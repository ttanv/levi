"""LLM producers: sample from archive and call LLM."""

import asyncio
import logging
import random

from ..artifacts import ArtifactAdapter, apply_diff as _apply_diff
from ..clients.base import client_name
from ..config import LeviConfig
from ..pool import CVTMAPElitesPool
from ..prompts import ProgramWithScore
from ..selection import ComponentSelector
from .state import BudgetLimitReached, PipelineState

logger = logging.getLogger(__name__)

# Backwards-compatible re-export used by existing tests and callers.
apply_diff = _apply_diff

FEEDBACK_MAX_FAILURES = 3


def _extract_failure_feedback(elite) -> list[str] | None:
    """Random-sample up to FEEDBACK_MAX_FAILURES failure-feedback strings.

    Uniform random over examples that scored below 1.0 (failures), so that
    repeated mutations of the same parent see different failure modes
    across calls instead of always the lowest-indexed three.
    """
    if elite is None:
        return None
    scores = elite.result.scores if elite.result is not None else None
    if not isinstance(scores, dict):
        return None
    fpe = scores.get("feedback_per_example")
    pes = scores.get("per_example_scores")
    if not fpe or not pes:
        return None
    failure_idx = [i for i, s in enumerate(pes) if s < 1.0 and i < len(fpe) and fpe[i]]
    if not failure_idx:
        return None
    k = min(FEEDBACK_MAX_FAILURES, len(failure_idx))
    return [fpe[i] for i in random.sample(failure_idx, k)]


async def llm_producer(
    worker_id: int,
    code_queue: asyncio.Queue,
    pool: CVTMAPElitesPool,
    archive_lock: asyncio.Lock,
    config: LeviConfig,
    artifact_adapter: ArtifactAdapter,
    state: PipelineState,
    stop_event: asyncio.Event,
    component_selector: ComponentSelector | None = None,
) -> None:
    while not stop_event.is_set():
        if state.budget_exhausted:
            break

        try:
            async with archive_lock:
                if pool.size() == 0:
                    logger.error(f"[LLM-{worker_id}] Archive is empty; stopping pipeline")
                    stop_event.set()
                    break
                sampler_name, model = pool.get_weighted_sampler_config()
                n_parents = config.pipeline.n_parents + config.pipeline.n_inspirations
                context = {"budget_progress": state.budget_progress}
                sample = pool.sample(sampler_name, n_parents=n_parents, context=context)

            parent = sample.parent
            inspirations = [p for p in sample.inspirations if random.random() < 0.8]
            parents = [parent] + inspirations

            # Determine output mode from config
            use_diff = config.pipeline.output_mode == "diff"
            model_key = client_name(model)

            is_bundle = getattr(artifact_adapter, "is_bundle_artifact", False)
            target: str | None = None
            if is_bundle and component_selector is not None:
                seed_bundle = getattr(artifact_adapter, "seed_bundle", None)
                if seed_bundle is not None:
                    async with archive_lock:
                        target = component_selector.select(list(seed_bundle.editable_targets))

            parent_elite = pool.get_elite(sample.metadata.get("source_cell"))
            feedback = _extract_failure_feedback(parent_elite)

            mutation_kwargs = {
                "meta_advice": (
                    state.current_meta_advice if state.current_meta_advice and random.random() < 0.8 else None
                ),
                "model": model,
                "use_diff": use_diff,
            }
            if target is not None:
                mutation_kwargs["target"] = target
            if feedback:
                mutation_kwargs["feedback"] = feedback

            prompt = artifact_adapter.build_mutation_prompt(
                [ProgramWithScore(p, None) for p in parents],
                **mutation_kwargs,
            )

            try:
                response = await state.acompletion(
                    model,
                    prompt=[{"role": "user", "content": prompt}],
                    temperature=config.pipeline.temperature,
                    max_tokens=config.pipeline.max_tokens,
                    timeout=300,
                )
                content = response.text
            except BudgetLimitReached:
                stop_event.set()
                break
            except Exception as e:
                logger.warning(f"[LLM-{worker_id}] [{model_key}] Error: {e}")
                await asyncio.sleep(1.0)
                continue

            # state.acompletion already accounts for cost centrally.

            if state.budget_exhausted:
                stop_event.set()
                break

            extract_kwargs: dict = {
                "parent_content": parent.content if (use_diff or target is not None) else None,
                "use_diff": use_diff,
            }
            if target is not None:
                extract_kwargs["target"] = target

            candidate_content = artifact_adapter.extract_candidate(content, **extract_kwargs)
            if not candidate_content:
                continue

            await code_queue.put(
                {
                    "content": candidate_content,
                    "sampler": sampler_name,
                    "source_cell": sample.metadata.get("source_cell"),
                    "model": model_key,
                    "target": target,
                }
            )

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"[LLM-{worker_id}] Unexpected error: {e}")
            await asyncio.sleep(1.0)
