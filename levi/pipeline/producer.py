"""LLM producers: sample from archive and call LLM."""

import asyncio
import logging
import random

from ..artifacts import ArtifactAdapter, apply_diff as _apply_diff
from ..clients.base import client_name
from ..config import LeviConfig
from ..pool import CVTMAPElitesPool
from ..prompts import ProgramWithScore
from .state import BudgetLimitReached, PipelineState

logger = logging.getLogger(__name__)

# Backwards-compatible re-export used by existing tests and callers.
apply_diff = _apply_diff


async def llm_producer(
    worker_id: int,
    code_queue: asyncio.Queue,
    pool: CVTMAPElitesPool,
    archive_lock: asyncio.Lock,
    config: LeviConfig,
    artifact_adapter: ArtifactAdapter,
    state: PipelineState,
    stop_event: asyncio.Event,
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
            prompt = artifact_adapter.build_mutation_prompt(
                [ProgramWithScore(p, None) for p in parents],
                meta_advice=state.current_meta_advice if state.current_meta_advice and random.random() < 0.8 else None,
                model=model,
                use_diff=use_diff,
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

            candidate_content = artifact_adapter.extract_candidate(
                content,
                parent_content=parent.content if use_diff else None,
                use_diff=use_diff,
            )
            if not candidate_content:
                continue

            await code_queue.put(
                {
                    "content": candidate_content,
                    "sampler": sampler_name,
                    "source_cell": sample.metadata.get("source_cell"),
                    "model": model_key,
                }
            )

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"[LLM-{worker_id}] Unexpected error: {e}")
            await asyncio.sleep(1.0)
