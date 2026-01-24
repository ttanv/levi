"""LLM producers: sample from archive and call LLM."""

import asyncio
import random
import re
import logging
from typing import Optional

from ..config import AlgoforgeConfig
from ..pool import CVTMAPElitesPool
from ..llm import PromptBuilder, ProgramWithScore, OutputMode, get_llm_client, LLMRetryExhaustedError
from ..utils import extract_code
from .state import PipelineState

logger = logging.getLogger(__name__)


def apply_diff(original: str, diff_response: str) -> Optional[str]:
    """Apply SEARCH/REPLACE diff blocks to original code."""
    result = original

    pattern = r'<<<<<<< SEARCH\s*(.*?)\s*=======\s*(.*?)\s*>>>>>>> REPLACE'
    matches = re.findall(pattern, diff_response, re.DOTALL)

    if not matches:
        # No diff blocks found, try to extract full code
        return extract_code(diff_response)

    for search, replace in matches:
        search = search.strip()
        replace = replace.strip()
        if search in result:
            result = result.replace(search, replace, 1)
        else:
            # Search block not found in original - diff failed
            return None

    return result


async def llm_producer(
    worker_id: int,
    code_queue: asyncio.Queue,
    pool: CVTMAPElitesPool,
    archive_lock: asyncio.Lock,
    config: AlgoforgeConfig,
    state: PipelineState,
    stop_event: asyncio.Event,
) -> None:
    while not stop_event.is_set():
        if state.budget_exhausted:
            break

        try:
            async with archive_lock:
                sampler_name, model = pool.get_weighted_sampler_config()
                n_parents = config.pipeline.n_parents + config.pipeline.n_inspirations
                context = {"budget_progress": state.budget_progress}
                sample = pool.sample(sampler_name, n_parents=n_parents, context=context)

            parent = sample.parent
            inspirations = [p for p in sample.inspirations if random.random() < 0.8]
            parents = [parent] + inspirations

            # Determine output mode from config
            use_diff = config.pipeline.output_mode == "diff"
            output_mode = OutputMode.DIFF if use_diff else OutputMode.FULL

            builder = PromptBuilder()
            builder.add_section("Problem", config.problem_description, priority=10)
            builder.add_section("Signature", f"```python\n{config.function_signature}\n```", priority=20)
            builder.add_parents([ProgramWithScore(p, None) for p in parents], priority=30)

            # Check for optimized mutation instructions for this model
            mutation_overrides = config.prompt_overrides.get("mutation", {})
            if model in mutation_overrides:
                builder.set_custom_output(mutation_overrides[model])
            else:
                builder.set_output_mode(output_mode)

            if state.current_meta_advice and random.random() < 0.8:
                builder.add_section("Meta-Advice", state.current_meta_advice, priority=100)

            prompt = builder.build()

            state.llm_in_flight += 1
            try:
                llm = get_llm_client()
                response = await llm.acompletion(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=config.pipeline.temperature,
                    max_tokens=config.pipeline.max_tokens,
                    timeout=300,
                )
                content = response.content
                cost = response.cost
            except LLMRetryExhaustedError as e:
                logger.warning(f"[LLM-{worker_id}] [{model}] Error after retries: {e.last_error}")
                continue
            except Exception as e:
                logger.warning(f"[LLM-{worker_id}] [{model}] Error: {e}")
                await asyncio.sleep(1.0)
                continue
            finally:
                state.llm_in_flight -= 1

            state.add_cost(cost)

            if state.budget_exhausted:
                stop_event.set()
                break

            # Extract or apply code based on mode
            if use_diff:
                code = apply_diff(parent.code, content)
            else:
                code = extract_code(content)
            if not code:
                continue

            await code_queue.put({
                "code": code,
                "sampler": sampler_name,
                "source_cell": sample.metadata.get("source_cell"),
                "model": model,
            })

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"[LLM-{worker_id}] Unexpected error: {e}")
            await asyncio.sleep(1.0)
