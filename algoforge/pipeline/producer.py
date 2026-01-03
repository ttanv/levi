"""LLM producers: sample from archive and call LLM."""

import asyncio
import random
import re
import logging
from typing import Optional

import litellm

from ..config import AlgoforgeConfig
from ..pool import CVTMAPElitesPool
from ..llm import PromptBuilder, ProgramWithScore, OutputMode
from .state import PipelineState

logger = logging.getLogger(__name__)


def extract_code(response: str) -> Optional[str]:
    """Extract Python code from LLM response."""
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    response = re.sub(r'<thinking>.*?</thinking>', '', response, flags=re.DOTALL)

    matches = re.findall(r'```python\s*(.*?)```', response, re.DOTALL)
    if matches:
        return matches[0].strip()

    matches = re.findall(r'```\s*(.*?)```', response, re.DOTALL)
    if matches:
        return matches[0].strip()

    stripped = response.strip()
    for pattern in ['def ', 'import ', 'from ', '# ', '"""', "'''"]:
        if stripped.startswith(pattern):
            return stripped

    for line in stripped.split('\n'):
        line = line.strip()
        if line.startswith(('def ', 'import ', 'from ', 'class ')):
            idx = stripped.find(line)
            return stripped[idx:].strip()

    return None


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
                sample = pool.sample(sampler_name, n_parents=n_parents)

            parent = sample.parent
            inspirations = [p for p in sample.inspirations if random.random() < 0.8]
            parents = [parent] + inspirations

            builder = PromptBuilder()
            builder.add_section("Problem", config.problem_description, priority=10)
            builder.add_section("Signature", f"```python\n{config.function_signature}\n```", priority=20)
            builder.add_parents([ProgramWithScore(p, None) for p in parents], priority=30)
            builder.set_output_mode(OutputMode.FULL)

            if state.current_meta_advice and random.random() < 0.8:
                builder.add_section("Meta-Advice", state.current_meta_advice, priority=100)

            prompt = builder.build()

            state.llm_in_flight += 1
            try:
                response = await litellm.acompletion(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=config.pipeline.temperature,
                    max_tokens=config.pipeline.max_tokens,
                    timeout=300,
                )
                content = response.choices[0].message.content
                cost = litellm.completion_cost(completion_response=response)
            except Exception as e:
                logger.warning(f"[LLM-{worker_id}] Error: {e}")
                await asyncio.sleep(1.0)
                continue
            finally:
                state.llm_in_flight -= 1

            state.add_cost(cost)

            if state.budget_exhausted:
                stop_event.set()
                break

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
