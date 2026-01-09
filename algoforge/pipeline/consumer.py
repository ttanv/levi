"""Eval consumers: evaluate code and update archive."""

import asyncio
import logging
import re
import resource
import types
from typing import Callable, Optional

from ..config import AlgoforgeConfig
from ..core import Program, EvaluationResult
from ..pool import CVTMAPElitesPool
from ..utils import ResilientProcessPool
from .state import PipelineState

SNAPSHOT_INTERVAL = 10  # Save snapshot every N evaluations

logger = logging.getLogger(__name__)

META_ADVISOR_PROMPT = """You are a lessons-learned advisor for an evolutionary code optimization system.

## Your Role
Analyze FAILURES from recent evaluations. Your lessons get injected into LLM prompts to help future solutions avoid the same mistakes.

## What You're Given
- **Failure count**: How many candidates failed (crashes, invalid code, timeouts, etc.)
- **Error patterns**: Specific error messages encountered (including timeouts)
- **Previous lessons**: What you advised last time (learn from what worked/didn't work)

## Your Task: Write Concise Lessons (150-200 words max)

### Focus ONLY on Failure Prevention
You do NOT see successful solutions. Your job is purely defensive:
1. **Identify error patterns** - What mistakes are being made repeatedly?
2. **Explain root causes** - Why are these errors happening?
3. **Give specific fixes** - Exactly how to avoid each error type
4. **Learn from previous advice** - If similar errors persist, strengthen the warning. If errors reduced, that advice worked.

### For Each Error Pattern:
- Quote the error briefly
- Explain what causes it
- Give a specific fix

## Output Format
Keep it SHORT and DIRECT:

**Avoid These Errors:**
- [Error pattern]: [How to fix]
- [Error pattern]: [How to fix]

---

{metrics_data}
"""


def extract_fn_name(fn_signature: str) -> str:
    match = re.search(r'def\s+(\w+)\s*\(', fn_signature)
    if match:
        return match.group(1)
    return 'solve'


def _evaluate_code(code: str, score_fn: Callable, inputs: list, fn_name: str) -> dict:
    """Runs in subprocess: parse code, extract callable, call score_fn."""
    # Limit process memory to 2GB to prevent VM crashes
    try:
        memory_bytes = 8 * 1024 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    except (ValueError, resource.error):
        pass  # May fail on some platforms

    namespace = {}
    try:
        exec(code, namespace)
    except SyntaxError as e:
        return {"error": f"Syntax error: {e}"}
    except MemoryError:
        return {"error": "MemoryError: code exceeded 8GB limit"}

    fn = namespace.get(fn_name)
    if not isinstance(fn, types.FunctionType):
        return {"error": f"Function '{fn_name}' not found (got {type(fn).__name__})"}

    try:
        return score_fn(fn, inputs)
    except MemoryError:
        return {"error": "MemoryError: code exceeded 8GB limit"}


async def eval_consumer(
    worker_id: int,
    code_queue: asyncio.Queue,
    pool: CVTMAPElitesPool,
    archive_lock: asyncio.Lock,
    executor: ResilientProcessPool,
    config: AlgoforgeConfig,
    state: PipelineState,
    stop_event: asyncio.Event,
    snapshot_callback: Optional[Callable[[], None]] = None,
) -> None:
    fn_name = extract_fn_name(config.function_signature)

    while not stop_event.is_set() or not code_queue.empty():
        try:
            item = await asyncio.wait_for(code_queue.get(), timeout=2.0)
        except asyncio.TimeoutError:
            continue
        except asyncio.CancelledError:
            break

        state.eval_in_flight += 1
        try:
            cascade = config.cascade
            if cascade.enabled and cascade.quick_inputs:
                quick_result = await executor.run(
                    _evaluate_code,
                    item["code"],
                    config.score_fn,
                    cascade.quick_inputs,
                    fn_name,
                    timeout=cascade.quick_timeout
                )
                if "error" in quick_result:
                    result = quick_result
                else:
                    quick_score = quick_result.get('score', 0)
                    threshold = state.best_score_so_far * cascade.min_score_ratio
                    if quick_score < threshold:
                        result = {"cascade_rejected": True, "quick_score": quick_score, "threshold": threshold}
                    else:
                        result = await executor.run(
                            _evaluate_code,
                            item["code"],
                            config.score_fn,
                            config.inputs,
                            fn_name,
                            timeout=config.pipeline.eval_timeout
                        )
            else:
                result = await executor.run(
                    _evaluate_code,
                    item["code"],
                    config.score_fn,
                    config.inputs,
                    fn_name,
                    timeout=config.pipeline.eval_timeout
                )
        except TimeoutError:
            result = {"error": "Timeout"}
        except Exception as e:
            result = {"error": str(e)}
        finally:
            state.eval_in_flight -= 1

        async with archive_lock:
            if "cascade_rejected" in result:
                pool.update_sampler(item["sampler"], item["source_cell"], success=False)
                state.record_reject()
                logger.info(
                    f"[Eval #{state.eval_count}] {item['sampler']:15s} "
                    f"CASCADE SKIP  | quick: {result['quick_score']:.1f} < {result['threshold']:.1f}"
                )
            elif "error" not in result:
                program = Program(code=item["code"])
                eval_result = EvaluationResult(
                    program_id=program.id,
                    scores=result,
                    is_valid=True,
                )
                accepted = pool.add(program, eval_result)
                pool.update_sampler(item["sampler"], item["source_cell"], success=accepted)

                if accepted:
                    state.record_accept()
                else:
                    state.record_reject()

                score = result.get('score', 0)

                is_new_best = score > state.best_score_so_far

                state.record_score(
                    score=score,
                    accepted=accepted,
                    sampler=item["sampler"],
                    archive_size=pool.size(),
                )

                if is_new_best:
                    status = "NEW BEST ★"
                elif accepted:
                    status = "accepted"
                else:
                    status = "rejected"

                logger.info(
                    f"[Eval #{state.eval_count}] {item['sampler']:15s} "
                    f"{status:12s} | "
                    f"score: {score:.1f} | best: {state.best_score_so_far:.1f} | "
                    f"cost: ${state.total_cost:.3f}"
                )
            else:
                pool.update_sampler(item["sampler"], item["source_cell"], success=False)
                state.record_error(result["error"])
                logger.info(f"[Eval #{state.eval_count}] {item['sampler']:15s} ERROR: {result['error'][:50]}")

        if config.meta_advice.enabled and state.should_generate_meta_advice(config.meta_advice.interval):
            asyncio.create_task(_generate_meta_advice(config, state))

        # Save snapshot every N evaluations
        if snapshot_callback and state.eval_count % SNAPSHOT_INTERVAL == 0:
            try:
                snapshot_callback()
            except Exception as e:
                logger.warning(f"[Snapshot] Failed to save: {e}")


def _format_metrics_for_llm(metrics: dict, previous_advice: str, progress_pct: float) -> str:
    total = metrics.get('acceptances', 0) + metrics.get('rejections', 0) + metrics.get('errors', 0)
    error_count = metrics.get('errors', 0)
    error_messages = metrics.get('error_messages', set())

    data = f"""## Progress: {progress_pct:.0f}% of budget consumed

## Recent Results ({total} candidates evaluated this period):
- Acceptances: {metrics.get('acceptances', 0)}
- Rejections: {metrics.get('rejections', 0)}
- Errors/Failures: {error_count}"""

    if error_messages:
        data += "\n\n## Error Patterns to Avoid:\n"
        for err in sorted(error_messages):
            data += f"- {err}\n"

    if previous_advice:
        data += f"\n\n## Your Previous Lessons (learn from these - did they help reduce errors?):\n{previous_advice}"

    return data


async def _generate_meta_advice(config: AlgoforgeConfig, state: PipelineState) -> None:
    import litellm

    if not config.meta_advice.model:
        return

    metrics = state.reset_period_metrics()
    progress_pct = 0.0
    if config.budget.dollars:
        progress_pct = (state.total_cost / config.budget.dollars) * 100

    metrics_data = _format_metrics_for_llm(metrics, state.previous_meta_advice, progress_pct)
    prompt = META_ADVISOR_PROMPT.format(metrics_data=metrics_data)

    try:
        call_kwargs = {
            "model": config.meta_advice.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": config.meta_advice.max_tokens,
            "timeout": 60,
        }

        if "deepseek" in config.meta_advice.model.lower():
            call_kwargs["reasoning"] = {"enabled": True}

        response = await litellm.acompletion(**call_kwargs)
        advice = response.choices[0].message.content.strip()
        cost = litellm.completion_cost(completion_response=response)

        state.previous_meta_advice = state.current_meta_advice
        state.current_meta_advice = advice
        state.add_cost(cost)

        logger.info(f"[Meta-Advice] Generated new advice (${cost:.4f})")
    except Exception as e:
        logger.warning(f"[Meta-Advice] Failed to generate: {e}")
