"""Eval consumers: evaluate code and update archive."""

import asyncio
import logging
from typing import Callable, Optional

from ..config import LeviConfig
from ..core import Program, EvaluationResult
from ..pool import CVTMAPElitesPool
from ..utils import ResilientProcessPool, extract_fn_name, evaluate_code, coerce_score
from .state import PipelineState, BudgetLimitReached

SNAPSHOT_INTERVAL = 10  # Save snapshot every N evaluations

logger = logging.getLogger(__name__)


def _short_model(model: str) -> str:
    return model.split("/")[-1]


def _model_label(item: dict) -> str:
    model = _short_model(item.get("model", "unknown"))
    sampler = item.get("sampler", "")
    if "_T" in sampler:
        return f"{model}{sampler[sampler.index('_T'):]}"
    return model


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


async def eval_consumer(
    worker_id: int,
    code_queue: asyncio.Queue,
    pool: CVTMAPElitesPool,
    archive_lock: asyncio.Lock,
    executor: ResilientProcessPool,
    config: LeviConfig,
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

        try:
            if not await state.try_start_evaluation():
                stop_event.set()
                break
            try:
                cascade = config.cascade
                if cascade.enabled and cascade.quick_inputs:
                    quick_result = await executor.run(
                        evaluate_code,
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
                                evaluate_code,
                                item["code"],
                                config.score_fn,
                                config.inputs,
                                fn_name,
                                timeout=config.pipeline.eval_timeout
                            )
                else:
                    result = await executor.run(
                        evaluate_code,
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

            async with archive_lock:
                if "cascade_rejected" in result:
                    pool.update_sampler(item["sampler"], item["source_cell"], success=False)
                    state.record_reject()
                    label = _model_label(item)
                    logger.info(
                        f"[Eval #{state.eval_count}] {label:30s} "
                        f"CASCADE SKIP  | quick: {result['quick_score']:.1f} < {result['threshold']:.1f}"
                    )
                elif "error" not in result:
                    score, score_error = coerce_score(result)
                    if score_error is not None:
                        pool.update_sampler(item["sampler"], item["source_cell"], success=False)
                        state.record_error(score_error)
                        label = _model_label(item)
                        logger.info(f"[Eval #{state.eval_count}] {label:30s} ERROR: {score_error[:50]}")
                    else:
                        result = dict(result)
                        result["score"] = score

                        program = Program(content=item["code"])
                        eval_result = EvaluationResult(
                            scores=result,
                            is_valid=True,
                        )
                        accepted, cell_index = pool.add(program, eval_result)
                        pool.update_sampler(item["sampler"], item["source_cell"], success=accepted)

                        if accepted:
                            state.record_accept()
                        else:
                            state.record_reject()

                        is_new_best = score > state.best_score_so_far

                        state.record_score(
                            score=score,
                            accepted=accepted,
                            sampler=item["sampler"],
                            archive_size=pool.size(),
                            cell_index=cell_index,
                        )

                        if is_new_best:
                            status = "NEW BEST ★"
                        elif accepted:
                            status = "accepted"
                        else:
                            status = "rejected"

                        label = _model_label(item)
                        logger.info(
                            f"[Eval #{state.eval_count}] {label:30s} {status:12s} | "
                            f"score: {score:.1f} | best: {state.best_score_so_far:.1f} | "
                            f"${state.total_cost:.3f}"
                        )
                else:
                    pool.update_sampler(item["sampler"], item["source_cell"], success=False)
                    state.record_error(result["error"])
                    label = _model_label(item)
                    logger.info(f"[Eval #{state.eval_count}] {label:30s} ERROR: {result['error'][:50]}")

            if config.meta_advice.enabled and state.should_generate_meta_advice(config.meta_advice.interval):
                asyncio.create_task(_generate_meta_advice(config, state))

            # Save snapshot every N evaluations
            if snapshot_callback and state.eval_count % SNAPSHOT_INTERVAL == 0:
                try:
                    snapshot_callback()
                except Exception as e:
                    logger.warning(f"[Snapshot] Failed to save: {e}")
            await state.finish_evaluation()
        except asyncio.CancelledError:
            await state.finish_evaluation()
            break
        except Exception as e:
            await state.finish_evaluation()
            logger.error(f"[Eval-{worker_id}] Unexpected error (continuing): {e}", exc_info=True)


def _format_metrics_for_llm(
    metrics: dict,
    previous_advice: str,
    progress_pct: float,
    problem_description: str = "",
    function_signature: str = "",
) -> str:
    total = metrics.get('acceptances', 0) + metrics.get('rejections', 0) + metrics.get('errors', 0)
    error_count = metrics.get('errors', 0)
    top_errors = metrics.get('top_errors', [])

    data = ""
    if problem_description:
        data += f"## Problem\n{problem_description}\n\n"
    if function_signature:
        data += f"## Function Signature\n```python\n{function_signature}\n```\n\n"

    data += f"""## Progress: {progress_pct:.0f}% of budget consumed

## Recent Results ({total} candidates evaluated this period):
- Acceptances: {metrics.get('acceptances', 0)}
- Rejections: {metrics.get('rejections', 0)}
- Errors/Failures: {error_count}"""

    if top_errors:
        data += "\n\n## Most Common Errors (across entire run):\n"
        for err, count in top_errors:
            data += f"- ({count}x) {err}\n"

    if previous_advice:
        data += f"\n\n## Your Previous Lessons:\n{previous_advice}"

    return data


async def _generate_meta_advice(config: LeviConfig, state: PipelineState) -> None:
    from ..llm import get_llm_client

    if not config.meta_advice.model:
        return

    metrics = state.reset_period_metrics()
    progress_pct = 0.0
    if config.budget.dollars:
        progress_pct = (state.total_cost / config.budget.dollars) * 100

    metrics_data = _format_metrics_for_llm(
        metrics, state.previous_meta_advice, progress_pct,
        config.problem_description, config.function_signature
    )
    prompt = META_ADVISOR_PROMPT.format(metrics_data=metrics_data)

    try:
        llm = get_llm_client()

        extras = {}
        if "deepseek" in config.meta_advice.model.lower():
            extras["reasoning"] = {"enabled": True}

        response = await state.acompletion(
            llm,
            model=config.meta_advice.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=config.meta_advice.max_tokens,
            timeout=60,
            **extras,
        )
        advice = response.content.strip()
        cost = response.cost
    except BudgetLimitReached:
        return
    except Exception as e:
        logger.warning(f"[Meta-Advice] Failed to generate: {e}")
        return

    try:
        state.previous_meta_advice = state.current_meta_advice
        state.current_meta_advice = advice

        logger.info(f"[Meta-Advice] Generated new advice (${cost:.4f})")
    except Exception as e:
        logger.warning(f"[Meta-Advice] Failed to update state: {e}")
