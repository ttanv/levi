"""
SandboxedEvaluator: Evaluator with process-isolated code execution.

Uses multiprocessing for isolation and reliable timeout enforcement.
Captures execution time per input for systems optimization scoring.
"""

import time
import multiprocessing as mp
from typing import Callable, Any, Optional

from ..core import Program, EvaluationResult, MetricDict
from ..budget import BudgetManager, ResourceType
from .protocol import EvaluationStage


def _run_in_process(code: str, inputs: list, queue: mp.Queue) -> None:
    """Worker function that runs in isolated process."""
    try:
        namespace = {}
        exec(code, namespace)

        # Find callable
        func = None
        for name in ['solve', 'main', 'solution', 'priority']:
            if name in namespace and callable(namespace[name]):
                func = namespace[name]
                break
        if func is None:
            for name, obj in namespace.items():
                if callable(obj) and not name.startswith('_'):
                    func = obj
                    break

        if func is None:
            queue.put(('error', 'No callable function found'))
            return

        results = {}
        for i, inp in enumerate(inputs):
            start = time.perf_counter()
            output = func(inp)
            elapsed = time.perf_counter() - start
            results[str(i)] = {'output': output, 'exec_time': elapsed}

        queue.put(('success', results))

    except MemoryError:
        queue.put(('error', 'MemoryError during execution'))
    except Exception as e:
        queue.put(('error', str(e)))


class SandboxedEvaluator:
    """
    Evaluator with process-isolated code execution.

    Supports multiple score functions for multi-objective optimization.
    Score functions receive (output, input, exec_time) to score
    correctness and execution properties.
    """

    def __init__(
        self,
        budget_manager: BudgetManager,
        score_functions: dict[str, Callable[[Any, Any, float], float]],
        timeout: float = 10.0,
    ) -> None:
        """
        Args:
            budget_manager: Budget manager for tracking evaluations
            score_functions: Dict of {metric_name: fn(output, input, exec_time) -> score}
            timeout: Maximum execution time per program (seconds)
        """
        self._budget_manager = budget_manager
        self._score_functions = score_functions
        self._timeout = timeout

    def evaluate(
        self,
        program: Program,
        inputs: list,
    ) -> EvaluationResult:
        """Evaluate program in isolated process."""
        self._budget_manager.check_budget()

        queue = mp.Queue()
        process = mp.Process(
            target=_run_in_process,
            args=(program.code, inputs, queue)
        )
        process.start()
        process.join(timeout=self._timeout)

        if process.is_alive():
            # First try graceful termination (SIGTERM)
            process.terminate()
            process.join(timeout=1.0)  # Give it 1 second to terminate gracefully

            # If still alive, force kill (SIGKILL) - prevents zombie processes
            if process.is_alive():
                process.kill()
                process.join(timeout=1.0)

            # Clean up process resources
            process.close()

            # Clean up queue to prevent resource leaks
            try:
                queue.close()
                queue.join_thread()
            except Exception:
                pass

            # Track consecutive timeouts for backpressure
            self._consecutive_timeouts = getattr(self, '_consecutive_timeouts', 0) + 1
            if self._consecutive_timeouts >= 3:
                # Backpressure: add small delay when timeouts cascade
                import time as time_mod
                time_mod.sleep(0.5 * min(self._consecutive_timeouts - 2, 5))

            self._budget_manager.try_consume(ResourceType.EVALUATIONS, 1)
            return EvaluationResult(
                is_valid=False,
                error=f"Timeout after {self._timeout}s",
            )

        if queue.empty():
            # Clean up process and queue resources
            try:
                process.close()
            except Exception:
                pass
            try:
                queue.close()
                queue.join_thread()
            except Exception:
                pass

            # Track as a timeout-like failure for backpressure
            self._consecutive_timeouts = getattr(self, '_consecutive_timeouts', 0) + 1

            self._budget_manager.try_consume(ResourceType.EVALUATIONS, 1)
            return EvaluationResult(
                is_valid=False,
                error="Process died without result",
            )

        status, result = queue.get()

        # Clean up process and queue resources (successful completion)
        try:
            process.close()
        except Exception:
            pass
        try:
            queue.close()
            queue.join_thread()
        except Exception:
            pass

        self._budget_manager.try_consume(ResourceType.EVALUATIONS, 1)

        if status == 'error':
            # Still an error, but process completed - mild backpressure
            self._consecutive_timeouts = max(0, getattr(self, '_consecutive_timeouts', 0) - 1)
            return EvaluationResult(
                is_valid=False,
                error=result,
            )

        scores: MetricDict = {}
        try:
            for metric_name, score_fn in self._score_functions.items():
                metric_scores = [
                    score_fn(
                        result[str(i)]['output'],
                        inp,
                        result[str(i)]['exec_time']
                    )
                    for i, inp in enumerate(inputs)
                ]
                scores[metric_name] = sum(metric_scores) / len(metric_scores)
        except Exception as e:
            return EvaluationResult(
                is_valid=False,
                error=f"Scoring error: {e}",
            )

        # Reset consecutive timeout counter on successful evaluation
        self._consecutive_timeouts = 0

        return EvaluationResult(
            scores=scores,
            is_valid=True,
        )

    def evaluate_cascade(
        self,
        program: Program,
        stages: list[EvaluationStage]
    ) -> Optional[EvaluationResult]:
        """Progressive evaluation with early termination."""
        for stage in stages:
            result = self.evaluate(program, stage.inputs)
            if not result.is_valid:
                return None
            if stage.validator and not stage.validator(result):
                return None
        return result
