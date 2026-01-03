"""AlgoForge: Main entry point for evolutionary code optimization."""

import asyncio
import logging
import re

from ..config import AlgoforgeConfig, AlgoforgeResult
from ..core import Program, EvaluationResult
from ..pool import CVTMAPElitesPool
from ..behavior import BehaviorExtractor
from ..pipeline import PipelineRunner
from ..init import Diversifier
from ..utils import ResilientProcessPool

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    """Configure logging for algoforge."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("LiteLLM").setLevel(logging.ERROR)
    logging.getLogger("litellm").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _extract_fn_name(fn_signature: str) -> str:
    match = re.match(r'def\s+(\w+)\s*\(', fn_signature)
    if match:
        return match.group(1)
    return 'solve'


def _evaluate_code(code: str, score_fn, inputs: list, fn_name: str) -> dict:
    """Runs in subprocess: parse code, extract callable, call score_fn."""
    namespace = {}
    exec(code, namespace)

    fn = namespace.get(fn_name)
    if fn is None:
        for name, obj in namespace.items():
            if callable(obj) and not name.startswith('_'):
                fn = obj
                break

    if fn is None:
        return {"error": f"Function '{fn_name}' not found in generated code"}

    return score_fn(fn, inputs)


def run(config: AlgoforgeConfig) -> AlgoforgeResult:
    """
    Run AlgoForge evolutionary optimization.

    This is a synchronous entry point that handles async internally.
    """
    return asyncio.run(_run_async(config))


async def _run_async(config: AlgoforgeConfig) -> AlgoforgeResult:
    """Internal async implementation."""
    _setup_logging()

    logger.info("[AlgoForge] Starting evolutionary optimization")
    logger.info(f"[AlgoForge] Budget: {config.budget}")
    logger.info(f"[AlgoForge] Sampler-model pairs: {len(config.sampler_model_pairs)}")

    # Create behavior extractor
    extractor = BehaviorExtractor(
        ast_features=config.behavior.ast_features,
        score_keys=config.behavior.score_keys,
        init_noise=config.behavior.init_noise,
    )

    # Create CVT pool
    pool = CVTMAPElitesPool(
        behavior_extractor=extractor,
        n_centroids=config.cvt.n_centroids,
        defer_centroids=config.cvt.defer_centroids,
    )

    for pair in config.sampler_model_pairs:
        pool.register_sampler_model_pair(pair.sampler, pair.model, pair.weight, pair.temperature)

    # Create executor
    executor = ResilientProcessPool(max_workers=config.pipeline.n_eval_processes)

    try:
        # Evaluate seed program
        fn_name = _extract_fn_name(config.function_signature)
        logger.info("[AlgoForge] Evaluating seed program")

        try:
            seed_result = await executor.run(
                _evaluate_code,
                config.seed_program,
                config.score_fn,
                config.inputs,
                fn_name,
                timeout=config.pipeline.eval_timeout
            )
        except Exception as e:
            logger.error(f"[AlgoForge] Failed to evaluate seed: {e}")
            raise RuntimeError(f"Seed program evaluation failed: {e}")

        if "error" in seed_result:
            raise RuntimeError(f"Seed program evaluation error: {seed_result['error']}")

        seed_score = seed_result.get('score', 0.0)
        logger.info(f"[AlgoForge] Seed score: {seed_score:.1f}")

        # Init phase
        if config.init.enabled:
            logger.info("[AlgoForge] Running init phase")
            diversifier = Diversifier(config, executor)
            init_cost = await diversifier.run(pool, config.seed_program, seed_result, extractor)
            logger.info(f"[AlgoForge] Init phase complete, cost: ${init_cost:.3f}")
        else:
            # Just add seed to archive
            program = Program(code=config.seed_program)
            eval_result = EvaluationResult(
                program_id=program.id,
                scores=seed_result,
                is_valid=True,
            )
            pool.add(program, eval_result)
            extractor.set_phase('evolution')

        # Run main pipeline
        logger.info("[AlgoForge] Starting evolution pipeline")
        runner = PipelineRunner(config, pool, executor, output_dir=config.output_dir)
        result = await runner.run()

        logger.info(f"[AlgoForge] Complete - best score: {result.best_score:.1f}, "
                    f"evals: {result.total_evaluations}, cost: ${result.total_cost:.3f}")

        return result

    finally:
        executor.shutdown()
