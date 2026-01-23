"""AlgoForge: Main entry point for evolutionary code optimization."""

import asyncio
import logging
import types

from ..config import AlgoforgeConfig, AlgoforgeResult
from ..core import Program, EvaluationResult
from ..pool import CVTMAPElitesPool
from ..behavior import BehaviorExtractor
from ..pipeline import PipelineRunner
from ..init import Diversifier
from ..utils import ResilientProcessPool, extract_fn_name
from ..llm import set_llm_client, clear_llm_client
from ..llm.unified_client import UnifiedLLMClient, UnifiedLLMClientConfig

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


def _evaluate_code(code: str, score_fn, inputs: list, fn_name: str) -> dict:
    """Runs in subprocess for seed evaluation (no memory limit - trusted code)."""
    namespace = {}
    try:
        exec(code, namespace)
    except SyntaxError as e:
        return {"error": f"Syntax error: {e}"}

    fn = namespace.get(fn_name)
    if not isinstance(fn, types.FunctionType):
        return {"error": f"Function '{fn_name}' not found (got {type(fn).__name__})"}

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

    # Initialize unified LLM client
    llm_config = UnifiedLLMClientConfig(
        local_endpoints=config.llm.local_endpoints,
        model_info=config.llm.model_info,
        max_retries=config.llm.max_retries,
        retry_delay=config.llm.retry_delay,
        retry_backoff=config.llm.retry_backoff,
        default_temperature=config.pipeline.temperature,
        default_max_tokens=config.pipeline.max_tokens,
        batch_size=config.llm.batch_size,
        batch_max_wait_ms=config.llm.batch_max_wait_ms,
    )
    llm_client = UnifiedLLMClient(llm_config)
    set_llm_client(llm_client)

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
        pool.register_sampler_model_pair(pair.sampler, pair.model, pair.weight, pair.temperature, pair.n_cycles)

    # Create executor
    executor = ResilientProcessPool(max_workers=config.pipeline.n_eval_processes)

    try:
        # Evaluate seed program
        fn_name = extract_fn_name(config.function_signature)
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
        init_cost = 0.0
        if config.init.enabled:
            logger.info("[AlgoForge] Running init phase")
            diversifier = Diversifier(config, executor)
            init_cost = await diversifier.run(pool, config.seed_program, seed_result, extractor)
            logger.info(f"[AlgoForge] Init phase complete, cost: ${init_cost:.3f}")
        else:
            # Load predefined centroids if configured
            if config.cvt.predefined_centroids_file:
                from ..init.diversifier import _load_predefined_centroids
                import numpy as np
                logger.info(f"[AlgoForge] Loading predefined centroids from {config.cvt.predefined_centroids_file}")
                result = _load_predefined_centroids(
                    config.cvt.predefined_centroids_file,
                    extractor.features,
                )

                # Set up extractor normalization
                if result.bounds is not None:
                    # Deterministic mode: use fixed bounds from centroids file
                    extractor.set_fixed_bounds(result.bounds)
                    logger.info(f"[AlgoForge] Using deterministic normalization with fixed bounds")
                else:
                    # Legacy mode: use z-score stats from centroid data
                    extractor.init_stats_from_data(result.raw_feature_data)
                    logger.info(f"[AlgoForge] Using adaptive normalization (legacy mode)")

                pool._centroids = result.centroids
                pool._n_centroids = len(result.centroids)
                pool._mins = np.zeros(len(extractor.features))
                pool._maxs = np.ones(len(extractor.features))
                pool._ranges = np.ones(len(extractor.features))
                logger.info(f"[AlgoForge] Loaded {len(result.centroids)} predefined centroids")

            # Just add seed to archive
            program = Program(code=config.seed_program)
            eval_result = EvaluationResult(
                scores=seed_result,
                is_valid=True,
            )
            pool.add(program, eval_result)
            extractor.set_phase('evolution')

        # Run main pipeline
        logger.info("[AlgoForge] Starting evolution pipeline")
        runner = PipelineRunner(config, pool, executor, output_dir=config.output_dir, init_cost=init_cost)
        result = await runner.run()

        logger.info(f"[AlgoForge] Complete - best score: {result.best_score:.1f}, "
                    f"evals: {result.total_evaluations}, cost: ${result.total_cost:.3f}")
        if config.output_dir:
            logger.info(f"[AlgoForge] Snapshot: {config.output_dir}/snapshot.json")
        code_preview = result.best_program[:500]
        if len(result.best_program) > 500:
            code_preview += "..."
        logger.info(f"[AlgoForge] Best program:\n{code_preview}")

        return result

    finally:
        await llm_client.close()
        clear_llm_client()
        executor.shutdown()
