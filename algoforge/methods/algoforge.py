"""AlgoForge: Main entry point for evolutionary code optimization."""

import asyncio
import logging
import time

from ..config import AlgoforgeConfig, AlgoforgeResult
from ..core import Program, EvaluationResult
from ..pool import CVTMAPElitesPool
from ..behavior import BehaviorExtractor
from ..pipeline import PipelineRunner
from ..pipeline.state import PipelineState
from ..init import Diversifier
from ..utils import ResilientProcessPool, extract_fn_name, evaluate_code
from ..llm import set_llm_client, clear_llm_client
from ..llm.unified_client import UnifiedLLMClient, UnifiedLLMClientConfig

logger = logging.getLogger(__name__)


def _collect_referenced_models(config: AlgoforgeConfig) -> set[str]:
    """Collect all model names referenced by runtime config."""
    models: set[str] = {pair.model for pair in config.sampler_model_pairs}

    if config.init.diversity_model:
        models.add(config.init.diversity_model)
    if config.init.variant_models:
        models.update(config.init.variant_models)
    if config.meta_advice.model:
        models.add(config.meta_advice.model)
    if config.punctuated_equilibrium.heavy_model:
        models.add(config.punctuated_equilibrium.heavy_model)
    if config.punctuated_equilibrium.variant_models:
        models.update(config.punctuated_equilibrium.variant_models)

    return models


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


def _restore_from_snapshot(
    pool: CVTMAPElitesPool,
    extractor: BehaviorExtractor,
    snapshot: dict,
) -> float:
    """
    Restore pool state from a snapshot dict.

    Returns the total_cost from the snapshot (used as init_cost for the runner).
    """
    import numpy as np

    elites = snapshot.get("elites", [])
    run_state = snapshot.get("run_state", {})
    resumed_cost = run_state.get("total_cost", 0.0)

    logger.info(f"[Resume] Restoring {len(elites)} elites, prior cost: ${resumed_cost:.3f}")

    # Generate centroids if deferred (init would normally do this)
    if pool._centroids is None:
        pool._centroids = pool._init_cvt_centroids()
        pool._mins = np.zeros(pool._n_dims)
        pool._maxs = np.ones(pool._n_dims)
        pool._ranges = np.ones(pool._n_dims)

    # Restore elites into pool
    restored = 0
    for elite_data in elites:
        program = Program(code=elite_data["code"])
        eval_result = EvaluationResult(
            scores=elite_data["scores"],
            is_valid=True,
        )
        accepted, _ = pool.add(program, eval_result)
        if accepted:
            restored += 1

    extractor.set_phase('evolution')
    logger.info(f"[Resume] Restored {restored}/{len(elites)} elites into pool")
    return resumed_cost


def run(config: AlgoforgeConfig, resume_snapshot: dict | None = None) -> AlgoforgeResult:
    """
    Run AlgoForge evolutionary optimization.

    This is a synchronous entry point that handles async internally.

    Args:
        config: AlgoForge configuration.
        resume_snapshot: Optional snapshot dict (from snapshot.json) to resume from.
                        When provided, skips init and loads elites from the snapshot.
    """
    return asyncio.run(_run_async(config, resume_snapshot=resume_snapshot))


async def _run_async(config: AlgoforgeConfig, resume_snapshot: dict | None = None) -> AlgoforgeResult:
    """Internal async implementation."""
    run_start_time = time.time()
    _setup_logging()
    state = PipelineState(config.budget, start_time=run_start_time)
    state.configure_llm_concurrency(config.pipeline.n_llm_workers)

    logger.info("[AlgoForge] Starting evolutionary optimization")
    logger.info(f"[AlgoForge] Budget: {config.budget}")
    logger.info(f"[AlgoForge] Sampler-model pairs: {len(config.sampler_model_pairs)}")

    # Initialize unified LLM client
    llm_config = UnifiedLLMClientConfig(
        local_endpoints=config.llm.local_endpoints,
        model_info=config.llm.model_info,
        known_models=_collect_referenced_models(config),
        max_retries=config.llm.max_retries,
        retry_delay=config.llm.retry_delay,
        retry_backoff=config.llm.retry_backoff,
        default_temperature=config.pipeline.temperature,
        default_max_tokens=config.pipeline.max_tokens,
    )
    llm_client = UnifiedLLMClient(llm_config)
    set_llm_client(llm_client)

    # Create behavior extractor
    extractor = BehaviorExtractor(
        ast_features=config.behavior.ast_features,
        score_keys=config.behavior.score_keys,
        init_noise=config.behavior.init_noise,
        custom_extractors=config.behavior.custom_extractors or None,
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

        if not await state.try_start_evaluation():
            raise RuntimeError("Budget exhausted before seed evaluation")

        try:
            seed_result = await executor.run(
                evaluate_code,
                config.seed_program,
                config.score_fn,
                config.inputs,
                fn_name,
                timeout=config.pipeline.eval_timeout
            )
        except Exception as e:
            logger.error(f"[AlgoForge] Failed to evaluate seed: {e}")
            raise RuntimeError(f"Seed program evaluation failed: {e}")
        finally:
            await state.finish_evaluation()

        if "error" in seed_result:
            state.record_error(str(seed_result["error"]))
            raise RuntimeError(f"Seed program evaluation error: {seed_result['error']}")

        seed_score = seed_result.get('score', 0.0)
        state.record_accept()
        state.record_score(
            score=seed_score,
            accepted=True,
            sampler="seed",
            archive_size=0,
            cell_index=None,
        )
        logger.info(f"[AlgoForge] Seed score: {seed_score:.1f}")

        # Resume from snapshot or run init
        init_cost = 0.0
        init_score_history = []
        if resume_snapshot is not None:
            init_cost = _restore_from_snapshot(pool, extractor, resume_snapshot)
            state.total_cost = state._coerce_finite_float(init_cost, default=0.0)
        else:
            if not config.init.enabled:
                logger.info("[AlgoForge] init.enabled=False ignored; running data-driven init")
            logger.info("[AlgoForge] Running init phase")
            diversifier = Diversifier(config, executor, start_time=run_start_time, state=state)
            init_cost, init_score_history = await diversifier.run(pool, config.seed_program, seed_result, extractor)
            logger.info(f"[AlgoForge] Init phase complete, cost: ${init_cost:.3f}")

        # Run main pipeline
        logger.info("[AlgoForge] Starting evolution pipeline")
        runner = PipelineRunner(
            config,
            pool,
            executor,
            output_dir=config.output_dir,
            init_cost=init_cost,
            init_score_history=init_score_history,
            start_time=run_start_time,
            state=state,
        )
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
