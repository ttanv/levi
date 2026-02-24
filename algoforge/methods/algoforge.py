"""AlgoForge: Main entry point for evolutionary code optimization."""

import asyncio
import logging
import time

import litellm

from ..config import AlgoforgeConfig, AlgoforgeResult
from ..core import Program, EvaluationResult
from ..pool import CVTMAPElitesPool
from ..behavior import BehaviorExtractor, FeatureVector
from ..pipeline import PipelineRunner
from ..pipeline.state import PipelineState
from ..init import Diversifier
from ..utils import ResilientProcessPool, extract_fn_name, evaluate_code
from ..llm import set_llm_client, clear_llm_client
from ..llm.unified_client import UnifiedLLMClient, UnifiedLLMClientConfig

logger = logging.getLogger(__name__)


def _register_models_with_litellm(config: AlgoforgeConfig) -> None:
    """Auto-register local endpoints and model info with litellm."""
    # Register local endpoints so litellm routes to them
    for model_name, base_url in config.local_endpoints.items():
        registration: dict = {"litellm_params": {
            "model": f"openai/{model_name}",
            "api_base": base_url,
            "api_key": "unused",
        }}
        # Merge model_info if available (for cost tracking)
        if model_name in config.model_info:
            registration.update(config.model_info[model_name])
        litellm.register_model({model_name: registration})

    # Register model_info for non-local models (cost tracking for custom cloud models)
    non_local_info = {k: v for k, v in config.model_info.items()
                      if k not in config.local_endpoints}
    if non_local_info:
        litellm.register_model(non_local_info)


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

    metadata = snapshot["metadata"]
    elites = snapshot.get("elites", [])
    run_state = snapshot.get("run_state", {})
    resumed_cost = run_state.get("total_cost", 0.0)

    logger.info(f"[Resume] Restoring {len(elites)} elites, prior cost: ${resumed_cost:.3f}")

    centroids_arr = np.asarray(metadata["centroids"], dtype=float)
    if centroids_arr.ndim != 2 or centroids_arr.shape[0] <= 0 or centroids_arr.shape[1] != pool._n_dims:
        raise RuntimeError(
            f"Invalid centroid matrix in snapshot: expected (N,{pool._n_dims}), "
            f"got shape {centroids_arr.shape}"
        )
    pool._centroids = centroids_arr
    pool._n_centroids = centroids_arr.shape[0]

    normalization = metadata["normalization"]
    mins_arr = np.asarray(normalization["mins"], dtype=float)
    maxs_arr = np.asarray(normalization["maxs"], dtype=float)
    ranges_arr = np.asarray(normalization["ranges"], dtype=float)
    if mins_arr.shape != (pool._n_dims,) or maxs_arr.shape != (pool._n_dims,) or ranges_arr.shape != (pool._n_dims,):
        raise RuntimeError(
            f"Invalid normalization bounds in snapshot: expected vectors of length {pool._n_dims}"
        )
    pool._mins = mins_arr
    pool._maxs = maxs_arr
    pool._ranges = ranges_arr

    # Restore elites into pool
    restored = 0
    for elite_data in elites:
        program = Program(
            content=elite_data["code"],
            metadata=elite_data.get("metadata", {}),
        )
        eval_result = EvaluationResult(
            scores=elite_data["scores"],
            is_valid=True,
        )
        behavior_values = elite_data["behavior"]
        if not isinstance(behavior_values, dict):
            raise RuntimeError("Invalid elite behavior in snapshot (expected dict)")
        behavior = FeatureVector({k: float(v) for k, v in behavior_values.items()})
        accepted = bool(pool.add_at_cell(int(elite_data["cell_index"]), program, eval_result, behavior))
        if not accepted:
            raise RuntimeError(f"Failed to restore elite for cell {elite_data['cell_index']}")

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

    # Register local endpoints and model info with litellm
    _register_models_with_litellm(config)

    # Run prompt optimization if enabled (before LLM client setup)
    if config.prompt_opt.enabled and not config.prompt_overrides:
        from ..prompt_opt import optimize_prompts
        overrides, opt_cost = optimize_prompts(config)
        config.prompt_overrides = overrides
        state.total_cost += opt_cost
        logger.info(f"[AlgoForge] Prompt optimization cost: ${opt_cost:.3f}")

    # Initialize unified LLM client
    llm_config = UnifiedLLMClientConfig(
        temperature=config.pipeline.temperature,
        max_tokens=config.pipeline.max_tokens,
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
