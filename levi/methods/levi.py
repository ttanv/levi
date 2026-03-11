"""Levi: Main entry point for evolutionary code optimization."""

import asyncio
import logging
import re
import time
from collections.abc import Callable
from typing import Any

import litellm

from ..behavior import BehaviorExtractor, FeatureVector
from ..config import BudgetConfig, LeviConfig, LeviResult
from ..core import EvaluationResult, Program
from ..init import Diversifier
from ..llm import clear_llm_client, set_llm_client
from ..llm.unified_client import UnifiedLLMClient, UnifiedLLMClientConfig
from ..pipeline import PipelineRunner
from ..pipeline.state import PipelineState
from ..pool import CVTMAPElitesPool
from ..utils import ResilientProcessPool, evaluate_code, extract_code, extract_fn_name

logger = logging.getLogger(__name__)


def _register_models_with_litellm(config: LeviConfig) -> None:
    """Auto-register local endpoints and model info with litellm."""
    # Register local endpoints so litellm routes to them
    for model_name, base_url in config.local_endpoints.items():
        registration: dict = {
            "litellm_params": {
                "model": f"openai/{model_name}",
                "api_base": base_url,
                "api_key": "unused",
            }
        }
        # Merge model_info if available (for cost tracking)
        if model_name in config.model_info:
            registration.update(config.model_info[model_name])
        litellm.register_model({model_name: registration})

    # Register model_info for non-local models (cost tracking for custom cloud models)
    non_local_info = {k: v for k, v in config.model_info.items() if k not in config.local_endpoints}
    if non_local_info:
        litellm.register_model(non_local_info)


def _setup_logging() -> None:
    """Configure logging for levi."""
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
            f"Invalid centroid matrix in snapshot: expected (N,{pool._n_dims}), got shape {centroids_arr.shape}"
        )
    pool._centroids = centroids_arr
    pool._n_centroids = centroids_arr.shape[0]

    normalization = metadata["normalization"]
    mins_arr = np.asarray(normalization["mins"], dtype=float)
    maxs_arr = np.asarray(normalization["maxs"], dtype=float)
    ranges_arr = np.asarray(normalization["ranges"], dtype=float)
    if mins_arr.shape != (pool._n_dims,) or maxs_arr.shape != (pool._n_dims,) or ranges_arr.shape != (pool._n_dims,):
        raise RuntimeError(f"Invalid normalization bounds in snapshot: expected vectors of length {pool._n_dims}")
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

    extractor.set_phase("evolution")
    logger.info(f"[Resume] Restored {restored}/{len(elites)} elites into pool")
    return resumed_cost


def _build_config(
    problem_description: str,
    function_signature: str,
    seed_program: str | None,
    score_fn: Callable[..., dict],
    inputs: list[Any] | None,
    model: str | list[str] | None,
    paradigm_model: str | list[str] | None,
    mutation_model: str | list[str] | None,
    budget_dollars: float | None,
    budget_evals: int | None,
    budget_seconds: float | None,
    target_score: float | None = None,
    **kwargs: Any,
) -> LeviConfig:
    """Build LeviConfig from evolve_code() parameters."""
    # Validate function_signature
    if not re.search(r"def\s+\w+\s*\(", function_signature):
        raise ValueError(
            f"Invalid function_signature: must contain a Python function definition "
            f"(e.g. 'def solve(x):'). Got: {function_signature!r}"
        )

    # Resolve models
    if model is not None and (paradigm_model is not None or mutation_model is not None):
        raise ValueError(
            "Cannot specify both 'model' and 'paradigm_model'/'mutation_model'. "
            "Use 'model' for a single model, or 'paradigm_model'/'mutation_model' for separate models."
        )

    if model is not None:
        resolved_paradigm = model
        resolved_mutation = model
    elif paradigm_model is not None or mutation_model is not None:
        resolved_paradigm = paradigm_model or mutation_model
        resolved_mutation = mutation_model or paradigm_model
    else:
        raise ValueError("Must specify 'model' (for both), or 'paradigm_model' and/or 'mutation_model'.")

    # Build budget
    if budget_dollars is None and budget_evals is None and budget_seconds is None:
        raise ValueError(
            "Must specify at least one budget constraint: budget_dollars, budget_evals, or budget_seconds."
        )
    budget = BudgetConfig(
        dollars=budget_dollars,
        evaluations=budget_evals,
        seconds=budget_seconds,
        target_score=target_score,
    )

    # Reject kwargs that overlap with explicit params
    _EXPLICIT_PARAMS = {
        "problem_description",
        "function_signature",
        "seed_program",
        "score_fn",
        "inputs",
        "paradigm_models",
        "mutation_models",
        "budget",
    }
    overlap = set(kwargs) & _EXPLICIT_PARAMS
    if overlap:
        raise ValueError(f"Use the explicit parameters instead of **kwargs for: {overlap}")

    config_dict: dict[str, Any] = {
        "problem_description": problem_description,
        "function_signature": function_signature,
        "seed_program": seed_program,
        "score_fn": score_fn,
        "paradigm_models": resolved_paradigm,
        "mutation_models": resolved_mutation,
        "budget": budget,
    }
    if inputs is not None:
        config_dict["inputs"] = inputs

    config_dict.update(kwargs)
    return LeviConfig(**config_dict)


def evolve_code(
    problem_description: str,
    *,
    function_signature: str,
    seed_program: str | None = None,
    score_fn: Callable[..., dict],
    inputs: list[Any] | None = None,
    model: str | list[str] | None = None,
    paradigm_model: str | list[str] | None = None,
    mutation_model: str | list[str] | None = None,
    budget_dollars: float | None = None,
    budget_evals: int | None = None,
    budget_seconds: float | None = None,
    target_score: float | None = None,
    resume_snapshot: dict | None = None,
    **kwargs: Any,
) -> LeviResult:
    """
    Run Levi evolutionary code optimization.

    Simple usage::

        import levi

        result = levi.evolve_code(
            "Optimize bin packing to minimize wasted space",
            function_signature="def pack(items, bin_capacity):",
            score_fn=my_scorer,
            model="openai/gpt-4o-mini",
            budget_dollars=5.0,
        )

    With separate paradigm/mutation models::

        result = levi.evolve_code(
            problem_description,
            function_signature=sig,
            seed_program=seed,
            score_fn=scorer,
            paradigm_model="openai/gpt-4o",
            mutation_model="openai/gpt-4o-mini",
            budget_dollars=10.0,
        )

    Power users can pass any LeviConfig field via **kwargs::

        result = levi.evolve_code(
            ...,
            punctuated_equilibrium=levi.PunctuatedEquilibriumConfig(enabled=True),
            pipeline=levi.PipelineConfig(n_llm_workers=8),
            output_dir="runs/experiment_1",
        )

    Args:
        problem_description: Natural language description of the optimization problem.
        function_signature: Python function signature to optimize (e.g. "def solve(x):").
        seed_program: Initial working implementation. If None, init generates
            starting programs automatically from the function signature.
        score_fn: Evaluation function returning a dict with at least {"score": float}.
            Accepts either score_fn(fn) or score_fn(fn, inputs).
        inputs: Optional test inputs passed to score_fn. Can be omitted if
            score_fn only takes one argument.
        model: Model for both paradigm shifts and mutations. Use litellm format
            (e.g. "openai/gpt-4o-mini"). Mutually exclusive with paradigm_model/mutation_model.
        paradigm_model: Model(s) for paradigm shifts (heavier, creative).
        mutation_model: Model(s) for incremental mutations (lighter, fast).
        budget_dollars: Maximum dollar spend.
        budget_evals: Maximum number of evaluations.
        budget_seconds: Maximum wall-clock seconds.
        resume_snapshot: Optional snapshot dict to resume a previous run.
        **kwargs: Advanced overrides — any LeviConfig field (e.g. pipeline,
            punctuated_equilibrium, output_dir, local_endpoints).

    Returns:
        LeviResult with best_program, best_score, and run statistics.
    """
    config = _build_config(
        problem_description=problem_description,
        function_signature=function_signature,
        seed_program=seed_program,
        score_fn=score_fn,
        inputs=inputs,
        model=model,
        paradigm_model=paradigm_model,
        mutation_model=mutation_model,
        budget_dollars=budget_dollars,
        budget_evals=budget_evals,
        budget_seconds=budget_seconds,
        target_score=target_score,
        **kwargs,
    )
    return asyncio.run(_run_async(config, resume_snapshot=resume_snapshot))


async def _evaluate_seed(
    config: LeviConfig,
    executor: ResilientProcessPool,
    state: PipelineState,
    fn_name: str,
) -> dict:
    """Evaluate the seed program and return its result dict."""
    logger.info("[Levi] Evaluating seed program")

    if not await state.try_start_evaluation():
        raise RuntimeError("Budget exhausted before seed evaluation")

    try:
        seed_result = await executor.run(
            evaluate_code,
            config.seed_program,
            config.score_fn,
            config.inputs,
            fn_name,
            timeout=config.pipeline.eval_timeout,
        )
    except Exception as e:
        logger.error(f"[Levi] Failed to evaluate seed: {e}")
        raise RuntimeError(f"Seed program evaluation failed: {e}")
    finally:
        await state.finish_evaluation()

    if "error" in seed_result:
        state.record_error(str(seed_result["error"]))
        raise RuntimeError(f"Seed program evaluation error: {seed_result['error']}")

    seed_score = seed_result.get("score", 0.0)
    state.record_accept()
    state.record_score(
        score=seed_score,
        accepted=True,
        sampler="seed",
        archive_size=0,
        cell_index=None,
    )
    logger.info(f"[Levi] Seed score: {seed_score:.17g}")
    return seed_result


async def _run_async(config: LeviConfig, resume_snapshot: dict | None = None) -> LeviResult:
    """Internal async implementation."""
    run_start_time = time.time()
    _setup_logging()
    state = PipelineState(config.budget, start_time=run_start_time)
    state.configure_llm_concurrency(config.pipeline.n_llm_workers)

    logger.info("[Levi] Starting evolutionary optimization")
    logger.info(f"[Levi] Budget: {config.budget}")
    logger.info(f"[Levi] Sampler-model pairs: {len(config.sampler_model_pairs)}")

    # Register local endpoints and model info with litellm
    _register_models_with_litellm(config)

    # Run prompt optimization if enabled (before LLM client setup)
    if config.prompt_opt.enabled and not config.prompt_overrides:
        from ..prompt_opt import optimize_prompts

        overrides, opt_cost = optimize_prompts(config)
        config.prompt_overrides = overrides
        state.total_cost += opt_cost
        logger.info(f"[Levi] Prompt optimization cost: ${opt_cost:.3f}")

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
        fn_name = extract_fn_name(config.function_signature)

        # Evaluate seed program (if provided)
        seed_result = await _evaluate_seed(config, executor, state, fn_name) if config.seed_program else None

        # Resume from snapshot or run init
        init_cost = 0.0
        init_score_history = []
        if resume_snapshot is not None:
            init_cost = _restore_from_snapshot(pool, extractor, resume_snapshot)
            state.total_cost = state._coerce_finite_float(init_cost, default=0.0)
        else:
            logger.info("[Levi] Running init phase")
            diversifier = Diversifier(config, executor, state)
            init_cost, init_score_history = await diversifier.run(
                pool, config.seed_program, seed_result, extractor
            )
            logger.info(f"[Levi] Init phase complete, cost: ${init_cost:.3f}")

        # Run main pipeline
        logger.info("[Levi] Starting evolution pipeline")
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

        logger.info(
            f"[Levi] Complete - best score: {result.best_score:.17g}, "
            f"evals: {result.total_evaluations}, cost: ${result.total_cost:.3f}"
        )
        if config.output_dir:
            logger.info(f"[Levi] Snapshot: {config.output_dir}/snapshot.json")
        code_preview = result.best_program[:500]
        if len(result.best_program) > 500:
            code_preview += "..."
        logger.info(f"[Levi] Best program:\n{code_preview}")

        return result

    finally:
        await llm_client.close()
        clear_llm_client()
        executor.shutdown()
