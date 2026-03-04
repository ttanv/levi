"""
DSPy prompt optimization for Levi.

Consolidates the ~800-line example scripts into a single module that
reuses core utilities (extract_code, evaluate_code) and is driven
entirely by LeviConfig.
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import dspy

from ..config import LeviConfig
from ..utils.code_extraction import extract_code, extract_fn_name
from ..utils.evaluation import evaluate_code

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DSPy TIPS configuration
# ---------------------------------------------------------------------------


def _configure_dspy_tips() -> None:
    """Patch DSPy TIPS dict for brevity/open-endedness."""
    try:
        from dspy.propose.grounded_proposer import TIPS

        TIPS["brevity"] = (
            "Keep the instruction concise (under 100 words). Brevity encourages creativity and exploration."
        )
        TIPS["open_ended"] = (
            "Keep instructions general and goal-focused. Describe WHAT to "
            "achieve, not HOW to achieve it. Avoid prescribing specific "
            "algorithms, techniques, or data structures."
        )
        if "description" in TIPS:
            del TIPS["description"]
        if "high_stakes" in TIPS:
            del TIPS["high_stakes"]
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# DSPy LM factory
# ---------------------------------------------------------------------------


def _make_dspy_lm(
    model: str,
    config: LeviConfig,
    temperature: float = 0.8,
    max_tokens: int = 8192,
) -> dspy.LM:
    """Create a ``dspy.LM`` from an Levi model name + ``config.local_endpoints``."""
    if model in config.local_endpoints:
        return dspy.LM(
            model=f"openai/{model}",
            api_base=config.local_endpoints[model],
            api_key="unused",
            temperature=temperature,
            max_tokens=max_tokens,
        )
    return dspy.LM(model=model, temperature=temperature, max_tokens=max_tokens)


# ---------------------------------------------------------------------------
# DSPy Signatures
# ---------------------------------------------------------------------------


class MutationSignature(dspy.Signature):
    """Generate an improved version of code.

    You are given a parent solution and one inspiration solution. Mutate the parent
    code to improve its score, optionally borrowing ideas from the inspiration.
    Output the complete improved code.
    """

    problem_description: str = dspy.InputField(desc="The optimization problem description")
    function_signature: str = dspy.InputField(desc="The exact function signature to implement")
    parent_code: str = dspy.InputField(desc="The parent code to mutate (v1)")
    parent_score: float = dspy.InputField(desc="Score of the parent code")
    inspiration_code: str = dspy.InputField(desc="An inspiration solution to optionally borrow ideas from (v2)")
    inspiration_score: float = dspy.InputField(desc="Score of the inspiration code")

    code: str = dspy.OutputField(desc="Complete improved Python code")


class ParadigmShiftSignature(dspy.Signature):
    """Generate a high-scoring algorithmic solution using a fundamentally new approach.

    You are given representative solutions from different behavioral regions.
    Analyze them, then propose a NEW solution using a different algorithmic paradigm.
    Do not simply tweak or combine existing solutions.
    """

    problem_description: str = dspy.InputField(desc="The optimization problem description")
    function_signature: str = dspy.InputField(desc="The exact function signature to implement")
    best_score: float = dspy.InputField(desc="The best score achieved so far")
    representative_solutions: str = dspy.InputField(desc="Current solutions with their scores")

    code: str = dspy.OutputField(desc="Complete Python code that achieves a high score")


# ---------------------------------------------------------------------------
# DSPy Modules
# ---------------------------------------------------------------------------


class MutationModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict(MutationSignature)

    def forward(
        self, problem_description, function_signature, parent_code, parent_score, inspiration_code, inspiration_score
    ):
        return self.generate(
            problem_description=problem_description,
            function_signature=function_signature,
            parent_code=parent_code,
            parent_score=parent_score,
            inspiration_code=inspiration_code,
            inspiration_score=inspiration_score,
        )


class ParadigmShiftModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict(ParadigmShiftSignature)

    def forward(self, problem_description, function_signature, best_score, representative_solutions):
        return self.generate(
            problem_description=problem_description,
            function_signature=function_signature,
            best_score=best_score,
            representative_solutions=representative_solutions,
        )


# ---------------------------------------------------------------------------
# Prompt quality helpers
# ---------------------------------------------------------------------------


def _length_penalty(prompt: str, threshold: int = 300, max_penalty: float = 0.15) -> float:
    if len(prompt) <= threshold:
        return 0.0
    penalty = ((len(prompt) - threshold) / 1000) * max_penalty
    return min(max_penalty, penalty)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _mutation_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    config: LeviConfig,
    fn_name: str,
    trace: Optional[Any] = None,
) -> float:
    """Cascading metric: has_code → extracted → compiles → runs → score_improvement - length_penalty."""
    output = prediction.code
    score = 0.0

    has_code_block = "```python" in output or "```\n" in output or "def " in output
    if has_code_block:
        score += 0.10

    mutated_code = extract_code(output)
    if mutated_code is None:
        if "def " in output:
            mutated_code = output
        else:
            return score

    score += 0.10

    try:
        compile(mutated_code, "<string>", "exec")
    except SyntaxError:
        return score

    score += 0.10

    result = evaluate_code(mutated_code, config.score_fn, config.inputs, fn_name)
    if "error" in result:
        return score

    score += 0.25  # Runs without error
    child_score = result.get("score", 0.0)
    parent_score = example.parent_score

    if child_score > parent_score:
        improvement = (child_score - parent_score) / max(parent_score, 1.0)
        score += min(improvement * 2, 1.0) * 0.25
    elif child_score >= parent_score * 0.95:
        score += 0.12
    elif child_score >= parent_score * 0.8:
        score += 0.05

    # Length penalty on the optimized instruction (from trace context)
    if trace:
        for step in trace:
            if hasattr(step, "signature") and hasattr(step.signature, "instructions"):
                instr = str(step.signature.instructions)
                score -= _length_penalty(instr, threshold=300, max_penalty=0.15)
                break

    return max(0.0, score)


def _paradigm_shift_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    config: LeviConfig,
    fn_name: str,
    reference_code: str,
    trace: Optional[Any] = None,
) -> float:
    """Hard gate (compile+run) then score(0.60) + diversity(0.20) - length_penalty."""
    code = extract_code(prediction.code) or prediction.code

    try:
        compile(code, "<string>", "exec")
    except SyntaxError:
        return 0.0

    result = evaluate_code(code, config.score_fn, config.inputs, fn_name)
    if "error" in result:
        return 0.0

    exec_score = result.get("score", 0.0)
    normalized_score = min(exec_score / 100.0, 1.0)
    score = normalized_score * 0.60

    # Diversity: simple structural distance from reference
    try:
        import ast as _ast

        ref_nodes = len(list(_ast.walk(_ast.parse(reference_code))))
        new_nodes = len(list(_ast.walk(_ast.parse(code))))
        diversity = abs(new_nodes - ref_nodes) / max(ref_nodes, new_nodes, 1)
        score += min(diversity * 2, 1.0) * 0.20
    except Exception:
        pass

    # Length penalty on optimized instruction
    if trace:
        for step in trace:
            if hasattr(step, "signature") and hasattr(step.signature, "instructions"):
                instr = str(step.signature.instructions)
                score -= _length_penalty(instr, threshold=400, max_penalty=0.10)
                break

    return max(0.0, score)


# ---------------------------------------------------------------------------
# Training data generation
# ---------------------------------------------------------------------------


def _generate_mutation_examples(
    config: LeviConfig,
    fn_name: str,
    seed_score: float,
    n_examples: int = 5,
) -> list[dspy.Example]:
    """Generate training examples from seed_program + score_fn.

    Uses the seed as both parent and sole inspiration (v1 limitation).
    Pure instruction optimization with ``max_bootstrapped_demos=0`` means
    examples are only used for metric evaluation, not few-shot.
    """
    examples = []
    for _ in range(n_examples):
        example = dspy.Example(
            problem_description=config.problem_description,
            function_signature=config.function_signature,
            parent_code=config.seed_program,
            parent_score=seed_score,
            inspiration_code=config.seed_program,
            inspiration_score=seed_score,
        ).with_inputs(
            "problem_description",
            "function_signature",
            "parent_code",
            "parent_score",
            "inspiration_code",
            "inspiration_score",
        )
        examples.append(example)
    return examples


def _generate_paradigm_shift_examples(
    config: LeviConfig,
    seed_score: float,
    n_examples: int = 4,
) -> list[dspy.Example]:
    """Generate training examples from seed + baseline score."""
    representative_solutions = f"### Region 1 - Seed (Score: {seed_score:.17g})\n```python\n{config.seed_program}\n```\n"

    examples = []
    for _ in range(n_examples):
        example = dspy.Example(
            problem_description=config.problem_description,
            function_signature=config.function_signature,
            best_score=seed_score,
            representative_solutions=representative_solutions,
        ).with_inputs(
            "problem_description",
            "function_signature",
            "best_score",
            "representative_solutions",
        )
        examples.append(example)
    return examples


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def _config_hash(config: LeviConfig) -> str:
    """Deterministic hash of config fields that affect prompt optimization."""
    payload = json.dumps(
        {
            "problem_description": config.problem_description,
            "function_signature": config.function_signature,
            "seed_program": config.seed_program,
            "mutation_models": sorted(config.mutation_models),
            "paradigm_models": sorted(config.paradigm_models),
            "n_trials": config.prompt_opt.n_trials,
            "num_candidates": config.prompt_opt.num_candidates,
            "optimize_mutation": config.prompt_opt.optimize_mutation,
            "optimize_paradigm_shift": config.prompt_opt.optimize_paradigm_shift,
            "teacher_model": config.prompt_opt.teacher_model,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _cache_path(config: LeviConfig) -> Path:
    cache_dir = config.prompt_opt.cache_dir or "."
    return Path(cache_dir) / "optimized_prompts.json"


def _load_cache(config: LeviConfig) -> Optional[dict]:
    """Load cached prompts if config hash matches."""
    path = _cache_path(config)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        stored_hash = data.get("metadata", {}).get("config_hash")
        if stored_hash == _config_hash(config):
            logger.info("[PromptOpt] Loaded cached prompts from %s", path)
            return data
        logger.info("[PromptOpt] Cache hash mismatch, re-optimizing")
        return None
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("[PromptOpt] Failed to load cache: %s", exc)
        return None


def _save_cache(overrides: dict, config: LeviConfig) -> None:
    """Save optimized prompts with config hash."""
    path = _cache_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        **overrides,
        "metadata": {
            "config_hash": _config_hash(config),
            "timestamp": datetime.now().isoformat(),
        },
    }
    path.write_text(json.dumps(data, indent=2))
    logger.info("[PromptOpt] Saved optimized prompts to %s", path)


# ---------------------------------------------------------------------------
# Cost tracking wrapper
# ---------------------------------------------------------------------------


class _CostTracker:
    """Accumulates litellm completion costs from DSPy calls."""

    def __init__(self):
        self.total: float = 0.0
        self._original_request = None

    def start(self) -> None:
        try:
            import litellm

            self._original_request = litellm.success_callback
            litellm.success_callback = [self._on_success]
        except Exception:
            pass

    def stop(self) -> None:
        try:
            import litellm

            if self._original_request is not None:
                litellm.success_callback = self._original_request
            else:
                litellm.success_callback = []
        except Exception:
            pass

    def _on_success(self, kwargs, completion_response, start_time, end_time):
        try:
            from litellm import completion_cost

            cost = completion_cost(completion_response=completion_response)
            self.total += cost
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def optimize_prompts(config: LeviConfig) -> tuple[dict, float]:
    """Run DSPy prompt optimization based on ``config.prompt_opt``.

    Returns ``(prompt_overrides, total_cost)`` where ``prompt_overrides``
    has the same shape consumed by ``config.prompt_overrides``:

    .. code-block:: python

        {
            "mutation": {"model_name": "optimized instruction", ...},
            "paradigm_shift": "optimized instruction" | None,
        }
    """
    from dspy.teleprompt import MIPROv2

    opt_cfg = config.prompt_opt

    # 0. Check cache (unless force=True)
    if not opt_cfg.force:
        cached = _load_cache(config)
        if cached is not None:
            overrides = {
                "mutation": cached.get("mutation", {}),
                "paradigm_shift": cached.get("paradigm_shift"),
            }
            return overrides, 0.0

    _configure_dspy_tips()

    fn_name = extract_fn_name(config.function_signature)
    cost_tracker = _CostTracker()
    cost_tracker.start()

    # 1. Evaluate seed to get baseline score
    logger.info("[PromptOpt] Evaluating seed program")
    seed_result = evaluate_code(config.seed_program, config.score_fn, config.inputs, fn_name)
    if "error" in seed_result:
        raise RuntimeError(f"Seed program evaluation failed during prompt optimization: {seed_result['error']}")
    seed_score = seed_result.get("score", 0.0)
    logger.info("[PromptOpt] Seed score: %.17g", seed_score)

    overrides: dict[str, Any] = {"mutation": {}, "paradigm_shift": None}

    # 2. Optimize mutation prompts (one per mutation model)
    if opt_cfg.optimize_mutation:
        trainset = _generate_mutation_examples(config, fn_name, seed_score)

        for model in config.mutation_models:
            logger.info("[PromptOpt] Optimizing mutation prompt for %s", model)

            target_lm = _make_dspy_lm(model, config)
            teacher_lm = _make_dspy_lm(opt_cfg.teacher_model, config)

            dspy.configure(lm=target_lm)

            def metric(example, prediction, trace=None, _cfg=config, _fn=fn_name):
                return _mutation_metric(example, prediction, _cfg, _fn, trace)

            optimizer = MIPROv2(
                metric=metric,
                auto=None,
                num_candidates=opt_cfg.num_candidates,
                verbose=True,
                num_threads=opt_cfg.num_threads,
                init_temperature=opt_cfg.init_temperature,
                teacher_settings=dict(lm=teacher_lm),
            )

            try:
                optimized_module = optimizer.compile(
                    MutationModule(),
                    trainset=trainset,
                    max_bootstrapped_demos=0,
                    max_labeled_demos=0,
                    num_trials=opt_cfg.n_trials,
                    minibatch=False,
                )

                for _name, predictor in optimized_module.named_predictors():
                    if hasattr(predictor, "signature"):
                        overrides["mutation"][model] = str(predictor.signature.instructions)
                        break
            except Exception as exc:
                logger.error("[PromptOpt] Failed to optimize mutation for %s: %s", model, exc)

    # 3. Optimize paradigm shift prompt (only if PE is enabled)
    if opt_cfg.optimize_paradigm_shift and config.punctuated_equilibrium.enabled:
        logger.info("[PromptOpt] Optimizing paradigm shift prompt")

        trainset = _generate_paradigm_shift_examples(config, seed_score)
        paradigm_model = config.paradigm_models[0]
        target_lm = _make_dspy_lm(paradigm_model, config)
        teacher_lm = _make_dspy_lm(opt_cfg.teacher_model, config)
        dspy.configure(lm=target_lm)

        def ps_metric(example, prediction, trace=None, _cfg=config, _fn=fn_name, _seed=config.seed_program):
            return _paradigm_shift_metric(example, prediction, _cfg, _fn, _seed, trace)

        optimizer = MIPROv2(
            metric=ps_metric,
            auto=None,
            num_candidates=opt_cfg.num_candidates,
            verbose=True,
            num_threads=max(1, opt_cfg.num_threads // 2),
            init_temperature=opt_cfg.init_temperature,
            teacher_settings=dict(lm=teacher_lm),
        )

        try:
            optimized_module = optimizer.compile(
                ParadigmShiftModule(),
                trainset=trainset,
                max_bootstrapped_demos=0,
                max_labeled_demos=0,
                num_trials=opt_cfg.n_trials,
                minibatch=False,
            )

            for _name, predictor in optimized_module.named_predictors():
                if hasattr(predictor, "signature"):
                    overrides["paradigm_shift"] = str(predictor.signature.instructions)
                    break
        except Exception as exc:
            logger.error("[PromptOpt] Failed to optimize paradigm shift: %s", exc)

    cost_tracker.stop()
    total_cost = cost_tracker.total

    # 4. Save cache
    _save_cache(overrides, config)

    logger.info("[PromptOpt] Optimization complete, cost: $%.3f", total_cost)
    return overrides, total_cost
