#!/usr/bin/env python3
"""Run Levi on ARC-AGI-2: evolve a transform function per task."""

import argparse
import json
import os
import random
import sys
import types
from datetime import datetime
from pathlib import Path

import levi
from problem import (
    FUNCTION_SIGNATURE,
    SEED_PROGRAM,
    build_problem_description,
    make_score_fn,
)

DEFAULT_MODEL = "openrouter/qwen/qwen3-30b-a3b-instruct-2507"

QWEN_LOCAL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
QWEN_LOCAL_ENDPOINT = "http://localhost:8001/v1"
QWEN_COST = {"input_cost_per_token": 0.0000001, "output_cost_per_token": 0.0000004}


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def resolve_data_dir(split: str) -> Path:
    """Locate the ARC-AGI-2 data directory for the given split."""
    root = os.environ.get("ARC_AGI_2_DATA_ROOT")
    if root:
        return Path(root) / split
    return Path(__file__).resolve().parent / "data" / split


def load_tasks(data_dir: Path) -> dict[str, dict]:
    """Read all *.json task files from a directory."""
    tasks = {}
    if not data_dir.is_dir():
        print(f"Error: data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)
    for path in sorted(data_dir.glob("*.json")):
        with open(path) as f:
            tasks[path.stem] = json.load(f)
    return tasks


def _max_grid_size(task: dict) -> int:
    """Return the largest grid dimension across all examples in a task."""
    sizes = []
    for example in task.get("train", []) + task.get("test", []):
        for key in ("input", "output"):
            grid = example.get(key, [])
            if grid:
                sizes.append(max(len(grid), len(grid[0])))
    return max(sizes) if sizes else 0


# ---------------------------------------------------------------------------
# Safe execution helpers
# ---------------------------------------------------------------------------

def _safe_transform(code: str, input_grid: list[list[int]]) -> list[list[int]]:
    """Exec code, call transform, fallback to identity on error."""
    try:
        namespace: dict = {}
        exec(code, namespace)
        fn = namespace.get("transform")
        if not isinstance(fn, types.FunctionType):
            return [row[:] for row in input_grid]
        result = fn(input_grid)
        if isinstance(result, list):
            return result
    except Exception:
        pass
    return [row[:] for row in input_grid]


def get_top_n_programs(snapshot_path: Path, n: int = 2) -> list[str]:
    """Load Levi snapshot, extract top-n distinct programs by score."""
    with open(snapshot_path) as f:
        snapshot = json.load(f)

    elites = snapshot.get("elites", [])
    # Already sorted by primary_score descending in snapshot
    seen_codes: set[str] = set()
    programs: list[str] = []
    for elite in elites:
        code = elite["code"]
        if code not in seen_codes:
            seen_codes.add(code)
            programs.append(code)
            if len(programs) >= n:
                break

    # Fallback: pad with identity if we don't have enough distinct programs
    while len(programs) < n:
        programs.append(SEED_PROGRAM)

    return programs


# ---------------------------------------------------------------------------
# Per-task evolution
# ---------------------------------------------------------------------------

def evolve_task(
    task_id: str,
    task: dict,
    model: str | None,
    paradigm_model: str | None,
    mutation_model: str | list[str] | None,
    budget_dollars: float | None,
    budget_seconds: float | None,
    output_dir: Path,
) -> dict:
    """Run Levi evolution for a single ARC task. Returns summary dict."""
    problem_desc = build_problem_description(task)
    train_examples = task["train"]
    score_fn = make_score_fn(train_examples)

    # Build score_keys from training examples for behavioral diversity
    score_keys = [f"train_{i}" for i in range(len(train_examples))]

    task_output_dir = str(output_dir / task_id)

    # Model routing: single model or paradigm/mutation split
    model_kwargs: dict = {}
    if paradigm_model or mutation_model:
        if paradigm_model:
            model_kwargs["paradigm_model"] = paradigm_model
        if mutation_model:
            model_kwargs["mutation_model"] = mutation_model
    else:
        model_kwargs["model"] = model or DEFAULT_MODEL

    # Wire local Qwen endpoint if available
    local_endpoints: dict[str, str] = {}
    model_info: dict[str, dict] = {}

    # Check all configured models for local Qwen
    all_models = []
    for v in model_kwargs.values():
        if isinstance(v, list):
            all_models.extend(v)
        elif v:
            all_models.append(v)

    if QWEN_LOCAL in all_models:
        local_endpoints[QWEN_LOCAL] = os.environ.get(
            "LEVI_LOCAL_ENDPOINT", QWEN_LOCAL_ENDPOINT
        )
        model_info[QWEN_LOCAL] = QWEN_COST

    extra_kwargs: dict = {}
    if local_endpoints:
        extra_kwargs["local_endpoints"] = local_endpoints
    if model_info:
        extra_kwargs["model_info"] = model_info

    # Budget: only pass non-None values
    budget_kwargs: dict = {}
    if budget_dollars is not None:
        budget_kwargs["budget_dollars"] = budget_dollars
    if budget_seconds is not None:
        budget_kwargs["budget_seconds"] = budget_seconds

    result = levi.evolve_code(
        problem_desc,
        function_signature=FUNCTION_SIGNATURE,
        seed_program=SEED_PROGRAM,
        score_fn=score_fn,
        **model_kwargs,
        **budget_kwargs,
        target_score=100.0,
        pipeline=levi.PipelineConfig(
            n_llm_workers=8,
            n_eval_processes=8,
            max_tokens=4096,
            eval_timeout=30,
        ),
        behavior=levi.BehaviorConfig(
            ast_features=["loop_count", "branch_count", "subscript_count"],
            score_keys=score_keys,
        ),
        cascade=levi.CascadeConfig(enabled=False),
        punctuated_equilibrium=levi.PunctuatedEquilibriumConfig(
            enabled=True, interval=8
        ),
        init=levi.InitConfig(n_diverse_seeds=3, n_variants_per_seed=12),
        cvt=levi.CVTConfig(n_centroids=30),
        output_dir=task_output_dir,
        **extra_kwargs,
    )

    # Extract Pass@2 predictions
    snapshot_path = Path(task_output_dir) / "snapshot.json"
    test_predictions = []
    if snapshot_path.exists():
        top_programs = get_top_n_programs(snapshot_path, n=2)
    else:
        top_programs = [result.best_program, SEED_PROGRAM]

    for test_example in task["test"]:
        test_input = test_example["input"]
        guesses = []
        for prog in top_programs:
            output = _safe_transform(prog, test_input)
            guesses.append({"output": output})
        test_predictions.append(guesses)

    return {
        "task_id": task_id,
        "best_score": result.best_score,
        "total_cost": result.total_cost,
        "total_evaluations": result.total_evaluations,
        "runtime_seconds": result.runtime_seconds,
        "test_predictions": test_predictions,
    }


# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Levi on ARC-AGI-2 tasks")
    parser.add_argument(
        "--split",
        default="training",
        choices=["training", "evaluation"],
        help="Dataset split (default: training)",
    )
    parser.add_argument(
        "--task-ids",
        nargs="+",
        help="Specific task IDs to run (overrides --max-tasks)",
    )
    parser.add_argument(
        "--budget-dollars",
        type=float,
        default=float(os.environ.get("LEVI_BUDGET_DOLLARS", "1.0")),
        help="Budget per task in dollars (default: $1.00, 0=no limit)",
    )
    parser.add_argument(
        "--budget-seconds",
        type=float,
        default=float(os.environ.get("LEVI_BUDGET_SECONDS", "120")),
        help="Time limit per task in seconds (default: 120, 0=no limit)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("LEVI_MODEL"),
        help="Single model for both paradigm and mutation",
    )
    parser.add_argument(
        "--paradigm-model",
        default=None,
        help="Model for paradigm shifts (creative/heavy)",
    )
    parser.add_argument(
        "--mutation-model",
        nargs="+",
        default=None,
        help="Model(s) for mutations (can specify multiple)",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Limit to N tasks",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Randomly sample tasks instead of picking smallest",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: runs/<timestamp>_arc_agi_2)",
    )
    args = parser.parse_args()

    # Resolve budget: 0 means no limit
    budget_dollars = args.budget_dollars if args.budget_dollars > 0 else None
    budget_seconds = args.budget_seconds if args.budget_seconds > 0 else None
    if budget_dollars is None and budget_seconds is None:
        parser.error("At least one of --budget-dollars or --budget-seconds must be > 0")

    # Resolve models
    if args.model and (args.paradigm_model or args.mutation_model):
        parser.error("Cannot use --model with --paradigm-model/--mutation-model")
    if not args.model and not args.paradigm_model and not args.mutation_model:
        args.model = DEFAULT_MODEL

    # Load dataset
    data_dir = resolve_data_dir(args.split)
    all_tasks = load_tasks(data_dir)
    print(f"Loaded {len(all_tasks)} tasks from {data_dir}")

    # Select tasks
    if args.task_ids:
        tasks = {tid: all_tasks[tid] for tid in args.task_ids if tid in all_tasks}
        missing = set(args.task_ids) - set(tasks)
        if missing:
            print(f"Warning: task IDs not found: {missing}", file=sys.stderr)
    elif args.random:
        all_ids = list(all_tasks)
        n = min(args.max_tasks or len(all_ids), len(all_ids))
        selected = random.sample(all_ids, n)
        tasks = {tid: all_tasks[tid] for tid in selected}
    else:
        # Sort by max grid size ascending (smallest/simplest first)
        sorted_ids = sorted(all_tasks, key=lambda tid: _max_grid_size(all_tasks[tid]))
        if args.max_tasks:
            sorted_ids = sorted_ids[: args.max_tasks]
        tasks = {tid: all_tasks[tid] for tid in sorted_ids}

    print(f"Running {len(tasks)} tasks (split={args.split})")

    # Output directory
    output_dir = Path(
        args.output_dir
        or f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}_arc_agi_2"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run tasks
    submission: dict[str, list] = {}
    results_summary: list[dict] = []
    total_cost = 0.0
    solved = 0

    for i, (task_id, task) in enumerate(tasks.items(), 1):
        print(f"\n{'='*60}")
        print(f"Task {i}/{len(tasks)}: {task_id}")
        print(f"  Train examples: {len(task['train'])}, Test examples: {len(task['test'])}")
        print(f"  Max grid size: {_max_grid_size(task)}")
        print(f"{'='*60}")

        try:
            result = evolve_task(
                task_id=task_id,
                task=task,
                model=args.model,
                paradigm_model=args.paradigm_model,
                mutation_model=args.mutation_model,
                budget_dollars=budget_dollars,
                budget_seconds=budget_seconds,
                output_dir=output_dir,
            )

            submission[task_id] = result["test_predictions"]
            results_summary.append(result)
            total_cost += result["total_cost"]

            is_solved = result["best_score"] >= 100.0
            if is_solved:
                solved += 1

            print(f"  Score: {result['best_score']:.1f}  Cost: ${result['total_cost']:.3f}  "
                  f"Evals: {result['total_evaluations']}  Time: {result['runtime_seconds']:.0f}s"
                  f"  {'SOLVED' if is_solved else ''}")

        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            # Write identity fallback for failed tasks
            fallback = []
            for test_example in task["test"]:
                identity = [row[:] for row in test_example["input"]]
                fallback.append([{"output": identity}, {"output": identity}])
            submission[task_id] = fallback

    # Write submission
    submission_path = output_dir / "submission.json"
    with open(submission_path, "w") as f:
        json.dump(submission, f)

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Tasks run:    {len(tasks)}")
    print(f"Solved:       {solved}/{len(tasks)}")
    print(f"Total cost:   ${total_cost:.2f}")
    print(f"Submission:   {submission_path}")
    print(f"Output dir:   {output_dir}")


if __name__ == "__main__":
    main()
