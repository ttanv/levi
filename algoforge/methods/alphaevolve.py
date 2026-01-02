"""
AlphaEvolve: High-throughput evolutionary search with CVT-MAP-Elites + Islands.

Key features:
- CVT-MAP-Elites maintains behavioral diversity
- Multiple islands prevent premature convergence
- LLM ensemble balances throughput and quality
- Periodic island culling removes stagnant populations
"""

import re
import time
from typing import Callable, Any, Optional

from ..core import Program, EvaluationResult
from ..budget import BudgetManager, BudgetExhausted, ResourceType
from ..pool import CVTMAPElitesPool
from ..evaluator import SandboxedEvaluator
from ..llm import LLMClient, LLMConfig, PromptBuilder, ProgramWithScore, OutputMode
from ..behavior import BehaviorExtractor
from .._config import get_config


def extract_code(response: str) -> Optional[str]:
    """Extract Python code from LLM response."""
    # Try ```python blocks first
    matches = re.findall(r'```python\s*(.*?)```', response, re.DOTALL)
    if matches:
        return matches[0].strip()

    # Try generic ``` blocks
    matches = re.findall(r'```\s*(.*?)```', response, re.DOTALL)
    if matches:
        return matches[0].strip()

    # Try raw code starting with common Python patterns
    stripped = response.strip()
    for pattern in ['def ', 'import ', 'from ', '# ', '"""', "'''"]:
        if stripped.startswith(pattern):
            return stripped

    # Last resort: find first line that looks like Python code
    for line in stripped.split('\n'):
        line = line.strip()
        if line.startswith(('def ', 'import ', 'from ', 'class ')):
            # Return from this line onwards
            idx = stripped.find(line)
            return stripped[idx:].strip()

    return None


def apply_diff(original: str, diff_response: str) -> Optional[str]:
    """Apply SEARCH/REPLACE diff blocks to original code."""
    result = original

    pattern = r'<<<<<<< SEARCH\s*(.*?)\s*=======\s*(.*?)\s*>>>>>>> REPLACE'
    matches = re.findall(pattern, diff_response, re.DOTALL)

    if not matches:
        return extract_code(diff_response)

    for search, replace in matches:
        search = search.strip()
        replace = replace.strip()
        if search in result:
            result = result.replace(search, replace, 1)
        else:
            return None

    return result


def alphaevolve(
    score_functions: dict[str, Callable[[Any, Any, float], float]],
    inputs: list,
    seed_program: str,
    problem_description: str,
    function_signature: str,
    budget_evaluations: Optional[int] = None,
    budget_dollars: Optional[float] = None,
    budget_seconds: Optional[float] = None,
    n_centroids: int = 100,
    n_islands: int = 10,
    n_parents: int = 2,
    epoch_interval: int = 100,
    use_diff_mode: bool = False,
    temperature: float = 0.7,
    verbose: bool = True,
) -> Program:
    """
    Run AlphaEvolve evolutionary optimization.

    Args:
        score_functions: Dict of {metric_name: fn(output, input, exec_time) -> score}
        inputs: List of inputs to evaluate programs on
        seed_program: Initial program code
        problem_description: Problem description for LLM
        function_signature: Function signature for LLM
        budget_evaluations: Max evaluations (optional)
        budget_dollars: Max LLM cost (optional)
        budget_seconds: Max wall time (optional)
        n_centroids: Number of CVT centroids per island
        n_islands: Number of independent islands
        n_parents: Parents to sample per mutation
        epoch_interval: Generations between island culling
        use_diff_mode: Whether to use text-based diff mutations
        temperature: LLM sampling temperature
        verbose: Print progress

    Returns:
        Best program found
    """
    config = get_config()
    start_time = time.time()

    # Create budget manager
    budget = BudgetManager(
        max_evaluations=budget_evaluations,
        max_llm_cost=budget_dollars,
        max_wall_time=budget_seconds,
    )

    # Create behavior extractor
    extractor = BehaviorExtractor()

    # Create CVT-MAP-Elites pool
    pool = CVTMAPElitesPool(
        behavior_extractor=extractor,
        n_centroids=n_centroids,
        n_islands=n_islands,
        temperature=1.0,
    )

    # Create evaluator
    evaluator = SandboxedEvaluator(
        budget_manager=budget,
        score_functions=score_functions,
        timeout=config.evaluation_timeout,
        memory_limit_mb=config.memory_limit_mb,
    )

    # Create LLM client
    llm = LLMClient(
        budget_manager=budget,
        models=config.models,
        default_config=LLMConfig(temperature=temperature),
    )

    # Evaluate and add seed to all islands
    seed = Program(code=seed_program)
    seed_result = evaluator.evaluate(seed, inputs)

    if not seed_result.is_valid:
        raise ValueError(f"Seed program invalid: {seed_result.error}")

    for island_idx in range(n_islands):
        seeded = Program(
            code=seed_program,
            metadata={"island_index": island_idx},
        )
        pool.add(seeded, seed_result)

    if verbose:
        print(f"{'='*60}")
        print(f"AlphaEvolve Started")
        print(f"{'='*60}")
        print(f"  Seed score:  {seed_result.primary_score:.4f}")
        print(f"  Islands:     {n_islands}")
        print(f"  Centroids:   {n_centroids}")
        print(f"  Parents:     {n_parents}")
        print(f"  Diff mode:   {use_diff_mode}")
        print(f"{'='*60}")

    generation = 0
    best_score = seed_result.primary_score
    best_program = seed
    accepted = 0
    invalid = 0
    failed_extract = 0

    # Main evolution loop
    try:
        while not budget.is_exhausted():
            generation += 1
            gen_start = time.time()

            # Sample parents from pool
            try:
                sample = pool.sample(n_parents=n_parents)
            except ValueError:
                if verbose:
                    print("[ERROR] Pool empty, cannot sample")
                break

            parents = [sample.parent] + sample.inspirations
            island_index = sample.metadata["island_index"]

            # Build prompt
            builder = PromptBuilder()
            builder.add_section("Problem", problem_description, priority=10)
            builder.add_section("Signature", f"```python\n{function_signature}\n```", priority=20)

            parent_with_scores = [ProgramWithScore(p, None) for p in parents]
            builder.add_parents(parent_with_scores, priority=30)

            if use_diff_mode:
                builder.set_output_mode(OutputMode.DIFF)
            else:
                builder.add_section(
                    "Task",
                    "Write an improved version. Be creative - try different algorithms, "
                    "optimize for edge cases, or improve efficiency.",
                    priority=40
                )

            prompt = builder.build()

            # Generate mutation
            try:
                response = llm.generate(prompt)
                model_used = response.model
            except BudgetExhausted:
                if verbose:
                    print(f"[Gen {generation}] Budget exhausted during LLM call")
                break

            # Extract/apply code
            if use_diff_mode:
                new_code = apply_diff(parents[0].code, response.content)
            else:
                new_code = extract_code(response.content)

            if new_code is None:
                failed_extract += 1
                if verbose:
                    print(f"[Gen {generation}] Failed to extract code | "
                          f"Island {island_index} | Model: {model_used}")
                continue

            # Create child with island metadata
            child = Program(
                code=new_code,
                parents=tuple(p.id for p in parents),
                metadata={"island_index": island_index, "generation": generation},
            )

            # Evaluate
            try:
                result = evaluator.evaluate(child, inputs)
            except BudgetExhausted:
                if verbose:
                    print(f"[Gen {generation}] Budget exhausted during evaluation")
                break

            gen_time = time.time() - gen_start
            elapsed = time.time() - start_time

            # Log generation result
            if result.is_valid:
                was_accepted = pool.add(child, result)
                score = result.primary_score

                if was_accepted:
                    accepted += 1
                    status = "ACCEPTED"

                    if score > best_score:
                        best_score = score
                        best_program = child
                        status = "NEW BEST"
                else:
                    status = "rejected"

                if verbose:
                    print(f"[Gen {generation:4d}] {status:10s} | "
                          f"Score: {score:7.4f} | Best: {best_score:7.4f} | "
                          f"Island {island_index:2d} | "
                          f"Elites: {pool.size():4d} | "
                          f"Time: {gen_time:.2f}s | "
                          f"Model: {model_used}")
            else:
                invalid += 1
                if verbose:
                    error_short = result.error[:40] if result.error else "unknown"
                    print(f"[Gen {generation:4d}] INVALID    | "
                          f"Error: {error_short}... | "
                          f"Island {island_index:2d} | "
                          f"Time: {gen_time:.2f}s")

            pool.on_generation_complete()

            # Periodic epoch (island culling)
            if generation % epoch_interval == 0:
                pool.on_epoch()
                stats = pool.get_stats()
                if verbose:
                    print(f"{'─'*60}")
                    print(f"[EPOCH] Generation {generation} | "
                          f"Elites: {stats['total_elites']} | "
                          f"Best: {best_score:.4f} | "
                          f"Elapsed: {elapsed:.1f}s")
                    print(f"        Island sizes: {stats['island_sizes']}")
                    print(f"{'─'*60}")

    except BudgetExhausted:
        if verbose:
            print(f"[INFO] Budget exhausted")

    total_time = time.time() - start_time
    stats = pool.get_stats()

    if verbose:
        print(f"{'='*60}")
        print(f"AlphaEvolve Complete")
        print(f"{'='*60}")
        print(f"  Generations:    {generation}")
        print(f"  Accepted:       {accepted}")
        print(f"  Invalid:        {invalid}")
        print(f"  Failed extract: {failed_extract}")
        print(f"  Total elites:   {stats['total_elites']}")
        print(f"  Best score:     {best_score:.4f}")
        print(f"  Total time:     {total_time:.1f}s")
        print(f"{'='*60}")

    return best_program
