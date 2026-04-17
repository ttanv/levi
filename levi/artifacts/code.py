"""Code artifact adapter for Levi's existing public API."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from ..clients.base import ClientSpec, client_name
from ..config import LeviConfig
from ..core import Program
from ..equilibrium.prompts import PARADIGM_SHIFT_PROMPTS, VARIANT_GENERATION_PROMPT, get_budget_stage
from ..prompts import OutputMode, ProgramWithScore, PromptBuilder
from ..utils import ResilientProcessPool, evaluate_code, extract_code, extract_fn_name
from .base import ArtifactAdapter

DIVERSITY_SEED_PROMPT = """# {problem_title}

## Problem
{problem_description}

## Function Signature
```python
{function_signature}
```

## Your Task: ALGORITHMIC DIVERSITY

You MUST design a solution using a **FUNDAMENTALLY DIFFERENT ALGORITHM** than the existing seeds.

**DO NOT:**
- Make minor variations or parameter tweaks to existing approaches
- Use the same core algorithm with different constants
- Reorder or refactor existing logic

**DO:**
- Analyze what algorithmic paradigm each existing seed uses
- Identify what aspects of the problem they exploit (or ignore)
- Design from first principles using a completely different strategy
- Think about what information in the problem they are NOT using
- Consider entirely different ways to model or decompose the problem

The goal is to explore different regions of the algorithm design space. A population of diverse algorithms will outperform a population of similar ones.

## Existing Seeds (analyze their algorithms, then do something DIFFERENT):
{existing_seeds}

## Output
Output ONLY the complete Python code in a ```python block.
"""


def apply_diff(original: str, diff_response: str) -> str | None:
    """Apply SEARCH/REPLACE diff blocks to original code."""
    result = original

    pattern = r"<<<<<<< SEARCH\s*(.*?)\s*=======\s*(.*?)\s*>>>>>>> REPLACE"
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


class CodeAdapter(ArtifactAdapter):
    """Adapter for Levi's existing code-evolution behavior."""

    artifact_type = "code"

    def __init__(self, config: LeviConfig):
        self.config = config
        self.fn_name = extract_fn_name(config.function_signature)

    def make_program(self, content: str, metadata: dict[str, Any] | None = None) -> Program:
        return Program(content=content, metadata=metadata or {})

    def snapshot_content(self, elite_data: Mapping[str, Any]) -> str:
        content = elite_data.get("content")
        if isinstance(content, str):
            return content

        legacy_code = elite_data.get("code")
        if isinstance(legacy_code, str):
            return legacy_code

        raise KeyError("content")

    async def evaluate(
        self,
        executor: ResilientProcessPool,
        content: str,
        *,
        inputs: list[Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        return await executor.run(
            evaluate_code,
            content,
            self.config.score_fn,
            self.config.inputs if inputs is None else inputs,
            self.fn_name,
            timeout=self.config.pipeline.eval_timeout if timeout is None else timeout,
        )

    def build_mutation_prompt(
        self,
        parents: Sequence[ProgramWithScore],
        *,
        meta_advice: str | None = None,
        model: ClientSpec | None = None,
        use_diff: bool = False,
    ) -> str:
        builder = PromptBuilder()
        builder.add_section("Problem", self.config.problem_description, priority=10)
        builder.add_section("Signature", f"```python\n{self.config.function_signature}\n```", priority=20)
        builder.add_parents(list(parents), priority=30)

        mutation_overrides = self.config.prompt_overrides.get("mutation", {})
        model_key = client_name(model) if model is not None else None
        if model_key and model_key in mutation_overrides:
            builder.set_custom_output(mutation_overrides[model_key])
        else:
            builder.set_output_mode(OutputMode.DIFF if use_diff else OutputMode.FULL)

        if meta_advice:
            builder.add_section("Meta-Advice", meta_advice, priority=100)

        return builder.build()

    def extract_candidate(
        self,
        response_text: str,
        *,
        parent_content: str | None = None,
        use_diff: bool = False,
    ) -> str | None:
        if use_diff:
            if parent_content is None:
                raise ValueError("parent_content is required when use_diff=True")
            return apply_diff(parent_content, response_text)
        return extract_code(response_text)

    def build_diversity_prompt(self, existing_candidates: Sequence[tuple[str, float]]) -> str:
        existing_seeds_text = "\n\n---\n\n".join(
            [
                f"### Seed {idx + 1} (Score: {score:.17g}):\n```python\n{content}\n```"
                for idx, (content, score) in enumerate(existing_candidates)
            ]
        )
        prompt_template = self.config.init.diversity_prompt or DIVERSITY_SEED_PROMPT
        return prompt_template.format(
            problem_title="Algorithm Optimization",
            problem_description=self.config.problem_description,
            function_signature=self.config.function_signature,
            existing_seeds=existing_seeds_text,
        )

    def build_init_variant_prompt(self, parents: Sequence[ProgramWithScore]) -> str:
        builder = PromptBuilder()
        builder.add_section("Problem", self.config.problem_description, priority=10)
        builder.add_section("Signature", f"```python\n{self.config.function_signature}\n```", priority=20)
        builder.add_parents(list(parents), priority=30)
        builder.set_output_mode(OutputMode.FULL)
        return builder.build()

    def build_paradigm_shift_prompt(
        self,
        representatives: Sequence[tuple[int, Any]],
        *,
        n_evaluations: int,
        budget_progress: float = 0.0,
    ) -> str:
        stage = get_budget_stage(budget_progress)

        rep_text_parts = []
        for idx, (cluster_id, elite) in enumerate(representatives):
            score = elite.result.primary_score
            content = elite.program.content
            rep_text_parts.append(
                f"### Region {idx + 1} (Cluster {cluster_id}, Score: {score:.17g})\n```python\n{content}\n```"
            )

        representative_solutions = "\n\n".join(rep_text_parts)

        override = self.config.prompt_overrides.get("paradigm_shift")
        if override:
            return f"""# Algorithmic Paradigm Shift Challenge

## Problem
{self.config.problem_description}

## Function Signature
```python
{self.config.function_signature}
```

## Current Best Solutions ({len(representatives)} regions, {n_evaluations} evaluations)

{representative_solutions}

## Your Task
{override}

Output ONLY complete, runnable Python code in a ```python block.
"""

        template = PARADIGM_SHIFT_PROMPTS[stage]
        return template.format(
            problem_description=self.config.problem_description,
            function_signature=self.config.function_signature,
            n_evaluations=n_evaluations,
            n_regions=len(representatives),
            representative_solutions=representative_solutions,
        )

    def build_variant_prompt(self, base_content: str, base_score: float) -> str:
        return VARIANT_GENERATION_PROMPT.format(
            problem_description=self.config.problem_description,
            function_signature=self.config.function_signature,
            base_code=base_content,
            base_score=base_score,
        )
