"""Prompt artifact adapter for single-prompt evolution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..clients.base import ClientSpec
from ..config import LeviConfig
from ..core import Program
from ..prompts import ProgramWithScore
from ..utils import ResilientProcessPool, evaluate_prompt
from .base import ArtifactAdapter

PROMPT_MUTATION_TEMPLATE = """# Prompt Optimization

## Objective
{problem_description}

## Current Best Prompt
Score: {parent_score}
--- PROMPT START ---
{parent_prompt}
--- PROMPT END ---

{inspirations_section}{meta_advice_section}## Task
Write a better prompt that improves the evaluation score.
Keep the same overall task, but make the instructions clearer, stronger, or more effective.

## Output
Output only the rewritten prompt text. Do not add explanation or markdown fences.
"""

PROMPT_INIT_TEMPLATE = """# Prompt Initialization

## Objective
{problem_description}

{existing_section}## Task
Write one high-quality prompt candidate for this objective.
It should be meaningfully different from the existing prompts when they are provided.

## Output
Output only the prompt text. Do not add explanation or markdown fences.
"""

PROMPT_PARADIGM_TEMPLATE = """# Prompt Paradigm Shift

## Objective
{problem_description}

## Current Representatives
{representatives}

## Task
Write a fundamentally different prompt strategy that could outperform the current prompt family.
Change the framing, structure, or reasoning strategy substantially rather than making a tiny edit.

## Output
Output only the rewritten prompt text. Do not add explanation or markdown fences.
"""

PROMPT_VARIANT_TEMPLATE = """# Prompt Local Variation

## Objective
{problem_description}

## Base Prompt
Score: {base_score}
--- PROMPT START ---
{base_prompt}
--- PROMPT END ---

## Task
Write one improved variation of this prompt.

## Output
Output only the rewritten prompt text. Do not add explanation or markdown fences.
"""


def _strip_prompt_response(response_text: str) -> str | None:
    text = response_text.strip()
    if not text:
        return None

    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2:
            inner = "\n".join(lines[1:-1]).strip()
            return inner or None

    return text


class PromptAdapter(ArtifactAdapter):
    """Adapter for single-prompt evolution."""

    artifact_type = "prompt"

    def __init__(self, config: LeviConfig):
        self.config = config

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
            evaluate_prompt,
            content,
            self.config.score_fn,
            self.config.inputs if inputs is None else inputs,
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
        parent = parents[0]
        parent_score = parent.result.primary_score if parent.result is not None else "N/A"

        inspirations = list(parents[1:])
        inspirations_section = ""
        if inspirations:
            parts = []
            for idx, inspiration in enumerate(inspirations, start=1):
                score = inspiration.result.primary_score if inspiration.result is not None else "N/A"
                parts.append(
                    f"### Inspiration {idx}\nScore: {score}\n--- PROMPT START ---\n"
                    f"{inspiration.program.content}\n--- PROMPT END ---"
                )
            inspirations_section = "## Inspiration Prompts\n" + "\n\n".join(parts) + "\n\n"

        meta_advice_section = f"## Meta-Advice\n{meta_advice}\n\n" if meta_advice else ""
        return PROMPT_MUTATION_TEMPLATE.format(
            problem_description=self.config.problem_description,
            parent_score=parent_score,
            parent_prompt=parent.program.content,
            inspirations_section=inspirations_section,
            meta_advice_section=meta_advice_section,
        )

    def extract_candidate(
        self,
        response_text: str,
        *,
        parent_content: str | None = None,
        use_diff: bool = False,
    ) -> str | None:
        del parent_content, use_diff
        return _strip_prompt_response(response_text)

    def build_diversity_prompt(self, existing_candidates: Sequence[tuple[str, float]]) -> str:
        if existing_candidates:
            lines = []
            for idx, (content, score) in enumerate(existing_candidates, start=1):
                lines.append(
                    f"### Prompt {idx}\nScore: {score}\n--- PROMPT START ---\n{content}\n--- PROMPT END ---"
                )
            existing_section = "## Existing Prompts\n" + "\n\n".join(lines) + "\n\n"
        else:
            existing_section = ""

        return PROMPT_INIT_TEMPLATE.format(
            problem_description=self.config.problem_description,
            existing_section=existing_section,
        )

    def build_init_variant_prompt(self, parents: Sequence[ProgramWithScore]) -> str:
        return self.build_mutation_prompt(parents)

    def build_paradigm_shift_prompt(
        self,
        representatives: Sequence[tuple[int, Any]],
        *,
        n_evaluations: int,
        budget_progress: float = 0.0,
    ) -> str:
        del n_evaluations, budget_progress
        parts = []
        for idx, (cluster_id, elite) in enumerate(representatives, start=1):
            parts.append(
                f"### Region {idx} (Cluster {cluster_id})\n"
                f"Score: {elite.result.primary_score}\n"
                f"--- PROMPT START ---\n{elite.program.content}\n--- PROMPT END ---"
            )

        return PROMPT_PARADIGM_TEMPLATE.format(
            problem_description=self.config.problem_description,
            representatives="\n\n".join(parts),
        )

    def build_variant_prompt(self, base_content: str, base_score: float) -> str:
        return PROMPT_VARIANT_TEMPLATE.format(
            problem_description=self.config.problem_description,
            base_prompt=base_content,
            base_score=base_score,
        )
