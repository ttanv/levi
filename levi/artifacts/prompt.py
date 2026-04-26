"""Prompt artifact adapter for single-prompt and bundle evolution."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from ..clients.base import ClientSpec
from ..config import LeviConfig
from ..core import Program
from ..prompts import ProgramWithScore, PromptBundle
from ..utils import ResilientProcessPool, evaluate_prompt
from ..utils.evaluation import evaluate_bundle
from .base import ArtifactAdapter

OUTPUT_INSTRUCTIONS = (
    "Wrap the new prompt inside `<prompt>` tags exactly as shown. Put NOTHING outside the tags — "
    "no preamble, no explanation, no markdown fences, no quotes. Only the literal prompt text "
    "between the tags will be used.\n\n"
    "<prompt>\n"
    "your new prompt text here\n"
    "</prompt>"
)

PROMPT_MUTATION_TEMPLATE = """# Prompt Optimization

## Objective
{problem_description}

## Current Best Prompt
Score: {parent_score}
--- PROMPT START ---
{parent_prompt}
--- PROMPT END ---

{feedback_section}{inspirations_section}{meta_advice_section}## Task
Rewrite the prompt to score higher on the evaluation.

Treat the failures above as the only window you'll get into the task — anything they reveal about the inputs, desired outputs, failure modes, or strategies that worked must end up encoded in the new prompt itself, because the assistant won't see this feedback at evaluation time.

Be concrete. When the failures show specific terms, formats, edge cases, or substitution patterns ("replace X with Y"), bake representative examples and explicit rules into the prompt instead of abstract directives. When the feedback suggests a generalizable strategy that worked, name it as a reusable rule. Detailed, factually grounded prompts that capture observed patterns reliably outperform short generic ones.

Keep the same overall task and goal, but make the new prompt richer and more specific than the current best.

## Output
""" + OUTPUT_INSTRUCTIONS + "\n"

PROMPT_INIT_TEMPLATE = """# Prompt Initialization

## Objective
{problem_description}

{existing_section}## Task
Write one high-quality prompt candidate for this objective.
It should be meaningfully different from the existing prompts when they are provided.

## Output
""" + OUTPUT_INSTRUCTIONS + "\n"

PROMPT_PARADIGM_TEMPLATE = """# Prompt Paradigm Shift

## Objective
{problem_description}

## Current Representatives
{representatives}

## Task
Write a fundamentally different prompt strategy that could outperform the current prompt family.
Change the framing, structure, or reasoning strategy substantially rather than making a tiny edit.

## Output
""" + OUTPUT_INSTRUCTIONS + "\n"

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
""" + OUTPUT_INSTRUCTIONS + "\n"


BUNDLE_MUTATION_TEMPLATE = """# Prompt Optimization — component: {target}

## Objective
{problem_description}

## Current bundle (parent score {parent_score})
{bundle_section}
{feedback_section}{inspirations_section}{meta_advice_section}## Task
Rewrite ONLY the `{target}` component. The other components will stay fixed.

Treat the failures above as your only signal about how this component is performing. Anything they reveal — specific terms, formats, edge cases, substitution patterns ("replace X with Y"), or strategies that succeeded — should end up encoded in the new component, because the assistant won't see this feedback at evaluation time.

Be concrete. Prefer representative examples and explicit "X → Y" rules over abstract directives. When the feedback suggests a generalizable strategy, name it as a reusable rule. Detailed, factually grounded components reliably outperform short generic ones.

## Output
""" + OUTPUT_INSTRUCTIONS + "\n"

BUNDLE_DIVERSITY_TEMPLATE = """# Prompt Initialization — component: {target}

## Objective
{problem_description}

## Fixed components (read-only)
{readonly_section}

## Seed for `{target}`
--- PROMPT START ---
{seed_text}
--- PROMPT END ---

{existing_section}## Task
Write one high-quality candidate for the `{target}` component only.
It should be meaningfully different from the existing candidates when provided.

## Output
""" + OUTPUT_INSTRUCTIONS + "\n"

BUNDLE_VARIANT_TEMPLATE = """# Prompt Local Variation — component: {target}

## Objective
{problem_description}

## Fixed components (read-only)
{readonly_section}

## Base `{target}` (score {base_score})
--- PROMPT START ---
{base_text}
--- PROMPT END ---

## Task
Write one improved variation of the `{target}` component only.

## Output
""" + OUTPUT_INSTRUCTIONS + "\n"

BUNDLE_PARADIGM_TEMPLATE = """# Prompt Paradigm Shift — component: {target}

## Objective
{problem_description}

## Fixed components (read-only)
{readonly_section}

## Current `{target}` variants across top regions
{representatives}

## Task
Write a fundamentally different strategy for the `{target}` component.
Change framing, structure, or reasoning strategy substantially rather than making a tiny edit.
The other components stay fixed.

## Output
""" + OUTPUT_INSTRUCTIONS + "\n"


def _render_feedback_section(feedback: Sequence[str] | None) -> str:
    if not feedback:
        return ""
    parts = []
    for idx, entry in enumerate(feedback, start=1):
        parts.append(f"### Failure {idx}\n{entry}")
    return "## Feedback from recent failures\n" + "\n\n".join(parts) + "\n\n"


_THINK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.DOTALL | re.IGNORECASE)
_PROMPT_OPEN = re.compile(r"<prompt\b[^>]*>", re.IGNORECASE)
_PROMPT_CLOSE = re.compile(r"</prompt\s*>", re.IGNORECASE)
# Match a label line ending in "prompt:", "version:", "rewrite:", possibly bold/italic-wrapped.
# Anchored to start of a line to avoid matching arbitrary mid-sentence colons.
_LABEL_RE = re.compile(
    r"(?:^|\n)[^\n]*\b(?:prompt|version|rewrite|rewritten|optimized|here(?:'s| is) (?:the|my))\b[^\n]*[:：]\s*\*{0,2}\s*\n+",
    re.IGNORECASE,
)


def _unwrap_quotes(text: str) -> str:
    """Strip a single layer of surrounding quotes/backticks."""
    text = text.strip()
    for q in ('"""', "'''"):
        if text.startswith(q) and text.endswith(q) and len(text) > 2 * len(q):
            return text[len(q) : -len(q)].strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'", "`"}:
        return text[1:-1].strip()
    return text


def _strip_code_fence(text: str) -> str:
    """If text is wrapped in a ```...``` block, return the inside."""
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2:
            return "\n".join(lines[1:-1]).strip()
    return text


def _strip_prompt_response(response_text: str) -> str | None:
    """Extract the actual prompt text from an LLM response.

    Robust to: <think> blocks, <prompt> tags, code fences, "Optimized X prompt:"
    preambles, and surrounding quotes/backticks.
    """
    text = response_text.strip()
    if not text:
        return None

    # Strip Qwen/DeepSeek-style chain-of-thought blocks anywhere in the response.
    text = _THINK_RE.sub("", text).strip()
    if not text:
        return None

    # Preferred path: explicit <prompt>...</prompt> markers. Use the first
    # opening and the last closing so stray inner mentions don't break extraction.
    open_match = _PROMPT_OPEN.search(text)
    close_matches = list(_PROMPT_CLOSE.finditer(text))
    if open_match and close_matches:
        last_close = close_matches[-1]
        if last_close.start() > open_match.end():
            inner = text[open_match.end() : last_close.start()].strip()
            inner = _strip_code_fence(inner)
            inner = _unwrap_quotes(inner)
            return inner or None

    # Whole response is a fenced code block.
    if text.startswith("```") and text.endswith("```"):
        inner = _strip_code_fence(text)
        inner = _unwrap_quotes(inner)
        return inner or None

    # Take everything after the LAST "...prompt:" / "Here's my version:" label line.
    matches = list(_LABEL_RE.finditer(text))
    if matches:
        tail = text[matches[-1].end() :].strip()
        tail = _strip_code_fence(tail)
        tail = _unwrap_quotes(tail)
        if tail:
            return tail

    # Fallback: light cleanup only.
    return _unwrap_quotes(_strip_code_fence(text)) or None


def _render_readonly(bundle: PromptBundle, target: str) -> str:
    parts = []
    for name in bundle.target_names:
        if name == target:
            continue
        parts.append(
            f"### {name}\n--- PROMPT START ---\n{bundle.get(name)}\n--- PROMPT END ---"
        )
    return "\n\n".join(parts) if parts else "(none)"


def _render_bundle_section(bundle: PromptBundle, target: str) -> str:
    parts = [
        f"### [EDITABLE] {target}\n--- PROMPT START ---\n{bundle.get(target)}\n--- PROMPT END ---"
    ]
    for name in bundle.target_names:
        if name == target:
            continue
        parts.append(
            f"### [READ-ONLY] {name}\n--- PROMPT START ---\n{bundle.get(name)}\n--- PROMPT END ---"
        )
    return "\n\n".join(parts)


class PromptAdapter(ArtifactAdapter):
    """Adapter for single-prompt and multi-component prompt evolution."""

    artifact_type = "prompt"

    def __init__(self, config: LeviConfig, seed_bundle: PromptBundle | None = None):
        self.config = config
        self.seed_bundle = seed_bundle
        self._is_bundle = seed_bundle is not None and len(seed_bundle.prompts) > 1

    @property
    def is_bundle_artifact(self) -> bool:
        return self._is_bundle

    def _deserialize(self, content: str) -> PromptBundle:
        return PromptBundle.deserialize_loose(content)

    def _as_bundle_dict(self, content: str) -> dict[str, str]:
        return self._deserialize(content).as_dict()

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
        eval_inputs = self.config.inputs if inputs is None else inputs
        eval_timeout = self.config.pipeline.eval_timeout if timeout is None else timeout

        if self._is_bundle:
            bundle_dict = self._as_bundle_dict(content)
            return await executor.run(
                evaluate_bundle,
                bundle_dict,
                self.config.score_fn,
                eval_inputs,
                timeout=eval_timeout,
            )

        return await executor.run(
            evaluate_prompt,
            content,
            self.config.score_fn,
            eval_inputs,
            timeout=eval_timeout,
        )

    # ---------- Single-prompt templates ----------

    def _build_single_mutation_prompt(
        self,
        parents: Sequence[ProgramWithScore],
        *,
        meta_advice: str | None,
        feedback: Sequence[str] | None = None,
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
            feedback_section=_render_feedback_section(feedback),
            inspirations_section=inspirations_section,
            meta_advice_section=meta_advice_section,
        )

    # ---------- Bundle templates ----------

    def _build_bundle_mutation_prompt(
        self,
        parents: Sequence[ProgramWithScore],
        *,
        target: str,
        meta_advice: str | None,
        feedback: Sequence[str] | None = None,
    ) -> str:
        parent = parents[0]
        parent_score = parent.result.primary_score if parent.result is not None else "N/A"
        parent_bundle = self._deserialize(parent.program.content)

        bundle_section = _render_bundle_section(parent_bundle, target)

        inspirations_section = ""
        inspirations = list(parents[1:])
        if inspirations:
            parts = []
            for idx, inspiration in enumerate(inspirations, start=1):
                score = inspiration.result.primary_score if inspiration.result is not None else "N/A"
                try:
                    insp_bundle = self._deserialize(inspiration.program.content)
                    insp_text = insp_bundle.get(target)
                except (KeyError, ValueError):
                    insp_text = inspiration.program.content
                parts.append(
                    f"### Inspiration {idx} (`{target}`, score {score})\n"
                    f"--- PROMPT START ---\n{insp_text}\n--- PROMPT END ---"
                )
            inspirations_section = "## Inspiration Components\n" + "\n\n".join(parts) + "\n\n"

        meta_advice_section = f"## Meta-Advice\n{meta_advice}\n\n" if meta_advice else ""
        return BUNDLE_MUTATION_TEMPLATE.format(
            problem_description=self.config.problem_description,
            target=target,
            parent_score=parent_score,
            bundle_section=bundle_section,
            feedback_section=_render_feedback_section(feedback),
            inspirations_section=inspirations_section,
            meta_advice_section=meta_advice_section,
        )

    def build_mutation_prompt(
        self,
        parents: Sequence[ProgramWithScore],
        *,
        meta_advice: str | None = None,
        model: ClientSpec | None = None,
        use_diff: bool = False,
        target: str | None = None,
        feedback: Sequence[str] | None = None,
    ) -> str:
        del model, use_diff
        if self._is_bundle and target is not None:
            return self._build_bundle_mutation_prompt(
                parents, target=target, meta_advice=meta_advice, feedback=feedback
            )
        return self._build_single_mutation_prompt(parents, meta_advice=meta_advice, feedback=feedback)

    def extract_candidate(
        self,
        response_text: str,
        *,
        parent_content: str | None = None,
        use_diff: bool = False,
        target: str | None = None,
    ) -> str | None:
        del use_diff
        new_text = _strip_prompt_response(response_text)
        if not new_text:
            return None

        if self._is_bundle and target is not None:
            if parent_content is None:
                parent_bundle = self.seed_bundle
            else:
                parent_bundle = self._deserialize(parent_content)
            if parent_bundle is None:
                return None
            updated = parent_bundle.with_updates({target: new_text})
            return updated.serialize()

        return new_text

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
        return self._build_single_mutation_prompt(parents, meta_advice=None)

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

    # ---------- Component-scoped bundle methods (init + PE) ----------

    def build_component_diversity_prompt(
        self,
        target: str,
        base_bundle: PromptBundle,
        existing_candidates: Sequence[tuple[str, float]],
    ) -> str:
        readonly_section = _render_readonly(base_bundle, target)
        if existing_candidates:
            lines = []
            for idx, (text, score) in enumerate(existing_candidates, start=1):
                lines.append(
                    f"### Candidate {idx}\nScore: {score}\n--- PROMPT START ---\n{text}\n--- PROMPT END ---"
                )
            existing_section = "## Existing candidates for `{target}`\n".format(target=target) + "\n\n".join(lines) + "\n\n"
        else:
            existing_section = ""

        return BUNDLE_DIVERSITY_TEMPLATE.format(
            problem_description=self.config.problem_description,
            target=target,
            seed_text=base_bundle.get(target),
            readonly_section=readonly_section,
            existing_section=existing_section,
        )

    def build_component_variant_prompt(
        self,
        target: str,
        base_bundle: PromptBundle,
        base_score: float,
    ) -> str:
        return BUNDLE_VARIANT_TEMPLATE.format(
            problem_description=self.config.problem_description,
            target=target,
            readonly_section=_render_readonly(base_bundle, target),
            base_text=base_bundle.get(target),
            base_score=base_score,
        )

    def build_component_paradigm_shift_prompt(
        self,
        target: str,
        representatives: Sequence[tuple[int, Any]],
        *,
        n_evaluations: int,
        budget_progress: float = 0.0,
    ) -> str:
        del n_evaluations, budget_progress
        rep_parts = []
        anchor_bundle: PromptBundle | None = None
        for idx, (cluster_id, elite) in enumerate(representatives, start=1):
            try:
                bundle = self._deserialize(elite.program.content)
            except (KeyError, ValueError):
                continue
            if target not in bundle.target_names:
                continue
            if anchor_bundle is None:
                anchor_bundle = bundle
            rep_parts.append(
                f"### Region {idx} (Cluster {cluster_id}) — score {elite.result.primary_score}\n"
                f"--- PROMPT START ---\n{bundle.get(target)}\n--- PROMPT END ---"
            )

        if anchor_bundle is None:
            anchor_bundle = self.seed_bundle

        readonly_section = _render_readonly(anchor_bundle, target) if anchor_bundle else "(none)"

        return BUNDLE_PARADIGM_TEMPLATE.format(
            problem_description=self.config.problem_description,
            target=target,
            readonly_section=readonly_section,
            representatives="\n\n".join(rep_parts) if rep_parts else "(none)",
        )
