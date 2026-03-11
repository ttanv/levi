"""
PromptBuilder: Flexible, composable prompt construction.

Supports:
- Multiple parent programs
- Arbitrary sections with priorities
- Full code and diff output modes
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from ...core import EvaluationResult, Program


class OutputMode(Enum):
    FULL = "full"
    DIFF = "diff"


@dataclass
class ProgramWithScore:
    program: Program
    result: Optional[EvaluationResult] = None

    @property
    def score(self) -> str:
        return str(self.result.primary_score) if self.result else "N/A"


@dataclass
class PromptSection:
    name: str
    content: str
    priority: int = 50


class PromptBuilder:
    """
    Flexible prompt builder with composable sections.

    Example:
        builder = PromptBuilder()
        builder.add_section("Problem", "Optimize bin packing...", priority=10)
        builder.add_section("Signature", "def solve(items):", priority=20)
        builder.add_parents(parents, priority=30)
        builder.add_section("Niche", "High-complexity region", priority=35)
        builder.set_output_mode(OutputMode.DIFF)
        prompt = builder.build()
    """

    def __init__(self) -> None:
        self._sections: list[PromptSection] = []
        self._output_mode: OutputMode = OutputMode.FULL
        self._custom_output: Optional[str] = None

    def add_section(self, name: str, content: str, priority: int = 50) -> "PromptBuilder":
        """Add a custom section to the prompt."""
        self._sections.append(PromptSection(name, content, priority))
        return self

    def add_parents(
        self,
        parents: list[ProgramWithScore],
        priority: int = 50,
    ) -> "PromptBuilder":
        """Add parent programs as v1, v2, v3, etc."""
        for i, p in enumerate(parents):
            label = f"v{i + 1}"
            content = f"Score: {p.score}\n```python\n{p.program.content}\n```"
            self._sections.append(PromptSection(label, content, priority + i))
        return self

    def add_feedback(self, feedback: str, priority: int = 60) -> "PromptBuilder":
        """Add feedback/traces section."""
        self._sections.append(PromptSection("Feedback", feedback, priority))
        return self

    def set_output_mode(self, mode: OutputMode) -> "PromptBuilder":
        """Set how the LLM should output changes."""
        self._output_mode = mode
        return self

    def set_custom_output(self, instructions: str) -> "PromptBuilder":
        """Set custom output instructions (replaces default output mode)."""
        self._custom_output = instructions
        return self

    def build(self) -> str:
        """Build the final prompt string."""
        sorted_sections = sorted(self._sections, key=lambda s: s.priority)

        parts = []
        for section in sorted_sections:
            parts.append(f"## {section.name}\n{section.content}")

        # Use custom output if set, otherwise use output mode instructions
        if self._custom_output:
            parts.append(f"## Output\n{self._custom_output}")
        else:
            parts.append(f"## Output\n{self._output_instructions()}")

        return "\n\n".join(parts)

    def _output_instructions(self) -> str:
        if self._output_mode == OutputMode.FULL:
            return """Write an improved version of the function.

CRITICAL REQUIREMENTS:
1. Your code must be COMPLETE and RUNNABLE as a standalone file
2. Include ALL necessary imports at the top of your code
3. The function signature must match exactly what is specified
4. Ensure there are no syntax errors (matching parentheses, quotes, indentation)

Your response must follow this structure:
```python
# all necessary imports here

def function_name(...):
    # your complete implementation
    return ...
```

DO NOT include any explanation or text outside the code block.
DO NOT assume any imports are already available - include every import your code needs."""

        elif self._output_mode == OutputMode.DIFF:
            return """Output your improved code using SEARCH/REPLACE blocks.

FORMAT:
<<<<<<< SEARCH
exact lines to find
=======
replacement lines
>>>>>>> REPLACE

CRITICAL REQUIREMENTS:
1. The resulting code must be COMPLETE and RUNNABLE
2. Do NOT remove import statements unless replacing with different imports
3. Ensure your replacements maintain valid Python syntax
4. If adding new functionality that needs imports, add them with a separate SEARCH/REPLACE block

RULES:
1. Make SURGICAL changes - small, focused edits (5-20 lines max per block)
2. Copy the SEARCH section EXACTLY from the original (including whitespace/indentation)
3. Use multiple small SEARCH/REPLACE blocks instead of one large block
4. Start your response immediately with <<<<<<< SEARCH
5. Do NOT include any explanation or text outside the blocks
6. Do NOT use ```python code blocks

GOOD: Replace a single function or a few lines
BAD: Replace the entire file or 100+ lines at once"""

        return ""
