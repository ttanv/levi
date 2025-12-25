"""
PromptBuilder: Flexible, composable prompt construction.

Supports:
- Multiple parent programs
- Arbitrary sections with priorities
- Diff mode and evolve-block mode
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from ...core import Program, EvaluationResult


class OutputMode(Enum):
    """How the LLM should output its changes."""
    FULL = "full"
    DIFF = "diff"
    EVOLVE_BLOCK = "evolve_block"
    STRUCTURED_DIFF = "structured_diff"


@dataclass
class ProgramWithScore:
    """A program paired with its evaluation result."""
    program: Program
    result: Optional[EvaluationResult] = None

    @property
    def score(self) -> str:
        return str(self.result.primary_score) if self.result else "N/A"


@dataclass
class PromptSection:
    """A section of the prompt with a priority for ordering."""
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

    def _add_line_numbers(self, code: str) -> str:
        """Add line numbers to code for LLM reference."""
        lines = code.split('\n')
        numbered = []
        for i, line in enumerate(lines, start=1):
            numbered.append(f"{i:4d} | {line}")
        return '\n'.join(numbered)

    def add_section(self, name: str, content: str, priority: int = 50) -> 'PromptBuilder':
        """Add a custom section to the prompt."""
        self._sections.append(PromptSection(name, content, priority))
        return self

    def add_parents(
        self,
        parents: list[ProgramWithScore],
        priority: int = 50,
    ) -> 'PromptBuilder':
        """Add parent programs as v1, v2, v3, etc."""
        for i, p in enumerate(parents):
            label = f"v{i + 1}"

            # Add line numbers for STRUCTURED_DIFF mode
            if self._output_mode == OutputMode.STRUCTURED_DIFF:
                code_display = self._add_line_numbers(p.program.code)
            else:
                code_display = p.program.code

            content = f"Score: {p.score}\n```python\n{code_display}\n```"
            self._sections.append(PromptSection(label, content, priority + i))
        return self

    def add_feedback(self, feedback: str, priority: int = 60) -> 'PromptBuilder':
        """Add feedback/traces section."""
        self._sections.append(PromptSection("Feedback", feedback, priority))
        return self

    def set_output_mode(self, mode: OutputMode) -> 'PromptBuilder':
        """Set how the LLM should output changes."""
        self._output_mode = mode
        return self

    def get_response_format(self) -> Optional[dict]:
        """
        Get JSON schema for structured output based on current mode.

        Returns:
            JSON schema dict for LiteLLM's response_format parameter, or None
            if current mode doesn't use structured outputs.
        """
        if self._output_mode == OutputMode.STRUCTURED_DIFF:
            from ..schemas import STRUCTURED_DIFF_SCHEMA
            return STRUCTURED_DIFF_SCHEMA
        return None

    def build(self) -> str:
        """Build the final prompt string."""
        sorted_sections = sorted(self._sections, key=lambda s: s.priority)

        parts = []
        for section in sorted_sections:
            parts.append(f"## {section.name}\n{section.content}")

        parts.append(f"## Output\n{self._output_instructions()}")

        return "\n\n".join(parts)

    def _output_instructions(self) -> str:
        if self._output_mode == OutputMode.FULL:
            return '''Write an improved version of the function.

IMPORTANT: Your response must contain ONLY a Python code block. No text before or after.

```python
def your_function(...):
    # your implementation
```

DO NOT include any explanation, commentary, or text outside the code block.'''

        elif self._output_mode == OutputMode.DIFF:
            return '''Output your improved code using SEARCH/REPLACE blocks.

FORMAT:
<<<<<<< SEARCH
exact lines to find
=======
replacement lines
>>>>>>> REPLACE

RULES:
1. Make SURGICAL changes - small, focused edits (5-20 lines max per block)
2. Copy the SEARCH section EXACTLY from the original (including whitespace/indentation)
3. Use multiple small SEARCH/REPLACE blocks instead of one large block
4. Start your response immediately with <<<<<<< SEARCH
5. Do NOT include any explanation or text outside the blocks
6. Do NOT use ```python code blocks

GOOD: Replace a single function or a few lines
BAD: Replace the entire file or 100+ lines at once'''

        elif self._output_mode == OutputMode.EVOLVE_BLOCK:
            return '''Modify ONLY the code between # EVOLVE-BLOCK START and # EVOLVE-BLOCK END.

Output the complete function in a ```python block. Keep all code outside the markers unchanged.

```python
def function(...):
    # unchanged code before
    # EVOLVE-BLOCK START
    # your changes here only
    # EVOLVE-BLOCK END
    # unchanged code after
```

DO NOT include any explanation outside the code block.'''

        elif self._output_mode == OutputMode.STRUCTURED_DIFF:
            return '''You will receive code with LINE NUMBERS (format: "  N | code").

Your task: Improve the code by making targeted edits.

CRITICAL RULES:
1. Line numbers are 1-indexed (first line is 1)
2. To replace lines 5-7, set start_line=5, end_line=7
3. To insert before line 3, set start_line=3, end_line=2
4. To delete lines, set new_content=""
5. Edits MUST NOT overlap
6. DO NOT include line numbers in new_content

Return JSON:
{
  "summary": "Brief description",
  "edits": [
    {
      "start_line": 5,
      "end_line": 7,
      "new_content": "    improved_code()\\n    return result",
      "explanation": "Why this improves the code"
    }
  ]
}

BEST PRACTICES:
- Make small, focused edits (surgical, not comprehensive)
- Preserve indentation in new_content
- List edits in order by start_line
- Explain your reasoning'''

        return ""

    def clear(self) -> 'PromptBuilder':
        """Clear all sections."""
        self._sections = []
        self._output_mode = OutputMode.FULL
        return self
