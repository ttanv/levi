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
            content = f"Score: {p.score}\n```python\n{p.program.code}\n```"
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
            return '''Choose ONE format (do NOT mix them):

OPTION 1 - SEARCH/REPLACE blocks (for small changes):
<<<<<<< SEARCH
exact code to find
=======
replacement code
>>>>>>> REPLACE

OPTION 2 - Full code block (for large changes):
```python
# complete rewritten code here
```

CRITICAL:
- Pick ONE format, use it for your ENTIRE response
- NEVER put ```python inside SEARCH/REPLACE blocks
- NEVER put <<<<<<< markers inside code blocks
- Start immediately with <<<<<<< SEARCH or ```python'''

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

        return ""

    def clear(self) -> 'PromptBuilder':
        """Clear all sections."""
        self._sections = []
        self._output_mode = OutputMode.FULL
        return self
