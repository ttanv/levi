"""
Tests for PromptBuilder: Flexible, composable prompt construction.

PromptBuilder is used to assemble prompts for LLM mutation generation.
"""

import pytest

from algoforge.core import Program, EvaluationResult
from algoforge.llm import PromptBuilder, ProgramWithScore, OutputMode


class TestOutputMode:
    """Tests for OutputMode enum."""

    def test_output_modes_exist(self):
        """All expected output modes are defined."""
        assert OutputMode.FULL.value == "full"
        assert OutputMode.DIFF.value == "diff"


class TestProgramWithScore:
    """Tests for ProgramWithScore helper class."""

    def test_creation_with_result(self):
        """ProgramWithScore can be created with a program and result."""
        prog = Program(code="def solve(x): return x")
        result = EvaluationResult(
            scores={"score": 0.85},
        )

        pws = ProgramWithScore(program=prog, result=result)

        assert pws.program == prog
        assert pws.result == result
        assert pws.score == "0.85"

    def test_creation_without_result(self):
        """ProgramWithScore can be created without a result."""
        prog = Program(code="def solve(x): return x")

        pws = ProgramWithScore(program=prog, result=None)

        assert pws.program == prog
        assert pws.result is None
        assert pws.score == "N/A"

    def test_score_property_uses_primary_score(self):
        """score property uses result's primary_score."""
        prog = Program(code="def solve(x): return x")
        result = EvaluationResult(
            scores={"accuracy": 0.9, "speed": 0.7},  # No 'score' key
        )

        pws = ProgramWithScore(program=prog, result=result)

        # primary_score returns first value when no 'score' key
        assert pws.score == "0.9"


class TestPromptBuilder:
    """Tests for PromptBuilder class."""

    def test_creation(self):
        """PromptBuilder can be created."""
        builder = PromptBuilder()

        # Should not raise
        prompt = builder.build()
        assert isinstance(prompt, str)

    def test_add_section(self):
        """add_section() adds content to the prompt."""
        builder = PromptBuilder()
        builder.add_section("Problem", "Optimize bin packing algorithm")

        prompt = builder.build()

        assert "## Problem" in prompt
        assert "Optimize bin packing algorithm" in prompt

    def test_add_section_with_priority(self):
        """Sections are ordered by priority."""
        builder = PromptBuilder()
        builder.add_section("Second", "This comes second", priority=20)
        builder.add_section("First", "This comes first", priority=10)
        builder.add_section("Third", "This comes third", priority=30)

        prompt = builder.build()

        first_idx = prompt.index("## First")
        second_idx = prompt.index("## Second")
        third_idx = prompt.index("## Third")

        assert first_idx < second_idx < third_idx

    def test_add_section_chainable(self):
        """add_section() returns self for chaining."""
        builder = PromptBuilder()

        result = builder.add_section("Test", "Content")

        assert result is builder

    def test_add_parents(self):
        """add_parents() adds parent programs to prompt."""
        builder = PromptBuilder()
        prog1 = Program(code="def solve(x): return x")
        prog2 = Program(code="def solve(x): return x * 2")

        parents = [
            ProgramWithScore(prog1, None),
            ProgramWithScore(prog2, None),
        ]
        builder.add_parents(parents)

        prompt = builder.build()

        assert "## v1" in prompt
        assert "## v2" in prompt
        assert "def solve(x): return x" in prompt
        assert "def solve(x): return x * 2" in prompt

    def test_add_parents_with_scores(self):
        """add_parents() includes scores when available."""
        builder = PromptBuilder()
        prog = Program(code="def solve(x): return x")
        result = EvaluationResult(
            scores={"score": 0.75},
        )

        parents = [ProgramWithScore(prog, result)]
        builder.add_parents(parents)

        prompt = builder.build()

        assert "Score: 0.75" in prompt

    def test_add_parents_with_priority(self):
        """add_parents() respects priority ordering."""
        builder = PromptBuilder()
        prog = Program(code="def solve(x): return x")

        builder.add_section("Before", "Before parents", priority=10)
        builder.add_parents([ProgramWithScore(prog, None)], priority=20)
        builder.add_section("After", "After parents", priority=30)

        prompt = builder.build()

        before_idx = prompt.index("## Before")
        v1_idx = prompt.index("## v1")
        after_idx = prompt.index("## After")

        assert before_idx < v1_idx < after_idx

    def test_add_parents_chainable(self):
        """add_parents() returns self for chaining."""
        builder = PromptBuilder()

        result = builder.add_parents([])

        assert result is builder

    def test_add_feedback(self):
        """add_feedback() adds feedback section."""
        builder = PromptBuilder()
        builder.add_feedback("The previous attempt timed out on large inputs.")

        prompt = builder.build()

        assert "## Feedback" in prompt
        assert "The previous attempt timed out" in prompt

    def test_add_feedback_chainable(self):
        """add_feedback() returns self for chaining."""
        builder = PromptBuilder()

        result = builder.add_feedback("Some feedback")

        assert result is builder

    def test_set_output_mode_full(self):
        """set_output_mode(FULL) sets full rewrite instructions."""
        builder = PromptBuilder()
        builder.set_output_mode(OutputMode.FULL)

        prompt = builder.build()

        assert "## Output" in prompt
        assert "improved version" in prompt.lower()
        assert "```python" in prompt.lower()

    def test_set_output_mode_diff(self):
        """set_output_mode(DIFF) sets SEARCH/REPLACE instructions."""
        builder = PromptBuilder()
        builder.set_output_mode(OutputMode.DIFF)

        prompt = builder.build()

        assert "## Output" in prompt
        assert "SEARCH" in prompt
        assert "REPLACE" in prompt
        assert "<<<<<<< SEARCH" in prompt
        assert "=======" in prompt
        assert ">>>>>>> REPLACE" in prompt

    def test_set_output_mode_chainable(self):
        """set_output_mode() returns self for chaining."""
        builder = PromptBuilder()

        result = builder.set_output_mode(OutputMode.DIFF)

        assert result is builder

    def test_build_empty(self):
        """build() works with no sections added."""
        builder = PromptBuilder()

        prompt = builder.build()

        # Should at least have output instructions
        assert "## Output" in prompt

    def test_build_complete_prompt(self):
        """build() assembles a complete prompt correctly."""
        builder = PromptBuilder()
        prog = Program(code="def solve(x): return x")

        builder.add_section("Problem", "Optimize for speed", priority=10)
        builder.add_section("Signature", "def solve(x) -> int:", priority=20)
        builder.add_parents([ProgramWithScore(prog, None)], priority=30)
        builder.add_feedback("Previous version was too slow", priority=40)
        builder.set_output_mode(OutputMode.DIFF)

        prompt = builder.build()

        # Check ordering
        sections = ["Problem", "Signature", "v1", "Feedback", "Output"]
        indices = [prompt.index(f"## {s}") for s in sections]
        assert indices == sorted(indices)

        # Check content
        assert "Optimize for speed" in prompt
        assert "def solve(x) -> int:" in prompt
        assert "def solve(x): return x" in prompt
        assert "too slow" in prompt
        assert "SEARCH" in prompt

    def test_chained_building(self):
        """All methods can be chained together."""
        prog = Program(code="def solve(x): return x")

        prompt = (
            PromptBuilder()
            .add_section("Problem", "Test problem", priority=10)
            .add_section("Signature", "def solve(x):", priority=20)
            .add_parents([ProgramWithScore(prog, None)], priority=30)
            .add_feedback("Test feedback", priority=40)
            .set_output_mode(OutputMode.DIFF)
            .build()
        )

        assert "## Problem" in prompt
        assert "## Signature" in prompt
        assert "## v1" in prompt
        assert "## Feedback" in prompt
        assert "## Output" in prompt

    def test_code_block_formatting(self):
        """Parent code is wrapped in markdown code blocks."""
        builder = PromptBuilder()
        prog = Program(code="def solve(x):\n    return x * 2")

        builder.add_parents([ProgramWithScore(prog, None)])

        prompt = builder.build()

        assert "```python" in prompt
        assert "```" in prompt
        assert "def solve(x):" in prompt

    def test_multiple_parents_incrementing_labels(self):
        """Multiple parents get v1, v2, v3 labels."""
        builder = PromptBuilder()
        programs = [
            Program(code=f"def solve(x): return x + {i}")
            for i in range(3)
        ]

        parents = [ProgramWithScore(p, None) for p in programs]
        builder.add_parents(parents)

        prompt = builder.build()

        assert "## v1" in prompt
        assert "## v2" in prompt
        assert "## v3" in prompt


class TestPromptBuilderIntegration:
    """Integration tests for realistic prompt building scenarios."""

    def test_alphaevolve_style_prompt(self):
        """Builds a prompt similar to AlphaEvolve usage."""
        builder = PromptBuilder()

        # Problem description
        builder.add_section(
            "Problem",
            "Optimize a scheduling algorithm for transaction ordering.",
            priority=10
        )

        # Function signature
        builder.add_section(
            "Signature",
            "```python\ndef get_random_costs() -> tuple[float, list[list[int]], float]:\n```",
            priority=20
        )

        # Parent programs
        parent = Program(code='''def get_random_costs():
    schedules = []
    for w in workloads:
        schedule = list(range(w.num_txns))
        random.shuffle(schedule)
        schedules.append(schedule)
    total = sum(w.get_opt_seq_cost(s) for w, s in zip(workloads, schedules))
    return total, schedules, 0.0''')

        builder.add_parents(
            [ProgramWithScore(parent, EvaluationResult(
                scores={"score": 45.2},
            ))],
            priority=30
        )

        # Task instruction
        builder.add_section(
            "Task",
            "Write an improved version. Try different algorithms or optimize for edge cases.",
            priority=40
        )

        prompt = builder.build()

        # Verify structure
        assert "## Problem" in prompt
        assert "transaction ordering" in prompt
        assert "## Signature" in prompt
        assert "get_random_costs" in prompt
        assert "## v1" in prompt
        assert "Score: 45.2" in prompt
        assert "random.shuffle" in prompt
        assert "## Task" in prompt
        assert "improved version" in prompt.lower()

    def test_diff_mode_prompt(self):
        """Builds a prompt with DIFF output mode."""
        builder = PromptBuilder()

        parent = Program(code="def solve(x): return x * 2")

        builder.add_section("Problem", "Make the function faster", priority=10)
        builder.add_parents([ProgramWithScore(parent, None)], priority=20)
        builder.set_output_mode(OutputMode.DIFF)

        prompt = builder.build()

        assert "## Output" in prompt
        assert "<<<<<<< SEARCH" in prompt
        assert "=======" in prompt
        assert ">>>>>>> REPLACE" in prompt
        assert "SEARCH/REPLACE" in prompt

    def test_feedback_driven_prompt(self):
        """Builds a prompt with execution feedback."""
        builder = PromptBuilder()

        parent = Program(code='''def solve(items):
    result = []
    for item in items:
        result.append(item * 2)
    return result''')

        result = EvaluationResult(
            scores={"score": 0.6},
        )

        builder.add_section("Problem", "Optimize list processing", priority=10)
        builder.add_parents([ProgramWithScore(parent, result)], priority=20)
        builder.add_feedback(
            "The function is slow on large inputs. "
            "Consider using list comprehension or numpy.",
            priority=30
        )

        prompt = builder.build()

        assert "Score: 0.6" in prompt
        assert "## Feedback" in prompt
        assert "slow on large inputs" in prompt
        assert "list comprehension" in prompt
