"""ARC-AGI-2 task-level building blocks: grid formatting, prompts, scoring."""

from collections.abc import Callable

FUNCTION_SIGNATURE = "def transform(input_grid: list[list[int]]) -> list[list[int]]:"

SEED_PROGRAM = """\
def transform(input_grid: list[list[int]]) -> list[list[int]]:
    return [row[:] for row in input_grid]
"""


def format_grid(grid: list[list[int]]) -> str:
    """Render grid as space-separated ints with a dimensions header."""
    rows, cols = len(grid), len(grid[0]) if grid else 0
    lines = [f"({rows}x{cols})"]
    for row in grid:
        lines.append(" ".join(str(v) for v in row))
    return "\n".join(lines)


def build_problem_description(task: dict) -> str:
    """Generate the LLM-facing prompt for a single ARC task."""
    parts = [
        "# ARC-AGI-2 Task",
        "",
        "You are solving an ARC-AGI-2 puzzle. Each puzzle defines a transformation",
        "from an input grid to an output grid. Grids are 2D arrays of integers 0-9",
        "(up to 30x30). You are given training examples showing input-output pairs.",
        "Your job is to figure out the transformation rule and implement it.",
        "",
        "## Training Examples",
    ]

    for i, example in enumerate(task["train"]):
        parts.append(f"\n### Example {i}")
        parts.append("Input:")
        parts.append(format_grid(example["input"]))
        parts.append("Output:")
        parts.append(format_grid(example["output"]))

    # Show test input dimensions so the LLM knows the scale
    test_input = task["test"][0]["input"]
    test_rows, test_cols = len(test_input), len(test_input[0]) if test_input else 0
    parts.extend([
        "",
        f"## Test Input Dimensions: {test_rows}x{test_cols}",
        "",
        "## Instructions",
        "- Study the training examples carefully to discover the pattern.",
        "- The output grid dimensions may differ from the input.",
        "- Use only Python standard library (no numpy, no external packages).",
        "- Your function receives a list[list[int]] and must return a list[list[int]].",
    ])

    return "\n".join(parts)


def compare_grids(
    predicted: list[list[int]], expected: list[list[int]]
) -> tuple[int, int]:
    """Cell-by-cell comparison handling size mismatches.

    Returns (correct_cells, total_cells) where total_cells is from expected.
    """
    exp_rows = len(expected)
    exp_cols = len(expected[0]) if expected else 0
    total = exp_rows * exp_cols

    pred_rows = len(predicted) if predicted else 0
    pred_cols = len(predicted[0]) if predicted else 0

    correct = 0
    for r in range(exp_rows):
        for c in range(exp_cols):
            if r < pred_rows and c < pred_cols and predicted[r][c] == expected[r][c]:
                correct += 1

    return correct, total


class ArcScorer:
    """Picklable scorer for one ARC task (needed for multiprocess evaluation)."""

    def __init__(self, train_examples: list[dict]) -> None:
        self.train_examples = train_examples

    def __call__(self, transform_fn, _inputs=None) -> dict:
        total_correct = 0
        total_cells = 0
        exact_matches = 0
        per_example = {}

        for i, example in enumerate(self.train_examples):
            inp = example["input"]
            expected = example["output"]

            try:
                predicted = transform_fn(inp)
            except Exception:
                predicted = []

            if not isinstance(predicted, list):
                predicted = []

            correct, total = compare_grids(predicted, expected)
            total_correct += correct
            total_cells += total

            is_exact = predicted == expected
            if is_exact:
                exact_matches += 1
            per_example[f"train_{i}"] = is_exact

        n = len(self.train_examples)
        cell_accuracy = total_correct / total_cells if total_cells > 0 else 0.0
        exact_ratio = exact_matches / n if n > 0 else 0.0

        # Score: cell accuracy × 40 + exact match ratio × 60 (0-100 scale)
        score = cell_accuracy * 40 + exact_ratio * 60

        result = {
            "score": score,
            "cell_accuracy": cell_accuracy,
            "exact_ratio": exact_ratio,
            "exact_matches": exact_matches,
        }
        result.update(per_example)
        return result


def make_score_fn(train_examples: list[dict]) -> ArcScorer:
    """Create a picklable scorer for one ARC task."""
    return ArcScorer(train_examples)
