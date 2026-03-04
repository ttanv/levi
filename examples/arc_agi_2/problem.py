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
        "",
        "## Self-Verification (IMPORTANT)",
        "Before writing your solution, mentally trace through EACH training example:",
        "1. For each example, apply your rule to the input and check if EVERY row of",
        "   your output matches the expected output row exactly.",
        "2. If any row doesn't match, your rule is wrong. Revise it before coding.",
        "3. Pay special attention to edge cases: boundary rows, single-cell patterns,",
        "   and rows where multiple colors interact.",
        "4. Only write code once you can explain why your rule produces the correct",
        "   output for ALL training examples.",
    ])

    return "\n".join(parts)


def _color_f1(predicted: list[list[int]], expected: list[list[int]]) -> float:
    """Per-color F1 score averaged across all colors in expected output.

    For each color value present in the expected grid, computes F1 of
    that color's cell positions between predicted and expected.
    """
    exp_rows = len(expected)
    exp_cols = len(expected[0]) if expected else 0

    # Collect all colors in expected
    colors: set[int] = set()
    for row in expected:
        colors.update(row)

    if not colors:
        return 1.0

    total_f1 = 0.0
    for color in colors:
        exp_positions: set[tuple[int, int]] = set()
        for r in range(exp_rows):
            for c in range(exp_cols):
                if expected[r][c] == color:
                    exp_positions.add((r, c))

        pred_positions: set[tuple[int, int]] = set()
        for r, row in enumerate(predicted):
            for c, v in enumerate(row):
                if v == color:
                    pred_positions.add((r, c))

        tp = len(exp_positions & pred_positions)
        fp = len(pred_positions - exp_positions)
        fn = len(exp_positions - pred_positions)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        total_f1 += f1

    return total_f1 / len(colors)


def _row_exact_ratio(predicted: list[list[int]], expected: list[list[int]]) -> tuple[float, int, int]:
    """Fraction of rows that match exactly.

    Returns (ratio, correct_rows, total_rows).
    """
    if not expected:
        return (1.0, 0, 0)
    if len(predicted) != len(expected):
        return (0.0, 0, len(expected))

    correct = sum(1 for p, e in zip(predicted, expected) if p == e)
    return (correct / len(expected), correct, len(expected))


def _score_example(predicted: list[list[int]], expected: list[list[int]]) -> dict:
    """Score a single predicted grid against expected.

    Hierarchical scoring designed so exact match always dominates:
    - Exact match: 100 pts
    - Otherwise: shape_match (5) + color_f1 (10) + row_exact_ratio (30) = max 45

    This creates clear tier separation: solving N+1 examples always beats N.
    Row-exact-match provides meaningful gradient (nearly impossible to game).
    """
    exp_rows = len(expected)
    exp_cols = len(expected[0]) if expected else 0
    pred_rows = len(predicted) if predicted else 0
    pred_cols = len(predicted[0]) if predicted else 0

    is_exact = predicted == expected
    if is_exact:
        return {
            "is_exact": True,
            "example_score": 100.0,
            "shape_match": 1.0,
            "color_f1": 1.0,
            "row_exact_ratio": 1.0,
            "rows_correct": exp_rows,
            "rows_total": exp_rows,
        }

    shape_match = 1.0 if (pred_rows == exp_rows and pred_cols == exp_cols) else 0.0
    color_f1 = _color_f1(predicted, expected)
    row_ratio, rows_correct, rows_total = _row_exact_ratio(predicted, expected)

    example_score = shape_match * 5.0 + color_f1 * 10.0 + row_ratio * 30.0

    return {
        "is_exact": False,
        "example_score": example_score,
        "shape_match": shape_match,
        "color_f1": color_f1,
        "row_exact_ratio": row_ratio,
        "rows_correct": rows_correct,
        "rows_total": rows_total,
    }


class ArcScorer:
    """Picklable scorer for one ARC task (needed for multiprocess evaluation).

    Hierarchical scoring (0-100 scale):
    - Per example: exact match = 100 pts, otherwise max 45 pts
      (shape 5 + color_f1 10 + row_exact_ratio 30)
    - Overall: average across examples

    Tier separation ensures solving N+1 examples always beats N:
    - 0/3 solved: max 45.0    (all partial)
    - 1/3 solved: max 63.3    (100 + 45 + 45) / 3
    - 2/3 solved: max 81.7    (100 + 100 + 45) / 3
    - 3/3 solved: 100.0
    """

    def __init__(self, train_examples: list[dict]) -> None:
        self.train_examples = train_examples

    def __call__(self, transform_fn, _inputs=None) -> dict:
        total_score = 0.0
        exact_matches = 0
        per_example: dict[str, float] = {}

        for i, example in enumerate(self.train_examples):
            inp = example["input"]
            expected = example["output"]

            try:
                predicted = transform_fn(inp)
            except Exception:
                predicted = []

            if not isinstance(predicted, list):
                predicted = []

            ex = _score_example(predicted, expected)
            total_score += ex["example_score"]

            if ex["is_exact"]:
                exact_matches += 1

            # Per-example keys for behavioral diversity (score_keys)
            per_example[f"train_{i}"] = 1.0 if ex["is_exact"] else 0.0
            # Per-example diagnostics for LLM feedback
            per_example[f"train_{i}_row_acc"] = ex["row_exact_ratio"]

        n = len(self.train_examples)
        score = total_score / n if n > 0 else 0.0

        result: dict[str, float] = {
            "score": score,
            "exact_matches": float(exact_matches),
        }
        result.update(per_example)
        return result


def make_score_fn(train_examples: list[dict]) -> ArcScorer:
    """Create a picklable scorer for one ARC task."""
    return ArcScorer(train_examples)
