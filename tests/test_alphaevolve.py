"""
Tests for code extraction and diff helpers.

These are critical for the refactoring since they handle code extraction
from LLM responses and DIFF application.
"""

import pytest

from algoforge.utils import extract_code
from algoforge.pipeline.producer import apply_diff


class TestExtractCode:
    """Tests for extract_code function."""

    def test_extract_python_code_block(self):
        """Extracts code from ```python blocks."""
        response = '''Here's an improved version:

```python
def solve(x):
    return x * 2
```

This doubles the input.'''

        result = extract_code(response)

        assert result == "def solve(x):\n    return x * 2"

    def test_extract_generic_code_block(self):
        """Extracts code from generic ``` blocks."""
        response = '''Here's the code:

```
def solve(x):
    return x + 1
```
'''

        result = extract_code(response)

        assert result == "def solve(x):\n    return x + 1"

    def test_extract_prefers_python_over_generic(self):
        """Prefers ```python blocks over generic ``` blocks."""
        response = '''
```
some text
```

```python
def solve(x):
    return x
```
'''

        result = extract_code(response)

        assert result == "def solve(x):\n    return x"

    def test_extract_raw_code_starting_with_def(self):
        """Extracts raw code starting with 'def '."""
        response = '''def solve(x):
    return x * 3'''

        result = extract_code(response)

        assert result == "def solve(x):\n    return x * 3"

    def test_extract_raw_code_starting_with_import(self):
        """Extracts raw code starting with 'import '."""
        response = '''import math

def solve(x):
    return math.sqrt(x)'''

        result = extract_code(response)

        assert result == "import math\n\ndef solve(x):\n    return math.sqrt(x)"

    def test_extract_raw_code_starting_with_from(self):
        """Extracts raw code starting with 'from '."""
        response = '''from typing import List

def solve(x: List[int]) -> int:
    return sum(x)'''

        result = extract_code(response)

        assert result == "from typing import List\n\ndef solve(x: List[int]) -> int:\n    return sum(x)"

    def test_extract_raw_code_starting_with_comment(self):
        """Extracts raw code starting with '# '."""
        response = '''# Solution for the problem
def solve(x):
    return x'''

        result = extract_code(response)

        assert result == "# Solution for the problem\ndef solve(x):\n    return x"

    def test_extract_finds_code_in_middle_of_text(self):
        """Finds code that starts mid-response."""
        response = '''I think we can improve this by using a loop.

Let me show you:

def solve(items):
    total = 0
    for item in items:
        total += item
    return total

This should be faster.'''

        result = extract_code(response)

        # Should find the def line and everything after
        assert result is not None
        assert "def solve(items):" in result
        assert "for item in items:" in result

    def test_extract_multiline_code_block(self):
        """Handles multi-line code blocks correctly."""
        response = '''```python
def solve(x):
    if x > 0:
        return x
    else:
        return -x
```'''

        result = extract_code(response)

        expected = """def solve(x):
    if x > 0:
        return x
    else:
        return -x"""
        assert result == expected

    def test_extract_returns_none_for_no_code(self):
        """Returns None when no code is found."""
        response = '''This is just plain text with no code at all.
Nothing to extract here!'''

        result = extract_code(response)

        assert result is None

    def test_extract_handles_empty_response(self):
        """Handles empty response."""
        result = extract_code("")

        assert result is None

    def test_extract_handles_whitespace_only(self):
        """Handles whitespace-only response."""
        result = extract_code("   \n\n   \t   ")

        assert result is None

    def test_extract_strips_whitespace(self):
        """Strips leading/trailing whitespace from extracted code."""
        response = '''```python

def solve(x):
    return x

```'''

        result = extract_code(response)

        assert result == "def solve(x):\n    return x"

    def test_extract_first_python_block_when_multiple(self):
        """Extracts first ```python block when multiple exist."""
        response = '''```python
def first():
    pass
```

```python
def second():
    pass
```'''

        result = extract_code(response)

        assert "first" in result
        assert "second" not in result

    def test_extract_class_definition(self):
        """Extracts code starting with class definition."""
        response = '''Here's a class:

class Solution:
    def solve(self, x):
        return x'''

        result = extract_code(response)

        assert result is not None
        assert "class Solution:" in result

    def test_extract_docstring_code(self):
        """Extracts code starting with docstring."""
        response = '''"""This module provides a solution."""

def solve(x):
    return x'''

        result = extract_code(response)

        assert result is not None
        assert '"""This module' in result


class TestApplyDiff:
    """Tests for apply_diff function."""

    def test_apply_single_diff_block(self):
        """Applies a single SEARCH/REPLACE block."""
        original = '''def solve(x):
    return x'''

        diff_response = '''<<<<<<< SEARCH
    return x
=======
    return x * 2
>>>>>>> REPLACE'''

        result = apply_diff(original, diff_response)

        assert result == '''def solve(x):
    return x * 2'''

    def test_apply_multiple_diff_blocks(self):
        """Applies multiple SEARCH/REPLACE blocks."""
        original = '''def solve(x):
    y = x + 1
    return y'''

        diff_response = '''<<<<<<< SEARCH
    y = x + 1
=======
    y = x * 2
>>>>>>> REPLACE

<<<<<<< SEARCH
    return y
=======
    return y + 10
>>>>>>> REPLACE'''

        result = apply_diff(original, diff_response)

        assert result == '''def solve(x):
    y = x * 2
    return y + 10'''

    def test_apply_diff_with_surrounding_text(self):
        """Applies diff even with explanation text around it."""
        original = '''def solve(x):
    return x'''

        diff_response = '''I'll change the return statement:

<<<<<<< SEARCH
    return x
=======
    return x + 1
>>>>>>> REPLACE

This adds 1 to the result.'''

        result = apply_diff(original, diff_response)

        assert result == '''def solve(x):
    return x + 1'''

    def test_apply_diff_multiline_search_replace(self):
        """Handles multi-line SEARCH/REPLACE content."""
        original = '''def solve(x):
    if x > 0:
        return x
    else:
        return 0'''

        diff_response = '''<<<<<<< SEARCH
    if x > 0:
        return x
    else:
        return 0
=======
    if x > 0:
        return x * 2
    elif x < 0:
        return -x
    else:
        return 0
>>>>>>> REPLACE'''

        result = apply_diff(original, diff_response)

        expected = '''def solve(x):
    if x > 0:
        return x * 2
    elif x < 0:
        return -x
    else:
        return 0'''
        assert result == expected

    def test_apply_diff_falls_back_to_extract_code(self):
        """Falls back to extract_code when no diff blocks found."""
        original = '''def solve(x):
    return x'''

        # Response has code but no diff markers
        diff_response = '''```python
def solve(x):
    return x * 3
```'''

        result = apply_diff(original, diff_response)

        assert result == "def solve(x):\n    return x * 3"

    def test_apply_diff_returns_none_when_search_not_found(self):
        """Returns None when SEARCH content not found in original."""
        original = '''def solve(x):
    return x'''

        diff_response = '''<<<<<<< SEARCH
    return y
=======
    return y * 2
>>>>>>> REPLACE'''

        result = apply_diff(original, diff_response)

        assert result is None

    def test_apply_diff_partial_failure(self):
        """Returns None if any diff block fails to match."""
        original = '''def solve(x):
    return x'''

        diff_response = '''<<<<<<< SEARCH
    return x
=======
    return x * 2
>>>>>>> REPLACE

<<<<<<< SEARCH
    nonexistent = True
=======
    different = False
>>>>>>> REPLACE'''

        result = apply_diff(original, diff_response)

        # First succeeds, second fails -> None
        assert result is None

    def test_apply_diff_empty_original(self):
        """Handles empty original code."""
        result = apply_diff("", '''<<<<<<< SEARCH
x
=======
y
>>>>>>> REPLACE''')

        assert result is None

    def test_apply_diff_empty_response(self):
        """Handles empty diff response."""
        original = "def solve(x): return x"

        result = apply_diff(original, "")

        # Falls back to extract_code, which returns None for empty
        assert result is None

    def test_apply_diff_preserves_indentation(self):
        """Preserves indentation in replacements."""
        original = '''class Solution:
    def solve(self, x):
        return x'''

        diff_response = '''<<<<<<< SEARCH
        return x
=======
        # Improved
        return x * 2
>>>>>>> REPLACE'''

        result = apply_diff(original, diff_response)

        expected = '''class Solution:
    def solve(self, x):
        # Improved
        return x * 2'''
        assert result == expected

    def test_apply_diff_first_occurrence_only(self):
        """Replaces only first occurrence of search text."""
        original = '''def solve(x):
    x = x + 1
    x = x + 1
    return x'''

        diff_response = '''<<<<<<< SEARCH
    x = x + 1
=======
    x = x + 2
>>>>>>> REPLACE'''

        result = apply_diff(original, diff_response)

        # Only first occurrence replaced
        assert result.count("x = x + 2") == 1
        assert result.count("x = x + 1") == 1

    def test_apply_diff_whitespace_handling(self):
        """Handles whitespace in search/replace correctly."""
        original = '''def solve(x):
    return x'''

        # Note: search has extra whitespace around it
        diff_response = '''<<<<<<< SEARCH

    return x

=======
    return x * 2
>>>>>>> REPLACE'''

        result = apply_diff(original, diff_response)

        # Should strip and still match
        assert result == '''def solve(x):
    return x * 2'''


class TestExtractCodeApplyDiffIntegration:
    """Integration tests for extract_code and apply_diff working together."""

    def test_diff_fallback_chain(self):
        """When diff fails, falls back to code extraction."""
        original = '''def solve(x):
    return x'''

        # LLM ignores diff format, just writes code
        response = '''I rewrote the function completely:

```python
def solve(x):
    result = x * 2 + 1
    return result
```'''

        result = apply_diff(original, response)

        assert result is not None
        assert "result = x * 2 + 1" in result

    def test_realistic_llm_diff_response(self):
        """Tests realistic LLM diff response format."""
        original = '''def priority(job):
    """Compute priority for scheduling."""
    return job.weight / job.duration'''

        response = '''To improve throughput, we should consider both weight and deadline:

<<<<<<< SEARCH
    return job.weight / job.duration
=======
    urgency = 1.0 / max(1, job.deadline - job.start_time)
    efficiency = job.weight / job.duration
    return urgency * 0.4 + efficiency * 0.6
>>>>>>> REPLACE

This balances urgency with efficiency.'''

        result = apply_diff(original, response)

        assert result is not None
        assert "urgency = 1.0 / max" in result
        assert "efficiency = job.weight" in result
        assert "return urgency * 0.4" in result

    def test_realistic_llm_full_response(self):
        """Tests realistic LLM full rewrite response."""
        response = '''Here's an optimized version using dynamic programming:

```python
def solve(items, capacity):
    n = len(items)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        weight, value = items[i-1]
        for w in range(capacity + 1):
            if weight <= w:
                dp[i][w] = max(dp[i-1][w], dp[i-1][w-weight] + value)
            else:
                dp[i][w] = dp[i-1][w]

    return dp[n][capacity]
```

This achieves O(n*W) complexity.'''

        result = extract_code(response)

        assert result is not None
        assert "def solve(items, capacity):" in result
        assert "dp = [[0]" in result
        assert "return dp[n][capacity]" in result
