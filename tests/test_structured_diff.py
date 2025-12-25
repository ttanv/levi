"""Tests for structured diff application."""

import pytest
from algoforge.methods.alphaevolve import apply_structured_diff


class TestApplyStructuredDiff:
    """Tests for apply_structured_diff function."""

    def test_single_line_replacement(self):
        """Replace a single line."""
        original = "def foo():\n    return 1\n    # end"

        edit = {
            "edits": [{
                "start_line": 2,
                "end_line": 2,
                "new_content": "    return 2",
                "explanation": "Change return value"
            }],
            "summary": "Update return"
        }

        result = apply_structured_diff(original, edit)
        expected = "def foo():\n    return 2\n    # end"

        assert result == expected

    def test_multi_line_replacement(self):
        """Replace multiple lines."""
        original = "def foo():\n    x = 1\n    y = 2\n    return x + y"

        edit = {
            "edits": [{
                "start_line": 2,
                "end_line": 3,
                "new_content": "    z = 3",
                "explanation": "Simplify"
            }],
            "summary": "Simplify"
        }

        result = apply_structured_diff(original, edit)
        expected = "def foo():\n    z = 3\n    return x + y"

        assert result == expected

    def test_line_deletion(self):
        """Delete lines using empty new_content."""
        original = "def foo():\n    x = 1\n    y = 2\n    return x"

        edit = {
            "edits": [{
                "start_line": 3,
                "end_line": 3,
                "new_content": "",
                "explanation": "Remove unused variable"
            }],
            "summary": "Cleanup"
        }

        result = apply_structured_diff(original, edit)
        expected = "def foo():\n    x = 1\n    return x"

        assert result == expected

    def test_line_insertion_at_beginning(self):
        """Insert at the beginning of file."""
        original = "def foo():\n    return 1"

        edit = {
            "edits": [{
                "start_line": 1,
                "end_line": 0,
                "new_content": "# Header comment",
                "explanation": "Add header"
            }],
            "summary": "Add comment"
        }

        result = apply_structured_diff(original, edit)
        expected = "# Header comment\ndef foo():\n    return 1"

        assert result == expected

    def test_line_insertion_at_end(self):
        """Insert at the end of file."""
        original = "def foo():\n    return 1"

        edit = {
            "edits": [{
                "start_line": 3,
                "end_line": 2,
                "new_content": "# Footer",
                "explanation": "Add footer"
            }],
            "summary": "Add footer"
        }

        result = apply_structured_diff(original, edit)
        expected = "def foo():\n    return 1\n# Footer"

        assert result == expected

    def test_line_insertion_in_middle(self):
        """Insert in the middle of file."""
        original = "def foo():\n    x = 1\n    return x"

        edit = {
            "edits": [{
                "start_line": 3,
                "end_line": 2,
                "new_content": "    y = 2",
                "explanation": "Add variable"
            }],
            "summary": "Add line"
        }

        result = apply_structured_diff(original, edit)
        expected = "def foo():\n    x = 1\n    y = 2\n    return x"

        assert result == expected

    def test_multiple_non_overlapping_edits(self):
        """Apply multiple edits in sequence."""
        original = "line1\nline2\nline3\nline4\nline5"

        edit = {
            "edits": [
                {
                    "start_line": 1,
                    "end_line": 1,
                    "new_content": "LINE1",
                    "explanation": "Uppercase line 1"
                },
                {
                    "start_line": 3,
                    "end_line": 3,
                    "new_content": "LINE3",
                    "explanation": "Uppercase line 3"
                },
                {
                    "start_line": 5,
                    "end_line": 5,
                    "new_content": "LINE5",
                    "explanation": "Uppercase line 5"
                }
            ],
            "summary": "Uppercase odd lines"
        }

        result = apply_structured_diff(original, edit)
        expected = "LINE1\nline2\nLINE3\nline4\nLINE5"

        assert result == expected

    def test_invalid_overlapping_edits(self):
        """Reject overlapping edits."""
        original = "line1\nline2\nline3\nline4"

        edit = {
            "edits": [
                {
                    "start_line": 1,
                    "end_line": 2,
                    "new_content": "NEW",
                    "explanation": "Edit 1"
                },
                {
                    "start_line": 2,  # Overlaps with previous edit
                    "end_line": 3,
                    "new_content": "NEW2",
                    "explanation": "Edit 2"
                }
            ],
            "summary": "Overlapping"
        }

        result = apply_structured_diff(original, edit)

        assert result is None

    def test_invalid_line_number_too_high(self):
        """Reject line numbers beyond file length."""
        original = "line1\nline2"

        edit = {
            "edits": [{
                "start_line": 5,  # Beyond file length
                "end_line": 5,
                "new_content": "NEW",
                "explanation": "Invalid"
            }],
            "summary": "Invalid"
        }

        result = apply_structured_diff(original, edit)

        assert result is None

    def test_invalid_line_number_zero(self):
        """Reject line number 0."""
        original = "line1\nline2"

        edit = {
            "edits": [{
                "start_line": 0,
                "end_line": 1,
                "new_content": "NEW",
                "explanation": "Invalid"
            }],
            "summary": "Invalid"
        }

        result = apply_structured_diff(original, edit)

        assert result is None

    def test_invalid_start_after_end(self):
        """Reject start_line > end_line + 1."""
        original = "line1\nline2\nline3"

        edit = {
            "edits": [{
                "start_line": 5,
                "end_line": 2,
                "new_content": "NEW",
                "explanation": "Invalid"
            }],
            "summary": "Invalid"
        }

        result = apply_structured_diff(original, edit)

        assert result is None

    def test_empty_edits_list(self):
        """Handle empty edits list (no changes)."""
        original = "line1\nline2"

        edit = {
            "edits": [],
            "summary": "No changes"
        }

        result = apply_structured_diff(original, edit)

        assert result == original

    def test_invalid_missing_edits_key(self):
        """Reject response missing 'edits' key."""
        original = "line1"

        edit = {
            "summary": "No edits key"
        }

        result = apply_structured_diff(original, edit)

        assert result is None

    def test_invalid_non_dict_response(self):
        """Reject non-dict response."""
        original = "line1"

        result = apply_structured_diff(original, "not a dict")

        assert result is None

    def test_invalid_non_list_edits(self):
        """Reject edits that is not a list."""
        original = "line1"

        edit = {
            "edits": "not a list",
            "summary": "Invalid"
        }

        result = apply_structured_diff(original, edit)

        assert result is None

    def test_multiline_new_content(self):
        """Handle new_content with multiple lines."""
        original = "def foo():\n    return 1"

        edit = {
            "edits": [{
                "start_line": 2,
                "end_line": 2,
                "new_content": "    x = 10\n    return x",
                "explanation": "Multi-line replacement"
            }],
            "summary": "Expand function"
        }

        result = apply_structured_diff(original, edit)
        expected = "def foo():\n    x = 10\n    return x"

        assert result == expected

    def test_preserves_indentation(self):
        """Ensure indentation in new_content is preserved."""
        original = "class Foo:\n    def bar(self):\n        return 1"

        edit = {
            "edits": [{
                "start_line": 3,
                "end_line": 3,
                "new_content": "        x = 5\n        return x",
                "explanation": "Add indented code"
            }],
            "summary": "Expand method"
        }

        result = apply_structured_diff(original, edit)
        expected = "class Foo:\n    def bar(self):\n        x = 5\n        return x"

        assert result == expected

    def test_missing_required_field(self):
        """Reject edits missing required fields."""
        original = "line1\nline2"

        edit = {
            "edits": [{
                "start_line": 1,
                # Missing end_line, new_content, explanation
            }],
            "summary": "Invalid"
        }

        result = apply_structured_diff(original, edit)

        assert result is None

    def test_invalid_line_type(self):
        """Reject edits with non-integer line numbers."""
        original = "line1\nline2"

        edit = {
            "edits": [{
                "start_line": "1",  # String instead of int
                "end_line": 1,
                "new_content": "NEW",
                "explanation": "Invalid"
            }],
            "summary": "Invalid"
        }

        result = apply_structured_diff(original, edit)

        assert result is None

    def test_delete_multiple_lines(self):
        """Delete multiple consecutive lines."""
        original = "line1\nline2\nline3\nline4\nline5"

        edit = {
            "edits": [{
                "start_line": 2,
                "end_line": 4,
                "new_content": "",
                "explanation": "Delete lines 2-4"
            }],
            "summary": "Cleanup"
        }

        result = apply_structured_diff(original, edit)
        expected = "line1\nline5"

        assert result == expected

    def test_replace_entire_file(self):
        """Replace entire file content."""
        original = "line1\nline2\nline3"

        edit = {
            "edits": [{
                "start_line": 1,
                "end_line": 3,
                "new_content": "new_line1\nnew_line2",
                "explanation": "Complete rewrite"
            }],
            "summary": "Rewrite"
        }

        result = apply_structured_diff(original, edit)
        expected = "new_line1\nnew_line2"

        assert result == expected

    def test_edits_sorted_automatically(self):
        """Edits should be sorted even if provided out of order."""
        original = "line1\nline2\nline3\nline4\nline5"

        edit = {
            "edits": [
                {
                    "start_line": 5,
                    "end_line": 5,
                    "new_content": "LINE5",
                    "explanation": "Edit line 5"
                },
                {
                    "start_line": 1,
                    "end_line": 1,
                    "new_content": "LINE1",
                    "explanation": "Edit line 1"
                },
                {
                    "start_line": 3,
                    "end_line": 3,
                    "new_content": "LINE3",
                    "explanation": "Edit line 3"
                }
            ],
            "summary": "Out of order edits"
        }

        result = apply_structured_diff(original, edit)
        expected = "LINE1\nline2\nLINE3\nline4\nLINE5"

        assert result == expected
