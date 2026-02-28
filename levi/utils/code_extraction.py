"""Utilities for extracting code and function names from LLM responses."""

import re
from typing import Optional


def extract_code(response: str) -> Optional[str]:
    """
    Extract Python code from LLM response.

    Handles multiple formats:
    1. Code blocks with ```python markers
    2. Generic code blocks with ``` markers
    3. Thinking tags that should be stripped (<think>, <thinking>)
    4. Raw Python code without markdown wrapping

    Args:
        response: Raw LLM response text

    Returns:
        Extracted Python code, or None if no valid code found
    """
    # Remove reasoning tags that some models include
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    response = re.sub(r"<thinking>.*?</thinking>", "", response, flags=re.DOTALL)

    # Try ```python blocks first
    matches = re.findall(r"```python\s*(.*?)```", response, re.DOTALL)
    if matches:
        return matches[0].strip()

    # Try generic ``` blocks
    matches = re.findall(r"```\s*(.*?)```", response, re.DOTALL)
    if matches:
        return matches[0].strip()

    # Try raw code starting with common Python patterns
    stripped = response.strip()
    for pattern in ["def ", "import ", "from ", "# ", '"""', "'''"]:
        if stripped.startswith(pattern):
            return stripped

    # Last resort: find first line that looks like Python code
    for line in stripped.split("\n"):
        line = line.strip()
        if line.startswith(("def ", "import ", "from ", "class ")):
            idx = stripped.find(line)
            return stripped[idx:].strip()

    return None


def extract_fn_name(fn_signature: str) -> str:
    """
    Extract function name from a function signature.

    Args:
        fn_signature: Function signature string (e.g., "def solve(x):")

    Returns:
        Function name, or 'solve' as default fallback
    """
    match = re.search(r"def\s+(\w+)\s*\(", fn_signature)
    if match:
        return match.group(1)
    return "solve"
