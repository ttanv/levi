"""Shared helpers for CLI-backed client implementations."""

import json

from .base import ClientInput


def render_cli_prompt(prompt: ClientInput) -> str:
    """Render a Levi prompt for a subscription CLI.

    The common Levi path is a single user message; pass its content verbatim
    so prompt templates and parent solutions stay byte-identical. For richer
    chat histories, fall back to a structured JSON transcript with a short
    instructional header so the CLI has enough context to continue the turn.
    """
    if isinstance(prompt, str):
        return prompt

    if len(prompt) == 1 and prompt[0].get("role") == "user":
        content = prompt[0].get("content", "")
        return content if isinstance(content, str) else json.dumps(content)

    transcript = json.dumps(prompt, ensure_ascii=True, indent=2)
    return (
        "Return the next assistant message for the chat payload below.\n\n"
        "Use only this payload as context. Do not inspect local files, run commands, "
        "or use external tools.\n\n"
        "```json\n"
        f"{transcript}\n"
        "```"
    )
