"""Claude Code CLI-backed client for users with a Claude subscription."""

import asyncio
import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from ._cli_common import render_cli_prompt
from .base import BaseClient, ClientInput, ClientResult
from .lm import DEFAULT_TIMEOUT

logger = logging.getLogger(__name__)


class ClaudeCodeClient(BaseClient):
    """Generation backend that routes through the ``claude`` CLI in print mode.

    Requires the Claude Code CLI to be installed locally and already
    authenticated via the user's Claude subscription. Each request spawns
    ``claude -p`` with all tools disabled so the model just returns the
    assistant's text response (the mutated code) — no filesystem access,
    no shell, no web.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        *,
        cli_path: str = "claude",
        tools: str = "",
        timeout: float = DEFAULT_TIMEOUT,
    ):
        super().__init__(model or "claude")
        self._claude_model = model
        self.cli_path = cli_path
        self.tools = tools
        self.timeout = timeout

    async def acompletion(self, prompt: ClientInput, **kwargs: Any) -> ClientResult:
        rendered = render_cli_prompt(prompt)
        timeout = kwargs.get("timeout") or self.timeout
        start = time.monotonic()

        cmd: list[str] = [
            self.cli_path,
            "-p",
            "--output-format",
            "text",
            "--tools",
            self.tools,
            "--disable-slash-commands",
        ]
        if self._claude_model:
            cmd.extend(["--model", self._claude_model])

        logger.info(
            "[%s] Claude Code CLI start | prompt_chars=%d | timeout=%ss",
            self.model,
            len(rendered),
            timeout,
        )

        with tempfile.TemporaryDirectory(prefix="levi_claude_cli_") as tmp_dir:
            workspace = Path(tmp_dir) / "workspace"
            workspace.mkdir()

            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(workspace),
                )
            except FileNotFoundError as error:
                raise RuntimeError(
                    f"[{self.model}] Claude CLI not found at {self.cli_path!r}. "
                    "Install Claude Code and ensure it's on PATH."
                ) from error

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(rendered.encode("utf-8")),
                    timeout=timeout,
                )
            except (asyncio.TimeoutError, TimeoutError) as error:
                process.kill()
                try:
                    await process.communicate()
                except Exception:
                    pass
                raise TimeoutError(
                    f"[{self.model}] Claude CLI timed out after {timeout} seconds"
                ) from error

        if process.returncode != 0:
            stderr_text = stderr.decode("utf-8", errors="replace").strip()
            stdout_text = stdout.decode("utf-8", errors="replace").strip()
            error_text = stderr_text or stdout_text or "Claude CLI failed with no output"
            elapsed = time.monotonic() - start
            logger.warning(
                "[%s] Claude Code CLI failed | elapsed=%.1fs | rc=%s | %s",
                self.model,
                elapsed,
                process.returncode,
                error_text[:400],
            )
            raise RuntimeError(f"[{self.model}] {error_text}")

        text = stdout.decode("utf-8", errors="replace").strip()
        elapsed = time.monotonic() - start
        logger.info(
            "[%s] Claude Code CLI complete | elapsed=%.1fs | output_chars=%d",
            self.model,
            elapsed,
            len(text),
        )

        return ClientResult(text=text, cost=0.0)
