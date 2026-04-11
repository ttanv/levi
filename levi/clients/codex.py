"""Codex CLI-backed client for users with an OpenAI Codex subscription."""

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


class CodexClient(BaseClient):
    """Generation backend that routes through the ``codex`` CLI.

    Requires the Codex CLI to be installed locally and already authenticated
    via the user's OpenAI Codex subscription. Each request spawns ``codex exec``
    in an ephemeral, read-only sandbox so Levi's prompts can't touch the host.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        *,
        cli_path: str = "codex",
        sandbox: str = "read-only",
        reasoning_effort: str = "minimal",
        timeout: float = DEFAULT_TIMEOUT,
    ):
        super().__init__(model or "codex")
        self._codex_model = model
        self.cli_path = cli_path
        self.sandbox = sandbox
        self.reasoning_effort = reasoning_effort
        self.timeout = timeout

    async def acompletion(self, prompt: ClientInput, **kwargs: Any) -> ClientResult:
        rendered = render_cli_prompt(prompt)
        timeout = kwargs.get("timeout") or self.timeout
        reasoning_effort = kwargs.get("reasoning_effort") or self.reasoning_effort
        start = time.monotonic()

        with tempfile.TemporaryDirectory(prefix="levi_codex_cli_") as tmp_dir:
            workspace = Path(tmp_dir) / "workspace"
            workspace.mkdir()
            output_path = Path(tmp_dir) / "last_message.txt"

            cmd: list[str] = [
                self.cli_path,
                "exec",
                "--skip-git-repo-check",
                "--ephemeral",
                "--sandbox",
                self.sandbox,
                "--color",
                "never",
                "--output-last-message",
                str(output_path),
                "-C",
                str(workspace),
            ]
            if reasoning_effort:
                cmd.extend(["-c", f'model_reasoning_effort="{reasoning_effort}"'])
            # web_search is incompatible with reasoning_effort="minimal" and
            # useless in a read-only ephemeral sandbox anyway.
            cmd.extend(["-c", 'web_search="disabled"'])
            if self._codex_model:
                cmd.extend(["--model", self._codex_model])
            cmd.append("-")

            logger.info(
                "[%s] Codex CLI start | prompt_chars=%d | timeout=%ss",
                self.model,
                len(rendered),
                timeout,
            )

            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except FileNotFoundError as error:
                raise RuntimeError(
                    f"[{self.model}] Codex CLI not found at {self.cli_path!r}. "
                    "Install the Codex CLI and ensure it's on PATH."
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
                    f"[{self.model}] Codex CLI timed out after {timeout} seconds"
                ) from error

            if process.returncode != 0:
                stderr_text = stderr.decode("utf-8", errors="replace").strip()
                stdout_text = stdout.decode("utf-8", errors="replace").strip()
                error_text = stderr_text or stdout_text or "Codex CLI failed with no output"
                elapsed = time.monotonic() - start
                logger.warning(
                    "[%s] Codex CLI failed | elapsed=%.1fs | rc=%s | %s",
                    self.model,
                    elapsed,
                    process.returncode,
                    error_text[:400],
                )
                raise RuntimeError(f"[{self.model}] {error_text}")

            text = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
            elapsed = time.monotonic() - start
            logger.info(
                "[%s] Codex CLI complete | elapsed=%.1fs | output_chars=%d",
                self.model,
                elapsed,
                len(text.strip()),
            )

        return ClientResult(text=text.strip(), cost=0.0)
