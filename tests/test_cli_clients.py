"""Tests for subscription-backed CLI clients (Codex + Claude Code)."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from levi.clients import ClaudeCodeClient, CodexClient


def _fake_process(*, returncode: int = 0, stdout: bytes = b"", stderr: bytes = b""):
    process = MagicMock()
    process.returncode = returncode
    process.communicate = AsyncMock(return_value=(stdout, stderr))
    process.kill = MagicMock()
    return process


class TestCodexClient:
    def test_defaults(self):
        client = CodexClient()
        assert client.cli_path == "codex"
        assert client.sandbox == "read-only"
        assert client.reasoning_effort == "minimal"

    def test_acompletion_builds_expected_argv_and_reads_output(self, tmp_path):
        client = CodexClient(model="gpt-5")
        process = _fake_process(returncode=0)

        captured_cmd: list[str] = []
        captured_stdin: list[bytes] = []

        async def fake_exec(*cmd, **kwargs):
            captured_cmd.extend(cmd)
            # Write the expected output file where Codex would.
            for idx, token in enumerate(cmd):
                if token == "--output-last-message":
                    Path(cmd[idx + 1]).write_text("mutated code here", encoding="utf-8")
                    break

            async def fake_communicate(stdin_bytes):
                captured_stdin.append(stdin_bytes)
                return b"", b""

            process.communicate = fake_communicate
            return process

        with patch("levi.clients.codex.asyncio.create_subprocess_exec", side_effect=fake_exec):
            result = asyncio.run(client.acompletion("rewrite my function"))

        assert result.text == "mutated code here"
        assert result.cost == 0.0

        assert captured_cmd[0] == "codex"
        assert "exec" in captured_cmd
        assert "--ephemeral" in captured_cmd
        assert "--sandbox" in captured_cmd
        assert captured_cmd[captured_cmd.index("--sandbox") + 1] == "read-only"
        assert "--model" in captured_cmd
        assert captured_cmd[captured_cmd.index("--model") + 1] == "gpt-5"
        assert captured_cmd[-1] == "-"

        assert captured_stdin == [b"rewrite my function"]

    def test_acompletion_raises_on_nonzero_return(self):
        client = CodexClient()

        async def fake_exec(*cmd, **kwargs):
            return _fake_process(returncode=2, stderr=b"boom")

        with patch("levi.clients.codex.asyncio.create_subprocess_exec", side_effect=fake_exec):
            with pytest.raises(RuntimeError, match="boom"):
                asyncio.run(client.acompletion("hi"))

    def test_acompletion_raises_on_timeout(self):
        client = CodexClient(timeout=0.01)
        process = _fake_process()

        async def slow_communicate(_stdin):
            await asyncio.sleep(1.0)
            return b"", b""

        process.communicate = slow_communicate

        async def fake_exec(*cmd, **kwargs):
            return process

        with patch("levi.clients.codex.asyncio.create_subprocess_exec", side_effect=fake_exec):
            with pytest.raises(TimeoutError, match="timed out"):
                asyncio.run(client.acompletion("hi"))

        process.kill.assert_called_once()

    def test_acompletion_raises_when_cli_missing(self):
        client = CodexClient(cli_path="/nonexistent/codex")

        async def fake_exec(*cmd, **kwargs):
            raise FileNotFoundError("no such file")

        with patch("levi.clients.codex.asyncio.create_subprocess_exec", side_effect=fake_exec):
            with pytest.raises(RuntimeError, match="Codex CLI not found"):
                asyncio.run(client.acompletion("hi"))


class TestClaudeCodeClient:
    def test_defaults(self):
        client = ClaudeCodeClient()
        assert client.cli_path == "claude"
        assert client.tools == ""

    def test_acompletion_builds_expected_argv_and_reads_stdout(self):
        client = ClaudeCodeClient(model="claude-opus-4-5")

        captured_cmd: list[str] = []
        captured_stdin: list[bytes] = []

        process = _fake_process(returncode=0, stdout=b"  mutated code  \n")

        async def fake_communicate(stdin_bytes):
            captured_stdin.append(stdin_bytes)
            return b"  mutated code  \n", b""

        process.communicate = fake_communicate

        async def fake_exec(*cmd, **kwargs):
            captured_cmd.extend(cmd)
            return process

        with patch("levi.clients.claude_code.asyncio.create_subprocess_exec", side_effect=fake_exec):
            result = asyncio.run(client.acompletion("rewrite my function"))

        assert result.text == "mutated code"
        assert result.cost == 0.0

        assert captured_cmd[0] == "claude"
        assert "-p" in captured_cmd
        assert "--output-format" in captured_cmd
        assert captured_cmd[captured_cmd.index("--output-format") + 1] == "text"
        assert "--tools" in captured_cmd
        assert captured_cmd[captured_cmd.index("--tools") + 1] == ""
        assert "--disable-slash-commands" in captured_cmd
        assert "--model" in captured_cmd
        assert captured_cmd[captured_cmd.index("--model") + 1] == "claude-opus-4-5"

        assert captured_stdin == [b"rewrite my function"]

    def test_acompletion_raises_on_nonzero_return(self):
        client = ClaudeCodeClient()

        async def fake_exec(*cmd, **kwargs):
            return _fake_process(returncode=1, stderr=b"auth failure")

        with patch("levi.clients.claude_code.asyncio.create_subprocess_exec", side_effect=fake_exec):
            with pytest.raises(RuntimeError, match="auth failure"):
                asyncio.run(client.acompletion("hi"))

    def test_acompletion_raises_on_timeout(self):
        client = ClaudeCodeClient(timeout=0.01)
        process = _fake_process()

        async def slow_communicate(_stdin):
            await asyncio.sleep(1.0)
            return b"", b""

        process.communicate = slow_communicate

        async def fake_exec(*cmd, **kwargs):
            return process

        with patch("levi.clients.claude_code.asyncio.create_subprocess_exec", side_effect=fake_exec):
            with pytest.raises(TimeoutError, match="timed out"):
                asyncio.run(client.acompletion("hi"))

        process.kill.assert_called_once()

    def test_acompletion_raises_when_cli_missing(self):
        client = ClaudeCodeClient(cli_path="/nonexistent/claude")

        async def fake_exec(*cmd, **kwargs):
            raise FileNotFoundError("no such file")

        with patch("levi.clients.claude_code.asyncio.create_subprocess_exec", side_effect=fake_exec):
            with pytest.raises(RuntimeError, match="Claude CLI not found"):
                asyncio.run(client.acompletion("hi"))


class TestRenderCliPrompt:
    def test_single_user_message_is_verbatim(self):
        from levi.clients._cli_common import render_cli_prompt

        assert render_cli_prompt("raw prompt") == "raw prompt"
        assert render_cli_prompt([{"role": "user", "content": "raw prompt"}]) == "raw prompt"

    def test_multi_turn_chat_is_wrapped(self):
        from levi.clients._cli_common import render_cli_prompt

        rendered = render_cli_prompt(
            [
                {"role": "system", "content": "be helpful"},
                {"role": "user", "content": "hi"},
            ]
        )
        assert "assistant" in rendered.lower()
        assert "```json" in rendered
        assert "be helpful" in rendered
