# Quickstart

Three minimal, self-contained runnable examples — one per path LEVI supports.
Pick whichever matches what you have access to.

| File                         | What it does                          | Needs                                             |
| ---------------------------- | -------------------------------------- | ------------------------------------------------- |
| `quickstart_api.py`          | Evolve a bin-packing function          | `OPENAI_API_KEY` (or change `MODEL` in the file)  |
| `quickstart_claude.py`       | Same problem, via Claude Code CLI      | `claude` CLI installed and signed in              |
| `quickstart_prompts.py`      | Evolve a sentiment-classifier prompt   | `OPENAI_API_KEY`                                  |

Each one is a single file plus the shared `problem.py`. Total cost per run is
a few cents on the API paths, and free on the Claude Code path.

## Run one

From the repo root:

```bash
uv sync
cd examples/quickstart
uv run python quickstart_api.py
```

Switching providers is a one-line edit — change `MODEL` at the top of the
file to any [litellm-supported](https://docs.litellm.ai/docs/providers) id
(e.g. `anthropic/claude-haiku-4-5`, `openrouter/google/gemini-2.5-flash`) and
set the matching API key in your environment.
