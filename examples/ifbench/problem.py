"""IFBench prompt-optimization task — port of gepa-artifact's 2-stage program.

Mirrors ``gepa-artifact/gepa_artifact/benchmarks/IFBench/ifbench_program.py``:

    - ``IFBenchCoT2StageProgram`` = two ``dspy.ChainOfThought`` modules
        1. ``generate_response_module`` over ``GenerateResponse(query -> response)``
        2. ``ensure_correct_response_module`` over
           ``EnsureCorrectResponse(query, response -> final_response)``
    - The final answer sent through IFBench's loose verifier is the
      ``final_response`` field from stage 2.
    - Scoring uses GEPA's merged IFEval+IFBench registry with the 8-variant
      response fold (``metric_with_feedback``'s loose accuracy).

Levi optimizes a 2-component prompt bundle ``{"generate_response",
"ensure_correct_response"}``. Each string is the instruction plugged into the
corresponding signature via ``pred.signature.with_instructions(...)``. Empty
seeds match GEPA's baseline (DSPy falls back to each signature's default
docstring).
"""

from __future__ import annotations

import json
import logging
import os
import random
import urllib.request
from pathlib import Path
from typing import Any

import dspy

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from official_adapter import load_gepa_registry
else:
    from .official_adapter import load_gepa_registry

logger = logging.getLogger(__name__)

TASK_MODEL = os.getenv("IFBENCH_TASK_MODEL", "openrouter/qwen/qwen3-8b")
TASK_PROVIDER_ONLY = os.getenv("IFBENCH_PROVIDER_ONLY", "alibaba")
TASK_TEMPERATURE = float(os.getenv("IFBENCH_TEMPERATURE", "0.6"))
TASK_TOP_P = float(os.getenv("IFBENCH_TOP_P", "0.95"))
TASK_TOP_K = int(os.getenv("IFBENCH_TOP_K", "20"))
TASK_MIN_P = float(os.getenv("IFBENCH_MIN_P", "0"))
TASK_MAX_TOKENS = int(os.getenv("IFBENCH_MAX_TOKENS", "32768"))
TASK_TIMEOUT = float(os.getenv("IFBENCH_TIMEOUT", "360"))
TASK_MAX_WORKERS = int(os.getenv("IFBENCH_MAX_WORKERS", "8"))

BUNDLE_KEYS = ("generate_response", "ensure_correct_response")

PROBLEM_DESCRIPTION = (
    "Optimize a 2-component prompt bundle for IFBench, a single-turn verifiable "
    "instruction-following benchmark. The DSPy program runs two ChainOfThought "
    "stages: `generate_response` drafts a reply to the user query, and "
    "`ensure_correct_response` rewrites that reply so every explicit constraint "
    "in the query is obeyed. Only the second stage's `final_response` is scored "
    "against the official IFBench loose verifier (prompt-level accuracy). "
    "Higher score means more constraints satisfied across the benchmark."
)

# Seed = empty -> DSPy uses each signature's default docstring (GEPA baseline).
SEED_BUNDLE: dict[str, str] = {key: "" for key in BUNDLE_KEYS}

IFBENCH_TRAIN_JSONL_URL = os.getenv(
    "IFBENCH_TRAIN_JSONL_URL",
    "https://raw.githubusercontent.com/gepa-ai/gepa-artifact/main/gepa_artifact/benchmarks/IFBench/data/IFBench_train.jsonl",
)
IFBENCH_TRAIN_CACHE_PATH = Path(
    os.getenv(
        "IFBENCH_TRAIN_CACHE_PATH",
        str(Path.home() / ".cache" / "levi" / "ifbench" / "IFBench_train.jsonl"),
    )
)


# ---------- DSPy program (port of IFBenchCoT2StageProgram) ----------


class GenerateResponse(dspy.Signature):
    """Respond to the query"""

    query = dspy.InputField()
    response = dspy.OutputField()


class EnsureCorrectResponse(dspy.Signature):
    """Ensure the response is correct and adheres to the given constraints. Your response will be used as the final response."""

    query = dspy.InputField()
    response = dspy.InputField()
    final_response = dspy.OutputField()


class IFBenchCoT2StageProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_response = dspy.ChainOfThought(GenerateResponse)
        self.ensure_correct_response = dspy.ChainOfThought(EnsureCorrectResponse)

    def forward(self, query: str):
        draft = self.generate_response(query=query).response
        final = self.ensure_correct_response(query=query, response=draft).final_response
        return dspy.Prediction(response=final, draft=draft)


# ---------- Data loading (port of IFBench.init_dataset) ----------


def _fetch_ifbench_train_jsonl() -> Path:
    if IFBENCH_TRAIN_CACHE_PATH.exists() and IFBENCH_TRAIN_CACHE_PATH.stat().st_size > 0:
        return IFBENCH_TRAIN_CACHE_PATH

    IFBENCH_TRAIN_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.info("[IFBench data] Fetching IFBench_train.jsonl from %s", IFBENCH_TRAIN_JSONL_URL)
    with urllib.request.urlopen(IFBENCH_TRAIN_JSONL_URL) as response:
        IFBENCH_TRAIN_CACHE_PATH.write_bytes(response.read())
    return IFBENCH_TRAIN_CACHE_PATH


def _row_to_example(row: dict[str, Any], source: str) -> dict[str, Any]:
    return {
        "id": f"ifbench_{source}_{row['key']}",
        "key": row["key"],
        "prompt": row["prompt"],
        "instruction_id_list": row["instruction_id_list"],
        "kwargs": row["kwargs"],
    }


def _load_train_examples() -> list[dict[str, Any]]:
    path = _fetch_ifbench_train_jsonl()
    examples: list[dict[str, Any]] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(_row_to_example(json.loads(line), "train"))
    return examples


def _load_test_examples() -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "This example requires the `datasets` package. Install it with:\n"
            "    uv run --with datasets python ...\n"
        ) from e

    raw_examples = list(load_dataset("allenai/IFBench_test")["train"])
    return [_row_to_example(row, "test") for row in raw_examples]


def load_splits(
    n_discovery: int = 150,
    *,
    shuffle_seed: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load IFBench using GEPA's split convention.

    Discovery pool comes from `IFBench_train.jsonl` (used for both proxy
    benchmark selection and Levi's optimization loop). Test set is the
    official `IFBench_test.jsonl` for final reference scoring.

    With `shuffle_seed=None` we keep the file order (matches GEPA's
    deterministic `train_set = train_val[300:600]` indexing). Pass an
    integer to deterministically shuffle before slicing.
    """
    train_examples = _load_train_examples()
    if shuffle_seed is not None:
        train_examples = list(train_examples)
        random.Random(shuffle_seed).shuffle(train_examples)
    discovery = train_examples[:n_discovery]
    testset = _load_test_examples()
    return discovery, testset


# ---------- DSPy LM configuration (per-process, lazy) ----------

_lm_configured = False


def _ensure_lm() -> None:
    global _lm_configured
    if _lm_configured:
        return
    kwargs: dict[str, Any] = dict(
        model=TASK_MODEL,
        temperature=TASK_TEMPERATURE,
        top_p=TASK_TOP_P,
        max_tokens=TASK_MAX_TOKENS,
        timeout=TASK_TIMEOUT,
    )
    extra_body: dict[str, Any] = {}
    if TASK_MODEL.startswith("openrouter/") and TASK_PROVIDER_ONLY:
        extra_body["provider"] = {"only": [TASK_PROVIDER_ONLY]}
    if TASK_TOP_K:
        extra_body["top_k"] = TASK_TOP_K
    extra_body["min_p"] = TASK_MIN_P
    kwargs["extra_body"] = extra_body
    dspy.configure(lm=dspy.LM(**kwargs))
    _lm_configured = True


# ---------- Scoring (8-variant loose fold — port of ifbench_metric.py) ----------


def _build_loose_response_variants(response: str) -> list[str]:
    """Port of GEPA's 8-variant response normalization (see ``ifbench_metric.py``)."""
    lines = response.split("\n")
    remove_first = "\n".join(lines[1:]).strip()
    remove_last = "\n".join(lines[:-1]).strip()
    remove_both = "\n".join(lines[1:-1]).strip()
    revised = response.replace("*", "")
    revised_first = remove_first.replace("*", "")
    revised_last = remove_last.replace("*", "")
    revised_both = remove_both.replace("*", "")
    return [
        response,
        revised,
        remove_first,
        remove_last,
        remove_both,
        revised_first,
        revised_last,
        revised_both,
    ]


def _check_item_following(
    registry_module: Any,
    item: dict[str, Any],
    response: str,
) -> tuple[list[bool], list[str], list[str]]:
    """Return (per-instruction follow list, correct_texts, incorrect_texts).

    Matches GEPA's loose fold: each instruction is marked followed if ANY of
    the 8 response variants passes its ``check_following`` check.
    """
    all_responses = _build_loose_response_variants(response)
    instruction_list = item["instruction_id_list"]
    kwargs_list = item.get("kwargs") or []
    follow_list: list[bool] = []
    correct_texts: list[str] = []
    incorrect_texts: list[str] = []

    for idx, instruction_id in enumerate(instruction_list):
        instruction_cls = registry_module.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)
        raw_kwargs = kwargs_list[idx] if idx < len(kwargs_list) else {}
        clean_kwargs = {k: v for k, v in (raw_kwargs or {}).items() if v is not None}
        ins_text = instruction.build_description(**clean_kwargs) or ""
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            ins_text = instruction.build_description(prompt=item["prompt"]) or ins_text

        is_following = False
        for variant in all_responses:
            if variant.strip() and instruction.check_following(variant):
                is_following = True
                break
        follow_list.append(is_following)
        (correct_texts if is_following else incorrect_texts).append(ins_text)

    return follow_list, correct_texts, incorrect_texts


def _build_feedback_text(
    item: dict[str, Any],
    response: str,
    correct_texts: list[str],
    incorrect_texts: list[str],
) -> str:
    """Port of GEPA's metric_with_feedback feedback_text construction."""
    parts: list[str] = [f"Q: {item['prompt']}", f"Response: {response!r}"]
    if correct_texts:
        parts.append(
            "Your response correctly followed the following instructions:\n"
            + "\n".join(correct_texts)
        )
    if incorrect_texts:
        header = (
            "However, your response did not follow the following instructions properly:"
            if correct_texts
            else "Your response did not follow the following instructions properly:"
        )
        parts.append(header + "\n" + "\n".join(incorrect_texts))
    return "\n".join(parts).strip()


# ---------- Program apply + per-example runner ----------


def _apply_bundle(program: IFBenchCoT2StageProgram, bundle: dict[str, str]) -> None:
    for key in BUNDLE_KEYS:
        text = bundle.get(key, "")
        if not text:
            continue
        pred = getattr(program, key).predict
        pred.signature = pred.signature.with_instructions(text)


def _score_one(
    bundle: dict[str, str],
    item: dict[str, Any],
    registry_module: Any,
) -> dict[str, Any]:
    program = IFBenchCoT2StageProgram()
    _apply_bundle(program, bundle)
    failure: dict[str, str] | None = None
    try:
        pred = program(query=item["prompt"])
        response = pred.response or ""
    except Exception as exc:
        response = ""
        failure = {"id": str(item["id"]), "error_type": type(exc).__name__, "message": str(exc)}

    # Run the registry check even on failure: an empty response naturally scores
    # 0 on every instruction, matching the old batch-failure semantics so that
    # instruction_weighted_score still counts the example's instructions in its
    # denominator.
    follow_list, correct, incorrect = _check_item_following(registry_module, item, response)
    if not follow_list:
        fraction = 0.0
        follow_all = 0.0
    else:
        followed = sum(bool(flag) for flag in follow_list)
        fraction = followed / len(follow_list)
        follow_all = 1.0 if followed == len(follow_list) else 0.0

    if failure:
        feedback = f"Q: {item['prompt']}\nResponse: (crash: {failure['error_type']})"
    else:
        feedback = _build_feedback_text(item, response, correct, incorrect)

    return {
        "id": item["id"],
        "response": response,
        "follow_list": follow_list,
        "loose_follow_all": follow_all,
        "loose_instruction_fraction": fraction,
        "feedback": feedback,
        "failure": failure,
    }


# ---------- Response dumping (preserved from old problem.py) ----------


def _dump_responses(
    bundle: dict[str, str],
    inputs: list[dict[str, Any]],
    per_example: list[dict[str, Any]],
    score: float,
) -> str | None:
    """Persist per-evaluation responses to disk; return the file path.

    Set ``IFBENCH_RESPONSES_DIR`` (run.py does this) to enable.
    """
    dir_env = os.getenv("IFBENCH_RESPONSES_DIR")
    if not dir_env:
        return None

    import hashlib
    import time

    out_dir = Path(dir_env)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None

    bundle_blob = json.dumps(bundle, sort_keys=True).encode("utf-8")
    digest = hashlib.sha1(bundle_blob).hexdigest()[:10]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_score{score:.4f}_{digest}_n{len(inputs)}.json"
    path = out_dir / filename

    failures = [p["failure"] for p in per_example if p["failure"]]
    payload = {
        "bundle": bundle,
        "score": score,
        "n_items": len(inputs),
        "request_failures": len(failures),
        "failures": failures,
        "items": [
            {"id": str(item["id"]), "prompt": item["prompt"], "response": p["response"]}
            for item, p in zip(inputs, per_example, strict=True)
        ],
    }
    try:
        path.write_text(json.dumps(payload, indent=2))
    except OSError:
        return None
    return str(path)


# ---------- Score function ----------


def score_fn(bundle: dict[str, str], inputs: list[dict[str, Any]]) -> dict[str, Any]:
    """Score a 2-component prompt bundle on IFBench using GEPA's loose fold.

    Headline ``score`` = mean of per-prompt loose-instruction fractions
    (each prompt weighted equally). The dense ``loose_instruction_fractions``
    vector is consumed by the proxy-benchmark selector.
    """
    import concurrent.futures

    if not inputs:
        return {
            "score": 0.0,
            "em": 0.0,
            "prompt_level_score": 0.0,
            "instruction_weighted_score": 0.0,
            "per_example_scores": [],
            "predictions": [],
            "feedback_per_example": [],
            "loose_follow_all": [],
            "loose_instruction_fractions": [],
            "request_failures": 0.0,
        }

    _ensure_lm()
    registry_module = load_gepa_registry()

    n = len(inputs)
    per_example: list[dict[str, Any]] = [{} for _ in range(n)]

    max_workers = min(TASK_MAX_WORKERS, n)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_score_one, bundle, item, registry_module): idx
            for idx, item in enumerate(inputs)
        }
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            per_example[idx] = future.result()

    failures = [p["failure"] for p in per_example if p["failure"]]
    if len(failures) == n:
        sample = ", ".join(f"{f['id']}:{f['error_type']}" for f in failures[:3])
        raise RuntimeError(f"All {n} IFBench requests failed for {TASK_MODEL}. Sample: {sample}")

    loose_instruction_fractions = [p["loose_instruction_fraction"] for p in per_example]
    loose_follow_all = [p["loose_follow_all"] for p in per_example]
    total_followed = sum(sum(bool(f) for f in p["follow_list"]) for p in per_example)
    total_instructions = sum(len(p["follow_list"]) for p in per_example)

    score = sum(loose_instruction_fractions) / len(loose_instruction_fractions)
    prompt_level_score = sum(loose_follow_all) / len(loose_follow_all)
    instruction_weighted_score = (
        total_followed / total_instructions if total_instructions else 0.0
    )

    # Response dumps go to disk — score_fn runs in a subprocess via
    # ResilientProcessPool, and 150 × multi-KB strings would overflow the
    # OS pipe buffer (~64KB on macOS) and wedge mp.Queue's feeder thread.
    responses_path = _dump_responses(bundle, inputs, per_example, score)

    return {
        "score": score,
        "em": score,
        "prompt_level_score": prompt_level_score,
        "instruction_weighted_score": instruction_weighted_score,
        "per_example_scores": loose_instruction_fractions,
        "predictions": [p["response"] for p in per_example],
        "feedback_per_example": [p["feedback"] for p in per_example],
        "loose_follow_all": loose_follow_all,
        "loose_instruction_fractions": loose_instruction_fractions,
        "request_failures": float(len(failures)),
        "responses_path": responses_path,
    }
