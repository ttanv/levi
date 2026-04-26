"""Tests for the PUPA / PAPILLON example scorer."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import dspy


def _load_pupa_problem_module():
    path = Path(__file__).resolve().parents[1] / "examples" / "pupa" / "problem.py"
    spec = importlib.util.spec_from_file_location("test_pupa_problem_module", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeUntrustedLM:
    def __call__(self, *_args, **_kwargs):
        return ["fake-external-response"]


class _FakeJudge:
    """Looks up a per-user_query verdict so we can assert aggregation."""

    def __init__(self, results_by_query):
        self._results = results_by_query

    def __call__(self, **kwargs):
        return self._results[kwargs["user_query"]]


def test_seed_bundle_has_two_components():
    module = _load_pupa_problem_module()
    assert set(module.SEED_BUNDLE) == {"craft_redacted_request", "respond_to_query"}
    assert all(value == "" for value in module.SEED_BUNDLE.values())
    assert module.BUNDLE_KEYS == ("craft_redacted_request", "respond_to_query")


def test_apply_bundle_skips_empty_and_sets_non_empty():
    module = _load_pupa_problem_module()

    program = module.PAPILLON(untrusted_model=_FakeUntrustedLM())
    original_redact_instr = program.craft_redacted_request.predict.signature.instructions
    original_respond_instr = program.respond_to_query.signature.instructions

    # Empty values should leave both default docstrings in place.
    module._apply_bundle(program, {"craft_redacted_request": "", "respond_to_query": ""})
    assert program.craft_redacted_request.predict.signature.instructions == original_redact_instr
    assert program.respond_to_query.signature.instructions == original_respond_instr

    # Non-empty values should mutate the corresponding signature.
    # ChainOfThought wraps Predict in `.predict.signature`; bare Predict has
    # `.signature` directly.
    module._apply_bundle(
        program,
        {"craft_redacted_request": "redact please", "respond_to_query": "be careful"},
    )
    assert program.craft_redacted_request.predict.signature.instructions == "redact please"
    assert program.respond_to_query.signature.instructions == "be careful"


def test_score_fn_aggregates_quality_and_leakage(monkeypatch):
    module = _load_pupa_problem_module()

    fake_predictions = {
        "q0": dspy.Prediction(llm_request="r0", llm_response="x", response="resp0"),
        "q1": dspy.Prediction(llm_request="r1", llm_response="x", response="resp1"),
    }
    monkeypatch.setattr(
        module.PAPILLON,
        "__call__",
        lambda self, user_query: fake_predictions[user_query],
    )

    fake_judge = _FakeJudge(
        {
            "q0": dspy.Prediction(quality=True, leakage=0.0),
            "q1": dspy.Prediction(quality=False, leakage=0.5),
        }
    )
    monkeypatch.setattr(
        module, "_ensure_lm", lambda: (None, _FakeUntrustedLM(), fake_judge)
    )

    inputs = [
        {"id": "p0", "user_query": "q0", "target_response": "t0", "pii_str": "a||b"},
        {"id": "p1", "user_query": "q1", "target_response": "t1", "pii_str": "c||d"},
    ]
    result = module.score_fn(dict(module.SEED_BUNDLE), inputs)

    # p0: quality=1, leakage=0  -> (1 + 1)/2 = 1.0
    # p1: quality=0, leakage=0.5 -> (0 + 0.5)/2 = 0.25
    assert result["per_example_scores"] == [1.0, 0.25]
    assert result["score"] == (1.0 + 0.25) / 2
    assert result["quality_mean"] == 0.5
    assert result["leakage_mean"] == 0.25
    assert result["predictions"] == ["resp0", "resp1"]
    assert all(result["feedback_per_example"])
    assert result["request_failures"] == 0.0


def test_score_fn_handles_empty_response_without_calling_judge(monkeypatch):
    module = _load_pupa_problem_module()

    monkeypatch.setattr(
        module.PAPILLON,
        "__call__",
        lambda self, user_query: dspy.Prediction(llm_request="", llm_response="", response=""),
    )

    class _ExplodingJudge:
        def __call__(self, **_kwargs):
            raise AssertionError("judge must not be invoked when response is empty")

    monkeypatch.setattr(
        module, "_ensure_lm", lambda: (None, _FakeUntrustedLM(), _ExplodingJudge())
    )

    inputs = [{"id": "p0", "user_query": "q0", "target_response": "t0", "pii_str": "a||b"}]
    result = module.score_fn(dict(module.SEED_BUNDLE), inputs)

    assert result["per_example_scores"] == [0.0]
    assert result["score"] == 0.0
    assert result["quality_mean"] == 0.0
    assert result["leakage_mean"] == 0.0
    assert result["predictions"] == [""]


def test_judge_quality_consistency_rule():
    """Port of GEPA's `judgment_1 or (judgment_1 == judgment_2)` should give
    full credit unless the judge consistently rules the gold response strictly
    better (judgment_1=False, judgment_2=True)."""
    module = _load_pupa_problem_module()

    judge = module.LLMJudge()

    class _Verdict:
        def __init__(self, value):
            self.judgment = value

    cases = [
        # (j1, j2, expected_quality)
        (True, True, True),    # both say new>=og -> tie or new better
        (True, False, True),   # new strictly better
        (False, True, False),  # only case the new response truly fails
        (False, False, True),  # contradictory rulings -> give credit
    ]

    for j1, j2, expected in cases:
        verdicts = [_Verdict(j1), _Verdict(j2)]

        def fake_quality_judge(**_kwargs):
            return verdicts.pop(0)

        # No PII to score so we don't need to stub fact_checker.
        judge.quality_judge = fake_quality_judge  # type: ignore[assignment]
        out = judge(
            user_query="q",
            og_resp="og",
            new_resp="new",
            updated_query="r",
            pii_str="",
        )
        assert out.quality is expected, f"{(j1, j2)} -> expected {expected}, got {out.quality}"
        assert out.leakage == 0.0


def test_judge_leakage_normalizes_by_unique_pii_count():
    module = _load_pupa_problem_module()

    judge = module.LLMJudge()

    class _Verdict:
        def __init__(self, value):
            self.judgment = value

    judge.quality_judge = lambda **_kwargs: _Verdict(True)  # type: ignore[assignment]

    class _Leak:
        def __init__(self, n):
            self.num_pii_leaked = n

    judge.fact_checker = lambda **_kwargs: _Leak(2)  # type: ignore[assignment]

    # "a||a||b||c" -> unique pii = {a, b, c}, leak=2 -> 2/3.
    out = judge(
        user_query="q", og_resp="og", new_resp="new", updated_query="r", pii_str="a||a||b||c"
    )
    assert out.leakage == 2 / 3
