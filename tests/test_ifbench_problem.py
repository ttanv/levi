"""Tests for the IFBench example scorer (2-stage DSPy port)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import dspy


def _load_ifbench_problem_module():
    path = Path(__file__).resolve().parents[1] / "examples" / "ifbench" / "problem.py"
    spec = importlib.util.spec_from_file_location("test_ifbench_problem_module", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeInstruction:
    def __init__(self, instruction_id, *, needs_prompt=False, followed_marker="loose-pass"):
        self._instruction_id = instruction_id
        self._needs_prompt = needs_prompt
        self._followed_marker = followed_marker

    def build_description(self, **_kwargs):
        return f"[desc:{self._instruction_id}]"

    def get_instruction_args(self):
        return {"prompt": None} if self._needs_prompt else {}

    def check_following(self, response):
        return self._followed_marker in response


def _fake_registry():
    return SimpleNamespace(
        INSTRUCTION_DICT={
            "dummy:alpha": lambda _id: _FakeInstruction(_id, followed_marker="loose-pass"),
            "dummy:beta": lambda _id: _FakeInstruction(_id, followed_marker="beta-pass"),
        }
    )


def test_build_loose_response_variants_has_eight_entries():
    module = _load_ifbench_problem_module()
    variants = module._build_loose_response_variants("*hello*\nworld\nagain")
    assert len(variants) == 8
    assert variants[0] == "*hello*\nworld\nagain"
    assert "*" not in variants[1]


def test_check_item_following_uses_eight_variant_fold():
    module = _load_ifbench_problem_module()
    registry = _fake_registry()

    # Asterisks obscure the follow marker; only the asterisk-stripped variant matches.
    follow_list, correct, incorrect = module._check_item_following(
        registry,
        {
            "prompt": "p",
            "instruction_id_list": ["dummy:alpha"],
            "kwargs": [{}],
        },
        "*loose-pass*",
    )
    assert follow_list == [True]
    assert correct == ["[desc:dummy:alpha]"]
    assert incorrect == []


def test_check_item_following_partitions_correct_and_incorrect():
    module = _load_ifbench_problem_module()
    registry = _fake_registry()

    follow_list, correct, incorrect = module._check_item_following(
        registry,
        {
            "prompt": "p",
            "instruction_id_list": ["dummy:alpha", "dummy:beta"],
            "kwargs": [{}, {}],
        },
        "loose-pass",
    )
    assert follow_list == [True, False]
    assert correct == ["[desc:dummy:alpha]"]
    assert incorrect == ["[desc:dummy:beta]"]


def _stub_program_responses(module, responses_by_id):
    """Make IFBenchCoT2StageProgram.__call__ return canned responses keyed by query."""

    def fake_call(self, query):
        return dspy.Prediction(response=responses_by_id[query], draft=responses_by_id[query])

    module.IFBenchCoT2StageProgram.__call__ = fake_call


def test_score_fn_aggregates_loose_metrics(monkeypatch):
    module = _load_ifbench_problem_module()
    monkeypatch.setattr(module, "load_gepa_registry", _fake_registry)
    monkeypatch.setattr(module, "_ensure_lm", lambda: None)

    _stub_program_responses(
        module,
        {"p0": "loose-pass beta-pass", "p1": "loose-pass"},
    )

    inputs = [
        {
            "id": "ifbench_0",
            "key": "0",
            "prompt": "p0",
            "instruction_id_list": ["dummy:alpha", "dummy:beta"],
            "kwargs": [{}, {}],
        },
        {
            "id": "ifbench_1",
            "key": "1",
            "prompt": "p1",
            "instruction_id_list": ["dummy:alpha"],
            "kwargs": [{}],
        },
    ]

    result = module.score_fn({"generate_response": "", "ensure_correct_response": ""}, inputs)

    # All instructions followed: per-prompt fractions = [1.0, 1.0] → mean 1.0.
    assert result["score"] == 1.0
    assert result["prompt_level_score"] == 1.0
    assert result["instruction_weighted_score"] == 1.0
    assert result["per_example_scores"] == [1.0, 1.0]
    assert result["loose_follow_all"] == [1.0, 1.0]
    assert result["loose_instruction_fractions"] == [1.0, 1.0]
    assert result["request_failures"] == 0.0
    assert result["predictions"] == ["loose-pass beta-pass", "loose-pass"]
    assert all(isinstance(entry, str) and entry for entry in result["feedback_per_example"])


def test_score_fn_handles_partial_follow_and_crash(monkeypatch):
    module = _load_ifbench_problem_module()
    monkeypatch.setattr(module, "load_gepa_registry", _fake_registry)
    monkeypatch.setattr(module, "_ensure_lm", lambda: None)

    def fake_call(self, query):
        if query == "p0":
            return dspy.Prediction(response="loose-pass", draft="loose-pass")
        raise RuntimeError("boom")

    module.IFBenchCoT2StageProgram.__call__ = fake_call

    inputs = [
        {
            "id": "ifbench_0",
            "key": "0",
            "prompt": "p0",
            "instruction_id_list": ["dummy:alpha", "dummy:beta"],
            "kwargs": [{}, {}],
        },
        {
            "id": "ifbench_1",
            "key": "1",
            "prompt": "p1",
            "instruction_id_list": ["dummy:alpha"],
            "kwargs": [{}],
        },
    ]

    result = module.score_fn({"generate_response": "", "ensure_correct_response": ""}, inputs)

    # Prompt 0: alpha followed (1), beta not (0) → fraction 0.5.
    # Prompt 1: program crash → empty response, no follows → fraction 0.0.
    assert result["score"] == 0.25
    assert result["prompt_level_score"] == 0.0
    assert result["instruction_weighted_score"] == 1 / 3
    assert result["loose_instruction_fractions"] == [0.5, 0.0]
    assert result["loose_follow_all"] == [0.0, 0.0]
    assert result["request_failures"] == 1.0


def test_seed_bundle_has_two_components():
    module = _load_ifbench_problem_module()
    assert set(module.SEED_BUNDLE) == {"generate_response", "ensure_correct_response"}
    assert all(value == "" for value in module.SEED_BUNDLE.values())


def test_apply_bundle_skips_empty_and_sets_non_empty(monkeypatch):
    module = _load_ifbench_problem_module()
    monkeypatch.setattr(module, "_ensure_lm", lambda: None)

    program = module.IFBenchCoT2StageProgram()
    original_gen_instruction = program.generate_response.predict.signature.instructions

    module._apply_bundle(program, {"generate_response": "", "ensure_correct_response": "be careful"})

    # Empty => leave DSPy default in place.
    assert program.generate_response.predict.signature.instructions == original_gen_instruction
    # Non-empty => signature instructions replaced.
    assert program.ensure_correct_response.predict.signature.instructions == "be careful"
