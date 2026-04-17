"""Tests for the minimal prompt bundle representation."""

from collections import OrderedDict

import pytest

from levi.prompts.bundle import (
    DEFAULT_PROMPT_TARGET,
    PromptBundle,
    normalize_prompt_bundle,
    serialize_prompt_bundle,
)


class TestPromptBundle:
    def test_single_prompt_uses_default_target(self):
        bundle = normalize_prompt_bundle("Be concise.")

        assert bundle.target_names == (DEFAULT_PROMPT_TARGET,)
        assert bundle.get(DEFAULT_PROMPT_TARGET) == "Be concise."
        assert bundle.editable_targets == (DEFAULT_PROMPT_TARGET,)
        assert bundle.is_single_target is True

    def test_single_prompt_can_be_frozen(self):
        bundle = PromptBundle.single("Do not change me.", editable=False)

        assert bundle.editable_targets == ()
        assert bundle.immutable_targets == (DEFAULT_PROMPT_TARGET,)

    def test_mapping_bundle_is_sorted_deterministically(self):
        bundle = normalize_prompt_bundle({"zeta": "last", "alpha": "first"})

        assert bundle.target_names == ("alpha", "zeta")
        assert bundle.editable_targets == ("alpha", "zeta")
        assert bundle.as_dict() == {"alpha": "first", "zeta": "last"}

    def test_mapping_inputs_accept_ordered_dict(self):
        bundle = normalize_prompt_bundle(
            OrderedDict([("selection", "Pick promising prompts."), ("mutation", "Edit one target.")])
        )

        assert bundle.target_names == ("mutation", "selection")

    def test_editable_targets_can_be_restricted(self):
        bundle = normalize_prompt_bundle(
            {"system": "Always be precise.", "mutation": "Make one targeted change."},
            editable_targets=("mutation",),
        )

        assert bundle.editable_targets == ("mutation",)
        assert bundle.immutable_targets == ("system",)
        assert bundle.is_editable("mutation") is True
        assert bundle.is_editable("system") is False

    def test_unknown_editable_target_is_rejected(self):
        with pytest.raises(ValueError, match="not present"):
            normalize_prompt_bundle({"mutation": "x"}, editable_targets=("system",))

    def test_with_updates_replaces_existing_prompt_text(self):
        bundle = normalize_prompt_bundle(
            {"system": "Stay concise.", "mutation": "Make one improvement."},
            editable_targets=("mutation",),
        )

        updated = bundle.with_updates({"mutation": "Make one measurable improvement."})

        assert updated.get("mutation") == "Make one measurable improvement."
        assert updated.get("system") == "Stay concise."
        assert updated.editable_targets == ("mutation",)

    def test_with_updates_rejects_unknown_target(self):
        bundle = normalize_prompt_bundle({"mutation": "Make one improvement."})

        with pytest.raises(KeyError, match="system"):
            bundle.with_updates({"system": "New system prompt"})


class TestPromptBundleSerialization:
    def test_serialize_prompt_bundle_is_canonical(self):
        serialized = serialize_prompt_bundle(
            {"b": "second", "a": "first"},
            editable_targets=("a",),
        )

        assert serialized == (
            '{\n'
            '  "editable_targets": [\n'
            '    "a"\n'
            "  ],\n"
            '  "prompts": {\n'
            '    "a": "first",\n'
            '    "b": "second"\n'
            "  }\n"
            "}"
        )

    def test_round_trip_from_serialized_payload(self):
        original = PromptBundle.from_mapping(
            {"mutation": "Mutate carefully.", "system": "Think step by step."},
            editable_targets=("mutation",),
        )

        restored = PromptBundle.from_serialized(original.serialize())

        assert restored == original

    def test_from_serialized_accepts_plain_mapping_for_backwards_compatibility(self):
        restored = PromptBundle.from_serialized('{"mutation": "Edit carefully."}')

        assert restored.as_dict() == {"mutation": "Edit carefully."}
        assert restored.editable_targets == ("mutation",)
