"""Tests for prompt bundle serialization and response parsing."""

from collections import OrderedDict

import pytest

from levi.prompts.bundle import (
    DEFAULT_PROMPT_TARGET,
    PromptBundle,
    PromptTarget,
    normalize_prompt_bundle,
    parse_prompt_bundle_response,
    serialize_prompt_bundle,
)


class TestPromptBundle:
    def test_single_prompt_uses_default_target(self):
        bundle = normalize_prompt_bundle("Be concise.")

        assert bundle.target_names == (DEFAULT_PROMPT_TARGET,)
        assert bundle.get(DEFAULT_PROMPT_TARGET) == "Be concise."
        assert bundle.is_single_target is True

    def test_mapping_bundle_is_sorted_deterministically(self):
        bundle = normalize_prompt_bundle({"zeta": "last", "alpha": "first"})

        assert bundle.target_names == ("alpha", "zeta")
        assert bundle.as_dict() == {"alpha": "first", "zeta": "last"}

    def test_mapping_inputs_accept_ordered_dict(self):
        bundle = normalize_prompt_bundle(OrderedDict([("selection", "Pick promising prompts."), ("mutation", "Edit one target.")]))

        assert bundle.target_names == ("mutation", "selection")

    def test_serialize_prompt_bundle_is_canonical(self):
        serialized = serialize_prompt_bundle({"b": "second", "a": "first"})

        assert (
            serialized
            == '{\n  "a": "first",\n  "b": "second"\n}'
        )

    def test_round_trip_from_serialized(self):
        original = PromptBundle.from_mapping({"mutation": "Mutate carefully.", "system": "Think step by step."})

        restored = PromptBundle.from_serialized(original.serialize())

        assert restored == original

    def test_duplicate_target_names_are_rejected(self):
        with pytest.raises(ValueError, match="must be unique"):
            PromptBundle((PromptTarget("mutation", "one"), PromptTarget("mutation", "two")))

    def test_non_string_target_value_is_rejected(self):
        with pytest.raises(ValueError, match="must be strings"):
            normalize_prompt_bundle({"mutation": 123})  # type: ignore[arg-type]


class TestPromptBundleParsing:
    def test_parses_fenced_json_mapping(self):
        response = """
Here is the updated bundle:

```json
{
  "paradigm_shift": "Try a radically different approach.",
  "mutation": "Make one focused improvement."
}
```
"""

        parsed = parse_prompt_bundle_response(response)

        assert parsed is not None
        assert parsed.as_dict() == {
            "mutation": "Make one focused improvement.",
            "paradigm_shift": "Try a radically different approach.",
        }

    def test_parses_wrapped_json_bundle(self):
        response = """
{
  "notes": "targeted mutation prompt only",
  "prompts": {
    "mutation": "Prefer minimal diffs.",
    "selection": "Pick the most promising parent."
  }
}
"""

        parsed = parse_prompt_bundle_response(response)

        assert parsed is not None
        assert parsed.as_dict() == {
            "mutation": "Prefer minimal diffs.",
            "selection": "Pick the most promising parent.",
        }

    def test_reference_filters_extra_json_fields(self):
        reference = PromptBundle.from_mapping({"mutation": "old mutation", "selection": "old selection"})
        response = """
{
  "mutation": "new mutation",
  "selection": "new selection",
  "rationale": "ignored"
}
"""

        parsed = parse_prompt_bundle_response(response, reference=reference)

        assert parsed is not None
        assert parsed.as_dict() == {
            "mutation": "new mutation",
            "selection": "new selection",
        }

    def test_single_target_reference_accepts_alias_field(self):
        reference = PromptBundle.single("old text", target="mutation")
        response = '{"prompt": "Try one sharper local change."}'

        parsed = parse_prompt_bundle_response(response, reference=reference)

        assert parsed is not None
        assert parsed.as_dict() == {"mutation": "Try one sharper local change."}

    def test_single_target_reference_accepts_raw_text(self):
        reference = PromptBundle.single("old text", target="mutation")

        parsed = parse_prompt_bundle_response("Try a more explicit improvement instruction.", reference=reference)

        assert parsed is not None
        assert parsed.as_dict() == {"mutation": "Try a more explicit improvement instruction."}

    def test_single_target_reference_accepts_generic_fence(self):
        reference = PromptBundle.single("old text", target="mutation")
        response = """
```text
Prefer measurable, local improvements.
```
"""

        parsed = parse_prompt_bundle_response(response, reference=reference)

        assert parsed is not None
        assert parsed.as_dict() == {"mutation": "Prefer measurable, local improvements."}

    def test_multi_target_reference_rejects_raw_text(self):
        reference = PromptBundle.from_mapping({"mutation": "old mutation", "selection": "old selection"})

        parsed = parse_prompt_bundle_response("Only update the mutation prompt.", reference=reference)

        assert parsed is None
