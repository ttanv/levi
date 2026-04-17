"""Canonical prompt-bundle helpers for prompt evolution."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any

DEFAULT_PROMPT_TARGET = "prompt"
_WRAPPER_KEYS = ("prompts", "bundle")
_SINGLE_VALUE_KEYS = ("prompt", "value", "content", "text")
_GENERIC_FENCED_BLOCK_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]+)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_JSON_FENCED_BLOCK_RE = re.compile(r"```json\s*(.*?)```", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True, order=True)
class PromptTarget:
    """Named prompt component within a bundle."""

    name: str
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("prompt target name must be a non-empty string")
        if not isinstance(self.text, str):
            raise ValueError("prompt target text must be a string")


@dataclass(frozen=True)
class PromptBundle:
    """Canonical representation of one or more prompt components."""

    targets: tuple[PromptTarget, ...]

    def __post_init__(self) -> None:
        if not self.targets:
            raise ValueError("prompt bundle must contain at least one target")

        sorted_targets = tuple(sorted(self.targets, key=lambda target: target.name))
        names = [target.name for target in sorted_targets]
        if len(set(names)) != len(names):
            raise ValueError("prompt bundle target names must be unique")

        object.__setattr__(self, "targets", sorted_targets)

    @classmethod
    def single(cls, text: str, *, target: str = DEFAULT_PROMPT_TARGET) -> "PromptBundle":
        """Build a bundle containing one named prompt."""
        return cls((PromptTarget(target, text),))

    @classmethod
    def from_mapping(cls, prompts: dict[str, str]) -> "PromptBundle":
        """Build a bundle from a mapping of target name -> prompt text."""
        if not prompts:
            raise ValueError("prompt mapping must contain at least one target")
        return cls(tuple(PromptTarget(name, text) for name, text in prompts.items()))

    @classmethod
    def from_value(
        cls,
        value: "PromptBundle | str | Mapping[str, str]",
        *,
        default_target: str = DEFAULT_PROMPT_TARGET,
    ) -> "PromptBundle":
        """Normalize a single prompt or prompt mapping into a bundle."""
        if isinstance(value, PromptBundle):
            return value
        if isinstance(value, str):
            return cls.single(value, target=default_target)
        if isinstance(value, Mapping):
            _validate_string_mapping(value)
            return cls.from_mapping(dict(value))
        raise TypeError("prompt bundle value must be a PromptBundle, string, or Mapping[str, str]")

    @classmethod
    def from_serialized(cls, serialized: str, *, reference: "PromptBundle | None" = None) -> "PromptBundle":
        """Deserialize a canonical JSON prompt bundle."""
        payload = json.loads(serialized)
        return cls.from_payload(payload, reference=reference)

    @classmethod
    def from_payload(cls, payload: Any, *, reference: "PromptBundle | None" = None) -> "PromptBundle":
        """Build a bundle from decoded JSON or other parsed payloads."""
        prompts = _coerce_prompt_mapping(payload, reference=reference)
        return cls.from_mapping(prompts)

    @property
    def target_names(self) -> tuple[str, ...]:
        return tuple(target.name for target in self.targets)

    @property
    def is_single_target(self) -> bool:
        return len(self.targets) == 1

    def get(self, target: str) -> str:
        for entry in self.targets:
            if entry.name == target:
                return entry.text
        raise KeyError(target)

    def as_dict(self) -> dict[str, str]:
        return {target.name: target.text for target in self.targets}

    def serialize(self) -> str:
        """Return a deterministic JSON representation for storage/evolution."""
        return json.dumps(self.as_dict(), indent=2, sort_keys=True, ensure_ascii=False)


def normalize_prompt_bundle(
    value: PromptBundle | str | Mapping[str, str],
    *,
    default_target: str = DEFAULT_PROMPT_TARGET,
) -> PromptBundle:
    """Normalize prompt inputs into a canonical bundle."""
    return PromptBundle.from_value(value, default_target=default_target)


def serialize_prompt_bundle(
    value: PromptBundle | str | Mapping[str, str],
    *,
    default_target: str = DEFAULT_PROMPT_TARGET,
) -> str:
    """Normalize prompt inputs and return canonical serialized content."""
    return normalize_prompt_bundle(value, default_target=default_target).serialize()


def parse_prompt_bundle_response(
    response_text: str,
    *,
    reference: PromptBundle | None = None,
) -> PromptBundle | None:
    """Parse a model response into a prompt bundle.

    Accepts:
    - Bare JSON object: ``{"mutation": "...", "system": "..."}``
    - Wrapped JSON: ``{"prompts": {...}}`` or ``{"bundle": {...}}``
    - Single-target raw text when ``reference`` names exactly one target
    """

    for payload in _iter_decoded_json_payloads(response_text):
        try:
            return PromptBundle.from_payload(payload, reference=reference)
        except (TypeError, ValueError):
            continue

    if reference is not None and reference.is_single_target:
        target_name = reference.target_names[0]
        raw_text = _strip_outer_fence(response_text).strip()
        if raw_text:
            return PromptBundle.single(raw_text, target=target_name)

    return None


def _validate_string_mapping(prompts: Mapping[str, str]) -> None:
    for name, text in prompts.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("prompt target names must be non-empty strings")
        if not isinstance(text, str):
            raise ValueError("prompt target values must be strings")


def _coerce_prompt_mapping(payload: Any, *, reference: PromptBundle | None = None) -> dict[str, str]:
    if isinstance(payload, str):
        target_name = DEFAULT_PROMPT_TARGET
        if reference is not None and reference.is_single_target:
            target_name = reference.target_names[0]
        return {target_name: payload}

    if not isinstance(payload, Mapping):
        raise TypeError("prompt bundle payload must be a string or object")

    mapping = dict(payload)
    for key in _WRAPPER_KEYS:
        wrapped = mapping.get(key)
        if isinstance(wrapped, Mapping):
            _validate_string_mapping(wrapped)
            return dict(wrapped)

    if reference is not None:
        matched = _select_reference_targets(mapping, reference)
        if matched is not None:
            return matched

    _validate_string_mapping(mapping)
    return mapping


def _select_reference_targets(mapping: dict[str, Any], reference: PromptBundle) -> dict[str, str] | None:
    reference_names = reference.target_names
    if all(name in mapping and isinstance(mapping[name], str) for name in reference_names):
        return {name: mapping[name] for name in reference_names}

    if reference.is_single_target:
        target_name = reference_names[0]
        for alias in _SINGLE_VALUE_KEYS:
            value = mapping.get(alias)
            if isinstance(value, str):
                return {target_name: value}

    return None


def _iter_decoded_json_payloads(text: str) -> list[Any]:
    decoder = json.JSONDecoder()
    payloads: list[Any] = []
    seen: set[str] = set()

    for candidate in _iter_json_candidates(text):
        stripped = candidate.strip()
        if not stripped or stripped in seen:
            continue
        seen.add(stripped)
        try:
            payloads.append(decoder.decode(stripped))
        except JSONDecodeError:
            continue

    stripped_text = text.strip()
    for idx, ch in enumerate(stripped_text):
        if ch not in '{"':
            continue
        fragment = stripped_text[idx:]
        try:
            payload, end = decoder.raw_decode(fragment)
        except JSONDecodeError:
            continue

        raw_candidate = fragment[:end].strip()
        if raw_candidate in seen:
            continue
        seen.add(raw_candidate)
        payloads.append(payload)

    return payloads


def _iter_json_candidates(text: str) -> list[str]:
    candidates = [text]
    candidates.extend(match.group(1) for match in _JSON_FENCED_BLOCK_RE.finditer(text))
    candidates.extend(match.group(1) for match in _GENERIC_FENCED_BLOCK_RE.finditer(text))
    return candidates


def _strip_outer_fence(text: str) -> str:
    stripped = text.strip()
    match = _GENERIC_FENCED_BLOCK_RE.fullmatch(stripped)
    if match:
        return match.group(1)
    return stripped
