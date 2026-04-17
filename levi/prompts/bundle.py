"""Minimal prompt-bundle helpers for prompt evolution."""

from __future__ import annotations

import json
from collections.abc import Collection
from typing import Any

DEFAULT_PROMPT_TARGET = "prompt"


class PromptBundle:
    """Canonical prompt artifact with explicit editable-target metadata."""

    def __init__(
        self,
        prompts: dict[str, str],
        editable_targets: Collection[str] | None = None,
    ) -> None:
        if not prompts:
            raise ValueError("prompt bundle must contain at least one prompt")

        self.prompts = dict(sorted(prompts.items()))

        if editable_targets is None:
            self.editable_targets = tuple(self.prompts.keys())
        else:
            self.editable_targets = tuple(sorted(dict.fromkeys(editable_targets)))
            for target in self.editable_targets:
                if target not in self.prompts:
                    raise ValueError(f"editable target {target!r} is not present in prompts")

    def __eq__(self, other: object) -> bool:
        return self.prompts == other.prompts and self.editable_targets == other.editable_targets

    def __repr__(self) -> str:
        return f"PromptBundle(prompts={self.prompts!r}, editable_targets={self.editable_targets!r})"

    @staticmethod
    def single(
        text: str,
        *,
        target: str = DEFAULT_PROMPT_TARGET,
        editable: bool = True,
    ) -> "PromptBundle":
        """Build a bundle containing one named prompt."""
        editable_targets = (target,) if editable else ()
        return PromptBundle({target: text}, editable_targets=editable_targets)

    @staticmethod
    def from_mapping(
        prompts: dict[str, str],
        *,
        editable_targets: Collection[str] | None = None,
    ) -> "PromptBundle":
        """Build a bundle from a mapping of target name -> prompt text."""
        if not prompts:
            raise ValueError("prompt mapping must contain at least one prompt")
        return PromptBundle(prompts, editable_targets=editable_targets)

    @staticmethod
    def from_value(
        value: "PromptBundle | str | dict[str, str]",
        *,
        default_target: str = DEFAULT_PROMPT_TARGET,
        editable_targets: Collection[str] | None = None,
    ) -> "PromptBundle":
        """Normalize a prompt string or mapping into a bundle."""
        if isinstance(value, PromptBundle):
            return value
        if isinstance(value, str):
            editable = editable_targets is None or default_target in editable_targets
            return PromptBundle.single(value, target=default_target, editable=editable)
        if isinstance(value, dict):
            return PromptBundle.from_mapping(value, editable_targets=editable_targets)
        return PromptBundle.from_mapping(value, editable_targets=editable_targets)  # type: ignore[arg-type]

    @staticmethod
    def from_payload(payload: Any) -> "PromptBundle":
        """Build a bundle from decoded JSON."""
        if "prompts" in payload:
            return PromptBundle.from_mapping(
                payload["prompts"],
                editable_targets=payload.get("editable_targets"),
            )

        return PromptBundle.from_mapping(payload)

    @staticmethod
    def from_serialized(serialized: str) -> "PromptBundle":
        """Deserialize a prompt bundle from JSON."""
        return PromptBundle.from_payload(json.loads(serialized))

    @property
    def target_names(self) -> tuple[str, ...]:
        return tuple(self.prompts.keys())

    @property
    def immutable_targets(self) -> tuple[str, ...]:
        editable = set(self.editable_targets)
        return tuple(name for name in self.target_names if name not in editable)

    @property
    def is_single_target(self) -> bool:
        return len(self.prompts) == 1

    def get(self, target: str) -> str:
        return self.prompts[target]

    def is_editable(self, target: str) -> bool:
        if target not in self.prompts:
            raise KeyError(target)
        return target in set(self.editable_targets)

    def as_dict(self) -> dict[str, str]:
        return dict(self.prompts)

    def as_payload(self) -> dict[str, Any]:
        return {
            "prompts": self.as_dict(),
            "editable_targets": list(self.editable_targets),
        }

    def with_updates(self, updates: dict[str, str]) -> "PromptBundle":
        """Return a new bundle with selected prompt texts replaced."""
        if not updates:
            return self

        updated_prompts = self.as_dict()
        for name, text in updates.items():
            if name not in updated_prompts:
                raise KeyError(name)
            updated_prompts[name] = text

        return PromptBundle(updated_prompts, editable_targets=self.editable_targets)

    def serialize(self) -> str:
        """Return a deterministic JSON representation for storage."""
        return json.dumps(self.as_payload(), indent=2, sort_keys=True, ensure_ascii=False)


def normalize_prompt_bundle(
    value: PromptBundle | str | dict[str, str],
    *,
    default_target: str = DEFAULT_PROMPT_TARGET,
    editable_targets: Collection[str] | None = None,
) -> PromptBundle:
    """Normalize prompt inputs into a canonical bundle."""
    return PromptBundle.from_value(
        value,
        default_target=default_target,
        editable_targets=editable_targets,
    )


def serialize_prompt_bundle(
    value: PromptBundle | str | dict[str, str],
    *,
    default_target: str = DEFAULT_PROMPT_TARGET,
    editable_targets: Collection[str] | None = None,
) -> str:
    """Normalize prompt inputs and return canonical serialized content."""
    return normalize_prompt_bundle(
        value,
        default_target=default_target,
        editable_targets=editable_targets,
    ).serialize()
