"""Internal artifact adapter abstractions for Levi."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from ..clients.base import ClientSpec
from ..core import Program
from ..prompts import ProgramWithScore
from ..utils import ResilientProcessPool


class ArtifactAdapter(ABC):
    """Internal boundary between the generic engine and an artifact domain."""

    artifact_type: str = "artifact"

    @abstractmethod
    def make_program(self, content: str, metadata: dict[str, Any] | None = None) -> Program:
        """Wrap raw artifact content into a Program."""

    @abstractmethod
    def snapshot_content(self, elite_data: Mapping[str, Any]) -> str:
        """Extract canonical content from a serialized snapshot entry."""

    @abstractmethod
    async def evaluate(
        self,
        executor: ResilientProcessPool,
        content: str,
        *,
        inputs: list[Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Evaluate artifact content with the domain-specific harness."""

    @abstractmethod
    def build_mutation_prompt(
        self,
        parents: Sequence[ProgramWithScore],
        *,
        meta_advice: str | None = None,
        model: ClientSpec | None = None,
        use_diff: bool = False,
    ) -> str:
        """Build the main mutation prompt for the producer pipeline."""

    @abstractmethod
    def extract_candidate(
        self,
        response_text: str,
        *,
        parent_content: str | None = None,
        use_diff: bool = False,
    ) -> str | None:
        """Extract candidate content from a model response."""

    @abstractmethod
    def build_diversity_prompt(self, existing_candidates: Sequence[tuple[str, float]]) -> str:
        """Build the init-phase diversity prompt."""

    @abstractmethod
    def build_init_variant_prompt(self, parents: Sequence[ProgramWithScore]) -> str:
        """Build the init-phase local-variation prompt."""

    @abstractmethod
    def build_paradigm_shift_prompt(
        self,
        representatives: Sequence[tuple[int, Any]],
        *,
        n_evaluations: int,
        budget_progress: float = 0.0,
    ) -> str:
        """Build the punctuated-equilibrium paradigm prompt."""

    @abstractmethod
    def build_variant_prompt(self, base_content: str, base_score: float) -> str:
        """Build a local-variation prompt around a paradigm-shift result."""
