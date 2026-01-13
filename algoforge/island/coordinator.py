"""
Island Coordinator for Distributed Quality-Diversity Evolution.

Manages multiple CVT-MAP-Elites archives (islands) with ring-based
migration topology and random elite selection.
"""

import logging
import random
from dataclasses import dataclass, field
from typing import Optional

from ..core import Program, EvaluationResult
from ..pool import CVTMAPElitesPool
from ..behavior import BehaviorExtractor

logger = logging.getLogger(__name__)


@dataclass
class Migrant:
    """A program being migrated between islands."""
    program: Program
    result: EvaluationResult
    raw_behavior: dict[str, float]


@dataclass
class Island:
    """An island with its own CVT-MAP-Elites archive."""
    index: int
    pool: CVTMAPElitesPool
    eval_count: int = 0
    acceptance_count: int = 0
    incoming_buffer: list[Migrant] = field(default_factory=list)
    last_migration_eval: int = 0

    @property
    def best_score(self) -> float:
        """Get best score in this island's archive."""
        if self.pool.size() == 0:
            return float('-inf')
        return self.pool._best_score

    def get_random_elites(self, n: int) -> list[Migrant]:
        """
        Get n random elites from the archive for migration.

        Uses random selection (not top-k by fitness) to maintain
        behavioral diversity during migration.
        """
        elites = self.pool.get_elites()
        if not elites:
            return []

        filled_cells = list(elites.keys())
        selected_cells = random.sample(filled_cells, min(n, len(filled_cells)))

        migrants = []
        for cell_idx in selected_cells:
            elite = elites[cell_idx]
            if elite.raw_behavior is not None:
                migrants.append(Migrant(
                    program=elite.program,
                    result=elite.result,
                    raw_behavior=elite.raw_behavior,
                ))
        return migrants

    def receive_migrants(self, migrants: list[Migrant]) -> int:
        """
        Add migrants to incoming buffer for later processing.

        Returns the number of migrants added to buffer.
        """
        self.incoming_buffer.extend(migrants)
        return len(migrants)

    def process_incoming(self) -> int:
        """
        Process all migrants in the incoming buffer.

        Attempts to add each migrant to this island's archive using
        the migrant's raw behavior for re-normalization.

        Returns the number of migrants accepted into the archive.
        """
        accepted = 0
        for migrant in self.incoming_buffer:
            if self.pool.add_with_raw_behavior(
                migrant.program,
                migrant.result,
                migrant.raw_behavior,
            ):
                accepted += 1
                self.acceptance_count += 1

        self.incoming_buffer.clear()
        return accepted


class IslandCoordinator:
    """
    Coordinates multiple islands with ring-based migration.

    Each island has its own independent CVT-MAP-Elites archive with
    its own centroids and adaptive bounds. Migration occurs in a
    ring topology (island i -> island (i+1) % n) using random elite
    selection to preserve behavioral diversity.
    """

    def __init__(
        self,
        n_islands: int,
        behavior_extractor: BehaviorExtractor,
        n_centroids: int = 50,
        migration_interval: int = 100,
        migrants_per_event: int = 5,
        subscore_keys: Optional[list[str]] = None,
    ):
        """
        Initialize the island coordinator.

        Args:
            n_islands: Number of islands to create
            behavior_extractor: Shared behavior extractor for all islands
            n_centroids: Number of CVT centroids per island
            migration_interval: Evals per island before migration triggers
            migrants_per_event: Number of random elites to migrate
            subscore_keys: Optional subscore keys for samplers
        """
        self.n_islands = n_islands
        self.behavior_extractor = behavior_extractor
        self.n_centroids = n_centroids
        self.migration_interval = migration_interval
        self.migrants_per_event = migrants_per_event
        self.subscore_keys = subscore_keys

        # Create islands with deferred centroid initialization
        self.islands: list[Island] = []
        for i in range(n_islands):
            pool = CVTMAPElitesPool(
                behavior_extractor=behavior_extractor,
                n_centroids=n_centroids,
                defer_centroids=True,  # Centroids set during island init
                subscore_keys=subscore_keys,
            )
            self.islands.append(Island(index=i, pool=pool))

        self._total_migrations = 0

    def get_island(self, index: int) -> Island:
        """Get island by index."""
        return self.islands[index]

    def add_to_island(
        self,
        island_idx: int,
        program: Program,
        result: EvaluationResult,
    ) -> bool:
        """
        Add a program to a specific island's archive.

        Also increments the island's eval count and triggers
        migration check if the interval is reached.

        Returns True if the program was accepted.
        """
        island = self.islands[island_idx]
        accepted, _ = island.pool.add(program, result)

        island.eval_count += 1
        if accepted:
            island.acceptance_count += 1

        # Check if migration should occur
        evals_since_migration = island.eval_count - island.last_migration_eval
        if evals_since_migration >= self.migration_interval:
            self.migrate(island_idx)
            island.last_migration_eval = island.eval_count

        return accepted

    def migrate(self, source_idx: int) -> int:
        """
        Migrate random elites from source island to next island in ring.

        Ring topology: island i -> island (i+1) % n

        Returns the number of migrants sent.
        """
        target_idx = (source_idx + 1) % self.n_islands
        source = self.islands[source_idx]
        target = self.islands[target_idx]

        # Get random elites (not top-k by fitness)
        migrants = source.get_random_elites(self.migrants_per_event)

        if migrants:
            target.receive_migrants(migrants)
            self._total_migrations += len(migrants)
            logger.info(
                f"[Migration] Island {source_idx} -> {target_idx}: "
                f"{len(migrants)} migrants sent"
            )

        return len(migrants)

    def process_all_incoming(self) -> dict[int, int]:
        """
        Process incoming migrants for all islands.

        Returns dict mapping island_idx -> number of migrants accepted.
        """
        results = {}
        for island in self.islands:
            if island.incoming_buffer:
                accepted = island.process_incoming()
                results[island.index] = accepted
                if accepted > 0:
                    logger.info(
                        f"[Migration] Island {island.index} accepted "
                        f"{accepted} migrants"
                    )
        return results

    def get_global_best(self) -> tuple[Optional[Program], float]:
        """Get the best program across all islands."""
        best_program = None
        best_score = float('-inf')

        for island in self.islands:
            if island.pool.size() > 0:
                island_best = island.pool.best()
                island_score = island.best_score
                if island_score > best_score:
                    best_score = island_score
                    best_program = island_best

        return best_program, best_score

    def get_stats(self) -> dict:
        """Get statistics for all islands."""
        island_stats = []
        for island in self.islands:
            island_stats.append({
                "index": island.index,
                "archive_size": island.pool.size(),
                "best_score": island.best_score,
                "eval_count": island.eval_count,
                "acceptance_count": island.acceptance_count,
                "acceptance_rate": (
                    island.acceptance_count / island.eval_count
                    if island.eval_count > 0 else 0.0
                ),
            })

        global_best_program, global_best_score = self.get_global_best()
        total_evals = sum(i.eval_count for i in self.islands)
        total_accepted = sum(i.acceptance_count for i in self.islands)

        return {
            "n_islands": self.n_islands,
            "total_evals": total_evals,
            "total_accepted": total_accepted,
            "total_migrations": self._total_migrations,
            "global_best_score": global_best_score,
            "islands": island_stats,
        }

    def get_total_archive_size(self) -> int:
        """Get total number of elites across all islands."""
        return sum(island.pool.size() for island in self.islands)

    def perform_culling(self, n_seed_elites: int = 10) -> dict:
        """
        Cull bottom half of islands and reseed from top half.

        Ranks islands by best score, clears the bottom half, then seeds
        each cleared island with top elites from a different top-half island.

        Args:
            n_seed_elites: Number of top elites to seed into each cleared island

        Returns:
            Dict with culling statistics
        """
        if self.n_islands < 2:
            return {"culled": 0, "seeded": 0}

        # Rank islands by best score
        ranked = sorted(
            self.islands,
            key=lambda i: i.best_score,
            reverse=True
        )

        n_top = (self.n_islands + 1) // 2  # Ceiling division for odd counts
        top_islands = ranked[:n_top]
        bottom_islands = ranked[n_top:]

        if not bottom_islands:
            return {"culled": 0, "seeded": 0}

        logger.info(
            f"[Culling] Top islands: {[i.index for i in top_islands]} | "
            f"Bottom islands: {[i.index for i in bottom_islands]}"
        )

        # Clear bottom islands
        total_cleared = 0
        for island in bottom_islands:
            n_cleared = island.pool.clear()
            total_cleared += n_cleared
            island.eval_count = 0
            island.acceptance_count = 0
            logger.info(f"[Culling] Cleared island {island.index}: {n_cleared} elites removed")

        # Seed each bottom island from a different top island (round-robin)
        total_seeded = 0
        for i, bottom_island in enumerate(bottom_islands):
            source_island = top_islands[i % len(top_islands)]
            top_elites = source_island.pool.get_top_elites(n_seed_elites)

            seeded = 0
            for elite in top_elites:
                if elite.raw_behavior is not None:
                    if bottom_island.pool.add_with_raw_behavior(
                        elite.program,
                        elite.result,
                        elite.raw_behavior,
                    ):
                        seeded += 1

            total_seeded += seeded
            logger.info(
                f"[Culling] Seeded island {bottom_island.index} from island "
                f"{source_island.index}: {seeded}/{len(top_elites)} elites"
            )

        return {
            "culled": len(bottom_islands),
            "cleared_elites": total_cleared,
            "seeded_elites": total_seeded,
            "top_islands": [i.index for i in top_islands],
            "bottom_islands": [i.index for i in bottom_islands],
        }
