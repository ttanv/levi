"""
Punctuated Equilibrium: Periodic paradigm shifts in evolutionary search.

Inspired by the biological theory that evolution proceeds in bursts of rapid
change separated by long periods of stasis. This module implements periodic
"paradigm shift" events that inject fundamentally new solutions into the archive.
"""

import asyncio
import logging
import random

from sklearn.cluster import KMeans

from ..artifacts import ArtifactAdapter
from ..clients.base import client_name
from ..config import LeviConfig
from ..core import EvaluationResult
from ..pipeline.state import BudgetLimitReached, PipelineState, coerce_finite_float
from ..pool import CVTMAPElitesPool
from ..pool.cvt_map_elites import Elite
from ..prompts import PromptBundle
from ..selection import ComponentSelector, make_component_selector
from ..utils import ResilientProcessPool, coerce_score

logger = logging.getLogger(__name__)


class PunctuatedEquilibrium:
    """
    Implements punctuated equilibrium for CVT-MAP-Elites.

    Periodically:
    1. Clusters occupied centroids into behavioral regions
    2. Selects best elite from each cluster as representative
    3. Generates paradigm shift solution using heavy model
    4. Generates variants using lighter models
    5. Inserts solutions into the archive
    """

    def __init__(
        self,
        config: LeviConfig,
        pool: CVTMAPElitesPool,
        executor: ResilientProcessPool,
        artifact_adapter: ArtifactAdapter,
        archive_lock: asyncio.Lock,
        state: PipelineState,
        main_component_selector: ComponentSelector | None = None,
    ):
        self.config = config
        self.pool = pool
        self.executor = executor
        self.artifact_adapter = artifact_adapter
        self.archive_lock = archive_lock
        self.state = state
        self.pe_config = config.punctuated_equilibrium
        self.main_component_selector = main_component_selector

        self.pe_component_selector: ComponentSelector | None = None
        self._is_bundle = getattr(artifact_adapter, "is_bundle_artifact", False)
        if self._is_bundle:
            self.pe_component_selector = make_component_selector(self.pe_config.component_selector)

    def _cluster_occupied_centroids(self) -> dict[int, list[int]]:
        """
        Cluster occupied centroids and return mapping of cluster_id -> cell_indices.

        Returns empty dict if not enough occupied cells for clustering.
        """
        elites = self.pool.get_elites()
        if len(elites) < self.pe_config.n_clusters:
            logger.info(f"[PE] Not enough elites ({len(elites)}) for {self.pe_config.n_clusters} clusters")
            return {}

        cell_indices = list(elites.keys())
        centroids = self.pool._centroids

        if centroids is None:
            logger.warning("[PE] Centroids not initialized")
            return {}

        # Get centroid vectors for occupied cells only
        occupied_centroids = centroids[cell_indices]

        n_clusters = min(self.pe_config.n_clusters, len(cell_indices))
        kmeans = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            n_init=3,
            random_state=None,  # Different clustering each time
        )
        labels = kmeans.fit_predict(occupied_centroids)

        # Build cluster -> cell_indices mapping
        clusters: dict[int, list[int]] = {}
        for idx, label in enumerate(labels):
            label_int = int(label)
            if label_int not in clusters:
                clusters[label_int] = []
            clusters[label_int].append(cell_indices[idx])

        return clusters

    def _select_cluster_representatives(
        self,
        clusters: dict[int, list[int]],
    ) -> list[tuple[int, Elite]]:
        """
        Select highest-performing elite from each cluster.

        Returns list of (cluster_id, elite) tuples.
        """
        representatives = []
        elites = self.pool.get_elites()

        for cluster_id, cell_indices in clusters.items():
            cluster_elites = [(idx, elites[idx]) for idx in cell_indices]
            best_idx, best_elite = max(cluster_elites, key=lambda x: x[1].result.primary_score)
            representatives.append((cluster_id, best_elite))

        return representatives

    def _build_paradigm_shift_prompt(
        self,
        representatives: list[tuple[int, Elite]],
        n_evaluations: int,
        budget_progress: float = 0.0,
        target: str | None = None,
    ) -> str:
        if target is not None and self._is_bundle:
            return self.artifact_adapter.build_component_paradigm_shift_prompt(
                target,
                representatives,
                n_evaluations=n_evaluations,
                budget_progress=budget_progress,
            )
        return self.artifact_adapter.build_paradigm_shift_prompt(
            representatives,
            n_evaluations=n_evaluations,
            budget_progress=budget_progress,
        )

    def _build_variant_prompt(
        self,
        base_code: str,
        base_score: float,
        target: str | None = None,
    ) -> str:
        if target is not None and self._is_bundle:
            base_bundle = PromptBundle.deserialize_loose(base_code)
            return self.artifact_adapter.build_component_variant_prompt(target, base_bundle, base_score)
        return self.artifact_adapter.build_variant_prompt(base_code, base_score)

    def _pick_pe_component(self) -> str | None:
        if not self._is_bundle or self.pe_component_selector is None:
            return None
        seed_bundle = getattr(self.artifact_adapter, "seed_bundle", None)
        if seed_bundle is None:
            return None
        context: dict = {}
        if self.pe_config.share_main_selector_stats and self.main_component_selector is not None:
            context["main_stats"] = self.main_component_selector.stats()
        return self.pe_component_selector.select(
            list(seed_bundle.editable_targets), context=context or None
        )

    async def _evaluate(self, code: str) -> dict:
        """Evaluate code using the executor."""
        return await self.artifact_adapter.evaluate(self.executor, code)

    async def trigger(self, n_evaluations: int, budget_progress: float = 0.0) -> dict:
        """
        Trigger a punctuated equilibrium event.

        Args:
            n_evaluations: Current evaluation count (for prompt context)
            budget_progress: Fraction of budget consumed (0-1)

        Returns:
            Dict with statistics about the PE event:
            {
                "triggered": bool,
                "paradigm_generated": bool,
                "paradigm_score": Optional[float],
                "paradigm_accepted": bool,
                "paradigm_cell": Optional[int],
                "variants_generated": int,
                "variants_accepted": int,
                "total_cost": float,
            }
        """
        stats = {
            "triggered": True,
            "paradigm_generated": False,
            "paradigm_score": None,
            "paradigm_accepted": False,
            "paradigm_cell": None,
            "variants_generated": 0,
            "variants_accepted": 0,
            "variant_cells": [],
            "total_cost": 0.0,
            "evaluations": [],
        }
        pe_evals_started = 0

        def can_start_pe_eval() -> bool:
            if self.state.budget_exhausted:
                return False
            if self.state.budget.evaluations is None:
                return True
            eval_limit = int(coerce_finite_float(self.state.budget.evaluations, default=0.0))
            if eval_limit <= 0:
                return False
            return (self.state.eval_count + pe_evals_started) < eval_limit

        # Step 1: Cluster occupied centroids
        async with self.archive_lock:
            clusters = self._cluster_occupied_centroids()

        if not clusters:
            stats["triggered"] = False
            return stats

        logger.info(f"[PE] Clustered {len(self.pool.get_elites())} elites into {len(clusters)} clusters")

        # Step 2: Select cluster representatives
        async with self.archive_lock:
            representatives = self._select_cluster_representatives(clusters)

        for cluster_id, elite in representatives:
            logger.info(f"[PE] Cluster {cluster_id} rep: score={elite.result.primary_score:.17g}")

        # Step 3: Generate paradigm shift solution
        heavy_models = self.pe_config.heavy_models
        if not heavy_models:
            heavy_models = [self.config.sampler_model_pairs[0].model]
        heavy_model = random.choice(heavy_models)

        pe_target = self._pick_pe_component()
        if pe_target is not None:
            logger.info(f"[PE] Selected component for paradigm shift: {pe_target}")
            stats["pe_target"] = pe_target
        prompt = self._build_paradigm_shift_prompt(
            representatives, n_evaluations, budget_progress, target=pe_target
        )

        try:
            extras = {}
            # Add reasoning_effort for DeepSeek models if configured
            if self.pe_config.reasoning_effort:
                if self.pe_config.reasoning_effort == "disabled":
                    # Disable reasoning entirely (e.g., for GLM models)
                    extras["extra_body"] = {"reasoning": {"enabled": False}}
                    logger.info("[PE] Reasoning disabled for paradigm shift")
                else:
                    extras["reasoning_effort"] = self.pe_config.reasoning_effort
                    logger.info(f"[PE] Using reasoning_effort={self.pe_config.reasoning_effort} for paradigm shift")

            response = await self.state.acompletion(
                heavy_model,
                prompt=[{"role": "user", "content": prompt}],
                temperature=self.pe_config.temperature,
                max_tokens=4096,
                timeout=300,
                **extras,
            )
            content = response.text
            cost = response.cost
            stats["total_cost"] += cost
        except BudgetLimitReached:
            logger.info("[PE] Budget exhausted before paradigm shift generation")
            return stats
        except Exception as e:
            logger.warning(f"[PE] Paradigm shift generation failed: {e}")
            return stats

        if pe_target is not None and representatives:
            anchor_elite = max(representatives, key=lambda item: item[1].result.primary_score)[1]
            paradigm_code = self.artifact_adapter.extract_candidate(
                content,
                parent_content=anchor_elite.program.content,
                target=pe_target,
            )
        else:
            paradigm_code = self.artifact_adapter.extract_candidate(content)
        if not paradigm_code:
            logger.warning("[PE] Failed to extract paradigm shift code")
            return stats

        stats["paradigm_generated"] = True
        logger.debug("[PE] Paradigm shift code generated (%d chars)", len(paradigm_code))

        # Step 4: Evaluate paradigm shift solution
        if not can_start_pe_eval():
            logger.info("[PE] Skipping paradigm shift evaluation (budget exhausted)")
            stats["evaluations"].append(
                {
                    "source": "paradigm_shift",
                    "model": client_name(heavy_model),
                    "error": "Budget exhausted",
                    "archive_size": self.pool.size(),
                }
            )
            return stats

        try:
            pe_evals_started += 1
            result = await self._evaluate(paradigm_code)
        except Exception as e:
            logger.warning(f"[PE] Paradigm shift evaluation failed: {e}")
            result = {"error": str(e)}

        if "error" not in result:
            score, score_error = coerce_score(result)
            if score_error is not None:
                logger.warning(f"[PE] Paradigm shift invalid score: {score_error}")
                result = {"error": score_error}
            else:
                result = dict(result)
                result["score"] = score

        if "error" not in result:
            stats["paradigm_score"] = score

            program = self.artifact_adapter.make_program(
                paradigm_code,
                metadata={
                    "source": "punctuated_equilibrium",
                    "pe_type": "paradigm_shift",
                },
            )
            eval_result = EvaluationResult(
                scores=result,
                is_valid=True,
            )

            async with self.archive_lock:
                accepted, cell_idx = self.pool.add(program, eval_result)

            stats["paradigm_accepted"] = accepted
            stats["paradigm_cell"] = cell_idx
            if pe_target is not None and self.pe_component_selector is not None:
                self.pe_component_selector.update(pe_target, accepted=accepted)
            stats["evaluations"].append(
                {
                    "source": "paradigm_shift",
                    "model": client_name(heavy_model),
                    "score": score,
                    "accepted": accepted,
                    "cell_index": cell_idx,
                    "archive_size": self.pool.size(),
                }
            )

            logger.info(f"[PE] Paradigm shift: score={score:.17g}, accepted={accepted}, cell={cell_idx}")
        else:
            error_message = str(result.get("error", "unknown"))
            logger.info(f"[PE] Paradigm shift eval error: {error_message[:100]}")
            stats["evaluations"].append(
                {
                    "source": "paradigm_shift",
                    "model": client_name(heavy_model),
                    "error": error_message,
                    "archive_size": self.pool.size(),
                }
            )
            paradigm_code = None  # Can't generate variants

        # Step 5: Generate variants (only if paradigm was valid)
        if paradigm_code and stats["paradigm_score"] is not None:
            variant_models = self.pe_config.variant_models
            if not variant_models:
                # Use lighter models from config, cycle through sampler models
                variant_models = [p.model for p in self.config.sampler_model_pairs[:3]]
                if not variant_models:
                    variant_models = [heavy_model]

            logger.info(
                f"[PE] Generating {self.pe_config.n_variants} variants using models: "
                f"{[client_name(model) for model in variant_models[:3]]}..."
            )

            variant_prompt = self._build_variant_prompt(
                paradigm_code, stats["paradigm_score"], target=pe_target
            )

            async def generate_variant(model, idx: int):
                try:
                    response = await self.state.acompletion(
                        model,
                        prompt=[{"role": "user", "content": variant_prompt}],
                        temperature=self.pe_config.temperature,
                        max_tokens=4096,
                        timeout=300,
                    )
                    return {
                        "idx": idx,
                        "content": response.text,
                        "cost": response.cost,
                        "model": client_name(model),
                    }
                except BudgetLimitReached:
                    return {"idx": idx, "error": "Budget exhausted", "model": client_name(model)}
                except Exception as e:
                    return {"idx": idx, "error": str(e), "model": client_name(model)}

            # Generate variants in parallel
            tasks = [
                generate_variant(variant_models[i % len(variant_models)], i) for i in range(self.pe_config.n_variants)
            ]
            variant_results = await asyncio.gather(*tasks)

            # Evaluate and insert variants
            for vr in variant_results:
                if "error" in vr:
                    logger.warning(
                        f"[PE] Variant {vr['idx']} generation failed ({vr.get('model', '?')}): {vr['error'][:100]}"
                    )
                    continue

                stats["total_cost"] += vr["cost"]

                if pe_target is not None:
                    variant_code = self.artifact_adapter.extract_candidate(
                        vr["content"],
                        parent_content=paradigm_code,
                        target=pe_target,
                    )
                else:
                    variant_code = self.artifact_adapter.extract_candidate(vr["content"])
                if not variant_code:
                    logger.warning(f"[PE] Variant {vr['idx']} code extraction failed ({vr.get('model', '?')})")
                    continue

                stats["variants_generated"] += 1

                if not can_start_pe_eval():
                    logger.info(f"[PE] Skipping variant {vr['idx']} evaluation (budget exhausted)")
                    stats["evaluations"].append(
                        {
                            "source": "variant",
                            "model": vr.get("model", "unknown"),
                            "error": "Budget exhausted",
                            "archive_size": self.pool.size(),
                        }
                    )
                    continue

                try:
                    pe_evals_started += 1
                    result = await self._evaluate(variant_code)
                except Exception as e:
                    logger.warning(f"[PE] Variant {vr['idx']} evaluation exception: {e}")
                    result = {"error": str(e)}

                if "error" not in result:
                    score, score_error = coerce_score(result)
                    if score_error is not None:
                        logger.warning(f"[PE] Variant {vr['idx']} invalid score: {score_error}")
                        result = {"error": score_error}
                    else:
                        result = dict(result)
                        result["score"] = score

                if "error" not in result:
                    program = self.artifact_adapter.make_program(
                        variant_code,
                        metadata={
                            "source": "punctuated_equilibrium",
                            "pe_type": "variant",
                        },
                    )
                    eval_result = EvaluationResult(
                        scores=result,
                        is_valid=True,
                    )

                    async with self.archive_lock:
                        accepted, cell_idx = self.pool.add(program, eval_result)

                    if accepted:
                        stats["variants_accepted"] += 1
                        stats["variant_cells"].append(cell_idx)
                    if pe_target is not None and self.pe_component_selector is not None:
                        self.pe_component_selector.update(pe_target, accepted=accepted)
                    stats["evaluations"].append(
                        {
                            "source": "variant",
                            "model": vr.get("model", "unknown"),
                            "score": score,
                            "accepted": accepted,
                            "cell_index": cell_idx,
                            "archive_size": self.pool.size(),
                        }
                    )

                    logger.info(f"[PE] Variant {vr['idx']}: score={score:.17g}, accepted={accepted}")
                else:
                    error_message = str(result.get("error", "unknown"))
                    logger.warning(f"[PE] Variant {vr['idx']} eval error: {error_message[:100]}")
                    stats["evaluations"].append(
                        {
                            "source": "variant",
                            "model": vr.get("model", "unknown"),
                            "error": error_message,
                            "archive_size": self.pool.size(),
                        }
                    )

        logger.info(
            f"[PE] Complete: paradigm_accepted={stats['paradigm_accepted']}, "
            f"variants={stats['variants_accepted']}/{stats['variants_generated']}, "
            f"cost=${stats['total_cost']:.3f}"
        )

        return stats
