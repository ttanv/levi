"""
Punctuated Equilibrium: Periodic paradigm shifts in evolutionary search.

Inspired by the biological theory that evolution proceeds in bursts of rapid
change separated by long periods of stasis. This module implements periodic
"paradigm shift" events that inject fundamentally new solutions into the archive.
"""

import asyncio
import logging
import random
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans

from ..config import LeviConfig
from ..core import Program, EvaluationResult
from ..pool import CVTMAPElitesPool
from ..pool.cvt_map_elites import Elite
from ..llm import get_llm_client
from ..utils import ResilientProcessPool, extract_code, extract_fn_name, evaluate_code, coerce_score
from ..pipeline.state import PipelineState, BudgetLimitReached
from .prompts import PARADIGM_SHIFT_PROMPT, PARADIGM_SHIFT_PROMPTS, VARIANT_GENERATION_PROMPT, get_budget_stage

logger = logging.getLogger(__name__)


class PunctuatedEquilibrium:
    """
    Implements punctuated equilibrium for CVT-MAP-Elites.

    Periodically:
    1. Clusters occupied centroids into behavioral regions
    2. Selects best elite from each cluster as representative
    3. Generates paradigm shift solution using heavy model
    4. Generates variants using lighter models
    5. Inserts solutions with behavior noise
    """

    def __init__(
        self,
        config: LeviConfig,
        pool: CVTMAPElitesPool,
        executor: ResilientProcessPool,
        archive_lock: asyncio.Lock,
        state: PipelineState,
    ):
        self.config = config
        self.pool = pool
        self.executor = executor
        self.archive_lock = archive_lock
        self.state = state
        self.pe_config = config.punctuated_equilibrium
        self.fn_name = extract_fn_name(config.function_signature)

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
            init='k-means++',
            n_init=3,
            random_state=None  # Different clustering each time
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
            best_idx, best_elite = max(
                cluster_elites,
                key=lambda x: x[1].result.primary_score
            )
            representatives.append((cluster_id, best_elite))

        return representatives

    def _build_paradigm_shift_prompt(
        self,
        representatives: list[tuple[int, Elite]],
        n_evaluations: int,
        budget_progress: float = 0.0,
    ) -> str:
        """Build prompt for paradigm shift generation.

        Args:
            representatives: Cluster representatives (cluster_id, elite) pairs.
            n_evaluations: Current evaluation count.
            budget_progress: Fraction of budget consumed (0-1). Controls prompt
                aggressiveness: early budget = radical shifts, late = refinement.
        """
        stage = get_budget_stage(budget_progress)
        logger.info(f"[PE] Budget progress: {budget_progress:.1%} -> stage={stage}")

        rep_text_parts = []
        for i, (cluster_id, elite) in enumerate(representatives):
            score = elite.result.primary_score
            code = elite.program.content
            rep_text_parts.append(
                f"### Region {i+1} (Cluster {cluster_id}, Score: {score:.1f})\n"
                f"```python\n{code}\n```"
            )

        representative_solutions = "\n\n".join(rep_text_parts)

        # Check for optimized paradigm shift instructions
        override = self.config.prompt_overrides.get("paradigm_shift")
        if override:
            # Use optimized prompt instead of default template
            return f"""# Algorithmic Paradigm Shift Challenge

## Problem
{self.config.problem_description}

## Function Signature
```python
{self.config.function_signature}
```

## Current Best Solutions ({len(representatives)} regions, {n_evaluations} evaluations)

{representative_solutions}

## Your Task
{override}

Output ONLY complete, runnable Python code in a ```python block.
"""

        # Use stage-appropriate prompt template
        template = PARADIGM_SHIFT_PROMPTS[stage]
        return template.format(
            problem_description=self.config.problem_description,
            function_signature=self.config.function_signature,
            n_evaluations=n_evaluations,
            n_regions=len(representatives),
            representative_solutions=representative_solutions,
        )

    def _build_variant_prompt(self, base_code: str, base_score: float) -> str:
        """Build prompt for variant generation."""
        return VARIANT_GENERATION_PROMPT.format(
            problem_description=self.config.problem_description,
            function_signature=self.config.function_signature,
            base_code=base_code,
            base_score=base_score,
        )

    async def _evaluate(self, code: str) -> dict:
        """Evaluate code using the executor."""
        return await self.executor.run(
            evaluate_code,
            code,
            self.config.score_fn,
            self.config.inputs,
            self.fn_name,
            timeout=self.config.pipeline.eval_timeout
        )

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
            eval_limit = int(self.state._coerce_finite_float(self.state.budget.evaluations, default=0.0))
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
            logger.info(f"[PE] Cluster {cluster_id} rep: score={elite.result.primary_score:.1f}")

        # Step 3: Generate paradigm shift solution
        heavy_models = self.pe_config.heavy_models
        if not heavy_models:
            heavy_models = [self.config.sampler_model_pairs[0].model]
        heavy_model = random.choice(heavy_models)

        prompt = self._build_paradigm_shift_prompt(representatives, n_evaluations, budget_progress)

        try:
            # Build call kwargs
            llm = get_llm_client()

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
                llm,
                model=heavy_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.pe_config.temperature,
                max_tokens=4096,
                timeout=300,
                **extras,
            )
            content = response.content
            cost = response.cost
            stats["total_cost"] += cost
        except BudgetLimitReached:
            logger.info("[PE] Budget exhausted before paradigm shift generation")
            return stats
        except Exception as e:
            logger.warning(f"[PE] Paradigm shift generation failed: {e}")
            return stats

        paradigm_code = extract_code(content)
        if not paradigm_code:
            logger.warning("[PE] Failed to extract paradigm shift code")
            return stats

        stats["paradigm_generated"] = True
        logger.debug("[PE] Paradigm shift code generated (%d chars)", len(paradigm_code))

        # Step 4: Evaluate paradigm shift solution
        if not can_start_pe_eval():
            logger.info("[PE] Skipping paradigm shift evaluation (budget exhausted)")
            stats["evaluations"].append({
                "source": "paradigm_shift",
                "model": heavy_model,
                "error": "Budget exhausted",
                "archive_size": self.pool.size(),
            })
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

            program = Program(content=paradigm_code, metadata={
                "source": "punctuated_equilibrium",
                "pe_type": "paradigm_shift",
            })
            eval_result = EvaluationResult(
                scores=result,
                is_valid=True,
            )

            async with self.archive_lock:
                accepted, cell_idx = self.pool.add_with_behavior_noise(
                    program,
                    eval_result,
                    noise_scale=self.pe_config.behavior_noise,
                )

            stats["paradigm_accepted"] = accepted
            stats["paradigm_cell"] = cell_idx
            stats["evaluations"].append({
                "source": "paradigm_shift",
                "model": heavy_model,
                "score": score,
                "accepted": accepted,
                "cell_index": cell_idx,
                "archive_size": self.pool.size(),
            })

            logger.info(f"[PE] Paradigm shift: score={score:.1f}, "
                       f"accepted={accepted}, cell={cell_idx}")
        else:
            error_message = str(result.get('error', 'unknown'))
            logger.info(f"[PE] Paradigm shift eval error: {error_message[:100]}")
            stats["evaluations"].append({
                "source": "paradigm_shift",
                "model": heavy_model,
                "error": error_message,
                "archive_size": self.pool.size(),
            })
            paradigm_code = None  # Can't generate variants

        # Step 5: Generate variants (only if paradigm was valid)
        if paradigm_code and stats["paradigm_score"] is not None:
            variant_models = self.pe_config.variant_models
            if not variant_models:
                # Use lighter models from config, cycle through sampler models
                variant_models = [p.model for p in self.config.sampler_model_pairs[:3]]
                if not variant_models:
                    variant_models = [heavy_model]

            logger.info(f"[PE] Generating {self.pe_config.n_variants} variants using models: {variant_models[:3]}...")

            variant_prompt = self._build_variant_prompt(
                paradigm_code, stats["paradigm_score"]
            )

            async def generate_variant(model: str, idx: int):
                try:
                    llm = get_llm_client()
                    response = await self.state.acompletion(
                        llm,
                        model=model,
                        messages=[{"role": "user", "content": variant_prompt}],
                        temperature=self.pe_config.temperature,
                        max_tokens=4096,
                        timeout=300,
                    )
                    return {"idx": idx, "content": response.content, "cost": response.cost, "model": model}
                except BudgetLimitReached:
                    return {"idx": idx, "error": "Budget exhausted", "model": model}
                except Exception as e:
                    return {"idx": idx, "error": str(e), "model": model}

            # Generate variants in parallel
            tasks = [
                generate_variant(
                    variant_models[i % len(variant_models)], i
                )
                for i in range(self.pe_config.n_variants)
            ]
            variant_results = await asyncio.gather(*tasks)

            # Evaluate and insert variants
            for vr in variant_results:
                if "error" in vr:
                    logger.warning(f"[PE] Variant {vr['idx']} generation failed ({vr.get('model', '?')}): {vr['error'][:100]}")
                    continue

                stats["total_cost"] += vr["cost"]

                variant_code = extract_code(vr["content"])
                if not variant_code:
                    logger.warning(f"[PE] Variant {vr['idx']} code extraction failed ({vr.get('model', '?')})")
                    continue

                stats["variants_generated"] += 1

                if not can_start_pe_eval():
                    logger.info(f"[PE] Skipping variant {vr['idx']} evaluation (budget exhausted)")
                    stats["evaluations"].append({
                        "source": "variant",
                        "model": vr.get("model", "unknown"),
                        "error": "Budget exhausted",
                        "archive_size": self.pool.size(),
                    })
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
                    program = Program(content=variant_code, metadata={
                        "source": "punctuated_equilibrium",
                        "pe_type": "variant",
                    })
                    eval_result = EvaluationResult(
                        scores=result,
                        is_valid=True,
                    )

                    async with self.archive_lock:
                        accepted, cell_idx = self.pool.add_with_behavior_noise(
                            program,
                            eval_result,
                            noise_scale=self.pe_config.behavior_noise,
                        )

                    if accepted:
                        stats["variants_accepted"] += 1
                        stats["variant_cells"].append(cell_idx)
                    stats["evaluations"].append({
                        "source": "variant",
                        "model": vr.get("model", "unknown"),
                        "score": score,
                        "accepted": accepted,
                        "cell_index": cell_idx,
                        "archive_size": self.pool.size(),
                    })

                    logger.info(f"[PE] Variant {vr['idx']}: score={score:.1f}, "
                               f"accepted={accepted}")
                else:
                    error_message = str(result.get('error', 'unknown'))
                    logger.warning(f"[PE] Variant {vr['idx']} eval error: {error_message[:100]}")
                    stats["evaluations"].append({
                        "source": "variant",
                        "model": vr.get("model", "unknown"),
                        "error": error_message,
                        "archive_size": self.pool.size(),
                    })

        logger.info(f"[PE] Complete: paradigm_accepted={stats['paradigm_accepted']}, "
                   f"variants={stats['variants_accepted']}/{stats['variants_generated']}, "
                   f"cost=${stats['total_cost']:.3f}")

        return stats
