"""
Punctuated Equilibrium: Periodic paradigm shifts in evolutionary search.

Inspired by the biological theory that evolution proceeds in bursts of rapid
change separated by long periods of stasis. This module implements periodic
"paradigm shift" events that inject fundamentally new solutions into the archive.
"""

import asyncio
import logging
import resource
import types
from typing import Optional

import litellm
import numpy as np
from sklearn.cluster import KMeans

from ..config import AlgoforgeConfig
from ..core import Program, EvaluationResult
from ..pool import CVTMAPElitesPool
from ..pool.cvt_map_elites import Elite
from ..utils import ResilientProcessPool, extract_code, extract_fn_name
from .prompts import PARADIGM_SHIFT_PROMPT, VARIANT_GENERATION_PROMPT

logger = logging.getLogger(__name__)


def _evaluate_code(code: str, score_fn, inputs: list, fn_name: str) -> dict:
    """Runs in subprocess: parse code, extract callable, call score_fn."""
    # Limit process memory to prevent VM crashes
    try:
        memory_bytes = 8 * 1024 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    except (ValueError, resource.error):
        pass  # May fail on some platforms

    namespace = {}
    try:
        exec(code, namespace)
    except SyntaxError as e:
        return {"error": f"Syntax error: {e}"}

    fn = namespace.get(fn_name)
    if not isinstance(fn, types.FunctionType):
        return {"error": f"Function '{fn_name}' not found (got {type(fn).__name__})"}

    try:
        result = score_fn(fn, inputs)
        return result
    except Exception as e:
        return {"error": f"Scoring error: {e}"}


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
        config: AlgoforgeConfig,
        pool: CVTMAPElitesPool,
        executor: ResilientProcessPool,
        archive_lock: asyncio.Lock,
    ):
        self.config = config
        self.pool = pool
        self.executor = executor
        self.archive_lock = archive_lock
        self.pe_config = config.punctuated_equilibrium
        self.fn_name = extract_fn_name(config.function_signature)

        # Track costs
        self.total_cost = 0.0

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
    ) -> str:
        """Build prompt for paradigm shift generation."""
        rep_text_parts = []
        for i, (cluster_id, elite) in enumerate(representatives):
            score = elite.result.primary_score
            code = elite.program.code
            rep_text_parts.append(
                f"### Region {i+1} (Cluster {cluster_id}, Score: {score:.1f})\n"
                f"```python\n{code}\n```"
            )

        representative_solutions = "\n\n".join(rep_text_parts)

        return PARADIGM_SHIFT_PROMPT.format(
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
            _evaluate_code,
            code,
            self.config.score_fn,
            self.config.inputs,
            self.fn_name,
            timeout=self.config.pipeline.eval_timeout
        )

    async def trigger(self, n_evaluations: int) -> dict:
        """
        Trigger a punctuated equilibrium event.

        Args:
            n_evaluations: Current evaluation count (for prompt context)

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
        }

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
        heavy_model = self.pe_config.heavy_model
        if not heavy_model:
            # Default to first sampler model
            heavy_model = self.config.sampler_model_pairs[0].model

        prompt = self._build_paradigm_shift_prompt(representatives, n_evaluations)

        try:
            # Build call kwargs
            call_kwargs = {
                "model": heavy_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.pe_config.temperature,
                "max_tokens": 30000,
                "timeout": 300,
            }

            # Add reasoning_effort for DeepSeek models if configured
            if self.pe_config.reasoning_effort:
                call_kwargs["reasoning_effort"] = self.pe_config.reasoning_effort
                logger.info(f"[PE] Using reasoning_effort={self.pe_config.reasoning_effort} for paradigm shift")

            if heavy_model in self.config.api_bases:
                call_kwargs["api_base"] = self.config.api_bases[heavy_model]

            response = await litellm.acompletion(**call_kwargs)
            content = response.choices[0].message.content
            cost = litellm.completion_cost(completion_response=response)
            stats["total_cost"] += cost
            self.total_cost += cost
        except Exception as e:
            logger.warning(f"[PE] Paradigm shift generation failed: {e}")
            return stats

        paradigm_code = extract_code(content)
        if not paradigm_code:
            logger.warning("[PE] Failed to extract paradigm shift code")
            return stats

        stats["paradigm_generated"] = True

        # Step 4: Evaluate paradigm shift solution
        try:
            result = await self._evaluate(paradigm_code)
        except Exception as e:
            logger.warning(f"[PE] Paradigm shift evaluation failed: {e}")
            result = {"error": str(e)}

        if "error" not in result:
            score = result.get("score", 0.0)
            stats["paradigm_score"] = score

            program = Program(code=paradigm_code, metadata={
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

            logger.info(f"[PE] Paradigm shift: score={score:.1f}, "
                       f"accepted={accepted}, cell={cell_idx}")
        else:
            logger.info(f"[PE] Paradigm shift eval error: {result.get('error', 'unknown')[:100]}")
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
                    kwargs = {
                        "model": model,
                        "messages": [{"role": "user", "content": variant_prompt}],
                        "temperature": self.pe_config.temperature,
                        "max_tokens": 30000,
                        "timeout": 300,
                    }
                    if model in self.config.api_bases:
                        kwargs["api_base"] = self.config.api_bases[model]
                    response = await litellm.acompletion(**kwargs)
                    content = response.choices[0].message.content
                    cost = litellm.completion_cost(completion_response=response)
                    return {"idx": idx, "content": content, "cost": cost, "model": model}
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
                self.total_cost += vr["cost"]

                variant_code = extract_code(vr["content"])
                if not variant_code:
                    logger.warning(f"[PE] Variant {vr['idx']} code extraction failed ({vr.get('model', '?')})")
                    continue

                stats["variants_generated"] += 1

                try:
                    result = await self._evaluate(variant_code)
                except Exception as e:
                    logger.warning(f"[PE] Variant {vr['idx']} evaluation exception: {e}")
                    result = {"error": str(e)}

                if "error" not in result:
                    score = result.get("score", 0.0)
                    program = Program(code=variant_code, metadata={
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

                    logger.info(f"[PE] Variant {vr['idx']}: score={score:.1f}, "
                               f"accepted={accepted}")
                else:
                    logger.warning(f"[PE] Variant {vr['idx']} eval error: {result.get('error', 'unknown')[:100]}")

        logger.info(f"[PE] Complete: paradigm_accepted={stats['paradigm_accepted']}, "
                   f"variants={stats['variants_accepted']}/{stats['variants_generated']}, "
                   f"cost=${stats['total_cost']:.3f}")

        return stats
