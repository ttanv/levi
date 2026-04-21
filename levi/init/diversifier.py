"""Init phase: diverse seed generation and archive initialization."""

import asyncio
import json
import logging
import random
from pathlib import Path

import numpy as np

from ..artifacts import ArtifactAdapter
from ..behavior import BehaviorExtractor
from ..clients.base import client_name
from ..config import LeviConfig
from ..core import EvaluationResult
from ..pipeline.state import BudgetLimitReached, PipelineState, ScoreHistoryEntry
from ..pool import CVTMAPElitesPool
from ..prompts import ProgramWithScore
from ..utils import ResilientProcessPool
from .proxy_benchmark import build_problem_score_matrix, select_inputs_by_index, select_proxy_problem_subset

logger = logging.getLogger(__name__)


class Diversifier:
    """Handles init phase: diverse seed generation and archive population."""

    def __init__(
        self,
        config: LeviConfig,
        executor: ResilientProcessPool,
        artifact_adapter: ArtifactAdapter,
        state: PipelineState,
    ):
        self.config = config
        self.executor = executor
        self.artifact_adapter = artifact_adapter
        self.state = state
        self.total_cost = 0.0
        self.best_score = 0.0

    async def _acompletion(self, *, model, prompt, **kwargs):
        """Route completion through state for budget tracking."""
        return await self.state.acompletion(model, prompt=prompt, **kwargs)

    def _record_score(self, score: float, sampler: str) -> None:
        """Record a score from the init phase."""
        self.state.record_accept()
        self.state.record_score(
            score=score,
            accepted=True,
            sampler=sampler,
            archive_size=0,
            cell_index=None,
        )
        self.best_score = self.state.best_score_so_far

    async def _cascade_eval(self, content: str) -> dict:
        """Evaluate with cascade: quick test first, full eval only if promising."""
        # Caller already reserved the eval slot.
        if self._should_bypass_cascade_for_proxy_discovery():
            result = await self.artifact_adapter.evaluate(self.executor, content)
            if "error" not in result:
                score = result.get("score", 0)
                self._record_score(score, sampler="init_variant")
            return result

        cascade = self.config.cascade
        if cascade.enabled and cascade.quick_inputs:
            quick_result = await self.artifact_adapter.evaluate(
                self.executor,
                content,
                inputs=cascade.quick_inputs,
                timeout=cascade.quick_timeout,
            )
            if "error" in quick_result:
                return quick_result
            quick_score = quick_result.get("score", 0)
            threshold = self.best_score * cascade.min_score_ratio
            if quick_score < threshold:
                return {"error": f"Cascade rejected: {quick_score:.17g} < {threshold:.17g}"}

        result = await self.artifact_adapter.evaluate(self.executor, content)
        if "error" not in result:
            score = result.get("score", 0)
            self._record_score(score, sampler="init_variant")
        return result

    async def run(
        self,
        pool: CVTMAPElitesPool,
        seed_program: str | None,
        seed_result: dict | None,
        extractor: BehaviorExtractor,
    ) -> tuple[float, list[ScoreHistoryEntry]]:
        """
        Run init phase: generate diverse seeds, expand with variants, build centroids.

        Returns (total_cost, score_history) from the init phase.
        """
        if seed_result is not None:
            seed_score = seed_result.get("score", 0.0)
            self.best_score = seed_score
            # Record the seed itself
            if self.state.eval_count == 0:
                self._record_score(seed_score, sampler="init_seed")

        # Phase 1: Generate diverse seeds
        diverse_seeds = await self._generate_diverse_seeds(seed_program, seed_result)

        # Phase 2: Generate variants
        valid_programs, behavior_vectors = await self._generate_variants(diverse_seeds, extractor)

        # Phase 2.5: Learn a proxy benchmark from init-stage prompt evaluations.
        behavior_vectors = self._maybe_configure_proxy_benchmark(valid_programs, extractor, behavior_vectors)

        # Phase 3: Build centroids and populate archive
        await self._populate_archive(
            pool,
            valid_programs,
            behavior_vectors,
            seed_program,
            seed_result,  # Always pass seed as fallback
        )

        return self.total_cost, list(self.state.score_history)

    async def _generate_diverse_seeds(
        self,
        seed_program: str | None,
        seed_result: dict | None,
    ) -> list[tuple[str, float, dict]]:
        """Phase 1: Generate diverse seeds sequentially with context accumulation."""
        n_seeds = self.config.init.n_diverse_seeds
        model = self.config.init.diversity_model or "gpt-4"

        # Store (code, score, full_result) tuples - include full result to avoid re-evaluation
        diverse_seeds: list[tuple[str, float, dict]] = []
        if seed_program is not None and seed_result is not None:
            diverse_seeds.append((seed_program, seed_result.get("score", 0.0), seed_result))
        else:
            # Generate more seeds to compensate for missing seed program
            n_seeds += 1

        logger.info(f"[Init Phase 1] Generating {n_seeds} diverse seeds with {client_name(model)}")

        max_retries = 3
        for i in range(n_seeds):
            if self.state.budget_exhausted:
                logger.info("[Init Phase 1] Stopping seed generation (budget exhausted)")
                break

            success = False
            for attempt in range(max_retries):
                if self.state.budget_exhausted:
                    break

                attempt_label = f"[Seed {i + 1}]" if attempt == 0 else f"[Seed {i + 1}, retry {attempt}]"

                prompt = self.artifact_adapter.build_diversity_prompt(
                    [(content, score) for content, score, _ in diverse_seeds]
                )

                try:
                    llm_kwargs = {
                        "max_tokens": 16384,
                        "timeout": 300,
                        **self.config.init.diversity_llm_kwargs,
                    }
                    response = await self._acompletion(
                        model=model,
                        prompt=[{"role": "user", "content": prompt}],
                        temperature=self.config.init.temperature,
                        **llm_kwargs,
                    )
                    content = response.text
                    self.total_cost += response.cost

                    new_content = self.artifact_adapter.extract_candidate(content)
                    if not new_content:
                        logger.info(f"  {attempt_label} Failed to extract code")
                        continue

                    if not await self.state.try_start_evaluation():
                        logger.info("[Init Phase 1] Stopping seed evaluation (budget exhausted)")
                        break

                    try:
                        result = await self.artifact_adapter.evaluate(self.executor, new_content)

                        if "error" not in result:
                            new_score = result.get("score", 0.0)
                            self._record_score(new_score, sampler="init_diversity")
                            diverse_seeds.append((new_content, new_score, result))
                            logger.info(f"  {attempt_label} OK - score: {new_score:.17g}")
                            success = True
                        else:
                            self.state.record_error(str(result.get("error", "unknown error")))
                            logger.info(f"  {attempt_label} Eval failed: {result['error'][:50]}")
                    finally:
                        await self.state.finish_evaluation()
                except BudgetLimitReached:
                    logger.info("[Init Phase 1] Stopping seed generation (budget exhausted)")
                    break
                except Exception as e:
                    logger.warning(f"  {attempt_label} Error: {e}")

                if success:
                    break

        logger.info(f"[Init Phase 1] Generated {len(diverse_seeds)} seeds")
        return diverse_seeds

    async def _generate_variants(
        self,
        diverse_seeds: list[tuple[str, float, dict]],
        extractor: BehaviorExtractor,
    ) -> tuple[list[dict], list[np.ndarray]]:
        """Phase 2: Generate variants in parallel using light model(s)."""
        n_variants_per_seed = self.config.init.n_variants_per_seed
        models = self.config.init.variant_models or ["gpt-4o-mini"]
        n_variants = n_variants_per_seed * len(diverse_seeds)

        logger.info(
            f"[Init Phase 2] Generating {n_variants} variants with {[client_name(model) for model in models]}"
        )

        # Build all seeds as ProgramWithScore for sampling
        all_parents = []
        for seed_code, seed_score, _ in diverse_seeds:
            prog = self.artifact_adapter.make_program(seed_code, metadata={"score": seed_score})
            eval_res = EvaluationResult(
                scores={"score": seed_score},
                is_valid=True,
            )
            all_parents.append(ProgramWithScore(prog, eval_res))

        # Build prompts - each variant randomly samples 2 seeds as inspirations
        prompts = []
        n_inspirations = min(2, len(all_parents))  # Sample 2 seeds as inspirations

        for seed_idx, (seed_code, seed_score, _) in enumerate(diverse_seeds):
            for _ in range(n_variants_per_seed):
                # Randomly sample inspirations from all seeds
                inspirations = random.sample(all_parents, n_inspirations)
                prompts.append(self.artifact_adapter.build_init_variant_prompt(inspirations))

        # Parallel LLM calls (cycle through models)
        async def generate_variant(idx: int, prompt: str, model: str) -> dict:
            try:
                response = await self._acompletion(
                    model=model,
                    prompt=[{"role": "user", "content": prompt}],
                    temperature=self.config.init.temperature,
                    max_tokens=4096,
                    timeout=300,
                )
                return {"idx": idx, "content": response.text, "cost": response.cost}
            except BudgetLimitReached:
                return {"idx": idx, "error": "Budget exhausted"}
            except Exception as e:
                return {"idx": idx, "error": str(e)}

        tasks = [generate_variant(i, prompts[i], models[i % len(models)]) for i in range(n_variants)]
        results = await asyncio.gather(*tasks)

        # Extract code from responses
        candidates = []
        for res in results:
            if "error" in res:
                continue
            self.total_cost += res["cost"]
            candidate_content = self.artifact_adapter.extract_candidate(res["content"])
            if candidate_content:
                candidates.append({"idx": res["idx"], "content": candidate_content})

        total_candidates = len(candidates)
        logger.info(f"[Init Phase 2] Evaluating {total_candidates} candidates")

        async def eval_candidate(cand: dict) -> tuple[int, dict]:
            if not await self.state.try_start_evaluation():
                return cand["idx"], {
                    "content": cand["content"],
                    "result": {"error": "Budget exhausted", "budget_stop": True},
                }
            try:
                result = await self._cascade_eval(cand["content"])
                if "error" in result:
                    error_msg = str(result.get("error", "unknown error"))
                    if error_msg.startswith("Cascade rejected:"):
                        self.state.record_reject()
                    elif error_msg != "Budget exhausted":
                        self.state.record_error(error_msg)
                return cand["idx"], {"content": cand["content"], "result": result}
            except Exception as e:
                self.state.record_error(str(e))
                return cand["idx"], {"content": cand["content"], "result": {"error": str(e)}}
            finally:
                await self.state.finish_evaluation()

        eval_tasks = [eval_candidate(c) for c in candidates]
        eval_results = await asyncio.gather(*eval_tasks)
        logger.info(f"[Init Phase 2] Finished evaluating {total_candidates} candidates")

        # Collect valid programs and behavior vectors
        valid_programs = []
        behavior_vectors = []
        failure_errors = []  # Track unique failure reasons

        next_init_order = len(diverse_seeds)
        for idx, data in sorted(eval_results, key=lambda item: item[0]):
            result = data["result"]
            if "error" in result:
                if result.get("budget_stop"):
                    continue
                # Collect unique failure errors for logging
                error_msg = result["error"]
                if len(failure_errors) < 5 and error_msg not in [e for e, _ in failure_errors]:
                    failure_errors.append((error_msg, data["content"][:100]))
                continue
            score = result.get("score", 0.0)
            program = self.artifact_adapter.make_program(data["content"], metadata={"primary_score": score})
            behavior = extractor.extract(program, result)
            valid_programs.append(
                {
                    "content": data["content"],
                    "score": score,
                    "result": result,
                    "behavior": behavior,
                    "init_order": next_init_order + idx,
                    "init_source": "variant",
                }
            )
            behavior_vectors.append(np.array([behavior[f] for f in extractor.features]))

        # Add diverse seeds directly (already evaluated in Phase 1, no need to re-evaluate)
        seed_programs = []
        seed_behavior_vectors = []
        for init_order, (seed_code, seed_score, seed_result) in enumerate(diverse_seeds):
            program = self.artifact_adapter.make_program(seed_code, metadata={"primary_score": seed_score})
            behavior = extractor.extract(program, seed_result)
            seed_programs.append(
                {
                    "content": seed_code,
                    "score": seed_score,
                    "result": seed_result,
                    "behavior": behavior,
                    "init_order": init_order,
                    "init_source": "seed",
                }
            )
            seed_behavior_vectors.append(np.array([behavior[f] for f in extractor.features]))

        valid_programs = seed_programs + valid_programs
        behavior_vectors = seed_behavior_vectors + behavior_vectors

        logger.info(f"[Init Phase 2] Valid programs: {len(valid_programs)}")

        # Log sample of failure reasons if there were failures
        if failure_errors:
            logger.info(f"[Init Phase 2] Sample failures ({len(failure_errors)} unique errors shown):")
            for error_msg, code_preview in failure_errors:
                logger.info(f"  - {error_msg[:100]}")
                logger.info(f"    Content: {code_preview}...")
        return valid_programs, behavior_vectors

    def _should_bypass_cascade_for_proxy_discovery(self) -> bool:
        proxy_cfg = self.config.proxy_benchmark
        return (
            proxy_cfg.enabled
            and self.artifact_adapter.artifact_type == "prompt"
            and bool(proxy_cfg.discovery_inputs)
            and not proxy_cfg.selected_indices
        )

    def _maybe_configure_proxy_benchmark(
        self,
        valid_programs: list[dict],
        extractor: BehaviorExtractor,
        behavior_vectors: list[np.ndarray],
    ) -> list[np.ndarray]:
        proxy_cfg = self.config.proxy_benchmark
        if not proxy_cfg.enabled:
            return behavior_vectors
        if self.artifact_adapter.artifact_type != "prompt":
            logger.info("[Init Proxy] Skipping proxy benchmark selection for non-prompt artifacts")
            return behavior_vectors
        if not proxy_cfg.discovery_inputs:
            logger.info("[Init Proxy] Skipping proxy benchmark selection without discovery inputs")
            return behavior_vectors

        discovery_programs = self._select_proxy_candidates(valid_programs, proxy_cfg.matrix_key)
        if len(discovery_programs) < 2:
            logger.info("[Init Proxy] Skipping proxy benchmark selection (need at least 2 init outputs with per-problem scores)")
            return behavior_vectors

        try:
            score_matrix = build_problem_score_matrix(
                [program["result"] for program in discovery_programs],
                key=proxy_cfg.matrix_key,
            )
        except ValueError as error:
            logger.info(f"[Init Proxy] Skipping proxy benchmark selection: {error}")
            return behavior_vectors

        if score_matrix.shape[1] != len(proxy_cfg.discovery_inputs):
            logger.info(
                "[Init Proxy] Skipping proxy benchmark selection: matrix width %s does not match discovery pool %s",
                score_matrix.shape[1],
                len(proxy_cfg.discovery_inputs),
            )
            return behavior_vectors

        if proxy_cfg.debug_logging:
            self._log_proxy_discovery_matrix(discovery_programs, score_matrix)
        selection = select_proxy_problem_subset(
            score_matrix,
            proxy_cfg.subset_size,
            trace_top_k=proxy_cfg.debug_top_k,
        )
        proxy_cfg.selected_indices = list(selection.selected_indices)
        self.config.inputs = select_inputs_by_index(proxy_cfg.discovery_inputs, proxy_cfg.selected_indices)
        refreshed_behavior_vectors = self._apply_proxy_scores(valid_programs, extractor)
        self._write_proxy_benchmark_artifact(discovery_programs, score_matrix, selection)
        if proxy_cfg.debug_logging:
            self._log_proxy_selection(selection)
            self._log_proxy_prompt_scores(valid_programs)

        logger.info(
            "[Init Proxy] Selected %d representative problems from %d init outputs; future evals now use the proxy subset",
            len(proxy_cfg.selected_indices),
            len(discovery_programs),
        )
        return refreshed_behavior_vectors

    def _select_proxy_candidates(self, valid_programs: list[dict], matrix_key: str) -> list[dict]:
        ordered = sorted(valid_programs, key=lambda item: (item.get("init_order", 10**9), -item.get("score", 0.0)))
        selected: list[dict] = []
        seen_contents: set[str] = set()

        for program in ordered:
            content = program["content"]
            if content in seen_contents:
                continue
            per_problem_scores = program.get("result", {}).get(matrix_key)
            if not isinstance(per_problem_scores, (list, tuple)):
                continue
            selected.append(program)
            seen_contents.add(content)
        return selected

    def _apply_proxy_scores(self, valid_programs: list[dict], extractor: BehaviorExtractor) -> list[np.ndarray]:
        selected_indices = self.config.proxy_benchmark.selected_indices
        matrix_key = self.config.proxy_benchmark.matrix_key
        behavior_vectors: list[np.ndarray] = []

        for program in valid_programs:
            result = dict(program["result"])
            per_problem_scores = result.get(matrix_key)
            if isinstance(per_problem_scores, (list, tuple)) and selected_indices:
                proxy_values = [float(per_problem_scores[idx]) for idx in selected_indices]
                result["full_benchmark_score"] = float(result.get("score", 0.0))
                result["score"] = sum(proxy_values) / len(proxy_values)
                program["score"] = result["score"]
            program["result"] = result

            artifact = self.artifact_adapter.make_program(
                program["content"],
                metadata={"primary_score": program["score"]},
            )
            behavior = extractor.extract(artifact, result)
            program["behavior"] = behavior
            behavior_vectors.append(np.array([behavior[f] for f in extractor.features]))

        return behavior_vectors

    def _write_proxy_benchmark_artifact(
        self,
        discovery_programs: list[dict],
        score_matrix: np.ndarray,
        selection,
    ) -> None:
        output_dir = self.config.output_dir
        if not output_dir:
            return

        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        artifact_path = path / "proxy_benchmark.json"
        problems = []
        selected_indices = set(selection.selected_indices)
        for idx, item in enumerate(self.config.proxy_benchmark.discovery_inputs):
            problems.append(
                {
                    "index": idx,
                    "id": self._problem_identifier(item, idx),
                    "selected": idx in selected_indices,
                }
            )

        prompts = []
        matrix_key = self.config.proxy_benchmark.matrix_key
        for program, row in zip(discovery_programs, score_matrix.tolist()):
            result = program.get("result", {})
            prompts.append(
                {
                    "init_order": int(program.get("init_order", -1)),
                    "init_source": program.get("init_source", "unknown"),
                    "proxy_score": float(program.get("score", 0.0)),
                    "full_score": float(result.get("full_benchmark_score", program.get("score", 0.0))),
                    "content": program["content"],
                    matrix_key: row,
                }
            )

        artifact = {
            "matrix_key": matrix_key,
            "prompt_count": len(prompts),
            "problem_count": int(score_matrix.shape[1]),
            **selection.as_dict(),
            "problems": problems,
            "prompts": prompts,
        }
        artifact_path.write_text(json.dumps(artifact, indent=2))

    def _problem_identifier(self, item: object, fallback_index: int) -> str:
        if isinstance(item, dict):
            for key in ("id", "problem_id", "name"):
                value = item.get(key)
                if value is not None:
                    return str(value)
            problem = item.get("problem")
            if isinstance(problem, str) and problem:
                return problem[:80]
        return f"problem_{fallback_index}"

    def _log_proxy_discovery_matrix(self, discovery_programs: list[dict], score_matrix: np.ndarray) -> None:
        logger.info(
            "[Init Proxy] Discovery matrix shape: prompts=%d problems=%d metric=%s",
            score_matrix.shape[0],
            score_matrix.shape[1],
            self.config.proxy_benchmark.matrix_key,
        )
        for row_idx, program in enumerate(discovery_programs):
            logger.info(
                "[Init Proxy] Row %02d | init_order=%s source=%s full_score=%.4f preview=%s",
                row_idx,
                program.get("init_order", -1),
                program.get("init_source", "unknown"),
                float(program.get("score", 0.0)),
                self._content_preview(program["content"]),
            )

    def _log_proxy_selection(self, selection) -> None:
        logger.info(
            "[Init Proxy] Final selected indices: %s | final ranking agreement=%.4f",
            selection.selected_indices,
            selection.final_ranking_score,
        )
        for step in selection.step_traces:
            chosen_id = self._problem_identifier(
                self.config.proxy_benchmark.discovery_inputs[step.chosen.problem_index],
                step.chosen.problem_index,
            )
            logger.info(
                "[Init Proxy] Step %02d chose #%d (%s) | obj=%.4f sep=%.4f rank=%.4f red=%.4f",
                step.step_number,
                step.chosen.problem_index,
                chosen_id,
                step.chosen.objective,
                step.chosen.separation_score,
                step.chosen.ranking_score,
                step.chosen.redundancy_penalty,
            )
            for candidate in step.top_candidates:
                candidate_id = self._problem_identifier(
                    self.config.proxy_benchmark.discovery_inputs[candidate.problem_index],
                    candidate.problem_index,
                )
                logger.info(
                    "[Init Proxy]   cand #%d (%s) | obj=%.4f sep=%.4f rank=%.4f red=%.4f",
                    candidate.problem_index,
                    candidate_id,
                    candidate.objective,
                    candidate.separation_score,
                    candidate.ranking_score,
                    candidate.redundancy_penalty,
                )

    def _log_proxy_prompt_scores(self, valid_programs: list[dict]) -> None:
        ordered = sorted(valid_programs, key=lambda item: (item.get("init_order", 10**9), item.get("init_source", "")))
        for program in ordered:
            result = program.get("result", {})
            logger.info(
                "[Init Proxy] Prompt init_order=%s source=%s | full_score=%.4f proxy_score=%.4f",
                program.get("init_order", -1),
                program.get("init_source", "unknown"),
                float(result.get("full_benchmark_score", program.get("score", 0.0))),
                float(program.get("score", 0.0)),
            )

    def _content_preview(self, content: str, limit: int = 80) -> str:
        compact = " ".join(content.strip().split())
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3] + "..."

    async def _backfill_quick_scores(self, programs: list[dict]) -> None:
        """Compute quick_scores for programs that don't have one yet.

        Modifies each program's 'result' dict in-place, adding 'quick_score'.
        Programs that already have a quick_score or where quick eval fails are skipped.
        """
        cascade = self.config.cascade
        if not (cascade.enabled and cascade.quick_inputs):
            return

        tasks = []
        indices = []
        for i, prog in enumerate(programs):
            if "quick_score" not in prog["result"]:
                tasks.append(
                    self.artifact_adapter.evaluate(
                        self.executor,
                        prog["content"],
                        inputs=cascade.quick_inputs,
                        timeout=cascade.quick_timeout,
                    )
                )
                indices.append(i)

        if not tasks:
            return

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, qr in zip(indices, results):
            if isinstance(qr, Exception) or "error" in qr:
                continue
            programs[i]["result"]["quick_score"] = qr.get("score", 0)

        logger.info(f"[Init Phase 3] Computed quick_scores for {len(tasks)} archive seeds")

    async def _populate_archive(
        self,
        pool: CVTMAPElitesPool,
        valid_programs: list[dict],
        behavior_vectors: list[np.ndarray],
        seed_program: str = None,
        seed_result: dict = None,
    ) -> None:
        """Phase 3: Build centroids and populate archive using k-means labels directly."""
        if not behavior_vectors:
            logger.warning("[Init Phase 3] No valid programs to build centroids")
            if seed_program and seed_result and "error" not in seed_result:
                fallback = [{"content": seed_program, "result": {**seed_result}}]
                await self._backfill_quick_scores(fallback)

                program = self.artifact_adapter.make_program(
                    seed_program,
                    metadata={"primary_score": seed_result.get("score", 0.0)},
                )
                eval_result = EvaluationResult(
                    scores=fallback[0]["result"],
                    is_valid=True,
                )
                accepted, _ = pool.add(program, eval_result)
                logger.info(
                    f"[Init Phase 3] Added seed program as fallback (accepted={accepted}), archive size: {pool.size()}"
                )
            return

        # Build centroids from init-generated programs (seed + diverse seeds + variants).
        n_centroids, labels = pool.set_centroids_from_data(
            behavior_vectors,
            n_centroids=self.config.cvt.n_centroids,
        )
        logger.info(f"[Init Phase 3] Built {n_centroids} centroids")

        # Group programs by their assigned cell (from k-means labels)
        cell_to_programs: dict[int, list[dict]] = {}
        for idx, prog in enumerate(valid_programs):
            cell = int(labels[idx])
            if cell not in cell_to_programs:
                cell_to_programs[cell] = []
            cell_to_programs[cell].append(prog)

        # Select best program per cell and backfill quick_scores for cascade
        best_per_cell = {cell_idx: max(progs, key=lambda p: p["score"]) for cell_idx, progs in cell_to_programs.items()}
        await self._backfill_quick_scores(list(best_per_cell.values()))

        # For each cell, add the best-scoring program directly (no re-extraction)
        n_accepted = 0
        for cell_idx, best_prog in best_per_cell.items():
            program = self.artifact_adapter.make_program(
                best_prog["content"],
                metadata={"primary_score": best_prog["score"]},
            )
            eval_result = EvaluationResult(
                scores=best_prog["result"],
                is_valid=True,
            )
            if pool.add_at_cell(cell_idx, program, eval_result, best_prog["behavior"]):
                n_accepted += 1

        best_score = max(p["score"] for p in valid_programs) if valid_programs else 0.0
        logger.info(
            f"[Init Phase 3] Done: {n_accepted} cells filled, archive size: {pool.size()}, best: {best_score:.17g}, cost: ${self.total_cost:.3f}"
        )
