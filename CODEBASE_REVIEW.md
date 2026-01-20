# AlgoForge Codebase Review Document

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [Data Flow Diagram](#data-flow-diagram)
4. [Module-by-Module Review Guide](#module-by-module-review-guide)
5. [Critical Code Paths](#critical-code-paths)
6. [Potential Review Concerns](#potential-review-concerns)
7. [Suggested Review Order](#suggested-review-order)
8. [Test Coverage Map](#test-coverage-map)
9. [Configuration Surface Area](#configuration-surface-area)
10. [Quick Reference: File Purposes](#quick-reference-file-purposes)

---

## Executive Summary

**AlgoForge** is an LLM-guided evolutionary optimization framework for discovering algorithms. It combines:
- **CVT-MAP-Elites**: Quality-diversity algorithm using Centroidal Voronoi Tessellation
- **Multi-strategy sampling**: UCB, softmax, uniform, and per-subscore samplers
- **Async pipeline**: Producer-consumer pattern with LLM workers and evaluators
- **Behavioral diversity**: AST-based code structure analysis

**Key Design Principle**: Uses deterministic code structure features (via AST analysis) to measure behavioral diversity, eliminating the complexity of multi-island populations.

**Lines of Code**: ~3,500 lines of Python across 25+ files

---

## Architecture Overview

```
                          ┌─────────────────────────────────────┐
                          │         AlgoforgeConfig             │
                          │  (problem, signature, budget, etc.) │
                          └────────────────┬────────────────────┘
                                           │
                          ┌────────────────▼────────────────────┐
                          │          algoforge.run()            │
                          │     (methods/algoforge.py:45)       │
                          └────────────────┬────────────────────┘
                                           │
            ┌──────────────────────────────┼──────────────────────────────┐
            │                              │                              │
            ▼                              ▼                              ▼
┌───────────────────────┐   ┌────────────────────────┐   ┌───────────────────────┐
│   BehaviorExtractor   │   │    CVTMAPElitesPool    │   │     Diversifier       │
│ (behavior/extractor)  │   │  (pool/cvt_map_elites) │   │  (init/diversifier)   │
│  - AST features       │   │  - Archive mgmt        │   │  - Diverse seeds      │
│  - Z-score normalize  │   │  - Multi-sampler       │   │  - Variants gen       │
│  - Sigmoid → [0,1]    │   │  - Cell assignment     │   │  - Centroid init      │
└───────────────────────┘   └────────────────────────┘   └───────────────────────┘
                                           │
                          ┌────────────────▼────────────────────┐
                          │         PipelineRunner              │
                          │      (pipeline/runner.py)           │
                          └────────────────┬────────────────────┘
                                           │
                 ┌─────────────────────────┼─────────────────────────┐
                 │                         │                         │
                 ▼                         ▼                         ▼
      ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
      │   LLM Producers  │     │  Eval Consumers  │     │  PE Monitor      │
      │  (producer.py)   │────▶│  (consumer.py)   │     │  (equilibrium/)  │
      │  - Sample parent │     │  - Subprocess    │     │  - Paradigm shift│
      │  - Build prompt  │     │  - Score fn      │     │  - Variants      │
      │  - Call LLM      │     │  - Update pool   │     │  - Noise inject  │
      └──────────────────┘     └──────────────────┘     └──────────────────┘
                 │                         │
                 │    ┌────────────────────┘
                 ▼    ▼
           ┌──────────────┐
           │  code_queue  │
           │ (asyncio.Q)  │
           └──────────────┘
```

---

## Data Flow Diagram

### Main Evolution Loop

```
1. INITIALIZATION
   ┌─────────────────────────────────────────────────────────────────────┐
   │ Seed Program → Evaluate → BehaviorExtractor.extract() → Pool.add() │
   └─────────────────────────────────────────────────────────────────────┘
                                     │
                         ┌───────────┼───────────┐
                         │ Init Phase (optional) │
                         │ Diversifier.run()     │
                         └───────────┼───────────┘
                                     │
2. EVOLUTION LOOP
   ┌─────────────────────────────────────────────────────────────────────┐
   │                                                                     │
   │  ┌──────────────┐    ┌─────────────┐    ┌──────────────────┐       │
   │  │ Pool.sample  │───▶│ PromptBuild │───▶│ LLM.acompletion  │       │
   │  │ (UCB/softmax │    │ (parents +  │    │ (litellm)        │       │
   │  │  /uniform)   │    │  problem)   │    └────────┬─────────┘       │
   │  └──────────────┘    └─────────────┘             │                 │
   │                                                  ▼                 │
   │                                        ┌─────────────────┐         │
   │                                        │  extract_code   │         │
   │                                        │  (from LLM resp)│         │
   │                                        └────────┬────────┘         │
   │                                                 ▼                  │
   │  ┌──────────────┐    ┌─────────────┐    ┌──────────────────┐       │
   │  │ Pool.add     │◀───│ Behavior    │◀───│ _evaluate_code   │       │
   │  │ (cell assign │    │ Extract     │    │ (subprocess)     │       │
   │  │  + elitism)  │    │             │    └──────────────────┘       │
   │  └──────────────┘    └─────────────┘                               │
   │        │                                                           │
   │        └──────── Pool.update_sampler(success=True/False) ──────────┘
   │
   └─────────────────────────────────────────────────────────────────────┘

3. PERIODIC EVENTS
   - Meta-Advice: Every N evals, analyze errors, inject advice into prompts
   - Punctuated Equilibrium: Cluster archive, generate paradigm shifts
   - Snapshot: Save archive state to JSON
```

---

## Module-by-Module Review Guide

### 1. Core Data Structures (`algoforge/core/`)

| File | Lines | Purpose | Review Focus |
|------|-------|---------|--------------|
| `program.py` | ~30 | Immutable program representation | Frozen dataclass, UUID generation, lineage tracking |
| `evaluation.py` | ~40 | Evaluation result container | Score dict structure, `primary_score` property |
| `types.py` | ~10 | Type aliases | MetricDict, OutputDict definitions |

**Key Questions to Review:**
- Is `Program` truly immutable (frozen=True)?
- Does `EvaluationResult.primary_score` correctly extract from scores dict?
- Are parent references properly maintained for lineage tracking?

### 2. Configuration (`algoforge/config/`)

| File | Lines | Purpose | Review Focus |
|------|-------|---------|--------------|
| `models.py` | ~143 | Pydantic config models | Validation rules, defaults, field relationships |

**Key Questions to Review:**
- Are all required fields properly marked?
- Do validators catch invalid configurations early?
- Are defaults sensible (e.g., n_centroids=50, eval_timeout=300)?
- Is `arbitrary_types_allowed` needed for `score_fn: Callable`?

### 3. Behavior Extraction (`algoforge/behavior/`)

| File | Lines | Purpose | Review Focus |
|------|-------|---------|--------------|
| `extractor.py` | ~168 | Z-score normalization, feature composition | Welford's algorithm correctness, sigmoid mapping |
| `features.py` | ~200 | 14 AST-based feature extractors | AST visitor correctness, edge cases |

**Key Questions to Review:**
- Is Welford's online algorithm implemented correctly? (lines 107-113)
- Is the sigmoid z-score mapping mathematically correct? (line 121-124)
- Do AST features handle malformed code gracefully?
- Is noise only applied during `init` phase?

**Critical Code Path - `extractor.py:126-167`:**
```python
def extract(self, program: Program, eval_result: Optional[dict] = None) -> FeatureVector:
    # 1. Parse AST (fallback to 0.5 on error)
    # 2. Extract raw values via AST visitors
    # 3. Update running stats (Welford)
    # 4. Z-score normalize → sigmoid → [0,1]
    # 5. Add noise during init phase
```

### 4. Archive/Pool (`algoforge/pool/`)

| File | Lines | Purpose | Review Focus |
|------|-------|---------|--------------|
| `cvt_map_elites.py` | ~842 | Main archive with multi-strategy sampling | Sampler implementations, cell assignment, elitism |
| `protocol.py` | ~30 | Sampler protocol, SampleResult | Interface contract |

**Key Questions to Review:**
- Is UCB correctly implemented? (lines 45-56)
- Is weighted sampling without replacement correct? (lines 149-167)
- Does `add()` correctly handle elitism (only accept if better)?
- Is nearest centroid finding efficient? (lines 489-493)
- Is `add_with_behavior_noise()` applying noise correctly? (lines 581-630)

**Critical Code Path - `cvt_map_elites.py:495-515`:**
```python
def add(self, program: Program, evaluation_result: EvaluationResult) -> tuple[bool, int]:
    # 1. Extract behavior
    # 2. Find nearest centroid (cell assignment)
    # 3. Check if cell empty → accept
    # 4. Check if new score > existing → accept
    # 5. Otherwise reject
```

### 5. Pipeline (`algoforge/pipeline/`)

| File | Lines | Purpose | Review Focus |
|------|-------|---------|--------------|
| `runner.py` | ~234 | Orchestration, task management | Async coordination, shutdown handling |
| `producer.py` | ~126 | LLM workers | Prompt building, diff mode, code extraction |
| `consumer.py` | ~287 | Eval workers | Subprocess evaluation, meta-advice, cascade |
| `state.py` | ~162 | Shared state | Budget tracking, score history, thread safety |

**Key Questions to Review:**
- Is `archive_lock` properly used everywhere it should be?
- Does producer correctly handle budget exhaustion mid-flight?
- Is subprocess memory limit (8GB) applied correctly?
- Does cascade evaluation logic correctly skip poor candidates?
- Is meta-advice generation properly async (non-blocking)?

**Critical Code Path - `producer.py:43-125`:**
```python
async def llm_producer(...):
    while not stop_event.is_set():
        # 1. Check budget
        # 2. Sample from pool (with lock)
        # 3. Build prompt
        # 4. Call LLM
        # 5. Extract/apply code
        # 6. Put on queue
```

**Critical Code Path - `consumer.py:83-210`:**
```python
async def eval_consumer(...):
    while not stop_event.is_set() or not code_queue.empty():
        # 1. Get item from queue
        # 2. Cascade eval (if enabled)
        # 3. Full eval
        # 4. Update pool (with lock)
        # 5. Update sampler stats
        # 6. Generate meta-advice (if interval)
        # 7. Save snapshot (if interval)
```

### 6. Initialization (`algoforge/init/`)

| File | Lines | Purpose | Review Focus |
|------|-------|---------|--------------|
| `diversifier.py` | ~519 | Diverse seed generation, centroid building | Prompt for diversity, k-means usage |

**Key Questions to Review:**
- Is `DIVERSITY_SEED_PROMPT` effective at generating diverse solutions?
- Is k-means++ initialization correct?
- Is the farthest-first traversal for diversity selection correct? (pool/cvt_map_elites.py:428-475)
- Are seeds added without re-evaluation (avoiding double work)?

### 7. Punctuated Equilibrium (`algoforge/equilibrium/`)

| File | Lines | Purpose | Review Focus |
|------|-------|---------|--------------|
| `equilibrium.py` | ~410 | Paradigm shift generation | Clustering, representative selection, variant generation |
| `prompts.py` | ~100 | PE-specific prompts | Prompt quality |

**Key Questions to Review:**
- Is cluster representative selection correct (best per cluster)?
- Is behavior noise applied correctly to PE solutions?
- Does variant generation properly handle LLM failures?
- Is cost tracking correct for PE events?

### 8. LLM Interface (`algoforge/llm/`)

| File | Lines | Purpose | Review Focus |
|------|-------|---------|--------------|
| `client.py` | ~50 | LiteLLM wrapper | Error handling, cost tracking |
| `prompts/builder.py` | ~140 | Composable prompt construction | Section ordering, output mode instructions |

**Key Questions to Review:**
- Is the DIFF output mode instruction clear enough for LLMs?
- Does `apply_diff()` in producer.py correctly handle SEARCH/REPLACE blocks?
- Are prompts properly escaped/formatted?

### 9. Utilities (`algoforge/utils/`)

| File | Lines | Purpose | Review Focus |
|------|-------|---------|--------------|
| `code_extraction.py` | ~50 | Extract code from LLM responses | Regex patterns, edge cases |
| `resilient_pool.py` | ~80 | Process pool with timeout | Process lifecycle, memory isolation |
| `ids.py` | ~10 | UUID generation | Uniqueness |

**Key Questions to Review:**
- Does `extract_code()` handle all LLM response formats?
- Is process pool properly cleaning up dead processes?
- Are timeouts correctly enforced?

---

## Critical Code Paths

### Path 1: Seed → Archive (First Program)

```
1. methods/algoforge.py:87-98  → Evaluate seed in subprocess
2. methods/algoforge.py:133-139 → Create Program + EvaluationResult
3. pool/cvt_map_elites.py:495-515 → Add to archive (always accepted, first entry)
```

### Path 2: LLM → Archive (Evolution)

```
1. producer.py:57-61  → Sample from pool (with lock)
2. producer.py:71-80  → Build prompt
3. producer.py:84-92  → Call LLM
4. producer.py:107-119 → Extract code, put on queue
5. consumer.py:98-140 → Evaluate (cascade + full)
6. consumer.py:157-163 → Create Program + EvaluationResult
7. consumer.py:163-169 → Add to pool, update sampler
```

### Path 3: Centroid Initialization

```
1. diversifier.py:210-212 → Generate diverse seeds
2. diversifier.py:297-431 → Generate variants in parallel
3. pool/cvt_map_elites.py:388-425 → K-means clustering on behavior vectors
4. diversifier.py:503-515 → Add best per cell to archive
```

### Path 4: Punctuated Equilibrium Event

```
1. runner.py:165-167 → Check if PE should trigger
2. equilibrium.py:224-226 → Cluster occupied centroids
3. equilibrium.py:235 → Select cluster representatives
4. equilibrium.py:246-275 → Generate paradigm shift (heavy model)
5. equilibrium.py:286-314 → Evaluate and add with behavior noise
6. equilibrium.py:331-403 → Generate and evaluate variants
```

---

## Potential Review Concerns

### 1. Concurrency & Thread Safety

| Location | Concern | What to Check |
|----------|---------|---------------|
| `producer.py:57-61` | Archive lock during sample | Is lock held for minimum time? |
| `consumer.py:148-199` | Archive lock during update | Could deadlock with PE? |
| `equilibrium.py:224-235` | PE holds lock during clustering | Is this blocking too long? |
| `state.py` | Shared state without locks | Are primitives thread-safe in asyncio? |

### 2. Memory Management

| Location | Concern | What to Check |
|----------|---------|---------------|
| `consumer.py:59-61` | 8GB memory limit via setrlimit | Works on all platforms? (comment says may fail) |
| `diversifier.py:64-65` | Same 8GB limit | Duplicate code |
| `equilibrium.py:32-36` | Same 8GB limit | Duplicate code (DRY violation) |

### 3. Error Handling

| Location | Concern | What to Check |
|----------|---------|---------------|
| `producer.py:93-96` | LLM error → sleep 1s → retry | No backoff, could hammer API |
| `consumer.py:141-144` | TimeoutError → record error | Is timeout vs Exception distinction correct? |
| `equilibrium.py:268-270` | PE generation failure → return early | Should retry? |

### 4. Code Execution Safety

| Location | Concern | What to Check |
|----------|---------|---------------|
| `consumer.py:66-67` | `exec(code, namespace)` | Arbitrary code execution (intended, but review isolation) |
| `diversifier.py:69-70` | Same exec pattern | Same concern |

### 5. Numerical Stability

| Location | Concern | What to Check |
|----------|---------|---------------|
| `extractor.py:115-119` | `_get_std()` returns max(sqrt(variance), 0.1) | Why 0.1 minimum? |
| `extractor.py:123` | Sigmoid clamping z to [-10, 10] | Correct for numerical stability |
| `cvt_map_elites.py:145` | Softmax with `exp((ns-1.0)/temperature)` | Subtracting 1.0 for numerical stability |

### 6. State Consistency

| Location | Concern | What to Check |
|----------|---------|---------------|
| `state.py:92-108` | record_accept/reject/error increment eval_count | Is order correct? (increment before or after?) |
| `state.py:143-157` | record_score updates best_score_so_far | Race condition possible? |
| `pool/cvt_map_elites.py:506-514` | `_best_score` update in add() | Thread-safe? |

### 7. Configuration Validation

| Location | Concern | What to Check |
|----------|---------|---------------|
| `models.py:12-14` | weight > 0 validation | Only positive, not zero check |
| `models.py:124-128` | sampler_model_pairs non-empty | Good |
| Various | Missing: n_centroids > 0, eval_timeout > 0 | Add validation? |

---

## Suggested Review Order

### Phase 1: Core Abstractions (30 min)
1. `core/program.py` - Understand the basic unit
2. `core/evaluation.py` - Understand what an evaluation produces
3. `config/models.py` - Understand all configuration options

### Phase 2: Behavioral Diversity (45 min)
4. `behavior/features.py` - Understand AST feature extractors
5. `behavior/extractor.py` - Understand normalization pipeline

### Phase 3: Archive Management (60 min)
6. `pool/protocol.py` - Understand sampler interface
7. `pool/cvt_map_elites.py` - This is the core algorithm
   - Focus on: Sampler classes (lines 59-306)
   - Focus on: CVTMAPElitesPool.add() (lines 495-579)
   - Focus on: CVTMAPElitesPool.sample() (lines 640-669)

### Phase 4: Pipeline (60 min)
8. `pipeline/state.py` - Understand shared state
9. `pipeline/runner.py` - Understand orchestration
10. `pipeline/producer.py` - Understand LLM generation
11. `pipeline/consumer.py` - Understand evaluation loop

### Phase 5: Advanced Features (45 min)
12. `init/diversifier.py` - Understand initialization
13. `equilibrium/equilibrium.py` - Understand PE mechanism
14. `llm/prompts/builder.py` - Understand prompt construction

### Phase 6: Entry Point (15 min)
15. `methods/algoforge.py` - See how it all comes together

### Phase 7: Cross-Cutting Concerns (30 min)
16. Review all `_evaluate_code` functions for consistency
17. Review all uses of `archive_lock` for correctness
18. Review error handling patterns

---

## Test Coverage Map

| Module | Test File | Coverage Notes |
|--------|-----------|----------------|
| `core/` | `tests/test_core.py` | Program, EvaluationResult |
| `pool/` | `tests/test_pool.py` | CVT-MAP-Elites pool |
| `budget/` | `tests/test_budget.py` | Budget manager |
| `llm/prompts/` | `tests/test_prompt_builder.py` | PromptBuilder |
| Integration | `tests/test_alphaevolve.py` | End-to-end tests |
| `behavior/` | **No tests found** | Gap! |
| `pipeline/` | **No tests found** | Gap! |
| `init/` | **No tests found** | Gap! |
| `equilibrium/` | **No tests found** | Gap! |

**Recommendation**: The behavior extraction and pipeline modules are critical but lack dedicated tests.

---

## Configuration Surface Area

### Required Configuration
```python
AlgoforgeConfig(
    problem_description: str,      # Problem statement for LLM
    function_signature: str,       # e.g., "def solve(items: list) -> int"
    seed_program: str,             # Initial solution code
    inputs: list[Any],             # Test inputs for evaluation
    score_fn: Callable,            # fn(candidate_fn, inputs) -> dict with 'score'
    budget: BudgetConfig,          # dollars/evaluations/seconds
    sampler_model_pairs: list,     # At least one required
)
```

### Optional Configuration (with defaults)

| Section | Key Settings | Default | Impact |
|---------|--------------|---------|--------|
| `cvt` | `n_centroids` | 50 | Archive capacity |
| `cvt` | `defer_centroids` | True | Data-driven vs random init |
| `init` | `enabled` | True | Whether to run diverse seeding |
| `init` | `n_diverse_seeds` | 5 | Exploration breadth |
| `init` | `n_variants_per_seed` | 25 | Population size |
| `meta_advice` | `interval` | 50 | How often to analyze errors |
| `pipeline` | `n_llm_workers` | 4 | Parallelism |
| `pipeline` | `n_eval_processes` | 4 | Parallelism |
| `pipeline` | `eval_timeout` | 300s | Max eval time |
| `pipeline` | `output_mode` | "full" | "full" vs "diff" |
| `cascade` | `enabled` | True | Pre-screening |
| `cascade` | `min_score_ratio` | 0.8 | Rejection threshold |
| `punctuated_equilibrium` | `enabled` | False | PE feature |
| `punctuated_equilibrium` | `interval` | 50 | Trigger frequency |

---

## Quick Reference: File Purposes

```
algoforge/
├── __init__.py              # Public API exports
├── methods/
│   └── algoforge.py         # Main entry point: run()
├── core/
│   ├── program.py           # Program dataclass (code + metadata)
│   ├── evaluation.py        # EvaluationResult dataclass
│   └── types.py             # Type aliases
├── config/
│   └── models.py            # All Pydantic config models
├── behavior/
│   ├── extractor.py         # BehaviorExtractor (z-score + sigmoid)
│   └── features.py          # 14 AST feature functions
├── pool/
│   ├── cvt_map_elites.py    # Main archive + samplers (UCB/softmax/uniform)
│   └── protocol.py          # Sampler protocol + SampleResult
├── pipeline/
│   ├── runner.py            # PipelineRunner orchestration
│   ├── producer.py          # LLM worker coroutine
│   ├── consumer.py          # Eval worker coroutine + meta-advice
│   └── state.py             # PipelineState (shared counters)
├── init/
│   └── diversifier.py       # Diverse seed generation + variants
├── equilibrium/
│   ├── equilibrium.py       # PunctuatedEquilibrium class
│   └── prompts.py           # PE-specific prompts
├── llm/
│   ├── client.py            # LLMClient (LiteLLM wrapper)
│   └── prompts/
│       └── builder.py       # PromptBuilder (composable prompts)
├── evaluator/
│   ├── protocol.py          # Evaluator protocol (unused?)
│   └── sandboxed.py         # Sandboxed evaluator (unused?)
├── database/
│   ├── protocol.py          # Database protocol (unused?)
│   └── memory.py            # In-memory database (unused?)
├── budget/
│   ├── manager.py           # Budget tracking
│   └── exceptions.py        # BudgetExhausted exception
├── island/
│   ├── coordinator.py       # Multi-island orchestration
│   ├── runner.py            # Island-specific runner
│   └── diversifier.py       # Island-specific init
└── utils/
    ├── code_extraction.py   # extract_code() from LLM responses
    ├── resilient_pool.py    # ResilientProcessPool
    └── ids.py               # UUID generation

tests/
├── test_core.py             # Program, EvaluationResult tests
├── test_pool.py             # CVT-MAP-Elites tests
├── test_budget.py           # Budget manager tests
├── test_prompt_builder.py   # PromptBuilder tests
└── test_alphaevolve.py      # Integration tests

examples/
├── eplb/                    # Expert Parallelism Load Balancer
├── prism/                   # Prism ML system
├── txn_scheduling/          # Transaction scheduling
├── cloudcast/               # Cloud resource prediction
├── llm_sql/                 # SQL query optimization
└── cant_be_late/            # Time-constrained scheduling
```

---

## Appendix: Code Smells to Watch For

1. **Duplicate `_evaluate_code` functions** - Same function in 3 files:
   - `consumer.py:56-81`
   - `diversifier.py:59-84`
   - `equilibrium.py:29-53`

2. **Hardcoded constants**:
   - `SNAPSHOT_INTERVAL = 10` in consumer.py
   - Memory limit `8 * 1024 * 1024 * 1024` in multiple places
   - Various prompt strings embedded in code

3. **Unused modules?**:
   - `database/` - appears to have protocol but no usage in main flow
   - `evaluator/protocol.py` and `sandboxed.py` - not used in pipeline

4. **Mixed abstraction levels**:
   - `PipelineState` is a dataclass but has methods with side effects
   - Some Pydantic models, some dataclasses, some plain classes

---

## Checklist for Your Review

- [ ] Core data structures are correct and complete
- [ ] Behavior extraction math is correct (Welford, sigmoid)
- [ ] Sampler implementations match their descriptions (UCB, softmax, etc.)
- [ ] Archive elitism is correctly enforced
- [ ] Pipeline concurrency is safe (no deadlocks, proper lock usage)
- [ ] Subprocess evaluation is properly isolated
- [ ] Budget tracking is accurate across all cost sources
- [ ] Error handling is consistent and recoverable
- [ ] Configuration validation catches invalid inputs
- [ ] Meta-advice is generated at correct intervals
- [ ] Punctuated equilibrium triggers and operates correctly
- [ ] Snapshot saves contain all necessary state
- [ ] Tests exist for critical paths (or note gaps)
