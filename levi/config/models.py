from collections.abc import Callable
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class SamplerModelPair(BaseModel):
    sampler: str
    model: str
    weight: float = 1.0
    temperature: Optional[float] = None  # For softmax sampler
    n_cycles: Optional[int] = None  # For cyclic_annealing sampler

    @field_validator("weight")
    @classmethod
    def weight_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("weight must be positive")
        return v


class BudgetConfig(BaseModel):
    dollars: Optional[float] = None
    evaluations: Optional[int] = None
    seconds: Optional[float] = None
    target_score: Optional[float] = None


class CVTConfig(BaseModel):
    n_centroids: int = 50
    data_driven_centroids: bool = True


class InitConfig(BaseModel):
    enabled: bool = True
    n_diverse_seeds: int = 4
    n_variants_per_seed: int = 20
    diversity_model: Optional[str] = None
    variant_models: Optional[list[str]] = None
    temperature: Optional[float] = None
    diversity_prompt: Optional[str] = None  # Custom prompt for diverse seed generation
    diversity_llm_kwargs: dict = Field(
        default_factory=dict
    )  # Extra kwargs passed to diversity LLM calls (e.g. reasoning_effort, max_tokens)


class MetaAdviceConfig(BaseModel):
    enabled: bool = True
    interval: int = 50
    model: Optional[str] = None
    max_tokens: int = 400
    temperature: Optional[float] = None


class BehaviorConfig(BaseModel):
    ast_features: list[str] = Field(default=["loop_count", "branch_count", "math_operators", "loop_nesting_max"])
    score_keys: list[str] = Field(default_factory=list)
    init_noise: float = 0.0

    # Custom extractors: Callable[[Program], float]. Unlike built-in AST extractors,
    # these receive only the Program, making them usable for non-code content types.
    custom_extractors: dict[str, Callable] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}


class CascadeConfig(BaseModel):
    enabled: bool = True
    quick_inputs: list[Any] = Field(default_factory=list)
    min_score_ratio: float = 0.8
    quick_timeout: float = 30.0


class PunctuatedEquilibriumConfig(BaseModel):
    """Configuration for Punctuated Equilibrium feature.

    Periodically triggers paradigm-shift generation using heavy model(s),
    creating fundamentally new solutions to escape local optima.
    """

    enabled: bool = True
    interval: int = 10
    n_clusters: int = 3
    n_variants: int = 3
    heavy_models: Optional[list[str]] = None
    variant_models: Optional[list[str]] = None
    behavior_noise: float = 0.0
    temperature: Optional[float] = None
    reasoning_effort: Optional[str] = None


class PromptOptConfig(BaseModel):
    enabled: bool = False
    teacher_model: Optional[str] = None  # Model for MIPROv2 instruction proposals; None = paradigm_models[0]
    n_trials: int = 12
    num_candidates: int = 4
    num_threads: int = 4
    init_temperature: float = 1.2
    optimize_mutation: bool = True
    optimize_paradigm_shift: bool = True  # Only runs if PE is enabled
    cache_dir: Optional[str] = None  # None = output_dir or cwd
    force: bool = False  # Re-optimize even if cached


class PipelineConfig(BaseModel):
    n_llm_workers: int = 4
    n_eval_processes: int = 4
    eval_timeout: float = 60.0
    temperature: Optional[float] = None
    max_tokens: int = 16384
    n_parents: int = 1
    n_inspirations: int = 1
    output_mode: str = "full"


class LeviConfig(BaseModel):
    # Required
    problem_description: str
    function_signature: str
    seed_program: str | None = None
    inputs: Optional[list[Any]] = None
    score_fn: Callable[..., dict]
    budget: BudgetConfig

    # Core model config
    paradigm_models: str | list[str] = "openai/gpt-4o"
    mutation_models: str | list[str] = "openai/gpt-4o-mini"
    local_endpoints: dict[str, str] = Field(default_factory=dict)

    # Optional: for cost tracking on custom/local models (auto-registers with litellm).
    model_info: dict[str, dict] = Field(default_factory=dict)

    # Auto-generated from mutation_models if not provided.
    # Pass explicitly to override (e.g. for custom sampler/temperature combos).
    sampler_model_pairs: list[SamplerModelPair] = Field(default_factory=list)

    # Optional with defaults
    cvt: CVTConfig = Field(default_factory=CVTConfig)
    init: InitConfig = Field(default_factory=InitConfig)
    meta_advice: MetaAdviceConfig = Field(default_factory=MetaAdviceConfig)
    behavior: BehaviorConfig = Field(default_factory=BehaviorConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    cascade: CascadeConfig = Field(default_factory=CascadeConfig)
    punctuated_equilibrium: PunctuatedEquilibriumConfig = Field(default_factory=PunctuatedEquilibriumConfig)
    prompt_opt: PromptOptConfig = Field(default_factory=PromptOptConfig)

    output_dir: Optional[str] = None  # Directory for snapshots

    # Prompt overrides from DSPy optimization
    prompt_overrides: dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def _auto_wire_models(self) -> "LeviConfig":
        # 1. Coerce str → list[str]
        if isinstance(self.paradigm_models, str):
            self.paradigm_models = [self.paradigm_models]
        if isinstance(self.mutation_models, str):
            self.mutation_models = [self.mutation_models]

        # 2. Auto-generate sampler_model_pairs if not provided
        if not self.sampler_model_pairs:
            pairs = []
            for model in self.mutation_models:
                for temp in [0.3, 0.7, 1.0, 1.2]:
                    pairs.append(
                        SamplerModelPair(
                            sampler="softmax",
                            model=model,
                            weight=1.0,
                            temperature=temp,
                        )
                    )
            self.sampler_model_pairs = pairs

        if not self.sampler_model_pairs:
            raise ValueError(
                "must have at least one sampler_model_pair (provide mutation_models or sampler_model_pairs)"
            )

        # 3. Auto-fill None model fields in sub-configs
        if self.init.diversity_model is None:
            self.init.diversity_model = self.paradigm_models[0]
        if self.init.variant_models is None:
            self.init.variant_models = list(self.mutation_models)

        if self.meta_advice.model is None:
            self.meta_advice.model = self.mutation_models[0]

        if self.punctuated_equilibrium.heavy_models is None:
            self.punctuated_equilibrium.heavy_models = list(self.paradigm_models)
        if self.punctuated_equilibrium.variant_models is None:
            self.punctuated_equilibrium.variant_models = list(self.mutation_models)

        if self.prompt_opt.teacher_model is None:
            self.prompt_opt.teacher_model = self.paradigm_models[0]
        if self.prompt_opt.cache_dir is None and self.output_dir:
            self.prompt_opt.cache_dir = self.output_dir

        # 4. Auto-generate output_dir if not set
        if self.output_dir is None:
            self.output_dir = f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        return self


class LeviResult(BaseModel):
    best_program: str
    best_score: float
    total_evaluations: int
    total_cost: float
    archive_size: int
    runtime_seconds: float
    score_history: Optional[list[float]] = None

    model_config = {"arbitrary_types_allowed": True}
