from pydantic import BaseModel, Field, field_validator
from typing import Callable, Optional, Any


class SamplerModelPair(BaseModel):
    sampler: str
    model: str
    weight: float = 1.0
    temperature: Optional[float] = None  # For softmax sampler
    n_cycles: Optional[int] = None  # For cyclic_annealing sampler

    @field_validator('weight')
    @classmethod
    def weight_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError('weight must be positive')
        return v


class BudgetConfig(BaseModel):
    dollars: Optional[float] = None
    evaluations: Optional[int] = None
    seconds: Optional[float] = None


class CVTConfig(BaseModel):
    n_centroids: int = 50
    defer_centroids: bool = True
    predefined_centroids_file: Optional[str] = None  


class InitConfig(BaseModel):
    enabled: bool = True
    n_diverse_seeds: int = 5
    n_variants_per_seed: int = 25
    diversity_model: Optional[str] = None
    variant_models: Optional[list[str]] = None  
    temperature: float = 0.8


class MetaAdviceConfig(BaseModel):
    enabled: bool = True
    interval: int = 50
    model: Optional[str] = None
    max_tokens: int = 400


class BehaviorConfig(BaseModel):
    ast_features: list[str] = Field(
        default=["loop_count", "branch_count", "math_operators", "loop_nesting_max"]
    )
    score_keys: list[str] = Field(default_factory=list)
    init_noise: float = 0.15


class CheckpointConfig(BaseModel):
    enabled: bool = False
    interval_seconds: float = 600.0
    interval_generations: Optional[int] = None
    path: Optional[str] = None


class CascadeConfig(BaseModel):
    enabled: bool = True
    quick_inputs: list[Any] = Field(default_factory=list)
    min_score_ratio: float = 0.8
    quick_timeout: float = 30.0


class PunctuatedEquilibriumConfig(BaseModel):
    """Configuration for Punctuated Equilibrium feature.

    Periodically triggers paradigm-shift generation using a heavy model,
    creating fundamentally new solutions to escape local optima.
    """
    enabled: bool = False
    interval: int = 50
    n_clusters: int = 3  
    n_variants: int = 5  
    heavy_model: Optional[str] = None  
    variant_models: Optional[list[str]] = None  
    behavior_noise: float = 0.05  
    temperature: float = 1.0
    reasoning_effort: Optional[str] = None  


class PipelineConfig(BaseModel):
    n_llm_workers: int = 4
    n_eval_processes: int = 4
    eval_timeout: float = 60.0
    temperature: float = 0.8
    max_tokens: int = 30000
    n_parents: int = 1
    n_inspirations: int = 2
    output_mode: str = "full"  


class AlgoforgeConfig(BaseModel):
    # Required
    problem_description: str
    function_signature: str
    seed_program: str
    inputs: list[Any]
    score_fn: Callable[[Callable, list], dict]
    budget: BudgetConfig
    sampler_model_pairs: list[SamplerModelPair]

    # Optional with defaults
    cvt: CVTConfig = Field(default_factory=CVTConfig)
    init: InitConfig = Field(default_factory=InitConfig)
    meta_advice: MetaAdviceConfig = Field(default_factory=MetaAdviceConfig)
    behavior: BehaviorConfig = Field(default_factory=BehaviorConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    cascade: CascadeConfig = Field(default_factory=CascadeConfig)
    punctuated_equilibrium: PunctuatedEquilibriumConfig = Field(default_factory=PunctuatedEquilibriumConfig)

    n_islands: int = 1
    migration_interval: Optional[int] = None
    output_dir: Optional[str] = None  # Directory for snapshots
    api_bases: dict[str, str] = Field(default_factory=dict)  # Model -> api_base URL mapping

    model_config = {"arbitrary_types_allowed": True}

    @field_validator('sampler_model_pairs')
    @classmethod
    def must_have_at_least_one(cls, v: list[SamplerModelPair]) -> list[SamplerModelPair]:
        if not v:
            raise ValueError('must have at least one sampler_model_pair')
        return v


class AlgoforgeResult(BaseModel):
    best_program: str
    best_score: float
    total_evaluations: int
    total_cost: float
    archive_size: int
    runtime_seconds: float
    per_sampler_stats: Optional[dict] = None
    score_history: Optional[list[float]] = None

    model_config = {"arbitrary_types_allowed": True}
