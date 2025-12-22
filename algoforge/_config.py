"""
Global configuration for AlgoForge.
"""

from dataclasses import dataclass, field
from typing import Optional

from .llm import ModelWeight


@dataclass
class AlgoForgeConfig:
    """Global configuration."""

    # LLM ensemble (weights should sum to ~1.0)
    models: list[ModelWeight] = field(default_factory=lambda: [
        ModelWeight("gpt-4o-mini", 0.8),
        ModelWeight("gpt-4o", 0.2),
    ])

    # Default budget limits
    default_max_evaluations: Optional[int] = None
    default_max_llm_cost: Optional[float] = None
    default_max_wall_time: Optional[float] = None

    # Execution settings
    evaluation_timeout: float = 30.0
    memory_limit_mb: int = 512


_global_config = AlgoForgeConfig()


def configure(
    models: Optional[list[ModelWeight]] = None,
    **kwargs
) -> None:
    """
    Configure global AlgoForge settings.

    Example:
        af.configure(models=[
            ModelWeight('claude-sonnet-4-20250514', 0.8),
            ModelWeight('claude-3-opus-20240229', 0.2),
        ])
    """
    global _global_config

    if models:
        _global_config.models = models

    for key, value in kwargs.items():
        if hasattr(_global_config, key):
            setattr(_global_config, key, value)
        else:
            raise ValueError(f"Unknown configuration key: {key}")


def get_config() -> AlgoForgeConfig:
    """Get current configuration."""
    return _global_config
