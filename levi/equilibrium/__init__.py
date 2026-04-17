"""Punctuated Equilibrium module for periodic paradigm shifts."""

__all__ = ["PunctuatedEquilibrium"]


def __getattr__(name: str):
    if name == "PunctuatedEquilibrium":
        from .equilibrium import PunctuatedEquilibrium

        return PunctuatedEquilibrium
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
