"""Compatibility exports for the shared paper-eta implementation."""

from vara import paper_eta as _implementation
from vara.paper_eta import *  # noqa: F401,F403

__all__ = _implementation.__all__


def __getattr__(name: str):
    """Delegate legacy attribute access, including private helper names."""
    return getattr(_implementation, name)


def __dir__() -> list[str]:
    """Expose the implementation namespace when inspecting this compatibility module."""
    return sorted(set(globals()) | set(dir(_implementation)))
