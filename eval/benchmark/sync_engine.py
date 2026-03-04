"""Backward-compat shim: SyncEngine is now AsyncEngine(forced_sync=True)."""

from core.engine import AsyncEngine as _AsyncEngine
from core.schema import UseCase as _UseCase


def SyncEngine(use_case: _UseCase, model: str = "gpt-4o") -> _AsyncEngine:
    """Return an AsyncEngine running in forced_sync mode (all tools run inline)."""
    return _AsyncEngine(use_case, forced_sync=True, model=model)


__all__ = ["SyncEngine"]
