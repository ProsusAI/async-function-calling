"""Re-export SyncEngine from core for backward compatibility with benchmark scripts."""

from core.sync_engine import SyncEngine, _SYNC_BASE_PROMPT  # noqa: F401

__all__ = ["SyncEngine", "_SYNC_BASE_PROMPT"]
