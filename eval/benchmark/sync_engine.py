"""SyncEngine — AsyncEngine subclass for synchronous tool execution.

The only behavioral difference from AsyncEngine:
  fire_tool_async() runs the tool inline (blocking) and returns the real
  result string immediately, instead of spawning a background thread and
  returning a job JSON.

Additional differences for a fair comparison:
  - System prompt: uses only the use_case system prompt (no async mechanics).
  - Tool list: await_job is excluded (there are no jobs to await).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.engine import AsyncEngine
from core.schema import UseCase


class SyncEngine(AsyncEngine):
    """Synchronous variant of AsyncEngine for benchmarking."""

    def __init__(self, use_case: UseCase):
        # Parent init uses "tool" mode to avoid a KeyError on BASE_SYSTEM_PROMPTS.
        # We override the system prompt and tools immediately after.
        super().__init__(use_case, injection_mode="tool")

        # No async mechanics in the system prompt — the model won't see job IDs.
        self._system_prompt = use_case.system_prompt
        self.messages[0] = {"role": "system", "content": self._system_prompt}

        # Remove await_job from the tool list — irrelevant for synchronous calls.
        self._tools_schema = list(use_case.tool_schemas)

    def fire_tool_async(self, tool_name: str, tool_args: dict) -> str:
        """Run tool synchronously; return real result, never a job JSON."""
        tool_fn = self.use_case.tool_functions[tool_name]
        return tool_fn(tool_args)
