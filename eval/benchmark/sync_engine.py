"""SyncEngine — AsyncEngine subclass for synchronous tool execution.

The only behavioral difference from AsyncEngine:
  fire_tool_async() runs the tool inline (blocking) and returns the real
  result string immediately, instead of spawning a background thread and
  returning a job JSON.

Additional differences for a fair comparison:
  - System prompt: _SYNC_BASE_PROMPT + use_case.system_prompt.
    The base prompt matches the synthesis/formatting instructions in
    BASE_SYSTEM_PROMPTS but omits all async job-ID mechanics (job_id JSON,
    await_job, "(System) Job X completed" parsing).  This ensures any
    formatting differences measured in the benchmark reflect injection
    mechanics only — not the presence or absence of synthesis guidance.
  - Tool list: await_job is excluded (there are no jobs to await).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.engine import AsyncEngine
from core.schema import UseCase

# Synthesis and formatting instructions shared with the async base prompts,
# with all async job-ID / await_job mechanics removed.
_SYNC_BASE_PROMPT = """\
When tool results are available, proactively synthesise them with the \
conversation context — do not wait for the user to ask "which is best." \
Filter and rank based on what you know: stated interests, companions, budget \
signals, or other context from earlier in the conversation. Explain briefly \
why the top picks fit their situation. Reserve a full flat list only when you \
have no context to work with.\
"""


class SyncEngine(AsyncEngine):
    """Synchronous variant of AsyncEngine for benchmarking."""

    def __init__(self, use_case: UseCase, model: str = "gpt-4o"):
        # Parent init uses "tool" mode to avoid a KeyError on BASE_SYSTEM_PROMPTS.
        # We override the system prompt and tools immediately after.
        super().__init__(use_case, injection_mode="tool", model=model)

        # Synthesis instructions + use-case prompt, but NO async job-ID mechanics.
        self._system_prompt = _SYNC_BASE_PROMPT + "\n\n---\n\n" + use_case.system_prompt
        self.messages[0] = {"role": "system", "content": self._system_prompt}

        # Remove await_job from the tool list — irrelevant for synchronous calls.
        self._tools_schema = list(use_case.tool_schemas)

    def fire_tool_async(self, tool_name: str, tool_args: dict) -> str:
        """Run tool synchronously; return real result, never a job JSON."""
        tool_fn = self.use_case.tool_functions[tool_name]
        return tool_fn(tool_args)
