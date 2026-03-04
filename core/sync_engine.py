"""SyncEngine — AsyncEngine subclass for synchronous tool execution.

The only behavioral difference from AsyncEngine:
  fire_tool_async() runs the tool inline (blocking) and returns the real
  result string immediately, instead of spawning a background thread and
  returning a job JSON.

Used by:
  - AgentTool: runs a sub-agent to completion and returns its final text
  - eval/benchmark: fair latency comparison against the async engine
"""

from .engine import AsyncEngine
from .schema import UseCase

# Synthesis instructions shared with the async base prompts, with all async
# job-ID / await_job mechanics stripped out. Used when there are no background
# jobs to manage (sub-agents, benchmarks).
_SYNC_BASE_PROMPT = """\
When tool results are available, proactively synthesise them with the \
conversation context — do not wait for the user to ask "which is best." \
Filter and rank based on what you know: stated interests, companions, budget \
signals, or other context from earlier in the conversation. Explain briefly \
why the top picks fit their situation. Reserve a full flat list only when you \
have no context to work with.\
"""


class SyncEngine(AsyncEngine):
    """Synchronous variant of AsyncEngine.

    All tools run inline — there are no background threads, no job IDs,
    and no await_job. Callers block until the full agent loop completes.
    """

    def __init__(self, use_case: UseCase, model: str = "gpt-4o"):
        # Parent init uses "tool" mode to avoid a KeyError on BASE_SYSTEM_PROMPTS.
        # We override the system prompt and tools immediately after.
        super().__init__(use_case, injection_mode="tool", model=model)

        # Synthesis instructions + use-case prompt, but NO async job-ID mechanics.
        self._system_prompt = _SYNC_BASE_PROMPT + "\n\n---\n\n" + use_case.system_prompt
        self.messages[0] = {"role": "system", "content": self._system_prompt}

        # Remove await_job from the tool list — irrelevant without background jobs.
        self._tools_schema = [t.schema for t in use_case.tools]

    def fire_tool_async(self, tool_name: str, tool_args: dict) -> str:
        """Run tool synchronously; return real result, never a job JSON."""
        return self._tool_map[tool_name].fn(tool_args)
