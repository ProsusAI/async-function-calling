from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Hooks:
    """
    Optional lifecycle hooks for a UseCase.

    before_tool(tool_name, args) -> dict | None
        Called before every tool invocation. Return a (possibly modified) args
        dict to proceed, or None to cancel the call entirely.

    after_tool(tool_name, args, result) -> str
        Called after every tool completes (sync result or async result before
        injection). Return a (possibly modified) result string.
    """

    before_tool: Callable[[str, dict], "dict | None"] | None = field(default=None)
    after_tool: Callable[[str, dict, str], str] | None = field(default=None)


@dataclass
class Tool:
    """
    A single tool available to an agent.

    The sync/async distinction lives here — not in a separate set.
      is_async=False  → runs inline, result returned immediately to the LLM
      is_async=True   → fires in a background thread (fire-and-forget), result
                        injected back into the conversation when ready
    """

    name: str
    description: str
    parameters: dict           # OpenAI function parameters schema
    fn: Callable[[dict], str]  # takes the parsed args dict, returns a string result
    is_async: bool = False

    @property
    def schema(self) -> dict:
        """OpenAI function call schema for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class UseCase:
    """
    Contract for a use case plugged into AsyncEngine.

    The framework owns:
      - await_job tool schema and handler
      - Injection modes (user / system / tool)
      - SSE broadcast machinery
      - OpenAI API loop
      - BASE_SYSTEM_PROMPT (async mechanics, job-ID rules, synthesis behaviour)

    The use case owns everything in this dataclass.
    """

    display_name: str
    """Human-readable name for the frontend header, e.g. "Travel Assistant"."""

    input_placeholder: str
    """Input-box placeholder hint, e.g. "Ask about hotels, flights…"."""

    system_prompt: str
    """
    Domain-specific prompt fragment appended after BASE_SYSTEM_PROMPT.

    Must include:
      - Persona ("You are a X assistant")
      - Available tool descriptions in natural language
      - Domain-specific follow-up suggestions after async tools fire
      - Any domain constraints (supported cities, formats, etc.)

    Must NOT include:
      - await_job instructions        (owned by base)
      - How job-completion messages work (owned by base)
      - Generic synthesis / ranking behaviour (owned by base)
      - Job-ID confidentiality rules   (owned by base)
    """

    tools: list[Tool]
    """
    All tools available to this use case. Each Tool carries its name, schema,
    implementation fn, and sync/async mode in one place.
    Do NOT include await_job — the engine appends it automatically.
    """

    hooks: "Hooks | None" = None
    """Optional lifecycle hooks (before_tool, after_tool)."""
