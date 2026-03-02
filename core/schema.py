from dataclasses import dataclass
from typing import Callable


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

    tool_schemas: list[dict]
    """
    OpenAI function schemas for domain tools.
    Do NOT include await_job — the engine appends it automatically.
    """

    tool_functions: dict[str, Callable]
    """
    Maps tool name -> callable(args: dict) -> str.
    Do NOT include "await_job" — the engine handles it internally.
    """

    slow_tools: set[str]
    """
    Names of tools that run asynchronously (subset of tool_functions keys).
    Instant tools (not in this set) run inline during handle_response().
    """
