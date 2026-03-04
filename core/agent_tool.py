"""AgentTool — wraps a UseCase as a callable tool for an orchestrator agent.

The sub-agent runs its own full AsyncEngine loop (with its own queue, lock, and
background threads). It signals completion by calling the `return_answer_to_parent`
framework tool, which unblocks the AgentTool via a threading.Event.

Parameters
----------
is_async : bool
    Controls how the sub-agent is invoked FROM THE PARENT's perspective.
    True  → parent fires sub-agent in a background thread (non-blocking, gets job_id).
    False → parent blocks until sub-agent finishes.

forced_sync : bool
    Controls how the SUB-AGENT's own tools run INTERNALLY.
    True  → sub-agent runs all its tools inline, sequentially (no internal parallelism).
    False → sub-agent can fire its own background threads for async tools.

These two flags are orthogonal — any combination is valid.

max_steps : int
    Maximum number of OpenAI call rounds the sub-agent may make before giving up.
"""

import threading

from .schema import Tool, UseCase


class AgentTool(Tool):
    """A Tool whose implementation is another agent (UseCase)."""

    def __init__(
        self,
        name: str,
        description: str,
        use_case: UseCase,
        is_async: bool = False,
        forced_sync: bool = False,
        max_steps: int = 20,
    ):
        # Import here to avoid circular import at module load time.
        from .engine import AsyncEngine

        def _run(args: dict) -> str:
            query = args.get("query", "")

            done_event = threading.Event()
            answer_box: dict = {}

            engine = AsyncEngine(
                use_case,
                forced_sync=forced_sync,
                done_event=done_event,
                answer_box=answer_box,
                max_steps=max_steps,
            )
            engine.messages.append({"role": "user", "content": query})

            # Kick off the sub-agent's ReACT loop. For sync tools this returns
            # only after all tool calls are done. For async tools in the sub-agent,
            # background threads may still be running when this returns — we wait
            # below on done_event for the explicit completion signal.
            response = engine.call_openai()
            engine.handle_response(response)

            # Block until return_answer_to_parent fires (or timeout).
            if done_event.wait(timeout=120.0):
                return answer_box.get("answer", "Sub-agent completed but returned no answer.")
            return "Sub-agent timed out without returning an answer."

        super().__init__(
            name=name,
            description=description,
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The full task to delegate to this specialist agent.",
                    },
                },
                "required": ["query"],
            },
            fn=_run,
            is_async=is_async,
        )
