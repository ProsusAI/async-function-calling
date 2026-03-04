"""AgentTool — wraps a UseCase as a callable tool for an orchestrator agent.

The sub-agent runs to completion via SyncEngine and returns its final
synthesized text as the tool result. The orchestrator treats this exactly
like any other tool response and can synthesize across multiple agents.

  is_async=False (default)
    The orchestrator blocks until the sub-agent finishes. Simpler flow,
    agents run sequentially.

  is_async=True
    The sub-agent fires in a background thread (fire-and-forget). The
    orchestrator receives a job_id immediately and can use await_job to
    chain follow-ups. Multiple AgentTools with is_async=True run in
    parallel automatically via the existing injection machinery.
"""

from .schema import Tool, UseCase


class AgentTool(Tool):
    """A Tool whose implementation is another agent (UseCase)."""

    def __init__(
        self,
        name: str,
        description: str,
        use_case: UseCase,
        is_async: bool = False,
    ):
        # Import here to avoid any potential import-order issues at module load.
        # SyncEngine → AsyncEngine → schema, all within core — no circular deps.
        from .sync_engine import SyncEngine

        def _run(args: dict) -> str:
            query = args.get("query", "")
            engine = SyncEngine(use_case)
            engine.messages.append({"role": "user", "content": query})
            response = engine.call_openai()
            return engine.handle_response(response)

        super().__init__(
            name=name,
            description=description,
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The full request to delegate to this specialist agent.",
                    },
                },
                "required": ["query"],
            },
            fn=_run,
            is_async=is_async,
        )
