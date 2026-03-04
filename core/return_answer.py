"""return_answer_to_parent — framework-owned tool for sub-agents.

Sub-agents MUST call this tool to return their final answer to the parent agent.
It is automatically added to every sub-agent's tool list by AsyncEngine when the
engine is constructed with a done_event (i.e. when it is acting as a sub-agent).
"""

RETURN_ANSWER_SCHEMA = {
    "type": "function",
    "function": {
        "name": "return_answer_to_parent",
        "description": (
            "Return your final synthesized answer to the agent that called you. "
            "You MUST call this tool when you have finished your task — do not stop without calling it. "
            "Provide a complete, self-contained answer; the calling agent cannot ask you follow-up questions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The complete synthesized answer to return to the parent agent.",
                }
            },
            "required": ["answer"],
        },
    },
}
