# Framework-owned tool: await_job
#
# This schema is appended automatically by AsyncEngine to every use case's
# tool list. Use cases must NOT include it themselves.

AWAIT_JOB_SCHEMA = {
    "type": "function",
    "function": {
        "name": "await_job",
        "description": (
            "Call this immediately after starting a slow (background) tool "
            "if you already know the next step. Pass the exact job_id from "
            "that tool's response and describe what to do with the result. "
            "Example: after firing get_flights(origin='tokyo', destination='amsterdam'), "
            "call await_job(job_id='<id>', followup_hint='call get_hotels(city=amsterdam) "
            "for 2 nights'). "
            "Do NOT call the follow-up tool now with guessed args — await_job ensures it "
            "runs with real data. If the job has already completed (its result is already "
            "in the conversation), call the follow-up tool directly instead of using await_job."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "The job_id from the still-running background job to wait for.",
                },
                "followup_hint": {
                    "type": "string",
                    "description": (
                        "Natural language: what tool to call and how to map the result to its args. "
                        "Example: 'call get_activities with destination city and tag=couple'. "
                        "If this is an intermediate step in a chain, also note the step after it."
                    ),
                },
            },
            "required": ["job_id", "followup_hint"],
        },
    },
}
