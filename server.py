import argparse
import asyncio
import json
import logging
import os
import threading
import uuid
from contextlib import asynccontextmanager
from queue import Queue

import openai
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from tools import TOOL_FUNCTIONS, TOOLS_SCHEMA, SLOW_TOOLS

load_dotenv()

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("async_tools")
log.setLevel(logging.DEBUG)

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------------------------------------------------------
# System prompts — one per injection mode
#
# "user" / "system" modes: the LLM needs to be told that "(System) Job X
#   completed:" lines are tool completions, not user speech.
#
# "tool" mode: results arrive as native tool messages in the history —
#   no special explanation needed; the LLM already knows how to read them.
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_MSG = """You are a travel assistant. Tools come in two speeds:

Instant tools (get_weather): return results immediately — present them normally.

Slow tools (get_hotels, get_activities, get_flights): run in the background.
You'll receive a JSON result like {"job_id": "a1b2c3d4", "status": "started", "tool": "...", "args": {...}}.
The tool name and args are included so you know exactly what's running.
Acknowledge the job started (do NOT show the raw job_id to the user in chat — but DO use it internally when calling await_job) and ask one smart follow-up:
- Started get_flights → ask if they'd like hotels or activities at the destination
- Started get_hotels → ask if they'd like activities nearby or the current weather
- Started get_activities → ask if they'd like hotel recommendations for that city
- Started multiple jobs → ask about whatever's still missing for a full trip plan

Results arrive as messages:
  (System) Job a1b2c3d4 completed: <result>
  (System) Job a1b2c3d4 FAILED: <error>
Treat these as tool completions, NOT user speech. When results arrive, proactively synthesize
them with the conversation context — do not wait for the user to ask "which is best."
Filter and rank based on what you know: stated interests, travel companions, budget signals,
or activity areas from earlier in the conversation. Explain briefly why the top picks fit
their situation. Reserve a full flat list only when you have no context to work with.
Note any still-running lookups (without mentioning job IDs), and suggest the next logical step.

Tool dependencies — chaining with await_job:
Immediately after you start a slow tool, if you already know the next step, call await_job
in your NEXT response — before asking the user anything. Do not wait for a later message.
- Use the exact job_id from the tool response's "job_id" field.
- Be concrete: e.g. followup_hint="call get_activities(city='amsterdam', tag='couple')".
- For multi-step chains, register ALL follow-ups in the same response turn.
Do NOT guess or hallucinate a job_id. Do NOT call the follow-up tool now with placeholder args.

If the job already completed (you see "(System) Job X completed" in the conversation):
call the follow-up tool directly with the real result — do not use await_job.

When triggered by a job completion, re-read the original user request. If further steps
remain, also register await_job for the next dependency in the same response turn."""

_SYSTEM_PROMPT_TOOL = """You are a travel assistant. Tools come in two speeds:

Instant tools (get_weather): return results immediately — present them normally.

Slow tools (get_hotels, get_activities, get_flights): run in the background.
You'll receive a JSON result like {"job_id": "a1b2c3d4", "status": "started", "tool": "...", "args": {...}}.
The tool name and args are included so you know exactly what's running.
Acknowledge the job started (do NOT show the raw job_id to the user in chat — but DO use it internally when calling await_job) and ask one smart follow-up:
- Started get_flights → ask if they'd like hotels or activities at the destination
- Started get_hotels → ask if they'd like activities nearby or the current weather
- Started get_activities → ask if they'd like hotel recommendations for that city
- Started multiple jobs → ask about whatever's still missing for a full trip plan

Background tool results arrive as tool messages in the conversation history — treat them
exactly like instant tool results. When a result appears, proactively synthesize it with
the conversation context — do not wait for the user to ask "which is best."
Filter and rank based on what you know: stated interests, travel companions, budget signals,
or activity areas from earlier in the conversation. Explain briefly why the top picks fit
their situation. Reserve a full flat list only when you have no context to work with.
Note any still-running lookups (without mentioning job IDs), and suggest the next logical step.

Tool dependencies — chaining with await_job:
Immediately after you start a slow tool, if you already know the next step, call await_job
in your NEXT response — before asking the user anything. Do not wait for a later message.
- Use the exact job_id from the tool response's "job_id" field.
- Be concrete: e.g. followup_hint="call get_activities(city='amsterdam', tag='couple')".
- For multi-step chains, register ALL follow-ups in the same response turn.
Do NOT guess or hallucinate a job_id. Do NOT call the follow-up tool now with placeholder args.

If the job already completed (you see its tool result in the conversation):
call the follow-up tool directly with the real result — do not use await_job.

When triggered by a job completion, re-read the original user request. If further steps
remain, also register await_job for the next dependency in the same response turn."""

_SYSTEM_PROMPTS = {
    "user":   _SYSTEM_PROMPT_MSG,
    "system": _SYSTEM_PROMPT_MSG,   # same instructions, different role
    "tool":   _SYSTEM_PROMPT_TOOL,
}

# ---------------------------------------------------------------------------
# Injection mode — set by CLI arg at startup, read by _inject_finished()
# ---------------------------------------------------------------------------

INJECTION_MODE: str = "tool"   # default; overridden in __main__

# ---------------------------------------------------------------------------
# Global state (single-user demo)
# ---------------------------------------------------------------------------

messages = [{"role": "system", "content": _SYSTEM_PROMPTS[INJECTION_MODE]}]
pending_tools = {}       # job_id -> {name, args}
results_queue = Queue()  # background threads drop results here
_lock = threading.Lock() # serialises messages writes and OpenAI calls
deferred_hints = {}      # job_id -> list[str] of followup hints

# SSE: one asyncio.Queue per connected browser tab
_sse_loop: asyncio.AbstractEventLoop = None
_sse_clients: list[asyncio.Queue] = []


def push_event(event_type: str, data: dict):
    """Thread-safe push to all connected SSE clients."""
    payload = json.dumps({"type": event_type, "data": data})
    if _sse_loop:
        for q in list(_sse_clients):
            _sse_loop.call_soon_threadsafe(q.put_nowait, payload)


# ---------------------------------------------------------------------------
# Async tool machinery
# ---------------------------------------------------------------------------

def fire_tool_async(tool_name, tool_args):
    job_id = uuid.uuid4().hex[:8]
    tool_fn = TOOL_FUNCTIONS[tool_name]
    log.info("TOOL START  job=%s  tool=%s  args=%s", job_id, tool_name, json.dumps(tool_args))

    def run():
        log.debug("BG THREAD   job=%s  tool=%s  executing...", job_id, tool_name)
        try:
            result = tool_fn(tool_args)
            preview = result[:120] + "..." if len(result) > 120 else result
            log.info("BG DONE     job=%s  tool=%s  result_preview=%r", job_id, tool_name, preview)
            results_queue.put((job_id, tool_name, tool_args, result, None))
        except Exception as e:
            log.error("BG FAILED   job=%s  tool=%s  error=%s", job_id, tool_name, e)
            results_queue.put((job_id, tool_name, tool_args, None, str(e)))
        threading.Thread(target=_run_injection, daemon=True).start()

    threading.Thread(target=run, daemon=True).start()
    pending_tools[job_id] = {"name": tool_name, "args": tool_args}
    return json.dumps({"job_id": job_id, "status": "started", "tool": tool_name, "args": tool_args})


def call_openai():
    log.info("OPENAI CALL  messages=%d  (last role: %s)", len(messages), messages[-1]["role"])
    response = client.chat.completions.create(
        model="gpt-4o",
        tools=TOOLS_SCHEMA,
        messages=messages,
    )
    msg = response.choices[0].message
    tool_names = [tc.function.name for tc in (msg.tool_calls or [])]
    content_preview = (msg.content or "")[:80].replace("\n", " ")
    log.info("OPENAI RESP  content=%r  tool_calls=%s", content_preview, tool_names)
    return msg


def handle_response(msg) -> str:
    """Process an OpenAI response. Returns the assistant text to display."""
    tool_names = [tc.function.name for tc in (msg.tool_calls or [])]
    log.debug("HANDLE RESP  has_content=%s  tool_calls=%s", bool(msg.content), tool_names)
    messages.append(msg.model_dump(exclude_none=True))

    collected = []
    if msg.content:
        collected.append(msg.content)

    if msg.tool_calls:
        for tc in msg.tool_calls:
            tool_name = tc.function.name
            tool_args = json.loads(tc.function.arguments)
            tool_call_id = tc.id

            if tool_name in SLOW_TOOLS:
                log.info("DISPATCH     slow tool=%s  args=%s", tool_name, json.dumps(tool_args))
                content = fire_tool_async(tool_name, tool_args)

            elif tool_name == "await_job":
                job_id = tool_args.get("job_id", "")
                hint   = tool_args.get("followup_hint", "")
                if job_id in pending_tools and hint:
                    deferred_hints.setdefault(job_id, []).append(hint)
                    log.info("AWAIT_JOB    registered  job=%s  hint=%r", job_id, hint)
                    content = json.dumps({
                        "status": "registered",
                        "job_id": job_id,
                        "message": "Follow-up intent recorded. You will be reminded when this job completes.",
                    })
                else:
                    reason = "job not in pending_tools" if job_id not in pending_tools else "empty hint"
                    log.warning("AWAIT_JOB    REJECTED  job=%r  reason=%s  active_jobs=%s",
                                job_id, reason, list(pending_tools.keys()))
                    content = json.dumps({
                        "status": "error",
                        "message": (
                            f"No active job with id '{job_id}'. "
                            "If this job already completed, its result is in the conversation — "
                            "call the follow-up tool directly with that result."
                        ),
                    })

            else:
                log.info("DISPATCH     instant tool=%s  args=%s", tool_name, json.dumps(tool_args))
                content = TOOL_FUNCTIONS[tool_name](tool_args)
                preview = content[:80] + "..." if len(content) > 80 else content
                log.debug("INSTANT RES  tool=%s  result=%r", tool_name, preview)

            messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": content})

        log.debug("FOLLOWUP     making follow-up OpenAI call after %d tool result(s)", len(msg.tool_calls))
        followup = call_openai()
        collected.append(handle_response(followup))

    return "\n\n".join(filter(None, collected))


# ---------------------------------------------------------------------------
# Injection strategies
# ---------------------------------------------------------------------------

def _inject_finished(finished: list):
    """Append completed job results into messages using INJECTION_MODE.

    "user"   — single {"role": "user",   "content": "(System) Job X completed: ..."}
    "system" — single {"role": "system", "content": "(System) Job X completed: ..."}
    "tool"   — per-job synthetic assistant tool_call + tool result pair
    """
    log.info("INJECT  mode=%s  jobs=%d  pending_before=%s",
             INJECTION_MODE, len(finished), list(pending_tools.keys()))

    if INJECTION_MODE == "tool":
        for job_id, tool_name, tool_args, result, error in finished:
            if error:
                log.error("JOB FAILED   job=%s  tool=%s  error=%s", job_id, tool_name, error)
            else:
                preview = result[:120] + "..." if len(result) > 120 else result
                log.info("JOB DONE     job=%s  tool=%s  result_preview=%r", job_id, tool_name, preview)

            # Synthetic assistant message that "called" this tool
            call_id = f"call_{uuid.uuid4().hex[:8]}"
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(tool_args),
                    },
                }],
            })
            # Tool result (or error) paired with that call
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": result if result is not None else f"ERROR: {error}",
            })

            hints = deferred_hints.pop(job_id, [])
            if hints:
                log.info("HINT FIRE    job=%s  %d hint(s): %s", job_id, len(hints), hints)
            for hint in hints:
                if not error:
                    messages.append({"role": "system", "content":
                        f"You had planned a follow-up: \"{hint}\". "
                        f"The result is now available above — call the appropriate tool(s)."})
                else:
                    log.warning("HINT FAIL    job=%s  hint=%r  skipped due to failure", job_id, hint)
                    messages.append({"role": "system", "content":
                        f"Follow-up \"{hint}\" cannot proceed — the tool failed. "
                        f"Inform the user and ask how to proceed."})

            pending_tools.pop(job_id, None)

        if pending_tools:
            still = [f"{v['name']}({json.dumps(v['args'])})" for v in pending_tools.values()]
            log.info("STILL PEND   %s", still)
            messages.append({"role": "system",
                              "content": f"Still running in the background: {', '.join(still)}"})

    else:  # "user" or "system"
        lines = []
        for job_id, tool_name, tool_args, result, error in finished:
            if error:
                log.error("JOB FAILED   job=%s  tool=%s  error=%s", job_id, tool_name, error)
                lines.append(
                    f"(System) Job {job_id} FAILED: {tool_name}({json.dumps(tool_args)}) → {error}")
            else:
                preview = result[:120] + "..." if len(result) > 120 else result
                log.info("JOB DONE     job=%s  tool=%s  result_preview=%r", job_id, tool_name, preview)
                lines.append(
                    f"(System) Job {job_id} completed: {tool_name}({json.dumps(tool_args)}) → {result}")

            hints = deferred_hints.pop(job_id, [])
            if hints:
                log.info("HINT FIRE    job=%s  %d hint(s): %s", job_id, len(hints), hints)
            for hint in hints:
                if not error:
                    lines.append(
                        f"(System) You had planned a follow-up after job {job_id}: \"{hint}\". "
                        f"Now that the result is available above, call the appropriate tool(s) with "
                        f"correct arguments. If the original user request implies further steps beyond "
                        f"this follow-up, also register await_job for the next dependency.")
                else:
                    log.warning("HINT FAIL    job=%s  hint=%r  skipped due to failure", job_id, hint)
                    lines.append(
                        f"(System) Note: Follow-up \"{hint}\" after job {job_id} cannot proceed — "
                        f"job FAILED. Inform the user and ask how to proceed.")

            pending_tools.pop(job_id, None)

        if pending_tools:
            still = [f"Job {jid} ({v['name']})" for jid, v in pending_tools.items()]
            log.info("STILL PEND   %s", still)
            lines.append(f"(System) Still pending: {', '.join(still)}")

        injection = "\n".join(lines)
        log.debug("INJECTION    message to OpenAI:\n%s", injection)
        messages.append({"role": INJECTION_MODE, "content": injection})


def _run_injection():
    """Called from a background thread when a job completes."""
    with _lock:
        finished = []
        while not results_queue.empty():
            try:
                finished.append(results_queue.get_nowait())
            except Exception:
                break

        if not finished:
            return

        _inject_finished(finished)
        response = call_openai()
        bot_text = handle_response(response)

    push_event("assistant", {"content": bot_text})


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _sse_loop
    _sse_loop = asyncio.get_event_loop()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/stream")
async def stream():
    q: asyncio.Queue = asyncio.Queue()
    _sse_clients.append(q)

    async def event_generator():
        try:
            while True:
                data = await q.get()
                yield f"data: {data}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            _sse_clients.remove(q)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/chat")
def chat(body: dict):
    user_text = (body.get("message") or "").strip()
    if not user_text:
        return {"ok": False, "error": "empty message"}

    log.info("USER MSG     %r", user_text[:100])

    with _lock:
        log.debug("LOCK         acquired by /chat")
        messages.append({"role": "user", "content": user_text})
        response = call_openai()
        bot_text = handle_response(response)
        log.debug("LOCK         releasing from /chat")

    push_event("assistant", {"content": bot_text})
    return {"ok": True}


@app.post("/reset")
def reset():
    """Clear conversation history (keep system prompt)."""
    global messages, pending_tools, deferred_hints
    with _lock:
        messages.clear()
        messages.append({"role": "system", "content": _SYSTEM_PROMPTS[INJECTION_MODE]})
        pending_tools.clear()
        deferred_hints.clear()
    push_event("reset", {})
    return {"ok": True}


# Serve static files last so API routes take precedence
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Async Tools Demo server")
    parser.add_argument(
        "--injection-mode",
        choices=["user", "system", "tool"],
        default="tool",
        help=(
            "How completed background job results are injected into the LLM context.\n"
            "  user   — appended as a user-role message (original BUG-4 behaviour)\n"
            "  system — appended as a system-role message (semantically cleaner)\n"
            "  tool   — injected as a synthetic assistant tool_call + tool result pair (native)"
        ),
    )
    args = parser.parse_args()

    INJECTION_MODE = args.injection_mode
    # Re-initialise messages with the prompt that matches the chosen mode
    messages.clear()
    messages.append({"role": "system", "content": _SYSTEM_PROMPTS[INJECTION_MODE]})

    log.info("=" * 50)
    log.info("Starting server  injection_mode=%s", INJECTION_MODE)
    log.info("=" * 50)
    print(f"Starting server with injection_mode={INJECTION_MODE!r}")

    uvicorn.run(app, host="0.0.0.0", port=7862, log_level="warning")
