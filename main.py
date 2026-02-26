import os
import time
import json
import uuid
import logging
import openai
import threading
from queue import Queue
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.WARNING,  # silence noisy third-party loggers (openai, httpx, etc.)
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("async_tools")
log.setLevel(logging.DEBUG)  # our logger shows everything

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

from tools import TOOL_FUNCTIONS, TOOLS_SCHEMA, SLOW_TOOLS

SYSTEM_PROMPT = """You are a travel assistant. Tools come in two speeds:

Instant tools (get_weather): return results immediately — present them normally.

Slow tools (get_hotels, get_activities, get_flights): run in the background.
You'll receive a JSON result like {"job_id": "a1b2c3d4", "status": "started", "tool": "...", "args": {...}}.
The tool name and args are included so you know exactly what's running.
Acknowledge the job started (do NOT mention the job ID to the user — it's internal bookkeeping) and ask one smart follow-up:
- Started get_flights → ask if they'd like hotels or activities at the destination
- Started get_hotels → ask if they'd like activities nearby or the current weather
- Started get_activities → ask if they'd like hotel recommendations for that city
- Started multiple jobs → ask about whatever's still missing for a full trip plan

Results arrive as user messages:
  (System) Job a1b2c3d4 completed: <result>
  (System) Job a1b2c3d4 FAILED: <error>
Treat these as tool completions, NOT user speech. When results arrive, proactively synthesize
them with the conversation context — do not wait for the user to ask "which is best."
Filter and rank based on what you know: stated interests, travel companions, budget signals,
or activity areas from earlier in the conversation. Explain briefly why the top picks fit
their situation. Reserve a full flat list only when you have no context to work with.
Note any still-running lookups (without mentioning job IDs), and suggest the next logical step.

Tool dependencies — when a follow-up tool needs a pending job's result:
Check if the job is still running (you started it but haven't seen a "(System) Job X completed"
message yet). If still running: call await_job with that job_id and a natural-language description
of what to call next. Do NOT call the follow-up tool now with guessed args.
If the job already completed: its result is in the conversation above — call the follow-up tool
directly with the real result now.

When you call an intermediate tool triggered by a job completion, re-read the original user
request. If the user intended further steps after this one, also register await_job for the
next dependency in the same response turn — do not wait for the next user message."""

# --- State ---

messages = [{"role": "system", "content": SYSTEM_PROMPT}]
pending_tools = {}   # job_id -> {name, args}
results_queue = Queue()
deferred_hints = {}  # job_id -> list[str] of followup hints


def fire_tool_async(tool_name, tool_args):
    job_id = uuid.uuid4().hex[:8]
    # Capture function reference NOW (before thread starts) so the background thread
    # doesn't read from TOOL_FUNCTIONS after a potential patch.dict context has exited.
    tool_fn = TOOL_FUNCTIONS[tool_name]
    log.info("TOOL START  job=%s  tool=%s  args=%s", job_id, tool_name, json.dumps(tool_args))
    log.debug("PENDING     now tracking %d job(s): %s",
              len(pending_tools) + 1,
              list(pending_tools.keys()) + [job_id])

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


def handle_response(msg):
    """Process OpenAI response: extract text and tool calls."""
    tool_names = [tc.function.name for tc in (msg.tool_calls or [])]
    log.debug("HANDLE RESP  has_content=%s  tool_calls=%s", bool(msg.content), tool_names)

    messages.append(msg.model_dump(exclude_none=True))

    if msg.content:
        print(f"\n🤖 Assistant: {msg.content}")

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
                    log.debug("DEFERRED     hints now: %s",
                              {k: v for k, v in deferred_hints.items()})
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
        handle_response(followup)


def collect_finished_results():
    finished = []
    while not results_queue.empty():
        finished.append(results_queue.get_nowait())
    return finished


def inject_results(finished):
    """Inject completed results as (System) job messages."""
    log.info("INJECT       %d result(s)  pending_before=%s",
             len(finished), list(pending_tools.keys()))
    lines = []
    for job_id, tool_name, tool_args, result, error in finished:
        if error:
            log.error("JOB FAILED   job=%s  tool=%s  error=%s", job_id, tool_name, error)
            lines.append(f"(System) Job {job_id} FAILED: {tool_name}({json.dumps(tool_args)}) → {error}")
        else:
            preview = result[:120] + "..." if len(result) > 120 else result
            log.info("JOB DONE     job=%s  tool=%s  result_preview=%r", job_id, tool_name, preview)
            lines.append(f"(System) Job {job_id} completed: {tool_name}({json.dumps(tool_args)}) → {result}")

        hints = deferred_hints.pop(job_id, [])
        if hints:
            log.info("HINT FIRE    job=%s  %d hint(s): %s", job_id, len(hints), hints)
        else:
            log.debug("HINT CHECK   job=%s  no deferred hints registered", job_id)

        for hint in hints:
            if not error:
                lines.append(
                    f"(System) You had planned a follow-up after job {job_id}: \"{hint}\". "
                    f"Now that the result is available above, call the appropriate tool(s) with "
                    f"correct arguments. If the original user request implies further steps beyond "
                    f"this follow-up, also register await_job for the next dependency."
                )
            else:
                log.warning("HINT FAIL    job=%s  hint=%r  skipped due to job failure", job_id, hint)
                lines.append(
                    f"(System) Note: Follow-up \"{hint}\" after job {job_id} cannot proceed — "
                    f"job FAILED. Inform the user and ask how to proceed."
                )
        pending_tools.pop(job_id, None)

    if pending_tools:
        still = [f"Job {jid} ({v['name']})" for jid, v in pending_tools.items()]
        log.info("STILL PEND   %s", still)
        lines.append(f"(System) Still pending: {', '.join(still)}")

    system_msg = "\n".join(lines)
    log.debug("INJECTION    message to OpenAI:\n%s", system_msg)
    print(f"\n📬 Injecting results:\n{system_msg}")

    messages.append({"role": "user", "content": system_msg})
    response = call_openai()
    handle_response(response)


# --- Non-blocking input ---

input_queue = Queue()


def _input_reader():
    """Runs in a background thread, puts typed lines into input_queue."""
    while True:
        try:
            line = input()
            input_queue.put(line)
        except EOFError:
            input_queue.put(None)
            break


# --- Main loop ---

def main():
    print("=" * 60)
    print("Async Tool Demo (OpenAI) — type your message.")
    print("Results will be delivered automatically when ready.")
    print("Type 'quit' to exit.")
    print("=" * 60)

    # Start background thread that reads stdin without blocking the main loop
    reader = threading.Thread(target=_input_reader, daemon=True)
    reader.start()

    print("\n👤 You: ", end="", flush=True)

    while True:
        # Poll for finished tool results every 0.1s
        time.sleep(0.1)

        finished = collect_finished_results()

        # Check if the user typed something
        user_input = None
        if not input_queue.empty():
            user_input = input_queue.get_nowait()
            if user_input is None:  # EOF
                break
            user_input = user_input.strip()

        if user_input and user_input.lower() == "quit":
            break

        if user_input and finished:
            # User typed something AND results arrived at the same time
            log.info("RACE         user input + %d result(s) arrived simultaneously", len(finished))
            lines = []
            for job_id, tool_name, tool_args, result, error in finished:
                if error:
                    log.error("JOB FAILED   job=%s  tool=%s  error=%s", job_id, tool_name, error)
                    lines.append(f"(System) Job {job_id} FAILED: {tool_name}({json.dumps(tool_args)}) → {error}")
                else:
                    preview = result[:120] + "..." if len(result) > 120 else result
                    log.info("JOB DONE     job=%s  tool=%s  result_preview=%r", job_id, tool_name, preview)
                    lines.append(f"(System) Job {job_id} completed: {tool_name}({json.dumps(tool_args)}) → {result}")
                hints = deferred_hints.pop(job_id, [])
                if hints:
                    log.info("HINT FIRE    job=%s  %d hint(s): %s", job_id, len(hints), hints)
                for hint in hints:
                    if not error:
                        lines.append(
                            f"(System) You had planned a follow-up after job {job_id}: \"{hint}\". "
                            f"Now that the result is available above, call the appropriate tool(s) with "
                            f"correct arguments. If the original user request implies further steps beyond "
                            f"this follow-up, also register await_job for the next dependency."
                        )
                    else:
                        log.warning("HINT FAIL    job=%s  hint=%r  skipped due to job failure", job_id, hint)
                        lines.append(
                            f"(System) Note: Follow-up \"{hint}\" after job {job_id} cannot proceed — "
                            f"job FAILED. Inform the user and ask how to proceed."
                        )
                pending_tools.pop(job_id, None)
            if pending_tools:
                still = [f"Job {jid} ({v['name']})" for jid, v in pending_tools.items()]
                log.info("STILL PEND   %s", still)
                lines.append(f"(System) Still pending: {', '.join(still)}")
            lines.append(f"\nUser says: {user_input}")
            log.info("USER MSG     %r  (combined with injection)", user_input[:100])
            log.debug("INJECTION    combined message:\n%s", "\n".join(lines))
            print(f"\n📬 Injecting results + user message")
            messages.append({"role": "user", "content": "\n".join(lines)})
            response = call_openai()
            handle_response(response)
            print("\n👤 You: ", end="", flush=True)

        elif finished and not user_input:
            # Results arrived, user is idle — auto-inject without waiting
            inject_results(finished)
            print("\n👤 You: ", end="", flush=True)

        elif user_input:
            log.info("USER MSG     %r", user_input[:100])
            messages.append({"role": "user", "content": user_input})
            response = call_openai()
            handle_response(response)
            print("\n👤 You: ", end="", flush=True)


if __name__ == "__main__":
    main()