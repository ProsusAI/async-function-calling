import os
import json
import uuid
import logging
import threading
import openai
import gradio as gr
from queue import Queue
from dotenv import load_dotenv

from tools import TOOL_FUNCTIONS, TOOLS_SCHEMA, SLOW_TOOLS

load_dotenv()

# ---------------------------------------------------------------------------
# Logging setup — visible in the terminal where `python app.py` runs
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.WARNING,  # silence noisy third-party loggers (gradio, httpx, etc.)
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("async_tools")
log.setLevel(logging.DEBUG)  # our logger shows everything

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """You are a travel assistant. Tools come in two speeds:

Instant tools (get_weather): return results immediately — present them normally.

Slow tools (get_hotels, get_activities, get_flights): run in the background.
You'll receive a JSON result like {"job_id": "a1b2c3d4", "status": "started", "tool": "...", "args": {...}}.
The tool name and args are included so you know exactly what's running.
Acknowledge the job started (do NOT show the raw job_id to the user in chat — but DO use it internally when calling await_job) and ask one smart follow-up:
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

# --- Global state (single-user experiment) ---

messages = [{"role": "system", "content": SYSTEM_PROMPT}]
pending_tools = {}       # job_id -> {name, args}
results_queue = Queue()  # background threads drop results here
_lock = threading.Lock() # serialises messages writes and OpenAI calls
deferred_hints = {}      # job_id -> list[str] of followup hints


# --- Async tool machinery ---

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
        collected.append(handle_response(followup))

    return "\n\n".join(filter(None, collected))


def collect_finished_results():
    finished = []
    while not results_queue.empty():
        finished.append(results_queue.get_nowait())
    return finished


# --- Gradio event handlers ---

def process_user_message(user_text: str, history: list):
    if not user_text.strip():
        yield history, ""
        return

    log.info("USER MSG     %r", user_text[:100])

    # Show user message immediately and clear the textbox
    history = history + [{"role": "user", "content": user_text}]
    yield history, ""

    # Now do the OpenAI call
    with _lock:
        log.debug("LOCK         acquired by process_user_message")
        messages.append({"role": "user", "content": user_text})
        response = call_openai()
        bot_text = handle_response(response)
        log.debug("LOCK         releasing from process_user_message")

    history = history + [{"role": "assistant", "content": bot_text}]
    yield history, ""


def check_and_inject(history: list):
    """Called by gr.Timer every 0.5s. Auto-pushes tool results when ready."""
    finished = collect_finished_results()  # Queue is thread-safe, no lock needed
    if not finished:
        log.debug("TIMER        no results ready  pending=%s", list(pending_tools.keys()))
        return history

    log.info("TIMER        %d result(s) ready to inject  pending_before=%s",
             len(finished), list(pending_tools.keys()))

    with _lock:
        log.debug("LOCK         acquired by check_and_inject")
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

        injection = "\n".join(lines)
        log.debug("INJECTION    message to OpenAI:\n%s", injection)
        messages.append({"role": "user", "content": injection})
        response = call_openai()
        bot_text = handle_response(response)
        log.debug("LOCK         releasing from check_and_inject")

    return history + [{"role": "assistant", "content": bot_text}]


# --- UI ---

with gr.Blocks(title="Async Tools Demo") as demo:
    gr.Markdown(
        "## Async Tools Demo\n"
        "Ask about **hotels**, **activities**, **flights** (Tokyo→Mumbai/Amsterdam), or **weather**. "
        "Tools run in the background — results appear automatically."
    )
    chatbot = gr.Chatbot(height=520, show_label=False, group_consecutive_messages=False)
    msg_box = gr.Textbox(
        placeholder="e.g. What hotels are in Mumbai? What flights go from Tokyo to Amsterdam?",
        show_label=False,
        autofocus=True,
    )
    timer = gr.Timer(value=0.5)

    msg_box.submit(process_user_message, inputs=[msg_box, chatbot], outputs=[chatbot, msg_box])
    timer.tick(check_and_inject, inputs=[chatbot], outputs=[chatbot])


if __name__ == "__main__":
    demo.queue()
    demo.launch(share=False)
