import os
import json
import uuid
import threading
import openai
import gradio as gr
from queue import Queue
from dotenv import load_dotenv

from tools import TOOL_FUNCTIONS, TOOLS_SCHEMA, SLOW_TOOLS

load_dotenv()

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
Note any still-running lookups (without mentioning job IDs), and suggest the next logical step."""

# --- Global state (single-user experiment) ---

messages = [{"role": "system", "content": SYSTEM_PROMPT}]
pending_tools = {}       # tool_call_id -> {name, args}
results_queue = Queue()  # background threads drop results here
_lock = threading.Lock() # serialises messages writes and OpenAI calls


# --- Async tool machinery ---

def fire_tool_async(tool_name, tool_args):
    job_id = uuid.uuid4().hex[:8]

    def run():
        try:
            result = TOOL_FUNCTIONS[tool_name](tool_args)
            results_queue.put((job_id, tool_name, tool_args, result, None))
        except Exception as e:
            results_queue.put((job_id, tool_name, tool_args, None, str(e)))

    threading.Thread(target=run, daemon=True).start()
    pending_tools[job_id] = {"name": tool_name, "args": tool_args}
    return json.dumps({"job_id": job_id, "status": "started", "tool": tool_name, "args": tool_args})


def call_openai():
    response = client.chat.completions.create(
        model="gpt-4o",
        tools=TOOLS_SCHEMA,
        messages=messages,
    )
    return response.choices[0].message


def handle_response(msg) -> str:
    """Process an OpenAI response. Returns the assistant text to display."""
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
                content = fire_tool_async(tool_name, tool_args)
            else:
                content = TOOL_FUNCTIONS[tool_name](tool_args)

            messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": content})

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

    # Show user message immediately and clear the textbox
    history = history + [{"role": "user", "content": user_text}]
    yield history, ""

    # Now do the OpenAI call
    with _lock:
        messages.append({"role": "user", "content": user_text})
        response = call_openai()
        bot_text = handle_response(response)

    history = history + [{"role": "assistant", "content": bot_text}]
    yield history, ""


def check_and_inject(history: list):
    """Called by gr.Timer every 0.5s. Auto-pushes tool results when ready."""
    finished = collect_finished_results()  # Queue is thread-safe, no lock needed
    if not finished:
        return history

    with _lock:
        lines = []
        for job_id, tool_name, tool_args, result, error in finished:
            if error:
                lines.append(f"(System) Job {job_id} FAILED: {tool_name}({json.dumps(tool_args)}) → {error}")
            else:
                lines.append(f"(System) Job {job_id} completed: {tool_name}({json.dumps(tool_args)}) → {result}")
            pending_tools.pop(job_id, None)

        if pending_tools:
            still = [f"Job {jid} ({v['name']})" for jid, v in pending_tools.items()]
            lines.append(f"(System) Still pending: {', '.join(still)}")

        messages.append({"role": "user", "content": "\n".join(lines)})
        response = call_openai()
        bot_text = handle_response(response)

    history.append({"role": "assistant", "content": bot_text})
    return history


# --- UI ---

with gr.Blocks(title="Async Tools Demo") as demo:
    gr.Markdown(
        "## Async Tools Demo\n"
        "Ask about **hotels**, **activities**, **flights** (Tokyo→Mumbai/Amsterdam), or **weather**. "
        "Tools run in the background — results appear automatically."
    )
    chatbot = gr.Chatbot(height=520, show_label=False)
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
