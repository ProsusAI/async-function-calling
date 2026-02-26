import os
import time
import json
import uuid
import openai
import threading
from queue import Queue
from dotenv import load_dotenv

load_dotenv()

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
Note any still-running lookups (without mentioning job IDs), and suggest the next logical step."""

# --- State ---

messages = [{"role": "system", "content": SYSTEM_PROMPT}]
pending_tools = {}  # tool_call_id -> {name, args, thread}
results_queue = Queue()


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


def handle_response(msg):
    """Process OpenAI response: extract text and tool calls."""

    # Append the raw assistant message to history
    # OpenAI needs the full message object for tool_calls context
    messages.append(msg.model_dump(exclude_none=True))

    if msg.content:
        print(f"\n🤖 Assistant: {msg.content}")

    # If there are tool calls, fire them async and return PENDING
    if msg.tool_calls:
        for tc in msg.tool_calls:
            tool_name = tc.function.name
            tool_args = json.loads(tc.function.arguments)
            tool_call_id = tc.id

            if tool_name in SLOW_TOOLS:
                print(f"    🔧 Tool called (slow): {tool_name}({json.dumps(tool_args)})")
                content = fire_tool_async(tool_name, tool_args)
            else:
                print(f"    ⚡ Tool called (instant): {tool_name}({json.dumps(tool_args)})")
                content = TOOL_FUNCTIONS[tool_name](tool_args)

            messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": content})

        followup = call_openai()
        handle_response(followup)


def collect_finished_results():
    finished = []
    while not results_queue.empty():
        finished.append(results_queue.get_nowait())
    return finished


def inject_results(finished):
    """Inject completed results as (System) job messages."""
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

    system_msg = "\n".join(lines)
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
            lines.append(f"\nUser says: {user_input}")
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
            messages.append({"role": "user", "content": user_input})
            response = call_openai()
            handle_response(response)
            print("\n👤 You: ", end="", flush=True)


if __name__ == "__main__":
    main()