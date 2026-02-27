"""
Experiment: inject a completed async job as a synthetic tool_call + tool result pair.

Instead of any of these ugly options:
  role=user   "(System) Job abc123 completed: ..."   ← BUG-4 current
  role=system "(System) Job abc123 completed: ..."   ← cleaner but not native

We inject two messages that look exactly like a synchronous tool call:
  [assistant] { tool_calls: [{ id, function: {name, arguments} }] }
  [tool]      { tool_call_id, content: <actual result> }

The LLM has been trained on this exact shape. No "(System)" prefix needed.
No role confusion. Just a normal-looking tool result that arrived late.
"""

import os
import uuid
import json
from dotenv import load_dotenv
import openai

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM = """You are a travel assistant.
Slow tools run in the background — their results appear as tool messages later in the conversation.
When a tool result arrives, synthesise it with the conversation and respond helpfully."""

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_hotels",
            "description": "Get hotels for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


def ask(label: str, messages: list):
    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    print(f"{'='*60}")
    for m in messages:
        role = m["role"]
        if role == "assistant" and m.get("tool_calls"):
            tc = m["tool_calls"][0]
            print(f"  [assistant/tool_call] {tc['function']['name']}({tc['function']['arguments']})")
        elif role == "tool":
            print(f"  [tool result] {m['content'][:80].replace(chr(10),' ')}")
        else:
            print(f"  [{role}] {m.get('content','')[:100].replace(chr(10),' ')}")
    print()
    resp = client.chat.completions.create(
        model="gpt-4o",
        tools=TOOLS_SCHEMA,
        messages=messages,
    )
    msg = resp.choices[0].message
    print(f"  → tool_calls: {[tc.function.name for tc in (msg.tool_calls or [])]}")
    print(f"  → content: {(msg.content or '')[:400]}")
    return msg


def make_synthetic_tool_pair(tool_name: str, tool_args: dict, result: str):
    """Build the two messages that represent a completed async tool call."""
    call_id = f"call_{uuid.uuid4().hex[:8]}"
    assistant_msg = {
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
    }
    tool_msg = {
        "role": "tool",
        "tool_call_id": call_id,
        "content": result,
    }
    return assistant_msg, tool_msg


# ── Test 1: basic synthetic injection ────────────────────────────────────────
tool_call_msg, tool_result_msg = make_synthetic_tool_pair(
    "get_hotels",
    {"city": "amsterdam"},
    "Hotels in Amsterdam:\n  • The Dylan — Jordaan — $350/night\n  • citizenM — Museum Quarter — $160/night\n  • Generator — East — $60/night"
)

ask("Basic synthetic tool injection", [
    {"role": "system",    "content": SYSTEM},
    {"role": "user",      "content": "Find hotels in Amsterdam."},
    {"role": "assistant", "content": "I've started the hotel search, it'll take a moment."},
    tool_call_msg,
    tool_result_msg,
    {"role": "user",      "content": "Which one should I pick? I'm on a budget."},
])


# ── Test 2: does the LLM try to call get_hotels again? ───────────────────────
# (It shouldn't — the result is already in context)
ask("Does LLM avoid re-calling the tool when result is already present?", [
    {"role": "system",    "content": SYSTEM},
    {"role": "user",      "content": "Find hotels in Amsterdam."},
    {"role": "assistant", "content": "Searching now..."},
    tool_call_msg,
    tool_result_msg,
    {"role": "user",      "content": "What hotels did you find?"},
])


# ── Test 3: two synthetic tool injections in a row ────────────────────────────
flights_call, flights_result = make_synthetic_tool_pair(
    "get_hotels",
    {"city": "amsterdam"},
    "Hotels: The Dylan $350, citizenM $160, Generator $60"
)
activities_call, activities_result = make_synthetic_tool_pair(
    "get_hotels",   # reusing same schema just to test two injections
    {"city": "amsterdam"},
    "Activities: Canal boat tour, Rijksmuseum, Vondelpark picnic, Heineken Experience"
)

ask("Two synthetic injections — does LLM synthesise both?", [
    {"role": "system",    "content": SYSTEM},
    {"role": "user",      "content": "Plan my Amsterdam trip."},
    {"role": "assistant", "content": "I've started searching for hotels and activities."},
    flights_call,
    flights_result,
    activities_call,
    activities_result,
    {"role": "user",      "content": "Give me a quick plan."},
])
