"""
Experiment: can we send multiple consecutive user messages before an assistant reply?

Bug-4 context: _run_injection() appends job-completion data as role="user",
which is semantically wrong. The question is:
  - Does the API accept consecutive user messages?
  - Does it accept role="system" mid-conversation?
  - How does the LLM behave differently in each case?

We test four message shapes and print the LLM's reply + any API error.
"""

import os
from dotenv import load_dotenv
import openai

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM = "You are a helpful assistant. Answer concisely."


def ask(label: str, messages: list):
    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    print(f"{'='*60}")
    for m in messages:
        role = m["role"]
        content = m["content"][:120].replace("\n", " ")
        print(f"  [{role}] {content}")
    print()

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
        )
        reply = resp.choices[0].message.content
        print(f"  → REPLY: {reply[:300]}")
    except Exception as e:
        print(f"  → ERROR: {e}")


# ── Test 1: baseline — normal alternating pattern ────────────────────────────
ask("Baseline (normal alternating)", [
    {"role": "system",    "content": SYSTEM},
    {"role": "user",      "content": "My name is Alice."},
    {"role": "assistant", "content": "Got it, Alice!"},
    {"role": "user",      "content": "What is my name?"},
])

# ── Test 2: two consecutive user messages ─────────────────────────────────────
ask("Two consecutive user messages", [
    {"role": "system",    "content": SYSTEM},
    {"role": "user",      "content": "My name is Alice."},
    {"role": "user",      "content": "What is my name?"},
])

# ── Test 3: user message followed by system message (mid-conversation) ────────
ask("Mid-conversation system message (current BUG-4 fix candidate)", [
    {"role": "system",    "content": SYSTEM},
    {"role": "user",      "content": "My name is Alice."},
    {"role": "assistant", "content": "Got it, Alice!"},
    {"role": "user",      "content": "Search for hotels started as job abc123."},
    {"role": "assistant", "content": "I've started the hotel search."},
    {"role": "system",    "content": "(System) Job abc123 completed: Hotels → The Grand $200/night, Budget Inn $80/night."},
    {"role": "user",      "content": "Which hotel should I pick?"},
])

# ── Test 4: three consecutive user messages ───────────────────────────────────
ask("Three consecutive user messages", [
    {"role": "system",    "content": SYSTEM},
    {"role": "user",      "content": "My name is Alice."},
    {"role": "user",      "content": "(System) Job abc123 completed: Hotels → The Grand $200/night."},
    {"role": "user",      "content": "Which hotel should I pick?"},
])

# ── Test 5: the current app pattern — injection then real user msg ────────────
ask("Injection as user msg, then another user msg (actual BUG-4 scenario)", [
    {"role": "system",    "content": SYSTEM},
    {"role": "user",      "content": "Find me hotels in Amsterdam."},
    {"role": "assistant", "content": "I've started searching for hotels in Amsterdam."},
    {"role": "user",      "content": "(System) Job abc123 completed: get_hotels({\"city\": \"amsterdam\"}) → Hotels in Amsterdam:\n  • The Grand $200/night\n  • Budget Inn $80/night"},
    {"role": "user",      "content": "Actually, make it Mumbai instead."},
])
