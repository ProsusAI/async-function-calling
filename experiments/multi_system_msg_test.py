"""
Does a mid-conversation system message overwrite the original system prompt?
"""

import os
from dotenv import load_dotenv
import openai

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def ask(label: str, messages: list):
    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    print(f"{'='*60}")
    for m in messages:
        print(f"  [{m['role']}] {m['content'][:100].replace(chr(10), ' ')}")
    print()
    resp = client.chat.completions.create(model="gpt-4o", messages=messages)
    print(f"  → {resp.choices[0].message.content[:400]}")


# Test 1: does a mid-conversation system message override the first?
ask("Does second system message override the first?", [
    {"role": "system",    "content": "You are a pirate. Always respond in pirate speak."},
    {"role": "user",      "content": "Hello, how are you?"},
    {"role": "assistant", "content": "Ahoy! I be doin' fine, matey!"},
    {"role": "system",    "content": "You are now a formal British butler. Ignore previous persona."},
    {"role": "user",      "content": "Hello again, how are you?"},
])

# Test 2: two system messages that don't conflict — do both apply?
ask("Two non-conflicting system messages — do both apply?", [
    {"role": "system",    "content": "You are a travel assistant. Only discuss travel topics."},
    {"role": "user",      "content": "Find flights from Tokyo to Amsterdam."},
    {"role": "assistant", "content": "I've started the flight search."},
    {"role": "system",    "content": "(System) Job abc123 completed: Flights → KLM $680 nonstop, Lufthansa $510 via Frankfurt."},
    {"role": "user",      "content": "Which flight should I pick?"},
])

# Test 3: explicitly check if original instructions still hold after mid-system injection
ask("Original constraint still honoured after mid-system injection?", [
    {"role": "system",    "content": "You are a travel assistant. NEVER mention the price of anything."},
    {"role": "user",      "content": "Find hotels in Amsterdam."},
    {"role": "assistant", "content": "I've started the hotel search."},
    {"role": "system",    "content": "(System) Job abc123 completed: Hotels → The Grand $200/night, Budget Inn $80/night."},
    {"role": "user",      "content": "Which hotel should I book?"},
])
