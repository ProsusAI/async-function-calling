# Async Tool Calling — Project Notes

## The Problem

LLM tool calling is synchronous by nature. When the model calls a tool, the entire
conversation blocks until that tool returns. If a tool takes 15–30 seconds, the user
stares at a spinner with no way to interact.

## The Core Approach

We fake async by exploiting the fact that neither OpenAI nor Anthropic's APIs require
tool results to be real — they just need to be present before the next model call.

**The trick:**
1. Model calls a slow tool → we immediately return a job ID as the tool result and fire
   the real work in a background thread
2. Model sees the job info, acknowledges it, and asks a smart follow-up question
3. User keeps chatting normally — can trigger more tool calls while results are pending
4. When a background tool finishes, its result is injected as a `(System) Job X completed`
   user message
5. Model resumes with full context, matched to the original job ID

The system prompt teaches the model this protocol — instant vs slow tools, what job JSON
looks like, and that `(System)` messages are tool completions not user speech.

---

## Two-Speed Tools

Not all tools need to be async. We classify at definition time in `tools.py`:

```python
SLOW_TOOLS = {"get_hotels", "get_activities", "get_flights"}
# anything not in this set runs synchronously and returns a real result immediately
```

`handle_response` branches on this per tool call:
- **Slow** → `fire_tool_async()` → background thread, returns job JSON to the model
- **Instant** (e.g. `get_weather`) → runs inline, real result in the same turn

**Why separate them:** Forcing every tool through async adds latency and complexity for
no reason. Instant tools (cheap lookups, cached data) should just respond immediately.

---

## Job IDs (not `[PENDING]`)

Early version returned a fake `"[PENDING]"` string as the tool result. The model had no
explicit link between that and the eventual result.

**Current approach:** slow tools return structured JSON:
```json
{"job_id": "a1b2c3d4", "status": "started", "tool": "get_hotels", "args": {"city": "mumbai"}}
```

Results arrive as:
```
(System) Job a1b2c3d4 completed: Hotels in Mumbai: ...
(System) Job a1b2c3d4 FAILED: <error message>
```

**Why:** The model has the tool name, args, and ID right there in the tool result message.
When the completion arrives, the match is explicit — no inference from context needed.
Cleaner as conversation history grows. Also handles failures gracefully with `FAILED`.

---

## Thread Error Handling

If a background thread throws an exception, the old code would silently hang — `pending_tools`
would never get cleaned up and the model would keep saying "still pending" forever.

**Fix:** thread bodies are wrapped in `try/except`. On failure, an error tuple goes into
`results_queue` just like a success, and the injection formats it as `Job X FAILED: ...`
so the model can tell the user something went wrong.

---

## Race Condition & The Lock

`app.py` has two concurrent paths that both write to `messages` and call OpenAI:
- `process_user_message` — triggered when the user submits a message
- `check_and_inject` — triggered by `gr.Timer` every 0.5s

If a slow tool finishes exactly when the user sends a new message, both run simultaneously.
Without protection: interleaved writes to `messages`, two OpenAI calls with inconsistent
state, malformed conversation history.

**Fix:** `threading.Lock` (`_lock`) that both functions acquire before touching `messages`
or calling OpenAI. The queue drain (`collect_finished_results`) is outside the lock since
`Queue` is already thread-safe.

**Why not just use Gradio's queue for this:** `demo.queue()` serialises Gradio event
handlers (fixing the chatbot display state race), but background tool threads are not
Gradio events — they live outside Gradio's control. The `_lock` is still needed to protect
`messages` from those threads' injections.

---

## Gradio-Specific Fixes

### Chatbot display state race (`demo.queue()`)

Gradio passes each event handler a snapshot of the chatbot state at call time. If two
handlers (timer + submit) get called nearly simultaneously, each receives the same stale
snapshot. Whichever returns last overwrites the other's update — a message silently disappears.

**Fix:** `demo.queue()` before `demo.launch()`. Gradio serialises all event handlers
through a queue, so they never run simultaneously from Gradio's perspective.

### User message appears immediately (generator handler)

Old `process_user_message` appended both the user bubble and the assistant response in
one shot — after OpenAI returned. So the user saw nothing for 5–10s, then both appeared
at once. Janky.

**Fix:** converted to a generator that yields twice:
```python
# yield 1: user bubble appears immediately, textbox clears
history = history + [{"role": "user", "content": user_text}]
yield history, ""

# ... OpenAI call happens here ...

# yield 2: assistant response appears
history = history + [{"role": "assistant", "content": bot_text}]
yield history, ""
```

`demo.queue()` is required for Gradio to stream generator yields to the browser.

---

---

## Tool Dependencies — `await_job`

### Motivation

The async system above handles a common case well: fire multiple independent slow tools in
parallel, inject results as they arrive. But there's a harder case — **when tool2's arguments
depend on tool1's output**.

Example: "find me flights from Tokyo to Amsterdam, then check couple activities there."
- `get_flights` returns flight data including the destination city
- `get_activities` needs that city as an argument

If the LLM calls both immediately, it has to guess the city for `get_activities` — wrong.
If it calls only `get_flights` and waits, it might forget to call `get_activities` later.
Neither is acceptable.

### The Mental Model: A Sticky Note

Think of `await_job` as the LLM leaving itself a **sticky note** on a pending job:

> "When job `a1b2c3d4` finishes, remind me to call `get_activities` with the city from the result."

When the job completes, the system reads the sticky note aloud as part of the injection
message. The LLM sees both the result and its own reminder, and immediately makes the
follow-up call with the now-available real arguments.

```
LLM fires get_flights   →  job_id = "a1b2c3d4"
LLM calls await_job("a1b2c3d4", "call get_activities with destination city, tag=couple")
                         ↑ sticky note stuck to job a1b2c3d4

~10s later, check_and_inject fires:

  (System) Job a1b2c3d4 completed: get_flights(...) → [flight data]
  (System) You had planned a follow-up after job a1b2c3d4:
           "call get_activities with destination city, tag=couple"
           Now that the result is available, call the appropriate tool(s).

LLM reads its own note + the actual result → calls get_activities(city="amsterdam", tag="couple")
```

The key insight: the LLM decides the argument mapping at injection time, not upfront. It sees
the real result and reasons about it fresh — no guessing, no templates, no extraction logic.

### Timing: When to Call `await_job`

`await_job` is not restricted to the moment the slow tool is fired. It can be called at
any point in the conversation **while the job is still pending**. Two cases:

| Situation | Action |
|-----------|--------|
| Job still running (no `(System) Job X completed` seen yet) | Call `await_job` with that job_id |
| Job already completed (result is in conversation history) | Call follow-up tool directly now |

The system returns an error if `await_job` references an unknown job_id, with a hint to
look for the result in the conversation and call the follow-up tool directly.

### Multi-Hop Chains (A → B → C)

Chains work naturally, one hop at a time. The key is that `check_and_inject` re-calls
OpenAI with **full conversation history**, so the LLM always sees the original user intent.

```
Turn 1 — user says "flights → lounges → menu":
  LLM fires get_flights           → job "aaa"
  LLM calls await_job("aaa", "call get_airline_lounge with airline from result")

Injection 1 — "aaa" completes:
  LLM sees: flight result + "you had planned to call get_airline_lounge"
  LLM also sees: original user request "flights → lounges → menu"
  LLM fires get_airline_lounge(airline="ANA")  → job "bbb"
  LLM calls await_job("bbb", "call get_lounge_menu with lounge_id")
  ← both calls in the same response turn, handle_response processes them recursively

Injection 2 — "bbb" completes:
  LLM calls get_lounge_menu(lounge_id="...")   ← chain complete
```

The LLM doesn't need to know the full chain upfront. It registers the *next* hop each time
it fires an intermediate tool, using the conversation history to know what comes after.
The system prompt instructs this explicitly: "also register await_job for the next dependency
in the same response turn."

### What `await_job` Does NOT Do

- It does not extract values from results automatically — the LLM does that reasoning at injection time
- It cannot pre-register a full chain in one shot (future job_ids don't exist yet)
- It does not guarantee the LLM will always use it — if the LLM forgets, the result still
  arrives via the normal injection path and the LLM can make the follow-up call then

---

## Observability — Structured Logging

Both `app.py` and `main.py` emit structured log lines to the terminal. Root-level loggers
(gradio, httpx, openai) are silenced at WARNING; only the `async_tools` logger runs at DEBUG.

Key tags to watch:

| Tag | Meaning |
|-----|---------|
| `TOOL START` | Slow tool fired, background thread launched |
| `BG DONE / BG FAILED` | Background thread finished |
| `AWAIT_JOB registered` | Deferred hint stored |
| `AWAIT_JOB REJECTED` | Invalid job_id — reason + active jobs shown |
| `TIMER` | check_and_inject fired (DEBUG = nothing ready, INFO = results injected) |
| `HINT FIRE` | Deferred hint included in injection message |
| `OPENAI CALL / RESP` | API round-trip with message count and tool_calls list |
| `DISPATCH` | Tool routed as slow, instant, or await_job |
| `INJECTION` | Full message text sent to OpenAI at DEBUG level |
| `LOCK` | Lock acquired/released — useful for spotting contention |

---

## Files

| File | Purpose |
|---|---|
| `tools.py` | Tool implementations, `TOOL_FUNCTIONS` registry, `SLOW_TOOLS` set, `await_job` schema |
| `app.py` | Gradio web UI (primary interface) |
| `main.py` | Terminal frontend (reference implementation) |
| `eval/` | Unit tests + LLM behaviour eval framework |

## Running

```bash
python app.py    # opens browser, prints a share URL with share=True
python main.py   # terminal version
```
