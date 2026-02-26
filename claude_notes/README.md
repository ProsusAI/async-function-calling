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

## Files

| File | Purpose |
|---|---|
| `tools.py` | Tool implementations, `TOOL_FUNCTIONS` registry, `SLOW_TOOLS` set |
| `app.py` | Gradio web UI (primary interface) |
| `main.py` | Terminal frontend (reference implementation) |

## Running

```bash
python app.py    # opens browser, prints a share URL with share=True
python main.py   # terminal version
```
