# Async Tool-Calling Demo

A travel assistant chatbot that demonstrates **async LLM tool calling** — slow tools run in the background while the model continues the conversation, and results are pushed to the browser via SSE when ready.

## Architecture

```
tools.py          ← tool implementations + OpenAI schemas (no web framework)
server.py         ← FastAPI backend: OpenAI loop + async dispatch + SSE push
static/index.html ← browser UI: fetch + EventSource, vanilla JS
experiments/      ← standalone scripts used to validate API behaviour
```

### Tool classification

Tools are classified as **slow** or **instant** at definition time (`tools.py`):

| Type | Tools | Behaviour |
|------|-------|-----------|
| Instant | `get_weather` | Runs inline; result returned synchronously in the same OpenAI turn |
| Slow | `get_hotels`, `get_flights`, `get_activities` | Dispatched to a background thread; returns `{"job_id": ..., "status": "started"}` immediately |

### Request flow

```
Browser POST /chat
  → acquire _lock
  → append user message
  → call OpenAI  (LLM may dispatch slow tools → background threads start)
  → handle_response() recurses until no tool calls remain
  → release _lock
  → push_event("assistant", ...) → SSE → browser renders bubble

Background thread finishes
  → results_queue.put(...)
  → spawn _run_injection thread
    → acquire _lock
    → drain queue
    → _inject_finished()   ← injection mode applied here
    → call OpenAI
    → handle_response()
    → release _lock
  → push_event("assistant", ...) → SSE → browser renders new bubble
```

### SSE (Server-Sent Events)

The browser opens a single persistent `GET /stream` connection at page load. The server keeps it alive and writes `data: {...}\n\n` whenever anything happens. The browser's built-in `EventSource` API handles reconnection automatically. No polling, no timers.

---

## Injection modes

When a background job completes, the result must be fed back into the LLM's message history. Three strategies are supported, selectable at startup via `--injection-mode`.

### `user` — role=user message *(original, BUG-4)*

```python
{"role": "user", "content": "(System) Job abc123 completed: get_hotels(...) → Hotels: ..."}
```

The LLM sees job completions as if the user typed them. This causes confusion when a real user message arrives in the same turn — the LLM can lose track of which text is user speech vs. system data.

### `system` — role=system message

```python
{"role": "system", "content": "(System) Job abc123 completed: get_hotels(...) → Hotels: ..."}
```

Semantically correct. Verified experimentally: mid-conversation `system` messages are accepted by the API, and the original system prompt's constraints are still honoured — the new message adds data without overriding existing instructions.

### `tool` — synthetic tool call + result pair *(default)*

For each completed job, two messages are appended:

```python
# 1. Synthetic assistant message that "called" the tool
{"role": "assistant", "content": None, "tool_calls": [{
    "id": "call_a1b2c3d4",
    "type": "function",
    "function": {"name": "get_hotels", "arguments": '{"city": "amsterdam"}'}
}]}

# 2. Paired tool result
{"role": "tool", "tool_call_id": "call_a1b2c3d4", "content": "Hotels in Amsterdam: ..."}
```

The LLM reads this exactly like a synchronous tool call — no role confusion, no `(System)` prefix, no special instructions in the system prompt needed. The LLM is trained on this shape.

**Dependent tool chaining in `tool` mode:**
When the LLM dispatches a slow tool, it can register a follow-up intent with `await_job`:

```
LLM fires get_flights(tokyo, amsterdam)  → job_id = "abc123"
LLM calls await_job(job_id="abc123", followup_hint="call get_hotels(city=amsterdam)")
  → stored in deferred_hints["abc123"]
```

When the job completes, `_inject_finished` appends the synthetic tool pair **plus** a `system` reminder:

```
[assistant] tool_call: get_flights({"origin": "tokyo", "destination": "amsterdam"})
[tool]      Flights from Tokyo to Amsterdam: KLM $680 nonstop, ...
[system]    You had planned a follow-up: "call get_hotels(city=amsterdam)".
            The result is now available above — call the appropriate tool(s).
```

OpenAI is then called. The LLM sees its completed result alongside its own earlier intent and immediately calls `get_hotels`. Without the hint the LLM would have to re-derive intent from context, which is unreliable across long conversations.

---

## Running

```bash
# Default (tool mode — recommended)
python server.py

# Choose injection mode explicitly
python server.py --injection-mode tool    # native tool call/result pair
python server.py --injection-mode system  # role=system message
python server.py --injection-mode user    # role=user message (original BUG-4 behaviour)

# Help
python server.py --help
```

Requires `OPENAI_API_KEY` in `.env`. Server listens on `http://0.0.0.0:7862`.

---

## Experiments

Standalone scripts in `experiments/` that validate API behaviour without the full server:

| Script | What it tests |
|--------|--------------|
| `multi_user_msg_test.py` | Does the API accept consecutive `user` messages? How does the LLM handle injection-as-user-role? |
| `multi_system_msg_test.py` | Does a mid-conversation `system` message override the original system prompt? |
| `synthetic_tool_msg_test.py` | Do synthetic tool call + result pairs work? Does the LLM avoid re-calling already-resolved tools? |

---

## Evaluation

Two evaluation tracks live in [`eval/`](eval/):

### Track 1 — Infrastructure Tests

Unit and integration tests for the async machinery (no LLM calls, runs in ~0.2s):

```bash
uv run pytest eval/ -v
```

61 tests covering:

| File | What it tests |
|------|--------------|
| `test_fire_tool_async.py` | Job JSON format, 8-char hex IDs, `pending_tools` registered before thread finishes, 10 concurrent calls all distinct |
| `test_routing.py` | `SLOW_TOOLS` membership, slow tools → `fire_tool_async`, instant tools → inline, tool message format |
| `test_queue_mechanics.py` | 5-tuple deposit structure, multi-thread deposits, `collect_finished_results` drains queue |
| `test_error_handling.py` | Exceptions → FAILED queue entry (no silent hangs), FAILED message format |
| `test_state_management.py` | `pending_tools` lifecycle, `_lock` prevents concurrent message corruption |
| `test_check_and_inject.py` | Completion/failure message format, still-pending line, history update |

### Track 2 — LLM Behavior Evaluation

Scripted conversations that check whether the model follows the async protocol:

```bash
uv run python eval/run_llm_eval.py
uv run python eval/run_llm_eval.py --scenario flights_basic
uv run python eval/run_llm_eval.py --output results.json
```

Requires `OPENAI_API_KEY` + `ANTHROPIC_API_KEY` (Claude used as judge).

**Scenarios**: `flights_basic`, `hotels_basic`, `result_synthesis`, `parallel_tools`, `instant_tool`, `error_injection`

**Criteria checked per turn**:
- No job ID leaked to user (deterministic regex)
- Acknowledgment present when slow tool fires (keyword check)
- Follow-up question asked (regex)
- No `(System)` text echoed back (string check)
- Synthesis quality, context-awareness, follow-up relevance (LLM judge, 0–5)
