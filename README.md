# Async Tool-Calling Framework

A reusable framework for **async LLM tool calling** — slow tools run in background threads while the model continues the conversation, and results are pushed to the browser via SSE when ready. Ships with a travel assistant use case.

## Structure

```
core/                        ← reusable async tool-calling framework
│   schema.py                  UseCase dataclass (plugin contract)
│   engine.py                  AsyncEngine: OpenAI loop, async dispatch, SSE push
│   prompts.py                 Base system prompts (async mechanics, mode-specific)
│   await_job.py               await_job tool schema (framework-owned)
│   __init__.py

use_cases/
└── travel/                  ← travel assistant use case
        tools.py               tool implementations + OpenAI schemas
        prompt.py              travel-specific system prompt fragment
        __init__.py            TravelUseCase = UseCase(...)

server.py                    ← thin FastAPI wiring (~100 lines)
static/index.html            ← browser UI: fetch + EventSource, vanilla JS
experiments/                 ← standalone scripts for validating API behaviour
eval/                        ← 61 infrastructure tests + LLM behaviour eval
```

## Running

```bash
# Default (tool injection mode — recommended)
uv run server.py

# Choose injection mode explicitly
uv run server.py --injection-mode tool    # synthetic tool call/result pair (default)
uv run server.py --injection-mode system  # role=system message
uv run server.py --injection-mode user    # role=user message

# Help
uv run server.py --help
```

Requires `OPENAI_API_KEY` in `.env`. Server listens on `http://0.0.0.0:7862`.

## Adding a new use case

Create `use_cases/<domain>/` with three files:

**`tools.py`** — tool implementations and OpenAI schemas:
```python
SLOW_TOOLS = {"slow_tool_a", "slow_tool_b"}   # run in background threads
TOOL_FUNCTIONS = {"slow_tool_a": fn_a, "instant_tool": fn_b, ...}
TOOL_SCHEMAS = [...]  # OpenAI function schemas — do NOT include await_job
```

**`prompt.py`** — domain-specific system prompt fragment:
```python
SYSTEM_PROMPT = "You are a ... assistant. Available tools: ..."
```

**`__init__.py`** — wire it together:
```python
from core import UseCase
from .tools import SLOW_TOOLS, TOOL_FUNCTIONS, TOOL_SCHEMAS
from .prompt import SYSTEM_PROMPT

MyUseCase = UseCase(
    display_name="My Assistant",
    input_placeholder="Ask me anything…",
    system_prompt=SYSTEM_PROMPT,
    tool_schemas=TOOL_SCHEMAS,
    tool_functions=TOOL_FUNCTIONS,
    slow_tools=SLOW_TOOLS,
)
```

Then in `server.py`, swap `TravelUseCase` → `MyUseCase`. Zero changes to `core/`.

---

## How it works

### Tool classification

Tools are classified as **slow** or **instant** in the use case's `slow_tools` set:

| Type | Behaviour |
|------|-----------|
| Instant | Runs inline; result returned synchronously in the same OpenAI turn |
| Slow | Dispatched to a background thread; model gets `{"job_id": ..., "status": "started"}` immediately |

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
    → drain queue, inject results (mode-specific)
    → call OpenAI → handle_response()
    → release _lock
  → push_event("assistant", ...) → SSE → browser renders new bubble
```

### SSE

The browser opens a single persistent `GET /stream` connection at page load. The server writes `data: {...}\n\n` whenever anything happens. `EventSource` auto-reconnects. No polling, no timers.

### System prompt composition

The full system prompt is:

```
BASE_SYSTEM_PROMPT[injection_mode]   ← async mechanics (framework-owned)
---
use_case.system_prompt               ← domain persona, tools, heuristics
```

`BASE_SYSTEM_PROMPT` covers: job IDs, slow vs. instant distinction, how results arrive back, proactive synthesis, `await_job` chaining rules. The use case only adds domain knowledge.

---

## Injection modes

When a background job completes the result must re-enter the LLM's message history. Three strategies are supported via `--injection-mode`.

### `tool` *(default)*

Two synthetic messages appended per completed job:

```python
{"role": "assistant", "content": None, "tool_calls": [{"id": "call_a1b2c3", ...}]}
{"role": "tool", "tool_call_id": "call_a1b2c3", "content": "Hotels in Amsterdam: ..."}
```

The LLM reads this like a normal synchronous tool call — no role confusion, no special instructions needed.

### `system`

```python
{"role": "system", "content": "(System) Job abc123 completed: get_hotels(...) → Hotels: ..."}
```

Mid-conversation `system` messages are accepted by the API and don't override the original prompt.

### `user`

```python
{"role": "user", "content": "(System) Job abc123 completed: get_hotels(...) → Hotels: ..."}
```

Original behaviour. Can confuse the LLM when a real user message arrives in the same turn.

### `await_job` — dependent tool chaining

The LLM can register a follow-up intent before the result arrives:

```
LLM fires get_flights(tokyo, amsterdam)  → job_id = "abc123"
LLM calls await_job(job_id="abc123", followup_hint="call get_hotels(city=amsterdam)")
```

When the job completes, the hint is appended alongside the result as a `system` reminder. The LLM sees its earlier intent and immediately chains the next call.

---

## Evaluation

### Infrastructure tests (no LLM, ~0.2s)

```bash
uv run pytest eval/ -v
```

61 tests across 6 files:

| File | What it tests |
|------|--------------|
| `test_fire_tool_async.py` | Job JSON format, hex IDs, `pending_tools` timing, 10 concurrent calls |
| `test_routing.py` | `SLOW_TOOLS` membership, slow → async, instant → inline, message format |
| `test_queue_mechanics.py` | 5-tuple deposit, multi-thread, `collect_finished_results` drains queue |
| `test_error_handling.py` | Exceptions → FAILED entry (no silent hangs), FAILED message format |
| `test_state_management.py` | `pending_tools` lifecycle, `_lock` prevents concurrent corruption |
| `test_check_and_inject.py` | Completion/failure message format, still-pending line, history update |

### LLM behaviour evaluation

```bash
uv run python eval/run_llm_eval.py
uv run python eval/run_llm_eval.py --scenario flights_basic
uv run python eval/run_llm_eval.py --output results.json
```

Requires `OPENAI_API_KEY` + `ANTHROPIC_API_KEY` (Claude used as judge).

**Scenarios**: `flights_basic`, `hotels_basic`, `result_synthesis`, `parallel_tools`, `instant_tool`, `error_injection`

**Criteria**: no job ID leaked, acknowledgment present, follow-up asked, no `(System)` echoed, synthesis quality (LLM judge 0–5).

---

## Experiments

Standalone scripts in `experiments/` that validate API behaviour without the full server:

| Script | What it tests |
|--------|--------------|
| `multi_user_msg_test.py` | Consecutive `user` messages; injection-as-user-role behaviour |
| `multi_system_msg_test.py` | Mid-conversation `system` message; original prompt still honoured? |
| `synthetic_tool_msg_test.py` | Synthetic tool pairs; LLM avoids re-calling resolved tools? |
