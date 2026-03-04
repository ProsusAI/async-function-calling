# Async Tool-Calling Framework

A reusable framework for **async LLM tool calling** — slow tools run in background threads while the model continues the conversation, and results are pushed to the browser via SSE when ready. Supports single-agent and **multi-agent** setups where agents delegate to specialist sub-agents.

## Structure

```
core/
│   schema.py          Tool and UseCase dataclasses (plugin contract)
│   engine.py          AsyncEngine: OpenAI loop, async dispatch, SSE, sub-agent support
│   prompts.py         Base system prompts (async mechanics, mode-specific)
│   await_job.py       await_job tool schema (framework-owned)
│   return_answer.py   return_answer_to_parent tool schema (framework-owned, sub-agents only)
│   agent_tool.py      AgentTool: wraps a UseCase as a callable tool for orchestrators
│   __init__.py

use_cases/
├── travel/            Travel assistant (flights, hotels, activities)
├── music/             Music discovery (artists, genres, playlists)
└── multi/             Multi-agent demo: orchestrator → travel + music sub-agents

server.py              Thin FastAPI wiring (~150 lines)
static/index.html      Browser UI: fetch + EventSource, vanilla JS
eval/                  Infrastructure tests + LLM behaviour eval + async/sync benchmark
experiments/           Standalone scripts for validating API behaviour
```

## Running

```bash
# Travel assistant (default)
uv run server.py

# Music discovery
uv run server.py --use-case music

# Multi-agent demo (orchestrator → travel + music sub-agents in parallel)
uv run server.py --use-case multi

# Choose injection mode for background job results
uv run server.py --injection-mode tool    # synthetic tool call/result pair (default)
uv run server.py --injection-mode system  # role=system message
uv run server.py --injection-mode user    # role=user message
```

Requires `OPENAI_API_KEY` in `.env`. Server listens on `http://0.0.0.0:7862`.

---

## Adding a single-agent use case

Create `use_cases/<domain>/` with:

**`tools.py`** — tool implementations:
```python
from core.schema import Tool

def _get_hotels(args: dict) -> str:
    city = args["city"]
    return f"Hotels in {city}: ..."

get_hotels = Tool(
    name="get_hotels",
    description="Find hotels in a city.",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    fn=_get_hotels,
    is_async=True,   # slow tool — runs in background thread
)

get_weather = Tool(
    name="get_weather",
    description="Get current weather.",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    fn=lambda args: f"Weather in {args['city']}: sunny",
    is_async=False,  # instant — runs inline
)
```

**`prompt.py`** — domain-specific system prompt fragment:
```python
SYSTEM_PROMPT = "You are a travel assistant. ..."
```

**`__init__.py`** — wire it together:
```python
from core.schema import UseCase
from .tools import get_hotels, get_weather
from .prompt import SYSTEM_PROMPT

MyUseCase = UseCase(
    display_name="My Assistant",
    input_placeholder="Ask me anything…",
    system_prompt=SYSTEM_PROMPT,
    tools=[get_hotels, get_weather],
)
```

Then pass `MyUseCase` to `AsyncEngine` in `server.py`. Zero changes to `core/`.

---

## Multi-agent setup

### How it works

An **orchestrator agent** calls specialist **sub-agents** as tools. Each sub-agent runs its own full `AsyncEngine` loop — with its own queue, lock, and background threads — completely isolated from the parent.

The sub-agent signals completion by calling `return_answer_to_parent`, a framework-owned tool automatically added to every sub-agent's tool list. The orchestrator's `AgentTool` blocks on a `threading.Event` until this is called, then injects the answer into the parent's conversation.

```
Orchestrator (AsyncEngine)
  │
  ├── calls music_agent(query="jazz for Amsterdam")   → is_async=True → BG thread
  ├── calls travel_agent(query="trip to Amsterdam")   → is_async=True → BG thread
  │                      │                                      │
  │           music sub-agent runs                  travel sub-agent runs
  │           its own ReACT loop                    its own ReACT loop
  │           fires search_artists ──────────────── fires get_hotels
  │           fires build_playlist   (parallel)     fires get_flights
  │           calls return_answer_to_parent(...)     calls return_answer_to_parent(...)
  │                      │                                      │
  │           done_event.set()                      done_event.set()
  │                      │                                      │
  ├── parent results_queue ←─────────────────────────────────── ┘
  └── _run_injection fires → parent synthesizes → SSE push to browser
```

### `AgentTool` — wrapping a use case as a tool

```python
from core.agent_tool import AgentTool
from use_cases.music import MusicUseCase

AgentTool(
    name="music_agent",
    description="Music specialist: recommendations, playlists, artists, genres, moods.",
    use_case=MusicUseCase,
    is_async=True,      # how the parent calls this agent (True = non-blocking, parallel)
    forced_sync=False,  # how this agent runs its own tools (False = internal parallelism)
    max_steps=20,       # max OpenAI call rounds before giving up
)
```

**`is_async` and `forced_sync` are orthogonal:**

| `is_async` | `forced_sync` | Meaning |
|-----------|--------------|---------|
| `True` | `False` | Sub-agent fires in parent background thread; sub-agent's own tools run in parallel. Best performance. |
| `True` | `True` | Sub-agent fires in parent background thread; sub-agent runs its own tools sequentially. |
| `False` | `False` | Parent blocks until sub-agent finishes; sub-agent's tools run in parallel. |
| `False` | `True` | Fully sequential end-to-end. Equivalent to old `SyncEngine` behaviour. |

### `return_answer_to_parent` — the sub-agent's exit signal

This framework-owned tool is automatically added to every sub-agent's tool list. Sub-agents **must** call it to return their answer — the parent gets a timeout error string if the sub-agent exhausts `max_steps` without calling it.

The sub-agent's system prompt is automatically prepended with:
> *"You are a specialist sub-agent. Do NOT ask the user for clarification. When your task is complete, you MUST call `return_answer_to_parent`. If you have background jobs still running, do NOT call it yet — wait for those results to arrive and include them in your final answer."*

**The framework enforces this at the engine level too.** If a sub-agent tries to call `return_answer_to_parent` (or exits naturally) while it still has pending background jobs, the call is rejected and the model is told to wait. Only when `pending_tools` is empty can a sub-agent successfully return. This prevents the sub-agent from prematurely returning a "looking for flights…" stub before the actual flight data arrives.

### Building a multi-agent use case

```python
from core.schema import UseCase
from core.agent_tool import AgentTool
from use_cases.music import MusicUseCase
from use_cases.travel import TravelUseCase

MultiUseCase = UseCase(
    display_name="Multi-Agent Demo",
    input_placeholder="e.g. Plan a jazz-themed trip to Amsterdam",
    system_prompt="You are a coordinator. Delegate to specialists. Synthesize their answers.",
    tools=[
        AgentTool("music_agent", "Music specialist.", MusicUseCase, is_async=True),
        AgentTool("travel_agent", "Travel specialist.", TravelUseCase, is_async=True),
    ],
)
```

**Orchestrator prompt discipline — domain boundaries matter.** The orchestrator LLM decides which specialist to call based on the agent descriptions in your `system_prompt`. Overlapping descriptions cause misrouting — e.g. if `travel_agent` is described as handling "activities", a "jazz activities" query will go there instead of `music_agent`. Be explicit and non-overlapping:

```
music_agent:  ALL music content — artists, playlists, concerts, jazz events, venues.
travel_agent: logistics ONLY — flights, hotels, weather. NOT music events.
```

Include concrete routing examples in the prompt for cross-domain queries:
```
"jazz trip to Amsterdam" → call BOTH: music_agent (jazz venues) AND travel_agent (flights + hotels).
Call each agent at most once.
```

Run it:
```bash
uv run server.py --use-case multi
```

### `forced_sync` on the parent

`forced_sync` also works on the parent engine directly — useful for testing or when you need deterministic sequential execution:

```python
# All tools (including AgentTools) run inline; no background threads
engine = AsyncEngine(use_case, forced_sync=True)
```

---

## Hooks

`UseCase` accepts an optional `hooks` parameter for observing and modifying tool calls without touching `core/`.

```python
from core import Hooks, UseCase

MyUseCase = UseCase(
    ...,
    hooks=Hooks(
        before_tool=lambda name, args: args,       # inspect / modify args, or return None to cancel
        after_tool=lambda name, args, result: result,  # inspect / modify result
    ),
)
```

### `before_tool(name, args) → dict | None`

Called before every tool invocation (sync and async). Two outcomes:

| Return value | Effect |
|---|---|
| `dict` | Tool is called with the returned args (may be the original or modified) |
| `None` | Tool call is cancelled; LLM receives `{"status": "cancelled"}` as the tool result |

```python
# Log and add a default arg
def before(name, args):
    print(f"→ {name}({args})")
    if name == "get_hotels" and "stars" not in args:
        args = {**args, "stars": 4}
    return args

# Block a specific tool
def before(name, args):
    if name == "get_flights":
        return None   # cancelled
    return args
```

### `after_tool(name, args, result) → str`

Called after every tool completes — for sync tools immediately, for async tools once the background thread finishes (before the result is injected into the conversation).

```python
# Cache results by tool + args
_cache = {}

def after(name, args, result):
    key = (name, json.dumps(args, sort_keys=True))
    _cache[key] = result
    return result
```

Both hooks are **no-ops when not set** — existing use cases without a `hooks=` argument are unaffected.

---

## How async tool dispatch works

### Tool classification

Each `Tool` carries its own `is_async` flag:

| `is_async` | Behaviour |
|-----------|-----------|
| `False` | Runs inline; real result returned synchronously in the same OpenAI turn |
| `True` | Dispatched to a background thread; model gets `{"job_id": ..., "status": "started"}` immediately |

### Request flow

```
Browser POST /chat
  → acquire _lock
  → append user message
  → call OpenAI  (may dispatch async tools → background threads start)
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

```
BASE_SYSTEM_PROMPT[injection_mode]   ← async mechanics (framework-owned)
---
use_case.system_prompt               ← domain persona and tool descriptions
```

For sub-agents, the engine prepends a sub-agent preamble before the base prompt.

---

## Injection modes

When a background job completes, the result re-enters the LLM's message history. Three strategies are supported via `--injection-mode`.

### `tool` *(default)*

Two synthetic messages appended per completed job:

```python
{"role": "assistant", "content": None, "tool_calls": [{"id": "call_a1b2c3", ...}]}
{"role": "tool", "tool_call_id": "call_a1b2c3", "content": "Hotels in Amsterdam: ..."}
```

### `system`

```python
{"role": "system", "content": "(System) Job abc123 completed: get_hotels(...) → Hotels: ..."}
```

### `user`

```python
{"role": "user", "content": "(System) Job abc123 completed: get_hotels(...) → Hotels: ..."}
```

### `await_job` — dependent tool chaining

The LLM can register a follow-up intent before the result arrives:

```
LLM fires get_flights(tokyo, amsterdam)  → job_id = "abc123"
LLM calls await_job(job_id="abc123", followup_hint="call get_hotels(city=amsterdam)")
```

When the job completes, the hint is appended alongside the result. The LLM sees its earlier intent and immediately chains the next call.

---

## Evaluation

### Infrastructure tests (no LLM, ~0.2s)

```bash
uv run pytest eval/ -v
```

### LLM behaviour evaluation

```bash
uv run python eval/run_llm_eval.py
uv run python eval/run_llm_eval.py --scenario flights_basic
uv run python eval/run_llm_eval.py --output results.json
```

Requires `OPENAI_API_KEY` + `ANTHROPIC_API_KEY` (Claude used as judge).

### `single_message_eval` — async vs sync benchmark

```bash
# Quick run (0.5s tools, 3 trials)
uv run python eval/benchmark/run_benchmark.py --tool-delay 0.5 --trials 3

# Specific scenarios and modes
uv run python eval/benchmark/run_benchmark.py --scenarios two_parallel chain --modes sync async/tool

# Full run with JSON output
uv run python eval/benchmark/run_benchmark.py --trials 10 --output results.json
```

**Four conditions** — same LLM, same tools; only result re-injection differs:

| Mode | How results re-enter the model |
|------|-------------------------------|
| `sync` | Tool runs inline; real result returned in the same turn (`forced_sync=True`) |
| `async/tool` | Synthetic `assistant` tool_call + `tool` result pair injected |
| `async/system` | `role=system` message with job completion text |
| `async/user` | `role=user` message with `(System) Job X completed: …` |

---

## Experiments

Standalone scripts in `experiments/` that validate API behaviour without the full server:

| Script | What it tests |
|--------|--------------|
| `multi_user_msg_test.py` | Consecutive `user` messages; injection-as-user-role behaviour |
| `multi_system_msg_test.py` | Mid-conversation `system` message; original prompt still honoured? |
| `synthetic_tool_msg_test.py` | Synthetic tool pairs; LLM avoids re-calling resolved tools? |
