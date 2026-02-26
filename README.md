# Async Tool-Calling Demo

A travel assistant chatbot that demonstrates **async LLM tool calling** — slow tools run in the background while the model continues the conversation, injecting results when ready.

See [claude_notes/README.md](claude_notes/README.md) for a deep-dive on the architecture.

## How It Works

Tools are classified as **slow** or **instant** at definition time:

- **Slow tools** (`get_hotels`, `get_flights`, `get_activities`): return a job ID immediately, run in a background thread, inject `(System) Job X completed: ...` when done
- **Instant tools** (`get_weather`): run inline and return results normally

The system prompt teaches the model this protocol — it acknowledges in-flight jobs, asks follow-up questions while tools run, and synthesizes results with context when they arrive.

## Running

```bash
python app.py    # Gradio web UI
python main.py   # Terminal version
```

Requires `OPENAI_API_KEY` in `.env`.

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
