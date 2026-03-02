# Benchmark Design: Async vs Sync Tool Calling

## Hypothesis

Async tool calling is harder for the model than synchronous calling. Instead of seeing a
tool result inline, the model receives a job ID and must later make sense of an injected
result. This changed protocol adds protocol overhead and stochasticity.

**The core tradeoff:** async is faster (parallel tools) but may degrade quality and,
critically, *consistency*. The benchmark makes this tradeoff quantifiable.

The τ-bench `pass^k` metric (from Sierra's benchmark) reveals this clearly: a system at
80% pass@1 looks fine, but `pass^5 = 0.80^5 = 33%` — it fails two out of three 5-trial
runs. Async injection adds stochasticity; `pass^k` exposes whether this hurts reliability
in a way that `pass@1` alone hides.

---

## Four Conditions

| Condition | How results re-enter the model |
|---|---|
| **sync** | Tool runs inline; real result returned in the same OpenAI turn (traditional) |
| **async/tool** | Synthetic `assistant` tool_call + `tool` result pair injected (our default) |
| **async/system** | `role=system` message with job completion text injected |
| **async/user** | `role=user` message with `(System) Job X completed: ...` text injected |

All four use the same LLM model, same tools, same tool results. Only the injection
mechanism differs. `SyncEngine` is a minimal subclass of `AsyncEngine` with one override:

```python
class SyncEngine(AsyncEngine):
    def fire_tool_async(self, tool_name, tool_args) -> str:
        return self.use_case.tool_functions[tool_name](tool_args)  # real result, not job JSON
```

This is the entire behavioral difference. All OpenAI loop logic is identical.

---

## Three Metric Families

### 1. Latency

| Metric | Sync | Async |
|---|---|---|
| **Total latency** | Tool execution blocks model; TTFR = total latency | TTFR ≈ instant (acknowledgment); total ends at synthesis |
| **TTFR** | High (waits for every tool) | Near-zero (model talks while tools run) |

For N parallel slow tools: sync takes N×delay, async takes ~1×delay.

### 2. Quality — deterministic (no LLM needed)

**success_marker**: A fixed substring that MUST appear in the final assistant response.
Derived from the deterministic dummy data in `use_cases/music/tools.py` — same input →
same output → same expected string.

- Enables `pass@1` = fraction of trials where success_marker appears
- Enables `pass^k = pass@1 ^ k` — the consistency metric (borrowed from τ-bench)
- No LLM judge needed for this → fast, cheap, repeatable

**job_id_leaked**: Regex check on all assistant messages for the job ID format
(`[0-9a-f]{8}`). Leaked job IDs are a protocol failure — the model exposed internals.

### 3. Quality — LLM judge (optional, needs `ANTHROPIC_API_KEY`)

Synthesis quality scored 0–5 by Claude Sonnet, using the judge already in
`eval/run_llm_eval.py` lines 218–304. Criteria:
- `synthesis_quality`: does the response mention specific results with rationale?
- `context_awareness`: does it reference the user's stated preference/context?

---

## Six Scenarios (Music Use Case)

All are single-turn: one user message → tool calls → final synthesis. Single-turn design
isolates the injection mechanism from multi-turn accumulation effects.

### `instant_only` — Control

```
User: "What genres of music are best for a study session?"
Tools: get_mood_genres (instant)
success_marker: "Ambient"
```

No slow tools → no injection → sync and async should be identical. Verifies the benchmark
itself isn't introducing spurious differences.

---

### `single_slow` — Overhead Baseline

```
User: "Recommend some jazz artists for me."
Tools: search_artists (slow, 2s)
success_marker: "Miles Davis"
```

Measures the pure overhead of async injection for one tool. Expect small latency difference;
small quality gap. Reveals whether the acknowledgment-then-inject pattern hurts at all.

---

### `mixed_instant_slow` — Routing Correctness

```
User: "Tell me about jazz as a genre and recommend some artists."
Tools: get_genre_info (instant) + search_artists (slow)
success_marker: "Miles Davis"
```

`get_genre_info` runs inline in the same turn; `search_artists` goes async. Tests whether
the model correctly handles mixed routing and incorporates both results in synthesis.

---

### `two_parallel` — 2× Latency Advantage

```
User: "Compare jazz and electronic music artists. Best from each genre."
Tools: search_artists(jazz) + search_artists(electronic) — both slow, both parallel
success_marker: "Miles Davis"
```

Async runs both searches simultaneously (~1×delay). Sync must wait sequentially (~2×delay).
Also tests whether the model correctly synthesises two injected results — harder than one.

---

### `three_parallel` — 3× Latency Advantage

```
User: "Find me jazz artists, build a chill study playlist, and search for ambient artists."
Tools: search_artists + build_playlist + search_artists — three parallel
success_marker: "Miles Davis"
```

Maximum parallelism advantage. Also the hardest case for async quality: three injected
results must all be used. `pass^k` expected to diverge most from sync here.

---

### `chain` — Dependent Tool Chaining (await_job)

```
User: "Find me jazz artists. Then get the full discography for Miles Davis."
Tools: search_artists → (await_job) → get_discography(Miles Davis)
success_marker: "Kind of Blue"
```

Tests dependent chaining. `async/tool` mode (with await_job) should match sync quality
because the hint tells the model exactly what to do when search results arrive.
`async/user` and `async/system` are expected to be weaker — the injection text must
convey the follow-up intent clearly enough for the model to chain without explicit help.

---

## Implementation Sketch

### File structure (when implemented)

```
eval/benchmark/
├── __init__.py
├── scenarios.py         BenchScenario dataclass + 6 scenario instances
├── sync_engine.py       SyncEngine(AsyncEngine) — single fire_tool_async override
├── metrics.py           TrialResult, ScenarioBenchmarkResult, pass^k property
├── runner.py            run_trial(scenario, engine, mode) → TrialResult
└── run_benchmark.py     CLI: --trials, --tool-delay, --scenarios, --modes, --output
```

### Timing a trial

**Sync:**
```
t0 = now
engine.messages += [user_msg]
final_text = handle_response(call_openai())   # blocks until all tools done
total_latency = ttfr = now - t0
```

**Async:**
```
t0 = now
engine.messages += [user_msg]
ack_text = handle_response(call_openai())     # returns fast (acknowledgment)
ttfr = now - t0
poll engine.pending_tools until empty (max 30s)
engine.check_and_inject()                     # synthesis
total_latency = now - t0
```

### Controlled tool delay

Real tool delays (2–4s) make k=10 trials slow. A `--tool-delay N` flag patches
`use_cases.music.tools.time.sleep` to a fixed value N. This keeps relative timing
correct (2 parallel tools still take 1×N in async, 2×N in sync) without long runs.
`pass^k` is timing-independent — it only cares about task success.

### CLI (planned)

```bash
uv run python eval/benchmark/run_benchmark.py --trials 5 --tool-delay 1.0
uv run python eval/benchmark/run_benchmark.py --trials 10 --scenarios two_parallel chain
uv run python eval/benchmark/run_benchmark.py --modes sync async/tool --trials 5
uv run python eval/benchmark/run_benchmark.py --output results.json
```

### Output per scenario

```
━━ two_parallel: "Compare jazz and electronic artists"  [n=10] ━━━━━━━━━━━━

  Metric                    Sync      async/tool   async/system  async/user
  ──────────────────────────────────────────────────────────────────────────
  Total latency (s)          4.1       2.1  ↑2x     2.1  ↑2x     2.2  ↑1.9x
  Time to 1st resp (s)       4.1       0.4  ↑10x    0.4           0.4
  Pass@1                    100%        80%           60%           60%
  Pass^5                    100%        33%            8%            8%
  Job ID leaked               0%         0%            0%           20%
```

---

## Expected Findings

| Scenario | Latency winner | Quality winner | Key insight |
|---|---|---|---|
| `instant_only` | tied | tied | Control validates benchmark |
| `single_slow` | async (small) | sync (small) | Even 1 tool: slight quality cost |
| `mixed_instant_slow` | async (small) | sync | Mixed routing works but adds noise |
| `two_parallel` | async **2×** | sync (meaningful gap) | Pass^k diverges sharply |
| `three_parallel` | async **3×** | sync (large gap) | Maximum quality cost of async |
| `chain` | tied | async/tool ≈ sync | await_job closes the quality gap for chains |

The headline finding: **async/tool gives Nx latency savings for N parallel tools, but
`pass^k` reveals a reliability cost that `pass@1` understates. The chain scenario shows
that `await_job` can recover quality for dependent calls.**

---

## Related Work

- **τ-bench** (Sierra): `pass^k` metric, user simulator pattern, domain-grounded evaluation
- **BFCL v4** (Berkeley): parallel function calling evaluation, serial vs parallel distinction
- **ToolBench** (OpenBMB): multi-step tool execution, error recovery metrics
- Paper: *Asynchronous LLM Function Calling* (arXiv 2412.07017) — shows 1.6–5.4× latency
  improvement for async; does not measure quality degradation (this benchmark fills that gap)
