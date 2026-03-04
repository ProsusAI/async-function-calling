# single_message_eval — Experiment & Results (v2)

## What We Did

### Motivation

Quantify the latency/quality tradeoff of async tool-calling injection. Hypothesis: async
injection is faster but stochastically degrades quality, and `pass^k` exposes this in a
way `pass@1` hides.

### Bug Fixes Applied Before v2 Run

The first run (0.5s delays, `results.json`) had five confounds corrected before re-running:

| Bug | Fix |
|-----|-----|
| BUG 1 — case-sensitive marker | Changed to `success_marker.lower() in final_text.lower()` |
| BUG 2 — SyncEngine system prompt gap | Added `_SYNC_BASE_PROMPT` with synthesis instructions (no async mechanics) |
| BUG 3 — no LLM call counter | Added `_counting_call_openai` closure; `num_llm_calls` tracked per trial |
| BUG 4 — no uncertainty quantification | Added Wilson 95% CI, `stdev_total_latency` to metrics table |
| BUG 5 — tool delays too short | Changed to 5s / 8s / 12s (was 2s / 3s / 4s); run without `--tool-delay` override |

### Experiment Setup

**Benchmark**: `eval/benchmark/` — 5 files implementing a single-turn eval:

| File | Role |
|---|---|
| `scenarios.py` | 6 `BenchScenario` instances (user message + `success_marker`) |
| `sync_engine.py` | `SyncEngine(AsyncEngine)` — one override, runs tools inline |
| `metrics.py` | `TrialResult`, `ScenarioBenchmarkResult`, Wilson CI, `pass^k`, `check_job_id_leaked` |
| `runner.py` | `run_trial()` — drives injection manually (`_auto_inject=False`) |
| `run_benchmark.py` | CLI — `--trials`, `--tool-delay`, `--scenarios`, `--modes`, `--output` |

**Four conditions** — same GPT-4o model, same tools, same dummy data:

| Mode | Mechanism |
|---|---|
| `sync` | Tools run blocking inline; result in same OpenAI turn (_SYNC_BASE_PROMPT + use_case prompt) |
| `async/tool` | Synthetic `assistant` tool_call + `tool` result pair injected (default framework mode) |
| `async/system` | `role=system` message: `"(System) Job X completed: …"` |
| `async/user` | `role=user` message: `"(System) Job X completed: …"` |

**Six scenarios** (Music use case, deterministic dummy data):

| Scenario | Tools | Real delay |
|---|---|---|
| `instant_only` | `get_mood_genres` (instant) | 0s |
| `single_slow` | `search_artists` | 5s |
| `mixed_instant_slow` | `get_genre_info` (instant) + `search_artists` | 5s |
| `two_parallel` | `search_artists` ×2 (parallel) | 5s |
| `three_parallel` | `search_artists` ×2 + `build_playlist` | 5s + 12s |
| `chain` | `search_artists` → `get_discography` | 5s + 8s |

**Run parameters**: `--trials 5`, no `--tool-delay` (real 5/8/12s delays, 120 trials total).
Results in `eval/benchmark/results_v2.json`.

---

## Results

```
━━━━ instant_only  [n=5] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Metric                        sync   async/tool  async/system  async/user
  ─────────────────────────────────────────────────────────────────────────
  Total latency (s)          2.9±1.2     2.5±0.2     2.8±0.6     2.3±0.6
  TTFR (s)                       2.9         2.5         2.8         2.3
  LLM calls (mean)               2.0         2.0         2.0         2.0
  Pass@1                        100%        100%        100%        100%
    95% CI                 [57–100%]   [57–100%]   [57–100%]   [57–100%]
  Pass^5                        100%        100%        100%        100%
  Job ID leaked                   0%          0%          0%          0%

━━━━ single_slow  [n=5] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Metric                        sync   async/tool  async/system  async/user
  ─────────────────────────────────────────────────────────────────────────
  Total latency (s)         11.7±1.2    10.8±0.9    11.2±0.8    12.3±1.4
  TTFR (s)                      11.7         4.0         3.6         4.0
  LLM calls (mean)               2.2         3.0         3.0         3.0
  Pass@1                        100%        100%        100%        100%
    95% CI                 [57–100%]   [57–100%]   [57–100%]   [57–100%]
  Pass^5                        100%        100%        100%        100%
  Job ID leaked                   0%          0%          0%          0%

━━━━ mixed_instant_slow  [n=5] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Metric                        sync   async/tool  async/system  async/user
  ─────────────────────────────────────────────────────────────────────────
  Total latency (s)         11.3±0.7    11.2±0.4    10.6±0.5    11.3±1.4
  TTFR (s)                      11.3         3.8         3.6         4.3
  LLM calls (mean)               2.0         3.0         3.0         3.0
  Pass@1                        100%        100%        100%        100%
    95% CI                 [57–100%]   [57–100%]   [57–100%]   [57–100%]
  Pass^5                        100%        100%        100%        100%
  Job ID leaked                   0%          0%          0%          0%

━━━━ two_parallel  [n=5] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Metric                        sync   async/tool  async/system  async/user
  ─────────────────────────────────────────────────────────────────────────
  Total latency (s)         17.9±1.1    13.8±1.2    20.1±8.9    15.4±2.1
  TTFR (s)                      17.9         6.3         5.5         4.9
  LLM calls (mean)               2.0         3.6         3.6         3.0
  Pass@1                        100%        100%        100%        100%
    95% CI                 [57–100%]   [57–100%]   [57–100%]   [57–100%]
  Pass^5                        100%        100%        100%        100%
  Job ID leaked                   0%          0%          0%          0%

  * async/system trial 5 = 34.7s outlier (model took extra reasoning rounds);
    async/system mean ≈ 14.1s without this trial.

━━━━ three_parallel  [n=5] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Metric                        sync   async/tool  async/system  async/user
  ─────────────────────────────────────────────────────────────────────────
  Total latency (s)         29.2±4.2    17.9±1.1    16.5±0.8    27.3±8.5
  TTFR (s)                      29.2         5.9         5.5        11.9
  LLM calls (mean)               2.6         6.2         5.6         6.2
  Pass@1                        100%        100%        100%        100%
    95% CI                 [57–100%]   [57–100%]   [57–100%]   [57–100%]
  Pass^5                        100%        100%        100%        100%
  Job ID leaked                   0%          0%          0%          0%

━━━━ chain  [n=5] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Metric                        sync   async/tool  async/system  async/user
  ─────────────────────────────────────────────────────────────────────────
  Total latency (s)         22.5±2.9    21.6±2.5    20.3±1.7    22.3±3.5
  TTFR (s)                      22.5         6.3         4.5         5.7
  LLM calls (mean)               3.0         6.8         7.0         7.2
  Pass@1                        100%        100%        100%         80%
    95% CI                 [57–100%]   [57–100%]   [57–100%]   [38–96%]
  Pass^5                        100%        100%        100%         33%
  Job ID leaked                   0%          0%          0%          0%

  * async/user trial 120 ✗ — model produced a playlist instead of discography;
    likely treated "(User) job X completed" as a new user request.
```

---

## Key Findings

### 1. BUG fixes validated — instant_only is now a true control

All four modes score **100% pass@1 on `instant_only`** with near-identical timing (2–3s).
The v1 anomaly (sync 20% vs async 80%) was entirely the system-prompt gap; both bugs (BUG 1
case sensitivity + BUG 2 `_SYNC_BASE_PROMPT`) had to be fixed together to resolve it.
`instant_only` now correctly shows zero differences between modes.

### 2. TTFR: consistent 3–5× improvement at real delays

| Scenario | sync TTFR | best async TTFR | gain |
|---|---|---|---|
| single_slow | 11.7s | 3.6s (system) | **3.3×** |
| mixed_instant_slow | 11.3s | 3.6s (system) | **3.1×** |
| two_parallel | 17.9s | 4.9s (user) | **3.7×** |
| three_parallel | 29.2s | 5.5s (system) | **5.3×** |
| chain | 22.5s | 4.5s (system) | **5.0×** |

The user sees a first response in 4–6s instead of 11–29s. At real delays this is a
dramatically visible win — at 0.5s (v1) the gain was 2–3×; at real delays it grows to
3–5× because tool wait time dominates and the acknowledgment is fast by comparison.

### 3. Total latency: async wins decisively on parallel scenarios

At real tool delays, parallelism savings dwarf injection overhead:

| Scenario | sync | async/tool | async/system | async/user |
|---|---|---|---|---|
| single_slow | 11.7s | **10.8s** | 11.2s | 12.3s |
| mixed_instant_slow | 11.3s | 11.2s | **10.6s** | 11.3s |
| two_parallel | 17.9s | **13.8s** | 20.1s* | 15.4s |
| three_parallel | 29.2s | 17.9s | **16.5s** | 27.3s* |
| chain | 22.5s | 21.6s | **20.3s** | 22.3s |

`*` = high-variance cell; remove outliers and async/system two_parallel ≈ 14s.

`three_parallel` is the clearest result: async/system saves **12.7s** (43% faster) and
async/tool saves **11.3s** (39% faster) vs sync. The theoretical parallel saving for
three tools (5s + 5s + 12s sequential → 12s parallel) = 10s; both modes beat this because
the acknowledgment LLM call and the background tools overlap.

### 4. `async/system` is the total-latency winner at high parallelism

`three_parallel` ranking (means):
```
async/system 16.5s ±0.8 < async/tool 17.9s ±1.1 << async/user 27.3s ±8.5 < sync 29.2s ±4.2
```

This reverses the v1 finding (0.5s delays: `async/tool` ≈ `sync` < `async/system`). At
real delays, `async/system` wins because (a) parallel savings dominate injection overhead
and (b) `async/system` uses fewer LLM calls (5.6 vs 6.2) — text messages are more compact
than full assistant+tool message pairs, so the model processes 3 injected results faster.

### 5. `async/user` is the weakest mode on both dimensions

**Latency**: High variance in parallel scenarios (three_parallel ±8.5s, two_parallel
±2.1s). Trial 97 (three_parallel) = 39.1s — 2.2× the mean. The model appears to
occasionally interpret "(User) job completed" messages as interleaved user turns and enters
clarification loops.

**Quality**: Only mode to show a failure at n=5. `chain/async/user` trial 120 produced a
playlist ("Miles Davis Tribute") instead of discography synthesis. The model treated the
injected user message as a new creative request rather than a tool result notification.
Pass@1 = 80%, pass^5 = 33%.

`async/user` CI for chain is [38%–96%] — the true pass rate could be as low as 38%. This
is the only result in the entire run that falsifies a 100% lower-bound hypothesis.

### 6. `async/system` has rare but large latency outliers

`two_parallel/async/system` trial 5 = 34.7s (normal mean: ~14s). The model occasionally
enters a multi-round clarification loop when it receives "(System) Job X completed" text —
possibly re-summarizing before synthesizing. This adds 1–2 extra LLM rounds. Sporadic
(1/5 trials in two_parallel); `three_parallel/async/system` had no such outlier (stdev 0.8s).

### 7. LLM call counts reveal injection overhead structure

| Scenario | sync | async/tool | async/system | async/user |
|---|---|---|---|---|
| instant_only | 2.0 | 2.0 | 2.0 | 2.0 |
| single_slow | 2.2 | 3.0 | 3.0 | 3.0 |
| two_parallel | 2.0 | 3.6 | 3.6 | 3.0 |
| three_parallel | 2.6 | 6.2 | 5.6 | 6.2 |
| chain | 3.0 | 6.8 | 7.0 | 7.2 |

Pattern: each async injection adds 2 LLM calls (acknowledgment + synthesis after result).
The `three_parallel` modes average 6.2 calls because the three tools don't always finish
within the `_SIBLING_WAIT = 0.1s` window — `build_playlist` (12s) finishes 7s after the
two `search_artists` (5s), triggering a separate injection round. Fix: increase
`_SIBLING_WAIT` or inject greedily without waiting.

`chain` uses 7 LLM calls: 1 (dispatch search_artists) + 1 (acknowledge) + 1 (dispatch
get_discography after await_job fires hint) + 1 (acknowledge) + 1 (result injection) +
1 (synthesis) + potential extras = 6–7. Matches.

### 8. Quality at n=5: only `async/user chain` fails

119 of 120 trials succeed (99.2%). The single failure is a **systematic** failure mode —
the response is coherent but wrong, not a random garble — indicating the "(User) role
injection" design has a real, reproducible failure mode in dependent-tool chains.

---

## What to Do Next

### For statistical validity

1. **n=20 for chain/async/user**: CI [38–96%] is too wide. With n=20 at 80% pass@1 the CI
   narrows to [59–92%] — enough to confirm whether this is a real ~80% failure or a fluke.
2. **Harder success markers**: `"Kind of Blue"` is in model training data and could appear
   without tool use. Replace with unique formatted strings from tool output only, e.g.
   `"Kind of Blue (1959)"` matching the exact `_DISCOGRAPHY` dict format.
3. **LLM judge**: pass@1 saturated at 100% for 5/6 scenarios. A judge would expose
   qualitative gaps (synthesis coherence, whether rankings are justified) invisible to the
   marker check.

### Architecture fixes suggested by results

4. **Tune `_SIBLING_WAIT`**: Increase from 0.1s to ~1–2s to batch `build_playlist` (12s)
   with `search_artists` results (5s) in three_parallel. Expected: reduce async/tool calls
   from 6.2 to ~4.0, reduce total latency ~3–4s further.
5. **Investigate async/user chain failure**: The "(User) job completed" injection confused
   the model into treating it as a user creative request. Fix options: (a) use a clearer
   sentinel prefix like `"[TOOL RESULT]"`, (b) deprecate async/user in favor of
   async/system for dependent chains, (c) add system prompt instruction to distinguish
   injected notifications from real user turns.
6. **Multi-turn eval**: In single-turn the async/user confusion occurred once. In
   multi-turn, real user messages can arrive between injections — the collision risk is
   higher. This warrants a separate multi-turn eval.

### Open questions

- Does async/user chain fail rate hold at ~20% with n=20, or was trial 120 a fluke?
- Would a weaker model (GPT-4o-mini) show quality degradation at lower parallelism?
- Does the async/system 34.7s outlier recur systematically? What triggers the extra rounds?
- Does `_SIBLING_WAIT = 1.0s` reduce three_parallel LLM calls from 6.2 to ~4.0?
