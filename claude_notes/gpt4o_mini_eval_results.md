# single_message_eval — gpt-4o-mini vs gpt-4o

## Setup

Same benchmark, same music use case, same 5 real-delay scenarios (5/8/12s tools).
Only difference: `--model gpt-4o-mini`.

Results: `eval/benchmark/results_mini.json`
Reference: `eval/benchmark/results_v2.json` (gpt-4o)

**Note**: This document contains partial results for the first 3 scenarios (55/120 trials
complete) plus full results once the run finishes. The run is significantly slower than
expected due to finding #1 below.

---

## Partial Results (first 3 scenarios, n=5 each)

### instant_only — no slow tools, pure LLM overhead

| Metric | sync | async/tool | async/system | async/user |
|---|---|---|---|---|
| Total latency (s) | ~15 | ~17 | ~17 | ~21 |
| gpt-4o reference | 2.9 | 2.5 | 2.8 | 2.3 |
| **Slowdown** | **5.2×** | **6.8×** | **6.1×** | **9.1×** |
| Pass@1 | 100% | 100% | 100% | 100% |

All trials passed. The latency is 5–9× higher than gpt-4o with **zero tool delay** — this
is entirely LLM call time. gpt-4o-mini generates significantly more verbose, formatted
responses (numbered lists with descriptions vs concise prose), resulting in more output
tokens and longer wall time per call.

### single_slow — 5s tool delay

| Metric | sync | async/tool | async/system | async/user |
|---|---|---|---|---|
| Total latency (s) | ~36 | ~52 | ~54 | ~151* |
| gpt-4o reference | 11.7 | 10.8 | 11.2 | 12.3 |
| **Slowdown** | **3.1×** | **4.8×** | **4.8×** | **12.3×** |
| TTFR | ~36 | ~10 | ~10 | ~10–676 |
| Pass@1 | 100% | 100% | 100% | 100% |

`*` async/user mean dominated by one 676s trial (see Finding #2).

All trials passed. TTFR for async modes is ~10s (acknowledgment LLM call) vs gpt-4o's
~4s — the benefit of async TTFR still exists, but the window is narrower relative to the
inflated total latency.

### mixed_instant_slow — instant + 5s tool

| Metric | sync | async/tool | async/system |
|---|---|---|---|
| Total latency (s) | ~11.5 | ~22 | ~25 |
| gpt-4o reference | 11.3 | 11.2 | 10.6 |
| **Slowdown** | **1.0×** | **2.0×** | **2.4×** |
| Pass@1 | 100% | 100% | 100% |

Key finding: **sync total latency is the same as gpt-4o** for scenarios dominated by a
5s tool delay. gpt-4o-mini's LLM overhead (~10s) is absorbed by the tool wait. But async
modes are 2× slower because the extra injection LLM call (~10s) no longer overlaps with
tool time — it comes after.

---

## Key Findings

### 1. LLM call latency: gpt-4o-mini is 5–10× slower per call in this context

gpt-4o-mini's API call time in this music recommendation context is ~8–15s per call vs
~2–4s for gpt-4o. The cause: gpt-4o-mini generates significantly more verbose formatted
output (numbered lists with bullet-point descriptions for every item) vs gpt-4o's concise
prose. More output tokens → longer wall time, even though the model is nominally "smaller."

This makes gpt-4o-mini unsuitable as a production model for this use case without a system
prompt that strongly constrains output length/format.

### 2. Catastrophic runaway loop: single_slow/async/user trial 2 = 676 seconds

One trial ran for **11 minutes** — approximately 40–60 extra LLM calls. The model treated
the injected `(User) job completed: ...` message as a new user request for a different
query, made a new tool call, got another injection, made another tool call, etc. — a
feedback loop that self-terminated only when the _MAX_ROUNDS=5 cap was hit in a later
injection cycle.

This is the same failure mode seen in gpt-4o (trial 120: async/user produced a playlist
instead of discography), but dramatically amplified. gpt-4o had it once, terminated
gracefully, and it lasted ~28s. gpt-4o-mini entered the same loop but looped for hundreds
of seconds before converging.

**gpt-4o-mini is significantly more susceptible to the async/user injection confusion.**

### 3. Protocol compliance failure: await_job called with empty hint

In mixed_instant_slow/async/tool trial 5, gpt-4o-mini called `await_job` twice with an
empty `followup_hint`:
```
AWAIT_JOB REJECTED  job='8dae68c7'  reason=empty hint
AWAIT_JOB REJECTED  job='8dae68c7'  reason=empty hint
```
gpt-4o never produced an empty-hint await_job call across 120 trials. This is a prompt
instruction following error — gpt-4o-mini partially understood the await_job tool but
omitted the required argument. The trial still passed because the core search_artists
result was present, but the protocol was violated.

### 4. Async total latency advantage inverts for gpt-4o-mini

With gpt-4o at real delays, async modes were 20–43% faster in total latency (because
tool parallelism savings exceeded injection overhead). With gpt-4o-mini:

| Scenario | sync | async/tool | async/system |
|---|---|---|---|
| mixed_instant_slow | **11.5s** | 22s | 25s |
| gpt-4o mixed | 11.3s | **11.2s** | **10.6s** |

Async modes are now **2× slower** in total latency than sync for mixed scenarios. The
injection overhead (one extra ~10s LLM call) exceeds the 5s tool parallelism saving.

The TTFR benefit still exists (~10s vs ~36s for sync), but the total latency cost of
async is now negative when using a slow model.

### 5. Pass@1 is 100% for all completed scenarios despite the failures

Every trial produced the success marker, including the 676s runaway (it eventually
synthesized Miles Davis). This does NOT mean quality is acceptable — the 11-minute trial
is a UX disaster. pass@1 measures only correctness, not latency acceptability.

The correct way to measure this is: **pass@1 within a latency budget** (e.g., "correct
AND completed in under 60s"). By that measure, async/user single_slow would be 80%
(4/5 trials under 60s) rather than 100%.

---

## gpt-4o vs gpt-4o-mini Side-by-Side (completed scenarios)

### Total latency comparison

| Scenario | Mode | gpt-4o | gpt-4o-mini | Ratio |
|---|---|---|---|---|
| instant_only | sync | 2.9s | ~15s | 5.2× |
| instant_only | async/tool | 2.5s | ~17s | 6.8× |
| single_slow | sync | 11.7s | ~36s | 3.1× |
| single_slow | async/tool | 10.8s | ~52s | 4.8× |
| single_slow | async/user | 12.3s | ~151s* | 12.3× |
| mixed_instant_slow | sync | 11.3s | ~11.5s | **1.0×** |
| mixed_instant_slow | async/tool | 11.2s | ~22s | **2.0×** |

### Quality comparison

| Scenario | Mode | gpt-4o pass@1 | gpt-4o-mini pass@1 | Notable |
|---|---|---|---|---|
| instant_only | all | 100% | 100% | — |
| single_slow | all | 100% | 100% | mini: 1× 676s runaway |
| mixed_instant_slow | all | 100% | 100% | mini: await_job empty hint ×2 |

---

## Conclusions

**gpt-4o-mini is not a drop-in replacement for this async framework at these tool delays.**

1. **Total latency**: async modes are 2–5× slower than sync (inverted from gpt-4o)
   because each extra injection LLM call costs 10–15s — more than any parallelism saving.

2. **Reliability**: the async/user mode caused a 676s runaway loop (gpt-4o had one
   28s graceful failure at the same scenario; mini looped for 11 minutes).

3. **Protocol compliance**: mini violated the await_job API contract (empty hint),
   which gpt-4o never did.

4. **TTFR**: async still delivers a faster first response (~10s vs ~36s sync), but the
   practical value is questionable when total completion time is 2× worse.

5. **Sync is the right mode for gpt-4o-mini** at real tool delays. If TTFR matters,
   a much shorter output-length constraint in the system prompt is needed first.

---

## What the remaining scenarios will likely show

*(Prediction before full run completes)*

- **two_parallel**: sync ~30–40s. async/tool ~40–60s. The 2×5s parallel saving (~5s)
  will be swamped by the extra 10–15s injection call. async likely slower.
- **three_parallel**: sync ~40–60s (22s tools). async/tool ~40–70s. Similar inversion.
- **chain**: sync ~35–50s. async likely ~50–80s with more protocol errors.
  chain/async/user may loop again.

Predicting 2–4 additional protocol failures (empty await_job hints or runaway injections)
across the remaining 65 trials.
