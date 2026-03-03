# single_message_eval — gpt-5-nano-2025-08-07 vs gpt-4o

## Setup

Same benchmark, same music use case, same 5 real-delay scenarios (5/8/12s tools).
Only difference: `--model gpt-5-nano-2025-08-07`.

Results: `eval/benchmark/results_nano.json`
Reference: `eval/benchmark/results_v2.json` (gpt-4o)

**Note**: Run was interrupted partway through. Data is complete for:
- `instant_only` — all 4 modes, 5 trials ✓
- `single_slow` — all 4 modes, 5 trials ✓
- `mixed_instant_slow` — all 4 modes, 5 trials ✓
- `two_parallel` — sync/async/tool/async/system full (5 trials), async/user partial (4 trials) ✓
- `three_parallel` — sync only (4 trials); async modes: 0 trials ✗
- `chain` — 0 trials ✗

---

## Results by Scenario

### instant_only — no slow tools, pure LLM overhead

| Metric | sync | async/tool | async/system | async/user |
|---|---|---|---|---|
| Total latency (s) | ~35.5 | ~56.0 | ~65.2 | ~61.8 |
| gpt-4o reference | 2.9 | 2.5 | 2.8 | 2.3 |
| **Slowdown** | **12.3×** | **22.2×** | **23.5×** | **26.5×** |
| Mean LLM calls | 2.4 | 3.8 | 4.0 | 3.6 |
| Pass@1 | 100% | 100% | 100% | 100% |

All trials passed. LLM overhead is 12–27× that of gpt-4o — worse than gpt-4o-mini (5–9×). The
extra LLM call per async mode (injection + re-synthesis) compounds the overhead.

### single_slow — 5s tool delay

| Metric | sync | async/tool | async/system | async/user |
|---|---|---|---|---|
| Total latency (s) | ~21.2 | ~67.5 | ~59.5 | ~56.2 |
| gpt-4o reference | 11.7 | 10.8 | 11.2 | 12.3 |
| **Slowdown** | **1.8×** | **6.3×** | **5.3×** | **4.6×** |
| TTFR (s) | ~21 | ~40 | ~41 | ~42 |
| gpt-4o TTFR | ~11.7 | ~4* | ~4* | ~4* |
| Pass@1 | 100% | 100% | 100% | 100% |

`*` Estimated gpt-4o TTFR from acknowledgment response timing.

**Critical TTFR inversion**: gpt-4o async modes achieve TTFR of ~4s vs sync's ~11.7s (a ~3×
improvement). With gpt-5-nano, async TTFR is ~40s — **worse** than sync's total latency of 21s.
The TTFR advantage of async, the primary UX benefit, completely disappears.

**Loop behavior in async/tool** (3/5 trials looped): Trials 2, 3, 4 had 10–11 LLM calls vs the
expected 3. The model appears to be calling tools repeatedly after injection. Normal-latency trials
(2 trials) averaged 42.3s; looped trials (3 trials) averaged 84.3s.

### mixed_instant_slow — instant + 5s tool

| Metric | sync | async/tool | async/system | async/user |
|---|---|---|---|---|
| Total latency (s) | ~148.6 | ~84.3 | ~95.5 | ~57.5 |
| (excl. 629s outlier) | ~28.5 | — | — | — |
| gpt-4o reference | 11.3 | 11.2 | 10.6 | 11.3 |
| **Slowdown** | **13.1×** | **7.5×** | **9.0×** | **5.1×** |
| Mean LLM calls | 2.0 | 5.8 | 6.0 | 4.4 |
| Pass@1 | 100% | 100% | 100% | 100% |

**API latency spike**: sync trial 4 took 629.1 seconds with only 2 LLM calls. This is not a
runaway loop — it is pure API latency (the API call itself took 10+ minutes). Excluding this
outlier, sync mean ≈ 28.5s — still 2.5× slower than gpt-4o but in a reasonable range.

**Async beats sync on mean** (including the outlier): async/user at 57.5s beats sync at 148.6s.
This is an artifact of the outlier dominating the sync mean; excluding it, sync (28.5s) beats
all async modes.

### two_parallel — two concurrent 5s tools

| Metric | sync | async/tool | async/system | async/user |
|---|---|---|---|---|
| Total latency (s) | ~32.6 | ~48.8 | ~52.3 | ~56.1 |
| gpt-4o reference | 17.9 | 13.8 | 20.1 | 15.4 |
| **Slowdown** | **1.8×** | **3.5×** | **2.6×** | **3.6×** |
| Mean LLM calls | 2.0 | 3.0 | 3.6 | 4.75 |
| Pass@1 | 100% | 100% | 100% | 100% (4 trials) |

Sync remains the fastest mode. Async modes show 1.5–2× overhead vs sync due to the injection
LLM call. gpt-4o-async/tool was fastest (13.8s) in two_parallel; nano-async/tool is slower
than nano-sync (48.8s vs 32.6s).

### three_parallel — three concurrent 5/8/12s tools (partial data)

| Metric | sync only |
|---|---|
| Total latency (s) | ~55.1 (4 trials) |
| gpt-4o sync reference | 29.2s |
| **Slowdown** | **1.9×** |
| Mean LLM calls | 2.0 |
| Pass@1 | 100% |

Async modes not captured (run interrupted). Based on the pattern from two_parallel, async modes
would likely be 1.5–2× slower than sync (est. 80–120s vs 55s sync).

---

## Key Findings

### 1. LLM call latency: gpt-5-nano is 12–27× slower per interaction than gpt-4o

For pure LLM overhead (instant_only), nano is 12× slower in sync mode and 22–27× slower in async
modes. Each additional injection LLM call adds ~15–25s. This is significantly worse than gpt-4o-mini
(which was 5–9× slower).

The model appears to generate verbose, formatted responses (similar to gpt-4o-mini's behavior)
that produce many output tokens and inflate wall time.

### 2. The TTFR advantage of async is completely lost

With gpt-4o, the core UX benefit of async was TTFR: ~4s acknowledgment vs ~11.7s sync total.
With gpt-5-nano, the acknowledgment LLM call itself takes ~40s — longer than sync's full response
of 21s. Async provides no TTFR benefit when the model is this slow.

This is the defining failure mode: **async is justified only when LLM call time << tool execution
time**. For gpt-5-nano at 5s tools, LLM time (~18s per call) ≫ tool time (5s), inverting all
async advantages.

### 3. Systematic loop behavior in async/tool (single_slow)

3 of 5 single_slow/async/tool trials had 10–11 LLM calls instead of the expected 3. The model
enters a loop of tool calls after injection. gpt-4o-mini had similar behavior (1 extreme 676s
trial). nano's loop is more frequent (60% of trials in this cell) but self-terminates without
runaway (max ~106s).

gpt-4o never exhibited this behavior. This suggests gpt-5-nano has weaker instruction-following
for the injection protocol than gpt-4o.

### 4. API latency spike: single 629s call in mixed/sync

One sync trial took 629.1 seconds with 2 LLM calls — pure API latency. This is not a model logic
error but an API reliability issue. The model took 10+ minutes to respond to a single prompt. At
this rate, the API is not suitable for production use without aggressive timeouts.

### 5. Pass@1 is 100% for all completed scenarios

Every trial produced the success marker despite loops, slow calls, and API spikes. Correctness
is not the issue — latency and reliability are.

### 6. Sync is unambiguously the best mode for gpt-5-nano

| Scenario | Best mode | Winner |
|---|---|---|
| instant_only | sync (35.5s) | sync |
| single_slow | sync (21.2s) | sync by 2.6–3.2× |
| mixed_instant_slow | async/user (57.5s) | async/user (artifact of 629s outlier) |
| two_parallel | sync (32.6s) | sync by 1.5–1.7× |

Excluding the 629s outlier, sync wins every scenario.

---

## gpt-4o vs gpt-5-nano Side-by-Side (completed scenarios)

### Total latency comparison

| Scenario | Mode | gpt-4o | gpt-5-nano | Ratio |
|---|---|---|---|---|
| instant_only | sync | 2.9s | 35.5s | 12.3× |
| instant_only | async/tool | 2.5s | 56.0s | 22.2× |
| instant_only | async/system | 2.8s | 65.2s | 23.5× |
| instant_only | async/user | 2.3s | 61.8s | 26.5× |
| single_slow | sync | 11.7s | 21.2s | 1.8× |
| single_slow | async/tool | 10.8s | 67.5s | 6.3× |
| single_slow | async/system | 11.2s | 59.5s | 5.3× |
| single_slow | async/user | 12.3s | 56.2s | 4.6× |
| mixed_instant_slow | sync | 11.3s | 148.6s* | 13.1× |
| mixed_instant_slow | async/tool | 11.2s | 84.3s | 7.5× |
| mixed_instant_slow | async/system | 10.6s | 95.5s | 9.0× |
| mixed_instant_slow | async/user | 11.3s | 57.5s | 5.1× |
| two_parallel | sync | 17.9s | 32.6s | 1.8× |
| two_parallel | async/tool | 13.8s | 48.8s | 3.5× |
| two_parallel | async/system | 20.1s | 52.3s | 2.6× |

`*` Dominated by 629s API spike outlier; true mean (excl. outlier) ≈ 28.5s (2.5×)

### TTFR comparison (single_slow — most relevant for async UX)

| Model | sync | async/tool | async/system | async/user |
|---|---|---|---|---|
| gpt-4o | ~11.7s | ~4s | ~4s | ~4s |
| gpt-4o-mini | ~36s | ~10s | ~10s | ~10s |
| gpt-5-nano | ~21.2s | ~40s | ~41s | ~42s |

gpt-5-nano is the only model where async TTFR is **higher** than sync total latency — a complete
inversion of the expected async benefit.

---

## Comparison Across All Three Models

| Model | instant_only/sync | LLM overhead | Async TTFR benefit | Loop issues |
|---|---|---|---|---|
| gpt-4o | 2.9s | baseline | Yes (3× faster TTFR) | None |
| gpt-4o-mini | ~15s | 5× | Partial (10s vs 36s sync) | Yes (1 extreme 676s) |
| gpt-5-nano | ~35.5s | 12× | None (TTFR > sync total) | Yes (60% of async/tool trials) |

gpt-5-nano is worse than gpt-4o-mini on both LLM overhead (12× vs 5×) and loop susceptibility,
despite being nominally a newer generation.

---

## Conclusions

**gpt-5-nano is not viable for this async framework at these tool delays.**

1. **The async TTFR advantage is completely eliminated.** The model is so slow per LLM call
   (~18s) that the 5s tool delay becomes negligible by comparison. The acknowledgment response
   alone takes longer than a full sync turn. No UX justification for async exists at this latency.

2. **Total latency is 12–27× worse than gpt-4o in pure-LLM scenarios.** Even for tool-bound
   scenarios (where tool delay dominates), async modes are 4.6–6.3× slower than gpt-4o.

3. **60% of single_slow/async/tool trials entered a call loop.** The model frequently violates
   the injection protocol by making redundant tool calls. gpt-4o never exhibited this.

4. **API latency is unreliable** (one 629s call, 0 retry logic). A production system would
   need hard timeouts (e.g., 60s) to avoid degraded UX.

5. **Pass@1 remains 100%**, so the model is functionally correct — just unusably slow for
   any interactive application.

6. **Sync is the only viable mode** if gpt-5-nano must be used. Even then, expect 2–12×
   slower total latency than gpt-4o, with the occasional extreme API spike.

---

## What the Missing Scenarios Would Likely Show

*(Prediction — three_parallel async modes and chain not completed)*

- **three_parallel async**: sync at 55.1s. Async modes likely 80–120s (1.5–2× overhead),
  same pattern as two_parallel.
- **chain**: sync likely 40–60s (search → discography, serial). Async/tool may loop
  (await_job hint scaffolding adds extra calls). Expect 60–100s for async modes.
  chain/async/user may trigger a runaway loop (as it did for gpt-4o-mini).

Predicting 1–3 additional loop events across the remaining 19 trials (if run completes).
