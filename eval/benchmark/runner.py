"""run_trial(): execute a single benchmark trial and return a TrialResult.

Design notes:
  - _auto_inject is forced False; this module drives injection manually.
  - For async modes: polls pending_tools, then drains the results_queue and
    injects in rounds. Handles chained slow tools (chain scenario) by looping
    until pending_tools is empty.
  - For sync mode (SyncEngine): fire_tool_async() blocks inline, so
    handle_response() returns the full result immediately — no injection needed.
"""

from __future__ import annotations

import sys
import time
import threading
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))                      # project root (core, use_cases)
sys.path.insert(0, str(Path(__file__).parent))      # benchmark package (metrics, scenarios, …)

from core.engine import AsyncEngine
from metrics import TrialResult, check_job_id_leaked
from scenarios import BenchScenario
from sync_engine import SyncEngine

_TIMEOUT = 30.0       # max seconds to wait for background tools
_SIBLING_WAIT = 0.1   # brief window for parallel tools to land together
_MAX_ROUNDS = 5       # safety cap on injection rounds (prevents infinite loops)


def run_trial(
    scenario: BenchScenario,
    engine: AsyncEngine,
    mode: str,
    trial_idx: int,
    llm_judge_fn=None,
) -> TrialResult:
    """Run one trial of a benchmark scenario.

    The engine must have been reset before this call (call reset_engine()).
    _auto_inject is forced to False here; injection is driven manually.

    Args:
        scenario:      BenchScenario to run.
        engine:        AsyncEngine (or SyncEngine) instance, already reset.
        mode:          Label — "sync", "async/tool", "async/system", "async/user".
        trial_idx:     0-based trial index (for result metadata only).
        llm_judge_fn:  Optional callable(messages, response) → (float, float)
                       returning (synthesis_quality, context_awareness).
    """
    is_sync = isinstance(engine, SyncEngine)
    engine._auto_inject = False  # benchmark drives injection; no background spawning

    # Count every call_openai() invocation for this trial.
    _llm_call_count = [0]
    _original_call_openai = engine.call_openai
    def _counting_call_openai():
        _llm_call_count[0] += 1
        return _original_call_openai()
    engine.call_openai = _counting_call_openai

    try:
        engine.messages.append({"role": "user", "content": scenario.user_message})

        t0 = time.perf_counter()

        # Initial LLM call — dispatches tools (slow ones go to background threads
        # for async modes; fire_tool_async() blocks for SyncEngine).
        initial_msg = engine.call_openai()
        initial_text = engine.handle_response(initial_msg)
        ttfr = time.perf_counter() - t0

        if is_sync or not engine.pending_tools:
            # Sync: all tools ran inline — done immediately.
            # Async with only instant tools (instant_only scenario): also done.
            total_latency = time.perf_counter() - t0
            final_text = initial_text
        else:
            # Async: wait for background jobs, inject results, synthesize.
            final_text = _wait_and_inject_all(engine, t0)
            total_latency = time.perf_counter() - t0

        # --- Deterministic quality checks ---
        # Case-insensitive: avoids false failures when system-prompt differences
        # affect capitalisation (e.g. "Ambient" vs "ambient").
        success = scenario.success_marker.lower() in final_text.lower()
        job_id_leaked = check_job_id_leaked(engine.messages)

        # --- Optional LLM judge ---
        synthesis_quality = None
        context_awareness = None
        if llm_judge_fn is not None:
            try:
                synthesis_quality, context_awareness = llm_judge_fn(engine.messages, final_text)
            except Exception:
                pass  # judge failure is non-fatal; scores stay None

        return TrialResult(
            scenario_name=scenario.name,
            mode=mode,
            trial_idx=trial_idx,
            total_latency=total_latency,
            ttfr=ttfr,
            success=success,
            job_id_leaked=job_id_leaked,
            synthesis_quality=synthesis_quality,
            context_awareness=context_awareness,
            final_response=final_text,
            num_llm_calls=_llm_call_count[0],
        )
    finally:
        engine.call_openai = _original_call_openai


def _wait_and_inject_all(engine: AsyncEngine, t0: float) -> str:
    """Poll until all background jobs complete, injecting results in rounds.

    Handles chained slow tools: after injection the model may dispatch another
    slow tool (e.g. chain scenario), so the loop continues until pending_tools
    is empty and the queue is drained.

    The _SIBLING_WAIT window lets parallel tools that finish close together
    be batched into a single injection round rather than N serial rounds.
    """
    deadline = t0 + _TIMEOUT
    final_text = ""

    for _ in range(_MAX_ROUNDS):
        if not engine.pending_tools:
            break

        # Wait for at least one result to land in the queue.
        while engine.results_queue.empty():
            if time.perf_counter() > deadline:
                pending = list(engine.pending_tools)
                raise TimeoutError(f"Tools timed out after {_TIMEOUT}s: {pending}")
            time.sleep(0.05)

        # Brief window for parallel siblings to also land before we drain.
        time.sleep(_SIBLING_WAIT)

        # Inject all available results in one round.
        finished = engine.collect_finished_results()
        if not finished:
            continue

        engine._inject_finished(finished)
        response = engine.call_openai()
        # handle_response may dispatch more slow tools (chain scenario) →
        # pending_tools grows again → outer loop continues.
        final_text = engine.handle_response(response)

    return final_text


def reset_engine(engine: AsyncEngine) -> None:
    """Reset engine to a clean single-system-prompt state for the next trial."""
    with engine._lock:
        engine.messages.clear()
        engine.messages.append({"role": "system", "content": engine._system_prompt})
        engine.pending_tools.clear()
        engine.deferred_hints.clear()
        engine._lock = threading.Lock()

    # Drain any results left over from the previous trial's background threads.
    while not engine.results_queue.empty():
        try:
            engine.results_queue.get_nowait()
        except Exception:
            break
