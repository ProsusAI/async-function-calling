#!/usr/bin/env python
"""single_message_eval — benchmark async vs sync tool-calling injection modes.

Each trial is: one user message → tool calls → final synthesis.
Single-turn design isolates the injection mechanism from multi-turn effects.

Usage:
    uv run python eval/benchmark/run_benchmark.py
    uv run python eval/benchmark/run_benchmark.py --trials 10 --tool-delay 1.0
    uv run python eval/benchmark/run_benchmark.py --scenarios two_parallel chain
    uv run python eval/benchmark/run_benchmark.py --modes sync async/tool --trials 5
    uv run python eval/benchmark/run_benchmark.py --output results.json
    uv run python eval/benchmark/run_benchmark.py --llm-judge --trials 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from unittest import mock

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))                      # project root (core, use_cases)
sys.path.insert(0, str(Path(__file__).parent))      # benchmark package (scenarios, runner, …)

from core.engine import AsyncEngine
from use_cases.music import MusicUseCase
from scenarios import SCENARIOS
from sync_engine import SyncEngine
from metrics import ScenarioBenchmarkResult, print_results_table, MODES
from runner import run_trial, reset_engine


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="single_message_eval: benchmark async vs sync tool-calling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--trials", type=int, default=5,
        help="trials per (scenario, mode) cell",
    )
    p.add_argument(
        "--tool-delay", type=float, default=None, metavar="SECS",
        help=(
            "Override time.sleep() in music tools to a fixed value. "
            "Relative timing is preserved (2 parallel tools still take 1×N in async). "
            "Useful for fast dry-runs; pass^k is timing-independent."
        ),
    )
    p.add_argument(
        "--scenarios", nargs="+", choices=list(SCENARIOS), default=None,
        help="scenarios to run (default: all 6)",
    )
    p.add_argument(
        "--modes", nargs="+", choices=list(MODES), default=None,
        help="injection modes to test (default: all 4)",
    )
    p.add_argument(
        "--output", type=str, default=None, metavar="FILE",
        help="write JSON results to FILE",
    )
    p.add_argument(
        "--llm-judge", action="store_true",
        help="score synthesis quality via Claude (requires ANTHROPIC_API_KEY)",
    )
    p.add_argument(
        "--pass-k", type=int, default=5, metavar="K",
        help="k for pass^k display",
    )
    p.add_argument(
        "--model", type=str, default="gpt-4o", metavar="MODEL",
        help="OpenAI model to use (default: gpt-4o)",
    )
    return p.parse_args()


def _make_llm_judge_fn():
    """Return a judge callable(messages, response) → (synthesis_quality, context_awareness).

    Wraps the existing llm_judge() from eval/run_llm_eval.py, requesting only
    the two music-relevant criteria.  Returns None if the import fails.
    """
    try:
        # run_llm_eval lives one level up from this file
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from run_llm_eval import llm_judge  # noqa: PLC0415
    except ImportError as e:
        print(f"WARNING: could not import llm_judge — skipping judge ({e})")
        return None

    criteria = ["synthesis_quality", "context_awareness"]

    def judge(messages: list, response: str):
        results = llm_judge(messages, response, criteria)
        scores = {r.name: r.score for r in results}
        return scores.get("synthesis_quality"), scores.get("context_awareness")

    return judge


def _build_engines(modes: list[str], model: str) -> dict[str, AsyncEngine]:
    """Instantiate one engine per mode (all share the same MusicUseCase)."""
    engines: dict[str, AsyncEngine] = {}
    for mode in modes:
        if mode == "sync":
            engines[mode] = SyncEngine(MusicUseCase, model=model)
        else:
            injection = mode.split("/")[1]   # "tool" | "system" | "user"
            engines[mode] = AsyncEngine(MusicUseCase, injection_mode=injection, model=model)
    return engines


def run_benchmark(args: argparse.Namespace) -> list[ScenarioBenchmarkResult]:
    scenarios = [SCENARIOS[n] for n in (args.scenarios or list(SCENARIOS))]
    modes = args.modes or list(MODES)
    judge_fn = _make_llm_judge_fn() if args.llm_judge else None
    engines = _build_engines(modes, args.model)
    results: list[ScenarioBenchmarkResult] = []

    # Optionally patch time.sleep in music tools to speed up runs.
    # Capture the real time.sleep BEFORE patching to avoid infinite recursion
    # (the patched side_effect must not call the patched version).
    sleep_patch = None
    if args.tool_delay is not None:
        import time as _time_module
        _real_sleep = _time_module.sleep
        delay = args.tool_delay
        sleep_patch = mock.patch(
            "use_cases.music.tools.time.sleep",
            side_effect=lambda _: _real_sleep(delay),
        )
        sleep_patch.start()
        print(f"Tool delay overridden → {delay}s")

    try:
        total = len(scenarios) * len(modes) * args.trials
        done = 0

        for scenario in scenarios:
            for mode in modes:
                engine = engines[mode]
                sbr = ScenarioBenchmarkResult(scenario_name=scenario.name, mode=mode)

                for trial_idx in range(args.trials):
                    done += 1
                    print(
                        f"  [{done:>{len(str(total))}}/{total}] "
                        f"{scenario.name:<22} {mode:<14} trial {trial_idx + 1}",
                        end="",
                        flush=True,
                    )
                    reset_engine(engine)

                    try:
                        result = run_trial(
                            scenario=scenario,
                            engine=engine,
                            mode=mode,
                            trial_idx=trial_idx,
                            llm_judge_fn=judge_fn,
                        )
                        sbr.trials.append(result)
                        status = "✓" if result.success else "✗"
                        preview = result.final_response[:60].replace("\n", " ").strip()
                        print(f"  {status}  {result.total_latency:.1f}s  {preview!r}")
                    except Exception as e:
                        print(f"  ERROR: {e}")

                results.append(sbr)

    finally:
        if sleep_patch:
            sleep_patch.stop()

    return results


def _to_dict(r: ScenarioBenchmarkResult, k: int) -> dict:
    lo, hi = r.pass_at_1_ci
    return {
        "scenario": r.scenario_name,
        "mode": r.mode,
        "n": r.n,
        "pass_at_1": r.pass_at_1,
        "pass_at_1_ci_lo": lo,
        "pass_at_1_ci_hi": hi,
        f"pass_{k}": r.pass_k(k),
        "mean_total_latency": r.mean_total_latency,
        "stdev_total_latency": r.stdev_total_latency,
        "mean_ttfr": r.mean_ttfr,
        "mean_num_llm_calls": r.mean_num_llm_calls,
        "leak_rate": r.leak_rate,
        "synthesis_quality": r.mean_synthesis_quality,
        "context_awareness": r.mean_context_awareness,
        "trials": [
            {
                "trial_idx": t.trial_idx,
                "success": t.success,
                "job_id_leaked": t.job_id_leaked,
                "total_latency": t.total_latency,
                "ttfr": t.ttfr,
                "num_llm_calls": t.num_llm_calls,
                "synthesis_quality": t.synthesis_quality,
                "context_awareness": t.context_awareness,
                "final_response": t.final_response,
            }
            for t in r.trials
        ],
    }


def main() -> None:
    args = parse_args()

    print(f"\nsingle_message_eval")
    print(f"  trials     : {args.trials}")
    print(f"  scenarios  : {args.scenarios or 'all'}")
    print(f"  modes      : {args.modes or 'all'}")
    print(f"  model      : {args.model}")
    print(f"  tool-delay : {args.tool_delay if args.tool_delay is not None else 'real (5–12s)'}")
    print(f"  llm-judge  : {'yes' if args.llm_judge else 'no'}")
    print()

    results = run_benchmark(args)
    print_results_table(results, k=args.pass_k)

    if args.output:
        data = [_to_dict(r, args.pass_k) for r in results]
        with open(args.output, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
