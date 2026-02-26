#!/usr/bin/env python3
"""
LLM Behavior Evaluation for the Async Tool-Calling Travel Assistant.

Runs scripted conversations against the live system (OpenAI GPT-4o),
captures assistant responses at each turn, and evaluates them using:
  (a) Deterministic checks (regex, string search)
  (b) LLM-as-judge using Claude claude-sonnet-4-6

Usage:
    python eval/run_llm_eval.py
    python eval/run_llm_eval.py --scenario flights_basic
    python eval/run_llm_eval.py --output results.json

Environment variables required:
    OPENAI_API_KEY    — for the travel assistant (GPT-4o)
    ANTHROPIC_API_KEY — for Claude judge
"""

import os
import re
import sys
import json
import time
import argparse
import threading
from queue import Queue, Empty
from pathlib import Path
from unittest.mock import patch
from dataclasses import dataclass, field

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import anthropic

# Import travel app AFTER load_dotenv so API keys are present
import app as travel_app


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CriterionResult:
    name: str
    passed: bool
    score: float          # 0.0 – 1.0
    method: str           # "deterministic" or "llm_judge"
    detail: str


@dataclass
class TurnResult:
    turn_index: int
    user_message: str
    assistant_response: str
    criteria: list = field(default_factory=list)

    @property
    def pass_rate(self):
        if not self.criteria:
            return 1.0
        return sum(1 for c in self.criteria if c.passed) / len(self.criteria)


@dataclass
class ScenarioResult:
    scenario_name: str
    turns: list = field(default_factory=list)

    @property
    def aggregate_score(self):
        all_c = [c for t in self.turns for c in t.criteria]
        if not all_c:
            return 1.0
        return sum(c.score for c in all_c) / len(all_c)

    @property
    def pass_rate(self):
        all_c = [c for t in self.turns for c in t.criteria]
        if not all_c:
            return 1.0
        return sum(1 for c in all_c if c.passed) / len(all_c)


# ---------------------------------------------------------------------------
# App state reset
# ---------------------------------------------------------------------------

def reset_app_state():
    travel_app.messages.clear()
    travel_app.messages.append({"role": "system", "content": travel_app.SYSTEM_PROMPT})
    travel_app.pending_tools.clear()
    while not travel_app.results_queue.empty():
        try:
            travel_app.results_queue.get_nowait()
        except Empty:
            break
    travel_app._lock = threading.Lock()


# ---------------------------------------------------------------------------
# Deterministic checks
# ---------------------------------------------------------------------------

HEX8_PATTERN = re.compile(r'\b[0-9a-f]{8}\b')


def check_no_job_id_leak(response: str, known_ids: set) -> CriterionResult:
    """C1: No 8-char hex job_id should appear in the assistant response."""
    for jid in known_ids:
        if jid in response:
            return CriterionResult(
                name="no_job_id_leak",
                passed=False,
                score=0.0,
                method="deterministic",
                detail=f"Known job_id '{jid}' found in response",
            )
    matches = HEX8_PATTERN.findall(response)
    if matches:
        return CriterionResult(
            name="no_job_id_leak",
            passed=True,
            score=0.7,
            method="deterministic",
            detail=f"No known IDs leaked but found hex-like strings: {matches}",
        )
    return CriterionResult(
        name="no_job_id_leak",
        passed=True,
        score=1.0,
        method="deterministic",
        detail="No job IDs or hex-like strings in response",
    )


def check_acknowledgment(response: str, tool_name: str) -> CriterionResult:
    """C2: Response should acknowledge the search started."""
    tool_nouns = {
        "get_hotels":     ["hotel", "accommodation"],
        "get_flights":    ["flight", "airline", "airfare"],
        "get_activities": ["activit", "attraction", "experience", "thing to do"],
    }
    ack_words = [
        "searching", "looking", "checking", "finding", "fetching",
        "on it", "working on", "underway", "started", "initiated",
        "kick", "heading", "begin", "launching", "scanning",
    ]
    rl = response.lower()
    if any(w in rl for w in ack_words) or any(n in rl for n in tool_nouns.get(tool_name, [])):
        return CriterionResult(
            name="acknowledgment_present",
            passed=True,
            score=1.0,
            method="deterministic",
            detail=f"Acknowledgment found for {tool_name}",
        )
    return CriterionResult(
        name="acknowledgment_present",
        passed=False,
        score=0.0,
        method="deterministic",
        detail=f"No acknowledgment vocabulary found. Response: {response[:150]}",
    )


def check_ends_with_question(response: str) -> CriterionResult:
    """C3: Response should contain a follow-up question."""
    stripped = response.strip()
    question_pos = stripped.rfind("?")
    if question_pos >= 0:
        is_near_end = question_pos >= len(stripped) * 0.4
        return CriterionResult(
            name="follow_up_question_present",
            passed=True,
            score=1.0 if is_near_end else 0.6,
            method="deterministic",
            detail=f"Question found at position {question_pos}/{len(stripped)}",
        )
    return CriterionResult(
        name="follow_up_question_present",
        passed=False,
        score=0.0,
        method="deterministic",
        detail="No '?' found — no follow-up question detected",
    )


def check_no_system_echo(response: str) -> CriterionResult:
    """C6: Model must not echo '(System) Job...' text back to the user."""
    if "(System)" in response or "(system)" in response.lower():
        return CriterionResult(
            name="no_system_echo",
            passed=False,
            score=0.0,
            method="deterministic",
            detail="Response contains '(System)' — model is echoing internal messages",
        )
    return CriterionResult(
        name="no_system_echo",
        passed=True,
        score=1.0,
        method="deterministic",
        detail="No '(System)' text in response",
    )


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

def llm_judge(conversation_history: list, assistant_response: str, criteria: list[str]) -> list[CriterionResult]:
    """
    Use Claude claude-sonnet-4-6 to score the assistant response on qualitative criteria.

    criteria: list of criterion names from {"synthesis_quality", "context_awareness",
              "follow_up_relevance", "pending_awareness"}
    """
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Format conversation history for the judge
    history_text = ""
    for msg in conversation_history[-6:]:  # Last 6 messages for context
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if role == "system":
            continue
        history_text += f"\n[{role.upper()}]: {content[:500]}"

    criteria_descriptions = {
        "synthesis_quality":  "When results arrived, did the model mention specific options with a rationale, rather than just listing everything?",
        "context_awareness":  "Did the synthesis reference earlier user preferences (travel companions, budget signals, interests)?",
        "follow_up_relevance": "Is the follow-up question relevant? (flights→hotels/activities, hotels→weather/activities, activities→hotels)",
        "pending_awareness":  "If multiple tools were running, did the model note still-running lookups without mentioning job IDs?",
    }

    criteria_to_score = {k: v for k, v in criteria_descriptions.items() if k in criteria}
    if not criteria_to_score:
        return []

    criteria_prompt = "\n".join(
        f"{i+1}. {name} (0-5): {desc}"
        for i, (name, desc) in enumerate(criteria_to_score.items())
    )

    prompt = f"""You are evaluating an AI travel assistant response for quality and protocol compliance.

CONVERSATION CONTEXT (last few turns):
{history_text}

ASSISTANT RESPONSE TO EVALUATE:
{assistant_response}

Score the response on these criteria (integer 0-5 each):
{criteria_prompt}

Respond ONLY with a JSON object. No explanation outside the JSON.
Example: {{"synthesis_quality": 4, "context_awareness": 3, "notes": "brief reason"}}
"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text

        # Extract JSON (may be wrapped in markdown code fences)
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not json_match:
            raise ValueError(f"No JSON in judge response: {raw}")
        scores = json.loads(json_match.group())
    except Exception as e:
        # If judge fails, return neutral scores
        return [
            CriterionResult(
                name=name,
                passed=False,
                score=0.5,
                method="llm_judge",
                detail=f"Judge error: {e}",
            )
            for name in criteria_to_score
        ]

    results = []
    for name in criteria_to_score:
        raw_score = scores.get(name, 0)
        normalized = raw_score / 5.0
        results.append(CriterionResult(
            name=name,
            passed=normalized >= 0.6,
            score=normalized,
            method="llm_judge",
            detail=f"Score: {raw_score}/5. {scores.get('notes', '')}",
        ))
    return results


# ---------------------------------------------------------------------------
# Collect pending job IDs from message history
# ---------------------------------------------------------------------------

def collect_pending_ids() -> set:
    ids = set()
    for msg in travel_app.messages:
        if msg.get("role") == "tool":
            try:
                parsed = json.loads(msg["content"])
                if "job_id" in parsed:
                    ids.add(parsed["job_id"])
            except (json.JSONDecodeError, TypeError):
                pass
    return ids


# ---------------------------------------------------------------------------
# Run a single conversation turn
# ---------------------------------------------------------------------------

def run_turn(user_msg: str, history: list, sleep_time: float = 0.15) -> tuple[str, list]:
    """
    Drive one turn through process_user_message (a generator).
    Returns (assistant_response, updated_history).
    """
    with patch("tools.time.sleep", lambda _: time.sleep(sleep_time)):
        for hist, _ in travel_app.process_user_message(user_msg, history):
            history = hist
    assistant_response = history[-1]["content"] if history else ""
    return assistant_response, history


def wait_and_inject(history: list, sleep_time: float = 0.15, wait_multiplier: int = 4) -> tuple[str, list]:
    """
    Wait for background tools to complete, then inject results via check_and_inject.
    Returns (injected_assistant_response, updated_history).
    """
    time.sleep(sleep_time * wait_multiplier)
    history = travel_app.check_and_inject(history)
    response = history[-1]["content"] if history else ""
    return response, history


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

SCENARIOS = {
    "flights_basic": {
        "description": "Single flight search — test acknowledgment, follow-up, no job ID leak",
        "turns": [
            {
                "user": "Find me flights from Tokyo to Mumbai.",
                "checks": ["no_job_id_leak", "acknowledgment:get_flights", "follow_up_question"],
                "judge_criteria": ["follow_up_relevance"],
                "wait_for_results": False,
            }
        ],
    },
    "hotels_basic": {
        "description": "Single hotel search — test acknowledgment, follow-up, no job ID leak",
        "turns": [
            {
                "user": "What hotels are available in Amsterdam?",
                "checks": ["no_job_id_leak", "acknowledgment:get_hotels", "follow_up_question"],
                "judge_criteria": ["follow_up_relevance"],
                "wait_for_results": False,
            }
        ],
    },
    "result_synthesis": {
        "description": "Hotel search then result arrives — test synthesis quality",
        "turns": [
            {
                "user": "I'm travelling solo to Mumbai. What hotels are there?",
                "checks": ["no_job_id_leak", "acknowledgment:get_hotels", "follow_up_question"],
                "judge_criteria": [],
                "wait_for_results": False,
            },
            {
                "user": None,  # Trigger result injection (no user message)
                "inject_results": True,
                "checks": ["no_job_id_leak", "no_system_echo"],
                "judge_criteria": ["synthesis_quality", "context_awareness"],
                "wait_for_results": True,
            },
        ],
    },
    "parallel_tools": {
        "description": "Two tools fired at once — test pending awareness",
        "turns": [
            {
                "user": "Find flights from Tokyo to Mumbai and also hotels in Mumbai.",
                "checks": ["no_job_id_leak", "follow_up_question"],
                "judge_criteria": ["pending_awareness"],
                "wait_for_results": False,
            }
        ],
    },
    "instant_tool": {
        "description": "Weather query (instant) — result appears inline without job acknowledgment",
        "turns": [
            {
                "user": "What's the weather like in Tokyo?",
                "checks": ["no_job_id_leak", "no_system_echo"],
                "judge_criteria": [],
                "wait_for_results": False,
            }
        ],
    },
    "error_injection": {
        "description": "Inject a FAILED result — model should inform user gracefully",
        "turns": [
            {
                "user": "Find flights from Tokyo to Amsterdam.",
                "checks": ["no_job_id_leak", "acknowledgment:get_flights"],
                "judge_criteria": [],
                "wait_for_results": False,
            },
            {
                "user": None,
                "inject_fake_failure": True,
                "checks": ["no_job_id_leak", "no_system_echo"],
                "judge_criteria": [],
                "wait_for_results": True,
            },
        ],
    },
}


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------

def run_scenario(name: str, definition: dict, sleep_time: float = 0.15, verbose: bool = True) -> ScenarioResult:
    reset_app_state()
    scenario_result = ScenarioResult(scenario_name=name)
    history = []
    pending_ids = set()

    if verbose:
        print(f"\n{'='*60}")
        print(f"Scenario: {name}")
        print(f"  {definition['description']}")
        print(f"{'='*60}")

    for i, turn_def in enumerate(definition["turns"]):
        criteria_results = []

        # -- User message turn --
        if turn_def.get("user"):
            user_msg = turn_def["user"]
            if verbose:
                print(f"\n  Turn {i+1} — User: {user_msg}")

            response, history = run_turn(user_msg, history, sleep_time)

            if verbose:
                print(f"  Assistant: {response[:200]}{'...' if len(response)>200 else ''}")

        # -- Result injection turn --
        elif turn_def.get("inject_results") or turn_def.get("inject_fake_failure"):
            if verbose:
                print(f"\n  Turn {i+1} — [Result injection]")

            if turn_def.get("inject_fake_failure"):
                # Inject a synthetic FAILED result for the most recent pending job
                pending_copy = dict(travel_app.pending_tools)
                for jid, info in pending_copy.items():
                    travel_app.results_queue.put((jid, info["name"], info["args"], None, "Simulated API failure"))
                time.sleep(0.1)

            response, history = wait_and_inject(history, sleep_time)

            if verbose:
                print(f"  Assistant: {response[:200]}{'...' if len(response)>200 else ''}")
        else:
            continue

        # Collect current pending job IDs
        pending_ids = collect_pending_ids()

        # -- Deterministic checks --
        for check in turn_def.get("checks", []):
            if check == "no_job_id_leak":
                criteria_results.append(check_no_job_id_leak(response, pending_ids))
            elif check.startswith("acknowledgment:"):
                tool = check.split(":")[1]
                criteria_results.append(check_acknowledgment(response, tool))
            elif check == "follow_up_question":
                criteria_results.append(check_ends_with_question(response))
            elif check == "no_system_echo":
                criteria_results.append(check_no_system_echo(response))

        # -- LLM judge --
        judge_criteria = turn_def.get("judge_criteria", [])
        if judge_criteria and os.getenv("ANTHROPIC_API_KEY"):
            criteria_results.extend(
                llm_judge(travel_app.messages, response, judge_criteria)
            )

        # -- Print criteria results --
        if verbose:
            for c in criteria_results:
                icon = "[PASS]" if c.passed else "[FAIL]"
                print(f"    {icon} {c.name} ({c.score:.1f}) — {c.method}")
                if not c.passed:
                    print(f"         Detail: {c.detail}")

        scenario_result.turns.append(TurnResult(
            turn_index=i,
            user_message=turn_def.get("user", "[injection]"),
            assistant_response=response,
            criteria=criteria_results,
        ))

    if verbose:
        print(f"\n  Scenario pass rate: {scenario_result.pass_rate:.0%}")
        print(f"  Aggregate score:    {scenario_result.aggregate_score:.2f}")

    return scenario_result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LLM behavior evaluation for async tool-calling system")
    parser.add_argument("--scenario", help="Run only this scenario (e.g. flights_basic)")
    parser.add_argument("--output", help="Write JSON results to this file")
    parser.add_argument("--sleep", type=float, default=0.15, help="Sleep time for mocked tool delay (default 0.15s)")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-turn output")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set. Add it to .env or environment.")
        sys.exit(1)

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("WARNING: ANTHROPIC_API_KEY not set. LLM judge criteria will be skipped.")

    to_run = SCENARIOS
    if args.scenario:
        if args.scenario not in SCENARIOS:
            print(f"ERROR: Unknown scenario '{args.scenario}'. Available: {list(SCENARIOS.keys())}")
            sys.exit(1)
        to_run = {args.scenario: SCENARIOS[args.scenario]}

    all_results = {}
    for name, definition in to_run.items():
        result = run_scenario(name, definition, sleep_time=args.sleep, verbose=not args.quiet)
        all_results[name] = result

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, r in all_results.items():
        print(f"  {name:25s}  pass={r.pass_rate:.0%}  score={r.aggregate_score:.2f}")

    all_criteria = [c for r in all_results.values() for t in r.turns for c in t.criteria]
    if all_criteria:
        overall_pass = sum(1 for c in all_criteria if c.passed) / len(all_criteria)
        overall_score = sum(c.score for c in all_criteria) / len(all_criteria)
        print(f"\n  Overall: {overall_pass:.0%} pass rate, {overall_score:.2f} aggregate score")

    if args.output:
        output_data = {}
        for name, r in all_results.items():
            output_data[name] = {
                "pass_rate": r.pass_rate,
                "aggregate_score": r.aggregate_score,
                "turns": [
                    {
                        "user": t.user_message,
                        "response_preview": t.assistant_response[:300],
                        "criteria": [
                            {"name": c.name, "passed": c.passed, "score": c.score,
                             "method": c.method, "detail": c.detail}
                            for c in t.criteria
                        ],
                    }
                    for t in r.turns
                ],
            }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
