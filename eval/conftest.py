# eval/conftest.py
#
# IMPORTANT: The openai.OpenAI patch MUST happen before any engine import,
# because AsyncEngine calls openai.OpenAI() in __init__.
# Placing this patch at the top of conftest.py ensures it runs at collection time.

import sys
import threading
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch as _patch

# Make the project root importable
sys.path.insert(0, str(Path(__file__).parent.parent))

# Patch openai.OpenAI before anything imports it
_openai_patcher = _patch("openai.OpenAI", return_value=MagicMock())
_openai_patcher.start()

from core.engine import AsyncEngine          # noqa: E402  (safe now — openai.OpenAI is mocked)
from use_cases.travel import TravelUseCase   # noqa: E402

# Single engine instance shared by all tests; reset between each test via fixture.
# Uses "user" injection mode so that check_and_inject tests can assert on
# "(System) Job X completed" text in role=user messages (the legacy mode).
engine = AsyncEngine(TravelUseCase, injection_mode="user")

# Disable auto-injection so background threads only deposit results into the queue
# without draining it. Tests that need injection call check_and_inject() explicitly.
engine._auto_inject = False

# Convenience alias so test files that previously did `import app` can do
# `from eval.conftest import engine as app` with zero further changes to
# attribute access (app.messages → engine.messages, etc.).
app = engine


# ---------------------------------------------------------------------------
# Autouse fixture: reset all engine state between tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_app_state():
    """
    Reset engine state before each test.

    AsyncEngine uses instance attributes (messages, pending_tools, …) that
    accumulate state across calls. Without resetting, test order matters and
    tests pollute each other.
    """
    # Setup: clean state
    engine.messages.clear()
    engine.messages.append({"role": "system", "content": engine._system_prompt})
    engine.pending_tools.clear()
    engine.deferred_hints.clear()
    _drain_queue()
    engine._lock = threading.Lock()

    yield

    # Teardown: clean up again (daemon threads may still be adding to queue)
    _drain_queue()
    engine.pending_tools.clear()
    engine.deferred_hints.clear()


def _drain_queue():
    while not engine.results_queue.empty():
        try:
            engine.results_queue.get_nowait()
        except Exception:
            break


# ---------------------------------------------------------------------------
# Helper: build a mock OpenAI message object
# ---------------------------------------------------------------------------

def make_mock_msg(content=None, tool_calls=None):
    """
    Build a mock that mimics the object returned by:
        client.chat.completions.create(...).choices[0].message

    AsyncEngine accesses:
        msg.content
        msg.tool_calls
        msg.model_dump(exclude_none=True)   -> dict appended to messages list
    """
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls if tool_calls is not None else []

    dump = {"role": "assistant"}
    if content is not None:
        dump["content"] = content
    if tool_calls:
        dump["tool_calls"] = [repr(tc) for tc in tool_calls]
    msg.model_dump.return_value = dump

    return msg
