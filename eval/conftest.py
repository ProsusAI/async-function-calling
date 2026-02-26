# eval/conftest.py
#
# IMPORTANT: The openai.OpenAI patch MUST happen before `import app` ever runs,
# because app.py calls openai.OpenAI() at module level (line 14).
# Placing this patch at the top of conftest.py ensures it runs at collection time.

import sys
import threading
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch as _patch

# Make the project root importable (so `import app` works from eval/)
sys.path.insert(0, str(Path(__file__).parent.parent))

# Patch openai.OpenAI before app is ever imported
_openai_patcher = _patch("openai.OpenAI", return_value=MagicMock())
_openai_patcher.start()

import app  # noqa: E402  (safe now — openai.OpenAI is mocked)


# ---------------------------------------------------------------------------
# Autouse fixture: reset all module-level globals between tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_app_state():
    """
    Reset app.py module-level globals before each test.

    app.py uses module-level globals (messages, pending_tools, results_queue, _lock)
    that accumulate state across calls. Without resetting, test order matters
    and tests pollute each other.
    """
    # Setup: clean state
    app.messages.clear()
    app.messages.append({"role": "system", "content": app.SYSTEM_PROMPT})
    app.pending_tools.clear()
    _drain_queue()
    app._lock = threading.Lock()

    yield

    # Teardown: clean up again (daemon threads may still be adding to queue)
    _drain_queue()
    app.pending_tools.clear()


def _drain_queue():
    while not app.results_queue.empty():
        try:
            app.results_queue.get_nowait()
        except Exception:
            break


# ---------------------------------------------------------------------------
# Helper: build a mock OpenAI message object
# ---------------------------------------------------------------------------

def make_mock_msg(content=None, tool_calls=None):
    """
    Build a mock that mimics the object returned by:
        client.chat.completions.create(...).choices[0].message

    app.py accesses:
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
