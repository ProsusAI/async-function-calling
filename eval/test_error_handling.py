# eval/test_error_handling.py
#
# Tests for background-thread error handling:
#   - Exceptions in tool functions must produce FAILED queue entries
#   - Silent hangs (pending_tools never cleaned) must not occur
#   - check_and_inject must format FAILED messages correctly

import json
import time
import pytest
from unittest.mock import patch, MagicMock

import app
from conftest import make_mock_msg


def _wait_for_queue(count=1, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if app.results_queue.qsize() >= count:
            return True
        time.sleep(0.05)
    return False


class TestExceptionHandling:
    """Tool exceptions must become FAILED entries, not silent hangs."""

    def test_exception_produces_failed_entry(self):
        """When a tool raises, the queue entry has error set and result=None."""
        def boom(args):
            raise RuntimeError("network timeout")

        with patch.dict(app.TOOL_FUNCTIONS, {"get_hotels": boom}):
            result = app.fire_tool_async("get_hotels", {"city": "mumbai"})

        job_id = json.loads(result)["job_id"]
        assert _wait_for_queue(1), "No error result appeared in queue"

        _, _, _, result_val, error = app.results_queue.get_nowait()
        assert result_val is None
        assert "network timeout" in error

    def test_failed_entry_job_id_matches(self):
        """The job_id in the failed entry must match what was returned."""
        def always_fail(args):
            raise ValueError("bad city")

        with patch.dict(app.TOOL_FUNCTIONS, {"get_hotels": always_fail}):
            raw = app.fire_tool_async("get_hotels", {"city": "atlantis"})

        job_id = json.loads(raw)["job_id"]
        assert _wait_for_queue(1)
        got_id, _, _, _, error = app.results_queue.get_nowait()
        assert got_id == job_id
        assert error is not None

    def test_error_is_string(self):
        """Error field must be a string (the exception message), not an Exception object."""
        def fail(args):
            raise Exception("something went wrong")

        with patch.dict(app.TOOL_FUNCTIONS, {"get_flights": fail}):
            app.fire_tool_async("get_flights", {"origin": "tokyo", "destination": "mumbai"})

        assert _wait_for_queue(1)
        _, _, _, _, error = app.results_queue.get_nowait()
        assert isinstance(error, str)

    def test_no_result_on_failure(self):
        """Result must be None when an error occurs."""
        def fail(args):
            raise RuntimeError("oops")

        with patch.dict(app.TOOL_FUNCTIONS, {"get_activities": fail}):
            app.fire_tool_async("get_activities", {"city": "mumbai"})

        assert _wait_for_queue(1)
        _, _, _, result, _ = app.results_queue.get_nowait()
        assert result is None

    def test_success_and_failure_both_deposit(self):
        """A mix of successful and failed tools must both deposit entries."""
        def fail(args):
            raise RuntimeError("fail")

        # Use patcher.start() so the mock stays active while background threads run
        sleep_patcher = patch("tools.time.sleep")
        sleep_patcher.start()

        try:
            with patch.dict(app.TOOL_FUNCTIONS, {"get_flights": fail}):
                app.fire_tool_async("get_hotels", {"city": "mumbai"})   # succeeds
                app.fire_tool_async("get_flights", {"origin": "tokyo", "destination": "mumbai"})  # fails

            assert _wait_for_queue(2), "Both results (success + failure) should be in queue"
        finally:
            sleep_patcher.stop()

        items = []
        while not app.results_queue.empty():
            items.append(app.results_queue.get_nowait())
        assert len(items) == 2

        errors = [e for _, _, _, _, e in items]
        assert any(e is None for e in errors), "At least one should have succeeded"
        assert any(e is not None for e in errors), "At least one should have failed"


class TestFailedMessageFormat:
    """check_and_inject must format FAILED messages correctly."""

    def test_failed_message_contains_failed_keyword(self):
        """Injected message for a failed job must contain 'FAILED'."""
        job_id = "deadbeef"
        app.pending_tools[job_id] = {"name": "get_hotels", "args": {"city": "mumbai"}}
        app.results_queue.put((job_id, "get_hotels", {"city": "mumbai"}, None, "network error"))

        terminal = make_mock_msg(content="Sorry, something went wrong.")
        with patch("app.call_openai", return_value=terminal):
            app.check_and_inject([])

        injected = next(
            m for m in app.messages
            if m.get("role") == "user" and "(System)" in m.get("content", "")
        )
        assert "FAILED" in injected["content"]
        assert "deadbeef" in injected["content"]
        assert "network error" in injected["content"]

    def test_failed_message_not_completed(self):
        """A failed job must not produce a 'completed' line."""
        job_id = "badc0ffe"
        app.pending_tools[job_id] = {"name": "get_hotels", "args": {"city": "mumbai"}}
        app.results_queue.put((job_id, "get_hotels", {"city": "mumbai"}, None, "timeout"))

        terminal = make_mock_msg(content="It failed.")
        with patch("app.call_openai", return_value=terminal):
            app.check_and_inject([])

        injected = next(
            m for m in app.messages
            if m.get("role") == "user" and "(System)" in m.get("content", "")
        )
        assert "completed" not in injected["content"].lower()

    def test_pending_tools_cleaned_after_failure(self):
        """After a failed job is injected, job_id must be removed from pending_tools."""
        job_id = "cafebabe"
        app.pending_tools[job_id] = {"name": "get_flights", "args": {}}
        app.results_queue.put((job_id, "get_flights", {}, None, "API error"))

        terminal = make_mock_msg(content="Search failed.")
        with patch("app.call_openai", return_value=terminal):
            app.check_and_inject([])

        assert job_id not in app.pending_tools
