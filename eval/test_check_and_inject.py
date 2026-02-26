# eval/test_check_and_inject.py
#
# Tests for check_and_inject():
#   - Completion message format: "(System) Job X completed: ..."
#   - Failure message format: "(System) Job X FAILED: ..."
#   - Still-pending line when other jobs are running
#   - No injection when queue is empty
#   - History receives assistant response after injection

import json
import pytest
from unittest.mock import patch

import app
from conftest import make_mock_msg


class TestNoInjectionWhenEmpty:
    def test_returns_history_unchanged_when_queue_empty(self):
        history = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
        result = app.check_and_inject(history)
        assert result is history  # same object returned

    def test_messages_unchanged_when_queue_empty(self):
        initial_len = len(app.messages)
        app.check_and_inject([])
        assert len(app.messages) == initial_len


class TestCompletionMessageFormat:
    """Successful job completion must produce the correct (System) message."""

    def test_completion_line_format(self):
        job_id = "aabb1122"
        args = {"city": "mumbai"}
        result_text = "Hotels in Mumbai:\n  • The Taj Mahal Palace — Colaba — $280/night"

        app.pending_tools[job_id] = {"name": "get_hotels", "args": args}
        app.results_queue.put((job_id, "get_hotels", args, result_text, None))

        terminal = make_mock_msg(content="Here are the hotels!")
        with patch("app.call_openai", return_value=terminal):
            app.check_and_inject([])

        system_msg = next(
            m for m in app.messages
            if m.get("role") == "user" and "(System)" in m.get("content", "")
        )
        content = system_msg["content"]

        assert f"(System) Job {job_id} completed:" in content
        assert "get_hotels" in content
        assert result_text in content

    def test_completion_line_contains_args(self):
        job_id = "ccdd3344"
        args = {"origin": "tokyo", "destination": "amsterdam"}
        app.pending_tools[job_id] = {"name": "get_flights", "args": args}
        app.results_queue.put((job_id, "get_flights", args, "Flights from Tokyo to Amsterdam: ...", None))

        with patch("app.call_openai", return_value=make_mock_msg(content="Got flights.")):
            app.check_and_inject([])

        system_msg = next(
            m for m in app.messages
            if m.get("role") == "user" and "(System)" in m.get("content", "")
        )
        assert "tokyo" in system_msg["content"].lower()
        assert "amsterdam" in system_msg["content"].lower()

    def test_multiple_completions_in_one_inject(self):
        """Multiple finished jobs must all appear in the same injected message."""
        ids = ["ee001122", "ff334455"]
        for jid, city in zip(ids, ["mumbai", "amsterdam"]):
            app.pending_tools[jid] = {"name": "get_hotels", "args": {"city": city}}
            app.results_queue.put((jid, "get_hotels", {"city": city}, f"Hotels in {city.title()}: ...", None))

        with patch("app.call_openai", return_value=make_mock_msg(content="Here you go.")):
            app.check_and_inject([])

        system_msg = next(
            m for m in app.messages
            if m.get("role") == "user" and "(System)" in m.get("content", "")
        )
        for jid in ids:
            assert jid in system_msg["content"]


class TestFailureMessageFormat:
    """Failed jobs must produce the correct FAILED (System) message."""

    def test_failure_line_format(self):
        job_id = "deadbeef"
        args = {"city": "atlantis"}
        app.pending_tools[job_id] = {"name": "get_hotels", "args": args}
        app.results_queue.put((job_id, "get_hotels", args, None, "City not found"))

        with patch("app.call_openai", return_value=make_mock_msg(content="Sorry about that.")):
            app.check_and_inject([])

        system_msg = next(
            m for m in app.messages
            if m.get("role") == "user" and "(System)" in m.get("content", "")
        )
        content = system_msg["content"]
        assert f"(System) Job {job_id} FAILED:" in content
        assert "City not found" in content

    def test_failure_not_completion(self):
        job_id = "cafebabe"
        app.pending_tools[job_id] = {"name": "get_flights", "args": {}}
        app.results_queue.put((job_id, "get_flights", {}, None, "API error"))

        with patch("app.call_openai", return_value=make_mock_msg(content="Flights search failed.")):
            app.check_and_inject([])

        system_msg = next(
            m for m in app.messages
            if m.get("role") == "user" and "(System)" in m.get("content", "")
        )
        assert "completed" not in system_msg["content"]


class TestStillPendingLine:
    """When other jobs are still running, a still-pending line must be appended."""

    def test_still_pending_appended_when_other_jobs_remain(self):
        finished_id = "aa000001"
        still_running_id = "bb000002"

        app.pending_tools[finished_id] = {"name": "get_hotels", "args": {"city": "mumbai"}}
        app.pending_tools[still_running_id] = {"name": "get_flights", "args": {}}
        app.results_queue.put((finished_id, "get_hotels", {"city": "mumbai"}, "Hotels...", None))

        with patch("app.call_openai", return_value=make_mock_msg(content="Hotels found, waiting on flights.")):
            app.check_and_inject([])

        system_msg = next(
            m for m in app.messages
            if m.get("role") == "user" and "(System)" in m.get("content", "")
        )
        assert "Still pending" in system_msg["content"]
        assert still_running_id in system_msg["content"]

    def test_no_still_pending_when_all_done(self):
        """When all pending jobs complete, no 'Still pending' line appears."""
        job_id = "cc000003"
        app.pending_tools[job_id] = {"name": "get_hotels", "args": {"city": "mumbai"}}
        app.results_queue.put((job_id, "get_hotels", {"city": "mumbai"}, "Hotels...", None))

        with patch("app.call_openai", return_value=make_mock_msg(content="All done.")):
            app.check_and_inject([])

        system_msg = next(
            m for m in app.messages
            if m.get("role") == "user" and "(System)" in m.get("content", "")
        )
        assert "Still pending" not in system_msg["content"]


class TestHistoryUpdate:
    """check_and_inject must append assistant response to history."""

    def test_assistant_message_appended_to_history(self):
        job_id = "dd000004"
        app.pending_tools[job_id] = {"name": "get_hotels", "args": {"city": "mumbai"}}
        app.results_queue.put((job_id, "get_hotels", {"city": "mumbai"}, "Hotels in Mumbai: ...", None))

        history = [{"role": "user", "content": "Find hotels"}]
        terminal = make_mock_msg(content="Here are the top hotels in Mumbai!")

        with patch("app.call_openai", return_value=terminal):
            updated_history = app.check_and_inject(history)

        last = updated_history[-1]
        assert last["role"] == "assistant"
        assert "Here are the top hotels in Mumbai!" in last["content"]

    def test_history_grows_by_one_after_inject(self):
        job_id = "ee000005"
        app.pending_tools[job_id] = {"name": "get_activities", "args": {"city": "amsterdam"}}
        app.results_queue.put((job_id, "get_activities", {"city": "amsterdam"}, "Activities in Amsterdam: ...", None))

        history = []
        original_len = len(history)
        with patch("app.call_openai", return_value=make_mock_msg(content="Great activities found.")):
            updated = app.check_and_inject(history)

        assert len(updated) == original_len + 1
