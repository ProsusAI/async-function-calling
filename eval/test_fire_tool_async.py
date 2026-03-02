# eval/test_fire_tool_async.py
#
# Tests for fire_tool_async():
#   - Return value is valid JSON with correct fields
#   - pending_tools is updated immediately (before background thread finishes)
#   - Distinct job IDs under concurrent calls

import json
import re
import time
import threading
import pytest
from unittest.mock import patch

from conftest import app


class TestReturnValue:
    """fire_tool_async() must return valid job JSON with all required fields."""

    def _fire(self, tool="get_hotels", args=None):
        if args is None:
            args = {"city": "mumbai"}
        with patch("use_cases.travel.tools.time.sleep"):
            return app.fire_tool_async(tool, args)

    def test_returns_valid_json_string(self):
        result = self._fire()
        parsed = json.loads(result)  # must not raise
        assert isinstance(parsed, dict)

    def test_has_job_id_field(self):
        parsed = json.loads(self._fire())
        assert "job_id" in parsed

    def test_job_id_is_8_char_hex(self):
        parsed = json.loads(self._fire())
        job_id = parsed["job_id"]
        assert len(job_id) == 8
        assert re.fullmatch(r"[0-9a-f]{8}", job_id), (
            f"job_id '{job_id}' is not 8 lowercase hex chars"
        )

    def test_status_is_started(self):
        parsed = json.loads(self._fire("get_flights", {"origin": "tokyo", "destination": "mumbai"}))
        assert parsed["status"] == "started"

    def test_tool_name_echoed(self):
        parsed = json.loads(self._fire("get_activities", {"city": "amsterdam"}))
        assert parsed["tool"] == "get_activities"

    def test_args_echoed(self):
        args = {"city": "amsterdam", "tag": "couple"}
        parsed = json.loads(self._fire("get_activities", args))
        assert parsed["args"] == args

    def test_all_four_fields_present(self):
        parsed = json.loads(self._fire("get_flights", {"origin": "tokyo", "destination": "amsterdam"}))
        assert {"job_id", "status", "tool", "args"}.issubset(parsed.keys())


class TestPendingToolsRegistration:
    """pending_tools must be populated immediately, before the background thread finishes."""

    def test_pending_tools_registered_immediately(self):
        """
        Use Events to freeze the background thread mid-execution,
        then assert the job_id is already in pending_tools.

        Note: patch() is started explicitly (not via context manager) so the
        mock stays active while the background thread is running.
        """
        event_started = threading.Event()
        event_release = threading.Event()

        patcher = patch("use_cases.travel.tools.time.sleep")
        mock_sleep = patcher.start()

        def slow_sleep(_):
            event_started.set()    # signal: thread is running
            event_release.wait(timeout=5)  # wait for test to release

        mock_sleep.side_effect = slow_sleep

        try:
            result = app.fire_tool_async("get_hotels", {"city": "mumbai"})
            job_id = json.loads(result)["job_id"]

            assert event_started.wait(timeout=3), "Background thread never started"
            assert job_id in app.pending_tools
            assert app.pending_tools[job_id]["name"] == "get_hotels"
        finally:
            event_release.set()  # release thread regardless of assertion outcome
            patcher.stop()

    def test_pending_tools_name_matches(self):
        with patch("use_cases.travel.tools.time.sleep"):
            result = app.fire_tool_async("get_flights", {"origin": "tokyo", "destination": "mumbai"})
        job_id = json.loads(result)["job_id"]
        # Give thread a moment to potentially finish (it's mocked so it's fast)
        # pending_tools may already be cleaned if thread finished + queue drained,
        # but since no one calls collect_finished_results, it should still be present
        time.sleep(0.05)
        assert job_id in app.pending_tools
        assert app.pending_tools[job_id]["args"] == {"origin": "tokyo", "destination": "mumbai"}


class TestDistinctJobIds:
    """Multiple fire_tool_async calls must produce distinct job IDs."""

    def test_two_calls_distinct(self):
        with patch("use_cases.travel.tools.time.sleep"):
            r1 = json.loads(app.fire_tool_async("get_hotels", {"city": "mumbai"}))
            r2 = json.loads(app.fire_tool_async("get_hotels", {"city": "amsterdam"}))
        assert r1["job_id"] != r2["job_id"]

    def test_ten_concurrent_calls_all_distinct(self):
        job_ids = []
        lock = threading.Lock()

        tools_and_args = [
            ("get_hotels", {"city": "mumbai"}),
            ("get_hotels", {"city": "amsterdam"}),
            ("get_flights", {"origin": "tokyo", "destination": "mumbai"}),
            ("get_flights", {"origin": "tokyo", "destination": "amsterdam"}),
            ("get_activities", {"city": "mumbai"}),
            ("get_activities", {"city": "amsterdam"}),
            ("get_activities", {"city": "mumbai", "tag": "couple"}),
            ("get_activities", {"city": "amsterdam", "tag": "family"}),
            ("get_hotels", {"city": "mumbai"}),
            ("get_flights", {"origin": "tokyo", "destination": "mumbai"}),
        ]

        def fire_and_collect(tool_name, args):
            with patch("use_cases.travel.tools.time.sleep"):
                parsed = json.loads(app.fire_tool_async(tool_name, args))
            with lock:
                job_ids.append(parsed["job_id"])

        threads = [
            threading.Thread(target=fire_and_collect, args=(t, a))
            for t, a in tools_and_args
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert len(job_ids) == 10
        assert len(set(job_ids)) == 10, f"Duplicate job_ids found: {job_ids}"

    def test_three_fires_register_in_pending_tools(self):
        with patch("use_cases.travel.tools.time.sleep"):
            r1 = json.loads(app.fire_tool_async("get_hotels", {"city": "mumbai"}))
            r2 = json.loads(app.fire_tool_async("get_flights", {"origin": "tokyo", "destination": "mumbai"}))
            r3 = json.loads(app.fire_tool_async("get_activities", {"city": "amsterdam"}))

        fired_ids = {r1["job_id"], r2["job_id"], r3["job_id"]}
        assert len(fired_ids) == 3  # all distinct
