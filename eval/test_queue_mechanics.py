# eval/test_queue_mechanics.py
#
# Tests for the results_queue / background thread deposit mechanism:
#   - Successful tool completion puts a 5-tuple in the queue
#   - collect_finished_results() drains the queue and returns all items
#   - Multiple concurrent threads all deposit their results

import json
import time
import threading
import pytest
from unittest.mock import patch

import app


def _wait_for_queue(count=1, timeout=3.0):
    """Poll until the queue has at least `count` items or timeout expires."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if app.results_queue.qsize() >= count:
            return True
        time.sleep(0.05)
    return False


class TestQueueDeposit:
    """Background threads must deposit results into results_queue."""

    def test_success_result_deposited(self):
        """After a slow tool completes, a success tuple appears in the queue."""
        with patch("tools.time.sleep"):
            result = app.fire_tool_async("get_hotels", {"city": "mumbai"})

        job_id = json.loads(result)["job_id"]

        assert _wait_for_queue(1), "No result appeared in queue within timeout"

        item = app.results_queue.get_nowait()
        got_job_id, tool_name, tool_args, result_str, error = item

        assert got_job_id == job_id
        assert tool_name == "get_hotels"
        assert tool_args == {"city": "mumbai"}
        assert "Hotels in Mumbai" in result_str
        assert error is None

    def test_five_tuple_structure(self):
        """Queue entry must be a 5-tuple: (job_id, tool_name, tool_args, result, error)."""
        with patch("tools.time.sleep"):
            app.fire_tool_async("get_flights", {"origin": "tokyo", "destination": "mumbai"})

        assert _wait_for_queue(1)
        item = app.results_queue.get_nowait()
        assert len(item) == 5, f"Expected 5-tuple, got {len(item)}-tuple"

    def test_success_has_none_error(self):
        with patch("tools.time.sleep"):
            app.fire_tool_async("get_activities", {"city": "amsterdam"})

        assert _wait_for_queue(1)
        _, _, _, result, error = app.results_queue.get_nowait()
        assert error is None
        assert result is not None

    def test_multiple_threads_all_deposit(self):
        """Three concurrent async tool calls must each deposit exactly one result."""
        with patch("tools.time.sleep"):
            app.fire_tool_async("get_hotels", {"city": "mumbai"})
            app.fire_tool_async("get_flights", {"origin": "tokyo", "destination": "amsterdam"})
            app.fire_tool_async("get_activities", {"city": "amsterdam"})

        assert _wait_for_queue(3), "Not all 3 results appeared in queue"
        items = []
        while not app.results_queue.empty():
            items.append(app.results_queue.get_nowait())
        assert len(items) == 3

    def test_tool_args_preserved_in_deposit(self):
        """The args deposited must exactly match what was passed to fire_tool_async."""
        args = {"city": "amsterdam", "tag": "couple"}
        with patch("tools.time.sleep"):
            app.fire_tool_async("get_activities", args)

        assert _wait_for_queue(1)
        _, _, deposited_args, _, _ = app.results_queue.get_nowait()
        assert deposited_args == args


class TestCollectFinishedResults:
    """collect_finished_results() must drain the queue and return all items."""

    def test_drains_queue(self):
        # Pre-populate with synthetic entries
        app.results_queue.put(("id1", "get_hotels", {}, "Hotels...", None))
        app.results_queue.put(("id2", "get_flights", {}, "Flights...", None))

        results = app.collect_finished_results()

        assert app.results_queue.empty()
        assert len(results) == 2

    def test_returns_all_items(self):
        for i in range(5):
            app.results_queue.put((f"id{i}", "get_hotels", {}, f"result{i}", None))

        results = app.collect_finished_results()
        assert len(results) == 5

    def test_empty_queue_returns_empty_list(self):
        results = app.collect_finished_results()
        assert results == []

    def test_items_in_order(self):
        app.results_queue.put(("first", "get_hotels", {}, "r1", None))
        app.results_queue.put(("second", "get_flights", {}, "r2", None))

        results = app.collect_finished_results()
        assert results[0][0] == "first"
        assert results[1][0] == "second"
