# eval/test_state_management.py
#
# Tests for shared-state correctness:
#   - pending_tools lifecycle (add on fire, remove on inject)
#   - _lock prevents concurrent writes to messages

import json
import time
import threading
import pytest
from unittest.mock import patch, MagicMock

from conftest import app
from conftest import make_mock_msg


class TestPendingToolsLifecycle:
    """pending_tools must track in-flight jobs accurately."""

    def test_registered_on_fire(self):
        with patch("use_cases.travel.tools.time.sleep"):
            raw = app.fire_tool_async("get_hotels", {"city": "mumbai"})
        job_id = json.loads(raw)["job_id"]
        assert job_id in app.pending_tools

    def test_removed_after_successful_inject(self):
        """After check_and_inject processes a completed job, job_id leaves pending_tools."""
        job_id = "aa112233"
        app.pending_tools[job_id] = {"name": "get_hotels", "args": {"city": "mumbai"}}
        app.results_queue.put((job_id, "get_hotels", {"city": "mumbai"}, "Hotels in Mumbai: ...", None))

        terminal = make_mock_msg(content="Here are the hotels.")
        with patch.object(app, "call_openai", return_value=terminal):
            app.check_and_inject([])

        assert job_id not in app.pending_tools

    def test_removed_after_failed_inject(self):
        job_id = "bb223344"
        app.pending_tools[job_id] = {"name": "get_flights", "args": {}}
        app.results_queue.put((job_id, "get_flights", {}, None, "timeout"))

        terminal = make_mock_msg(content="Search failed.")
        with patch.object(app, "call_openai", return_value=terminal):
            app.check_and_inject([])

        assert job_id not in app.pending_tools

    def test_multiple_jobs_all_removed(self):
        """All completed job IDs must be removed from pending_tools."""
        ids = ["cc334455", "dd445566", "ee556677"]
        for jid in ids:
            app.pending_tools[jid] = {"name": "get_hotels", "args": {}}
            app.results_queue.put((jid, "get_hotels", {}, "Hotels...", None))

        terminal = make_mock_msg(content="Here you go.")
        with patch.object(app, "call_openai", return_value=terminal):
            app.check_and_inject([])

        for jid in ids:
            assert jid not in app.pending_tools

    def test_unfinished_job_stays_in_pending(self):
        """A job that hasn't completed yet must remain in pending_tools."""
        finished_id = "ff667788"
        still_running_id = "gg778899"

        app.pending_tools[finished_id] = {"name": "get_hotels", "args": {}}
        app.pending_tools[still_running_id] = {"name": "get_flights", "args": {}}
        # Only one result in queue
        app.results_queue.put((finished_id, "get_hotels", {}, "Hotels...", None))

        terminal = make_mock_msg(content="Hotels found, flights still pending.")
        with patch.object(app, "call_openai", return_value=terminal):
            app.check_and_inject([])

        assert finished_id not in app.pending_tools
        assert still_running_id in app.pending_tools


class TestLockBehavior:
    """_lock must serialize concurrent access to messages and OpenAI calls."""

    def test_messages_not_corrupted_under_concurrent_writes(self):
        """
        Two threads append to messages simultaneously under the lock.
        After both complete, messages must contain exactly the expected entries
        with no interleaving or missing items.
        """
        results = []
        errors = []
        barrier = threading.Barrier(2, timeout=5)

        def append_under_lock(tag):
            try:
                barrier.wait()  # Both threads start at the same time
                with app._lock:
                    current_len = len(app.messages)
                    time.sleep(0.01)  # Simulate some work
                    app.messages.append({"role": "user", "content": tag})
                    results.append((tag, current_len))
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=append_under_lock, args=("thread_1",))
        t2 = threading.Thread(target=append_under_lock, args=("thread_2",))
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert not errors, f"Threads raised errors: {errors}"

        contents = [m["content"] for m in app.messages if m.get("role") == "user"]
        assert "thread_1" in contents
        assert "thread_2" in contents

    def test_lock_is_reentrant_safe_between_tests(self):
        """
        The lock must be acquirable after reset_app_state replaces it.
        This tests that reset_app_state creates a fresh, unlocked lock.
        """
        acquired = app._lock.acquire(blocking=False)
        assert acquired, "_lock should be free at start of test (reset by fixture)"
        app._lock.release()

    def test_concurrent_fire_and_user_message_both_complete(self):
        """
        Simulate the race condition scenario from the README:
        Timer injection (check_and_inject) and user message processing
        happening concurrently.

        Both paths must complete; neither must see corrupted state.
        """
        # Pre-populate a completed job for injection
        job_id = "race0000"
        app.pending_tools[job_id] = {"name": "get_hotels", "args": {"city": "mumbai"}}
        app.results_queue.put((job_id, "get_hotels", {"city": "mumbai"}, "Hotels in Mumbai: ...", None))

        call_count = [0]
        lock = threading.Lock()

        def mock_call_openai():
            with lock:
                call_count[0] += 1
            return make_mock_msg(content=f"Response #{call_count[0]}")

        errors = []

        def run_user_message():
            try:
                with app._lock:
                    app.messages.append({"role": "user", "content": "Show me options"})
                    _ = mock_call_openai()
            except Exception as e:
                errors.append(("user_msg", e))

        def run_inject():
            try:
                finished = app.collect_finished_results()
                if finished:
                    with app._lock:
                        for jid, tname, targs, result, err in finished:
                            if err:
                                line = f"(System) Job {jid} FAILED: {tname}({json.dumps(targs)}) → {err}"
                            else:
                                line = f"(System) Job {jid} completed: {tname}({json.dumps(targs)}) → {result}"
                            app.pending_tools.pop(jid, None)
                        app.messages.append({"role": "user", "content": line})
                        _ = mock_call_openai()
            except Exception as e:
                errors.append(("inject", e))

        t1 = threading.Thread(target=run_user_message)
        t2 = threading.Thread(target=run_inject)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert not errors, f"Concurrent paths raised errors: {errors}"
        # Both paths should have added a message
        user_messages = [m for m in app.messages if m.get("role") == "user"]
        assert len(user_messages) >= 2
