# eval/test_routing.py
#
# Tests for tool routing logic in handle_response():
#   - Slow tools (get_hotels, get_flights, get_activities) go async via fire_tool_async
#   - Instant tools (get_weather) run inline and return real results
#   - Tool messages are appended to app.messages with correct structure

import json
import pytest
from unittest.mock import patch, MagicMock, call

import app
from tools import SLOW_TOOLS, TOOL_FUNCTIONS
from conftest import make_mock_msg


class TestSlowToolsSet:
    """SLOW_TOOLS must contain exactly the right tool names."""

    def test_get_hotels_in_slow_tools(self):
        assert "get_hotels" in SLOW_TOOLS

    def test_get_flights_in_slow_tools(self):
        assert "get_flights" in SLOW_TOOLS

    def test_get_activities_in_slow_tools(self):
        assert "get_activities" in SLOW_TOOLS

    def test_get_weather_not_in_slow_tools(self):
        assert "get_weather" not in SLOW_TOOLS


class TestSlowToolRouting:
    """Slow tools must be dispatched to fire_tool_async, not called inline."""

    def _make_tool_call(self, tool_name, args, call_id="tc_001"):
        tc = MagicMock()
        tc.function.name = tool_name
        tc.function.arguments = json.dumps(args)
        tc.id = call_id
        return tc

    def test_hotels_calls_fire_async(self):
        tc = self._make_tool_call("get_hotels", {"city": "mumbai"})
        msg = make_mock_msg(tool_calls=[tc])
        terminal = make_mock_msg(content="Hotels are being searched.")

        with patch("app.fire_tool_async", return_value='{"job_id":"aa000000","status":"started","tool":"get_hotels","args":{"city":"mumbai"}}') as mock_fire, \
             patch("app.call_openai", return_value=terminal):
            app.handle_response(msg)

        mock_fire.assert_called_once_with("get_hotels", {"city": "mumbai"})

    def test_flights_calls_fire_async(self):
        tc = self._make_tool_call("get_flights", {"origin": "tokyo", "destination": "mumbai"})
        msg = make_mock_msg(tool_calls=[tc])
        terminal = make_mock_msg(content="Flights are being searched.")

        with patch("app.fire_tool_async", return_value='{"job_id":"bb000000","status":"started","tool":"get_flights","args":{}}') as mock_fire, \
             patch("app.call_openai", return_value=terminal):
            app.handle_response(msg)

        mock_fire.assert_called_once_with("get_flights", {"origin": "tokyo", "destination": "mumbai"})

    def test_activities_calls_fire_async(self):
        tc = self._make_tool_call("get_activities", {"city": "amsterdam", "tag": "couple"})
        msg = make_mock_msg(tool_calls=[tc])
        terminal = make_mock_msg(content="Activities are being searched.")

        with patch("app.fire_tool_async", return_value='{"job_id":"cc000000","status":"started","tool":"get_activities","args":{}}') as mock_fire, \
             patch("app.call_openai", return_value=terminal):
            app.handle_response(msg)

        mock_fire.assert_called_once_with("get_activities", {"city": "amsterdam", "tag": "couple"})

    def test_slow_tool_appends_job_json_as_tool_message(self):
        """The tool message in messages must contain the job JSON, not a real result."""
        tc = self._make_tool_call("get_hotels", {"city": "mumbai"}, call_id="tc_slow")
        msg = make_mock_msg(tool_calls=[tc])
        terminal = make_mock_msg(content="I'll look into that.")

        job_json = '{"job_id":"dd000000","status":"started","tool":"get_hotels","args":{"city":"mumbai"}}'
        with patch("app.fire_tool_async", return_value=job_json), \
             patch("app.call_openai", return_value=terminal):
            app.handle_response(msg)

        tool_messages = [m for m in app.messages if m.get("role") == "tool"]
        assert len(tool_messages) == 1
        assert tool_messages[0]["tool_call_id"] == "tc_slow"
        assert tool_messages[0]["content"] == job_json

    def test_slow_tool_content_is_job_json_not_real_data(self):
        """Confirm the content is job JSON (has job_id), not hotel listing text."""
        tc = self._make_tool_call("get_hotels", {"city": "mumbai"})
        msg = make_mock_msg(tool_calls=[tc])
        terminal = make_mock_msg(content="Searching...")

        with patch("app.fire_tool_async", return_value='{"job_id":"ee000000","status":"started","tool":"get_hotels","args":{}}'), \
             patch("app.call_openai", return_value=terminal):
            app.handle_response(msg)

        tool_msg = next(m for m in app.messages if m.get("role") == "tool")
        parsed = json.loads(tool_msg["content"])
        assert "job_id" in parsed
        assert "Hotels in" not in tool_msg["content"]


class TestInstantToolRouting:
    """Instant tools (get_weather) must run inline and return real results."""

    def _make_tool_call(self, tool_name, args, call_id="tc_instant"):
        tc = MagicMock()
        tc.function.name = tool_name
        tc.function.arguments = json.dumps(args)
        tc.id = call_id
        return tc

    def test_weather_tool_runs_inline(self):
        """get_weather must NOT call fire_tool_async."""
        tc = self._make_tool_call("get_weather", {"city": "tokyo"})
        msg = make_mock_msg(tool_calls=[tc])
        terminal = make_mock_msg(content="The weather in Tokyo is rainy.")

        with patch("app.fire_tool_async") as mock_fire, \
             patch("app.call_openai", return_value=terminal):
            app.handle_response(msg)

        mock_fire.assert_not_called()

    def test_weather_result_is_real_string(self):
        """get_weather tool message content must be weather data, not job JSON."""
        tc = self._make_tool_call("get_weather", {"city": "tokyo"}, call_id="tc_wx")
        msg = make_mock_msg(tool_calls=[tc])
        terminal = make_mock_msg(content="Tokyo is rainy.")

        with patch("app.call_openai", return_value=terminal):
            app.handle_response(msg)

        tool_messages = [m for m in app.messages if m.get("role") == "tool"]
        assert len(tool_messages) == 1
        assert tool_messages[0]["tool_call_id"] == "tc_wx"
        # Must be plain text, not job JSON
        try:
            parsed = json.loads(tool_messages[0]["content"])
            assert "job_id" not in parsed, "Weather result should not be job JSON"
        except json.JSONDecodeError:
            pass  # Not JSON at all → definitely correct behavior

    def test_weather_result_contains_city_data(self):
        tc = self._make_tool_call("get_weather", {"city": "tokyo"})
        msg = make_mock_msg(tool_calls=[tc])
        terminal = make_mock_msg(content="It's rainy.")

        with patch("app.call_openai", return_value=terminal):
            app.handle_response(msg)

        tool_msg = next(m for m in app.messages if m.get("role") == "tool")
        # tools.py returns "18°C, rainy" for tokyo
        assert "°C" in tool_msg["content"] or "tokyo" in tool_msg["content"].lower()

    def test_tool_call_id_preserved_in_message(self):
        """The tool_call_id in the appended message must match the original tool call."""
        tc = self._make_tool_call("get_weather", {"city": "paris"}, call_id="tc_paris_wx")
        msg = make_mock_msg(tool_calls=[tc])
        terminal = make_mock_msg(content="Paris weather noted.")

        with patch("app.call_openai", return_value=terminal):
            app.handle_response(msg)

        tool_msg = next(m for m in app.messages if m.get("role") == "tool")
        assert tool_msg["tool_call_id"] == "tc_paris_wx"
