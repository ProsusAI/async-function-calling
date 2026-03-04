import asyncio
import json
import logging
import os
import threading
import uuid
from queue import Queue

import openai
from dotenv import load_dotenv

from .await_job import AWAIT_JOB_SCHEMA
from .prompts import BASE_SYSTEM_PROMPTS
from .return_answer import RETURN_ANSWER_SCHEMA
from .schema import UseCase

load_dotenv()

log = logging.getLogger("async_tools")

# Synthesis-only base prompt: no async job-ID mechanics.
# Used when forced_sync=True (no background threads, no job IDs).
_FORCED_SYNC_BASE_PROMPT = (
    "When tool results are available, proactively synthesise them with the "
    "conversation context — do not wait for the user to ask \"which is best.\" "
    "Filter and rank based on what you know: stated interests, companions, budget "
    "signals, or other context from earlier in the conversation. Explain briefly "
    "why the top picks fit their situation. Reserve a full flat list only when you "
    "have no context to work with."
)

# Prepended to any sub-agent's system prompt (when done_event is provided).
_SUBAGENT_PREAMBLE = (
    "You are a specialist sub-agent invoked by an orchestrating agent.\n"
    "Do NOT ask the user for clarification — work with the information given to you.\n"
    "When your task is complete, you MUST call `return_answer_to_parent` with your "
    "full synthesized answer. Do not stop without calling it."
)


class AsyncEngine:
    """
    Core async tool-calling engine.

    Can operate in three modes depending on constructor arguments:

    Parent agent (done_event=None):
      - Standard async tool dispatch with background threads and SSE injection.
      - forced_sync=True: all tools run inline, no job IDs, no await_job.

    Sub-agent (done_event provided):
      - Has its own independent queue, pending_tools, and lock.
      - Does NOT push SSE events (no _sse_loop attached).
      - Adds `return_answer_to_parent` to its tool list.
      - AgentTool blocks on done_event.wait() until the sub-agent calls that tool.
      - forced_sync=True: sub-agent's own tools run inline (no internal parallelism).
    """

    def __init__(
        self,
        use_case: UseCase,
        injection_mode: str = "tool",
        model: str = "gpt-4o",
        forced_sync: bool = False,
        done_event: "threading.Event | None" = None,
        answer_box: "dict | None" = None,
        max_steps: int = 20,
    ):
        self.use_case = use_case
        # Sub-agents always use "tool" injection mode internally.
        self.injection_mode = "tool" if done_event is not None else injection_mode
        self.model = model
        self._forced_sync = forced_sync
        self._done_event = done_event
        self._answer_box = answer_box if answer_box is not None else {}
        self._max_steps = max_steps
        self._step_count = 0
        self._terminated = False

        # -----------------------------------------------------------------
        # System prompt composition
        # -----------------------------------------------------------------
        is_subagent = done_event is not None

        if is_subagent and forced_sync:
            base = _FORCED_SYNC_BASE_PROMPT
        elif is_subagent:
            base = BASE_SYSTEM_PROMPTS["tool"]
        elif forced_sync:
            base = _FORCED_SYNC_BASE_PROMPT
        else:
            base = BASE_SYSTEM_PROMPTS[injection_mode]

        if is_subagent:
            self._system_prompt = (
                _SUBAGENT_PREAMBLE + "\n\n---\n\n" + base + "\n\n---\n\n" + use_case.system_prompt
            )
        else:
            self._system_prompt = base + "\n\n---\n\n" + use_case.system_prompt

        # -----------------------------------------------------------------
        # Tool schema: domain tools + framework-owned tools
        # -----------------------------------------------------------------
        self._tool_map = {t.name: t for t in use_case.tools}
        domain_schemas = [t.schema for t in use_case.tools]

        framework_schemas = []
        if is_subagent:
            framework_schemas.append(RETURN_ANSWER_SCHEMA)
            if not forced_sync:
                framework_schemas.append(AWAIT_JOB_SCHEMA)
        else:
            if not forced_sync:
                framework_schemas.append(AWAIT_JOB_SCHEMA)

        self._tools_schema = domain_schemas + framework_schemas

        # OpenAI client
        self._client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # -----------------------------------------------------------------
        # Mutable state
        # -----------------------------------------------------------------
        self.messages: list = [{"role": "system", "content": self._system_prompt}]
        self.pending_tools: dict = {}       # job_id -> {name, args}
        self.results_queue: Queue = Queue() # background threads drop results here
        self._lock: threading.Lock = threading.Lock()
        self.deferred_hints: dict = {}      # job_id -> list[str] of follow-up hints

        # SSE: one asyncio.Queue per connected browser tab.
        # Sub-agents leave this None — push_event becomes a no-op.
        self._sse_loop: asyncio.AbstractEventLoop = None
        self._sse_clients: list = []

        # When False, background threads deposit results into the queue but do NOT
        # auto-spawn _run_injection. Useful in tests to avoid injection races.
        self._auto_inject: bool = True

    # ------------------------------------------------------------------
    # SSE
    # ------------------------------------------------------------------

    def push_event(self, event_type: str, data: dict):
        """Thread-safe push to all connected SSE clients."""
        payload = json.dumps({"type": event_type, "data": data})
        if self._sse_loop:
            for q in list(self._sse_clients):
                self._sse_loop.call_soon_threadsafe(q.put_nowait, payload)

    # ------------------------------------------------------------------
    # Async / sync tool machinery
    # ------------------------------------------------------------------

    def fire_tool_async(self, tool_name: str, tool_args: dict) -> str:
        """Dispatch a tool marked is_async=True.

        forced_sync=True  → run inline, return real result immediately.
        forced_sync=False → spawn background thread, return job JSON immediately.
        """
        tool_fn = self._tool_map[tool_name].fn

        if self._forced_sync:
            log.info("TOOL SYNC   tool=%s  args=%s", tool_name, json.dumps(tool_args))
            return tool_fn(tool_args)

        job_id = uuid.uuid4().hex[:8]
        log.info("TOOL START  job=%s  tool=%s  args=%s", job_id, tool_name, json.dumps(tool_args))

        def run():
            log.debug("BG THREAD   job=%s  tool=%s  executing...", job_id, tool_name)
            try:
                result = tool_fn(tool_args)
                preview = result[:120] + "..." if len(result) > 120 else result
                log.info("BG DONE     job=%s  tool=%s  result_preview=%r", job_id, tool_name, preview)
                self.results_queue.put((job_id, tool_name, tool_args, result, None))
            except Exception as e:
                log.error("BG FAILED   job=%s  tool=%s  error=%s", job_id, tool_name, e)
                self.results_queue.put((job_id, tool_name, tool_args, None, str(e)))
            if self._auto_inject:
                threading.Thread(target=self._run_injection, daemon=True).start()

        threading.Thread(target=run, daemon=True).start()
        self.pending_tools[job_id] = {"name": tool_name, "args": tool_args}
        return json.dumps({"job_id": job_id, "status": "started", "tool": tool_name, "args": tool_args})

    def call_openai(self):
        """Call the OpenAI chat completions API with current messages."""
        log.info("OPENAI CALL  messages=%d  (last role: %s)", len(self.messages), self.messages[-1]["role"])
        response = self._client.chat.completions.create(
            model=self.model,
            tools=self._tools_schema,
            messages=self.messages,
        )
        msg = response.choices[0].message
        tool_names = [tc.function.name for tc in (msg.tool_calls or [])]
        content_preview = (msg.content or "")[:80].replace("\n", " ")
        log.info("OPENAI RESP  content=%r  tool_calls=%s", content_preview, tool_names)
        return msg

    def handle_response(self, msg) -> str:
        """Process an OpenAI response. Returns the final assistant text to display.

        For sub-agents this return value is NOT the answer — the answer flows via
        return_answer_to_parent → done_event → AgentTool.wait(). The return value
        here is only used internally (e.g. by _run_injection for SSE push).
        """
        # max_steps is only enforced for sub-agents (done_event is set).
        # The parent engine lives for the entire server session — a lifetime
        # counter would silently stop tool processing after enough turns.
        if self._done_event is not None:
            self._step_count += 1
            if self._step_count > self._max_steps:
                log.warning("MAX STEPS (%d) reached — sub-agent giving up.", self._max_steps)
                if not self._terminated:
                    self._answer_box["answer"] = "Sub-agent exceeded maximum steps without completing."
                    self._terminated = True
                    self._done_event.set()
                return ""

        tool_names = [tc.function.name for tc in (msg.tool_calls or [])]
        log.debug("HANDLE RESP  has_content=%s  tool_calls=%s", bool(msg.content), tool_names)
        self.messages.append(msg.model_dump(exclude_none=True))

        # Final response: no tool calls remaining.
        if not msg.tool_calls:
            content = msg.content or ""
            # Sub-agent natural exit: the model returned content without calling
            # return_answer_to_parent. Treat this as an implicit return — signal
            # done_event with the content so the parent isn't left waiting 120s.
            if self._done_event is not None and not self._terminated:
                log.info("NATURAL EXIT  sub-agent auto-signaling with content")
                self._answer_box["answer"] = content
                self._terminated = True
                self._done_event.set()
            return content

        finished_with_answer = False

        for tc in msg.tool_calls:
            tool_name = tc.function.name
            tool_args = json.loads(tc.function.arguments)
            tool_call_id = tc.id

            if tool_name == "return_answer_to_parent":
                answer = tool_args.get("answer", "")
                log.info("RETURN_ANSWER  preview=%r", answer[:80])
                self._answer_box["answer"] = answer
                self._terminated = True
                if self._done_event is not None:
                    self._done_event.set()
                content = json.dumps({"status": "ok", "message": "Answer returned to parent."})
                finished_with_answer = True

            elif tool_name in self._tool_map and self._tool_map[tool_name].is_async:
                log.info("DISPATCH     async tool=%s  args=%s", tool_name, json.dumps(tool_args))
                content = self.fire_tool_async(tool_name, tool_args)

            elif tool_name == "await_job":
                job_id = tool_args.get("job_id", "")
                hint   = tool_args.get("followup_hint", "")
                if job_id in self.pending_tools and hint:
                    self.deferred_hints.setdefault(job_id, []).append(hint)
                    log.info("AWAIT_JOB    registered  job=%s  hint=%r", job_id, hint)
                    content = json.dumps({
                        "status": "registered",
                        "job_id": job_id,
                        "message": "Follow-up intent recorded. You will be reminded when this job completes.",
                    })
                else:
                    reason = "job not in pending_tools" if job_id not in self.pending_tools else "empty hint"
                    log.warning("AWAIT_JOB    REJECTED  job=%r  reason=%s  active_jobs=%s",
                                job_id, reason, list(self.pending_tools.keys()))
                    content = json.dumps({
                        "status": "error",
                        "message": (
                            f"No active job with id '{job_id}'. "
                            "If this job already completed, its result is in the conversation — "
                            "call the follow-up tool directly with that result."
                        ),
                    })

            else:
                log.info("DISPATCH     sync tool=%s  args=%s", tool_name, json.dumps(tool_args))
                content = self._tool_map[tool_name].fn(tool_args)
                preview = content[:80] + "..." if len(content) > 80 else content
                log.debug("INSTANT RES  tool=%s  result=%r", tool_name, preview)

            self.messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": content})

        # Do not recurse if the sub-agent has returned its answer.
        if finished_with_answer:
            return ""

        log.debug("FOLLOWUP     making follow-up OpenAI call after %d tool result(s)", len(msg.tool_calls))
        followup = self.call_openai()
        return self.handle_response(followup)

    # ------------------------------------------------------------------
    # Injection strategies
    # ------------------------------------------------------------------

    def collect_finished_results(self) -> list:
        """Drain results_queue and return all completed jobs as a list of tuples."""
        finished = []
        while not self.results_queue.empty():
            try:
                finished.append(self.results_queue.get_nowait())
            except Exception:
                break
        return finished

    def _inject_finished(self, finished: list):
        """Append completed job results into messages using self.injection_mode.

        "user"   — single {"role": "user",   "content": "(System) Job X completed: ..."}
        "system" — single {"role": "system", "content": "(System) Job X completed: ..."}
        "tool"   — per-job synthetic assistant tool_call + tool result pair
        """
        log.info("INJECT  mode=%s  jobs=%d  pending_before=%s",
                 self.injection_mode, len(finished), list(self.pending_tools.keys()))

        if self.injection_mode == "tool":
            for job_id, tool_name, tool_args, result, error in finished:
                if error:
                    log.error("JOB FAILED   job=%s  tool=%s  error=%s", job_id, tool_name, error)
                else:
                    preview = result[:120] + "..." if len(result) > 120 else result
                    log.info("JOB DONE     job=%s  tool=%s  result_preview=%r", job_id, tool_name, preview)

                # Synthetic assistant message that "called" this tool
                call_id = f"call_{uuid.uuid4().hex[:8]}"
                self.messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(tool_args),
                        },
                    }],
                })
                # Tool result (or error) paired with that call
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": result if result is not None else f"ERROR: {error}",
                })

                hints = self.deferred_hints.pop(job_id, [])
                if hints:
                    log.info("HINT FIRE    job=%s  %d hint(s): %s", job_id, len(hints), hints)
                for hint in hints:
                    if not error:
                        self.messages.append({"role": "system", "content":
                            f"You had planned a follow-up: \"{hint}\". "
                            f"The result is now available above — call the appropriate tool(s)."})
                    else:
                        log.warning("HINT FAIL    job=%s  hint=%r  skipped due to failure", job_id, hint)
                        self.messages.append({"role": "system", "content":
                            f"Follow-up \"{hint}\" cannot proceed — the tool failed. "
                            f"Inform the user and ask how to proceed."})

                self.pending_tools.pop(job_id, None)

            if self.pending_tools:
                still = [f"{v['name']}({json.dumps(v['args'])})" for v in self.pending_tools.values()]
                log.info("STILL PEND   %s", still)
                self.messages.append({"role": "system",
                                      "content": f"Still running in the background: {', '.join(still)}"})

        else:  # "user" or "system"
            lines = []
            for job_id, tool_name, tool_args, result, error in finished:
                if error:
                    log.error("JOB FAILED   job=%s  tool=%s  error=%s", job_id, tool_name, error)
                    lines.append(
                        f"(System) Job {job_id} FAILED: {tool_name}({json.dumps(tool_args)}) → {error}")
                else:
                    preview = result[:120] + "..." if len(result) > 120 else result
                    log.info("JOB DONE     job=%s  tool=%s  result_preview=%r", job_id, tool_name, preview)
                    lines.append(
                        f"(System) Job {job_id} completed: {tool_name}({json.dumps(tool_args)}) → {result}")

                hints = self.deferred_hints.pop(job_id, [])
                if hints:
                    log.info("HINT FIRE    job=%s  %d hint(s): %s", job_id, len(hints), hints)
                for hint in hints:
                    if not error:
                        lines.append(
                            f"(System) You had planned a follow-up after job {job_id}: \"{hint}\". "
                            f"Now that the result is available above, call the appropriate tool(s) with "
                            f"correct arguments. If the original user request implies further steps beyond "
                            f"this follow-up, also register await_job for the next dependency.")
                    else:
                        log.warning("HINT FAIL    job=%s  hint=%r  skipped due to failure", job_id, hint)
                        lines.append(
                            f"(System) Note: Follow-up \"{hint}\" after job {job_id} cannot proceed — "
                            f"job FAILED. Inform the user and ask how to proceed.")

                self.pending_tools.pop(job_id, None)

            if self.pending_tools:
                still = [f"Job {jid} ({v['name']})" for jid, v in self.pending_tools.items()]
                log.info("STILL PEND   %s", still)
                lines.append(f"(System) Still pending: {', '.join(still)}")

            injection = "\n".join(lines)
            log.debug("INJECTION    message to OpenAI:\n%s", injection)
            self.messages.append({"role": self.injection_mode, "content": injection})

    def check_and_inject(self, history=None):
        """Drain the results queue, inject all finished jobs, call OpenAI, return result.

        Must be called with self._lock already held (or in a single-threaded test context).
        """
        finished = self.collect_finished_results()
        if not finished:
            return history
        self._inject_finished(finished)
        response = self.call_openai()
        bot_text = self.handle_response(response)
        if history is not None:
            history.append({"role": "assistant", "content": bot_text})
        return history

    def _run_injection(self):
        """Called from a background thread when a job completes."""
        # Guard: if the engine has been terminated (sub-agent called return_answer_to_parent
        # while async tools were still running), skip injection entirely.
        if self._terminated:
            log.debug("INJECTION SKIPPED — engine terminated (dangling thread)")
            return

        with self._lock:
            if self._terminated:  # re-check after acquiring lock
                return
            finished = self.collect_finished_results()
            if not finished:
                return
            self._inject_finished(finished)
            response = self.call_openai()
            bot_text = self.handle_response(response)

        self.push_event("assistant", {"content": bot_text, "async": True})

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self):
        """Clear conversation history (keep system prompt)."""
        with self._lock:
            self.messages.clear()
            self.messages.append({"role": "system", "content": self._system_prompt})
            self.pending_tools.clear()
            self.deferred_hints.clear()
            self._step_count = 0
            self._terminated = False
