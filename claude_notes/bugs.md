# Bugs Found (Live Session — 2026-02-27)

Discovered by running `app.py`, chatting through several flows, and correlating
the Gradio UI with server logs.

---

## BUG-1 (Critical) — Timer-injected replies all collapse into one chat bubble

**File:** `app.py:263`

`check_and_inject` appends new assistant entries to `history` in-place and
returns the mutated list. Gradio's Chatbot does not render consecutive
same-role messages as separate bubbles — they all flow into the preceding
assistant turn. Every result injection appears glued to the previous bot
message.

**Observed:** Amsterdam test — "I've started looking for flights…" (initial)
+ "The flights are as follows…" (timer injection) rendered as one block with
one copy icon. Paris/Kolkata test — 4 distinct LLM responses crammed into a
single bubble.

**Cause:** `history.append(...)` mutates in place, vs. `process_user_message`
which correctly uses `history + [...]` to create a fresh list.

---

## BUG-2 (Critical) — `await_job` is never called; dependency tracking is dead code

**File:** `app.py:134`, `main.py:131`

The LLM consistently ignores the `await_job` tool even when the system prompt
explicitly instructs it to register a follow-up dependency before a slow job
completes. In every chained test the `deferred_hints` dict stayed empty.

**Observed in logs:**
```
OPENAI RESP  tool_calls=[]   ← no await_job after get_flights dispatch
HINT CHECK   job=8f05ea07  no deferred hints registered
```

Chains still worked *accidentally* because the timer injection re-exposes the
full conversation context and the LLM re-reads the original user intent. This
is fragile — if the user sends another message before the job completes the
LLM may lose track of the planned follow-up.

---

## BUG-3 (High) — Parallel slow tools produce fragmented, incoherent UX

**File:** `app.py:207`

When N slow tools are dispatched and each finishes at a different time, the
timer fires N separate OpenAI calls. The user receives N separate bot turns,
each aware only of the results that have arrived so far.

**Observed:** 3-job Amsterdam test produced 3 consecutive bot messages within
6 seconds — partial flight info, then partial hotel info, then activities —
instead of one consolidated response.

---

## BUG-4 (High) — System status injected as `user` role message

**File:** `app.py:258`

`check_and_inject` sends the entire injection block — job completion data
AND `(System) Still pending: Job X` status lines — as a single
`{"role": "user", "content": ...}` message. The OpenAI API sees these as
user turns, which is semantically wrong and inflates context costs.

```python
messages.append({"role": "user", "content": injection})  # should be "system"
```

---

## BUG-5 (High) — Results dequeued before lock acquisition; OpenAI call can miss them

**File:** `app.py:209`

```python
finished = collect_finished_results()   # dequeues OUTSIDE the lock
if not finished:
    return history
with _lock:                             # lock acquired AFTER dequeue
    ...
```

Between the dequeue and the lock, `process_user_message` can grab the lock
and call OpenAI. The LLM answers the user without knowing about the
just-dequeued results. Then `check_and_inject` injects those results as a
separate follow-up message.

---

## BUG-6 (Medium) — `history` mutated in-place in timer vs. freshly copied in submit handler

**File:** `app.py:192` vs `app.py:263`

`process_user_message` uses `history + [...]` (new list, safe).
`check_and_inject` uses `history.append(...)` (mutation, unsafe).

Mutating the input before returning it can cause state leakage between
concurrent Gradio events even with `demo.queue()` active.

---

## BUG-7 (Medium) — No mechanism to cancel a registered `await_job` hint

**File:** `app.py:138`, `main.py:134`

Once a hint is stored in `deferred_hints`, it fires unconditionally when the
job completes. If the user says "forget about hotels" after registering an
`await_job` for hotels, the system ignores them and fires the follow-up
anyway. The system prompt has no instruction for the LLM to cancel hints.

---

## BUG-8 (Medium) — Race-condition handler in `main.py` duplicates `inject_results()` logic

**File:** `main.py:275–315`

The `if user_input and finished:` branch in the main loop is ~40 lines of
code that is almost identical to `inject_results()`. Any fix to one path must
be duplicated to the other. Should be refactored to call `inject_results()`
directly.

---

## BUG-9 (Low) — DEBUG timer spam makes logs unreadable

**File:** `app.py:211`

```python
log.debug("TIMER  no results ready  pending=%s", list(pending_tools.keys()))
```

Fires every 0.5 s even when `pending_tools` is empty. After a short session
the log file is 99%+ these no-op lines, burying real events.

---

## BUG-10 (Low) — "No data" tool responses are silent successes, not errors

**File:** `tools.py:117`

`get_flights("tokyo", "paris")` returns the string
`"No flight data for tokyo → paris."` rather than raising an exception.
The background thread logs it as `BG DONE` (success). The `FAILED` injection
path and its error-recovery prompt language are never exercised for missing
data — only for actual Python exceptions.

---

## BUG-11 (High) — User message submitted during timer callback overwrites injected results

**File:** `app.py:184`, `app.py:207`

When the user submits a message while `check_and_inject` is executing, Gradio
captures a stale `history` snapshot as input to `process_user_message`. After
`check_and_inject` returns its updated history (with injected results), the UI
briefly shows those results — then `process_user_message` runs its first `yield`
with the older snapshot, silently overwriting them.

**Observed:** After sending "Find flights from Tokyo to Amsterdam, then find
hotels in Amsterdam," `check_and_inject` held the lock for ~6 seconds (while
dispatching follow-up tools and making two LLM calls for `await_job`). Typing
"no thank you!" during this window caused the injected flight results to flash
on screen then vanish as `process_user_message` replaced them with a shorter
history.

**Cause:** Gradio passes `history` to each event handler at the moment the
event fires, not at the moment execution begins. Both handlers run concurrently;
whichever yields last wins. `process_user_message` is unaware of the results
`check_and_inject` just appended, so its yield produces a history that omits
them entirely.
