# Frontend Architecture Discussion (2026-02-27)

## Context

BUG-11 exposed a fundamental mismatch between the app's async architecture and
Gradio's event model. This prompted a discussion about whether to stay with
Gradio or move to a custom frontend.

---

## The Core Problem with Gradio (BUG-11)

Gradio passes `history` to each event handler as a **snapshot captured at the
moment the event fires**, not when execution begins. Since `check_and_inject`
(timer) and `process_user_message` (submit) run concurrently, they can each
hold a different stale snapshot. Whichever handler yields last overwrites the
other's output.

Gradio was designed for stateless ML demos (input → output). This app has state
that evolves from two independent sources simultaneously: user messages and
background job completions. That's a structural mismatch.

---

## Options Considered

### Option A — Keep Gradio, move `history` to a global

**Change:** Store `history` as a module-level list (like `messages` already is).
Both `check_and_inject` and `process_user_message` read/write the global instead
of using Gradio's passed-in snapshot.

- **Effort:** ~10 lines changed in `app.py`
- **Fixes:** BUG-11 specifically
- **Doesn't fix:** Gradio's other structural constraints (BUG-3 fragmented UX,
  the `main.py` duplication, general inflexibility)
- **Best for:** Quick patch, staying within current architecture

### Option B — FastAPI + SSE + one plain HTML file

**Stack:**
- `FastAPI` + `uvicorn` for the backend
- Server-Sent Events (`text/event-stream`) for pushing updates to the browser
- One `static/index.html` with ~80 lines of vanilla JS (no npm, no build step)

**New file structure:**
```
server.py          ← replaces app.py (FastAPI + /chat POST + /stream SSE)
static/index.html  ← single HTML file, uses fetch + EventSource API
tools.py           ← unchanged
main.py            ← can be deleted (server.py unifies both paths)
```

**Why SSE fits this app perfectly:**
- The server pushes a message whenever anything happens: user turn, LLM
  response, background job completion
- No polling, no snapshot problem — `history` is a real shared list in
  `server.py`, mutated by both the `/chat` handler and job completion callbacks
- Browser's native `EventSource` API handles reconnection automatically
- Simpler than WebSockets (one-way push is all we need)

**Wins beyond BUG-11:**
- Eliminates the `app.py` / `main.py` duplication (BUG-8)
- Fixes BUG-3 (fragmented UX) naturally — SSE streams each update as it
  arrives, the UI can accumulate them rather than replacing
- Fixes BUG-4 (injection as `user` role) — injection messages stay server-side,
  never need a role at all in the UI
- Makes the codebase more general: any client (web, CLI, tests) can talk to the
  FastAPI backend

**Effort:** Medium — `server.py` is a restructuring of `app.py`'s logic,
`index.html` is new but small.

### Option C — Streamlit

Not recommended — has similar snapshot/state management issues as Gradio for
this use case.

---

## Decision

**Pending.** Both Option A and Option B are valid depending on goals:

| Goal | Recommendation |
|------|---------------|
| Quick bug fix, stay in Gradio | Option A |
| Cleaner architecture, generalize codebase | Option B |

The user's stated goal is "simple and easy, not production ready" — Option B
(FastAPI + SSE) is still simple enough (no npm, no framework) while fixing the
root architectural issue and eliminating `main.py` duplication.
