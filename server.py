import argparse
import asyncio
import logging
import threading
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from core import AsyncEngine
from use_cases.travel import TravelUseCase

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("async_tools")
log.setLevel(logging.DEBUG)

# ---------------------------------------------------------------------------
# Engine — created in __main__ after CLI args are parsed.
# Exposed at module level so tests and tooling can import it directly.
# ---------------------------------------------------------------------------

engine: AsyncEngine = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    engine._sse_loop = asyncio.get_event_loop()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/config")
def config():
    """Use-case metadata consumed by the frontend at page load."""
    return {
        "display_name": engine.use_case.display_name,
        "placeholder":  engine.use_case.input_placeholder,
    }


@app.get("/stream")
async def stream():
    q: asyncio.Queue = asyncio.Queue()
    engine._sse_clients.append(q)

    async def event_generator():
        try:
            while True:
                data = await q.get()
                yield f"data: {data}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            engine._sse_clients.remove(q)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/chat")
def chat(body: dict):
    user_text = (body.get("message") or "").strip()
    if not user_text:
        return {"ok": False, "error": "empty message"}

    log.info("USER MSG     %r", user_text[:100])

    with engine._lock:
        log.debug("LOCK         acquired by /chat")
        engine.messages.append({"role": "user", "content": user_text})
        response = engine.call_openai()
        bot_text = engine.handle_response(response)
        log.debug("LOCK         releasing from /chat")

    engine.push_event("assistant", {"content": bot_text})
    return {"ok": True}


@app.post("/reset")
def reset():
    """Clear conversation history (keep system prompt)."""
    engine.reset()
    engine.push_event("reset", {})
    return {"ok": True}


# Serve static files last so API routes take precedence
app.mount("/", StaticFiles(directory="static", html=True), name="static")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Async Tools Demo server")
    parser.add_argument(
        "--injection-mode",
        choices=["user", "system", "tool"],
        default="tool",
        help=(
            "How completed background job results are injected into the LLM context.\n"
            "  user   — appended as a user-role message\n"
            "  system — appended as a system-role message\n"
            "  tool   — injected as a synthetic assistant tool_call + tool result pair (default)"
        ),
    )
    args = parser.parse_args()

    engine = AsyncEngine(TravelUseCase, injection_mode=args.injection_mode)

    log.info("=" * 50)
    log.info("Starting server  injection_mode=%s", args.injection_mode)
    log.info("=" * 50)
    print(f"Starting server with injection_mode={args.injection_mode!r}")

    uvicorn.run(app, host="0.0.0.0", port=7862, log_level="warning")
