import asyncio
import logging
import threading
from typing import TYPE_CHECKING

from .schema import Tool

if TYPE_CHECKING:
    from mcp import StdioServerParameters

log = logging.getLogger("async_tools")

try:
    import mcp.types as _types
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamable_http_client

    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False


class MCPClient:
    """
    Connects to an MCP server and exposes its tools as Tool instances.

    Two transports are supported:
      stdio  — pass a StdioServerParameters; spawns a subprocess.
      HTTP   — pass a URL string; connects to a running HTTP MCP server.

    The session lives for the lifetime of this object on a persistent
    background event loop. Use as a context manager or call close() explicitly.

    Usage (stdio):
        from mcp import StdioServerParameters
        from core.mcp_client import MCPClient

        with MCPClient(StdioServerParameters(command="uvx", args=["mcp-server-fetch"])) as client:
            use_case = UseCase(..., tools=client.tools(is_async=True))

    Usage (HTTP):
        with MCPClient("http://localhost:8000/mcp") as client:
            use_case = UseCase(..., tools=client.tools(is_async=True))
    """

    def __init__(
        self,
        params: "StdioServerParameters | str",
        connect_timeout: float = 30.0,
    ):
        if not _MCP_AVAILABLE:
            raise ImportError(
                "MCP support requires the 'mcp' package. "
                "Install with: uv add mcp"
            )

        self._params = params
        self._connect_timeout = connect_timeout
        self._session = None
        self._client_cm = None
        self._session_cm = None

        # Persistent event loop in a daemon thread.
        # All MCP I/O (connect, list_tools, call_tool) runs here.
        # Using a dedicated thread avoids fighting with any outer asyncio loop.
        self._loop = asyncio.new_event_loop()
        t = threading.Thread(target=self._loop.run_forever, daemon=True)
        t.start()

        self._run(self._connect(), timeout=connect_timeout)
        log.info("MCPClient connected  params=%s", params)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run(self, coro, timeout: float = 60.0):
        """Block the calling thread until `coro` completes on the background loop."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout=timeout)

    async def _connect(self):
        if isinstance(self._params, StdioServerParameters):
            self._client_cm = stdio_client(self._params)
        else:
            # URL string → streamable HTTP transport
            self._client_cm = streamable_http_client(self._params)

        read, write = await self._client_cm.__aenter__()

        self._session_cm = ClientSession(read, write)
        self._session = await self._session_cm.__aenter__()
        await self._session.initialize()

    async def _disconnect(self):
        if self._session_cm:
            await self._session_cm.__aexit__(None, None, None)
        if self._client_cm:
            await self._client_cm.__aexit__(None, None, None)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tools(self, is_async: bool = True) -> list[Tool]:
        """
        Fetch the tool list from the MCP server and return as Tool instances.

        Args:
            is_async: When True (default), tools run in background threads —
                      correct for most MCP tools that call external services.
                      Set False only for tools known to be instant.
        """
        result = self._run(self._session.list_tools())
        wrapped = [self._wrap(t, is_async) for t in result.tools]
        log.info(
            "MCPClient loaded %d tool(s): %s",
            len(wrapped),
            [t.name for t in wrapped],
        )
        return wrapped

    def _wrap(self, mcp_tool, is_async: bool) -> Tool:
        name = mcp_tool.name

        # MCP's inputSchema is already JSON Schema — same structure as Tool.parameters.
        # Fall back to a bare object schema if absent (shouldn't happen per spec).
        parameters = mcp_tool.inputSchema or {"type": "object", "properties": {}}

        def fn(args: dict) -> str:
            result = self._run(self._session.call_tool(name, arguments=args))
            # isError is None (falsy) on success, True on tool-reported errors.
            prefix = "ERROR: " if result.isError else ""
            parts = [
                block.text
                for block in result.content
                if isinstance(block, _types.TextContent)
            ]
            return prefix + ("\n".join(parts) if parts else "(no output)")

        return Tool(
            name=name,
            description=mcp_tool.description or "",
            parameters=parameters,
            fn=fn,
            is_async=is_async,
        )

    def close(self):
        """Disconnect from the MCP server and stop the background event loop."""
        try:
            self._run(self._disconnect(), timeout=10.0)
        except Exception as e:
            log.warning("MCPClient close error: %s", e)
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
        log.info("MCPClient closed")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
