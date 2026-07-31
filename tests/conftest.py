"""Define configuration options across tests."""

import asyncio
import os
import threading
from collections.abc import Coroutine, Generator
from concurrent.futures import Future as ConcurrentFuture
from typing import Any, TypeVar

import pytest
from mcp import Client, Tool
from mcp.types import CallToolResult

_T = TypeVar("_T")

# Set conversion mode before any docling_mcp module is imported so that the
# pydantic-settings singleton reads the correct value at first import time.
os.environ["DOCLING_MCP_CONVERSION_MODE"] = "local"


class MCPClient:
    """Thin wrapper that dispatches async Client calls to a background event loop.

    The MCP Client uses anyio cancel scopes that must enter and exit in the
    same task.  pytest-asyncio 1.x runs each fixture teardown in a fresh task,
    which violates that constraint.  We work around this by keeping the Client
    alive inside a single coroutine that runs on a dedicated background loop
    (managed by a daemon thread).  Test methods submit coroutines to that loop
    via ``asyncio.run_coroutine_threadsafe`` and await the results through an
    executor future, so they never have to share tasks with the Client.
    """

    def __init__(self, client: Client, bg_loop: asyncio.AbstractEventLoop) -> None:
        self._client = client
        self._bg_loop = bg_loop

    async def _async_call(
        self, coro: Coroutine[Any, Any, _T], timeout: float = 60.0
    ) -> _T:
        """Submit *coro* to the background loop and await the result."""
        fut: ConcurrentFuture[_T] = asyncio.run_coroutine_threadsafe(
            coro, self._bg_loop
        )
        # Await via run_in_executor so we don't block the pytest event loop.
        return await asyncio.get_event_loop().run_in_executor(None, fut.result, timeout)

    async def list_tools(self) -> list[str]:
        response = await self._async_call(self._client.list_tools())
        return [tool.name for tool in response.tools]

    async def get_tools(self) -> list[Tool]:
        response = await self._async_call(self._client.list_tools())
        return response.tools

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any] | None = None
    ) -> CallToolResult:
        return await self._async_call(
            self._client.call_tool(tool_name, arguments or {})
        )


@pytest.fixture()
def mcp_client() -> Generator[MCPClient, None, None]:
    """Provide an MCPClient backed by an in-process MCPServer.

    The Client async context manager is entered and exited inside a single
    coroutine running on a background daemon thread, so anyio cancel scopes
    are always entered and exited by the same task.
    """
    # Import tool/prompt modules so their @mcp.tool / @mcp.prompt decorators
    # fire and register against the shared singleton.  Modules only execute
    # once, so subsequent tests reuse the already-registered singleton.
    import docling_mcp.prompts.conversion
    import docling_mcp.prompts.generation
    import docling_mcp.prompts.manipulation
    import docling_mcp.tools.conversion
    import docling_mcp.tools.generation
    import docling_mcp.tools.manipulation
    from docling_mcp.shared import mcp

    # Synchronisation primitives shared between the test thread and the bg thread.
    ready_event = threading.Event()
    client_holder: list[Client] = []
    exc_holder: list[BaseException] = []
    bg_loop_holder: list[asyncio.AbstractEventLoop] = []
    # Will hold a callable that sets the asyncio.Event inside the bg loop.
    shutdown_fn: list[Any] = []

    def _run() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        bg_loop_holder.append(loop)

        async def _lifecycle() -> None:
            try:
                async with Client(mcp) as client:
                    client_holder.append(client)
                    # Wire up the shutdown hook before signalling readiness.
                    shutdown = asyncio.Event()
                    shutdown_fn.append(lambda: loop.call_soon_threadsafe(shutdown.set))
                    ready_event.set()
                    # Wait for the test to finish before tearing down the Client.
                    # asyncio.Event avoids spinning and releases the event loop.
                    await shutdown.wait()
            except BaseException as exc:
                exc_holder.append(exc)
                ready_event.set()  # unblock the test thread on startup failure

        loop.run_until_complete(_lifecycle())
        loop.close()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    ready_event.wait(timeout=30)

    if exc_holder:
        raise exc_holder[0]

    yield MCPClient(client_holder[0], bg_loop_holder[0])

    # Signal the background coroutine to exit the Client context manager.
    shutdown_fn[0]()
    thread.join(timeout=30)

    if exc_holder:
        raise exc_holder[0]
