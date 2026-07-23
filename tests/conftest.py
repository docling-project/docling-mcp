"""Define configuration options across tests."""

import os
import sys

# Keep tests hermetic: use the in-memory document store so persisted documents
# from earlier runs cannot change conversion results. Set unconditionally,
# before any docling_mcp import, so an inherited environment cannot leak the
# real persistent cache into test runs.
os.environ["DOCLING_MCP_CACHE_PERSIST"] = "false"
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack
from typing import Any

import pytest_asyncio
from mcp import ClientSession, StdioServerParameters, Tool
from mcp.client.stdio import stdio_client


class MCPClient:
    def __init__(self) -> None:
        # Initialize session and client objects
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()

    async def connect_to_server(self, server_script_path: str) -> None:
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script
        """
        if not server_script_path.endswith(".py"):
            raise ValueError("Server script must be a .py file")

        # Set up test environment to use local conversion mode
        # This ensures tests work without requiring Docling Serve access
        test_env = os.environ.copy()
        test_env["DOCLING_CONVERSION_MODE"] = "local"

        # Explicitly use STDIO transport for tests (server default is now streamable-http)
        # Run as module instead of script to ensure proper Typer CLI initialization
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "docling_mcp.servers.mcp_server", "--transport", "stdio"],
            env=test_env,
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        await self.session.initialize()

    async def list_tools(self) -> list[str]:
        assert self.session
        response = await self.session.list_tools()
        tools = [tool.name for tool in response.tools]

        return tools

    async def get_tools(self) -> list[Tool]:
        assert self.session
        response = await self.session.list_tools()

        return response.tools

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any] | None = None
    ) -> Any:
        assert self.session
        response = await self.session.call_tool(tool_name, arguments)

        return response

    async def cleanup(self) -> None:
        """Clean up resources"""
        await self.exit_stack.aclose()


@pytest_asyncio.fixture()
async def mcp_client() -> AsyncGenerator[Any, Any]:
    client = MCPClient()
    await client.connect_to_server("docling_mcp/servers/mcp_server.py")
    yield client
    # Intentionally not awaited: the MCP stdio client's anyio cancel scopes
    # are entered across tasks under pytest-asyncio, and closing the stack
    # here raises "Attempted to exit cancel scope in a different task".
    # Leaked subprocesses are reaped when the pytest process exits.
    # await client.cleanup()
