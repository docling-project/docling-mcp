"""Define configuration options across tests."""

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

        server_params = StdioServerParameters(
            command="python", args=[server_script_path], env=None
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
    # await client.cleanup()
