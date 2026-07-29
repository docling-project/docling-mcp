"""Define configuration options across tests."""

import os
from collections.abc import AsyncGenerator
from typing import Any

import pytest
from mcp import Client, Tool
from mcp.types import CallToolResult

# Set conversion mode before any docling_mcp module is imported so that the
# pydantic-settings singleton reads the correct value at first import time.
os.environ["DOCLING_CONVERSION_MODE"] = "local"

# Set conversion mode before any docling_mcp module is imported so that the
# pydantic-settings singleton reads the correct value at first import time.
os.environ["DOCLING_MCP_CONVERSION_MODE"] = "local"


class MCPClient:
    """Thin wrapper around Client used by test assertions."""

    def __init__(self, client: Client) -> None:
        self._client = client

    async def list_tools(self) -> list[str]:
        response = await self._client.list_tools()
        return [tool.name for tool in response.tools]

    async def get_tools(self) -> list[Tool]:
        response = await self._client.list_tools()
        return response.tools

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any] | None = None
    ) -> CallToolResult:
        return await self._client.call_tool(tool_name, arguments or {})


@pytest.fixture()
async def mcp_client() -> AsyncGenerator[MCPClient, None]:
    # Import tool/prompt modules so their @mcp.tool / @mcp.prompt decorators
    # fire and register against the shared singleton. Modules only execute
    # once, so subsequent tests reuse the already-registered singleton.
    import docling_mcp.prompts.conversion
    import docling_mcp.prompts.generation
    import docling_mcp.prompts.manipulation
    import docling_mcp.tools.conversion
    import docling_mcp.tools.generation
    import docling_mcp.tools.manipulation
    from docling_mcp.shared import mcp

    async with Client(mcp) as client:
        yield MCPClient(client)
