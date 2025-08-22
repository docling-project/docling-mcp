"""Test the Docling MCP server tools with a dummy client."""

import json
import shutil
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

import anyio
import pytest
import pytest_asyncio
from mcp import ClientSession, StdioServerParameters, Tool
from mcp.client.stdio import stdio_client
from mcp.types import TextContent


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


@pytest.mark.asyncio
async def test_list_tools(mcp_client: AsyncGenerator[Any, Any]) -> None:
    tools = await mcp_client.list_tools()  # type: ignore[attr-defined]
    assert isinstance(tools, list)
    gold_tools = [
        "is_document_in_local_cache",
        "convert_document_into_docling_document",
        "convert_directory_files_into_docling_document",
        # "convert_attachments_into_docling_document",
        "create_new_docling_document",
        "export_docling_document_to_markdown",
        "save_docling_document",
        "add_title_to_docling_document",
        "add_section_heading_to_docling_document",
        "add_paragraph_to_docling_document",
        "open_list_in_docling_document",
        "close_list_in_docling_document",
        "add_list_items_to_list_in_docling_document",
        "add_table_in_html_format_to_docling_document",
        "get_overview_of_document_anchors",
        "search_for_text_in_document_anchors",
        "get_text_of_document_item_at_anchor",
        "update_text_of_document_item_at_anchor",
        "delete_document_items_at_anchors",
    ]

    assert tools == gold_tools


@pytest.mark.asyncio
async def test_get_tools(mcp_client: AsyncGenerator[Any, Any]) -> None:
    tools: list[Tool] = await mcp_client.get_tools()  # type: ignore[attr-defined]

    sample_tool = next(
        item for item in tools if item.name == "add_paragraph_to_docling_document"
    )
    async with await anyio.open_file(
        "tests/data/gt_tool_add_paragraph.json", encoding="utf-8"
    ) as input_file:
        contents = await input_file.read()
        gold_tool = json.loads(contents)
        assert gold_tool == sample_tool.model_dump()


@pytest.mark.asyncio
async def test_call_tool(mcp_client: AsyncGenerator[Any, Any]) -> None:
    res = await mcp_client.call_tool(  # type: ignore[attr-defined]
        "create_new_docling_document", {"prompt": "A new Docling document for testing"}
    )

    # always check if there's been a parsing error through `isError`, since no
    # exception will be raised
    assert not res.isError
    assert isinstance(res.content, list)
    assert len(res.content) == 1
    # there are 2 results: text as an MCP TextContent type...
    assert res.content[0].type == "text"
    assert res.content[0].text.startswith('{\n  "document_key": ')
    # ...the structured output
    assert res.structuredContent["prompt"] == "A new Docling document for testing"
    assert len(res.structuredContent["document_key"]) == 32

    # if no structured output, a schema is infered with the field `result`
    res = await mcp_client.call_tool(  # type: ignore[attr-defined]
        "create_new_docling_document", {}
    )
    assert isinstance(res.content, list)
    assert len(res.content) == 1
    assert "validation error" in res.content[0].text
    assert res.structuredContent is None


@pytest.mark.asyncio
async def test_convert_directory_files_into_docling_document(
    mcp_client: AsyncGenerator[Any, Any], tmp_path: Path
) -> None:
    test_dir = Path(__file__).parent
    test_files = [
        test_dir / "data" / "lorem_ipsum.docx.json",
        test_dir / "data" / "amt_handbook_sample.json",
        test_dir / "data" / "2203.01017v2.json",
    ]
    for item in test_files:
        shutil.copy(item, tmp_path)

    res = await mcp_client.call_tool(  # type: ignore[attr-defined]
        "convert_directory_files_into_docling_document", {"source": str(tmp_path)}
    )

    # returned content block text content
    assert isinstance(res.content, list)
    assert len(res.content) == 3
    assert isinstance(res.content[0], TextContent)
    assert res.content[0].type == "text"
    assert res.content[0].text.startswith(
        '{\n  "from_cache": false,\n  "document_key":'
    )

    # returned structured content
    assert isinstance(res.structuredContent, dict)
    assert "result" in res.structuredContent
    assert isinstance(res.structuredContent["result"], list)
    assert len(res.structuredContent["result"]) == 3
    for item in res.structuredContent["result"]:
        assert isinstance(item, dict)
        assert "from_cache" in item
        assert not item.get("from_cache")
        assert item.get("document_key", None)
