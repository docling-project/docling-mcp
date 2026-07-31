"""Test the Docling MCP server tools with a dummy client."""

import inspect
import json

import anyio
import pytest
from mcp import Tool
from mcp.types import TextContent

from tests.conftest import MCPClient


@pytest.mark.anyio
async def test_list_tools(mcp_client: MCPClient) -> None:
    tools = await mcp_client.list_tools()
    assert isinstance(tools, list)
    # Check that all expected tools are registered; order is not guaranteed
    # when tests share an in-process MCPServer singleton.
    expected_tools = {
        "is_document_in_local_cache",
        "convert_document_into_docling_document",
        "convert_directory_files_into_docling_document",
        "create_new_docling_document",
        "export_docling_document_to_markdown",
        "save_docling_document",
        "page_thumbnail",
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
    }

    assert set(tools) == expected_tools


@pytest.mark.anyio()
async def test_get_tools(mcp_client: MCPClient) -> None:
    tools: list[Tool] = await mcp_client.get_tools()

    sample_tool = next(
        item for item in tools if item.name == "add_paragraph_to_docling_document"
    )
    async with await anyio.open_file(
        "tests/data/gt_tool_add_paragraph.json", encoding="utf-8"
    ) as input_file:
        contents = await input_file.read()
        gold_tool = json.loads(contents)
        # Normalise descriptions: Python 3.14 changed inspect.cleandoc to strip
        # leading indentation from docstring continuation lines, so the MCP SDK
        # now returns a cleaned description regardless of Python version.
        sample_dump = sample_tool.model_dump()
        if gold_tool.get("description"):
            gold_tool["description"] = inspect.cleandoc(gold_tool["description"])
        if sample_dump.get("description"):
            sample_dump["description"] = inspect.cleandoc(sample_dump["description"])
        assert gold_tool == sample_dump


@pytest.mark.anyio()
async def test_call_tool(mcp_client: MCPClient) -> None:
    res = await mcp_client.call_tool(
        "create_new_docling_document", {"prompt": "A new Docling document for testing"}
    )

    # always check if there's been a parsing error through `is_error`, since no
    # exception will be raised
    assert not res.is_error
    assert isinstance(res.content, list)
    assert len(res.content) == 1
    # there are 2 results: text as an MCP TextContent type...
    assert isinstance(res.content[0], TextContent)
    assert res.content[0].text.startswith('{\n  "document_key": ')
    # ...the structured output
    assert res.structured_content["prompt"] == "A new Docling document for testing"
    assert len(res.structured_content["document_key"]) == 32

    # if no structured output, a schema is infered with the field `result`
    res = await mcp_client.call_tool("create_new_docling_document", {})
    assert isinstance(res.content, list)
    assert len(res.content) == 1
    assert isinstance(res.content[0], TextContent)
    assert "validation error" in res.content[0].text
    assert res.structured_content is None
