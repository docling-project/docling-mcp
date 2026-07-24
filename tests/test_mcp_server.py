"""Test the Docling MCP server tools with a dummy client."""

import json
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import patch

import anyio
import pytest
from mcp import Tool

from docling_mcp.servers.mcp_server import TransportType, main


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
    ]

    assert tools == gold_tools


@pytest.mark.asyncio()
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


@pytest.mark.asyncio()
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


# ---------------------------------------------------------------------------
# main() — Roots wiring (--allowed-directories / install_roots_handlers)
# ---------------------------------------------------------------------------


@patch("docling_mcp._roots_wiring.install_roots_handlers")
@patch("docling_mcp.servers.mcp_server.mcp")
@patch("docling_mcp.servers.mcp_server.allowed_roots")
def test_main_sets_static_roots_when_allowed_directories_given(
    mock_allowed_roots: Any, mock_mcp: Any, mock_install_roots_handlers: Any
) -> None:
    """--allowed-directories seeds the static roots set and wires the
    roots notification handlers.

    install_roots_handlers is patched at its definition site
    (docling_mcp._roots_wiring), not at docling_mcp.servers.mcp_server,
    because main() imports it with a deferred `from ... import` inside
    the function body — the name doesn't exist on the mcp_server module
    until that line executes, so patching it there would fail.
    """
    main(
        transport=TransportType.STDIO,
        tools=None,
        allowed_directories=["/some/path", "/other/path"],
    )

    mock_allowed_roots.set_static_roots.assert_called_once_with(
        ["/some/path", "/other/path"]
    )
    mock_install_roots_handlers.assert_called_once()
    mock_mcp.run.assert_called_once_with(transport=TransportType.STDIO.value)


@patch("docling_mcp._roots_wiring.install_roots_handlers")
@patch("docling_mcp.servers.mcp_server.mcp")
@patch("docling_mcp.servers.mcp_server.allowed_roots")
def test_main_skips_static_roots_when_no_allowed_directories(
    mock_allowed_roots: Any, mock_mcp: Any, mock_install_roots_handlers: Any
) -> None:
    """No --allowed-directories → set_static_roots is never called, but
    the roots notification handlers are still installed (client-sent
    Roots remain fully supported)."""
    main(transport=TransportType.STDIO, tools=None, allowed_directories=None)

    mock_allowed_roots.set_static_roots.assert_not_called()
    mock_install_roots_handlers.assert_called_once()
