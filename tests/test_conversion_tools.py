"""Test the Docling MCP server conversion tools."""

import shutil
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.server.fastmcp import Context
from mcp.shared.exceptions import McpError
from mcp.types import TextContent

from docling_mcp.tools.conversion import (
    convert_directory_files_into_docling_document,
    convert_document_into_docling_document,
)


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


# ---------------------------------------------------------------------------
# Roots authorization — PermissionError translates to McpError
# ---------------------------------------------------------------------------


@patch("docling_mcp.tools.conversion.get_converter")
@patch("docling_mcp.tools.conversion.allowed_roots")
def test_convert_document_raises_mcp_error_on_permission_denied(
    mock_allowed_roots: Any, mock_get_converter: Any
) -> None:
    """A PermissionError from allowed_roots.validate_source() surfaces as
    an McpError instead of propagating, and the real converter is never
    reached."""
    mock_allowed_roots.validate_source.side_effect = PermissionError(
        "some path outside allowed roots"
    )

    with pytest.raises(McpError) as exc_info:
        convert_document_into_docling_document("/forbidden/doc.pdf")

    assert (
        exc_info.value.error.message
        == "Unexpected error: some path outside allowed roots"
    )
    mock_get_converter.assert_not_called()


@pytest.mark.asyncio
@patch("docling_mcp.tools.conversion.get_converter")
@patch("docling_mcp.tools.conversion.allowed_roots")
async def test_convert_directory_raises_mcp_error_on_permission_denied(
    mock_allowed_roots: Any, mock_get_converter: Any
) -> None:
    """Same PermissionError-to-McpError translation for the directory
    variant, which additionally requires a Context."""
    mock_allowed_roots.validate_source.side_effect = PermissionError(
        "some path outside allowed roots"
    )
    ctx = MagicMock(spec=Context)
    ctx.info = AsyncMock()
    ctx.report_progress = AsyncMock()
    ctx.debug = AsyncMock()

    with pytest.raises(McpError) as exc_info:
        await convert_directory_files_into_docling_document("/forbidden/dir", ctx)

    assert (
        exc_info.value.error.message
        == "Unexpected error: some path outside allowed roots"
    )
    mock_get_converter.assert_not_called()
    ctx.info.assert_not_awaited()
