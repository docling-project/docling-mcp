"""Test the Docling MCP server conversion tools."""

import shutil
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pytest
from mcp.types import TextContent


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
