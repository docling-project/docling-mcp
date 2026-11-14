"""Test the Docling MCP server conversion tools."""

import shutil
import sys
from pathlib import Path

import pytest
from mcp.types import TextContent

import docling_mcp.tools.conversion
from tests.conftest import MCPClient


@pytest.mark.anyio
async def test_convert_directory_files_into_docling_document(
    mcp_client: MCPClient, tmp_path: Path
) -> None:
    test_dir = Path(__file__).parent
    test_files = [
        test_dir / "data" / "lorem_ipsum.docx.json",
        test_dir / "data" / "amt_handbook_sample.json",
        test_dir / "data" / "2203.01017v2.json",
    ]
    for item in test_files:
        shutil.copy(item, tmp_path)

    res = await mcp_client.call_tool(
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
    assert isinstance(res.structured_content, dict)
    assert "result" in res.structured_content
    assert isinstance(res.structured_content["result"], list)
    assert len(res.structured_content["result"]) == 3
    for item in res.structured_content["result"]:
        assert isinstance(item, dict)
        assert "from_cache" in item
        assert not item.get("from_cache")
        assert item.get("document_key", None)


@pytest.mark.anyio
async def test_converter_error_message_reaches_client(
    mcp_client: MCPClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "s3fs", None)  # simulate the [s3] extra missing

    res = await mcp_client.call_tool(
        "convert_document_into_docling_document", {"source": "s3://bucket/file.pdf"}
    )

    assert res.is_error
    assert isinstance(res.content[0], TextContent)
    assert "docling-mcp[s3]" in res.content[0].text


@pytest.mark.anyio
async def test_unexpected_error_message_stays_hidden(
    mcp_client: MCPClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _crash() -> None:
        raise RuntimeError("internal secret detail")

    monkeypatch.setattr(docling_mcp.tools.conversion, "get_converter", _crash)

    res = await mcp_client.call_tool(
        "convert_document_into_docling_document", {"source": "file.pdf"}
    )

    assert res.is_error
    assert isinstance(res.content[0], TextContent)
    assert "internal secret detail" not in res.content[0].text
