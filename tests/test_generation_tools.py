"""Test the Docling MCP server generation tools."""

import re
from pathlib import Path

import pytest
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import TextContent

from docling_core.types.doc.base import BoundingBox, Size
from docling_core.types.doc.common.reference import ProvenanceItem
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel

import docling_mcp.tools.generation as generation_mod
from docling_mcp.logger import setup_logger
from docling_mcp.settings.service_client import settings
from docling_mcp.shared import _LRUCaches, local_document_cache
from docling_mcp.tools.generation import (
    ExportDocumentMarkdownOutput,
    NewDoclingDocumentOutput,
    SaveDocumentOutput,
    UpdateDocumentOutput,
    add_table_in_html_format_to_docling_document,
    create_new_docling_document,
    export_docling_document_to_markdown,
    save_docling_document,
)
from tests.conftest import MCPClient

logger = setup_logger()

DATA_DIR = Path(__file__).parent / "data"
PAGED_KEY = "paged"
# Page 2 exists but has no content.
PAGE_TEXTS = {1: "Alpha on page one", 3: "Gamma on page three"}


@pytest.fixture
def doc_key() -> str:
    reply = create_new_docling_document(prompt="test-document")

    assert isinstance(reply, NewDoclingDocumentOutput)
    key = reply.document_key
    assert key in local_document_cache
    match = re.match(r"[a-fA-F0-9]{32}$", key)
    assert match is not None
    assert reply.prompt == "test-document"

    return key


def test_table_in_html_format_to_docling_document(doc_key: str) -> None:
    html_table: str = (
        "<table><tr><th colspan='2'>Demographics</th></tr><tr><th>Name</th><th>Age"
        "</th></tr><tr><td>John</td><td rowspan='2'>30</td></tr><tr><td>Jane</td></tr>"
        "</table>"
    )

    reply = add_table_in_html_format_to_docling_document(
        document_key=doc_key,
        html_table=html_table,
        table_captions=["Table 2: Complex demographic data with merged cells"],
    )

    assert isinstance(reply, UpdateDocumentOutput)
    assert reply.document_key == doc_key


@pytest.fixture
def isolated_caches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _LRUCaches:
    caches = _LRUCaches(max_size=10)
    monkeypatch.setattr(generation_mod, "local_document_cache", caches.documents)
    monkeypatch.setattr(generation_mod, "get_cache_dir", lambda: tmp_path)
    return caches


@pytest.fixture
def paged_doc(isolated_caches: _LRUCaches) -> DoclingDocument:
    doc = DoclingDocument(name="paged")
    for page_no in (1, 2, 3):
        doc.add_page(page_no=page_no, size=Size(width=100, height=100))
    bbox = BoundingBox(l=0, t=0, r=10, b=10)
    for page_no, text in PAGE_TEXTS.items():
        doc.add_text(
            label=DocItemLabel.TEXT,
            text=text,
            prov=ProvenanceItem(page_no=page_no, bbox=bbox, charspan=(0, len(text))),
        )
    isolated_caches.put(PAGED_KEY, doc, [])
    return doc


def test_export_markdown_defaults_to_all_pages(paged_doc: DoclingDocument) -> None:
    full = paged_doc.export_to_markdown(image_mode=settings.image_export_mode)

    reply = export_docling_document_to_markdown(document_key=PAGED_KEY)

    assert isinstance(reply, ExportDocumentMarkdownOutput)
    assert reply.markdown == full
    assert all(text in full for text in PAGE_TEXTS.values())
    explicit = export_docling_document_to_markdown(document_key=PAGED_KEY, page_no=None)
    assert explicit.markdown == full


@pytest.mark.parametrize(
    ("page_no", "expected"), [(1, PAGE_TEXTS[1]), (2, ""), (3, PAGE_TEXTS[3])]
)
def test_export_markdown_of_a_single_page(
    paged_doc: DoclingDocument, page_no: int, expected: str
) -> None:
    reply = export_docling_document_to_markdown(document_key=PAGED_KEY, page_no=page_no)

    assert reply.markdown == expected


def test_export_markdown_truncates_after_page_selection(
    paged_doc: DoclingDocument,
) -> None:
    # A positional max_size still truncates the whole document.
    assert export_docling_document_to_markdown(PAGED_KEY, 5).markdown == "Alpha"

    reply = export_docling_document_to_markdown(PAGED_KEY, max_size=5, page_no=3)

    assert reply.markdown == "Gamma"


@pytest.mark.parametrize("page_no", [4, 99])
def test_export_markdown_rejects_missing_page(
    paged_doc: DoclingDocument, page_no: int
) -> None:
    with pytest.raises(ToolError, match=r"Available pages are: 1, 2, 3$"):
        export_docling_document_to_markdown(document_key=PAGED_KEY, page_no=page_no)


def test_export_markdown_rejects_page_of_unpaged_document(
    isolated_caches: _LRUCaches,
) -> None:
    doc = DoclingDocument.load_from_json(filename=DATA_DIR / "lorem_ipsum.docx.json")
    isolated_caches.put("docx", doc, [])

    with pytest.raises(ToolError, match="Available pages are: none"):
        export_docling_document_to_markdown(document_key="docx", page_no=1)
    assert export_docling_document_to_markdown(document_key="docx").markdown


def test_export_markdown_page_of_converted_document(
    isolated_caches: _LRUCaches,
) -> None:
    doc = DoclingDocument.load_from_json(filename=DATA_DIR / "2203.01017v2.json")
    isolated_caches.put("paper", doc, [])
    full = export_docling_document_to_markdown(document_key="paper").markdown

    page = export_docling_document_to_markdown(document_key="paper", page_no=5).markdown

    assert page
    assert len(page) < len(full)
    assert page == doc.export_to_markdown(
        image_mode=settings.image_export_mode, page_no=5
    )


def test_save_document_page_keeps_full_exports(
    paged_doc: DoclingDocument, tmp_path: Path
) -> None:
    full = save_docling_document(document_key=PAGED_KEY)
    page_1 = save_docling_document(document_key=PAGED_KEY, page_no=1)
    page_3 = save_docling_document(document_key=PAGED_KEY, page_no=3)

    assert isinstance(page_3, SaveDocumentOutput)
    assert full.md_file == str(tmp_path / f"{PAGED_KEY}.md")
    assert page_1.md_file == str(tmp_path / f"{PAGED_KEY}-p1.md")
    assert page_3.md_file == str(tmp_path / f"{PAGED_KEY}-p3.md")
    # Page saves leave the full markdown export in place.
    full_md = Path(full.md_file).read_text(encoding="utf-8")
    assert all(text in full_md for text in PAGE_TEXTS.values())
    assert Path(page_1.md_file).read_text(encoding="utf-8").strip() == PAGE_TEXTS[1]
    assert Path(page_3.md_file).read_text(encoding="utf-8").strip() == PAGE_TEXTS[3]
    # The JSON file always holds the full document.
    assert page_1.json_file == page_3.json_file == full.json_file
    saved = DoclingDocument.load_from_json(filename=Path(page_3.json_file))
    assert saved.export_to_dict() == paged_doc.export_to_dict()


def test_save_document_rejects_missing_page_without_writing(
    paged_doc: DoclingDocument, tmp_path: Path
) -> None:
    with pytest.raises(ToolError, match="page_no=4"):
        save_docling_document(document_key=PAGED_KEY, page_no=4)

    assert list(tmp_path.iterdir()) == []


@pytest.mark.anyio
async def test_page_no_schema(mcp_client: MCPClient) -> None:
    tools = {tool.name: tool for tool in await mcp_client.get_tools()}

    for name in ("export_docling_document_to_markdown", "save_docling_document"):
        schema = tools[name].input_schema
        page_no = schema["properties"]["page_no"]
        assert page_no["anyOf"] == [
            {"minimum": 1, "type": "integer"},
            {"type": "null"},
        ]
        assert page_no["default"] is None
        assert "1-based" in page_no["description"]
        assert "page_no" not in schema.get("required", [])


@pytest.mark.anyio
async def test_export_page_through_mcp(
    mcp_client: MCPClient, paged_doc: DoclingDocument
) -> None:
    tool = "export_docling_document_to_markdown"

    res = await mcp_client.call_tool(tool, {"document_key": PAGED_KEY, "page_no": 3})
    assert not res.is_error
    assert res.structured_content == {
        "document_key": PAGED_KEY,
        "markdown": PAGE_TEXTS[3],
    }

    res = await mcp_client.call_tool(tool, {"document_key": PAGED_KEY, "page_no": None})
    assert not res.is_error
    assert res.structured_content is not None
    assert res.structured_content["markdown"] == paged_doc.export_to_markdown(
        image_mode=settings.image_export_mode
    )

    # A missing page is reported to the client with the available pages.
    res = await mcp_client.call_tool(tool, {"document_key": PAGED_KEY, "page_no": 99})
    assert res.is_error
    assert isinstance(res.content[0], TextContent)
    assert "Available pages are: 1, 2, 3" in res.content[0].text

    for invalid in (0, -1, "abc"):
        res = await mcp_client.call_tool(
            tool, {"document_key": PAGED_KEY, "page_no": invalid}
        )
        assert res.is_error
        assert isinstance(res.content[0], TextContent)
        assert "validation error" in res.content[0].text


@pytest.mark.anyio
async def test_save_page_through_mcp(
    mcp_client: MCPClient, paged_doc: DoclingDocument, tmp_path: Path
) -> None:
    res = await mcp_client.call_tool(
        "save_docling_document", {"document_key": PAGED_KEY, "page_no": 3}
    )

    assert not res.is_error
    assert res.structured_content == {
        "md_file": str(tmp_path / f"{PAGED_KEY}-p3.md"),
        "json_file": str(tmp_path / f"{PAGED_KEY}.json"),
    }
