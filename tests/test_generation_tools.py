import re

import pytest

from docling_mcp.logger import setup_logger
from docling_mcp.shared import local_document_cache
from docling_mcp.tools.generation import (
    NewDoclingDocumentOutput,
    UpdateDocumentOutput,
    add_table_in_html_format_to_docling_document,
    create_new_docling_document,
)

logger = setup_logger()


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
