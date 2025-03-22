import re

# Create a default project logger
from docling_mcp.logger import setup_logger
from docling_mcp.shared import local_document_cache
from docling_mcp.tools.generation import (  # noqa: F401
    add_listitem_to_list_in_docling_document,
    add_paragraph_to_docling_document,
    add_section_heading_to_docling_document,
    add_table_in_html_format_to_docling_document,
    add_title_to_docling_document,
    close_list_in_docling_document,
    create_new_docling_document,
    export_docling_document_to_markdown,
    open_list_in_docling_document,
    save_docling_document,
)

logger = setup_logger()


def test_create_docling_document():
    reply = create_new_docling_document(prompt="test-document")
    key = extract_key_from_reply(reply=reply)

    assert key in local_document_cache


def extract_key_from_reply(reply: str) -> str:
    match = re.search(r"document-key:\s*([a-fA-F0-9]{32})", reply)
    if match:
        return match.group(1)

    return "<key-not-found>"


def test_table_in_html_format_to_docling_document():
    reply = create_new_docling_document(prompt="test-document")
    key = extract_key_from_reply(reply=reply)

    html_table: str = "<table><tr><th colspan='2'>Demographics</th></tr><tr><th>Name</th><th>Age</th></tr><tr><td>John</td><td rowspan='2'>30</td></tr><tr><td>Jane</td></tr></table>"

    reply = add_table_in_html_format_to_docling_document(
        document_key=key,
        html_table=html_table,
        table_captions=["Table 2: Complex demographic data with merged cells"],
    )

    assert reply == f"Added table to a document with key: {key}"
