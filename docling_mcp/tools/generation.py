"""Tools for generating Docling documents."""

import hashlib
from dataclasses import dataclass
from io import BytesIO
from typing import Annotated

from pydantic import Field

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.document import (
    ConversionResult,
)
from docling.document_converter import DocumentConverter
from docling_core.types.doc.document import (
    ContentLayer,
    DoclingDocument,
    GroupItem,
    LevelNumber,
)
from docling_core.types.doc.labels import (
    DocItemLabel,
    GroupLabel,
)
from docling_core.types.io import DocumentStream

from docling_mcp.docling_cache import get_cache_dir
from docling_mcp.logger import setup_logger
from docling_mcp.shared import local_document_cache, local_stack_cache, mcp

# Create a default project logger
logger = setup_logger()


def hash_string_md5(input_string: str) -> str:
    """Creates an md5 hash-string from the input string."""
    return hashlib.md5(input_string.encode()).hexdigest()


@dataclass
class NewDoclingDocumentOutput:
    """Output of the create_new_docling_document tool."""

    document_key: Annotated[
        str, Field(description="The unique key that identifies the new document.")
    ]
    prompt: Annotated[str, Field(description="The original prompt.")]


@mcp.tool(title="Create new Docling document")
def create_new_docling_document(
    prompt: Annotated[
        str, Field(description="The prompt text to include in the new document.")
    ],
) -> NewDoclingDocumentOutput:
    """Create a new Docling document from a provided prompt string.

    This function generates a new document in the local document cache with the
    provided prompt text. The document is assigned a unique key derived from an MD5
    hash of the prompt text.
    """
    doc = DoclingDocument(name="Generated Document")

    item = doc.add_text(
        label=DocItemLabel.TEXT,
        text=f"prompt: {prompt}",
        content_layer=ContentLayer.FURNITURE,
    )

    document_key = hash_string_md5(prompt)

    local_document_cache[document_key] = doc
    local_stack_cache[document_key] = [item]

    return NewDoclingDocumentOutput(document_key, prompt)


@dataclass
class ExportDocumentMarkdownOutput:
    """Output of the export_docling_document_to_markdown tool."""

    document_key: Annotated[
        str,
        Field(description="The unique identifier of the document in the local cache."),
    ]
    markdown: Annotated[
        str, Field(description="The representation of the document in markdown format.")
    ]


@mcp.tool(title="Export Docling document to markdown format")
def export_docling_document_to_markdown(
    document_key: Annotated[
        str,
        Field(description="The unique identifier of the document in the local cache."),
    ],
) -> ExportDocumentMarkdownOutput:
    """Export a document from the local document cache to markdown format.

    This tool converts a Docling document that exists in the local cache into
    a markdown formatted string, which can be used for display or further processing.
    """
    if document_key not in local_document_cache:
        doc_keys = ", ".join(local_document_cache.keys())
        raise ValueError(
            f"document-key: {document_key} is not found. Existing document-keys are: {doc_keys}"
        )

    markdown = local_document_cache[document_key].export_to_markdown()

    return ExportDocumentMarkdownOutput(document_key, markdown)


@dataclass
class SaveDocumentOutput:
    """Output of the save_docling_document tool."""

    md_file: Annotated[
        str,
        Field(
            description="The path in the cache directory to the file in markdown format."
        ),
    ]
    json_file: Annotated[
        str,
        Field(
            description="The path in the cache directory to the file in JSON format."
        ),
    ]


@mcp.tool(title="Save Docling document")
def save_docling_document(
    document_key: Annotated[
        str,
        Field(description="The unique identifier of the document in the local cache."),
    ],
) -> SaveDocumentOutput:
    """Save a document from the local document cache to disk in both markdown and JSON formats.

    This tool takes a document that exists in the local cache and saves it to the specified
    cache directory with filenames based on the document key. Both markdown and JSON versions
    of the document are saved.
    """
    if document_key not in local_document_cache:
        doc_keys = ", ".join(local_document_cache.keys())
        raise ValueError(
            f"document-key: {document_key} is not found. Existing document-keys are: {doc_keys}"
        )

    cache_dir = get_cache_dir()
    md_file = str(cache_dir / f"{document_key}.md")
    json_file = str(cache_dir / f"{document_key}.json")

    local_document_cache[document_key].save_as_markdown(filename=md_file, text_width=72)
    local_document_cache[document_key].save_as_json(filename=json_file)

    return SaveDocumentOutput(md_file, json_file)


@dataclass
class UpdateDocumentOutput:
    """Output of the Docling document content generation tools."""

    document_key: Annotated[
        str,
        Field(description="The unique identifier of the document in the local cache."),
    ]


@mcp.tool(title="Add or update title to Docling document")
def add_title_to_docling_document(
    document_key: Annotated[
        str,
        Field(description="The unique identifier of the document in the local cache."),
    ],
    title: Annotated[
        str, Field(description="The title text to add or update to the document.")
    ],
) -> UpdateDocumentOutput:
    """Add or update the title of a document in the local document cache.

    This tool modifies an existing document that has already been processed
    and stored in the local cache. It requires that the document already exists
    in the cache before a title can be added.
    """
    if document_key not in local_document_cache:
        doc_keys = ", ".join(local_document_cache.keys())
        raise ValueError(
            f"document-key: {document_key} is not found. Existing document-keys are: {doc_keys}"
        )

    if len(local_stack_cache[document_key]) == 0:
        raise ValueError(
            f"Stack size is zero for document with document-key: {document_key}. Abort document generation"
        )

    parent = local_stack_cache[document_key][-1]

    if isinstance(parent, GroupItem):
        if parent.label == GroupLabel.LIST or parent.label == GroupLabel.ORDERED_LIST:
            raise ValueError(
                "A list is currently opened. Please close the list before adding a title!"
            )

    item = local_document_cache[document_key].add_title(text=title)
    local_stack_cache[document_key][-1] = item

    return UpdateDocumentOutput(document_key)


@mcp.tool(title="Add section heading to Docling document")
def add_section_heading_to_docling_document(
    document_key: Annotated[
        str,
        Field(description="The unique identifier of the document in the local cache."),
    ],
    section_heading: Annotated[
        str, Field(description="The text to use for the section heading.")
    ],
    section_level: Annotated[
        LevelNumber,
        Field(
            description="The level of the heading, starting from 1, where 1 is the highest level."
        ),
    ],
) -> UpdateDocumentOutput:
    """Add a section heading to an existing document in the local document cache.

    This tool inserts a section heading with the specified heading text and level
    into a document that has already been processed and stored in the local cache.
    Section levels typically represent heading hierarchy (e.g., 1 for H1, 2 for H2).
    """
    if document_key not in local_document_cache:
        doc_keys = ", ".join(local_document_cache.keys())
        raise ValueError(
            f"document-key: {document_key} is not found. Existing document-keys are: {doc_keys}"
        )

    if len(local_stack_cache[document_key]) == 0:
        raise ValueError(
            f"Stack size is zero for document with document-key: {document_key}. Abort document generation"
        )

    parent = local_stack_cache[document_key][-1]

    if isinstance(parent, GroupItem):
        if parent.label == GroupLabel.LIST or parent.label == GroupLabel.ORDERED_LIST:
            raise ValueError(
                "A list is currently opened. Please close the list before adding a section-heading!"
            )

    item = local_document_cache[document_key].add_heading(
        text=section_heading, level=section_level
    )
    local_stack_cache[document_key][-1] = item

    return UpdateDocumentOutput(document_key)


@mcp.tool(title="Add paragraph to Docling document")
def add_paragraph_to_docling_document(
    document_key: Annotated[
        str,
        Field(description="The unique identifier of the document in the local cache."),
    ],
    paragraph: Annotated[
        str, Field(description="The text content to add as a paragraph.")
    ],
) -> UpdateDocumentOutput:
    """Add a paragraph of text to an existing document in the local document cache.

    This tool inserts a new paragraph under the specified section header and level
    into a document that has already been processed and stored in the cache.
    """
    if document_key not in local_document_cache:
        doc_keys = ", ".join(local_document_cache.keys())
        raise ValueError(
            f"document-key: {document_key} is not found. Existing document-keys are: {doc_keys}"
        )

    if len(local_stack_cache[document_key]) == 0:
        raise ValueError(
            f"Stack size is zero for document with document-key: {document_key}. Abort document generation"
        )

    parent = local_stack_cache[document_key][-1]

    if isinstance(parent, GroupItem):
        if parent.label == GroupLabel.LIST or parent.label == GroupLabel.ORDERED_LIST:
            raise ValueError(
                "A list is currently opened. Please close the list before adding a paragraph!"
            )

    item = local_document_cache[document_key].add_text(
        label=DocItemLabel.TEXT, text=paragraph
    )
    local_stack_cache[document_key][-1] = item

    return UpdateDocumentOutput(document_key)


@mcp.tool(title="Open list in Docling document")
def open_list_in_docling_document(
    document_key: Annotated[
        str,
        Field(description="The unique identifier of the document in the local cache."),
    ],
) -> UpdateDocumentOutput:
    """Open a new list group in an existing document in the local document cache.

    This tool creates a new list structure within a document that has already been
    processed and stored in the local cache. It requires that the document already exists
    and that there is at least one item in the document's stack cache.
    """
    if document_key not in local_document_cache:
        doc_keys = ", ".join(local_document_cache.keys())
        raise ValueError(
            f"document-key: {document_key} is not found. Existing document-keys are: {doc_keys}"
        )

    if len(local_stack_cache[document_key]) == 0:
        raise ValueError(
            f"Stack size is zero for document with document-key: {document_key}. Abort document generation"
        )

    item = local_document_cache[document_key].add_group(label=GroupLabel.LIST)
    local_stack_cache[document_key].append(item)

    return UpdateDocumentOutput(document_key)


@mcp.tool(title="Close list in Docling document")
def close_list_in_docling_document(
    document_key: Annotated[
        str,
        Field(description="The unique identifier of the document in the local cache."),
    ],
) -> UpdateDocumentOutput:
    """Closes a list group in an existing document in the local document cache.

    This tool closes a previously opened list structure within a document.
    It requires that the document exists and that there is more than one item
    in the document's stack cache.
    """
    if document_key not in local_document_cache:
        doc_keys = ", ".join(local_document_cache.keys())
        raise ValueError(
            f"document-key: {document_key} is not found. Existing document-keys are: {doc_keys}"
        )

    if len(local_stack_cache[document_key]) <= 1:
        raise ValueError(
            f"Stack size is zero for document with document-key: {document_key}. Abort document generation"
        )

    local_stack_cache[document_key].pop()

    return UpdateDocumentOutput(document_key)


@dataclass
class ListItem:
    """A class to represent a list item pairing."""

    list_item_text: Annotated[str, Field(description="The text of a list item.")]
    list_marker_text: Annotated[str, Field(description="The marker of a list item.")]


@mcp.tool(title="Add items to list in Docling document")
def add_list_items_to_list_in_docling_document(
    document_key: Annotated[
        str,
        Field(description="The unique identifier of the document in the local cache."),
    ],
    list_items: Annotated[
        list[ListItem],
        Field(description="A list of list_item_text and list_marker_text items."),
    ],
) -> UpdateDocumentOutput:
    """Add list items to an open list in an existing document in the local document cache.

    This tool inserts new list items with the specified text and marker into an
    open list within a document. It requires that the document exists and that
    there is at least one item in the document's stack cache.
    """
    if document_key not in local_document_cache:
        doc_keys = ", ".join(local_document_cache.keys())
        raise ValueError(
            f"document-key: {document_key} is not found. Existing document-keys are: {doc_keys}"
        )

    if len(local_stack_cache[document_key]) == 0:
        raise ValueError(
            f"Stack size is zero for document with document-key: {document_key}. Abort document generation"
        )

    parent = local_stack_cache[document_key][-1]

    if isinstance(parent, GroupItem):
        if parent.label != GroupLabel.LIST and parent.label != GroupLabel.ORDERED_LIST:
            raise ValueError(
                "No list is currently opened. Please open a list before adding list-items!"
            )
    else:
        raise ValueError(
            "No list is currently opened. Please open a list before adding list-items!"
        )

    for list_item in list_items:
        local_document_cache[document_key].add_list_item(
            text=list_item.list_item_text,
            marker=list_item.list_marker_text,
            parent=parent,
        )

    return UpdateDocumentOutput(document_key)


@mcp.tool(title="Add HTML table to Docling document")
def add_table_in_html_format_to_docling_document(
    document_key: Annotated[
        str,
        Field(description="The unique identifier of the document in the local cache."),
    ],
    html_table: Annotated[
        str,
        Field(
            description="The HTML string representation of the table to add.",
            examples=[
                "<table><tr><th>Name</th><th>Age</th></tr><tr><td>John</td><td>30</td></tr></table>",
                "<table><tr><th colspan='2'>Demographics</th></tr><tr><th>Name</th><th>Age</th></tr><tr><td>John</td><td rowspan='2'>30</td></tr><tr><td>Jane</td></tr></table>",
            ],
        ),
    ],
    table_captions: Annotated[
        list[str] | None,
        Field(description="A list of caption strings to associate with the table.."),
    ] = None,
    table_footnotes: Annotated[
        list[str] | None,
        Field(description="A list of footnote strings to associate with the table."),
    ] = None,
) -> UpdateDocumentOutput:
    """Add an HTML-formatted table to an existing document in the local document cache.

    This tool parses the provided HTML table string, converts it to a structured table
    representation, and adds it to the specified document. It also supports optional
    captions and footnotes for the table.
    """
    if document_key not in local_document_cache:
        doc_keys = ", ".join(local_document_cache.keys())
        raise ValueError(
            f"document-key: {document_key} is not found. Existing document-keys are: {doc_keys}"
        )

    doc = local_document_cache[document_key]

    if len(local_stack_cache[document_key]) == 0:
        raise ValueError(
            f"Stack size is zero for document with document-key: {document_key}. Abort document generation"
        )

    html_doc: str = f"<html><body>{html_table}</body></html>"

    buff = BytesIO(html_doc.encode("utf-8"))
    doc_stream = DocumentStream(name="tmp", stream=buff)

    converter = DocumentConverter(allowed_formats=[InputFormat.HTML])
    conv_result: ConversionResult = converter.convert(doc_stream)

    if (
        conv_result.status == ConversionStatus.SUCCESS
        and len(conv_result.document.tables) > 0
    ):
        table = doc.add_table(data=conv_result.document.tables[0].data)

        for _ in table_captions or []:
            caption = doc.add_text(label=DocItemLabel.CAPTION, text=_)
            table.captions.append(caption.get_ref())

        for _ in table_footnotes or []:
            footnote = doc.add_text(label=DocItemLabel.FOOTNOTE, text=_)
            table.footnotes.append(footnote.get_ref())
    else:
        raise ValueError(
            "Could not parse the html string of the table! Please fix the html and try again!"
        )

    return UpdateDocumentOutput(document_key)
