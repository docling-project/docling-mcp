"""Tools for generating Docling documents."""

import uuid
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
    NodeItem,
    RefItem,
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


def resolve(doc_key: str, anchor: str | RefItem) -> NodeItem:
    """Resolves a NodeItem in a Docling Document from its anchor (RefItem) reference."""
    ref: RefItem = None

    if isinstance(anchor, RefItem):
        ref = anchor
    else:
        ref = RefItem(cref=anchor)
    return ref.resolve(local_document_cache[doc_key])


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

    document_key = str(uuid.uuid4()).replace("-", "")

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


@mcp.tool(title="Insert or append a title to Docling document")
def add_title_to_docling_document(
    document_key: Annotated[
        str,
        Field(description="The unique identifier of the document in the local cache."),
    ],
    title: Annotated[
        str, Field(description="The title text to add or update to the document.")
    ],
    sibling_anchor: Annotated[
        str | None,
        Field(
            description="The anchor of the sibling item to insert the title before/after."
        ),
    ] = None,
    insert_after: Annotated[
        bool,
        Field(
            description="Whether to insert the title after the sibling item. Defaults to inserting before."
        ),
    ] = False,
    parent_anchor: Annotated[
        str | None,
        Field(description="The anchor of the parent item to insert the title under."),
    ] = None,
) -> UpdateDocumentOutput:
    """Insert a title by specifying sibling_anchor or append a title by specifying parent_anchor in a Docling Document object."""
    if document_key not in local_document_cache:
        doc_keys = ", ".join(local_document_cache.keys())
        raise ValueError(
            f"document-key: {document_key} is not found. Existing document-keys are: {doc_keys}"
        )

    if len(local_stack_cache[document_key]) == 0:
        raise ValueError(
            f"Stack size is zero for document with document-key: {document_key}. Abort document generation"
        )

    if sibling_anchor:
        try:
            sibling = resolve(document_key, sibling_anchor)

            if (
                sibling.parent is None
                or sibling.parent == local_document_cache[document_key].body.get_ref()
            ):
                parent = local_document_cache[document_key].body
            else:
                parent = resolve(document_key, sibling.parent)
        except ValueError as e:
            raise ValueError(f"Invalid sibling-anchor: {sibling_anchor}. ") from e

        if isinstance(parent, GroupItem):
            if (
                parent.label == GroupLabel.LIST
                or parent.label == GroupLabel.ORDERED_LIST
            ):
                raise ValueError(
                    "You are attempting to insert a title within a list, which is not allowed. Please choose a different location to insert the title"
                )

        local_document_cache[document_key].insert_title(
            sibling=sibling, text=title, after=insert_after
        )

        return UpdateDocumentOutput(document_key)
    if parent_anchor:
        try:
            parent = resolve(document_key, parent_anchor)
        except ValueError as e:
            raise ValueError(f"Invalid parent-anchor: {parent_anchor}. ") from e

        if isinstance(parent, GroupItem):
            if (
                parent.label == GroupLabel.LIST
                or parent.label == GroupLabel.ORDERED_LIST
            ):
                raise ValueError(
                    "You are attempting to append a title within a list, which is not allowed. Please choose a different location to append the title"
                )

        local_document_cache[document_key].add_title(parent=parent, text=title)

        return UpdateDocumentOutput(document_key)

    parent = local_stack_cache[document_key][-1]

    if isinstance(parent, GroupItem):
        if parent.label == GroupLabel.LIST or parent.label == GroupLabel.ORDERED_LIST:
            raise ValueError(
                "A list is currently opened. Please close the list before adding a title!"
            )

    item = local_document_cache[document_key].add_title(text=title)
    local_stack_cache[document_key][-1] = item

    return UpdateDocumentOutput(document_key)


@mcp.tool(title="Insert or append a section heading to Docling document")
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
    sibling_anchor: Annotated[
        str | None,
        Field(
            description="The anchor of the sibling item to insert the section heading before/after."
        ),
    ] = None,
    insert_after: Annotated[
        bool,
        Field(
            description="Whether to insert the section heading after the sibling item. Defaults to inserting before."
        ),
    ] = False,
    parent_anchor: Annotated[
        str | None,
        Field(
            description="The anchor of the parent item to insert the section heading under."
        ),
    ] = None,
) -> UpdateDocumentOutput:
    """Insert a section heading by specifying sibling_anchor or append a section heading by specifying parent_anchor in a Docling Document object.

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

    if sibling_anchor:
        try:
            sibling = resolve(document_key, sibling_anchor)

            if (
                sibling.parent is None
                or sibling.parent == local_document_cache[document_key].body.get_ref()
            ):
                parent = local_document_cache[document_key].body
            else:
                parent = resolve(document_key, sibling.parent)
        except ValueError as e:
            raise ValueError(f"Invalid sibling-anchor: {sibling_anchor}. ") from e

        if isinstance(parent, GroupItem):
            if (
                parent.label == GroupLabel.LIST
                or parent.label == GroupLabel.ORDERED_LIST
            ):
                raise ValueError(
                    "You are attempting to insert a section heading within a list, which is not allowed. Please choose a different location to insert the section heading"
                )

        local_document_cache[document_key].insert_heading(
            sibling=sibling,
            text=section_heading,
            level=section_level,
            after=insert_after,
        )

        return UpdateDocumentOutput(document_key)
    if parent_anchor:
        try:
            parent = resolve(document_key, parent_anchor)
        except ValueError as e:
            raise ValueError(f"Invalid parent-anchor: {parent_anchor}. ") from e

        if isinstance(parent, GroupItem):
            if (
                parent.label == GroupLabel.LIST
                or parent.label == GroupLabel.ORDERED_LIST
            ):
                raise ValueError(
                    "You are attempting to append a section heading within a list, which is not allowed. Please choose a different location to append the section heading"
                )

        local_document_cache[document_key].add_heading(
            parent=parent, text=section_heading, level=section_level
        )

        return UpdateDocumentOutput(document_key)

    parent = local_stack_cache[document_key][-1]

    if isinstance(parent, GroupItem):
        if parent.label == GroupLabel.LIST or parent.label == GroupLabel.ORDERED_LIST:
            raise ValueError(
                "A list is currently opened. Please close the list before adding a section heading!"
            )

    item = local_document_cache[document_key].add_heading(
        text=section_heading, level=section_level
    )
    local_stack_cache[document_key][-1] = item

    return UpdateDocumentOutput(local_document_cache[document_key].export_to_dict())


@mcp.tool(title="Insert or append a paragraph to Docling document")
def add_paragraph_to_docling_document(
    document_key: Annotated[
        str,
        Field(description="The unique identifier of the document in the local cache."),
    ],
    paragraph: Annotated[
        str, Field(description="The text content to add as a paragraph.")
    ],
    sibling_anchor: Annotated[
        str | None,
        Field(
            description="The anchor of the sibling item to insert the paragraph before/after."
        ),
    ] = None,
    insert_after: Annotated[
        bool,
        Field(
            description="Whether to insert the paragraph after the sibling item. Defaults to inserting before."
        ),
    ] = False,
    parent_anchor: Annotated[
        str | None,
        Field(
            description="The anchor of the parent item to insert the paragraph under."
        ),
    ] = None,
) -> UpdateDocumentOutput:
    """Insert a paragraph by specifying sibling_anchor or append a title by specifying parent_anchor in a Docling Document object."""
    if document_key not in local_document_cache:
        doc_keys = ", ".join(local_document_cache.keys())
        raise ValueError(
            f"document-key: {document_key} is not found. Existing document-keys are: {doc_keys}"
        )

    if len(local_stack_cache[document_key]) == 0:
        raise ValueError(
            f"Stack size is zero for document with document-key: {document_key}. Abort document generation"
        )

    if sibling_anchor:
        try:
            sibling = resolve(document_key, sibling_anchor)

            if (
                sibling.parent is None
                or sibling.parent == local_document_cache[document_key].body.get_ref()
            ):
                parent = local_document_cache[document_key].body
            else:
                parent = resolve(document_key, sibling.parent)
        except ValueError as e:
            raise ValueError(f"Invalid sibling-anchor: {sibling_anchor}. ") from e

        if isinstance(parent, GroupItem):
            if (
                parent.label == GroupLabel.LIST
                or parent.label == GroupLabel.ORDERED_LIST
            ):
                raise ValueError(
                    "You are attempting to insert a paragraph within a list, which is not allowed. Please choose a different location to insert the paragraph"
                )

        local_document_cache[document_key].insert_text(
            sibling=sibling, text=paragraph, label=DocItemLabel.TEXT, after=insert_after
        )

        return UpdateDocumentOutput(local_document_cache[document_key].export_to_dict())
    if parent_anchor:
        try:
            parent = resolve(document_key, parent_anchor)
        except ValueError as e:
            raise ValueError(f"Invalid parent-anchor: {parent_anchor}. ") from e

        if isinstance(parent, GroupItem):
            if (
                parent.label == GroupLabel.LIST
                or parent.label == GroupLabel.ORDERED_LIST
            ):
                raise ValueError(
                    "You are attempting to append a paragraph within a list, which is not allowed. Please choose a different location to append the paragraph"
                )

        local_document_cache[document_key].add_text(
            parent=parent, text=paragraph, label=DocItemLabel.TEXT
        )

        return UpdateDocumentOutput(local_document_cache[document_key].export_to_dict())

    parent = local_stack_cache[document_key][-1]

    if isinstance(parent, GroupItem):
        if parent.label == GroupLabel.LIST or parent.label == GroupLabel.ORDERED_LIST:
            raise ValueError(
                "A list is currently opened. Please close the list before adding a paragraph!"
            )

    item = local_document_cache[document_key].add_text(
        text=paragraph, label=DocItemLabel.TEXT
    )
    local_stack_cache[document_key][-1] = item

    return UpdateDocumentOutput(local_document_cache[document_key].export_to_dict())


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


@mcp.tool(title="Insert or append items to list in Docling document")
def add_list_items_to_list_in_docling_document(
    document_key: Annotated[
        str,
        Field(description="The unique identifier of the document in the local cache."),
    ],
    list_items: Annotated[
        list[ListItem],
        Field(description="A list of list_item_text and list_marker_text items."),
    ],
    sibling_anchor: Annotated[
        str | None,
        Field(
            description="The anchor of the sibling item to insert the list items before/after."
        ),
    ] = None,
    insert_after: Annotated[
        bool,
        Field(
            description="Whether to insert the list items after the sibling item. Defaults to inserting before."
        ),
    ] = False,
    parent_anchor: Annotated[
        str | None,
        Field(
            description="The anchor of the parent item to insert the list items under."
        ),
    ] = None,
) -> UpdateDocumentOutput:
    """Insert list items by specifying sibling_anchor or append list items by specifying parent_anchor in a Docling Document object.

    List items will be added with their specified text and marker.
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

    if sibling_anchor:
        try:
            sibling = resolve(document_key, sibling_anchor)

            if (
                sibling.parent is None
                or sibling.parent == local_document_cache[document_key].body.get_ref()
            ):
                parent = local_document_cache[document_key].body
            else:
                parent = resolve(document_key, sibling.parent)
        except ValueError as e:
            raise ValueError(f"Invalid sibling-anchor: {sibling_anchor}. ") from e

        if not isinstance(parent, GroupItem) or parent.label not in (
            GroupLabel.LIST,
            GroupLabel.ORDERED_LIST,
        ):
            raise ValueError(
                "You are attempting to insert list items outside of a list, which is not allowed. Please choose a different location to insert the list items."
            )

        for list_item in reversed(list_items):
            local_document_cache[document_key].insert_list_item(
                text=list_item.list_item_text,
                marker=list_item.list_marker_text,
                sibling=sibling,
                after=insert_after,
            )

        return UpdateDocumentOutput(document_key)
    if parent_anchor:
        try:
            parent = resolve(document_key, parent_anchor)
        except ValueError as e:
            raise ValueError(f"Invalid parent-anchor: {parent_anchor}. ") from e

        if not isinstance(parent, GroupItem) or parent.label not in (
            GroupLabel.LIST,
            GroupLabel.ORDERED_LIST,
        ):
            raise ValueError(
                "You are attempting to append list items outside of a list, which is not allowed. Please choose a different parent under which to append the list items."
            )

        for list_item in reversed(list_items):
            local_document_cache[document_key].add_list_item(
                text=list_item.list_item_text,
                marker=list_item.list_marker_text,
                parent=parent,
                after=insert_after,
            )

        return UpdateDocumentOutput(document_key)

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


@mcp.tool(title="Insert or append an HTML table to Docling document")
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
    sibling_anchor: Annotated[
        str | None,
        Field(
            description="The anchor of the sibling item to insert the table before/after."
        ),
    ] = None,
    insert_after: Annotated[
        bool,
        Field(
            description="Whether to insert the table after the sibling item. Defaults to inserting before."
        ),
    ] = False,
    parent_anchor: Annotated[
        str | None,
        Field(description="The anchor of the parent item to insert the table under."),
    ] = None,
    table_captions: Annotated[
        list[str] | None,
        Field(description="A list of caption strings to associate with the table.."),
    ] = None,
    table_footnotes: Annotated[
        list[str] | None,
        Field(description="A list of footnote strings to associate with the table."),
    ] = None,
) -> UpdateDocumentOutput:
    """Insert an HTML-formatted table by specifying sibling_anchor or append an HTML-formatted table by specifying parent_anchor in a Docling Document object.

    This tool parses the provided HTML table string, converts it to a structured table
    representation, and inserts/appends it to the existing shared Docling Document
    object. It also supports optional captions and footnotes for the table.
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

    html_doc: str = f"<html><body>{html_table}</body></html>"

    buff = BytesIO(html_doc.encode("utf-8"))
    doc_stream = DocumentStream(name="tmp", stream=buff)

    converter = DocumentConverter(allowed_formats=[InputFormat.HTML])
    conv_result: ConversionResult = converter.convert(doc_stream)

    if (
        conv_result.status == ConversionStatus.SUCCESS
        and len(conv_result.document.tables) > 0
    ):
        if sibling_anchor:
            try:
                sibling = resolve(document_key, sibling_anchor)

                if (
                    sibling.parent is None
                    or sibling.parent
                    == local_document_cache[document_key].body.get_ref()
                ):
                    parent = local_document_cache[document_key].body
                else:
                    parent = resolve(document_key, sibling.parent)
            except ValueError as e:
                raise ValueError(f"Invalid sibling-anchor: {sibling_anchor}. ") from e

            if isinstance(parent, GroupItem):
                if (
                    parent.label == GroupLabel.LIST
                    or parent.label == GroupLabel.ORDERED_LIST
                ):
                    raise ValueError(
                        "You are attempting to insert a table within a list, which is not allowed. Please choose a different location to insert the table"
                    )

            table = local_document_cache[document_key].insert_table(
                data=conv_result.document.tables[0].data,
                sibling=sibling,
                after=insert_after,
            )

            for _ in reversed(table_footnotes or []):
                footnote = local_document_cache[document_key].insert_text(
                    label=DocItemLabel.FOOTNOTE,
                    text=_,
                    sibling=table,
                    after=insert_after,
                )
                table.footnotes.insert(0, footnote.get_ref())

            for _ in reversed(table_captions or []):
                caption = local_document_cache[document_key].insert_text(
                    label=DocItemLabel.CAPTION,
                    text=_,
                    sibling=table,
                    after=insert_after,
                )
                table.captions.insert(0, caption.get_ref())

            return UpdateDocumentOutput(document_key)
        if parent_anchor:
            try:
                parent = resolve(document_key, parent_anchor)
            except ValueError as e:
                raise ValueError(f"Invalid parent-anchor: {parent_anchor}. ") from e

            if isinstance(parent, GroupItem):
                if (
                    parent.label == GroupLabel.LIST
                    or parent.label == GroupLabel.ORDERED_LIST
                ):
                    raise ValueError(
                        "You are attempting to append a title within a list, which is not allowed. Please choose a different location to append the title"
                    )

            table = local_document_cache[document_key].add_table(
                data=conv_result.document.tables[0].data, parent=parent
            )

            for _ in table_captions or []:
                caption = local_document_cache[document_key].add_text(
                    label=DocItemLabel.CAPTION, text=_, parent=parent
                )
                table.captions.append(caption.get_ref())

            for _ in table_footnotes or []:
                footnote = local_document_cache[document_key].add_text(
                    label=DocItemLabel.FOOTNOTE, text=_, parent=parent
                )
                table.footnotes.append(footnote.get_ref())

            return UpdateDocumentOutput(document_key)

        parent = local_stack_cache[document_key][-1]

        if isinstance(parent, GroupItem):
            if (
                parent.label == GroupLabel.LIST
                or parent.label == GroupLabel.ORDERED_LIST
            ):
                raise ValueError(
                    "A list is currently opened. Please close the list before adding a table!"
                )

        table = local_document_cache[document_key].add_table(
            data=conv_result.document.tables[0].data
        )

        for _ in table_captions or []:
            caption = local_document_cache[document_key].add_text(
                label=DocItemLabel.CAPTION, text=_
            )
            table.captions.append(caption.get_ref())

        for _ in table_footnotes or []:
            footnote = local_document_cache[document_key].add_text(
                label=DocItemLabel.FOOTNOTE, text=_
            )
            table.footnotes.append(footnote.get_ref())

        local_stack_cache[document_key][-1] = table

        return UpdateDocumentOutput(document_key)
    else:
        raise ValueError(
            "Could not parse the html string of the table! Please fix the html and try again!"
        )
