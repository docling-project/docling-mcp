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
from docling_mcp.shared import local_document_cache, mcp

# Create a default project logger
logger = setup_logger()


def resolve(doc_key: str, anchor: str | RefItem) -> NodeItem:
    """Resolves a NodeItem in a Docling Document from its anchor (RefItem) reference."""
    ref: RefItem

    if isinstance(anchor, RefItem):
        ref = anchor
    else:
        ref = RefItem(cref=anchor)
    item = ref.resolve(local_document_cache[doc_key])
    if isinstance(item, NodeItem):
        return item

    raise ValueError("The anchor does not resolve to a NodeItem.")


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

    doc.add_text(
        label=DocItemLabel.TEXT,
        text=f"prompt: {prompt}",
        content_layer=ContentLayer.FURNITURE,
    )

    document_key = str(uuid.uuid4()).replace("-", "")

    local_document_cache[document_key] = doc

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
class DocumentUpdateOutput:
    """Output of the Docling document content generation tools."""

    document_key: Annotated[
        str,
        Field(description="The unique identifier of the document in the local cache."),
    ]

    anchor: Annotated[
        str | None,
        Field(
            description="The document anchor of the item that was updated or created."
        ),
    ]


@mcp.tool(title="Insert or append a title to Docling document")
def add_title_to_docling_document(
    document_key: Annotated[
        str,
        Field(description="The unique identifier of the document in the local cache."),
    ],
    title: Annotated[str, Field(description="The text of the new title.")],
    sibling_anchor: Annotated[
        str | None,
        Field(
            description="The anchor of the sibling item to insert the new title before/after."
        ),
    ] = None,
    insert_after: Annotated[
        bool,
        Field(
            description="Whether to insert the new title after the sibling item. Defaults to inserting before."
        ),
    ] = False,
    parent_anchor: Annotated[
        str | None,
        Field(
            description="The anchor of the parent item to insert the new title under."
        ),
    ] = None,
) -> DocumentUpdateOutput:
    """Insert a title by specifying sibling_anchor or append a title by specifying parent_anchor in a Docling Document object.

    If sibling_anchor is provided, the value of parent_anchor does not matter. Only omit sibling_anchor and specify parent_anchor
    when appending a title to the end of the document or a group. To append a title to the end of the entire document, do not specify either anchor.
    """
    if document_key not in local_document_cache:
        doc_keys = ", ".join(local_document_cache.keys())
        raise ValueError(
            f"document-key: {document_key} is not found. Existing document-keys are: {doc_keys}"
        )

    parent: NodeItem

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
            raise ValueError(f"Invalid sibling anchor: {sibling_anchor}.") from e

        if isinstance(parent, GroupItem):
            if (
                parent.label == GroupLabel.LIST
                or parent.label == GroupLabel.ORDERED_LIST
            ):
                raise ValueError(
                    "You are attempting to insert a title within a list, which is not allowed. Please choose a different location to insert the title"
                )

        item = local_document_cache[document_key].insert_title(
            sibling=sibling, text=title, after=insert_after
        )

        return DocumentUpdateOutput(document_key, item.self_ref)
    if parent_anchor:
        try:
            parent = resolve(document_key, parent_anchor)
        except ValueError as e:
            raise ValueError(f"Invalid parent anchor: {parent_anchor}.") from e

        if isinstance(parent, GroupItem):
            if (
                parent.label == GroupLabel.LIST
                or parent.label == GroupLabel.ORDERED_LIST
            ):
                raise ValueError(
                    "You are attempting to append a title within a list, which is not allowed. Please choose a different location to append the title"
                )

        item = local_document_cache[document_key].add_title(parent=parent, text=title)

        return DocumentUpdateOutput(document_key, item.self_ref)

    item = local_document_cache[document_key].add_title(text=title)

    return DocumentUpdateOutput(document_key, item.self_ref)


@mcp.tool(title="Insert or append a section heading to Docling document")
def add_section_heading_to_docling_document(
    document_key: Annotated[
        str,
        Field(description="The unique identifier of the document in the local cache."),
    ],
    section_heading: Annotated[
        str, Field(description="The text of the new section heading.")
    ],
    section_level: Annotated[
        LevelNumber,
        Field(
            description="The level of the new section heading, starting from 1, where 1 is the highest level."
        ),
    ],
    sibling_anchor: Annotated[
        str | None,
        Field(
            description="The anchor of the sibling item to insert the new section heading before/after."
        ),
    ] = None,
    insert_after: Annotated[
        bool,
        Field(
            description="Whether to insert the new section heading after the sibling item. Defaults to inserting before."
        ),
    ] = False,
    parent_anchor: Annotated[
        str | None,
        Field(
            description="The anchor of the parent item to insert the new section heading under."
        ),
    ] = None,
) -> DocumentUpdateOutput:
    """Insert a section heading by specifying sibling_anchor or append a section heading by specifying parent_anchor in a Docling Document object.

    If sibling_anchor is provided, the value of parent_anchor does not matter. Only omit sibling_anchor and specify parent_anchor
    when appending a section heading to the end of the document or a group. To append a section heading to the end of the entire document, do not specify either anchor.

    Section levels typically represent heading hierarchy (e.g., 1 for H1, 2 for H2).
    """
    if document_key not in local_document_cache:
        doc_keys = ", ".join(local_document_cache.keys())
        raise ValueError(
            f"document-key: {document_key} is not found. Existing document-keys are: {doc_keys}"
        )

    parent: NodeItem

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
            raise ValueError(f"Invalid sibling anchor: {sibling_anchor}.") from e

        if isinstance(parent, GroupItem):
            if (
                parent.label == GroupLabel.LIST
                or parent.label == GroupLabel.ORDERED_LIST
            ):
                raise ValueError(
                    "You are attempting to insert a section heading within a list, which is not allowed. Please choose a different location to insert the section heading"
                )

        item = local_document_cache[document_key].insert_heading(
            sibling=sibling,
            text=section_heading,
            level=section_level,
            after=insert_after,
        )

        return DocumentUpdateOutput(document_key, item.self_ref)
    if parent_anchor:
        try:
            parent = resolve(document_key, parent_anchor)
        except ValueError as e:
            raise ValueError(f"Invalid parent anchor: {parent_anchor}.") from e

        if isinstance(parent, GroupItem):
            if (
                parent.label == GroupLabel.LIST
                or parent.label == GroupLabel.ORDERED_LIST
            ):
                raise ValueError(
                    "You are attempting to append a section heading within a list, which is not allowed. Please choose a different location to append the section heading"
                )

        item = local_document_cache[document_key].add_heading(
            parent=parent, text=section_heading, level=section_level
        )

        return DocumentUpdateOutput(document_key, item.self_ref)

    item = local_document_cache[document_key].add_heading(
        text=section_heading, level=section_level
    )

    return DocumentUpdateOutput(document_key, item.self_ref)


@mcp.tool(title="Insert or append a paragraph to Docling document")
def add_paragraph_to_docling_document(
    document_key: Annotated[
        str,
        Field(description="The unique identifier of the document in the local cache."),
    ],
    paragraph: Annotated[str, Field(description="The text of the new paragraph.")],
    sibling_anchor: Annotated[
        str | None,
        Field(
            description="The anchor of the sibling item to insert the new paragraph before/after."
        ),
    ] = None,
    insert_after: Annotated[
        bool,
        Field(
            description="Whether to insert the new paragraph after the sibling item. Defaults to inserting before."
        ),
    ] = False,
    parent_anchor: Annotated[
        str | None,
        Field(
            description="The anchor of the parent item to insert the new paragraph under."
        ),
    ] = None,
) -> DocumentUpdateOutput:
    """Insert a paragraph by specifying sibling_anchor or append a paragraph by specifying parent_anchor in a Docling Document object.

    If sibling_anchor is provided, the value of parent_anchor does not matter. Only omit sibling_anchor and specify parent_anchor
    when appending a paragraph to the end of a group. To append a paragraph to the end of the entire document, do not specify either anchor.
    """
    if document_key not in local_document_cache:
        doc_keys = ", ".join(local_document_cache.keys())
        raise ValueError(
            f"document-key: {document_key} is not found. Existing document-keys are: {doc_keys}"
        )

    parent: NodeItem

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
            raise ValueError(f"Invalid sibling anchor: {sibling_anchor}.") from e

        if isinstance(parent, GroupItem):
            if (
                parent.label == GroupLabel.LIST
                or parent.label == GroupLabel.ORDERED_LIST
            ):
                raise ValueError(
                    "You are attempting to insert a paragraph within a list, which is not allowed. Please choose a different location to insert the paragraph"
                )

        item = local_document_cache[document_key].insert_text(
            sibling=sibling, text=paragraph, label=DocItemLabel.TEXT, after=insert_after
        )

        return DocumentUpdateOutput(document_key, item.self_ref)
    if parent_anchor:
        try:
            parent = resolve(document_key, parent_anchor)
        except ValueError as e:
            raise ValueError(f"Invalid parent anchor: {parent_anchor}.") from e

        if isinstance(parent, GroupItem):
            if (
                parent.label == GroupLabel.LIST
                or parent.label == GroupLabel.ORDERED_LIST
            ):
                raise ValueError(
                    "You are attempting to append a paragraph within a list, which is not allowed. Please choose a different location to append the paragraph"
                )

        item = local_document_cache[document_key].add_text(
            parent=parent, text=paragraph, label=DocItemLabel.TEXT
        )

        return DocumentUpdateOutput(document_key, item.self_ref)

    item = local_document_cache[document_key].add_text(
        text=paragraph, label=DocItemLabel.TEXT
    )

    return DocumentUpdateOutput(document_key, item.self_ref)


@mcp.tool(title="Insert or append a list group to Docling document")
def add_list_group_to_docling_document(
    document_key: Annotated[
        str,
        Field(description="The unique identifier of the document in the local cache."),
    ],
    sibling_anchor: Annotated[
        str | None,
        Field(
            description="The anchor of the sibling item to insert the new list group before/after."
        ),
    ] = None,
    insert_after: Annotated[
        bool,
        Field(
            description="Whether to insert the new list group after the sibling item. Defaults to inserting before."
        ),
    ] = False,
    parent_anchor: Annotated[
        str | None,
        Field(
            description="The anchor of the parent item to insert the new list group under."
        ),
    ] = None,
) -> DocumentUpdateOutput:
    """Insert a list group by specifying sibling_anchor or append a list group by specifying parent_anchor in a Docling Document object.

    If sibling_anchor is provided, the value of parent_anchor does not matter. Only omit sibling_anchor and specify parent_anchor
    when appending a list group to the end of the document or a group. To append a list group to the end of the entire document, do not specify either anchor.

    List items can only be added to a list group, so this tool is the necessary first step when attempting to create a new list.
    """
    if document_key not in local_document_cache:
        doc_keys = ", ".join(local_document_cache.keys())
        raise ValueError(
            f"document-key: {document_key} is not found. Existing document-keys are: {doc_keys}"
        )

    parent: NodeItem

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
            raise ValueError(f"Invalid sibling anchor: {sibling_anchor}.") from e

        item = local_document_cache[document_key].insert_list_group(
            sibling=sibling, after=insert_after
        )

        return DocumentUpdateOutput(document_key, item.self_ref)
    if parent_anchor:
        try:
            parent = resolve(document_key, parent_anchor)
        except ValueError as e:
            raise ValueError(f"Invalid parent anchor: {parent_anchor}.") from e

        item = local_document_cache[document_key].add_list_group(parent=parent)

        return DocumentUpdateOutput(document_key, item.self_ref)

    item = local_document_cache[document_key].add_list_group()

    return DocumentUpdateOutput(document_key, item.self_ref)


@dataclass
class ListItem:
    """A class to represent a list item pairing."""

    list_item_text: Annotated[str, Field(description="The text of a list item.")]
    list_marker_text: Annotated[str, Field(description="The marker of a list item.")]


@dataclass
class DocumentBatchUpdateOutput:
    """Output of the tools that update the Docling document with multiple items."""

    document_key: Annotated[
        str,
        Field(description="The unique identifier of the document in the local cache."),
    ]

    anchors: Annotated[
        list[str] | None,
        Field(
            description="A list of the document anchors of the items that were updated or created."
        ),
    ]


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
            description="The anchor of the sibling item to insert the new list items before/after."
        ),
    ] = None,
    insert_after: Annotated[
        bool,
        Field(
            description="Whether to insert the new list items after the sibling item. Defaults to inserting before."
        ),
    ] = False,
    parent_anchor: Annotated[
        str | None,
        Field(
            description="The anchor of the parent item to insert the new list items under."
        ),
    ] = None,
) -> DocumentBatchUpdateOutput:
    """Insert list items by specifying sibling_anchor or append list items by specifying parent_anchor in a Docling Document object.

    If sibling_anchor is provided, the value of parent_anchor does not matter. Only omit sibling_anchor and specify parent_anchor
    when appending list items to the end of a list group.

    List items will be added with their specified text and marker. When inserting new list items, marker values (such as numbers) will automatically be updated
    """
    if document_key not in local_document_cache:
        doc_keys = ", ".join(local_document_cache.keys())
        raise ValueError(
            f"document-key: {document_key} is not found. Existing document-keys are: {doc_keys}"
        )

    parent: NodeItem

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
            raise ValueError(f"Invalid sibling anchor: {sibling_anchor}.") from e

        if not isinstance(parent, GroupItem) or parent.label not in (
            GroupLabel.LIST,
            GroupLabel.ORDERED_LIST,
        ):
            raise ValueError(
                "You are attempting to insert list items outside of a list, which is not allowed. Please choose a different location to insert the list items."
            )

        refs = []

        for list_item in reversed(list_items):
            item = local_document_cache[document_key].insert_list_item(
                text=list_item.list_item_text,
                marker=list_item.list_marker_text,
                sibling=sibling,
                after=insert_after,
            )

            refs.append(item.self_ref)

        return DocumentBatchUpdateOutput(document_key, refs)
    if parent_anchor:
        try:
            parent = resolve(document_key, parent_anchor)
        except ValueError as e:
            raise ValueError(f"Invalid parent anchor: {parent_anchor}.") from e

        if not isinstance(parent, GroupItem) or parent.label not in (
            GroupLabel.LIST,
            GroupLabel.ORDERED_LIST,
        ):
            raise ValueError(
                "You are attempting to append list items outside of a list, which is not allowed. Please choose a different parent under which to append the list items."
            )

        refs = []

        for list_item in list_items:
            item = local_document_cache[document_key].add_list_item(
                text=list_item.list_item_text,
                marker=list_item.list_marker_text,
                parent=parent,
            )

            refs.append(item.self_ref)

        return DocumentBatchUpdateOutput(document_key, refs)

    raise ValueError(
        "List items much be added under a group (list or ordered list) parent. Thus, either a sibling_anchor or parent_anchor must be provided."
    )


@mcp.tool(title="Insert or append an HTML table to Docling document")
def add_table_in_html_format_to_docling_document(
    document_key: Annotated[
        str,
        Field(description="The unique identifier of the document in the local cache."),
    ],
    html_table: Annotated[
        str,
        Field(
            description="The HTML string representation of the new table.",
            examples=[
                "<table><tr><th>Name</th><th>Age</th></tr><tr><td>John</td><td>30</td></tr></table>",
                "<table><tr><th colspan='2'>Demographics</th></tr><tr><th>Name</th><th>Age</th></tr><tr><td>John</td><td rowspan='2'>30</td></tr><tr><td>Jane</td></tr></table>",
            ],
        ),
    ],
    sibling_anchor: Annotated[
        str | None,
        Field(
            description="The anchor of the sibling item to insert the new table before/after."
        ),
    ] = None,
    insert_after: Annotated[
        bool,
        Field(
            description="Whether to insert the new table after the sibling item. Defaults to inserting before."
        ),
    ] = False,
    parent_anchor: Annotated[
        str | None,
        Field(
            description="The anchor of the parent item to insert the new table under."
        ),
    ] = None,
    table_captions: Annotated[
        list[str] | None,
        Field(description="A list of caption strings to associate with the new table."),
    ] = None,
    table_footnotes: Annotated[
        list[str] | None,
        Field(
            description="A list of footnote strings to associate with the new table."
        ),
    ] = None,
) -> DocumentUpdateOutput:
    """Insert an HTML-formatted table by specifying sibling_anchor or append an HTML-formatted table by specifying parent_anchor in a Docling Document object.

    If sibling_anchor is provided, the value of parent_anchor does not matter. Only omit sibling_anchor and specify parent_anchor
    when appending an HTML-formatted table to the end of the document or a group. To append an HTML-formatted table to the end of the entire document, do not specify either anchor.

    This tool parses the provided HTML table string, converts it to a structured table
    representation, and inserts/appends it to the existing shared Docling Document
    object. It also supports optional captions and footnotes for the table.
    """
    if document_key not in local_document_cache:
        doc_keys = ", ".join(local_document_cache.keys())
        raise ValueError(
            f"document-key: {document_key} is not found. Existing document-keys are: {doc_keys}"
        )

    parent: NodeItem

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
                raise ValueError(f"Invalid sibling anchor: {sibling_anchor}.") from e

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

            return DocumentUpdateOutput(document_key, table.self_ref)
        if parent_anchor:
            try:
                parent = resolve(document_key, parent_anchor)
            except ValueError as e:
                raise ValueError(f"Invalid parent anchor: {parent_anchor}.") from e

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

            return DocumentUpdateOutput(document_key, table.self_ref)

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

        return DocumentUpdateOutput(document_key, table.self_ref)
    else:
        raise ValueError(
            "Could not parse the HTML string of the table! Please fix the HTML and try again!"
        )
