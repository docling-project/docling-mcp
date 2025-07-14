"""Tools for manipulating Docling documents."""

import re
from dataclasses import dataclass
from typing import Annotated

from pydantic import Field

from docling_core.types.doc.document import (
    DocItem,
    GroupItem,
    RefItem,
    SectionHeaderItem,
    TextItem,
    TitleItem,
)

from docling_mcp.logger import setup_logger
from docling_mcp.shared import local_document_cache, mcp

# Create a default project logger
logger = setup_logger()


# TODO: Provide a proper structure instead of a single string
@dataclass
class DocumentAnchorOutput:
    """Output of the get_overview_of_document_anchors tool."""

    structure: Annotated[
        str,
        Field(
            description=(
                "A string containing the hierarchical structure of the document with "
                "indentation to show nesting levels, along with anchor references."
            )
        ),
    ]


@mcp.tool(title="Get overview of Docling document anchors")
def get_overview_of_document_anchors(
    document_key: Annotated[
        str,
        Field(description="The unique identifier of the document in the local cache."),
    ],
) -> DocumentAnchorOutput:
    """Retrieve a structured overview of a document from the local document cache.

    This tool returns a text representation of the Docling document's structure,
    showing the hierarchy and types of elements within the document. Each line in the
    output includes the document anchor reference and item label.
    """
    if document_key not in local_document_cache:
        doc_keys = ", ".join(local_document_cache.keys())
        raise ValueError(
            f"document-key: {document_key} is not found. Existing document-keys are: "
            f"{doc_keys}"
        )

    doc = local_document_cache[document_key]

    lines = []
    slevel = 0
    for item, level in doc.iterate_items():
        ref = item.get_ref()

        if isinstance(item, DocItem):
            if isinstance(item, TitleItem):
                lines.append(f"[anchor:{ref.cref}] {item.label}: {item.text}")

            elif isinstance(item, SectionHeaderItem):
                slevel = item.level
                indent = "  " * (level + slevel)
                lines.append(
                    f"{indent}[anchor:{ref.cref}] {item.label}-{level}: {item.text}"
                )

            else:
                indent = "  " * (level + slevel + 1)
                lines.append(f"{indent}[anchor:{ref.cref}] {item.label}")

        elif isinstance(item, GroupItem):
            indent = "  " * (level + slevel + 1)
            lines.append(f"{indent}[anchor:{ref.cref}] {item.label}")

    return DocumentAnchorOutput("\n".join(lines))


@dataclass
class TextSearchOutput:
    """Output of the search_for_text_in_document_anchors tool."""

    result: Annotated[
        str,
        Field(
            description=(
                "A string listing the result of searching for text in the document's "
                "anchors. If matches were found, the result indicates what text matched "
                "at which anchors, along with the number of occurrences. If no matches "
                "were found, the result indicates that no matches were found."
            )
        ),
    ]


@mcp.tool(title="Search for text in Docling document anchors")
def search_for_text_in_document_anchors(
    document_key: Annotated[
        str,
        Field(description="The unique identifier of the document in the local cache."),
    ],
    text: Annotated[
        str,
        Field(
            description="The string of text to search for in the document's anchors."
        ),
    ],
) -> TextSearchOutput:
    """Search for specific text and keywords within a document's anchors.

    This tool takes a string of text to search for and returns a string of all
    document anchors that contain the exact text. The search is case-insensitive.
    If the exact text is not found, the tool will search for individual keywords
    within the text, splitting it on non-alphanumeric characters. If keywords
    are found, they are listed alongside their number of occurrences in parentheses.
    """
    if document_key not in local_document_cache:
        doc_keys = ", ".join(local_document_cache.keys())
        raise ValueError(
            f"document-key: {document_key} is not found. Existing document-keys are: {doc_keys}"
        )

    doc = local_document_cache[document_key]
    exact_matches = []
    matches = []
    keywords_set = {word for word in set(re.findall(r"\b\w+\b", text.lower())) if word}

    for item, _ in doc.iterate_items():
        if isinstance(item, TextItem):
            ref = item.get_ref()

            if text.lower() in item.text.lower():
                exact_matches.append(f"[anchor:{ref.cref}]")

            if not exact_matches:
                keyword_occurrences: dict[str, int] = {}
                total_matches = 0

                strings = re.findall(r"\b\w+\b", item.text.lower())

                for string in strings:
                    if string in keywords_set:
                        total_matches += 1
                        if string in keyword_occurrences:
                            keyword_occurrences[string] += 1
                        else:
                            keyword_occurrences[string] = 1

                if keyword_occurrences:
                    matches.append(
                        (
                            f"[anchor:{ref.cref}] keyword matches ({total_matches} total):{','.join([f' {k} ({v} occurrences)' for k, v in sorted(keyword_occurrences.items(), key=lambda x: x[1], reverse=True)])}",
                            total_matches,
                        )
                    )

    if exact_matches:
        return TextSearchOutput(
            "Found exact text matches in the following anchors:\n"
            + "\n".join(exact_matches)
        )
    if matches:
        return TextSearchOutput(
            "No exact text matches were found. Found individual keyword matches in the following anchors:\n"
            + "\n".join(
                [
                    match[0]
                    for match in sorted(
                        matches, key=lambda match: match[1], reverse=True
                    )
                ]
            )
        )
    return TextSearchOutput(
        f"No exact text matches nor individual keyword matches found for '{text}' in document with key {document_key}."
    )


@dataclass
class DocumentItemText:
    """Text content of a Docling document item."""

    text: Annotated[
        str,
        Field(description="The text content of an item in a Docling document."),
    ]


@mcp.tool(title="Get text of Docling document item at anchor")
def get_text_of_document_item_at_anchor(
    document_key: Annotated[
        str,
        Field(description="The unique identifier of the document in the local cache."),
    ],
    document_anchor: Annotated[
        str,
        Field(
            description=(
                "The anchor reference that identifies the specific item within the "
                "document."
            ),
            examples=["#/texts/2"],
        ),
    ],
) -> DocumentItemText:
    """Retrieve the text content of a specific document item identified by its anchor.

    This tool extracts the text from a Docling document item at the specified anchor
    location within a document that exists in the local document cache.
    """
    if document_key not in local_document_cache:
        doc_keys = ", ".join(local_document_cache.keys())
        raise ValueError(
            f"document-key: {document_key} is not found. Existing document-keys are: {doc_keys}"
        )

    doc = local_document_cache[document_key]

    ref = RefItem(cref=document_anchor)
    item = ref.resolve(doc=doc)

    if isinstance(item, TextItem):
        text = item.text
    else:
        raise ValueError(
            f"Item at {document_anchor} for document-key: {document_key} is not a textual item."
        )

    return DocumentItemText(text)


@mcp.tool(title="Update text of Docling document item at anchor")
def update_text_of_document_item_at_anchor(
    document_key: Annotated[
        str,
        Field(description="The unique identifier of the document in the local cache."),
    ],
    document_anchor: Annotated[
        str,
        Field(
            description=(
                "The anchor reference that identifies the specific item within the "
                "document."
            ),
            examples=["#/texts/6"],
        ),
    ],
    updated_text: Annotated[
        str,
        Field(description="The new text content to replace the existing content."),
    ],
) -> bool:
    """Update the text content of a specific document item identified by its anchor.

    This tool modifies the text of an existing document item at the specified anchor
    location within a document that exists in the local document cache. It returns
    True if the update was successful.
    """
    if document_key not in local_document_cache:
        doc_keys = ", ".join(local_document_cache.keys())
        raise ValueError(
            f"document-key: {document_key} is not found. Existing document-keys are: "
            f"{doc_keys}"
        )

    doc = local_document_cache[document_key]

    ref = RefItem(cref=document_anchor)
    item = ref.resolve(doc=doc)

    if isinstance(item, TextItem):
        item.text = updated_text
    else:
        raise ValueError(
            f"Item at {document_anchor} for document-key: {document_key} is not a "
            "textual item."
        )

    return True


@mcp.tool(title="Delete Docling document items at anchors")
def delete_document_items_at_anchors(
    document_key: Annotated[
        str,
        Field(description="The unique identifier of the document in the local cache."),
    ],
    document_anchors: Annotated[
        list[str],
        Field(
            description=(
                "A list of anchor references identifying the items to be deleted from the "
                "document."
            ),
            examples=["#/texts/2", "#/tables/1"],
        ),
    ],
) -> bool:
    """Delete multiple document items identified by their anchors.

    This tool removes specified items from a Docling document that exists in the local
    document cache, based on their anchor references. It returns True if the all the
    items were successfully removed.
    """
    if document_key not in local_document_cache:
        doc_keys = ", ".join(local_document_cache.keys())
        raise ValueError(
            f"document-key: {document_key} is not found. Existing document-keys are: {doc_keys}"
        )

    doc = local_document_cache[document_key]

    items = []
    for _ in document_anchors:
        ref = RefItem(cref=_)
        items.append(ref.resolve(doc=doc))

    doc.delete_items(node_items=items)

    return True
