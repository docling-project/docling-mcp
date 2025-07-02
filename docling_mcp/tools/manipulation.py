"""Tools for manipulating Docling documents."""

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


@mcp.tool()
def get_overview_of_document_anchors(document_key: str) -> str:
    """Retrieves a structured overview of a document from the local document cache.

    This tool returns a text representation of the document's structure, showing
    the hierarchy and types of elements within the document. Each line in the
    output includes the document anchor reference and item label.

    Args:
        document_key (str): The unique identifier for the document in the local cache.

    Returns:
        str: A string containing the hierarchical structure of the document with
             indentation to show nesting levels, along with anchor references.

    Raises:
        ValueError: If the specified document_key does not exist in the local cache.

    Example:
        get_overview_of_document_anchors(document_key="doc123")
    """
    if document_key not in local_document_cache:
        doc_keys = ", ".join(local_document_cache.keys())
        raise ValueError(
            f"document-key: {document_key} is not found. Existing document-keys are: {doc_keys}"
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

    return "\n".join(lines)


@mcp.tool()
def get_text_of_document_item_at_anchor(document_key: str, document_anchor: str) -> str:
    """Retrieves the text content of a specific document item identified by its anchor.

    This tool extracts the text from a document item at the specified anchor location
    within a document that exists in the local document cache.

    Args:
        document_key (str): The unique identifier for the document in the local cache.
        document_anchor (str): The anchor reference that identifies the specific item
                               within the document.

    Returns:
        str: A formatted string containing the text content of the specified item,
             wrapped in code block formatting.

    Raises:
        ValueError: If the specified document_key does not exist in the local cache.
        ValueError: If the item at the specified anchor is not a textual item.

    Example:
        get_text_of_document_item_at_anchor(document_key="doc123", document_anchor="#/texts/2")
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

    return f"The text of {document_anchor} for document-key with {document_key} is:\n\n```{text}```\n\n"


@mcp.tool()
def update_text_of_document_item_at_anchor(
    document_key: str, document_anchor: str, updated_text: str
) -> str:
    """Updates the text content of a specific document item identified by its anchor.

    This tool modifies the text of an existing document item at the specified anchor
    location within a document that exists in the local document cache.

    Args:
        document_key (str): The unique identifier for the document in the local cache.
        document_anchor (str): The anchor reference that identifies the specific item
                               within the document.
        updated_text (str): The new text content to replace the existing content.

    Returns:
        str: A confirmation message indicating the text was successfully updated.

    Raises:
        ValueError: If the specified document_key does not exist in the local cache.
        ValueError: If the item at the specified anchor is not a textual item.

    Example:
        update_text_of_document_item_at_anchor(document_key="doc123", document_anchor="#/texts/2", updated_text="This is the new content.")
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
        item.text = updated_text
    else:
        raise ValueError(
            f"Item at {document_anchor} for document-key: {document_key} is not a textual item."
        )

    return f"Updated the text at {document_anchor} for document with key {document_key}"


@mcp.tool()
def delete_document_items_at_anchors(
    document_key: str, document_anchors: list[str]
) -> str:
    """Deletes multiple document items identified by their anchors.

    This tool removes specified items from a document that exists in the local
    document cache, based on their anchor references.

    Args:
        document_key (str): The unique identifier for the document in the local cache.
        document_anchors (list[str]): A list of anchor references identifying the items
                                      to be deleted from the document.

    Returns:
        str: A confirmation message indicating the items were successfully deleted.

    Raises:
        ValueError: If the specified document_key does not exist in the local cache.

    Example:
        delete_document_items_at_anchors(document_key="doc123", document_anchors=["#/texts/2", "#/tables/1"])
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

    return f"Deleted the {document_anchors} for document with key {document_key}"
