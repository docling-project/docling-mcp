"""Prompts for editing and searching Docling documents."""

from typing import Annotated

from pydantic import Field

from docling_mcp.shared import mcp


@mcp.prompt(title="Review and edit")
def review_and_edit_document(
    document_key: Annotated[
        str,
        Field(description="The unique identifier of the document in the local cache."),
    ],
) -> str:
    """Review the structure of a cached document and interactively edit it."""
    return (
        f"I want to review and edit the document with key '{document_key}'.\n\n"
        "Start by calling get_overview_of_document_anchors to display the full document "
        "structure, including all anchor references (e.g. #/texts/2, #/tables/0).\n"
        "Present the structure clearly, then ask me which items I want to change and how.\n\n"
        "For each requested change, use the appropriate tool:\n"
        "- update_text_of_document_item_at_anchor — to replace the text at a specific anchor.\n"
        "- delete_document_items_at_anchors — to remove one or more items by anchor.\n"
        "- get_text_of_document_item_at_anchor — to inspect an item's text before editing.\n\n"
        "After all edits are complete, offer to save the document with save_docling_document."
    )


@mcp.prompt(title="Find and replace")
def find_and_replace_in_document(
    document_key: Annotated[
        str,
        Field(description="The unique identifier of the document in the local cache."),
    ],
    search_text: Annotated[str, Field(description="The text to search for.")],
    replacement_text: Annotated[str, Field(description="The text to replace it with.")],
) -> str:
    """Find text in a cached document and replace it at the correct anchor."""
    return (
        f"In the document with key '{document_key}', find '{search_text}' "
        f"and replace it with '{replacement_text}'.\n\n"
        "Use this two-step approach:\n"
        f"1. Call search_for_text_in_document_anchors with document_key='{document_key}' "
        f"and text='{search_text}' to locate all matching anchors.\n"
        "2. For each matching anchor, call update_text_of_document_item_at_anchor with "
        f"the anchor and updated_text='{replacement_text}'.\n\n"
        "If the search returns only keyword matches (not exact matches), show me the candidates "
        "and confirm which ones to update before making any changes."
    )
