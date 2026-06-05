"""Prompts for generating and rewriting Docling documents."""

from docling_mcp.shared import mcp


@mcp.prompt()
def author_structured_document(topic: str, sections: str) -> str:
    """Create a new structured Docling document on a given topic.

    Args:
        topic: The subject of the document to write.
        sections: Comma-separated list of section headings to include.
    """
    return (
        f"Write a well-structured Docling document about '{topic}' "
        f"with the following sections: {sections}. "
        "Use this exact sequence of tools to build the document:\n"
        "1. create_new_docling_document — create a blank document and get the document_key.\n"
        "2. add_title_to_docling_document — set the document title.\n"
        "3. For each section:\n"
        "   a. add_section_heading_to_docling_document (use section_level=1 for top-level sections).\n"
        "   b. add_paragraph_to_docling_document — add body text for that section.\n"
        "   c. If bullet points are needed: open_list_in_docling_document, then "
        "add_list_items_to_list_in_docling_document, then close_list_in_docling_document.\n"
        "IMPORTANT: Always close an open list before calling add_section_heading_to_docling_document "
        "or add_paragraph_to_docling_document, otherwise the tool will return an error.\n"
        "4. save_docling_document — save the finished document and return the file paths."
    )


@mcp.prompt()
def convert_and_rewrite(source: str, instructions: str) -> str:
    """Convert a document and rewrite its content following specific instructions.

    Args:
        source: The URL or local file path to the source document.
        instructions: Rewriting instructions, e.g. 'translate to French' or
            'simplify for a non-technical audience'.
    """
    return (
        f"Convert the document at '{source}' and rewrite it following these instructions: "
        f"{instructions}\n\n"
        "Use this sequence of tools:\n"
        "1. convert_document_into_docling_document — convert and cache the source document.\n"
        "2. export_docling_document_to_markdown — read the full content of the original.\n"
        "3. create_new_docling_document — create a blank document for the rewritten version.\n"
        "4. Rebuild the content using add_title_to_docling_document, "
        "add_section_heading_to_docling_document, add_paragraph_to_docling_document, "
        "and (if needed) open_list_in_docling_document / add_list_items_to_list_in_docling_document "
        "/ close_list_in_docling_document — applying the rewrite instructions throughout.\n"
        "IMPORTANT: Always close an open list before adding a heading or paragraph.\n"
        "5. save_docling_document — persist the rewritten document and return the file paths."
    )
