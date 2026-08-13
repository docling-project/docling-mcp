"""Prompts for converting documents with Docling."""

from typing import Annotated

from pydantic import Field

from docling_mcp.shared import mcp


@mcp.prompt(title="Convert document")
def generate_docling_document_from_pdf(
    file_path: Annotated[
        str,
        Field(
            description="The absolute or relative path to a local file (PDF, DOCX, XLSX, HTML, Markdown, EPUB, …)."
        ),
    ],
) -> str:
    """Convert a local file into a Docling document and return its document key."""
    return (
        f"Convert the file at '{file_path}' into a Docling document by calling "
        "convert_document_into_docling_document with the file path as the source. "
        "Once conversion is complete, return the document_key so I can use it with "
        "other tools. Also confirm whether the document was served from cache "
        "(from_cache=true) or freshly converted (from_cache=false)."
    )


@mcp.prompt(title="Convert and summarize")
def convert_and_summarize(
    source: Annotated[
        str, Field(description="The URL or local file path to the document.")
    ],
) -> str:
    """Convert a document and produce a structured summary."""
    return (
        f"Convert the document at '{source}' by calling "
        "convert_document_into_docling_document. "
        "Once you have the document_key, export the document to markdown using "
        "export_docling_document_to_markdown. "
        "Then produce a structured summary that includes: "
        "(1) the document title, "
        "(2) a list of the main sections, and "
        "(3) 3 to 5 key takeaways from the content."
    )


@mcp.prompt(title="Convert directory")
def convert_directory_and_list(
    directory: Annotated[
        str, Field(description="The path to a local directory containing documents.")
    ],
) -> str:
    """Convert all files in a directory and list the results."""
    return (
        f"Convert all files in the directory '{directory}' by calling "
        "convert_directory_files_into_docling_document. "
        "Once the conversion is complete, present the results as a table with two columns: "
        "'Document Key' and 'From Cache'. "
        "Also report the total number of files converted and how many were served from cache."
    )
