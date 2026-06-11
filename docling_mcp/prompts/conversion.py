"""Prompts for converting documents with Docling."""

from docling_mcp.shared import mcp


@mcp.prompt()
def generate_docling_document_from_pdf(file_path: str) -> str:
    """Convert a local PDF file into a Docling document and return its document key.

    Args:
        file_path: The absolute or relative path to a local PDF file.
    """
    return (
        f"Convert the PDF file at '{file_path}' into a Docling document by calling "
        "convert_document_into_docling_document with the file path as the source. "
        "Once conversion is complete, return the document_key so I can use it with "
        "other tools. Also confirm whether the document was served from cache "
        "(from_cache=true) or freshly converted (from_cache=false)."
    )


@mcp.prompt()
def convert_and_summarize(source: str) -> str:
    """Convert a document and produce a structured summary.

    Args:
        source: The URL or local file path to the document.
    """
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


@mcp.prompt()
def convert_directory_and_list(directory: str) -> str:
    """Convert all files in a directory and list the results.

    Args:
        directory: The path to a local directory containing documents.
    """
    return (
        f"Convert all files in the directory '{directory}' by calling "
        "convert_directory_files_into_docling_document. "
        "Once the conversion is complete, present the results as a table with two columns: "
        "'Document Key' and 'From Cache'. "
        "Also report the total number of files converted and how many were served from cache."
    )
