"""This module initializes and runs the Docling MCP server."""

import enum
import os

import typer

from docling_mcp.logger import setup_logger
from docling_mcp.shared import mcp
from docling_mcp.tools.conversion import (
    convert_pdf_document_into_json_docling_document_from_uri_path,
    is_document_in_local_cache,
)
from docling_mcp.tools.generation import (
    add_list_items_to_list_in_docling_document,
    add_paragraph_to_docling_document,
    add_section_heading_to_docling_document,
    add_title_to_docling_document,
    close_list_in_docling_document,
    create_new_docling_document,
    export_docling_document_to_markdown,
    open_list_in_docling_document,
    save_docling_document,
)

if (
    os.getenv("RAG_ENABLED") == "true"
    and os.getenv("OLLAMA_MODEL") != ""
    and os.getenv("EMBEDDING_MODEL") != ""
):
    from docling_mcp.tools.applications import (
        export_docling_document_to_vector_db,
        search_documents,
    )

app = typer.Typer()


class TansportType(str, enum.Enum):
    """List of available protocols."""

    STDIO = "stdio"
    SSE = "sse"


@app.command()
def main(
    transport: TansportType = TansportType.STDIO,
    sse_port: int = 8000,
) -> None:
    """Initialize and run the Docling MCP server."""
    # Create a default project logger
    logger = setup_logger()
    logger.info("starting up Docling MCP-server ...")

    # Initialize and run the server
    mcp.settings.port = sse_port
    mcp.run(transport=transport.value)


if __name__ == "__main__":
    app()
