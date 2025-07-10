"""This module initializes and runs the Docling MCP server."""

import enum
import os

import typer

import docling_mcp.tools.conversion
import docling_mcp.tools.generation
import docling_mcp.tools.manipulation
from docling_mcp.logger import setup_logger
from docling_mcp.shared import mcp

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


class TransportType(str, enum.Enum):
    """List of available protocols."""

    STDIO = "stdio"
    SSE = "sse"
    STREAMABLE_HTTP = "streamable-http"


@app.command()
def main(
    transport: TransportType = TransportType.STDIO,
    http_port: int = 8000,
) -> None:
    """Initialize and run the Docling MCP server."""
    # Create a default project logger
    logger = setup_logger()
    logger.info("starting up Docling MCP-server ...")

    # Initialize and run the server
    mcp.settings.port = http_port
    mcp.run(transport=transport.value)


if __name__ == "__main__":
    main()
