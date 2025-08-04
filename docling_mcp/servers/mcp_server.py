"""This module initializes and runs the Docling MCP server."""

import enum
from typing import Annotated

import typer

from docling_mcp.logger import setup_logger
from docling_mcp.shared import mcp

app = typer.Typer()


class ToolGroups(str, enum.Enum):
    """List of available toolsgroups."""

    CONVERSION = "conversion"
    GENERATION = "generation"
    MANIPULATION = "manipulation"
    LLAMA_INDEX_RAG = "llama-index-rag"


class TransportType(str, enum.Enum):
    """List of available protocols."""

    STDIO = "stdio"
    SSE = "sse"
    STREAMABLE_HTTP = "streamable-http"


_DEFAULT_TOOLS = [ToolGroups.CONVERSION, ToolGroups.GENERATION, ToolGroups.MANIPULATION]


@app.command()
def main(
    transport: TransportType = TransportType.STDIO,
    host: str = "localhost",
    port: int = 8000,
    tools: Annotated[
        list[ToolGroups] | None,
        typer.Argument(
            help=f"Tools to be loaded in the server. The default list is {', '.join(_DEFAULT_TOOLS)}"
        ),
    ] = None,
) -> None:
    """Initialize and run the Docling MCP server."""
    # Create a default project logger
    logger = setup_logger()

    if tools is None:
        tools = [*_DEFAULT_TOOLS]

    if ToolGroups.CONVERSION in tools:
        logger.info("loading conversion tools...")
        import docling_mcp.tools.conversion

    if ToolGroups.GENERATION in tools:
        logger.info("loading generation tools...")
        import docling_mcp.tools.generation

    if ToolGroups.MANIPULATION in tools:
        logger.info("loading manipulation tools...")
        import docling_mcp.tools.manipulation

    if ToolGroups.LLAMA_INDEX_RAG in tools:
        logger.info("loading Llama Index RAG tools...")
        import docling_mcp.tools.applications

    # Initialize and run the server
    logger.info("starting up Docling MCP-server ...")
    mcp.settings.host = host
    mcp.settings.port = port
    mcp.run(transport=transport.value)


if __name__ == "__main__":
    main()
