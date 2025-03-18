from docling_mcp.shared import mcp
from docling_mcp.tools import (
    conversion,
)  # This imports your tools module, registering the tools

from docling_mcp.logger import setup_logger

if __name__ == "__main__":
    # Create a default project logger
    logger = setup_logger()

    logger.info("starting up Docling MCP-server ...")

    # Initialize and run the server
    mcp.run(transport="stdio")
