from docling_mcp.logger import setup_logger
from docling_mcp.shared import mcp

if __name__ == "__main__":
    # Create a default project logger
    logger = setup_logger()
    logger.info("starting up Docling MCP-server ...")

    # Initialize and run the server
    mcp.run(transport="stdio")
