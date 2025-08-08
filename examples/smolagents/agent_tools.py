import logging


from typing import Optional, Literal, List, Dict, Any
from pydantic import BaseModel, Field, validator

from smolagents.tools import Tool
from smolagents import (
    MCPClient,
    ToolCollection,
    Tool,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MCPConfig(BaseModel):
    """Configuration for MCP server connection."""

    connection_type: Literal["stdio", "sse"] = Field(
        default="sse", description="Type of MCP connection"
    )
    server_url: str = Field(
        default="http://localhost:8000/sse", description="URL for SSE MCP server"
    )
    command: str = Field(
        default="mcp-server-lls", description="Command to start stdio MCP server"
    )
    args: List[str] = Field(
        default_factory=list, description="Arguments for stdio MCP server"
    )


def setup_mcp_tools(config: MCPConfig) -> list[Tool]:
    """Setup MCP tools based on configuration."""
    if config.connection_type == "sse":
        server_parameters = {
            "url": config.server_url,
            "transport": "sse",
        }
        logger.info(f"Connecting to MCP server at {config.server_url}")
    else:
        server_parameters = StdioServerParameters(
            command=config.command,
            args=config.args,
            env=os.environ.copy(),
        )
        logger.info(f"Starting MCP server with command: {config.command}")

    try:
        mcp_client = MCPClient(server_parameters)
        tools = mcp_client.get_tools()

        logger.info(f"Successfully loaded {len(tools)} tools from MCP server")
        return tools
    except Exception as e:
        logger.error(f"Failed to connect to MCP server: {e}")
        raise


def test_docling_mcp_tools():
    config = MCPConfig()

    tools = setup_mcp_tools(config=config)

    for tool in tools:
        print(tool.name, "\t", tool.description)


def main():
    """Main function to run the demonstrations."""

    test_docling_mcp_tools()


if __name__ == "__main__":
    main()
