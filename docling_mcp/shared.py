# shared.py
from mcp.server.fastmcp import FastMCP

# Create a single shared FastMCP instance
mcp = FastMCP("docling")

# Define your shared cache here if it's used by multiple tools
local_document_cache = {}
