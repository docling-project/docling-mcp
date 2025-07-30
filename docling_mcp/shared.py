"""This module defines shared resources."""

from mcp.server.fastmcp import FastMCP

from docling_core.types.doc.document import (
    DoclingDocument,
    NodeItem,
)

# Create a single shared FastMCP instance
mcp = FastMCP("docling")

# Define your shared cache here if it's used by multiple tools
local_document_cache: dict[str, DoclingDocument] = {}
local_stack_cache: dict[str, list[NodeItem]] = {}
