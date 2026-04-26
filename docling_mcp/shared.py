"""This module defines shared resources."""

from mcp.server.fastmcp import FastMCP

from docling_core.types.doc.document import (
    DoclingDocument,
    NodeItem,
)

# Create a single shared FastMCP instance
mcp = FastMCP("docling")

# Server-side state for the MCP Roots capability. Populated at startup
# from --allowed-directories and refreshed at runtime from
# notifications/roots/list_changed. See docling_mcp.roots.
from docling_mcp.roots import AllowedRootsRegistry  # noqa: E402

allowed_roots = AllowedRootsRegistry()

# Define your shared cache here if it's used by multiple tools
local_document_cache: dict[str, DoclingDocument] = {}
local_stack_cache: dict[str, list[NodeItem]] = {}
