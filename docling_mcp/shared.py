"""Shared resources for the Docling MCP server.

Attributes:
    mcp: The MCPServer singleton used by all tool and prompt modules.
    local_document_cache: In-memory cache mapping document keys to
        `DoclingDocument` instances, shared across all tools.
    local_stack_cache: In-memory cache mapping document keys to lists of
        `NodeItem` instances, used by Llama Stack tools.
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.items.node import NodeItem

mcp: MCPServer = MCPServer("docling")

local_document_cache: dict[str, DoclingDocument] = {}
local_stack_cache: dict[str, list[NodeItem]] = {}
