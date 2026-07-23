"""This module defines shared resources."""

from collections import defaultdict

from mcp.server.fastmcp import FastMCP

from docling_core.types.doc.items.node import NodeItem

from docling_mcp.store.base import DocumentStoreProtocol
from docling_mcp.store.factory import create_document_store

# Create a single shared FastMCP instance
mcp = FastMCP("docling")

# Shared document cache used by multiple tools. Converted documents are
# persisted across server restarts unless persistence is disabled.
local_document_cache: DocumentStoreProtocol = create_document_store()

# In-progress authoring state; memory-only by design. A defaultdict so that
# documents recovered from the persistent store (whose stack did not survive
# the restart) fail with a clear empty-stack error instead of a KeyError.
local_stack_cache: dict[str, list[NodeItem]] = defaultdict(list)
