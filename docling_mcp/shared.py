from typing import List, Dict, Union
from docling_core.types.doc.document import (DoclingDocument, NodeItem, DocItem, GroupItem)

from mcp.server.fastmcp import FastMCP

# Create a single shared FastMCP instance
mcp = FastMCP("docling")

# Define your shared cache here if it's used by multiple tools
local_document_cache: Dict[str, DoclingDocument] = {}
local_stack_cache: Dict[str, List[NodeItem] ] = {}
