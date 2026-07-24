"""Shared resources for the Docling MCP server.

Attributes:
    mcp: The FastMCP singleton used by all tool and prompt modules.
        A default instance is created at import time so that tool modules
        can import the name unconditionally. In server operation it is
        replaced by `init_mcp()` before any tool or prompt module is
        imported, ensuring the MCP SDK's DNS-rebinding-protection logic
        runs with the correct bind address.
    local_document_cache: In-memory cache mapping document keys to
        `DoclingDocument` instances, shared across all tools.
    local_stack_cache: In-memory cache mapping document keys to lists of
        `NodeItem` instances, used by Llama Stack tools.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.items.node import NodeItem

mcp: FastMCP = FastMCP("docling")


def init_mcp(host: str, port: int) -> FastMCP:
    """Create and register the shared FastMCP instance.

    Must be called once, before any tool or prompt module is imported.
    Passing `host` and `port` at construction time lets the MCP SDK apply
    its own DNS-rebinding-protection logic with the correct values:

    - `localhost` / `127.0.0.1` / `::1` → protection enabled,
      allowlist restricted to loopback addresses.
    - Any other host (e.g. `0.0.0.0`) → protection disabled, which is the
      safe default for container / Kubernetes deployments where the network
      perimeter is enforced externally.

    Args:
        host: The hostname or IP address the server will bind to.
        port: The TCP port the server will listen on.

    Returns:
        The newly created FastMCP instance, also stored as the module-level
        `mcp` name so that tool and prompt decorators resolve correctly.
    """
    global mcp
    mcp = FastMCP("docling", host=host, port=port)
    return mcp


local_document_cache: dict[str, DoclingDocument] = {}
local_stack_cache: dict[str, list[NodeItem]] = {}
