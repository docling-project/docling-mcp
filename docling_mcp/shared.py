"""This module defines shared resources."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.items.node import NodeItem

# The FastMCP singleton.  In normal server operation it is replaced by
# `init_mcp()` (called from `docling_mcp.servers.mcp_server.main`) *before*
# any tool or prompt module is imported, so the SDK's DNS-rebinding-protection
# logic runs with the correct bind address.
#
# The default instance below exists solely to satisfy imports that happen
# outside of `main()` (e.g. direct unit-test imports of tool modules).  It
# uses the SDK default host ("127.0.0.1") so DNS-rebinding protection is
# enabled, which is the safe default.
mcp: FastMCP = FastMCP("docling")


def init_mcp(host: str, port: int) -> FastMCP:
    """Create and register the shared FastMCP instance.

    Must be called once, before any tool or prompt module is imported.
    Passing ``host`` and ``port`` at construction time lets the MCP SDK apply
    its own DNS-rebinding-protection logic with the correct values:

    - ``localhost`` / ``127.0.0.1`` / ``::1`` → protection enabled,
      allowlist restricted to loopback addresses.
    - Any other host (e.g. ``0.0.0.0``) → protection disabled, which is the
      safe default for container / Kubernetes deployments where the network
      perimeter is enforced externally.

    Args:
        host: The hostname or IP address the server will bind to.
        port: The TCP port the server will listen on.

    Returns:
        The newly created FastMCP instance, also stored as the module-level
        ``mcp`` name so that tool and prompt decorators resolve correctly.
    """
    global mcp
    mcp = FastMCP("docling", host=host, port=port)
    return mcp


# Define your shared cache here if it's used by multiple tools
local_document_cache: dict[str, DoclingDocument] = {}
local_stack_cache: dict[str, list[NodeItem]] = {}
