"""Glue between the MCP protocol layer and ``AllowedRootsRegistry``.

The MCP Python SDK's ``Server.notification_handlers`` dispatch site
(``mcp.server.lowlevel.server.Server._handle_notification``) does **not**
propagate the active ``ServerSession`` to handler callables — only the
notification itself is passed. To call ``session.list_roots()`` from inside
a ``notifications/roots/list_changed`` handler we therefore have to capture
the session from the surrounding ``_handle_message`` scope into a
``ContextVar``. That's what this module does.

The patch is applied once at server startup via :func:`install_roots_handlers`;
it is a no-op if invoked twice on the same server instance.
"""

from __future__ import annotations

import contextvars
from typing import TYPE_CHECKING, Any

from mcp.types import (
    InitializedNotification,
    RootsListChangedNotification,
)

from docling_mcp.logger import setup_logger
from docling_mcp.shared import allowed_roots, mcp

if TYPE_CHECKING:
    from mcp.server.session import ServerSession

logger = setup_logger()

# ContextVar that holds the active ServerSession during message dispatch.
# Notification handlers fish the session out of here when they need to issue
# server-to-client requests like ``roots/list``.
_active_session: contextvars.ContextVar[ServerSession | None] = contextvars.ContextVar(
    "docling_mcp_active_session", default=None
)

# Sentinel attribute we set on the patched bound method so a second
# install_roots_handlers() call is a no-op.
_PATCH_MARKER = "_docling_mcp_roots_patched"


async def _refresh_from_client(session: ServerSession) -> None:
    """Pull the current roots list from the client and rebuild the registry."""
    caps = session.client_params.capabilities if session.client_params else None
    if caps is None or caps.roots is None:
        logger.debug("client has no roots capability; skipping list_roots()")
        return
    try:
        result = await session.list_roots()
    except Exception:
        logger.exception("session.list_roots() failed")
        return
    allowed_roots.update_from_client_roots([str(r.uri) for r in result.roots])


async def _on_initialized(notify: InitializedNotification) -> None:
    """Seed the registry from the client's roots once the handshake completes."""
    session = _active_session.get()
    if session is None:
        logger.warning("notifications/initialized fired with no active session")
        return
    await _refresh_from_client(session)


async def _on_roots_list_changed(notify: RootsListChangedNotification) -> None:
    """Refresh the registry when the client signals roots have changed."""
    session = _active_session.get()
    if session is None:
        logger.warning("notifications/roots/list_changed fired with no active session")
        return
    await _refresh_from_client(session)


def install_roots_handlers() -> None:
    """Wire roots notification handlers onto the shared FastMCP server.

    Idempotent — safe to call more than once.
    """
    server = mcp._mcp_server  # access the underlying low-level Server

    # Register notification handlers (keyed by type, per the SDK).
    server.notification_handlers[InitializedNotification] = _on_initialized
    server.notification_handlers[RootsListChangedNotification] = _on_roots_list_changed

    # Patch _handle_message to expose session via the ContextVar.
    if getattr(server._handle_message, _PATCH_MARKER, False):
        return

    original = server._handle_message

    async def _handle_message_with_session(
        message: Any,
        session: Any,
        lifespan_context: Any,
        raise_exceptions: bool = False,
    ) -> Any:
        token = _active_session.set(session)
        try:
            return await original(message, session, lifespan_context, raise_exceptions)
        finally:
            _active_session.reset(token)

    setattr(_handle_message_with_session, _PATCH_MARKER, True)
    server._handle_message = _handle_message_with_session  # type: ignore[method-assign]
    logger.info("roots handlers installed (initialized + list_changed)")
