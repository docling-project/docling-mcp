"""Roots-aware allowed-paths registry for the Docling MCP server.

This module implements the server-side state for the Model Context Protocol
``roots`` capability (see ``modelcontextprotocol/servers/tree/main/src/filesystem``
for the canonical reference implementation in TypeScript). It tracks the set
of filesystem directories the active client has authorized the server to
access.

Semantics, per the canonical MCP docs:

    "MCP clients that support Roots can dynamically update the Allowed
     directories. Roots notified by Client to Server, completely replace
     any server-side Allowed directories when provided."

So when the client sends a ``roots/list_changed`` notification, the registry
*replaces* (not unions) its allowed set. When the client never sends roots,
the registry falls back to the static set seeded from the
``--allowed-directories`` CLI flag, preserving backward compatibility for
clients that don't speak the roots protocol.
"""

from __future__ import annotations

import threading
from pathlib import Path
from urllib.parse import unquote, urlparse

from docling_mcp._path_utils import is_remote_url, to_filesystem_path
from docling_mcp.logger import setup_logger

logger = setup_logger()


class AllowedRootsRegistry:
    """Thread-safe registry of filesystem paths the server is allowed to read.

    The registry has two layers:

    * ``_static_roots`` — populated once at startup from the
      ``--allowed-directories`` CLI flag. Used only when no client roots
      have been received.
    * ``_client_roots`` — populated dynamically from
      ``roots/list_changed`` notifications. When non-empty, this set fully
      replaces the static set as the authoritative allowed list.

    Path comparisons are done on resolved absolute paths so that symlinks,
    ``..``, and ``~`` do not bypass authorization.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._static_roots: set[Path] = set()
        self._client_roots: set[Path] | None = None

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------
    def set_static_roots(self, raw_paths: list[str]) -> None:
        """Replace the static-root set from ``--allowed-directories`` values."""
        resolved = {Path(p).expanduser().resolve() for p in raw_paths}
        with self._lock:
            self._static_roots = resolved
        logger.info(f"static allowed-roots seeded: {sorted(str(p) for p in resolved)}")

    def update_from_client_roots(self, roots_uris: list[str]) -> None:
        """Replace the client-root set from a ``ListRootsResult``.

        ``roots_uris`` is the list of ``Root.uri`` strings the client returned
        — typically ``file://`` URLs. Non-``file://`` roots are ignored with
        a warning, since the server only authorizes filesystem access.
        """
        new_set: set[Path] = set()
        for uri in roots_uris:
            parsed = urlparse(uri)
            if parsed.scheme.lower() != "file":
                logger.warning(
                    f"ignoring non-file root from client: {uri!r} "
                    "(only file:// roots constrain filesystem access)"
                )
                continue
            # Decode percent-escapes so file:///path/with%20space matches
            # the real filesystem path. Clients (Claude Desktop, Cowork)
            # frequently send roots like file:///Users/x/Library/Application%20Support/...
            new_set.add(Path(unquote(parsed.path)).expanduser().resolve())

        with self._lock:
            self._client_roots = new_set
        logger.info(
            f"client allowed-roots refreshed ({len(new_set)} entries): "
            f"{sorted(str(p) for p in new_set)}"
        )

    def clear_client_roots(self) -> None:
        """Drop the client-root set, falling back to static roots."""
        with self._lock:
            self._client_roots = None
        logger.info("client allowed-roots cleared; falling back to static roots")

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------
    @property
    def is_unconstrained(self) -> bool:
        """Return True iff no client roots AND no static roots are configured.

        In this state the server validates nothing — preserves the existing
        behavior for users who haven't opted into the roots protocol or
        passed ``--allowed-directories``.
        """
        with self._lock:
            return self._client_roots is None and not self._static_roots

    def active_roots(self) -> set[Path]:
        """Return the currently-authoritative allowed-paths set.

        Client roots, when present, fully replace the static set per the
        canonical MCP semantics quoted at module top.
        """
        with self._lock:
            if self._client_roots is not None:
                return set(self._client_roots)
            return set(self._static_roots)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate_source(self, source: str) -> None:
        """Raise ``PermissionError`` if ``source`` is not under any allowed root.

        Remote URLs (http/https/ftp/etc.) are passed through unchecked —
        roots authorize filesystem access, not network access.

        If the registry is unconstrained (no client roots, no static
        roots), this is a no-op, preserving the pre-roots behavior of the
        server.
        """
        if is_remote_url(source):
            return

        if self.is_unconstrained:
            return

        target = to_filesystem_path(source)
        roots = self.active_roots()
        for root in roots:
            try:
                target.relative_to(root)
            except ValueError:
                continue
            return

        raise PermissionError(
            f"path {str(target)!r} is not under any allowed root. "
            f"active roots: {sorted(str(r) for r in roots)}"
        )
