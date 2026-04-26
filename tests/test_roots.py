"""Tests for the MCP Roots protocol implementation.

These tests exercise the registry, the path-utility helpers, and the
notification-wiring scaffold in isolation. They do NOT spin up a real MCP
server end-to-end — that's covered by the manual local-test matrix
documented in the PR description (Claude Desktop + Claude Code via uvx).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from docling_mcp._path_utils import is_remote_url, to_filesystem_path
from docling_mcp.roots import AllowedRootsRegistry


# ---------------------------------------------------------------------------
# _path_utils
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/foo.pdf",
        "https://example.com/foo.pdf",
        "ftp://example.com/foo.pdf",
        "ftps://example.com/foo.pdf",
        "s3://bucket/key.pdf",
        "HTTPS://example.com/CASE.pdf",  # case-insensitive
    ],
)
def test_is_remote_url_true(url: str) -> None:
    assert is_remote_url(url) is True


@pytest.mark.parametrize(
    "src",
    [
        "/tmp/foo.pdf",
        "relative/foo.pdf",
        "~/Documents/foo.pdf",
        "file:///tmp/foo.pdf",  # file:// is filesystem, not remote
        "file://localhost/tmp/foo.pdf",
    ],
)
def test_is_remote_url_false(src: str) -> None:
    assert is_remote_url(src) is False


def test_to_filesystem_path_plain() -> None:
    p = to_filesystem_path("/tmp/foo.pdf")
    assert p.is_absolute()
    assert str(p).endswith("foo.pdf")


def test_to_filesystem_path_file_url() -> None:
    p = to_filesystem_path("file:///tmp/foo.pdf")
    assert p.is_absolute()
    assert str(p) == "/tmp/foo.pdf"


def test_to_filesystem_path_rejects_remote() -> None:
    with pytest.raises(ValueError, match="remote URL"):
        to_filesystem_path("https://example.com/foo.pdf")


# ---------------------------------------------------------------------------
# AllowedRootsRegistry
# ---------------------------------------------------------------------------


def test_registry_unconstrained_by_default() -> None:
    r = AllowedRootsRegistry()
    assert r.is_unconstrained
    assert r.active_roots() == set()
    # Unconstrained validate is a no-op for any path
    r.validate_source("/etc/hostname")
    r.validate_source("https://example.com/x.pdf")


def test_registry_set_static_roots() -> None:
    r = AllowedRootsRegistry()
    with tempfile.TemporaryDirectory() as tmp:
        r.set_static_roots([tmp])
        assert not r.is_unconstrained
        assert Path(tmp).resolve() in r.active_roots()


def test_registry_validates_inside_static_root() -> None:
    r = AllowedRootsRegistry()
    with tempfile.TemporaryDirectory() as tmp:
        inside = os.path.join(tmp, "doc.pdf")
        Path(inside).touch()
        r.set_static_roots([tmp])
        r.validate_source(inside)  # no exception
        r.validate_source(f"file://{inside}")  # no exception


def test_registry_rejects_outside_static_root() -> None:
    r = AllowedRootsRegistry()
    with tempfile.TemporaryDirectory() as tmp:
        r.set_static_roots([tmp])
        with pytest.raises(PermissionError, match="not under any allowed root"):
            r.validate_source("/etc/hostname")


def test_registry_remote_url_passthrough_when_constrained() -> None:
    r = AllowedRootsRegistry()
    with tempfile.TemporaryDirectory() as tmp:
        r.set_static_roots([tmp])
        # Even though we're constrained, http URLs aren't filesystem paths
        r.validate_source("https://example.com/foo.pdf")
        r.validate_source("s3://bucket/key.pdf")


def test_registry_client_roots_replace_static() -> None:
    """Per canonical MCP semantics: client roots fully REPLACE static set."""
    r = AllowedRootsRegistry()
    with tempfile.TemporaryDirectory() as static_dir:
        with tempfile.TemporaryDirectory() as client_dir:
            r.set_static_roots([static_dir])
            r.update_from_client_roots([f"file://{client_dir}"])

            # Static path is no longer authorized — client roots replaced it
            with pytest.raises(PermissionError):
                r.validate_source(os.path.join(static_dir, "x.pdf"))

            # Client path IS authorized
            client_file = os.path.join(client_dir, "y.pdf")
            Path(client_file).touch()
            r.validate_source(client_file)


def test_registry_clear_client_roots_restores_static() -> None:
    r = AllowedRootsRegistry()
    with tempfile.TemporaryDirectory() as static_dir:
        with tempfile.TemporaryDirectory() as client_dir:
            r.set_static_roots([static_dir])
            r.update_from_client_roots([f"file://{client_dir}"])
            r.clear_client_roots()

            # Static path is authorized again
            sf = os.path.join(static_dir, "x.pdf")
            Path(sf).touch()
            r.validate_source(sf)

            # Old client path is NOT
            with pytest.raises(PermissionError):
                r.validate_source(os.path.join(client_dir, "y.pdf"))


def test_registry_update_ignores_non_file_uris() -> None:
    r = AllowedRootsRegistry()
    r.update_from_client_roots(
        ["https://example.com/", "file:///tmp/ok"]
    )
    # The https root is dropped; only the file:// root is kept
    assert r.active_roots() == {Path("/tmp/ok").resolve()}


def test_registry_symlink_resolution() -> None:
    """Resolve symlinks before comparing — prevents bypass via symlink."""
    r = AllowedRootsRegistry()
    with tempfile.TemporaryDirectory() as tmp:
        real = Path(tmp) / "real"
        real.mkdir()
        link = Path(tmp) / "link"
        link.symlink_to(real)
        target_file = real / "doc.pdf"
        target_file.touch()

        r.set_static_roots([str(real)])
        # Path through the symlink should still validate (resolved is the same)
        r.validate_source(str(link / "doc.pdf"))


def test_registry_dotdot_resolution() -> None:
    """`..` segments are resolved before validation."""
    r = AllowedRootsRegistry()
    with tempfile.TemporaryDirectory() as parent:
        sub = os.path.join(parent, "sub")
        os.makedirs(sub)
        # Path that escapes the root via ..
        escape = os.path.join(sub, "..", "..", "etc", "hostname")
        r.set_static_roots([sub])
        with pytest.raises(PermissionError):
            r.validate_source(escape)


# ---------------------------------------------------------------------------
# _roots_wiring (notification handlers + ContextVar)
# ---------------------------------------------------------------------------


def test_install_roots_handlers_registers_both() -> None:
    from docling_mcp._roots_wiring import install_roots_handlers
    from docling_mcp.shared import mcp
    from mcp.types import InitializedNotification, RootsListChangedNotification

    install_roots_handlers()
    handlers = mcp._mcp_server.notification_handlers
    assert InitializedNotification in handlers
    assert RootsListChangedNotification in handlers


def test_install_roots_handlers_is_idempotent() -> None:
    from docling_mcp._roots_wiring import install_roots_handlers
    from docling_mcp.shared import mcp

    install_roots_handlers()
    first_patched_method = mcp._mcp_server._handle_message
    install_roots_handlers()
    second_patched_method = mcp._mcp_server._handle_message
    # Same patched method object — second call did not re-wrap
    assert first_patched_method is second_patched_method
    assert getattr(first_patched_method, "_docling_mcp_roots_patched", False)


@pytest.mark.asyncio
async def test_on_initialized_seeds_from_session() -> None:
    """Handler queries session.list_roots() and updates the registry."""
    from docling_mcp import _roots_wiring as wiring
    from docling_mcp.shared import allowed_roots
    from mcp.types import (
        InitializedNotification,
        ListRootsResult,
        Root,
        RootsCapability,
        ClientCapabilities,
    )

    # Ensure clean state for this test
    allowed_roots.clear_client_roots()

    fake_session = MagicMock()
    fake_session.client_params = MagicMock()
    fake_session.client_params.capabilities = ClientCapabilities(
        roots=RootsCapability(listChanged=True)
    )
    fake_session.list_roots = AsyncMock(
        return_value=ListRootsResult(
            roots=[Root(uri="file:///tmp/from-client")]
        )
    )

    token = wiring._active_session.set(fake_session)
    try:
        await wiring._on_initialized(
            InitializedNotification(method="notifications/initialized")
        )
    finally:
        wiring._active_session.reset(token)

    fake_session.list_roots.assert_awaited_once()
    assert Path("/tmp/from-client").resolve() in allowed_roots.active_roots()


@pytest.mark.asyncio
async def test_on_initialized_skips_when_client_lacks_roots() -> None:
    """No client roots capability → don't call list_roots()."""
    from docling_mcp import _roots_wiring as wiring
    from mcp.types import ClientCapabilities, InitializedNotification

    fake_session = MagicMock()
    fake_session.client_params = MagicMock()
    # roots is not advertised
    fake_session.client_params.capabilities = ClientCapabilities()
    fake_session.list_roots = AsyncMock()

    token = wiring._active_session.set(fake_session)
    try:
        await wiring._on_initialized(
            InitializedNotification(method="notifications/initialized")
        )
    finally:
        wiring._active_session.reset(token)

    fake_session.list_roots.assert_not_awaited()


@pytest.mark.asyncio
async def test_on_list_changed_refreshes_registry() -> None:
    """Handler refreshes the registry on a list_changed notification."""
    from docling_mcp import _roots_wiring as wiring
    from docling_mcp.shared import allowed_roots
    from mcp.types import (
        ClientCapabilities,
        ListRootsResult,
        Root,
        RootsCapability,
        RootsListChangedNotification,
    )

    allowed_roots.clear_client_roots()
    fake_session = MagicMock()
    fake_session.client_params = MagicMock()
    fake_session.client_params.capabilities = ClientCapabilities(
        roots=RootsCapability(listChanged=True)
    )
    fake_session.list_roots = AsyncMock(
        return_value=ListRootsResult(
            roots=[
                Root(uri="file:///tmp/a"),
                Root(uri="file:///tmp/b"),
            ]
        )
    )

    token = wiring._active_session.set(fake_session)
    try:
        await wiring._on_roots_list_changed(
            RootsListChangedNotification(
                method="notifications/roots/list_changed"
            )
        )
    finally:
        wiring._active_session.reset(token)

    active = allowed_roots.active_roots()
    assert Path("/tmp/a").resolve() in active
    assert Path("/tmp/b").resolve() in active
