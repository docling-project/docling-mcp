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
    assert p == Path("/tmp/foo.pdf").resolve()


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
    r.update_from_client_roots(["https://example.com/", "file:///tmp/ok"])
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
    from mcp.types import InitializedNotification, RootsListChangedNotification

    from docling_mcp._roots_wiring import install_roots_handlers
    from docling_mcp.shared import mcp

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
    from mcp.types import (
        ClientCapabilities,
        InitializedNotification,
        ListRootsResult,
        Root,
        RootsCapability,
    )

    from docling_mcp import _roots_wiring as wiring
    from docling_mcp.shared import allowed_roots

    # Ensure clean state for this test
    allowed_roots.clear_client_roots()

    fake_session = MagicMock()
    fake_session.client_params = MagicMock()
    fake_session.client_params.capabilities = ClientCapabilities(
        roots=RootsCapability(listChanged=True)
    )
    fake_session.list_roots = AsyncMock(
        return_value=ListRootsResult(roots=[Root(uri="file:///tmp/from-client")])
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
    from mcp.types import ClientCapabilities, InitializedNotification

    from docling_mcp import _roots_wiring as wiring

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
    from mcp.types import (
        ClientCapabilities,
        ListRootsResult,
        Root,
        RootsCapability,
        RootsListChangedNotification,
    )

    from docling_mcp import _roots_wiring as wiring
    from docling_mcp.shared import allowed_roots

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

    expected_a = Path("/tmp/a").resolve()
    expected_b = Path("/tmp/b").resolve()
    token = wiring._active_session.set(fake_session)
    try:
        await wiring._on_roots_list_changed(
            RootsListChangedNotification(method="notifications/roots/list_changed")
        )
    finally:
        wiring._active_session.reset(token)

    active = allowed_roots.active_roots()
    assert expected_a in active
    assert expected_b in active


# ---------------------------------------------------------------------------
# Regression tests — URL-encoded paths
# ---------------------------------------------------------------------------


def test_to_filesystem_path_decodes_percent_escapes() -> None:
    """file:// URLs with %20 (space) decode to the real filesystem path."""
    p = to_filesystem_path("file:///Users/me/Application%20Support/foo.pdf")
    assert p == Path("/Users/me/Application Support/foo.pdf").resolve()
    assert "%20" not in str(p)


def test_to_filesystem_path_handles_special_chars() -> None:
    """Percent-encoded special characters (colons, parens) also decode."""
    p = to_filesystem_path("file:///tmp/yerk%3A%20Career%20Debugger/x.pdf")
    assert "%3A" not in str(p)
    assert "%20" not in str(p)
    assert "yerk: Career Debugger" in str(p)


def test_registry_client_root_with_percent_encoded_space() -> None:
    """Roots like file:///Application%20Support/... validate real paths."""
    r = AllowedRootsRegistry()
    with tempfile.TemporaryDirectory() as parent:
        # Build a path with a real space, the way macOS does
        real_dir = os.path.join(parent, "App Support")
        os.makedirs(real_dir)
        target = os.path.join(real_dir, "doc.pdf")
        Path(target).touch()

        # Client sends the percent-encoded form (the way urlparse emits it)
        r.update_from_client_roots([f"file://{parent}/App%20Support"])

        # The real-disk path with a literal space should validate
        r.validate_source(target)


def test_registry_logs_decoded_paths_not_encoded() -> None:
    """active_roots() exposes decoded paths so error messages are readable."""
    r = AllowedRootsRegistry()
    r.update_from_client_roots(
        ["file:///Users/me/Application%20Support/Claude/uploads"]
    )
    roots = r.active_roots()
    assert all("%20" not in str(p) for p in roots)
    assert all("Application Support" in str(p) for p in roots)


# ---------------------------------------------------------------------------
# --preload-models — intentionally absent (scope tripwire)
# ---------------------------------------------------------------------------
#
# These two tests exist to PREVENT accidental re-introduction of a
# `--preload-models` CLI flag that was scoped OUT of this PR during the
# 2026-06-11 merge with upstream/main (merge commit 6408167).
#
# Why the flag is not here:
#   1. Out of scope for the MCP Roots PR. The flag was added during local
#      live-test debugging to dodge Claude Desktop's per-request timeout —
#      it has nothing to do with path authorization.
#   2. Upstream PR #97 (feat!: Use remote Docling within tool) made
#      conversion remote-by-default with an opt-in `[local]` extra. Local
#      models are no longer imported at server boot in the default config,
#      which removes the cold-start problem the flag worked around.
#
# If you have a real reason to re-add this flag (e.g., for the local-extra
# path where boot-time warmup still matters), do it as a separate PR and
# delete or invert these two tests as part of that change.


def test_preload_flag_intentionally_absent_from_cli() -> None:
    """`--preload-models` MUST NOT appear in the CLI — see comment above."""
    from typer.testing import CliRunner

    from docling_mcp.servers import mcp_server

    runner = CliRunner()
    result = runner.invoke(mcp_server.app, ["--help"])
    assert result.exit_code == 0
    assert "--preload-models" not in result.output, (
        "--preload-models was re-introduced to the CLI. "
        "If intentional, see the comment above this test and delete it."
    )


def test_preload_param_intentionally_absent_from_main() -> None:
    """`preload_models` MUST NOT be a parameter of main() — see comment above."""
    import inspect

    from docling_mcp.servers.mcp_server import main

    sig = inspect.signature(main)
    assert "preload_models" not in sig.parameters, (
        "preload_models was re-added to main()'s signature. "
        "If intentional, see the comment above this test and delete it."
    )
