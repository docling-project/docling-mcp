"""Tests for per-call pipeline-option overrides on convert_document.

The MCP `convert_document_into_docling_document` tool exposes optional
per-call overrides for `do_ocr`, `do_table_structure`, `keep_images`, and
`conversion_mode`. These tests verify the override plumbing without
spinning up a real DocumentConverter.
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Factory: conversion_mode override
# ---------------------------------------------------------------------------


def test_get_converter_accepts_mode_override() -> None:
    """get_converter accepts a conversion_mode argument."""
    from docling_mcp.tools.converters.factory import get_converter

    sig = inspect.signature(get_converter)
    assert "conversion_mode" in sig.parameters
    assert sig.parameters["conversion_mode"].default is None


def test_get_converter_override_routes_to_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Passing conversion_mode='local' routes to LocalDocumentConverter regardless of env.

    Note: factory uses lazy imports inside get_converter(), so we patch the
    symbols at their source modules rather than on the factory module. That
    means this test requires the [local] extra (docling) to be importable.
    """
    pytest.importorskip("docling")
    from docling_mcp.settings.service_client import ConversionMode, settings
    from docling_mcp.tools.converters import factory

    # Pretend env said REMOTE so the override is what changes routing
    monkeypatch.setattr(settings, "conversion_mode", ConversionMode.REMOTE)

    # Patch the source symbols (factory imports them inside the function)
    with patch(
        "docling_mcp.tools.converters.remote.RemoteDocumentConverter"
    ) as remote_cls:
        with patch(
            "docling_mcp.tools.converters.local.LOCAL_CONVERSION_AVAILABLE", True
        ):
            with patch(
                "docling_mcp.tools.converters.local.LocalDocumentConverter"
            ) as local_cls:
                factory.get_converter(conversion_mode="local")
                local_cls.assert_called_once()
                remote_cls.assert_not_called()


# ---------------------------------------------------------------------------
# LocalDocumentConverter: per-call pipeline overrides
# ---------------------------------------------------------------------------


def test_local_get_converter_accepts_overrides() -> None:
    """LocalDocumentConverter._get_converter exposes the three pipeline overrides."""
    pytest.importorskip("docling")  # only runs when [local] extra is installed
    from docling_mcp.tools.converters.local import LocalDocumentConverter

    sig = inspect.signature(LocalDocumentConverter._get_converter)
    for name in ("do_ocr", "do_table_structure", "keep_images"):
        assert name in sig.parameters, f"missing param: {name}"
        assert sig.parameters[name].default is None


def test_local_convert_document_accepts_overrides() -> None:
    """LocalDocumentConverter.convert_document forwards overrides to _get_converter."""
    pytest.importorskip("docling")
    from docling_mcp.tools.converters.local import LocalDocumentConverter

    sig = inspect.signature(LocalDocumentConverter.convert_document)
    for name in ("do_ocr", "do_table_structure", "keep_images"):
        assert name in sig.parameters, f"missing param: {name}"
        assert sig.parameters[name].default is None


def test_local_converter_cache_keyed_by_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Different option tuples yield different cached DocumentConverter instances."""
    pytest.importorskip("docling")
    from docling_mcp.tools.converters import local as local_module
    from docling_mcp.tools.converters.local import LocalDocumentConverter

    # Replace the underlying DocumentConverter with a counter mock
    construct_calls: list[dict[str, object]] = []

    def fake_dc(format_options: object = None, **kw: object) -> MagicMock:
        construct_calls.append({"format_options": format_options})
        return MagicMock(name="FakeDocumentConverter")

    monkeypatch.setattr(local_module, "DocumentConverter", fake_dc)

    conv = LocalDocumentConverter()

    # Two calls with same options → one underlying instance
    a1 = conv._get_converter(do_ocr=False, do_table_structure=False, keep_images=False)
    a2 = conv._get_converter(do_ocr=False, do_table_structure=False, keep_images=False)
    assert a1 is a2

    # Different option tuple → new instance
    b = conv._get_converter(do_ocr=True, do_table_structure=False, keep_images=False)
    assert b is not a1

    assert len(construct_calls) == 2  # one per unique tuple


# ---------------------------------------------------------------------------
# Remote converter: override plumbing (signature only — no live API)
# ---------------------------------------------------------------------------


def test_remote_convert_document_accepts_overrides() -> None:
    """RemoteDocumentConverter.convert_document accepts the three pipeline overrides."""
    pytest.importorskip("docling")
    from docling_mcp.tools.converters.remote import RemoteDocumentConverter

    sig = inspect.signature(RemoteDocumentConverter.convert_document)
    for name in ("do_ocr", "do_table_structure", "keep_images"):
        assert name in sig.parameters, f"missing param: {name}"
        assert sig.parameters[name].default is None


# ---------------------------------------------------------------------------
# MCP tool surface
# ---------------------------------------------------------------------------


def test_convert_tool_exposes_all_four_overrides() -> None:
    """The convert_document_into_docling_document MCP tool surface includes the overrides."""
    pytest.importorskip("docling")
    from docling_mcp.tools.conversion import convert_document_into_docling_document

    sig = inspect.signature(convert_document_into_docling_document)
    for name in ("do_ocr", "do_table_structure", "keep_images", "conversion_mode"):
        assert name in sig.parameters, f"missing tool param: {name}"
        assert sig.parameters[name].default is None
