"""Test resolution of object-storage source URIs."""

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

from docling_core.types.doc.document import DoclingDocument
from docling_core.types.io import DocumentStream

from docling_mcp.tools.converters.sources import (
    fetched_source,
    fetched_stream,
    supported_uri_schemes,
)

fsspec = pytest.importorskip("fsspec")

OBJECT_URI = "memory://bucket/spec.pdf"


def _register_memory_scheme(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route the fsspec in-memory filesystem through the shim for tests."""
    import docling_mcp.tools.converters.sources as sources

    monkeypatch.setitem(sources._FSSPEC_SCHEMES, "memory", ("fsspec", "s3"))


def test_local_paths_pass_through(tmp_path: Path) -> None:
    source = str(tmp_path / "doc.pdf")
    with fetched_source(source) as resolved:
        assert resolved == source


def test_http_urls_pass_through() -> None:
    url = "https://example.com/spec.pdf"
    with fetched_source(url) as resolved:
        assert resolved == url


def test_object_uri_fetched_to_temp_file(monkeypatch: pytest.MonkeyPatch) -> None:
    _register_memory_scheme(monkeypatch)
    fsspec.filesystem("memory").pipe("/bucket/spec.pdf", b"fake pdf bytes")

    with fetched_source(OBJECT_URI) as resolved:
        path = Path(resolved)
        assert resolved != OBJECT_URI
        # The suffix selects the input format, so it must survive the fetch.
        assert path.suffix == ".pdf"
        assert path.read_bytes() == b"fake pdf bytes"

    assert not path.exists()


def test_fetch_failure_removes_temp_file(monkeypatch: pytest.MonkeyPatch) -> None:
    _register_memory_scheme(monkeypatch)

    with pytest.raises(FileNotFoundError):
        with fetched_source("memory://bucket/absent.pdf"):
            pass


def test_missing_provider_package_raises_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Make s3fs unimportable so fsspec raises the same ImportError it would
    # raise without the extra installed, whether or not s3fs is present here.
    monkeypatch.setitem(sys.modules, "s3fs", None)

    with pytest.raises(ValueError, match=r"docling-mcp\[s3\]"):
        with fetched_source("s3://bucket/key.pdf"):
            pass


def test_missing_provider_package_raises_install_hint_for_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "s3fs", None)

    with pytest.raises(ValueError, match=r"docling-mcp\[s3\]"):
        fetched_stream("s3://bucket/key.pdf")


def test_missing_fsspec_raises_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    """fsspec itself absent is reported as the provider extra being missing."""
    monkeypatch.setitem(sys.modules, "fsspec", None)

    with pytest.raises(ValueError, match=r"docling-mcp\[s3\]"):
        with fetched_source("s3://bucket/key.pdf"):
            pass

    with pytest.raises(ValueError, match=r"docling-mcp\[s3\]"):
        fetched_stream("s3://bucket/key.pdf")


def test_supported_uri_schemes_is_sorted_and_complete() -> None:
    import docling_mcp.tools.converters.sources as sources

    assert supported_uri_schemes() == tuple(sorted(sources._FSSPEC_SCHEMES))


def test_local_and_http_sources_pass_through_stream(tmp_path: Path) -> None:
    local = str(tmp_path / "doc.pdf")
    assert fetched_stream(local) == local
    assert fetched_stream("https://example.com/spec.pdf") == (
        "https://example.com/spec.pdf"
    )


def test_object_uri_fetched_into_document_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _register_memory_scheme(monkeypatch)
    fsspec.filesystem("memory").pipe("/bucket/spec.pdf", b"fake pdf bytes")

    resolved = fetched_stream(OBJECT_URI)

    assert isinstance(resolved, DocumentStream)
    # The name carries the suffix, which is what selects the input format.
    assert resolved.name == "spec.pdf"
    assert resolved.stream.getvalue() == b"fake pdf bytes"


@patch("docling_mcp.tools.converters.remote.DoclingServiceClient")
@patch("docling_mcp.tools.converters.remote.settings")
def test_remote_converter_fetches_object_uri(
    mock_settings: Any, mock_client_class: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The remote converter converts a fetched local copy of an object URI."""
    import docling_mcp.tools.converters.remote as remote_mod
    from docling_mcp.tools.converters.remote import RemoteDocumentConverter

    mock_settings.service_url = "https://serve.example.com"
    mock_settings.service_api_key = None
    cache: dict[str, DoclingDocument] = {}
    monkeypatch.setattr(remote_mod, "local_document_cache", cache)
    monkeypatch.setattr(remote_mod, "local_stack_cache", defaultdict(list))

    _register_memory_scheme(monkeypatch)
    fsspec.filesystem("memory").pipe("/bucket/spec.pdf", b"object bytes")

    sent: dict[str, Any] = {}

    def fake_convert(source: Any, options: Any) -> Any:
        sent["source"] = source
        result = Mock()
        result.status.is_error = False
        result.document = DoclingDocument(name="spec")
        return result

    mock_client_class.return_value.convert.side_effect = fake_convert

    converter = RemoteDocumentConverter()
    output = converter.convert_document(OBJECT_URI)

    # The body is uploaded from memory; no temp file is written for it.
    assert output.from_cache is False
    assert isinstance(sent["source"], DocumentStream)
    assert sent["source"].name == "spec.pdf"
    assert sent["source"].stream.getvalue() == b"object bytes"

    # The converted document records the original URI as its source.
    ((key, doc),) = cache.items()
    assert key == output.document_key
    assert any(t.text == f"source: {OBJECT_URI}" for t in doc.texts)


@patch("docling_mcp.tools.converters.remote.DoclingServiceClient")
@patch("docling_mcp.tools.converters.remote.settings")
def test_remote_cache_hit_does_not_fetch(
    mock_settings: Any, mock_client_class: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cache hit must not touch the object store at all."""
    import docling_mcp.tools.converters.remote as remote_mod
    from docling_mcp.tools.converters.remote import RemoteDocumentConverter

    mock_settings.service_url = "https://serve.example.com"
    mock_settings.service_api_key = None
    monkeypatch.setattr(remote_mod, "local_document_cache", {})
    monkeypatch.setattr(remote_mod, "local_stack_cache", defaultdict(list))

    _register_memory_scheme(monkeypatch)
    fsspec.filesystem("memory").pipe("/bucket/spec.pdf", b"object bytes")

    mock_client_class.return_value.convert.side_effect = lambda source, options: Mock(
        status=Mock(is_error=False), document=DoclingDocument(name="spec")
    )

    converter = RemoteDocumentConverter()
    first = converter.convert_document(OBJECT_URI)

    fetches: list[str] = []
    real_open = fsspec.open

    def counting_open(urlpath: str, *args: Any, **kwargs: Any) -> Any:
        fetches.append(urlpath)
        return real_open(urlpath, *args, **kwargs)

    monkeypatch.setattr(fsspec, "open", counting_open)

    again = converter.convert_document(OBJECT_URI)

    assert again.from_cache is True
    assert again.document_key == first.document_key
    assert fetches == []


@patch("docling_mcp.tools.converters.local.DocumentConverter")
def test_local_cache_hit_does_not_fetch(
    mock_converter_class: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The local converter also resolves the cache before fetching."""
    import docling_mcp.tools.converters.local as local_mod
    from docling_mcp.tools.converters.local import LocalDocumentConverter

    monkeypatch.setattr(local_mod, "local_document_cache", {})
    monkeypatch.setattr(local_mod, "local_stack_cache", defaultdict(list))

    _register_memory_scheme(monkeypatch)
    fsspec.filesystem("memory").pipe("/bucket/spec.pdf", b"object bytes")

    mock_converter_class.return_value.convert.return_value = Mock(
        status=Mock(is_error=False), document=DoclingDocument(name="spec")
    )

    converter = LocalDocumentConverter()
    first = converter.convert_document(OBJECT_URI)

    fetches: list[str] = []
    real_open = fsspec.open

    def counting_open(urlpath: str, *args: Any, **kwargs: Any) -> Any:
        fetches.append(urlpath)
        return real_open(urlpath, *args, **kwargs)

    monkeypatch.setattr(fsspec, "open", counting_open)

    again = converter.convert_document(OBJECT_URI)

    assert again.from_cache is True
    assert again.document_key == first.document_key
    assert fetches == []
