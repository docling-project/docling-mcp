"""Test fetch-to-temp resolution of object-storage source URIs."""

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

from docling_core.types.doc.document import DoclingDocument

from docling_mcp.tools.converters.sources import fetched_source

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
    monkeypatch.delitem(sys.modules["fsspec.registry"]._registry, "s3", raising=False)

    with pytest.raises(ValueError, match=r"docling-mcp\[s3\]"):
        with fetched_source("s3://bucket/key.pdf"):
            pass


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

    fetched: dict[str, Any] = {}

    def fake_convert(source: str, options: Any) -> Any:
        fetched["source"] = source
        fetched["existed"] = Path(source).exists()
        result = Mock()
        result.status.is_error = False
        result.document = DoclingDocument(name="spec")
        return result

    mock_client_class.return_value.convert.side_effect = fake_convert

    converter = RemoteDocumentConverter()
    output = converter.convert_document(OBJECT_URI)

    # The service client received a local temp copy, not the object URI.
    assert output.from_cache is False
    assert fetched["existed"] is True
    assert fetched["source"] != OBJECT_URI
    assert fetched["source"].endswith(".pdf")

    # The converted document records the original URI as its source.
    ((key, doc),) = cache.items()
    assert key == output.document_key
    assert any(t.text == f"source: {OBJECT_URI}" for t in doc.texts)

    # The key comes from the URI, not the temp copy, so a repeat call hits
    # the cache instead of re-fetching under a new key.
    again = converter.convert_document(OBJECT_URI)
    assert again.from_cache is True
    assert again.document_key == output.document_key
