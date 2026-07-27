"""Test the conversion cache key."""

import importlib
import os
from pathlib import Path
from typing import Any

import pytest

from docling_mcp.docling_cache import (
    _NOT_OUTPUT_RELEVANT,
    get_cache_key,
    local_conversion_context,
    remote_conversion_context,
)


def test_cache_key_dedupes_identical_content(tmp_path: Path) -> None:
    file_one = tmp_path / "one.pdf"
    file_two = tmp_path / "sub" / "two.pdf"
    file_two.parent.mkdir()
    file_one.write_bytes(b"same bytes")
    file_two.write_bytes(b"same bytes")

    assert get_cache_key(str(file_one)) == get_cache_key(str(file_two))


def test_cache_key_distinguishes_file_formats(tmp_path: Path) -> None:
    file_one = tmp_path / "doc.html"
    file_two = tmp_path / "doc.md"
    file_one.write_bytes(b"same bytes")
    file_two.write_bytes(b"same bytes")

    # The suffix selects the input format, so identical bytes under
    # different extensions must convert separately.
    assert get_cache_key(str(file_one)) != get_cache_key(str(file_two))


def test_cache_key_changes_when_content_changes(tmp_path: Path) -> None:
    source = tmp_path / "doc.pdf"
    source.write_bytes(b"version one")
    key_one = get_cache_key(str(source))

    source.write_bytes(b"version two, longer")

    assert get_cache_key(str(source)) != key_one


def test_cache_key_detects_same_size_rewrite_with_preserved_mtime(
    tmp_path: Path,
) -> None:
    source = tmp_path / "doc.pdf"
    source.write_bytes(b"aaaa")
    key_one = get_cache_key(str(source))
    stat = source.stat()

    source.write_bytes(b"bbbb")
    os.utime(source, (stat.st_atime, stat.st_mtime))

    assert get_cache_key(str(source)) != key_one


def test_cache_key_for_urls_uses_source_string() -> None:
    url = "https://example.com/spec.pdf"

    assert get_cache_key(url) == get_cache_key(url)
    assert get_cache_key(url) != get_cache_key(url + "?v=2")


def test_cache_key_for_directories_uses_source_string(tmp_path: Path) -> None:
    # A directory is not a file, so it must not be hashed as one.
    assert get_cache_key(str(tmp_path)) == get_cache_key(str(tmp_path))


def test_cache_key_uses_converter_supplied_context(tmp_path: Path) -> None:
    source = tmp_path / "doc.pdf"
    source.write_bytes(b"stable bytes")

    local_key = get_cache_key(str(source), conversion=local_conversion_context())
    remote_key = get_cache_key(str(source), conversion=remote_conversion_context())

    # A fallback conversion executed locally must never share a key with a
    # remote conversion of the same source.
    assert local_key != remote_key


def test_cache_key_follows_the_configured_mode(tmp_path: Path) -> None:
    from docling_mcp.settings.service_client import (
        ConversionMode,
        settings as service_settings,
    )

    source = tmp_path / "doc.pdf"
    source.write_bytes(b"stable bytes")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(service_settings, "conversion_mode", ConversionMode.LOCAL)
        local_key = get_cache_key(str(source))

        mp.setattr(service_settings, "conversion_mode", ConversionMode.REMOTE)
        mp.setattr(service_settings, "service_url", "https://serve-a.example.com")
        remote_key_a = get_cache_key(str(source))

        mp.setattr(service_settings, "service_url", "https://serve-b.example.com")
        remote_key_b = get_cache_key(str(source))

    assert local_key != remote_key_a
    assert remote_key_a != remote_key_b


@pytest.mark.parametrize(
    ("module", "context"),
    [
        ("docling_mcp.settings.conversion", local_conversion_context),
        ("docling_mcp.settings.service_client", remote_conversion_context),
    ],
)
def test_cache_key_covers_every_output_relevant_setting(
    tmp_path: Path, module: str, context: Any
) -> None:
    """Every setting that reaches the converter must change the key.

    Asserting over the model fields rather than a hand-written list means a
    setting added later fails this test instead of silently serving a
    conversion produced under a different configuration.
    """
    settings = importlib.import_module(module).settings

    source = tmp_path / "doc.pdf"
    source.write_bytes(b"stable bytes")
    baseline = get_cache_key(str(source), conversion=context())

    covered = [n for n in type(settings).model_fields if n not in _NOT_OUTPUT_RELEVANT]
    assert covered, "no output-relevant settings found"

    for name in covered:
        current = getattr(settings, name)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(settings, name, _other_value(current))
            changed = get_cache_key(str(source), conversion=context())
        assert changed != baseline, f"{module}.{name} does not affect the cache key"


def _other_value(current: object) -> object:
    """Return a value of the same type that differs from the current one."""
    if isinstance(current, bool):
        return not current
    if isinstance(current, int):
        return current + 1
    if isinstance(current, float):
        return current + 1.0
    if isinstance(current, str):
        return current + "-changed"
    return "changed"
