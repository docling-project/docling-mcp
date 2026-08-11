"""Test the conversion cache key."""

import os
from enum import Enum
from pathlib import Path
from typing import Any

import pytest

from docling_mcp.docling_cache import (
    _NOT_OUTPUT_RELEVANT,
    _NOT_RELEVANT_LOCALLY,
    _looks_like_local_path,
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


# Pinned so the source-branch tests below compare only the source, and cannot
# pass because the conversion context happened to differ between two calls.
_FIXED_CONTEXT: dict[str, object] = {}


def test_url_cache_key_changes_with_query_string() -> None:
    url = "https://example.com/spec.pdf"

    assert get_cache_key(url, conversion=_FIXED_CONTEXT) != get_cache_key(
        url + "?v=2", conversion=_FIXED_CONTEXT
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("relative.pdf", True),
        ("./relative.pdf", True),
        ("/abs/doc.pdf", True),
        (r"C:\doc.pdf", True),
        ("C:/doc.pdf", True),
        (r"\\server\share\doc.pdf", True),
        ("//server/share/doc.pdf", True),
        ("https://example.com/spec.pdf", False),
        ("s3://bucket/key.pdf", False),
        ("abfs://container/key.pdf", False),
        ("file:///tmp/doc.pdf", False),
    ],
)
def test_looks_like_local_path(source: str, expected: bool) -> None:
    """Windows drive letters parse as schemes, so they need separate handling."""
    assert _looks_like_local_path(source) is expected


def test_url_is_never_keyed_by_a_local_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file at the path a URL collapses to must not stand in for the URL.

    Path("https://example.com/spec.pdf") is the relative path
    "https:/example.com/spec.pdf", so without a scheme check a file planted
    there would be hashed as the URL's content.
    """
    url = "https://example.com/spec.pdf"
    before = get_cache_key(url, conversion=_FIXED_CONTEXT)

    monkeypatch.chdir(tmp_path)
    planted = Path(url)
    planted.parent.mkdir(parents=True, exist_ok=True)
    planted.write_bytes(b"not the real document")
    assert planted.is_file()

    assert get_cache_key(url, conversion=_FIXED_CONTEXT) == before


def test_url_and_file_content_keys_differ_for_the_same_text(tmp_path: Path) -> None:
    """A URL must not collide with a file whose content is that same URL."""
    url = "https://example.com/spec.pdf"
    source = tmp_path / "spec.pdf"
    source.write_text(url, encoding="utf-8")

    assert get_cache_key(url, conversion=_FIXED_CONTEXT) != get_cache_key(
        str(source), conversion=_FIXED_CONTEXT
    )


def test_file_and_missing_path_keys_differ(tmp_path: Path) -> None:
    """The same string keys differently once it stops naming a file."""
    source = tmp_path / "doc.pdf"
    source.write_bytes(b"some bytes")
    content_key = get_cache_key(str(source), conversion=_FIXED_CONTEXT)

    source.unlink()

    assert get_cache_key(str(source), conversion=_FIXED_CONTEXT) != content_key


def test_cache_key_for_directories_uses_source_string(tmp_path: Path) -> None:
    """A directory is not a file, so it must not be hashed as one."""
    source = tmp_path / "directory"
    missing_key = get_cache_key(str(source), conversion=_FIXED_CONTEXT)

    source.mkdir()

    assert get_cache_key(str(source), conversion=_FIXED_CONTEXT) == missing_key


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
    ("context", "also_ignored"),
    [
        (local_conversion_context, _NOT_RELEVANT_LOCALLY),
        (remote_conversion_context, frozenset()),
    ],
)
def test_cache_key_covers_every_output_relevant_setting(
    tmp_path: Path, context: Any, also_ignored: frozenset[str]
) -> None:
    """Every setting that reaches the converter must change the key.

    Asserting over the model fields rather than a hand-written list means a
    setting added later fails this test instead of silently serving a
    conversion produced under a different configuration.
    """
    from docling_mcp.settings.service_client import settings

    source = tmp_path / "doc.pdf"
    source.write_bytes(b"stable bytes")
    baseline = get_cache_key(str(source), conversion=context())

    ignored = _NOT_OUTPUT_RELEVANT | also_ignored
    covered = [n for n in type(settings).model_fields if n not in ignored]
    assert covered, "no output-relevant settings found"

    for name in covered:
        current = getattr(settings, name)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(settings, name, _other_value(current))
            changed = get_cache_key(str(source), conversion=context())
        assert changed != baseline, f"{name} does not affect the cache key"


def test_settings_that_do_not_change_output_are_excluded(tmp_path: Path) -> None:
    """Ignored settings must not invalidate a cached local conversion.

    The expected sets are written out here rather than read from the
    production constants. Otherwise adding an output-relevant setting to the
    denylist would make the coverage test above stop checking it while this
    test congratulates the implementation for excluding it.
    """
    from docling_mcp.settings.service_client import settings

    assert _NOT_OUTPUT_RELEVANT == {
        "conversion_mode",
        "fallback_to_local",
        "service_api_key",
        "service_max_retries",
        "service_timeout",
    }
    assert _NOT_RELEVANT_LOCALLY == {"service_url"}

    source = tmp_path / "doc.pdf"
    source.write_bytes(b"stable bytes")
    baseline = get_cache_key(str(source), conversion=local_conversion_context())

    for name in sorted(_NOT_OUTPUT_RELEVANT | _NOT_RELEVANT_LOCALLY):
        current = getattr(settings, name)
        other = _other_value(current)
        assert other != current, f"{name} was not actually changed"

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(settings, name, other)
            assert (
                get_cache_key(str(source), conversion=local_conversion_context())
                == baseline
            ), f"{name} unexpectedly affects the local cache key"


def _other_value(current: object) -> object:
    """Return a valid value of the same type that differs from the current one."""
    if isinstance(current, Enum):
        alternatives = [member for member in type(current) if member != current]
        assert alternatives, f"{type(current).__name__} has no other member"
        return alternatives[0]
    if isinstance(current, bool):
        return not current
    if isinstance(current, int):
        return current + 1
    if isinstance(current, float):
        return current + 1.0
    if isinstance(current, str):
        return current + "-changed"
    if current is None:
        # Every currently None-valued setting is annotated str | None.
        return "changed"
    raise AssertionError(f"unsupported settings value type: {type(current)!r}")
