"""Helpers for distinguishing URL-style and filesystem-style ``source`` strings.

The ``convert_document_into_docling_document`` tool accepts both remote URLs
and local filesystem paths in a single ``source`` parameter. Roots-based
authorization only applies to filesystem paths, so we need a small predicate
to decide which validator to apply.

This module is deliberately tiny and dependency-free so it can be imported
from both the registry and the conversion tools without introducing cycles.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

# URL schemes we treat as remote/non-filesystem and therefore exempt from
# roots authorization. ``file://`` is intentionally excluded so it falls
# through to filesystem-path handling.
_REMOTE_SCHEMES: frozenset[str] = frozenset({"http", "https", "ftp", "ftps", "s3"})


def is_remote_url(source: str) -> bool:
    """Return ``True`` if ``source`` is a remote URL (not a filesystem path).

    ``file://`` URLs are considered filesystem paths, not remote URLs, so
    they are subject to roots authorization.
    """
    parsed = urlparse(source)
    return parsed.scheme.lower() in _REMOTE_SCHEMES


def to_filesystem_path(source: str) -> Path:
    """Resolve ``source`` to an absolute filesystem ``Path``.

    Accepts either a plain path string or a ``file://`` URL. Raises
    ``ValueError`` for any other URL scheme — callers should gate this with
    :func:`is_remote_url` first.
    """
    parsed = urlparse(source)
    scheme = parsed.scheme.lower()

    if scheme in _REMOTE_SCHEMES:
        raise ValueError(
            f"to_filesystem_path() called on a remote URL: {source!r}. "
            "Use is_remote_url() to gate this call."
        )

    if scheme == "file":
        # urlparse on "file:///a/b" gives path="/a/b"
        candidate = Path(parsed.path)
    else:
        candidate = Path(source)

    return candidate.expanduser().resolve()
