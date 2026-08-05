"""Resolution of object-storage source URIs."""

import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Final
from urllib.parse import urlsplit

from docling_core.types.io import DocumentStream

from docling_mcp.logger import setup_logger

logger = setup_logger()

_FSSPEC_SCHEMES: Final[dict[str, tuple[str, str]]] = {
    "s3": ("s3fs", "s3"),
    "gs": ("gcsfs", "gcs"),
    "gcs": ("gcsfs", "gcs"),
    "abfs": ("adlfs", "azure"),
    "az": ("adlfs", "azure"),
}
"""Object-storage schemes resolved through fsspec.

Maps each scheme to the provider package implementing it and the docling-mcp
extra that installs that package.
"""

_COPY_BUFFER_SIZE = 1 << 20


def supported_uri_schemes() -> tuple[str, ...]:
    """Return the object-storage URI schemes this module can resolve, sorted."""
    return tuple(sorted(_FSSPEC_SCHEMES))


def _install_hint(scheme: str, package: str, extra: str) -> str:
    return (
        f"Reading {scheme}:// sources requires the '{package}' package. "
        f"Install it with the docling-mcp[{extra}] extra."
    )


def _fsspec_for(source: str) -> tuple[str, str, str] | None:
    """Return (scheme, package, extra) when a source needs fetching."""
    scheme = urlsplit(source).scheme.lower()
    if scheme not in _FSSPEC_SCHEMES:
        return None
    package, extra = _FSSPEC_SCHEMES[scheme]
    return scheme, package, extra


@contextmanager
def fetched_source(source: str) -> Iterator[str]:
    """Yield a local path for a source, fetching object-storage URIs.

    Sources with an fsspec-style object-storage scheme are downloaded to a
    temporary file that is removed when the context exits; the file keeps the
    URI's suffix so input-format detection behaves as it does for a local
    file. Any other source is yielded unchanged.

    Use this for the local converter, which reads the document from disk.
    The remote converter uploads the body instead, so it uses
    fetched_stream() and never writes a temporary file.

    Credentials are resolved by the provider's standard chain (environment
    variables, configuration files, instance roles). They are deliberately
    never read from MCP client configuration, which several clients store in
    plaintext.

    Args:
        source: A local path, URL, or object-storage URI.

    Yields:
        A local filesystem path for object-storage URIs, otherwise the
        source unchanged.

    Raises:
        ValueError: When the provider package for the URI scheme is not
            installed.
    """
    resolved = _fsspec_for(source)
    if resolved is None:
        yield source
        return

    scheme, package, extra = resolved
    try:
        import fsspec
    except ImportError as exc:
        raise ValueError(_install_hint(scheme, package, extra)) from exc

    suffix = PurePosixPath(urlsplit(source).path).suffix
    handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp_path = Path(handle.name)
    try:
        try:
            with fsspec.open(source, "rb") as remote_file:
                shutil.copyfileobj(remote_file, handle, _COPY_BUFFER_SIZE)
        except ImportError as exc:
            raise ValueError(_install_hint(scheme, package, extra)) from exc
        finally:
            handle.close()
        logger.info(f"Fetched object-storage source to {tmp_path}: {source}")
        yield str(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def fetched_stream(source: str) -> str | DocumentStream:
    """Return an in-memory stream for object-storage URIs.

    The remote converter uploads the document body to docling-serve, so the
    bytes never need to reach the local filesystem. Object-storage URIs are
    read straight into a DocumentStream, whose name carries the URI's
    filename so the service detects the input format as it would for an
    upload. Any other source is returned unchanged, including http(s) URLs,
    which docling-serve fetches itself.

    The whole body is held in memory, unlike fetched_source(), which streams
    it through a temporary file.

    Args:
        source: A local path, URL, or object-storage URI.

    Returns:
        A DocumentStream for object-storage URIs, otherwise the source
        unchanged.

    Raises:
        ValueError: When the provider package for the URI scheme is not
            installed.
    """
    resolved = _fsspec_for(source)
    if resolved is None:
        return source

    scheme, package, extra = resolved
    try:
        import fsspec
    except ImportError as exc:
        raise ValueError(_install_hint(scheme, package, extra)) from exc

    try:
        with fsspec.open(source, "rb") as remote_file:
            body = remote_file.read()
    except ImportError as exc:
        raise ValueError(_install_hint(scheme, package, extra)) from exc

    name = PurePosixPath(urlsplit(source).path).name
    logger.info(f"Fetched object-storage source into memory: {source}")
    return DocumentStream(name=name, stream=BytesIO(body))
