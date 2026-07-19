"""Fetch-to-temp resolution of object-storage source URIs."""

import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from docling_mcp.logger import setup_logger

logger = setup_logger()

# Object-storage schemes resolved through fsspec, mapped to the provider
# package implementing them and the docling-mcp extra that installs it.
# IBM Cloud Object Storage is reached through the S3 protocol.
_FSSPEC_SCHEMES: dict[str, tuple[str, str]] = {
    "s3": ("s3fs", "s3"),
    "gs": ("gcsfs", "gcs"),
    "gcs": ("gcsfs", "gcs"),
    "abfs": ("adlfs", "azure"),
    "az": ("adlfs", "azure"),
}

_COPY_BUFFER_SIZE = 1 << 20


def _install_hint(scheme: str, package: str, extra: str) -> str:
    return (
        f"Reading {scheme}:// sources requires the '{package}' package. "
        f"Install it with the docling-mcp[{extra}] extra."
    )


@contextmanager
def fetched_source(source: str) -> Iterator[str]:
    """Yield a local path for a source, fetching object-storage URIs.

    Sources with an fsspec-style object-storage scheme (s3://, gs://,
    abfs://) are downloaded to a temporary file that is removed when the
    context exits; the file keeps the URI's suffix so input-format detection
    and content-hash cache keys behave exactly as for a local file. Any
    other source is yielded unchanged.

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
    scheme = urlsplit(source).scheme.lower()
    if scheme not in _FSSPEC_SCHEMES:
        yield source
        return

    package, extra = _FSSPEC_SCHEMES[scheme]
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
