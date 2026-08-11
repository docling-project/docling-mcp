"""This module manages the cache directory to run Docling MCP tools."""

import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path, PureWindowsPath
from typing import Final
from urllib.parse import urlsplit

from docling_mcp.logger import setup_logger

# Create a default project logger
logger = setup_logger()


def hash_string(input_string: str) -> str:
    """Creates a hash-string from the input string."""
    return hashlib.sha256(input_string.encode(), usedforsecurity=False).hexdigest()


def get_cache_dir() -> Path:
    """Get the cache directory for the application.

    Returns:
        Path: A Path object pointing to the cache directory.

    The function will:
    1. First check for an environment variable 'CACHE_DIR'
    2. If not found, create a '_cache' directory in the root of the current package
    3. Ensure the directory exists before returning
    """
    # Check if cache directory is specified in environment variable
    cache_dir = os.environ.get("CACHE_DIR")

    if cache_dir:
        # Use the directory specified in the environment variable
        cache_path = Path(cache_dir)
    else:
        # Determine the package root directory
        if getattr(sys, "frozen", False):
            # Handle PyInstaller case
            package_root = Path(os.path.dirname(sys.executable))
        else:
            # Get the directory of the caller's module
            caller_file = sys._getframe(1).f_globals.get("__file__")

            if caller_file:
                # If running as a script or module
                current_path = Path(caller_file).resolve()

                # Find the package root by looking for the highest directory with an __init__.py
                package_root = current_path.parent
                while package_root.joinpath("__init__.py").exists():
                    package_root = package_root.parent
            else:
                # Fallback to current working directory if __file__ is not available
                package_root = Path.cwd()

        logger.info(f"package-root: {package_root}")

        # Create the cache directory path
        cache_path = package_root / "_cache"

    # Ensure cache directory exists
    logger.info(f"cache-path: {cache_path}")
    os.makedirs(cache_path, exist_ok=True)

    return cache_path


def _file_content_digest(path: Path) -> str:
    """Return the SHA-256 digest of a file's content.

    The content is hashed on every call: the cost is negligible next to a
    conversion, and stat-based caching cannot reliably detect same-size
    rewrites with preserved timestamps.
    """
    hasher = hashlib.sha256(usedforsecurity=False)
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


_VERSION_STAMP: Final[dict[str, str | None]] = {
    "docling_mcp": _package_version("docling-mcp"),
    "docling": _package_version("docling-slim") or _package_version("docling"),
}
"""Package versions that participate in the cache key.

Invalidates cached results when the packages producing them change behavior.
Computed once at import time, since the installed versions cannot change while
the process runs.
"""

_NOT_OUTPUT_RELEVANT: Final[frozenset[str]] = frozenset(
    {
        "conversion_mode",
        "fallback_to_local",
        "service_api_key",
        "service_max_retries",
        "service_timeout",
    }
)
"""Settings that never change what a conversion produces.

Every other setting enters the cache key, so one added later over-invalidates
cached conversions until it is listed here, rather than silently serving a
result produced under a different configuration.
"""

_NOT_RELEVANT_LOCALLY: Final[frozenset[str]] = frozenset({"service_url"})
"""Settings that additionally do not apply when converting in this process."""


def _conversion_options(exclude: frozenset[str] = frozenset()) -> dict[str, object]:
    """Return the settings that change what a conversion produces."""
    from docling_mcp.settings.service_client import settings

    ignored = _NOT_OUTPUT_RELEVANT | exclude
    return {
        name: value
        for name, value in settings.model_dump(mode="json").items()
        if name not in ignored
    }


def local_conversion_context() -> dict[str, object]:
    """Return the cache-key context for conversions executed locally."""
    return {
        "mode": "local",
        "options": _conversion_options(_NOT_RELEVANT_LOCALLY),
        "versions": _VERSION_STAMP,
    }


def remote_conversion_context() -> dict[str, object]:
    """Return the cache-key context for conversions delegated to docling-serve."""
    return {
        "mode": "remote",
        "options": _conversion_options(),
        "versions": _VERSION_STAMP,
    }


def _default_conversion_context() -> dict[str, object]:
    """Return the context for the configured mode.

    Reads conversion_mode to pick between the remote and local contexts. Only
    used when a caller does not supply its own context; the converters pass
    theirs so a fallback conversion is keyed by the converter that ran.
    """
    from docling_mcp.settings.service_client import ConversionMode, settings

    if settings.conversion_mode == ConversionMode.REMOTE:
        return remote_conversion_context()
    return local_conversion_context()


def _looks_like_local_path(source: str) -> bool:
    """Return whether a source should be resolved against the filesystem.

    A URI scheme means the source names a remote object, so it is keyed by
    its string and never read from disk. Windows drive letters parse as
    one-character schemes, so they are detected separately and still treated
    as paths. A schemeless source containing a colon before the first slash
    (a legal POSIX filename) is treated as a URI and keyed by string.
    """
    if PureWindowsPath(source).drive:
        return True
    return not urlsplit(source).scheme


def get_cache_key(
    source: str,
    conversion: dict[str, object] | None = None,
) -> str:
    """Generate a cache key for the document conversion.

    Local files are keyed by their content digest, so identical files reached
    through different paths share one conversion and an edited file triggers a
    new one. Sources carrying a URI scheme (URLs and object-storage URIs) are
    keyed by the source string and never read from disk, so a file that
    happens to sit at the path such a URI collapses to cannot stand in for it.
    The key also covers the conversion configuration, so a result produced
    under a different mode or pipeline setup is not reused. Converters should
    pass their own `conversion` context so a fallback conversion is keyed by
    the converter that actually ran, not by the configured mode.
    """
    key_data: dict[str, object] = {
        "conversion": conversion
        if conversion is not None
        else _default_conversion_context(),
    }

    path = Path(source)
    is_file = False
    if _looks_like_local_path(source):
        try:
            is_file = path.is_file()
        except OSError:
            is_file = False

    if is_file:
        key_data["content"] = _file_content_digest(path)
        # The suffix selects the input format, so identical bytes under
        # different extensions must not share a conversion.
        key_data["format"] = path.suffix.lower()
    else:
        key_data["source"] = source

    key_str = json.dumps(key_data, sort_keys=True)
    hash = hash_string(key_str)
    return hash[:32]
