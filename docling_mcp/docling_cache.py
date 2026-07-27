"""This module manages the cache directory to run Docling MCP tools."""

import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

from pydantic_settings import BaseSettings

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


# Version stamps invalidate cached results when the packages that produce
# them change behavior.
_VERSION_STAMP = {
    "docling_mcp": _package_version("docling-mcp"),
    "docling": _package_version("docling-slim") or _package_version("docling"),
}

# Settings that do not influence the converted output. Everything else in a
# settings model enters the cache key, so a setting added later invalidates
# cached conversions until it is listed here rather than silently serving a
# result produced under a different configuration.
_NOT_OUTPUT_RELEVANT = frozenset(
    {
        "conversion_mode",
        "fallback_to_local",
        "service_api_key",
        "service_max_retries",
        "service_timeout",
    }
)


def _output_relevant(settings: BaseSettings) -> dict[str, object]:
    """Return the settings that change what a conversion produces."""
    return {
        name: value
        for name, value in settings.model_dump(mode="json").items()
        if name not in _NOT_OUTPUT_RELEVANT
    }


def local_conversion_context() -> dict[str, object]:
    """Return the cache-key context for conversions executed locally."""
    from docling_mcp.settings.conversion import settings as conversion_settings

    return {
        "mode": "local",
        "options": _output_relevant(conversion_settings),
        "versions": _VERSION_STAMP,
    }


def remote_conversion_context() -> dict[str, object]:
    """Return the cache-key context for conversions delegated to docling-serve."""
    from docling_mcp.settings.service_client import settings as service_settings

    return {
        "mode": "remote",
        "options": _output_relevant(service_settings),
        "versions": _VERSION_STAMP,
    }


def _default_conversion_context() -> dict[str, object]:
    from docling_mcp.settings.service_client import (
        ConversionMode,
        settings as service_settings,
    )

    if service_settings.conversion_mode == ConversionMode.REMOTE:
        return remote_conversion_context()
    return local_conversion_context()


def get_cache_key(
    source: str,
    enable_ocr: bool = False,
    ocr_language: list[str] | None = None,
    conversion: dict[str, object] | None = None,
) -> str:
    """Generate a cache key for the document conversion.

    Local files are keyed by their content digest, so identical files reached
    through different paths share one conversion and an edited file triggers a
    new one. Other sources (URLs) are keyed by the source string. The key also
    covers the conversion configuration, so a result produced under a
    different mode or pipeline setup is not reused. Converters should pass
    their own `conversion` context so a fallback conversion is keyed by the
    converter that actually ran, not by the configured mode.
    """
    key_data: dict[str, object] = {
        "enable_ocr": enable_ocr,
        "ocr_language": ocr_language or [],
        "conversion": conversion
        if conversion is not None
        else _default_conversion_context(),
    }

    try:
        path = Path(source)
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
