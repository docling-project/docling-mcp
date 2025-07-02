"""This module manages the cache directory to run Docling MCP tools."""

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Optional

from docling_mcp.logger import setup_logger

# Create a default project logger
logger = setup_logger()


def hash_string_md5(input_string: str) -> str:
    """Creates an md5 hash-string from the input string."""
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


def get_cache_key(
    source: str, enable_ocr: bool = False, ocr_language: Optional[list[str]] = None
) -> str:
    """Generate a cache key for the document conversion."""
    key_data = {
        "source": source,
        "enable_ocr": enable_ocr,
        "ocr_language": ocr_language or [],
    }
    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(key_str.encode()).hexdigest()
