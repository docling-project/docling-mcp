"""Tools for accessing Docling documents stored on disk as JSON files."""

import os
from pathlib import Path

from docling_core.types.doc.document import (
    ContentLayer,
    DoclingDocument,
)
from docling_core.types.doc.labels import DocItemLabel

from docling_mcp.docling_cache import get_cache_dir
from docling_mcp.logger import setup_logger
from docling_mcp.shared import local_document_cache, local_stack_cache, mcp

# Create a default project logger
logger = setup_logger()


@mcp.tool()
def get_json_cache_keys() -> str:
    """Lists all cache keys that correspond to json files on the disk.

    Args:
      None

    Returns:
      This tool returns a string with all cache keys that correspond to json files on the disk.

    Raises:
      ValueError: If no json files are found in the cache directory.
    """
    cache_dir = get_cache_dir()
    json_files = [f.name for f in Path(cache_dir).glob("*.json")]

    if not json_files:
        raise ValueError("No json files found in the cache directory.")

    return "Cache keys corresponding to json files on the disk:\n\n" + "\n".join(
        json_files
    )


@mcp.tool()
def add_docling_document_from_disk_to_cache(cache_key: str) -> str:
    """Loads a Docling Document from a json file on the disk and stores in the local cache.

    Args:
      cache_key (str): Document identifier from the original item in the local cache that was saved to json. Current name of the file on the disk

    Returns:
      This tool returns a string that indicates the status/result of the operation

    Raises:
      ValueError: If the specified cache_key does not exist as a json file on the disk.
      ValueError: If the specified cache_key exists as a json file on the disk, but cannot be loaded as a DoclingDocument.
    """
    cache_dir = get_cache_dir()
    file_path = cache_dir / f"{cache_key}.json"

    if os.path.exists(file_path):
        try:
            # Load the JSON file as a DoclingDocument
            logger.info(f"Loading DoclingDocument from {file_path}")
            docling_document = DoclingDocument.load_from_json(file_path)

            local_document_cache[cache_key] = docling_document

            item = docling_document.add_text(
                label=DocItemLabel.TEXT,
                text=f"document loaded from json with key: {cache_key}",
                content_layer=ContentLayer.FURNITURE,
            )

            local_stack_cache[cache_key] = [item]

            logger.info(
                f"Successfully loaded and cached DoclingDocument for cache_key: {cache_key}"
            )
            return f"DoclingDocument with cache_key {cache_key} loaded and cached successfully."
        except Exception as e:
            raise ValueError(
                f"JSON file for cache_key {cache_key} exists, but could not be loaded as a DoclingDocument."
            ) from e

    raise ValueError(
        f"JSON file for cache_key {cache_key} does not exist. Use the 'get_json_cache_keys' tools to list cache keys that correspond to json files on the disk."
    )
