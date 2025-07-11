"""Tools for accessing Docling documents stored on disk as JSON files."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from pydantic import Field

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


@dataclass
class JSONKeys:
    """Cache keys for JSON files on disk."""

    files_exist: Annotated[
        bool,
        Field(
            description="Indicates whether any JSON files exist in the cache directory."
        ),
    ]

    cache_keys: Annotated[
        str,
        Field(
            description="A string containing all cache keys that correspond to json files on the disk."
        ),
    ]


@mcp.tool()
def get_json_cache_keys() -> JSONKeys:
    """Lists all cache keys that correspond to json files on the disk."""
    cache_dir = get_cache_dir()
    json_files = [f.name for f in Path(cache_dir).glob("*.json")]

    if not json_files:
        return JSONKeys(
            files_exist=False, cache_keys="No JSON files found in the cache directory."
        )

    return JSONKeys(
        files_exist=True,
        cache_keys="Cache keys corresponding to json files on the disk:\n\n"
        + "\n".join(
            json_files,
        ),
    )


@dataclass
class JSONLoadResult:
    """Result of loading a Docling Document from a JSON file to the local cache."""

    success: Annotated[
        bool,
        Field(description="Indicates whether the load operation was successful."),
    ]

    message: Annotated[
        str,
        Field(description="Status message detailing the result of the load operation."),
    ]


@mcp.tool()
def add_docling_document_from_disk_to_cache(
    cache_key: Annotated[
        str,
        Field(
            description="Document identifier from the original item in the local cache that was saved to json. Current name of the file on the disk without the .json extension."
        ),
    ],
) -> JSONLoadResult:
    """Loads a Docling Document from a json file on the disk and stores in the local cache."""
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
            return JSONLoadResult(
                success=True,
                message=f"DoclingDocument with cache_key {cache_key} loaded and cached successfully.",
            )
        except Exception:
            return JSONLoadResult(
                success=False,
                message=f"JSON file for cache_key {cache_key} exists, but could not be loaded as a DoclingDocument.",
            )

    return JSONLoadResult(
        success=False,
        message=f"JSON file for cache_key {cache_key} does not exist. Use the 'get_json_cache_keys' tools to list cache keys that correspond to json files on the disk.",
    )
