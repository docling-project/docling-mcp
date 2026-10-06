"""Tools for converting documents into DoclingDocument objects."""

import asyncio
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from mcp.server.mcpserver import Context
from mcp.types import ToolAnnotations
from pydantic import Field

from docling_mcp.logger import setup_logger
from docling_mcp.shared import drop_document, local_document_cache, mcp

from .converters.base import ConversionOutput
from .converters.factory import get_converter
from .converters.sources import supported_uri_schemes

# Create a default project logger
logger = setup_logger()

# Derived from the scheme table so the tool description cannot drift from the
# schemes actually resolved.
_SOURCE_DESCRIPTION = (
    "The URL or local file path to the document. Object-storage URIs "
    f"({', '.join(f'{s}://' for s in supported_uri_schemes())}) are supported "
    "when the matching provider extra is installed."
)


def cleanup_memory() -> None:
    """Run a CPython garbage-collection cycle.

    This releases any cyclic garbage that Python's reference-counter missed.
    It does not evict documents from the in-memory cache; those are evicted
    automatically by the LRU policy or by calling remove_document_from_local_cache.
    """
    gc.collect()
    logger.info("Performed garbage collection")


@dataclass
class CheckDocumentInCacheOutput:
    """Output of the check_document_in_local_cache tool."""

    in_cache: Annotated[
        bool,
        Field(
            description=(
                "Whether the document is already converted and in the local cache."
            )
        ),
    ]


@mcp.tool(
    title="Check if Docling document is in cache",
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False),
)
def check_document_in_local_cache(
    document_key: Annotated[
        str,
        Field(description="The unique identifier of the document in the local cache."),
    ],
) -> CheckDocumentInCacheOutput:
    """Check whether a Docling document is already converted and present in the local cache."""
    return CheckDocumentInCacheOutput(document_key in local_document_cache)


@dataclass
class RemoveDocumentFromCacheOutput:
    """Output of the remove_document_from_local_cache tool."""

    dropped: Annotated[
        bool,
        Field(
            description=(
                "True if the document was present in the cache and has been "
                "removed; False if the key was not found."
            )
        ),
    ]


@mcp.tool(
    title="Remove document from local cache",
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True),
)
def remove_document_from_local_cache(
    document_key: Annotated[
        str,
        Field(
            description="The unique identifier of the document to remove from the local cache."
        ),
    ],
) -> RemoveDocumentFromCacheOutput:
    """Remove a document from the local cache and release its memory.

    Call this tool when a client is finished with a document and wants to
    release the memory it occupies. After a successful call the key is no
    longer present in the cache.
    """
    removed = drop_document(document_key)
    if removed:
        logger.info(f"Removed document from cache: {document_key}")
    else:
        logger.debug(f"remove_document_from_local_cache: key not found: {document_key}")
    return RemoveDocumentFromCacheOutput(dropped=removed)


@mcp.tool(
    title="Convert document into Docling document",
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False),
)
def convert_document_into_docling_document(
    source: Annotated[
        str,
        Field(description=_SOURCE_DESCRIPTION),
    ],
) -> ConversionOutput:
    """Convert a document from a URL or local path into a Docling document.

    Use this tool when you have an existing file (PDF, DOCX, HTML, image, etc.)
    that you want to load and parse.

    This tool takes a document's URL or local file path, converts it using
    the configured converter (remote API or local), and stores the resulting
    Docling document in a local cache. It returns an output with a boolean
    set to False along with the document's unique cache key. If the document
    was already in the local cache, the conversion is skipped and the output
    boolean is set to True.
    """
    converter = get_converter()
    result = converter.convert_document(source)

    # Clean up memory after conversion
    cleanup_memory()

    return result


@mcp.tool(
    title="Convert files from directory into Docling document",
    structured_output=True,
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False),
)
async def convert_directory_files_into_docling_document(
    source: Annotated[
        str,
        Field(description="The path to a local directory"),
    ],
    ctx: Context,
) -> list[ConversionOutput]:
    """Convert all files from a local directory path and store them in local cache.

    This tool takes a local directory path, converts every file in the directory using
    the configured converter (remote API or local) and stores the resulting Docling
    documents in a local cache. It returns a list of conversion outputs, where each
    output consists of a boolean set to False along with a document's unique cache key.
    If a document was already in the local cache, the conversion is skipped and the
    output boolean is set to True.
    """
    # Remove any quotes from the source string
    source = source.strip("\"'")
    directory = Path(source)
    files: list[Path] = await asyncio.to_thread(
        lambda: [f for f in directory.iterdir() if f.is_file()]
    )
    out: list[ConversionOutput] = []

    logger.info(f"Converting {len(files)} files from directory: {source}")
    converter = get_converter()

    for i, file in enumerate(files):
        logger.info(f"Processing file {file}")
        await ctx.report_progress(i + 1, len(files))

        try:
            result = converter.convert_document(str(file))
            out.append(result)
            logger.debug(
                f"Completed step {i + 1} with Docling document key: {result.document_key}"
            )
        except Exception as e:
            logger.error(f"Failed to convert {file}: {e}")
            # Continue with other files
            continue

    cleanup_memory()

    return out


@dataclass
class ListCachedDocumentsOutput:
    """Output of the list_cached_documents tool."""

    document_keys: Annotated[
        list[str],
        Field(
            description=(
                "The list of document keys currently held in the local cache. "
                "Empty when no documents are cached."
            )
        ),
    ]


@mcp.tool(
    title="List cached Docling documents",
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False),
)
def list_cached_documents() -> ListCachedDocumentsOutput:
    """Return the keys of all Docling documents currently held in the local cache.

    Use this tool to discover which documents are available before calling any
    tool that requires a document_key. An empty list means the cache is empty
    and a document must first be loaded with convert_document_into_docling_document
    or created with create_new_docling_document.
    """
    return ListCachedDocumentsOutput(document_keys=list(local_document_cache.keys()))
