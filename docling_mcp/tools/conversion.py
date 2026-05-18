"""Tools for converting documents into DoclingDocument objects."""

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from mcp.server.fastmcp import Context
from mcp.shared.exceptions import McpError
from mcp.types import INTERNAL_ERROR, ErrorData, ToolAnnotations
from pydantic import Field

from docling_mcp.logger import setup_logger
from docling_mcp.shared import local_document_cache, mcp

from .converters.base import ConversionOutput
from .converters.factory import get_converter

# Create a default project logger
logger = setup_logger()


def cleanup_memory() -> None:
    """Force garbage collection to free up memory."""
    logger.info("Performed memory cleanup")
    gc.collect()


@dataclass
class IsDoclingDocumentInCacheOutput:
    """Output of the is_document_in_local_cache tool."""

    in_cache: Annotated[
        bool,
        Field(
            description=(
                "Whether the document is already converted and in the local cache."
            )
        ),
    ]


@mcp.tool(
    title="Is Docling document in cache",
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False),
)
def is_document_in_local_cache(
    document_key: Annotated[
        str,
        Field(description="The unique identifier of the document in the local cache."),
    ],
) -> IsDoclingDocumentInCacheOutput:
    """Verify if a Docling document is already converted and in the local cache."""
    return IsDoclingDocumentInCacheOutput(document_key in local_document_cache)


@mcp.tool(
    title="Convert document into Docling document",
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False),
)
def convert_document_into_docling_document(
    source: Annotated[
        str,
        Field(description="The URL or local file path to the document."),
    ],
) -> ConversionOutput:
    """Convert a document of any type from a URL or local path and store in local cache.

    This tool takes a document's URL or local file path, converts it using
    the configured converter (remote API or local), and stores the resulting
    Docling document in a local cache. It returns an output with a boolean
    set to False along with the document's unique cache key. If the document
    was already in the local cache, the conversion is skipped and the output
    boolean is set to True.
    """
    try:
        converter = get_converter()
        result = converter.convert_document(source)

        # Clean up memory after conversion
        cleanup_memory()

        return result

    except Exception as e:
        logger.exception(f"Error converting document: {source}")
        raise McpError(
            ErrorData(code=INTERNAL_ERROR, message=f"Unexpected error: {e!s}")
        ) from e


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
    ctx: Context,  # type: ignore[type-arg]
) -> list[ConversionOutput]:
    """Convert all files from a local directory path and store them in local cache.

    This tool takes a local directory path, converts every file in the directory using
    the configured converter (remote API or local) and stores the resulting Docling
    documents in a local cache. It returns a list of conversion outputs, where each
    output consists of a boolean set to False along with a document's unique cache key.
    If a document was already in the local cache, the conversion is skipped and the
    output boolean is set to True.
    """
    try:
        # Remove any quotes from the source string
        source = source.strip("\"'")
        directory = Path(source)
        files: list[Path] = [f for f in directory.iterdir() if f.is_file()]
        out: list[ConversionOutput] = []

        logger.info(f"Converting {len(files)} files from directory: {source}")
        converter = get_converter()

        for i, file in enumerate(files):
            # Track progress
            await ctx.info(f"Processing file {file}")
            await ctx.report_progress(i + 1, len(files))

            logger.info(f"Processing file {file}")

            try:
                result = converter.convert_document(str(file))
                out.append(result)

                await ctx.debug(
                    f"Completed step {i + 1} with Docling document key: {result.document_key}"
                )
            except Exception as e:
                logger.error(f"Failed to convert {file}: {e}")
                # Continue with other files
                continue

        cleanup_memory()

        return out

    except Exception as e:
        logger.exception(f"Error converting files in directory: {source}")
        raise McpError(
            ErrorData(code=INTERNAL_ERROR, message=f"Unexpected error: {e!s}")
        ) from e
