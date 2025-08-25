"""Tools for converting documents into DoclingDocument objects."""

import gc
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from mcp.server.fastmcp import Context
from mcp.shared.exceptions import McpError
from mcp.types import INTERNAL_ERROR, ErrorData
from pydantic import Field

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, FormatOption, PdfFormatOption
from docling_core.types.doc.document import (
    ContentLayer,
)
from docling_core.types.doc.labels import (
    DocItemLabel,
)

from docling_mcp.docling_cache import get_cache_key
from docling_mcp.logger import setup_logger
from docling_mcp.settings.conversion import settings
from docling_mcp.shared import local_document_cache, local_stack_cache, mcp

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


@mcp.tool(title="Is Docling document in cache")
def is_document_in_local_cache(
    document_key: Annotated[
        str,
        Field(description="The unique identifier of the document in the local cache."),
    ],
) -> IsDoclingDocumentInCacheOutput:
    """Verify if a Docling document is already converted and in the local cache."""
    return IsDoclingDocumentInCacheOutput(document_key in local_document_cache)


@dataclass
class ConvertDocumentOutput:
    """Output of the convert_document_into_docling_document tool."""

    from_cache: Annotated[
        bool,
        Field(
            description=(
                "Whether the document was already converted in the local cache."
            )
        ),
    ]
    document_key: Annotated[
        str,
        Field(description="The unique identifier of the document in the local cache."),
    ]


@lru_cache
def _get_converter() -> DocumentConverter:
    pipeline_options = PdfPipelineOptions()
    # pipeline_options.do_ocr = False  # Skip OCR for faster processing (enable for scanned docs)
    pipeline_options.generate_page_images = settings.keep_images

    format_options: dict[InputFormat, FormatOption] = {
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        InputFormat.IMAGE: PdfFormatOption(pipeline_options=pipeline_options),
    }

    logger.info(f"Creating DocumentConverter with format_options: {format_options}")
    return DocumentConverter(format_options=format_options)


@mcp.tool(title="Convert document into Docling document")
def convert_document_into_docling_document(
    source: Annotated[
        str,
        Field(description="The URL or local file path to the document."),
    ],
) -> ConvertDocumentOutput:
    """Convert a document of any type from a URL or local path and store in local cache.

    This tool takes a document's URL or local file path, converts it using
    Docling's DocumentConverter, and stores the resulting Docling document in a
    local cache. It returns an output with a boolean set to False along with the
    document's unique cache key. If the document was already in the local cache,
    the conversion is skipped and the output boolean is set to True.
    """
    try:
        # Remove any quotes from the source string
        source = source.strip("\"'")

        # Log the cleaned source
        logger.info(f"Processing document from source: {source}")

        # Generate cache key
        cache_key = get_cache_key(source)

        if cache_key in local_document_cache:
            logger.info(f"{source} has previously been added.")
            return ConvertDocumentOutput(True, cache_key)

        # Get converter
        converter = _get_converter()

        # Convert the document
        logger.info("Start conversion")
        result = converter.convert(source)

        # Check for errors - handle different API versions
        has_error = False
        error_message = ""

        # Try different ways to check for errors based on the API version
        if hasattr(result, "status"):
            if hasattr(result.status, "is_error"):
                has_error = result.status.is_error
            elif hasattr(result.status, "error"):
                has_error = result.status.error

        if hasattr(result, "errors") and result.errors:
            has_error = True
            error_message = str(result.errors)

        if has_error:
            error_msg = f"Conversion failed: {error_message}"
            raise McpError(ErrorData(code=INTERNAL_ERROR, message=error_msg))

        local_document_cache[cache_key] = result.document

        item = result.document.add_text(
            label=DocItemLabel.TEXT,
            text=f"source: {source}",
            content_layer=ContentLayer.FURNITURE,
        )

        local_stack_cache[cache_key] = [item]

        # Log completion
        logger.info(f"Successfully created the Docling document: {source}")

        # Clean up memory
        cleanup_memory()

        return ConvertDocumentOutput(False, cache_key)

    except Exception as e:
        logger.exception(f"Error converting document: {source}")
        raise McpError(
            ErrorData(code=INTERNAL_ERROR, message=f"Unexpected error: {e!s}")
        ) from e


@mcp.tool(
    title="Convert files from directory into Docling document", structured_output=True
)
async def convert_directory_files_into_docling_document(
    source: Annotated[
        str,
        Field(description="The path to a local directory"),
    ],
    ctx: Context,  # type: ignore[type-arg]
) -> list[ConvertDocumentOutput]:
    """Convert all files from a local directory path and store them in local cache.

    This tool takes a local directory path, converts every file in the directory using
    Docling's DocumentConverter and stores the resulting Docling documents in a local
    cache. It returns a list of conversion outputs, where each output consists of a
    boolean set to False along with a document's unique cache key. If a document was
    already in the local cache, the conversion is skipped and the output boolean is set
    to True.
    """
    try:
        # Remove any quotes from the source string
        source = source.strip("\"'")
        directory = Path(source)
        files: list[Path] = list(directory.iterdir())
        out: list[ConvertDocumentOutput] = []
        logger.info("Getting the converter")
        converter = _get_converter()

        for i, file in enumerate(files):
            if not file.is_file():
                continue

            # Track progress
            await ctx.info(f"Processing file {file}")
            await ctx.report_progress(i + 1, len(files))

            logger.info(f"Processing file {file}")
            cache_key = get_cache_key(str(file))
            if cache_key in local_document_cache:
                logger.info(f"{file} has been previously converted.")
                out.append(ConvertDocumentOutput(True, cache_key))
            else:
                # Convert the document
                logger.info("Start conversion")
                result = converter.convert(file)
                has_error = False
                error_message = ""
                if hasattr(result, "status"):
                    if hasattr(result.status, "is_error"):
                        has_error = result.status.is_error
                    elif hasattr(result.status, "error"):
                        has_error = result.status.error

                if hasattr(result, "errors") and result.errors:
                    has_error = True
                    error_message = str(result.errors)

                if has_error:
                    error_msg = f"Conversion failed: {error_message}"
                    raise McpError(ErrorData(code=INTERNAL_ERROR, message=error_msg))

                local_document_cache[cache_key] = result.document

                item = result.document.add_text(
                    label=DocItemLabel.TEXT,
                    text=f"source: {file}",
                    content_layer=ContentLayer.FURNITURE,
                )

                local_stack_cache[cache_key] = [item]

                await ctx.debug(
                    f"Completed step {i + 1} with Docling document key: {cache_key}"
                )
                logger.info(f"Successfully created the Docling document: {file}")
                out.append(ConvertDocumentOutput(False, cache_key))

        cleanup_memory()

        return out

    except Exception as e:
        logger.exception(f"Error converting files in directory: {source}")
        raise McpError(
            ErrorData(code=INTERNAL_ERROR, message=f"Unexpected error: {e!s}")
        ) from e
