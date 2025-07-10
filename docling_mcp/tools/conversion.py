"""Tools for converting documents into DoclingDocument objects."""

import gc
from dataclasses import dataclass
from typing import Annotated

from mcp.shared.exceptions import McpError
from mcp.types import INTERNAL_ERROR, ErrorData
from pydantic import Field

from docling.datamodel.accelerator_options import AcceleratorDevice
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
class ConvertPdfDocumentOutput:
    """Output of the convert_pdf_document_into_docling_document tool."""

    success: Annotated[
        bool,
        Field(
            description=(
                "Whether the document has been successfully converted or it was already "
                "converted and in the local cache."
            )
        ),
    ]
    document_key: Annotated[
        str,
        Field(description="The unique identifier of the document in the local cache."),
    ]


@mcp.tool(title="Convert PDF document into Docling document")
def convert_pdf_document_into_docling_document(
    source: Annotated[
        str,
        Field(description="The URL or local file path to the PDF document."),
    ],
) -> ConvertPdfDocumentOutput:
    """Convert a PDF document from a URL or local path and store in local cache.

    This tool takes a PDF document's URL or local file path, converts it using
    Docling's DocumentConverter and stores the resulting Docling document in a
    local cache. It returns an output with a boolean set to True along with the
    document's unique cache key. If the document was already in the local cache,
    the conversion is skipped and the output boolean is set to False.
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
            return ConvertPdfDocumentOutput(False, cache_key)

        # Log the start of processing
        logger.info("Set up pipeline options")

        # Configure pipeline
        # ocr_options = EasyOcrOptions(lang=ocr_language or ["en"])
        pipeline_options = PdfPipelineOptions(
            # do_ocr=False,
            # ocr_options=ocr_options,
            accelerator_device=AcceleratorDevice.MPS  # Explicitly set MPS
        )
        format_options: dict[InputFormat, FormatOption] = {
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }

        # Create converter with MPS acceleration
        logger.info(f"Creating DocumentConverter with format_options: {format_options}")
        converter = DocumentConverter(format_options=format_options)

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

        return ConvertPdfDocumentOutput(True, cache_key)

    except Exception as e:
        logger.exception(f"Error converting document: {source}")
        raise McpError(
            ErrorData(code=INTERNAL_ERROR, message=f"Unexpected error: {e!s}")
        ) from e


# @mcp.tool()
# def convert_attachments_into_docling_document(
#     pdf_payloads: list[Annotated[bytes, {"media_type": "application/octet-stream"}]],
# ) -> list[dict[str, Any]]:
#     """Process a pdf files attachment from Claude Desktop.

#     Args:
#         pdf_payloads: PDF document as binary data from the attachment

#     Returns:
#         A dictionary with processed results
#     """
#     results = []
#     for pdf_payload in pdf_payloads:
#         # Example processing - you can replace this with your actual processing logic
#         file_size = len(pdf_payload)

#         # First few bytes as hex for identification
#         header_bytes = pdf_payload[:10].hex()

#         # You can implement file type detection, parsing, or any other processing here
#         # For example, if it's an image, you might use PIL to process it

#         results.append(
#             {
#                 "file_size_bytes": file_size,
#                 "header_hex": header_bytes,
#                 "status": "processed",
#             }
#         )

#     return results
