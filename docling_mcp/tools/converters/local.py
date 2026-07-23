"""Local document converter using DocumentConverter."""

from pathlib import Path

from docling_core.types.doc.common.content_layer import ContentLayer
from docling_core.types.doc.labels import DocItemLabel

from docling_mcp.docling_cache import get_cache_key
from docling_mcp.logger import setup_logger
from docling_mcp.settings.conversion import settings
from docling_mcp.shared import local_document_cache, local_stack_cache

from .base import ConversionOutput

# Import DocumentConverter only if available
try:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import (
        DocumentConverter,
        FormatOption,
        PdfFormatOption,
    )

    LOCAL_CONVERSION_AVAILABLE = True
except ImportError:
    LOCAL_CONVERSION_AVAILABLE = False

logger = setup_logger()


class LocalDocumentConverter:
    """Converter using local DocumentConverter."""

    def __init__(self) -> None:
        """Initialize local converter."""
        if not LOCAL_CONVERSION_AVAILABLE:
            raise ImportError(
                "Local conversion requires docling-mcp[local] extra. "
                "Install with: pip install docling-mcp[local]"
            )
        self._converter: DocumentConverter | None = None
        logger.info("Initialized local document converter")

    def _get_converter(self) -> "DocumentConverter":
        """Get or create DocumentConverter instance."""
        if self._converter is not None:
            return self._converter

        pipeline_options = PdfPipelineOptions()
        pipeline_options.generate_page_images = settings.keep_images
        pipeline_options.do_ocr = settings.do_ocr
        pipeline_options.do_table_structure = settings.do_table_structure

        format_options: dict[InputFormat, FormatOption] = {
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            InputFormat.IMAGE: PdfFormatOption(pipeline_options=pipeline_options),
        }

        logger.info(f"Creating DocumentConverter with options: {format_options}")
        self._converter = DocumentConverter(format_options=format_options)
        return self._converter

    def convert_document(self, source: str) -> ConversionOutput:
        """Convert document using local converter."""
        source = source.strip("\"'")
        logger.info(f"Converting document locally: {source}")

        cache_key = get_cache_key(source)

        if cache_key in local_document_cache:
            logger.info(f"Document found in cache: {cache_key}")
            return ConversionOutput(True, cache_key)

        # Get converter and convert
        converter = self._get_converter()
        result = converter.convert(source)

        # Check for errors
        has_error = False
        if hasattr(result, "status"):
            if hasattr(result.status, "is_error"):
                has_error = result.status.is_error
            elif hasattr(result.status, "error"):
                has_error = result.status.error

        if has_error:
            raise Exception(f"Local conversion failed: {result.errors}")

        # Cache the result
        local_document_cache[cache_key] = result.document

        # Add source metadata
        item = result.document.add_text(
            label=DocItemLabel.TEXT,
            text=f"source: {source}",
            content_layer=ContentLayer.FURNITURE,
        )
        local_stack_cache[cache_key] = [item]

        logger.info(f"Successfully converted document: {cache_key}")
        return ConversionOutput(False, cache_key)

    def convert_directory(self, source: str) -> list[ConversionOutput]:
        """Convert all files in a directory using local converter."""
        source = source.strip("\"'")
        directory = Path(source)
        files: list[Path] = [f for f in directory.iterdir() if f.is_file()]
        out: list[ConversionOutput] = []

        logger.info(f"Converting {len(files)} files from directory: {source}")

        for file in files:
            try:
                result = self.convert_document(str(file))
                out.append(result)
            except Exception as e:
                logger.error(f"Failed to convert {file}: {e}")
                # Continue with other files
                continue

        return out

    def is_available(self) -> bool:
        """Check if local converter is available."""
        return LOCAL_CONVERSION_AVAILABLE
