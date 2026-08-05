"""Remote document converter using Docling Serve API."""

from pathlib import Path

from docling.datamodel.base_models import OutputFormat
from docling.datamodel.service.options import ConvertDocumentsOptions
from docling.service_client import DoclingServiceClient
from docling_core.types.doc.common.content_layer import ContentLayer
from docling_core.types.doc.labels import DocItemLabel
from docling_core.types.io import DocumentStream

from docling_mcp.docling_cache import get_cache_key
from docling_mcp.logger import setup_logger
from docling_mcp.settings.service_client import settings
from docling_mcp.shared import local_document_cache, local_stack_cache

from .base import ConversionOutput
from .sources import fetched_stream

logger = setup_logger()


class RemoteDocumentConverter:
    """Converter using Docling Serve API."""

    def __init__(self) -> None:
        """Initialize remote converter."""
        if not settings.service_url:
            raise ValueError(
                "DOCLING_MCP_SERVICE_URL is not set but "
                "DOCLING_MCP_CONVERSION_MODE=remote. "
                "Set it via environment variable or in a .env file. "
                "See .env.example for all available configuration variables."
            )

        # DoclingServiceClient requires api_key to be str, not Optional[str]
        api_key = (
            settings.service_api_key if settings.service_api_key is not None else ""
        )

        self.client = DoclingServiceClient(
            url=settings.service_url,
            api_key=api_key,
        )
        logger.info(f"Initialized remote converter with URL: {settings.service_url}")

    def convert_document(self, source: str) -> ConversionOutput:
        """Convert document using remote API.

        The cache is resolved before the source is fetched, so a repeat call
        for an object-storage URI does not download the body again only to
        discard it. Object bodies are uploaded from memory rather than
        staged through a temporary file.
        """
        source = source.strip("\"'")
        cache_key = get_cache_key(source)

        if cache_key in local_document_cache:
            logger.info(f"Document found in cache: {cache_key}")
            return ConversionOutput(True, cache_key)

        logger.info(f"Converting document via remote API: {source}")
        return self._convert_resolved_source(source, fetched_stream(source), cache_key)

    def _convert_resolved_source(
        self, source: str, resolved: str | DocumentStream, cache_key: str
    ) -> ConversionOutput:
        """Convert a resolved source, recording the original source."""
        # Configure conversion options
        options = ConvertDocumentsOptions(
            do_ocr=settings.do_ocr,
            do_table_structure=settings.do_table_structure,
            include_images=settings.keep_images,
            images_scale=settings.images_scale,
            to_formats=[OutputFormat.JSON],
            abort_on_error=False,
        )

        # Convert via remote API
        result = self.client.convert(source=resolved, options=options)

        # Check for errors
        if hasattr(result, "status") and hasattr(result.status, "is_error"):
            if result.status.is_error:
                raise Exception(f"Remote conversion failed: {result.errors}")

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
        """Convert all files in a directory using remote API."""
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
        """Check if remote service is available."""
        try:
            health = self.client.health()
            return health.status == "healthy"
        except Exception as e:
            logger.warning(f"Remote service health check failed: {e}")
            return False
