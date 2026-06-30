"""Base classes and protocols for document converters."""

from dataclasses import dataclass
from typing import Protocol


@dataclass
class ConversionOutput:
    """Output of document conversion."""

    from_cache: bool
    document_key: str


class DocumentConverterProtocol(Protocol):
    """Protocol for document converters."""

    def convert_document(
        self,
        source: str,
        do_ocr: bool | None = None,
        do_table_structure: bool | None = None,
        keep_images: bool | None = None,
    ) -> ConversionOutput:
        """Convert a single document.

        All pipeline-option overrides are optional; when None, the converter
        falls back to its environment-variable defaults (DOCLING_MCP_DO_OCR,
        DOCLING_MCP_DO_TABLE_STRUCTURE, DOCLING_MCP_KEEP_IMAGES).
        """
        ...

    def convert_directory(self, source: str) -> list[ConversionOutput]:
        """Convert all files in a directory."""
        ...

    def is_available(self) -> bool:
        """Check if this converter is available."""
        ...
