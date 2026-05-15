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

    def convert_document(self, source: str) -> ConversionOutput:
        """Convert a single document."""
        ...

    def convert_directory(self, source: str) -> list[ConversionOutput]:
        """Convert all files in a directory."""
        ...

    def is_available(self) -> bool:
        """Check if this converter is available."""
        ...
