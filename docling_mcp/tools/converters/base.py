"""Base classes and protocols for document converters."""

from dataclasses import dataclass
from typing import Annotated, Protocol

from pydantic import Field


@dataclass
class ConversionOutput:
    """Output of document conversion."""

    from_cache: Annotated[
        bool,
        Field(description="Whether the document was served from the local cache."),
    ]
    document_key: Annotated[
        str,
        Field(description="The unique identifier of the document in the local cache."),
    ]


class DocumentConverterProtocol(Protocol):
    """Protocol for document converters."""

    def convert_document(self, source: str) -> ConversionOutput:
        """Convert a single document from a URL or local path.

        Args:
            source: A local file path, URL, or object-storage URI.

        Returns:
            A `ConversionOutput` with the cache key and a flag indicating
            whether the result was served from cache.
        """
        ...

    def convert_directory(self, source: str) -> list[ConversionOutput]:
        """Convert all files found directly inside a local directory.

        Args:
            source: Path to a local directory.

        Returns:
            A list of `ConversionOutput` instances, one per successfully
            converted file.
        """
        ...

    def is_available(self) -> bool:
        """Check whether this converter is ready to accept conversions.

        Returns:
            `True` if the converter can process documents, `False` otherwise.
        """
        ...
