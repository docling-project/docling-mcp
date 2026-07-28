"""Conversion tools package with remote and local support."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import ConversionOutput, DocumentConverterProtocol

if TYPE_CHECKING:
    from .local import LocalDocumentConverter
    from .remote import RemoteDocumentConverter


# Use lazy import for factory to avoid importing DocumentConverter unnecessarily
def get_converter() -> RemoteDocumentConverter | LocalDocumentConverter:
    """Get the appropriate converter based on settings."""
    from .factory import get_converter as _get_converter

    return _get_converter()


__all__ = ["ConversionOutput", "DocumentConverterProtocol", "get_converter"]
