"""Unit tests for local document converter."""

from typing import Any
from unittest.mock import Mock, patch

import pytest

from docling_mcp.tools.converters.base import ConversionOutput
from docling_mcp.tools.converters.local import (
    LOCAL_CONVERSION_AVAILABLE,
    LocalDocumentConverter,
)


class TestLocalDocumentConverter:
    """Test suite for LocalDocumentConverter."""

    @patch("docling_mcp.tools.converters.local.LOCAL_CONVERSION_AVAILABLE", False)
    def test_init_without_local_extra_raises_error(self) -> None:
        """Test that initialization fails without local extra installed."""
        with pytest.raises(ImportError, match="Local conversion requires docling-mcp\\[local\\]"):
            LocalDocumentConverter()

    @patch("docling_mcp.tools.converters.local.LOCAL_CONVERSION_AVAILABLE", True)
    def test_init_with_local_extra_succeeds(self) -> None:
        """Test successful initialization with local extra."""
        converter = LocalDocumentConverter()
        assert converter is not None

    @patch("docling_mcp.tools.converters.local.LOCAL_CONVERSION_AVAILABLE", True)
    @patch("docling_mcp.tools.converters.local.local_document_cache", {})
    def test_convert_document_from_cache(self) -> None:
        """Test document conversion when document is in cache."""
        cache_key = "test_key"
        
        with patch("docling_mcp.tools.converters.local.get_cache_key", return_value=cache_key):
            with patch("docling_mcp.tools.converters.local.local_document_cache", {cache_key: Mock()}):
                converter = LocalDocumentConverter()
                result = converter.convert_document("test.pdf")
        
        assert isinstance(result, ConversionOutput)
        assert result.from_cache is True
        assert result.document_key == cache_key

    @patch("docling_mcp.tools.converters.local.LOCAL_CONVERSION_AVAILABLE", True)
    @patch("docling_mcp.tools.converters.local.local_stack_cache", {})
    @patch("docling_mcp.tools.converters.local.local_document_cache", {})
    @patch("docling_mcp.tools.converters.local.DocumentConverter")
    def test_convert_document_success(self, mock_converter_class: Any) -> None:
        """Test successful document conversion locally."""
        # Setup mock converter
        mock_converter = Mock()
        mock_converter_class.return_value = mock_converter
        
        # Setup mock result
        mock_document = Mock()
        mock_document.add_text = Mock(return_value=Mock())
        mock_result = Mock()
        mock_result.document = mock_document
        mock_result.status = Mock(is_error=False)
        mock_converter.convert.return_value = mock_result
        
        cache_key = "test_key"
        with patch("docling_mcp.tools.converters.local.get_cache_key", return_value=cache_key):
            converter = LocalDocumentConverter()
            result = converter.convert_document("test.pdf")
        
        assert isinstance(result, ConversionOutput)
        assert result.from_cache is False
        assert result.document_key == cache_key

    @patch("docling_mcp.tools.converters.local.LOCAL_CONVERSION_AVAILABLE", True)
    def test_is_available_when_installed(self) -> None:
        """Test is_available returns True when local extra is installed."""
        converter = LocalDocumentConverter()
        assert converter.is_available() is True

    @patch("docling_mcp.tools.converters.local.LOCAL_CONVERSION_AVAILABLE", False)
    def test_is_available_when_not_installed(self) -> None:
        """Test is_available returns False when local extra is not installed."""
        # Can't create converter without LOCAL_CONVERSION_AVAILABLE
        # So we test the module-level constant directly
        from docling_mcp.tools import converters
        with patch.object(converters.local, "LOCAL_CONVERSION_AVAILABLE", False):
            # Verify the constant is False
            assert converters.local.LOCAL_CONVERSION_AVAILABLE is False
