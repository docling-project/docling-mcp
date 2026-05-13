"""Unit tests for remote document converter."""

from typing import Any
from unittest.mock import Mock, patch

import pytest

from docling_mcp.tools.converters.base import ConversionOutput
from docling_mcp.tools.converters.remote import RemoteDocumentConverter


class TestRemoteDocumentConverter:
    """Test suite for RemoteDocumentConverter."""

    @patch("docling_mcp.tools.converters.remote.settings")
    def test_init_without_service_url_raises_error(self, mock_settings: Any) -> None:
        """Test that initialization fails without service URL."""
        mock_settings.service_url = None
        
        with pytest.raises(ValueError, match="DOCLING_SERVICE_URL must be set"):
            RemoteDocumentConverter()

    @patch("docling_mcp.tools.converters.remote.DoclingServiceClient")
    @patch("docling_mcp.tools.converters.remote.settings")
    def test_init_with_service_url_succeeds(self, mock_settings: Any, mock_client_class: Any) -> None:
        """Test successful initialization with service URL."""
        mock_settings.service_url = "https://test.example.com"
        mock_settings.service_api_key = "test-key"
        
        converter = RemoteDocumentConverter()
        
        assert converter is not None
        mock_client_class.assert_called_once_with(
            url="https://test.example.com",
            api_key="test-key"
        )

    @patch("docling_mcp.tools.converters.remote.local_document_cache", {})
    @patch("docling_mcp.tools.converters.remote.DoclingServiceClient")
    @patch("docling_mcp.tools.converters.remote.settings")
    def test_convert_document_from_cache(self, mock_settings: Any, mock_client_class: Any) -> None:
        """Test document conversion when document is in cache."""
        mock_settings.service_url = "https://test.example.com"
        mock_settings.service_api_key = None
        
        # Setup cache
        cache_key = "test_key"
        with patch("docling_mcp.tools.converters.remote.get_cache_key", return_value=cache_key):
            with patch("docling_mcp.tools.converters.remote.local_document_cache", {cache_key: Mock()}):
                converter = RemoteDocumentConverter()
                result = converter.convert_document("test.pdf")
        
        assert isinstance(result, ConversionOutput)
        assert result.from_cache is True
        assert result.document_key == cache_key

    @patch("docling_mcp.tools.converters.remote.local_stack_cache", {})
    @patch("docling_mcp.tools.converters.remote.local_document_cache", {})
    @patch("docling_mcp.tools.converters.remote.DoclingServiceClient")
    @patch("docling_mcp.tools.converters.remote.settings")
    def test_convert_document_success(self, mock_settings: Any, mock_client_class: Any) -> None:
        """Test successful document conversion via remote API."""
        mock_settings.service_url = "https://test.example.com"
        mock_settings.service_api_key = None
        
        # Setup mock client
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Setup mock result
        mock_document = Mock()
        mock_document.add_text = Mock(return_value=Mock())
        mock_result = Mock()
        mock_result.document = mock_document
        mock_result.status = Mock(is_error=False)
        mock_client.convert.return_value = mock_result
        
        cache_key = "test_key"
        with patch("docling_mcp.tools.converters.remote.get_cache_key", return_value=cache_key):
            converter = RemoteDocumentConverter()
            result = converter.convert_document("test.pdf")
        
        assert isinstance(result, ConversionOutput)
        assert result.from_cache is False
        assert result.document_key == cache_key
        mock_client.convert.assert_called_once()

    @patch("docling_mcp.tools.converters.remote.DoclingServiceClient")
    @patch("docling_mcp.tools.converters.remote.settings")
    def test_is_available_healthy(self, mock_settings: Any, mock_client_class: Any) -> None:
        """Test is_available returns True when service is healthy."""
        mock_settings.service_url = "https://test.example.com"
        mock_settings.service_api_key = None
        
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_health = Mock(status="healthy")
        mock_client.health.return_value = mock_health
        
        converter = RemoteDocumentConverter()
        assert converter.is_available() is True

    @patch("docling_mcp.tools.converters.remote.DoclingServiceClient")
    @patch("docling_mcp.tools.converters.remote.settings")
    def test_is_available_unhealthy(self, mock_settings: Any, mock_client_class: Any) -> None:
        """Test is_available returns False when service is unhealthy."""
        mock_settings.service_url = "https://test.example.com"
        mock_settings.service_api_key = None
        
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.health.side_effect = Exception("Service unavailable")
        
        converter = RemoteDocumentConverter()
        assert converter.is_available() is False
