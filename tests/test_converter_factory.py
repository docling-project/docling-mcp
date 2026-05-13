"""Unit tests for converter factory and mode selection."""

from typing import Any
from unittest.mock import Mock, patch

import pytest

from docling_mcp.settings.service_client import ConversionMode
from docling_mcp.tools.converters.factory import get_converter


class TestConverterFactory:
    """Test suite for converter factory."""

    @patch("docling_mcp.tools.converters.remote.DoclingServiceClient")
    @patch("docling_mcp.tools.converters.remote.settings")
    @patch("docling_mcp.tools.converters.factory.settings")
    def test_get_converter_remote_mode(
        self, mock_factory_settings: Any, mock_remote_settings: Any, mock_client: Any
    ) -> None:
        """Test factory returns remote converter in remote mode."""
        mock_factory_settings.conversion_mode = ConversionMode.REMOTE
        mock_factory_settings.fallback_to_local = False
        mock_remote_settings.service_url = "https://test.example.com"
        mock_remote_settings.service_api_key = None

        # Mock the client health check
        mock_client_instance = Mock()
        mock_client_instance.health.return_value = Mock(status="healthy")
        mock_client.return_value = mock_client_instance

        result = get_converter()

        assert result is not None
        assert result.__class__.__name__ == "RemoteDocumentConverter"

    @patch("docling_mcp.tools.converters.local.LOCAL_CONVERSION_AVAILABLE", True)
    @patch("docling_mcp.tools.converters.factory.settings")
    def test_get_converter_local_mode(self, mock_settings: Any) -> None:
        """Test factory returns local converter in local mode."""
        mock_settings.conversion_mode = ConversionMode.LOCAL

        result = get_converter()

        assert result is not None
        assert result.__class__.__name__ == "LocalDocumentConverter"

    @patch("docling_mcp.tools.converters.local.LOCAL_CONVERSION_AVAILABLE", False)
    @patch("docling_mcp.tools.converters.factory.settings")
    def test_get_converter_local_mode_not_available(self, mock_settings: Any) -> None:
        """Test factory raises error when local mode requested but not available."""
        mock_settings.conversion_mode = ConversionMode.LOCAL

        with pytest.raises(ImportError, match="Local conversion mode requires"):
            get_converter()

    @patch("docling_mcp.tools.converters.local.LOCAL_CONVERSION_AVAILABLE", True)
    @patch("docling_mcp.tools.converters.remote.DoclingServiceClient")
    @patch("docling_mcp.tools.converters.remote.settings")
    @patch("docling_mcp.tools.converters.factory.settings")
    def test_get_converter_fallback_to_local(
        self, mock_factory_settings: Any, mock_remote_settings: Any, mock_client: Any
    ) -> None:
        """Test factory falls back to local when remote fails and fallback enabled."""
        mock_factory_settings.conversion_mode = ConversionMode.REMOTE
        mock_factory_settings.fallback_to_local = True
        mock_remote_settings.service_url = (
            None  # This will cause RemoteDocumentConverter to fail
        )

        result = get_converter()

        assert result is not None
        assert result.__class__.__name__ == "LocalDocumentConverter"

    @patch("docling_mcp.tools.converters.local.LOCAL_CONVERSION_AVAILABLE", False)
    @patch("docling_mcp.tools.converters.remote.settings")
    @patch("docling_mcp.tools.converters.factory.settings")
    def test_get_converter_no_fallback_when_local_unavailable(
        self, mock_factory_settings: Any, mock_remote_settings: Any
    ) -> None:
        """Test factory raises error when remote fails and local not available."""
        mock_factory_settings.conversion_mode = ConversionMode.REMOTE
        mock_factory_settings.fallback_to_local = True
        mock_remote_settings.service_url = (
            None  # This will cause RemoteDocumentConverter to fail
        )

        with pytest.raises(ValueError, match="DOCLING_SERVICE_URL must be set"):
            get_converter()

    @patch("docling_mcp.tools.converters.remote.DoclingServiceClient")
    @patch("docling_mcp.tools.converters.remote.settings")
    @patch("docling_mcp.tools.converters.factory.settings")
    def test_get_converter_remote_unavailable_no_fallback(
        self, mock_factory_settings: Any, mock_remote_settings: Any, mock_client: Any
    ) -> None:
        """Test factory returns converter even when remote unavailable and fallback disabled."""
        mock_factory_settings.conversion_mode = ConversionMode.REMOTE
        mock_factory_settings.fallback_to_local = False
        mock_remote_settings.service_url = "https://test.example.com"
        mock_remote_settings.service_api_key = None

        # Mock unhealthy service
        mock_client_instance = Mock()
        mock_client_instance.health.side_effect = Exception("Service unavailable")
        mock_client.return_value = mock_client_instance

        # Should still return the converter even if not available
        result = get_converter()
        assert result is not None
        assert result.__class__.__name__ == "RemoteDocumentConverter"

    @patch("docling_mcp.tools.converters.factory.settings")
    def test_get_converter_unknown_mode(self, mock_settings: Any) -> None:
        """Test factory raises error for unknown conversion mode."""
        mock_settings.conversion_mode = "invalid_mode"

        with pytest.raises(ValueError, match="Unknown conversion mode"):
            get_converter()
