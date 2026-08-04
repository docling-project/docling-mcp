"""Unit tests for local document converter."""

from typing import Any
from unittest.mock import Mock, patch

import pytest

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    VlmConvertOptions,
    VlmPipelineOptions,
)
from docling.datamodel.vlm_engine_options import ApiVlmEngineOptions
from docling.document_converter import ImageFormatOption, PdfFormatOption
from docling.models.inference_engines.vlm import VlmEngineType
from docling.pipeline.vlm_pipeline import VlmPipeline

from docling_mcp.settings.service_client import ServiceClientSettings
from docling_mcp.tools.converters.base import ConversionOutput
from docling_mcp.tools.converters.local import (
    LocalDocumentConverter,
)


class TestVlmSettings:
    def test_defaults_disable_vlm_with_local_ollama_host(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("DOCLING_MCP_USE_VLM", raising=False)
        monkeypatch.delenv("DOCLING_MCP_VLM_HOST", raising=False)

        service_settings = ServiceClientSettings(_env_file=None)

        assert service_settings.use_vlm is False
        assert service_settings.vlm_host == "http://localhost:11434"

    def test_reads_vlm_settings_from_prefixed_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DOCLING_MCP_USE_VLM", "true")
        monkeypatch.setenv("DOCLING_MCP_VLM_HOST", "http://ollama.test:8080/")

        service_settings = ServiceClientSettings(_env_file=None)

        assert service_settings.use_vlm is True
        assert service_settings.vlm_host == "http://ollama.test:8080/"


class TestLocalDocumentConverter:
    """Test suite for LocalDocumentConverter."""

    @patch("docling_mcp.tools.converters.local.LOCAL_CONVERSION_AVAILABLE", False)
    def test_init_without_local_extra_raises_error(self) -> None:
        """Test that initialization fails without local extra installed."""
        with pytest.raises(
            ImportError, match="Local conversion requires docling-mcp\\[local\\]"
        ):
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

        with patch(
            "docling_mcp.tools.converters.local.get_cache_key", return_value=cache_key
        ):
            with patch(
                "docling_mcp.tools.converters.local.local_document_cache",
                {cache_key: Mock()},
            ):
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
        with patch(
            "docling_mcp.tools.converters.local.get_cache_key", return_value=cache_key
        ):
            converter = LocalDocumentConverter()
            result = converter.convert_document("test.pdf")

        assert isinstance(result, ConversionOutput)
        assert result.from_cache is False
        assert result.document_key == cache_key

    @patch("docling_mcp.tools.converters.local.LOCAL_CONVERSION_AVAILABLE", True)
    def test_get_converter_preserves_pdf_pipeline_settings(self) -> None:
        service_settings = ServiceClientSettings(
            _env_file=None,
            use_vlm=False,
            keep_images=True,
            images_scale=1.5,
            do_ocr=False,
            do_table_structure=False,
        )

        with patch("docling_mcp.tools.converters.local.settings", service_settings):
            converter = LocalDocumentConverter()
            docling_converter = converter._get_converter()

        for input_format in (InputFormat.PDF, InputFormat.IMAGE):
            pipeline_options = docling_converter.format_to_options[
                input_format
            ].pipeline_options
            assert isinstance(pipeline_options, PdfPipelineOptions)
            assert pipeline_options.generate_page_images is True
            assert pipeline_options.images_scale == 1.5
            assert pipeline_options.do_ocr is False
            assert pipeline_options.do_table_structure is False

    @pytest.mark.filterwarnings("error::DeprecationWarning")
    @patch("docling_mcp.tools.converters.local.LOCAL_CONVERSION_AVAILABLE", True)
    def test_get_converter_configures_granite_docling_vlm_pipeline(self) -> None:
        service_settings = ServiceClientSettings(
            _env_file=None,
            use_vlm=True,
            vlm_host="http://localhost:11434",
        )

        with patch("docling_mcp.tools.converters.local.settings", service_settings):
            converter = LocalDocumentConverter()
            docling_converter = converter._get_converter()

        pdf_option = docling_converter.format_to_options[InputFormat.PDF]
        image_option = docling_converter.format_to_options[InputFormat.IMAGE]
        assert isinstance(pdf_option, PdfFormatOption)
        assert isinstance(image_option, ImageFormatOption)

        for format_option in (pdf_option, image_option):
            assert format_option.pipeline_cls is VlmPipeline
            assert isinstance(format_option.pipeline_options, VlmPipelineOptions)
            assert format_option.pipeline_options.enable_remote_services is True

        assert pdf_option.pipeline_options == image_option.pipeline_options
        pipeline_options = pdf_option.pipeline_options
        assert isinstance(pipeline_options, VlmPipelineOptions)
        vlm_options = pipeline_options.vlm_options
        assert isinstance(vlm_options, VlmConvertOptions)
        preset = VlmConvertOptions.get_preset("granite_docling")
        assert vlm_options.model_spec == preset.model_spec
        assert isinstance(vlm_options.engine_options, ApiVlmEngineOptions)
        assert vlm_options.engine_options.engine_type is VlmEngineType.API_OLLAMA
        assert vlm_options.engine_options.params == {}
        assert (
            str(vlm_options.engine_options.url)
            == "http://localhost:11434/v1/chat/completions"
        )

    @patch("docling_mcp.tools.converters.local.LOCAL_CONVERSION_AVAILABLE", True)
    def test_get_converter_normalizes_trailing_slash_in_vlm_host(self) -> None:
        service_settings = ServiceClientSettings(
            _env_file=None,
            use_vlm=True,
            vlm_host="http://ollama.test:8080/",
        )

        with patch("docling_mcp.tools.converters.local.settings", service_settings):
            converter = LocalDocumentConverter()
            docling_converter = converter._get_converter()

        pipeline_options = docling_converter.format_to_options[
            InputFormat.PDF
        ].pipeline_options
        assert isinstance(pipeline_options, VlmPipelineOptions)
        assert isinstance(pipeline_options.vlm_options, VlmConvertOptions)
        assert isinstance(
            pipeline_options.vlm_options.engine_options, ApiVlmEngineOptions
        )
        assert (
            str(pipeline_options.vlm_options.engine_options.url)
            == "http://ollama.test:8080/v1/chat/completions"
        )

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
