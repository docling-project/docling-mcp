"""Settings for the Docling MCP server (service client and conversion pipeline)."""

import os
import warnings
from enum import Enum
from typing import Annotated

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from docling_core.types.doc.base import ImageRefMode


class ConversionMode(str, Enum):
    """Conversion operation modes."""

    REMOTE = "remote"
    """Use the Docling Serve REST API for document conversion."""

    LOCAL = "local"
    """Use a local DocumentConverter instance (requires the `local` extra)."""


class ServiceClientSettings(BaseSettings):
    """Settings for the Docling MCP server.

    All settings are read from environment variables with the `DOCLING_MCP_`
    prefix (or from a `.env` file).  The conversion pipeline options
    (`keep_images`, `images_scale`, `do_ocr`, `do_table_structure`) are
    shared by both the remote and local converters so that users only need to
    set them once.
    """

    model_config = SettingsConfigDict(
        env_prefix="DOCLING_MCP_",
        env_file=".env",
        extra="ignore",
    )

    # Operation mode
    conversion_mode: Annotated[
        ConversionMode,
        Field(
            description=(
                "Conversion backend to use. `remote` delegates to a Docling Serve "
                "API endpoint; `local` runs the DocumentConverter in-process "
                "(requires the `local` extra)."
            )
        ),
    ] = ConversionMode.REMOTE

    # Remote service connection
    service_url: Annotated[
        str | None,
        Field(
            description=(
                "Base URL of the Docling Serve instance. "
                "Required when `conversion_mode` is `remote`."
            )
        ),
    ] = None

    service_api_key: Annotated[
        str | None,
        Field(description="API key for authenticating with Docling Serve."),
    ] = None

    service_timeout: Annotated[
        float,
        Field(
            description="Request timeout in seconds for the remote Docling Serve API."
        ),
    ] = 300.0

    service_max_retries: Annotated[
        int,
        Field(description="Maximum number of retry attempts for remote API requests."),
    ] = 3

    # Fallback behavior
    fallback_to_local: Annotated[
        bool,
        Field(
            description=(
                "If `true`, fall back to local conversion when the remote service "
                "is unreachable (requires the `local` extra)."
            )
        ),
    ] = False

    # Conversion pipeline options (shared by both local and remote converters)
    keep_images: Annotated[
        bool,
        Field(
            description=(
                "Retain page images in the converted document. "
                "Required when using `page_thumbnail` or `image_export_mode=embedded`."
            )
        ),
    ] = False

    images_scale: Annotated[
        float,
        Field(
            description=(
                "Scale factor applied to page images during conversion. "
                "Increase to avoid tensor padding errors."
            )
        ),
    ] = 1.0

    do_ocr: Annotated[
        bool,
        Field(description="Run the OCR pipeline on converted documents."),
    ] = True

    do_table_structure: Annotated[
        bool,
        Field(
            description="Detect and reconstruct table structure in converted documents."
        ),
    ] = True

    # Markdown export options
    image_export_mode: Annotated[
        ImageRefMode,
        Field(
            description=(
                "Controls how images are rendered when exporting a document to "
                "Markdown. Accepted values mirror docling-core's `ImageRefMode`: "
                "`placeholder` (default, emits `<!-- image -->`), `embedded` "
                "(base64 data-URI), or `referenced` (file path / URL). "
                "Set via the `DOCLING_MCP_IMAGE_EXPORT_MODE` environment variable."
            )
        ),
    ] = ImageRefMode.PLACEHOLDER

    def model_post_init(self, __context: object) -> None:
        """Warn when deprecated (pre-refactor) environment variable names are set."""
        _RENAMED: dict[str, str] = {
            "DOCLING_SERVICE_URL": "DOCLING_MCP_SERVICE_URL",
            "DOCLING_SERVICE_API_KEY": "DOCLING_MCP_SERVICE_API_KEY",
            "DOCLING_CONVERSION_MODE": "DOCLING_MCP_CONVERSION_MODE",
            "DOCLING_SERVICE_TIMEOUT": "DOCLING_MCP_SERVICE_TIMEOUT",
            "DOCLING_SERVICE_MAX_RETRIES": "DOCLING_MCP_SERVICE_MAX_RETRIES",
            "DOCLING_FALLBACK_TO_LOCAL": "DOCLING_MCP_FALLBACK_TO_LOCAL",
            "DOCLING_KEEP_IMAGES": "DOCLING_MCP_KEEP_IMAGES",
            "DOCLING_IMAGES_SCALE": "DOCLING_MCP_IMAGES_SCALE",
            "DOCLING_DO_OCR": "DOCLING_MCP_DO_OCR",
            "DOCLING_DO_TABLE_STRUCTURE": "DOCLING_MCP_DO_TABLE_STRUCTURE",
            "DOCLING_MCP_LI_OLLAMA_MODEL": "DOCLING_MCP_LI_MODEL_ID",
        }
        for old, new in _RENAMED.items():
            if os.environ.get(old):
                warnings.warn(
                    f"Environment variable {old!r} is ignored. Use {new!r} instead.",
                    UserWarning,
                    stacklevel=2,
                )


settings = ServiceClientSettings()
