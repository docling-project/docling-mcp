"""Settings for the Docling MCP server (service client and conversion pipeline)."""

import os
import warnings
from enum import Enum

from pydantic_settings import BaseSettings, SettingsConfigDict


class ConversionMode(str, Enum):
    """Conversion operation modes."""

    REMOTE = "remote"  # Use Docling Serve API
    LOCAL = "local"  # Use local DocumentConverter


class ServiceClientSettings(BaseSettings):
    """Settings for the Docling MCP server.

    All settings are read from environment variables with the ``DOCLING_MCP_``
    prefix (or from a ``.env`` file).  The conversion pipeline options
    (``keep_images``, ``images_scale``, ``do_ocr``, ``do_table_structure``) are
    shared by both the remote and local converters so that users only need to
    set them once.
    """

    model_config = SettingsConfigDict(
        env_prefix="DOCLING_MCP_",
        env_file=".env",
        extra="ignore",
    )

    # Operation mode
    conversion_mode: ConversionMode = ConversionMode.REMOTE

    # Remote service connection
    service_url: str | None = None
    service_api_key: str | None = None
    service_timeout: float = 300.0
    service_max_retries: int = 3

    # Fallback behavior
    fallback_to_local: bool = False  # If remote fails, try local (if available)

    # Conversion pipeline options (shared by both local and remote converters)
    keep_images: bool = False
    images_scale: float = 1.0
    do_ocr: bool = True
    do_table_structure: bool = True
    use_vlm: bool = False
    vlm_host: str = "http://localhost:11434"

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
