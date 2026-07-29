"""Settings for the Docling MCP server (service client and conversion pipeline)."""

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


settings = ServiceClientSettings()
