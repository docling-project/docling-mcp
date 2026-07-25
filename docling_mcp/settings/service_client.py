"""Settings for Docling service client."""

from enum import Enum

from pydantic_settings import BaseSettings, SettingsConfigDict


class ConversionMode(str, Enum):
    """Conversion operation modes."""

    REMOTE = "remote"  # Use Docling Serve API
    LOCAL = "local"  # Use local DocumentConverter


class ServiceClientSettings(BaseSettings):
    """Settings for Docling service client."""

    model_config = SettingsConfigDict(
        env_prefix="DOCLING_",
        env_file=".env",
    )

    # Service configuration
    service_url: str | None = None
    service_api_key: str | None = None

    # Operation mode
    conversion_mode: ConversionMode = ConversionMode.REMOTE

    # Timeouts and retries
    service_timeout: float = 300.0
    service_max_retries: int = 3

    # Conversion Settings
    keep_images: bool = False
    images_scale: float = 1.0
    do_ocr: bool = True
    do_table_structure: bool = True

    # Fallback behavior
    fallback_to_local: bool = False  # If remote fails, try local (if available)


settings = ServiceClientSettings()
