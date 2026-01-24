"""Settings for local conversion tools."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class ConversionSettings(BaseSettings):
    """Settings for local conversion tools."""

    model_config = SettingsConfigDict(
        env_prefix="DOCLING_MCP_",
        env_file=".env",
        extra="ignore",  # Ignore extra env vars like DOCLING_SERVICE_URL
    )

    keep_images: bool = False
    images_scale: float = 1.0
    do_ocr: bool = True
    do_table_structure: bool = True


settings = ConversionSettings()
