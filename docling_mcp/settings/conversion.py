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
    # Add local-specific settings
    do_ocr: bool = True
    do_table_structure: bool = True

    # Use the VLM pipeline (e.g. granite-docling via a local Ollama instance)
    # instead of the standard OCR/layout pipeline.
    use_vlm: bool = False
    vlm_host: str = "http://localhost:11434"


settings = ConversionSettings()
