"""This module contains the settings for conversion tools."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings for the conversion tools."""

    model_config = SettingsConfigDict(
        env_prefix="DOCLING_MCP_",
        env_file=".env",
        # extra="allow",
    )
    keep_images: bool = False


settings = Settings()
