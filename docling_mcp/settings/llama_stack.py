"""This module contains the settings for the Llama Stack usages."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings for the Llama Stack usages."""

    model_config = SettingsConfigDict(
        env_prefix="DOCLING_MCP_LLS_",
        env_file=".env",
        # extra="allow",
    )
    url: str = "http://localhost:8321"
    vdb_embedding: str = "all-MiniLM-L6-v2"
    extraction_model: str = "openai/gpt-oss-20b"


settings = Settings()
