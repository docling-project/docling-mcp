"""This module contains the settings for the Llama Index usages."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings for the Llama Index usages."""

    model_config = SettingsConfigDict(
        env_prefix="DOCLING_MCP_LI_",
        env_file=".env",
        # extra="allow",
    )
    ollama_model: str = "granite3.2:latest"
    embedding_model: str = "BAAI/bge-base-en-v1.5"


settings = Settings()
