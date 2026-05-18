"""This module contains the settings for the Llama Index usages."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings for the Llama Index usages."""

    model_config = SettingsConfigDict(
        env_prefix="DOCLING_MCP_LI_",
        env_file=".env",
        # extra="allow",
    )
    api_base: str = "http://127.0.0.1:1234/v1"
    api_key: str = "none"
    model_id: str = "ibm/granite-3.2-8b"
    embedding_model: str = "BAAI/bge-base-en-v1.5"


settings = Settings()
