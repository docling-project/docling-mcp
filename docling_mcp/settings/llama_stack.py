"""This module contains the settings for the Llama Stack usages."""

from typing import Annotated

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings for Llama Stack tool integrations.

    All settings are read from environment variables with the
    `DOCLING_MCP_LLS_` prefix (or from a `.env` file).
    """

    model_config = SettingsConfigDict(
        env_prefix="DOCLING_MCP_LLS_",
        env_file=".env",
        # extra="allow",
    )
    url: Annotated[str, Field(description="Base URL of the Llama Stack server.")] = (
        "http://localhost:8321"
    )
    vdb_embedding: Annotated[
        str,
        Field(description="Embedding model name used for vector-database ingestion."),
    ] = "all-MiniLM-L6-v2"
    extraction_model: Annotated[
        str,
        Field(
            description="Model identifier used for structured information extraction."
        ),
    ] = "openai/gpt-oss-20b"


settings = Settings()
