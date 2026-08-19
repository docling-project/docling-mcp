"""This module contains the settings for the Llama Index usages."""

from typing import Annotated

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings for Llama Index tool integrations.

    All settings are read from environment variables with the
    `DOCLING_MCP_LI_` prefix (or from a `.env` file).
    """

    model_config = SettingsConfigDict(
        env_prefix="DOCLING_MCP_LI_",
        env_file=".env",
        # extra="allow",
    )
    api_base: Annotated[
        str,
        Field(description="Base URL of the OpenAI-compatible inference endpoint."),
    ] = "http://127.0.0.1:1234/v1"
    api_key: Annotated[
        str, Field(description="API key for the inference endpoint.")
    ] = "none"
    model_id: Annotated[
        str, Field(description="Model identifier for the LLM used in RAG queries.")
    ] = "ibm/granite-3.2-8b"
    embedding_model: Annotated[
        str,
        Field(description="HuggingFace embedding model name for vector indexing."),
    ] = "BAAI/bge-base-en-v1.5"


settings = Settings()
