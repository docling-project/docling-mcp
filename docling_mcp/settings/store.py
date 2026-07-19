"""Settings for the document store."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class StoreSettings(BaseSettings):
    """Settings for the document store."""

    model_config = SettingsConfigDict(
        env_prefix="DOCLING_MCP_",
        env_file=".env",
        extra="ignore",
    )

    # Base cache directory; persisted documents live in a "documents"
    # subdirectory beneath it. Defaults to the application cache dir.
    cache_dir: Path | None = None

    # Persist converted documents to disk so they survive server restarts
    cache_persist: bool = True


settings = StoreSettings()
