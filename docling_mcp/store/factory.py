"""Factory for creating the document store based on configuration."""

from pathlib import Path

from docling_mcp.docling_cache import get_cache_dir
from docling_mcp.logger import setup_logger
from docling_mcp.settings.store import settings
from docling_mcp.store.base import DocumentStoreProtocol
from docling_mcp.store.local import InMemoryDocumentStore, LocalDocumentStore

logger = setup_logger()


def _fallback_cache_dir() -> Path:
    return Path.home() / ".cache" / "docling-mcp"


def create_document_store() -> DocumentStoreProtocol:
    """Create the document store selected by the store settings.

    Falls back to a per-user cache directory when the default location is not
    writable (for example a read-only site-packages install), and to the
    in-memory store when no writable location exists at all.
    """
    if not settings.cache_persist:
        logger.info("Document persistence disabled; using in-memory store")
        return InMemoryDocumentStore()

    if settings.cache_dir is not None:
        candidates = [settings.cache_dir]
    else:
        candidates = []
        try:
            candidates.append(get_cache_dir())
        except OSError:
            logger.warning("Default cache directory is not available")
        candidates.append(_fallback_cache_dir())

    for base_dir in candidates:
        cache_dir = base_dir / "documents"
        try:
            store = LocalDocumentStore(cache_dir=cache_dir)
        except OSError:
            logger.warning(f"Document store location not writable: {cache_dir}")
            continue
        logger.info(f"Using local document store at: {cache_dir}")
        return store

    logger.error("No writable document store location; falling back to in-memory store")
    return InMemoryDocumentStore()
