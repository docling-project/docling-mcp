"""Document store package providing persistent caching of Docling documents."""

from docling_mcp.store.base import (
    CorpusSearchHit,
    DocumentMetadata,
    DocumentStoreProtocol,
)
from docling_mcp.store.factory import create_document_store
from docling_mcp.store.local import InMemoryDocumentStore, LocalDocumentStore
