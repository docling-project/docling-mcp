"""Base types and protocols for document stores."""

from collections.abc import Iterator, KeysView
from typing import Protocol

from pydantic import BaseModel

from docling_core.types.doc.document import DoclingDocument


class DocumentMetadata(BaseModel):
    """Metadata record describing a stored document."""

    document_key: str
    name: str | None = None
    source_filename: str | None = None
    mimetype: str | None = None
    binary_hash: int | None = None
    num_pages: int = 0
    stored_at: str | None = None

    @classmethod
    def from_document(
        cls, document_key: str, document: DoclingDocument, stored_at: str | None
    ) -> "DocumentMetadata":
        """Build a metadata record from a document instance."""
        origin = document.origin
        return cls(
            document_key=document_key,
            name=document.name,
            source_filename=origin.filename if origin is not None else None,
            mimetype=origin.mimetype if origin is not None else None,
            binary_hash=origin.binary_hash if origin is not None else None,
            num_pages=len(document.pages),
            stored_at=stored_at,
        )


class CorpusSearchHit(BaseModel):
    """One ranked match from a search across the stored documents."""

    document_key: str
    anchor: str
    snippet: str
    page: int | None = None


class DocumentStoreProtocol(Protocol):
    """Protocol for document stores.

    Stores behave as mutable mappings from document key to DoclingDocument and
    additionally expose metadata listing and lexical search for corpus-level
    tools.
    """

    def __getitem__(self, document_key: str) -> DoclingDocument:
        """Return the document stored under a key."""
        ...

    def __setitem__(self, document_key: str, document: DoclingDocument) -> None:
        """Store a document under a key."""
        ...

    def __delitem__(self, document_key: str) -> None:
        """Remove a document and its metadata."""
        ...

    def __contains__(self, document_key: object) -> bool:
        """Return whether a document exists for the key."""
        ...

    def __iter__(self) -> Iterator[str]:
        """Iterate over the stored document keys."""
        ...

    def __len__(self) -> int:
        """Return the number of stored documents."""
        ...

    def keys(self) -> KeysView[str]:
        """Return a view of the stored document keys."""
        ...

    def list_metadata(self) -> list[DocumentMetadata]:
        """Return metadata records for all stored documents."""
        ...

    def search_corpus(
        self,
        query: str,
        max_results: int,
        document_keys: list[str] | None = None,
    ) -> list[CorpusSearchHit]:
        """Return ranked lexical matches across the stored documents."""
        ...
