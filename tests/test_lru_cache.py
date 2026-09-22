"""Tests for the LRU document cache (shared._LRUCaches) and the drop tool."""

from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.items.node import NodeItem

from docling_mcp.shared import _LRUCaches
from docling_mcp.tools.conversion import (
    DropDocumentFromCacheOutput,
    drop_document_from_local_cache,
)
from docling_mcp.tools.generation import create_new_docling_document


def _make_doc(name: str = "doc") -> DoclingDocument:
    """Create a minimal DoclingDocument for testing."""
    return DoclingDocument(name=name)


class TestLRUCaches:
    """Unit tests for the _LRUCaches eviction logic."""

    def test_insert_and_retrieve(self) -> None:
        cache = _LRUCaches(max_size=3)
        doc = _make_doc()
        cache.put("k1", doc, [])
        assert cache.documents["k1"] is doc
        assert len(cache) == 1

    def test_evicts_lru_when_full(self) -> None:
        cache = _LRUCaches(max_size=2)
        doc_a, doc_b, doc_c = _make_doc("a"), _make_doc("b"), _make_doc("c")

        cache.put("a", doc_a, [])
        cache.put("b", doc_b, [])
        # Cache is full; inserting "c" must evict "a" (oldest).
        cache.put("c", doc_c, [])

        assert len(cache) == 2
        assert "a" not in cache.documents
        assert "b" in cache.documents
        assert "c" in cache.documents

    def test_access_refreshes_lru_order(self) -> None:
        cache = _LRUCaches(max_size=2)
        doc_a, doc_b, doc_c = _make_doc("a"), _make_doc("b"), _make_doc("c")

        cache.put("a", doc_a, [])
        cache.put("b", doc_b, [])
        # Touch "a" so "b" becomes the LRU.
        _ = cache.documents["a"]
        cache.put("c", doc_c, [])

        assert "b" not in cache.documents, "b should have been evicted (LRU)"
        assert "a" in cache.documents
        assert "c" in cache.documents

    def test_refresh_existing_does_not_grow(self) -> None:
        cache = _LRUCaches(max_size=2)
        doc = _make_doc()
        cache.put("k", doc, [])
        cache.put("k", _make_doc(), [])  # update, not insert
        assert len(cache) == 1

    def test_drop_removes_from_both_sub_caches(self) -> None:
        cache = _LRUCaches(max_size=3)
        cache.put("k", _make_doc(), [])

        removed = cache.drop("k")

        assert removed is True
        assert "k" not in cache.documents
        assert "k" not in cache.stacks

    def test_drop_missing_key_returns_false(self) -> None:
        cache = _LRUCaches(max_size=3)
        assert cache.drop("nonexistent") is False

    def test_stack_proxy_setitem_updates_in_place(self) -> None:
        cache = _LRUCaches(max_size=3)
        cache.put("k", _make_doc(), [])
        sentinel: list[NodeItem] = []
        cache.stacks["k"] = sentinel
        assert cache.stacks["k"] is sentinel

    def test_eviction_also_clears_stack(self) -> None:
        cache = _LRUCaches(max_size=1)
        cache.put("a", _make_doc(), [])
        cache.put("b", _make_doc(), [])  # evicts "a"

        assert "a" not in cache.stacks


class TestDropDocumentTool:
    """Integration tests for the drop_document_from_local_cache MCP tool."""

    def test_drop_existing_document(self) -> None:
        result = create_new_docling_document(prompt="drop-me")
        key = result.document_key

        out = drop_document_from_local_cache(document_key=key)

        assert isinstance(out, DropDocumentFromCacheOutput)
        assert out.dropped is True

    def test_drop_nonexistent_document(self) -> None:
        out = drop_document_from_local_cache(document_key="does-not-exist")

        assert isinstance(out, DropDocumentFromCacheOutput)
        assert out.dropped is False
