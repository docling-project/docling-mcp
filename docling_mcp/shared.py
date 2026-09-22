"""Shared resources for the Docling MCP server.

Attributes:
    mcp: The MCPServer singleton used by all tool and prompt modules.
    local_document_cache: In-memory LRU cache mapping document keys to
        `DoclingDocument` instances, shared across all tools.
    local_stack_cache: In-memory LRU cache mapping document keys to lists of
        `NodeItem` instances, used by Llama Stack tools.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator

from mcp.server.mcpserver import MCPServer

from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.items.node import NodeItem

mcp: MCPServer = MCPServer("docling")


class _LRUCaches:
    """Coupled LRU store for ``local_document_cache`` and ``local_stack_cache``.

    Both caches share a single capacity counter and a single eviction order so
    that one document key always maps to entries in both caches or neither.
    The capacity is read once at construction time from
    ``settings.cache_max_documents`` (``DOCLING_MCP_CACHE_MAX_DOCUMENTS``).

    Args:
        max_size: Maximum number of documents to retain. Must be >= 1.
    """

    def __init__(self, max_size: int) -> None:
        self._max_size = max_size
        self._docs: OrderedDict[str, DoclingDocument] = OrderedDict()
        self._stacks: OrderedDict[str, list[NodeItem]] = OrderedDict()

    def _evict_lru(self) -> None:
        """Remove the least-recently-used entry from both caches."""
        if self._docs:
            key, _ = self._docs.popitem(last=False)
            self._stacks.pop(key, None)

    def _touch(self, key: str) -> None:
        """Mark *key* as most-recently used."""
        if key in self._docs:
            self._docs.move_to_end(key)
        if key in self._stacks:
            self._stacks.move_to_end(key)

    def put(
        self,
        key: str,
        document: DoclingDocument,
        stack: list[NodeItem],
    ) -> None:
        """Insert or refresh *key* in both caches, evicting LRU if needed.

        Args:
            key: The document cache key.
            document: The converted `DoclingDocument`.
            stack: The associated node-item stack.
        """
        if key in self._docs:
            # Refresh existing entry in-place.
            self._docs[key] = document
            self._stacks[key] = stack
            self._touch(key)
            return

        if len(self._docs) >= self._max_size:
            self._evict_lru()

        self._docs[key] = document
        self._stacks[key] = stack

    def drop(self, key: str) -> bool:
        """Remove *key* from both caches.

        Args:
            key: The document cache key to remove.

        Returns:
            ``True`` if the key was present and removed, ``False`` otherwise.
        """
        removed = key in self._docs
        self._docs.pop(key, None)
        self._stacks.pop(key, None)
        return removed

    def __len__(self) -> int:
        return len(self._docs)

    @property
    def documents(self) -> _DocumentProxy:
        """Dict-like view over the document sub-cache."""
        return _DocumentProxy(self)

    @property
    def stacks(self) -> _StackProxy:
        """Dict-like view over the stack sub-cache."""
        return _StackProxy(self)


class _DocumentProxy:
    """Thin dict-compatible proxy for the document side of _LRUCaches.

    Supports the subset of dict used by the tool modules: __contains__,
    __getitem__, __setitem__, and keys().

    Note:
        __setitem__ on this proxy inserts a document with an empty stack.
        Callers that need to write both should use _LRUCaches.put directly,
        but tool code that only sets the document side still works correctly.

    Args:
        cache: The parent _LRUCaches instance.
    """

    def __init__(self, cache: _LRUCaches) -> None:
        self._cache = cache

    def __contains__(self, key: object) -> bool:
        return key in self._cache._docs

    def __getitem__(self, key: str) -> DoclingDocument:
        doc = self._cache._docs[key]
        self._cache._touch(key)
        return doc

    def __setitem__(self, key: str, value: DoclingDocument) -> None:
        existing_stack = self._cache._stacks.get(key, [])
        self._cache.put(key, value, existing_stack)

    def keys(self) -> Iterator[str]:
        """Return an iterator over the cached document keys."""
        return iter(self._cache._docs)


class _StackProxy:
    """Thin dict-compatible proxy for the stack side of _LRUCaches.

    Supports __contains__, __getitem__, __setitem__, and keys().

    Note:
        __setitem__ on this proxy updates the stack for a key that must already
        exist in the document cache; it does not create new entries.

    Args:
        cache: The parent _LRUCaches instance.
    """

    def __init__(self, cache: _LRUCaches) -> None:
        self._cache = cache

    def __contains__(self, key: object) -> bool:
        return key in self._cache._stacks

    def __getitem__(self, key: str) -> list[NodeItem]:
        stack = self._cache._stacks[key]
        self._cache._touch(key)
        return stack

    def __setitem__(self, key: str, value: list[NodeItem]) -> None:
        self._cache._stacks[key] = value
        if key in self._cache._docs:
            self._cache._touch(key)

    def keys(self) -> Iterator[str]:
        """Return an iterator over the cached stack keys."""
        return iter(self._cache._stacks)


def _build_caches() -> _LRUCaches:
    """Create the module-level LRU caches using the configured max size."""
    from docling_mcp.settings.service_client import settings

    return _LRUCaches(max_size=settings.cache_max_documents)


_caches: _LRUCaches = _build_caches()

local_document_cache: _DocumentProxy = _caches.documents
local_stack_cache: _StackProxy = _caches.stacks
