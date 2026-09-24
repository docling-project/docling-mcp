"""Shared resources for the Docling MCP server.

Attributes:
    mcp: The MCPServer singleton used by all tool and prompt modules.
    local_document_cache: In-memory LRU cache mapping document keys to
        `DoclingDocument` instances, shared across all tools.
    local_stack_cache: In-memory LRU cache mapping document keys to lists of
        `NodeItem` instances, used by Llama Stack tools.
"""

from __future__ import annotations

import threading
from collections import OrderedDict

from mcp.server.mcpserver import MCPServer

from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.items.node import NodeItem

mcp: MCPServer = MCPServer("docling")


class _LRUCaches:
    """Coupled LRU store for `local_document_cache` and `local_stack_cache`.

    Both caches share a single capacity counter and a single eviction order so
    that one document key always maps to entries in both caches or neither.
    The capacity is read once at construction time from
    `settings.cache_max_documents` (`DOCLING_MCP_CACHE_MAX_DOCUMENTS`); it
    cannot be changed after construction.

    Args:
        max_size: Maximum number of documents to retain. Must be >= 1.
    """

    def __init__(self, max_size: int) -> None:
        if max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {max_size}")
        self._max_size = max_size
        self._lock = threading.RLock()
        self._docs: OrderedDict[str, DoclingDocument] = OrderedDict()
        self._stacks: OrderedDict[str, list[NodeItem]] = OrderedDict()
        # Assigned here so every access returns the same object; a @property
        # would allocate a new proxy on each read.
        self.documents: _DocumentProxy = _DocumentProxy(self)
        self.stacks: _StackProxy = _StackProxy(self)

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
        """Insert or refresh *key* atomically in both caches, evicting LRU if needed.

        This is the single authoritative write path. All callers that need to
        store a document together with its stack must use this method so that
        eviction, LRU ordering, and the coupled-cache invariant are maintained
        consistently.

        Args:
            key: The document cache key.
            document: The converted `DoclingDocument`.
            stack: The associated node-item stack.
        """
        with self._lock:
            existing = key in self._docs
            if not existing and len(self._docs) >= self._max_size:
                self._evict_lru()

            self._docs[key] = document
            self._stacks[key] = stack
            if existing:
                self._touch(key)

    def drop(self, key: str) -> bool:
        """Remove *key* from both caches.

        Args:
            key: The document cache key to remove.

        Returns:
            True if the key was present and removed, False otherwise.
        """
        with self._lock:
            removed = key in self._docs
            self._docs.pop(key, None)
            self._stacks.pop(key, None)
            return removed

    def __len__(self) -> int:
        with self._lock:
            return len(self._docs)


class _DocumentProxy:
    """Read-only dict-compatible view for the document side of _LRUCaches.

    Supports the subset of the dict protocol used by the tool modules:
    `__contains__`, `__getitem__`, and `keys()`. Write access is
    intentionally not supported; callers must use `_LRUCaches.put` directly to
    ensure the coupled-cache invariant is maintained.

    Args:
        cache: The parent _LRUCaches instance.
    """

    def __init__(self, cache: _LRUCaches) -> None:
        self._cache = cache

    def __contains__(self, key: object) -> bool:
        with self._cache._lock:
            return key in self._cache._docs

    def __getitem__(self, key: str) -> DoclingDocument:
        with self._cache._lock:
            doc = self._cache._docs[key]
            self._cache._touch(key)
            return doc

    def __len__(self) -> int:
        with self._cache._lock:
            return len(self._cache._docs)

    def keys(self) -> list[str]:
        """Return a snapshot of the cached document keys."""
        with self._cache._lock:
            return list(self._cache._docs)


class _StackProxy:
    """Read-only dict-compatible view for the stack side of _LRUCaches.

    Supports `__contains__`, `__getitem__`, and `keys()`. Replacing the list
    stored under a key (`cache[key] = new_list`) is intentionally not
    supported; callers must use `_LRUCaches.put` to ensure the coupled-cache
    invariant is maintained. In-place mutations of the returned list — such as
    `cache[key].append(item)`, `cache[key].pop()`, or `cache[key][-1] = item`
    — are permitted and are the intended pattern used by the document-generation
    tools.

    Args:
        cache: The parent _LRUCaches instance.
    """

    def __init__(self, cache: _LRUCaches) -> None:
        self._cache = cache

    def __contains__(self, key: object) -> bool:
        with self._cache._lock:
            return key in self._cache._stacks

    def __getitem__(self, key: str) -> list[NodeItem]:
        with self._cache._lock:
            stack = self._cache._stacks[key]
            self._cache._touch(key)
            return stack

    def __len__(self) -> int:
        with self._cache._lock:
            return len(self._cache._stacks)

    def keys(self) -> list[str]:
        """Return a snapshot of the cached stack keys."""
        with self._cache._lock:
            return list(self._cache._stacks)


def _build_caches() -> _LRUCaches:
    """Create the module-level LRU caches using the configured max size.

    Returns:
        A new _LRUCaches instance sized from `settings.cache_max_documents`.
        The capacity is fixed for the lifetime of the process; changing the
        environment variable after import has no effect.
    """
    from docling_mcp.settings.service_client import settings

    return _LRUCaches(max_size=settings.cache_max_documents)


_caches: _LRUCaches = _build_caches()

local_document_cache: _DocumentProxy = _caches.documents
local_stack_cache: _StackProxy = _caches.stacks


def put_document(
    key: str,
    document: DoclingDocument,
    stack: list[NodeItem],
) -> None:
    """Insert or refresh a document and its stack in the in-memory cache.

    Args:
        key: The document cache key.
        document: The converted `DoclingDocument`.
        stack: The associated node-item stack.
    """
    _caches.put(key, document, stack)


def drop_document(key: str) -> bool:
    """Remove a document and its associated stack from the in-memory cache.

    Args:
        key: The document cache key to remove.

    Returns:
        True if the key was present and removed, False otherwise.
    """
    return _caches.drop(key)
