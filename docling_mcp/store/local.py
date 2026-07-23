"""Local document store implementations."""

import json
import os
import re
import threading
import uuid
from collections.abc import Iterator, MutableMapping
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeGuard

from docling_core.types.doc.document import DoclingDocument

from docling_mcp.logger import setup_logger
from docling_mcp.store.base import CorpusSearchHit, DocumentMetadata
from docling_mcp.store.index import CorpusIndex

logger = setup_logger()

_KEY_PATTERN = re.compile(r"[0-9a-f]{32}")
_META_SUFFIX = ".meta.json"
_INDEX_FILENAME = "corpus-index.sqlite3"


def is_valid_document_key(document_key: object) -> TypeGuard[str]:
    """Return whether a value is a well-formed document key."""
    return (
        isinstance(document_key, str)
        and _KEY_PATTERN.fullmatch(document_key) is not None
    )


class InMemoryDocumentStore(MutableMapping[str, DoclingDocument]):
    """Document store keeping everything in process memory.

    Preserves the pre-persistence behavior of the plain dict cache, with the
    same key-format contract as the persistent store. Built on MutableMapping
    so bulk operations such as update() and setdefault() also go through the
    validating __setitem__.
    """

    def __init__(self) -> None:
        self._documents: dict[str, DoclingDocument] = {}
        self._index = CorpusIndex(db_path=None)

    def __setitem__(self, document_key: str, document: DoclingDocument) -> None:
        """Store a document under a well-formed key."""
        if not is_valid_document_key(document_key):
            raise ValueError(f"Invalid document key: {document_key!r}")
        self._documents[document_key] = document
        # Only conversion artifacts are searchable; replacing one with an
        # in-progress authored document also drops its stale index rows.
        if document.origin is not None:
            self._index.index_document(document_key, document)
        else:
            self._index.remove_document(document_key)

    def __getitem__(self, document_key: str) -> DoclingDocument:
        """Return the document stored under a key."""
        return self._documents[document_key]

    def __delitem__(self, document_key: str) -> None:
        """Remove a document."""
        del self._documents[document_key]
        self._index.remove_document(document_key)

    def __iter__(self) -> Iterator[str]:
        """Iterate over the stored document keys."""
        return iter(self._documents)

    def __len__(self) -> int:
        """Return the number of stored documents."""
        return len(self._documents)

    def list_metadata(self) -> list[DocumentMetadata]:
        """Return metadata records synthesized from the in-memory documents."""
        return [
            DocumentMetadata.from_document(key, doc, stored_at=None)
            for key, doc in list(self._documents.items())
        ]

    def search_corpus(
        self,
        query: str,
        max_results: int,
        document_keys: list[str] | None = None,
    ) -> list[CorpusSearchHit]:
        """Return ranked lexical matches across the stored documents."""
        return self._index.search(query, max_results, document_keys)


class LocalDocumentStore(MutableMapping[str, DoclingDocument]):
    """Document store persisting converted documents to a local directory.

    Documents that carry a conversion origin are written through to disk as
    DoclingDocument JSON with a metadata sidecar, so they survive server
    restarts. Documents without an origin (in-progress authored documents)
    stay memory-only.

    Persisted documents are conversion artifacts: in-place edits made by
    tools live in session memory only and are not written back under the
    conversion key, so a cache hit always returns document state derived
    from the converted source. Use the save/export tools to make edited
    state durable. A lexical search index derived from the conversion
    artifacts lives beside the document files and is reconciled with them
    on every search, so it can always be rebuilt from disk.

    Documents are never evicted from memory during a session, so object
    identity is stable for as long as the process lives. Durability is best
    effort: a failed write leaves any previous disk copy in place (same
    content hash, so the same conversion) and is logged. Explicit deletion
    is strict and raises when disk state cannot be removed. Concurrent
    multi-process access is last-writer-wins; in-process access is guarded
    by a lock. Keys are validated against the 32-hex key format before any
    path is derived from them. The cache directory is created owner-only
    and data files are written with owner-only permissions.
    """

    def __init__(self, cache_dir: Path) -> None:
        self._dir = cache_dir
        self._dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        # mkdir(exist_ok=True) does not tighten a pre-existing directory
        os.chmod(self._dir, 0o700)
        # Probe writability now: mkdir(exist_ok=True) accepts an existing
        # read-only directory, and the factory needs the failure at
        # construction time to fall back to another location. Any OSError,
        # including a failing unlink, propagates as "not writable".
        probe = self._dir / f".probe-{os.getpid()}-{uuid.uuid4().hex}"
        try:
            probe.touch()
        finally:
            probe.unlink(missing_ok=True)
        self._memory: dict[str, DoclingDocument] = {}
        self._lock = threading.RLock()
        self._index = CorpusIndex(db_path=self._dir / _INDEX_FILENAME)

    def _doc_path(self, document_key: str) -> Path:
        return self._dir / f"{document_key}.json"

    def _meta_path(self, document_key: str) -> Path:
        return self._dir / f"{document_key}{_META_SUFFIX}"

    def _persist(self, document_key: str, document: DoclingDocument) -> None:
        # Unique temp names keep concurrent same-key writers from clobbering
        # each other's in-flight files; os.replace keeps readers consistent
        # per file. The document/sidecar pair is not atomic as a pair: the
        # sidecar is derived, advisory data and listings tolerate a missing
        # or stale one.
        token = f"{os.getpid()}-{uuid.uuid4().hex}"
        doc_tmp = self._dir / f"{document_key}.{token}.doc.tmp"
        meta_tmp = self._dir / f"{document_key}.{token}.meta.tmp"

        try:
            document.save_as_json(filename=doc_tmp)
            os.chmod(doc_tmp, 0o600)
            os.replace(doc_tmp, self._doc_path(document_key))

            metadata = DocumentMetadata.from_document(
                document_key,
                document,
                stored_at=datetime.now(timezone.utc).isoformat(),
            )
            meta_tmp.write_text(metadata.model_dump_json(), encoding="utf-8")
            os.chmod(meta_tmp, 0o600)
            os.replace(meta_tmp, self._meta_path(document_key))
        finally:
            for tmp in (doc_tmp, meta_tmp):
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    logger.warning(f"Could not remove temp file: {tmp}")

    def _try_persist(self, document_key: str, document: DoclingDocument) -> None:
        """Persist a document; on failure keep the memory copy authoritative.

        Nothing is deleted on failure: under content-hash keys, any
        pre-existing disk copy for the same key was produced from the same
        source bytes and configuration, so it stays valid for restart-time
        discovery. Losing durability is logged rather than raised.
        Programming errors are not swallowed.
        """
        try:
            self._persist(document_key, document)
        except (OSError, ValueError, TypeError):
            logger.exception(f"Failed to persist document: {document_key}")

    def _drop_persisted(self, document_key: str) -> None:
        """Remove disk state for a key, raising on the first failure."""
        for path in (self._doc_path(document_key), self._meta_path(document_key)):
            path.unlink(missing_ok=True)

    def __setitem__(self, document_key: str, document: DoclingDocument) -> None:
        """Store a document, persisting it when it has a conversion origin."""
        if not is_valid_document_key(document_key):
            raise ValueError(f"Invalid document key: {document_key!r}")

        with self._lock:
            if document.origin is not None:
                self._memory[document_key] = document
                self._try_persist(document_key, document)
                self._index.index_document(document_key, document)
            else:
                # Replacing a persisted document with a memory-only one must
                # not let the old disk copy resurface after a restart; if the
                # disk copy cannot be removed, fail the operation instead of
                # succeeding in memory only.
                self._drop_persisted(document_key)
                self._memory[document_key] = document
                self._index.remove_document(document_key)

    def __getitem__(self, document_key: str) -> DoclingDocument:
        """Return a document from memory or load it from disk.

        Unreadable or corrupt disk entries behave like absent keys.
        """
        if not is_valid_document_key(document_key):
            raise KeyError(document_key)

        with self._lock:
            if document_key in self._memory:
                return self._memory[document_key]

            doc_path = self._doc_path(document_key)
            if doc_path.exists():
                try:
                    document = DoclingDocument.load_from_json(filename=doc_path)
                except (OSError, ValueError) as exc:
                    logger.warning(
                        f"Unreadable document file for {document_key}: {exc}"
                    )
                    # Quarantine the corrupt entry so membership checks stop
                    # reporting it and the source can be converted again.
                    quarantine = self._dir / (
                        f"{document_key}.corrupt-{os.getpid()}-{uuid.uuid4().hex}"
                    )
                    try:
                        os.replace(doc_path, quarantine)
                        logger.warning(f"Quarantined corrupt entry to: {quarantine}")
                    except OSError:
                        logger.warning(f"Could not quarantine: {doc_path}")
                    raise KeyError(document_key) from exc
                self._memory[document_key] = document
                return document

        raise KeyError(document_key)

    def __delitem__(self, document_key: str) -> None:
        """Remove a document from memory and disk.

        Raises OSError when the disk copy cannot be removed, so deletion is
        never silently incomplete.
        """
        if not is_valid_document_key(document_key):
            raise KeyError(document_key)
        with self._lock:
            found = document_key in self
            self._drop_persisted(document_key)
            self._memory.pop(document_key, None)
            self._index.remove_document(document_key)
            if not found:
                raise KeyError(document_key)

    def __contains__(self, document_key: object) -> bool:
        """Return whether a document exists in memory or on disk."""
        if not is_valid_document_key(document_key):
            return False
        with self._lock:
            return document_key in self._memory or self._doc_path(document_key).exists()

    def _disk_keys(self) -> list[str]:
        keys = []
        for path in self._dir.glob("*.json"):
            if path.name.endswith(_META_SUFFIX):
                continue
            if _KEY_PATTERN.fullmatch(path.stem):
                keys.append(path.stem)
        return keys

    def __iter__(self) -> Iterator[str]:
        """Iterate over all document keys in memory and on disk."""
        with self._lock:
            seen = set(self._memory.keys())
            disk = [key for key in self._disk_keys() if key not in seen]
        yield from seen
        yield from disk

    def __len__(self) -> int:
        """Return the number of distinct documents in memory and on disk."""
        with self._lock:
            return len(set(self._memory.keys()) | set(self._disk_keys()))

    def list_metadata(self) -> list[DocumentMetadata]:
        """Return metadata for all stored documents.

        In-memory documents are authoritative for their descriptive fields;
        the persisted timestamp is taken from the sidecar when one exists.
        A sidecar is only trusted when its embedded key matches its filename.
        """
        with self._lock:
            records: dict[str, DocumentMetadata] = {}

            for key in self._disk_keys():
                stored_at = None
                try:
                    record = DocumentMetadata.model_validate(
                        json.loads(self._meta_path(key).read_text(encoding="utf-8"))
                    )
                    if record.document_key == key:
                        stored_at = record.stored_at
                        records[key] = record
                    else:
                        logger.warning(f"Sidecar key mismatch for: {key}")
                except (OSError, ValueError):
                    logger.warning(f"Missing or invalid metadata sidecar for: {key}")
                if key not in records:
                    # The document file itself proves this is a persisted
                    # document; fall back to its mtime so listings that
                    # filter on persistence do not hide it.
                    try:
                        mtime = self._doc_path(key).stat().st_mtime
                        stored_at = datetime.fromtimestamp(
                            mtime, tz=timezone.utc
                        ).isoformat()
                    except OSError:
                        stored_at = None
                    records[key] = DocumentMetadata(
                        document_key=key, stored_at=stored_at
                    )

            for key, doc in self._memory.items():
                disk_stored_at = records[key].stored_at if key in records else None
                records[key] = DocumentMetadata.from_document(
                    key, doc, stored_at=disk_stored_at
                )

            return sorted(records.values(), key=lambda r: r.document_key)

    def _sync_index(self) -> None:
        """Reconcile the search index with the stored conversion artifacts.

        Rebuilds any missing entries (a deleted or freshly recreated index
        database) and drops entries whose document no longer exists, so the
        index is always derivable from the store. Documents only on disk are
        loaded transiently for indexing without pinning them in memory.
        """
        indexed = self._index.indexed_keys()
        current: dict[str, DoclingDocument | None] = {
            key: doc for key, doc in self._memory.items() if doc.origin is not None
        }
        for key in self._disk_keys():
            current.setdefault(key, None)

        for key in indexed - set(current):
            self._index.remove_document(key)

        for key, doc in current.items():
            if key in indexed:
                continue
            if doc is None:
                try:
                    doc = DoclingDocument.load_from_json(filename=self._doc_path(key))
                except (OSError, ValueError) as exc:
                    logger.warning(f"Could not index document {key}: {exc}")
                    continue
            self._index.index_document(key, doc)

    def search_corpus(
        self,
        query: str,
        max_results: int,
        document_keys: list[str] | None = None,
    ) -> list[CorpusSearchHit]:
        """Return ranked lexical matches across the stored documents.

        The index reflects immutable conversion artifacts: session-only tool
        edits are not searchable, matching what a restart would serve.
        """
        with self._lock:
            self._sync_index()
            return self._index.search(query, max_results, document_keys)
