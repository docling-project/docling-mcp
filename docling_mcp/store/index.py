"""SQLite FTS5 search index over the items of stored documents."""

import os
import re
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from pathlib import Path

from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.items.node import DocItem
from docling_core.types.doc.items.table.table import TableItem

from docling_mcp.logger import setup_logger
from docling_mcp.store.base import CorpusSearchHit

logger = setup_logger()

_SCHEMA_VERSION = 1

# Column order of the corpus table; snippet() addresses text by position.
_TEXT_COLUMN = 4

_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


def _iter_index_rows(
    document_key: str, document: DoclingDocument
) -> Iterator[tuple[str, str, str, int | None, str]]:
    """Yield one (key, anchor, label, page, text) row per searchable item.

    Rows come from the document body via iterate_items(), so furniture such
    as the converter-added source line is not indexed. Tables contribute the
    concatenated text of their cells.
    """
    for item, _level in document.iterate_items():
        if not isinstance(item, DocItem):
            continue
        if isinstance(item, TableItem):
            text = " ".join(cell.text for cell in item.data.table_cells if cell.text)
        else:
            text = getattr(item, "text", "") or ""
        text = text.strip()
        if not text:
            continue
        page = item.prov[0].page_no if item.prov else None
        yield (document_key, item.get_ref().cref, item.label.value, page, text)


def _fts_match_expression(query: str) -> str:
    """Build a safe FTS5 MATCH expression from a free-text query.

    Each word is quoted so FTS5 query syntax in the input (operators,
    parentheses, dangling quotes) cannot raise a parse error; the quoted
    words combine with FTS5's implicit AND.
    """
    tokens = _TOKEN_PATTERN.findall(query)
    return " ".join(f'"{token}"' for token in tokens)


class CorpusIndex:
    """Lexical search index stored beside the persisted documents.

    The index is derived data with the same trust model as the metadata
    sidecars: it can always be rebuilt from the stored documents, so index
    writes never fail document storage. A corrupt database file is
    quarantined and replaced with an empty index, which the owning store
    repopulates during its sync-on-search pass. When SQLite lacks the FTS5
    extension, indexing degrades to a no-op and searches raise a clear
    error; see the fallback open question in the design notes.

    Access is serialized by an internal lock because MCP tool calls can run
    on different threads while SQLite connections default to single-thread
    affinity.
    """

    def __init__(self, db_path: Path | None) -> None:
        self._path = db_path
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self._fts5_available = True
        self._connect()

    def _connect(self) -> None:
        target = str(self._path) if self._path is not None else ":memory:"
        try:
            conn = sqlite3.connect(target, check_same_thread=False)
        except sqlite3.Error:
            logger.exception(f"Could not open corpus index: {target}")
            self._conn = None
            return
        try:
            self._init_schema(conn)
        except sqlite3.DatabaseError:
            conn.close()
            if self._quarantine():
                try:
                    conn = sqlite3.connect(target, check_same_thread=False)
                    self._init_schema(conn)
                except sqlite3.Error:
                    logger.exception(f"Could not recreate corpus index: {target}")
                    self._conn = None
                    return
            else:
                self._conn = None
                return
        self._conn = conn
        if self._path is not None:
            try:
                os.chmod(self._path, 0o600)
            except OSError:
                logger.warning(f"Could not restrict corpus index mode: {self._path}")

    def _init_schema(self, conn: sqlite3.Connection) -> None:
        conn.execute("PRAGMA busy_timeout = 5000")
        try:
            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS corpus USING fts5("
                "document_key UNINDEXED, anchor UNINDEXED, label UNINDEXED, "
                "page UNINDEXED, text)"
            )
        except sqlite3.OperationalError as exc:
            if "fts5" in str(exc).lower():
                logger.warning(
                    f"SQLite FTS5 unavailable; corpus search disabled: {exc}"
                )
                self._fts5_available = False
                return
            raise
        conn.execute(
            "CREATE TABLE IF NOT EXISTS indexed_documents("
            "document_key TEXT PRIMARY KEY)"
        )
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version != _SCHEMA_VERSION:
            with conn:
                conn.execute("DELETE FROM corpus")
                conn.execute("DELETE FROM indexed_documents")
                conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")

    def _quarantine(self) -> bool:
        """Move a corrupt database file aside; return whether a retry makes sense."""
        if self._path is None:
            return False
        quarantine = self._path.with_name(
            f"{self._path.name}.corrupt-{os.getpid()}-{uuid.uuid4().hex}"
        )
        try:
            os.replace(self._path, quarantine)
        except OSError:
            logger.warning(f"Could not quarantine corpus index: {self._path}")
            return False
        logger.warning(f"Quarantined corrupt corpus index to: {quarantine}")
        return True

    def _recover(self) -> None:
        """Drop the broken connection, quarantine the file, and reconnect."""
        if self._conn is not None:
            try:
                self._conn.close()
            except sqlite3.Error:
                logger.warning("Could not close broken corpus index connection")
            self._conn = None
        self._quarantine()
        self._connect()

    @property
    def available(self) -> bool:
        """Return whether the index can serve searches."""
        return self._conn is not None and self._fts5_available

    def index_document(self, document_key: str, document: DoclingDocument) -> None:
        """Replace the indexed rows for a document; never raises."""
        rows = list(_iter_index_rows(document_key, document))
        with self._lock:
            if not self.available:
                return
            assert self._conn is not None
            try:
                with self._conn:
                    self._conn.execute(
                        "DELETE FROM corpus WHERE document_key = ?", (document_key,)
                    )
                    self._conn.executemany(
                        "INSERT INTO corpus(document_key, anchor, label, page, text) "
                        "VALUES (?, ?, ?, ?, ?)",
                        rows,
                    )
                    self._conn.execute(
                        "INSERT OR REPLACE INTO indexed_documents(document_key) "
                        "VALUES (?)",
                        (document_key,),
                    )
            except sqlite3.DatabaseError:
                logger.exception(f"Could not index document: {document_key}")
                self._recover()

    def remove_document(self, document_key: str) -> None:
        """Remove a document's rows from the index; never raises."""
        with self._lock:
            if not self.available:
                return
            assert self._conn is not None
            try:
                with self._conn:
                    self._conn.execute(
                        "DELETE FROM corpus WHERE document_key = ?", (document_key,)
                    )
                    self._conn.execute(
                        "DELETE FROM indexed_documents WHERE document_key = ?",
                        (document_key,),
                    )
            except sqlite3.DatabaseError:
                logger.exception(f"Could not deindex document: {document_key}")
                self._recover()

    def indexed_keys(self) -> set[str]:
        """Return the keys currently present in the index; never raises."""
        with self._lock:
            if not self.available:
                return set()
            assert self._conn is not None
            try:
                rows = self._conn.execute(
                    "SELECT document_key FROM indexed_documents"
                ).fetchall()
            except sqlite3.DatabaseError:
                logger.exception("Could not read indexed document keys")
                self._recover()
                return set()
            return {row[0] for row in rows}

    def search(
        self,
        query: str,
        max_results: int,
        document_keys: list[str] | None = None,
    ) -> list[CorpusSearchHit]:
        """Return ranked matches for a free-text query.

        Raises RuntimeError when the index cannot serve searches because
        SQLite FTS5 is unavailable or the index database cannot be opened.
        """
        with self._lock:
            if not self.available:
                raise RuntimeError(
                    "Corpus search is unavailable: this Python's SQLite build "
                    "lacks the FTS5 extension or the search index cannot be "
                    "opened."
                )
            assert self._conn is not None

            match = _fts_match_expression(query)
            if not match:
                return []
            if document_keys is not None and len(document_keys) == 0:
                return []

            sql = (
                "SELECT document_key, anchor, page, "
                f"snippet(corpus, {_TEXT_COLUMN}, '', '', ' … ', 20) "
                "FROM corpus WHERE corpus MATCH ?"
            )
            params: list[object] = [match]
            if document_keys is not None:
                placeholders = ", ".join("?" for _ in document_keys)
                sql += f" AND document_key IN ({placeholders})"
                params.extend(document_keys)
            sql += " ORDER BY rank LIMIT ?"
            params.append(max(0, max_results))

            try:
                rows = self._conn.execute(sql, params).fetchall()
            except sqlite3.DatabaseError:
                logger.exception("Corpus search failed; resetting the index")
                self._recover()
                return []

            return [
                CorpusSearchHit(
                    document_key=row[0],
                    anchor=row[1],
                    page=int(row[2]) if row[2] is not None else None,
                    snippet=row[3],
                )
                for row in rows
            ]
