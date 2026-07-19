"""Test lexical search across the stored document corpus."""

from pathlib import Path

import pytest

from docling_core.types.doc.base import BoundingBox
from docling_core.types.doc.common.origin import DocumentOrigin
from docling_core.types.doc.common.reference import ProvenanceItem
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.items.table.table_data import TableCell, TableData
from docling_core.types.doc.labels import DocItemLabel

from docling_mcp.store.local import InMemoryDocumentStore, LocalDocumentStore

KEY_A = "a" * 32
KEY_B = "b" * 32

_INDEX_FILENAME = "corpus-index.sqlite3"


def _prov(page: int, length: int) -> ProvenanceItem:
    return ProvenanceItem(
        page_no=page,
        bbox=BoundingBox(l=0.0, t=0.0, r=1.0, b=1.0),
        charspan=(0, length),
    )


def make_corpus_document(
    name: str, entries: list[tuple[str, int]], converted: bool = True
) -> DoclingDocument:
    """Build a document whose text items carry page provenance."""
    doc = DoclingDocument(name=name)
    if converted:
        doc.origin = DocumentOrigin(
            mimetype="application/pdf",
            binary_hash=hash(name) & 0xFFFFFFFF,
            filename=f"{name}.pdf",
        )
    for text, page in entries:
        doc.add_text(label=DocItemLabel.TEXT, text=text, prov=_prov(page, len(text)))
    return doc


def test_search_returns_anchor_page_and_snippet(tmp_path: Path) -> None:
    store = LocalDocumentStore(cache_dir=tmp_path)
    doc = make_corpus_document(
        "doc-a",
        [
            ("The flux capacitor requires 1.21 gigawatts of power.", 3),
            ("Unrelated prose about gardening tools.", 5),
        ],
    )
    store[KEY_A] = doc

    hits = store.search_corpus("flux capacitor", max_results=10)

    assert len(hits) == 1
    hit = hits[0]
    assert hit.document_key == KEY_A
    assert hit.anchor == doc.texts[0].get_ref().cref
    assert hit.page == 3
    assert "flux capacitor" in hit.snippet


def test_search_limits_results(tmp_path: Path) -> None:
    store = LocalDocumentStore(cache_dir=tmp_path)
    entries = [(f"widget number {i} in the catalog", i + 1) for i in range(5)]
    store[KEY_A] = make_corpus_document("doc-a", entries)

    hits = store.search_corpus("widget", max_results=2)

    assert len(hits) == 2


def test_search_filters_by_document_keys(tmp_path: Path) -> None:
    store = LocalDocumentStore(cache_dir=tmp_path)
    store[KEY_A] = make_corpus_document("doc-a", [("shared banana term", 1)])
    store[KEY_B] = make_corpus_document("doc-b", [("shared banana term", 2)])

    hits = store.search_corpus("banana", max_results=10, document_keys=[KEY_B])

    assert [hit.document_key for hit in hits] == [KEY_B]


def test_search_index_survives_restart(tmp_path: Path) -> None:
    store = LocalDocumentStore(cache_dir=tmp_path)
    store[KEY_A] = make_corpus_document("doc-a", [("persistent aardvark facts", 4)])

    reborn = LocalDocumentStore(cache_dir=tmp_path)
    hits = reborn.search_corpus("aardvark", max_results=10)

    assert len(hits) == 1
    assert hits[0].document_key == KEY_A
    assert hits[0].page == 4


def test_search_index_rebuilds_when_missing(tmp_path: Path) -> None:
    store = LocalDocumentStore(cache_dir=tmp_path)
    store[KEY_A] = make_corpus_document("doc-a", [("rebuildable zeppelin manual", 2)])
    (tmp_path / _INDEX_FILENAME).unlink()

    reborn = LocalDocumentStore(cache_dir=tmp_path)
    hits = reborn.search_corpus("zeppelin", max_results=10)

    assert len(hits) == 1
    assert hits[0].document_key == KEY_A
    assert hits[0].anchor == store[KEY_A].texts[0].get_ref().cref


def test_corrupt_index_does_not_break_conversion(tmp_path: Path) -> None:
    store = LocalDocumentStore(cache_dir=tmp_path)
    store[KEY_A] = make_corpus_document("doc-a", [("first quokka document", 1)])

    (tmp_path / _INDEX_FILENAME).write_bytes(b"this is not a sqlite database")

    # Storing a new conversion must not raise even though the index is junk.
    store[KEY_B] = make_corpus_document("doc-b", [("second quokka document", 2)])

    hits = store.search_corpus("quokka", max_results=10)
    assert {hit.document_key for hit in hits} == {KEY_A, KEY_B}


def test_deleted_document_removed_from_search(tmp_path: Path) -> None:
    store = LocalDocumentStore(cache_dir=tmp_path)
    store[KEY_A] = make_corpus_document("doc-a", [("shared mongoose topic", 1)])
    store[KEY_B] = make_corpus_document("doc-b", [("shared mongoose topic", 2)])

    del store[KEY_A]
    hits = store.search_corpus("mongoose", max_results=10)

    assert [hit.document_key for hit in hits] == [KEY_B]


def test_search_drops_entries_for_missing_documents(tmp_path: Path) -> None:
    store = LocalDocumentStore(cache_dir=tmp_path)
    store[KEY_A] = make_corpus_document("doc-a", [("externally removed ocelot", 1)])

    # Simulate another process deleting the document but not the index rows.
    (tmp_path / f"{KEY_A}.json").unlink()
    (tmp_path / f"{KEY_A}.meta.json").unlink()

    reborn = LocalDocumentStore(cache_dir=tmp_path)
    assert reborn.search_corpus("ocelot", max_results=10) == []


def test_session_edits_do_not_change_index(tmp_path: Path) -> None:
    store = LocalDocumentStore(cache_dir=tmp_path)
    doc = make_corpus_document("doc-a", [("original narwhal wording", 1)])
    store[KEY_A] = doc

    # Tool edits mutate session memory only; the index keeps reflecting the
    # immutable conversion artifact.
    doc.texts[0].text = "edited kraken wording"

    assert store.search_corpus("kraken", max_results=10) == []
    assert len(store.search_corpus("narwhal", max_results=10)) == 1


def test_authored_documents_are_not_indexed(tmp_path: Path) -> None:
    store = LocalDocumentStore(cache_dir=tmp_path)
    store[KEY_A] = make_corpus_document(
        "draft", [("unpublished ibex draft", 1)], converted=False
    )

    assert store.search_corpus("ibex", max_results=10) == []


def test_search_handles_special_query_characters(tmp_path: Path) -> None:
    store = LocalDocumentStore(cache_dir=tmp_path)
    store[KEY_A] = make_corpus_document(
        "doc-a", [("the don't-panic guide to towels", 1)]
    )

    for query in ["don't", 'panic - "guide"', "AND OR NOT", "(towels", "*"]:
        hits = store.search_corpus(query, max_results=10)
        assert isinstance(hits, list)

    assert len(store.search_corpus("don't-panic", max_results=10)) == 1


def test_table_content_is_searchable(tmp_path: Path) -> None:
    store = LocalDocumentStore(cache_dir=tmp_path)
    doc = make_corpus_document("doc-a", [("intro paragraph", 1)])
    cells = [
        TableCell(
            text=text,
            start_row_offset_idx=0,
            end_row_offset_idx=1,
            start_col_offset_idx=col,
            end_col_offset_idx=col + 1,
        )
        for col, text in enumerate(["Voltage", "1.21 gigawatts"])
    ]
    doc.add_table(
        data=TableData(num_rows=1, num_cols=2, table_cells=cells),
        prov=_prov(7, 0),
    )
    store[KEY_A] = doc

    hits = store.search_corpus("voltage", max_results=10)

    assert len(hits) == 1
    assert hits[0].anchor == doc.tables[0].get_ref().cref
    assert hits[0].page == 7


def test_in_memory_store_search_corpus() -> None:
    store = InMemoryDocumentStore()
    store[KEY_A] = make_corpus_document("doc-a", [("volatile wombat notes", 6)])

    hits = store.search_corpus("wombat", max_results=10)
    assert len(hits) == 1
    assert hits[0].page == 6

    del store[KEY_A]
    assert store.search_corpus("wombat", max_results=10) == []


def test_search_without_fts5_raises_clear_error(tmp_path: Path) -> None:
    store = LocalDocumentStore(cache_dir=tmp_path)
    store[KEY_A] = make_corpus_document("doc-a", [("any content", 1)])

    store._index._fts5_available = False

    with pytest.raises(RuntimeError, match="FTS5"):
        store.search_corpus("content", max_results=10)


def test_search_across_documents_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import docling_mcp.tools.corpus as corpus

    store = LocalDocumentStore(cache_dir=tmp_path)
    store[KEY_A] = make_corpus_document(
        "doc-a", [("the tardigrade survives in vacuum", 9)]
    )
    monkeypatch.setattr(corpus, "local_document_cache", store)

    hits = corpus.search_across_documents(query="tardigrade", max_results=5)

    assert len(hits) == 1
    assert hits[0].document_key == KEY_A
    assert hits[0].anchor == store[KEY_A].texts[0].get_ref().cref
    assert hits[0].page == 9
    assert "tardigrade" in hits[0].snippet


def test_search_across_documents_tool_rejects_empty_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import docling_mcp.tools.corpus as corpus

    monkeypatch.setattr(
        corpus, "local_document_cache", LocalDocumentStore(cache_dir=tmp_path)
    )

    with pytest.raises(ValueError, match="query"):
        corpus.search_across_documents(query="   ", max_results=5)


def test_search_across_documents_tool_rejects_malformed_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import docling_mcp.tools.corpus as corpus

    monkeypatch.setattr(
        corpus, "local_document_cache", LocalDocumentStore(cache_dir=tmp_path)
    )

    with pytest.raises(ValueError, match="document key"):
        corpus.search_across_documents(
            query="anything", max_results=5, document_keys=["../../escape"]
        )
