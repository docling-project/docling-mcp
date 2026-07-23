"""Test the persistent document store and corpus tools."""

import json
import os
from pathlib import Path

import pytest

from docling_core.types.doc.common.origin import DocumentOrigin
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel

from docling_mcp.docling_cache import get_cache_key
from docling_mcp.store.local import InMemoryDocumentStore, LocalDocumentStore

not_root = pytest.mark.skipif(
    getattr(os, "geteuid", lambda: 1)() == 0,
    reason="permission bits do not bind root",
)


def make_document(name: str, text: str, converted: bool = True) -> DoclingDocument:
    doc = DoclingDocument(name=name)
    if converted:
        doc.origin = DocumentOrigin(
            mimetype="application/pdf",
            binary_hash=hash(name) & 0xFFFFFFFF,
            filename=f"{name}.pdf",
        )
    doc.add_text(label=DocItemLabel.TEXT, text=text)
    return doc


KEY_A = "a" * 32
KEY_B = "b" * 32
KEY_C = "c" * 32


def test_roundtrip_and_mapping_semantics(tmp_path: Path) -> None:
    store = LocalDocumentStore(cache_dir=tmp_path)

    assert KEY_A not in store
    store[KEY_A] = make_document("doc-a", "hello world")

    assert KEY_A in store
    assert store[KEY_A].name == "doc-a"
    assert set(store.keys()) == {KEY_A}
    assert len(store) == 1


def test_restart_survival(tmp_path: Path) -> None:
    store = LocalDocumentStore(cache_dir=tmp_path)
    doc = make_document("doc-a", "survives restarts")
    store[KEY_A] = doc

    # A new instance over the same directory simulates a server restart.
    reborn = LocalDocumentStore(cache_dir=tmp_path)

    assert KEY_A in reborn
    loaded = reborn[KEY_A]
    assert loaded.name == "doc-a"
    assert loaded.export_to_markdown() == doc.export_to_markdown()


def test_authored_documents_stay_memory_only(tmp_path: Path) -> None:
    store = LocalDocumentStore(cache_dir=tmp_path)
    store[KEY_A] = make_document("draft", "work in progress", converted=False)

    assert KEY_A in store

    reborn = LocalDocumentStore(cache_dir=tmp_path)
    assert KEY_A not in reborn


def test_edits_are_session_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import docling_mcp.tools.manipulation as manipulation

    store = LocalDocumentStore(cache_dir=tmp_path)
    doc = make_document("doc-a", "original text")
    store[KEY_A] = doc
    anchor = doc.texts[0].get_ref().cref
    monkeypatch.setattr(manipulation, "local_document_cache", store)

    manipulation.update_text_of_document_item_at_anchor(
        document_key=KEY_A, document_anchor=anchor, updated_text="edited text"
    )

    # The edit is visible in this session but the persisted conversion
    # artifact stays pristine: a restart serves the converted source.
    assert "edited text" in store[KEY_A].export_to_markdown()
    reborn = LocalDocumentStore(cache_dir=tmp_path)
    assert "original text" in reborn[KEY_A].export_to_markdown()


def test_persist_failure_keeps_memory_authoritative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = LocalDocumentStore(cache_dir=tmp_path)

    def boom(self: DoclingDocument, filename: Path) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(DoclingDocument, "save_as_json", boom)
    store[KEY_A] = make_document("doc-a", "unpersisted")

    assert "unpersisted" in store[KEY_A].export_to_markdown()
    assert not (tmp_path / f"{KEY_A}.json").exists()


def test_originless_overwrite_removes_disk_copy(tmp_path: Path) -> None:
    store = LocalDocumentStore(cache_dir=tmp_path)
    store[KEY_A] = make_document("doc-a", "converted")
    store[KEY_A] = make_document("draft", "replaced in memory", converted=False)

    assert "replaced in memory" in store[KEY_A].export_to_markdown()

    reborn = LocalDocumentStore(cache_dir=tmp_path)
    assert KEY_A not in reborn


def test_delete_removes_memory_and_disk(tmp_path: Path) -> None:
    store = LocalDocumentStore(cache_dir=tmp_path)
    store[KEY_A] = make_document("doc-a", "to delete")

    del store[KEY_A]

    assert KEY_A not in store
    assert not list(tmp_path.glob("*.json"))
    with pytest.raises(KeyError):
        del store[KEY_A]


@not_root
def test_failed_disk_removal_raises_on_delete(tmp_path: Path) -> None:
    store = LocalDocumentStore(cache_dir=tmp_path)
    store[KEY_A] = make_document("doc-a", "undeletable")

    tmp_path.chmod(0o500)
    try:
        with pytest.raises(OSError):
            del store[KEY_A]
    finally:
        tmp_path.chmod(0o700)

    # Deletion did not silently half-succeed: the document is still there.
    assert KEY_A in store
    assert "undeletable" in store[KEY_A].export_to_markdown()


def test_corrupt_document_file_behaves_like_absent_key(tmp_path: Path) -> None:
    store = LocalDocumentStore(cache_dir=tmp_path)
    (tmp_path / f"{KEY_A}.json").write_text("{not a document", encoding="utf-8")
    assert KEY_A in store

    with pytest.raises(KeyError):
        store[KEY_A]

    # The corrupt entry is quarantined, so membership no longer reports it
    # and the source can be converted again.
    assert KEY_A not in store
    assert not (tmp_path / f"{KEY_A}.json").exists()


def test_invalid_keys_are_rejected(tmp_path: Path) -> None:
    store = LocalDocumentStore(cache_dir=tmp_path)
    evil = "../../escape"

    with pytest.raises(ValueError):
        store[evil] = make_document("doc", "text")
    assert evil not in store
    with pytest.raises(KeyError):
        store[evil]
    with pytest.raises(KeyError):
        del store[evil]
    assert not (tmp_path.parent / "escape.json").exists()


def test_keys_with_trailing_newline_are_rejected(tmp_path: Path) -> None:
    from docling_mcp.store.local import is_valid_document_key

    sneaky = "a" * 32 + "\n"
    assert not is_valid_document_key(sneaky)

    store = LocalDocumentStore(cache_dir=tmp_path)
    with pytest.raises(ValueError):
        store[sneaky] = make_document("doc", "text")


def test_bulk_mutation_apis_validate_keys(tmp_path: Path) -> None:
    doc = make_document("doc", "text")

    memory_store = InMemoryDocumentStore()
    with pytest.raises(ValueError):
        memory_store.update({"../../escape": doc})
    with pytest.raises(ValueError):
        memory_store.setdefault("../../escape", doc)

    local_store = LocalDocumentStore(cache_dir=tmp_path)
    with pytest.raises(ValueError):
        local_store.update({"../../escape": doc})
    with pytest.raises(ValueError):
        local_store.setdefault("../../escape", doc)


def test_disk_scan_ignores_malformed_filenames(tmp_path: Path) -> None:
    store = LocalDocumentStore(cache_dir=tmp_path)
    store[KEY_A] = make_document("doc-a", "real")
    (tmp_path / (KEY_B + "\n.json")).write_text("{}", encoding="utf-8")

    assert set(store.keys()) == {KEY_A}
    assert len(store) == 1


@not_root
def test_unwritable_directory_raises_at_construction(tmp_path: Path) -> None:
    # An owner-restricted directory is auto-tightened to 0700, so true
    # unwritability is modeled by a locked parent the store cannot create
    # its directory under.
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        with pytest.raises(OSError):
            LocalDocumentStore(cache_dir=locked / "cache")
    finally:
        locked.chmod(0o700)


@not_root
def test_factory_falls_back_to_memory_when_unwritable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from docling_mcp.settings.store import settings as store_settings
    from docling_mcp.store.factory import create_document_store

    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    monkeypatch.setattr(store_settings, "cache_persist", True)
    monkeypatch.setattr(store_settings, "cache_dir", locked / "cache")
    try:
        store = create_document_store()
    finally:
        locked.chmod(0o700)

    assert isinstance(store, InMemoryDocumentStore)


@not_root
def test_store_files_are_owner_only(tmp_path: Path) -> None:
    base = tmp_path / "cache"
    store = LocalDocumentStore(cache_dir=base)
    store[KEY_A] = make_document("doc-a", "private")

    assert (base.stat().st_mode & 0o777) == 0o700
    assert ((base / f"{KEY_A}.json").stat().st_mode & 0o777) == 0o600
    assert ((base / f"{KEY_A}.meta.json").stat().st_mode & 0o777) == 0o600


@not_root
def test_pre_existing_directory_is_tightened(tmp_path: Path) -> None:
    base = tmp_path / "cache"
    base.mkdir(mode=0o755)

    LocalDocumentStore(cache_dir=base)

    assert (base.stat().st_mode & 0o777) == 0o700


def test_list_metadata(tmp_path: Path) -> None:
    store = LocalDocumentStore(cache_dir=tmp_path)
    store[KEY_A] = make_document("doc-a", "converted")
    store[KEY_B] = make_document("draft", "authored", converted=False)

    records = {r.document_key: r for r in store.list_metadata()}

    assert set(records) == {KEY_A, KEY_B}
    assert records[KEY_A].source_filename == "doc-a.pdf"
    assert records[KEY_A].stored_at is not None
    assert records[KEY_B].stored_at is None


def test_list_metadata_prefers_live_memory_state(tmp_path: Path) -> None:
    store = LocalDocumentStore(cache_dir=tmp_path)
    doc = make_document("doc-a", "converted")
    store[KEY_A] = doc

    # Mutate the live object; descriptive fields must reflect memory while
    # the persisted timestamp is retained from the sidecar.
    doc.name = "renamed"
    records = {r.document_key: r for r in store.list_metadata()}

    assert records[KEY_A].name == "renamed"
    assert records[KEY_A].stored_at is not None


def test_list_metadata_rejects_mismatched_sidecar(tmp_path: Path) -> None:
    store = LocalDocumentStore(cache_dir=tmp_path)
    store[KEY_A] = make_document("doc-a", "converted")

    meta_path = tmp_path / f"{KEY_A}.meta.json"
    record = json.loads(meta_path.read_text(encoding="utf-8"))
    record["document_key"] = KEY_B
    meta_path.write_text(json.dumps(record), encoding="utf-8")

    reborn = LocalDocumentStore(cache_dir=tmp_path)
    records = {r.document_key: r for r in reborn.list_metadata()}

    # The mismatched sidecar must not advertise KEY_B; the document file
    # still proves KEY_A exists.
    assert KEY_B not in records
    assert KEY_A in records
    assert records[KEY_A].stored_at is not None


def test_in_memory_store_list_metadata() -> None:
    store = InMemoryDocumentStore()
    store[KEY_A] = make_document("doc-a", "in memory")

    records = store.list_metadata()

    assert len(records) == 1
    assert records[0].document_key == KEY_A
    assert records[0].stored_at is None


def test_in_memory_store_rejects_invalid_keys() -> None:
    store = InMemoryDocumentStore()
    with pytest.raises(ValueError):
        store["../../escape"] = make_document("doc", "text")


def test_cache_key_dedupes_identical_content(tmp_path: Path) -> None:
    file_one = tmp_path / "one.pdf"
    file_two = tmp_path / "sub" / "two.pdf"
    file_two.parent.mkdir()
    file_one.write_bytes(b"same bytes")
    file_two.write_bytes(b"same bytes")

    assert get_cache_key(str(file_one)) == get_cache_key(str(file_two))


def test_cache_key_distinguishes_file_formats(tmp_path: Path) -> None:
    file_one = tmp_path / "doc.html"
    file_two = tmp_path / "doc.md"
    file_one.write_bytes(b"same bytes")
    file_two.write_bytes(b"same bytes")

    # The suffix selects the input format, so identical bytes under
    # different extensions must convert separately.
    assert get_cache_key(str(file_one)) != get_cache_key(str(file_two))


def test_cache_key_changes_when_content_changes(tmp_path: Path) -> None:
    source = tmp_path / "doc.pdf"
    source.write_bytes(b"version one")
    key_one = get_cache_key(str(source))

    source.write_bytes(b"version two, longer")
    key_two = get_cache_key(str(source))

    assert key_one != key_two


def test_cache_key_detects_same_size_rewrite_with_preserved_mtime(
    tmp_path: Path,
) -> None:
    source = tmp_path / "doc.pdf"
    source.write_bytes(b"aaaa")
    key_one = get_cache_key(str(source))
    stat = source.stat()

    source.write_bytes(b"bbbb")
    os.utime(source, (stat.st_atime, stat.st_mtime))

    assert get_cache_key(str(source)) != key_one


def test_cache_key_for_urls_uses_source_string() -> None:
    url = "https://example.com/spec.pdf"

    assert get_cache_key(url) == get_cache_key(url)
    assert get_cache_key(url) != get_cache_key(url + "?v=2")


def test_cache_key_uses_converter_supplied_context(tmp_path: Path) -> None:
    from docling_mcp.docling_cache import (
        local_conversion_context,
        remote_conversion_context,
    )

    source = tmp_path / "doc.pdf"
    source.write_bytes(b"stable bytes")

    local_key = get_cache_key(str(source), conversion=local_conversion_context())
    remote_key = get_cache_key(str(source), conversion=remote_conversion_context())

    # A fallback conversion executed locally must never share a key with a
    # remote conversion of the same source.
    assert local_key != remote_key


def test_cache_key_covers_conversion_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from docling_mcp.settings.service_client import (
        ConversionMode,
        settings as service_settings,
    )

    source = tmp_path / "doc.pdf"
    source.write_bytes(b"stable bytes")

    monkeypatch.setattr(service_settings, "conversion_mode", ConversionMode.LOCAL)
    local_key = get_cache_key(str(source))

    monkeypatch.setattr(service_settings, "conversion_mode", ConversionMode.REMOTE)
    monkeypatch.setattr(service_settings, "service_url", "https://serve-a.example.com")
    remote_key_a = get_cache_key(str(source))

    monkeypatch.setattr(service_settings, "service_url", "https://serve-b.example.com")
    remote_key_b = get_cache_key(str(source))

    assert local_key != remote_key_a
    assert remote_key_a != remote_key_b


def test_list_converted_documents_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import docling_mcp.tools.corpus as corpus

    store = LocalDocumentStore(cache_dir=tmp_path)
    store[KEY_A] = make_document("doc-a", "listed")
    store[KEY_B] = make_document("draft", "not listed", converted=False)
    monkeypatch.setattr(corpus, "local_document_cache", store)

    entries = corpus.list_converted_documents()

    assert len(entries) == 1
    assert entries[0].document_key == KEY_A
    assert entries[0].source_filename == "doc-a.pdf"


def test_list_converted_documents_survives_corrupt_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import docling_mcp.tools.corpus as corpus

    store = LocalDocumentStore(cache_dir=tmp_path)
    store[KEY_A] = make_document("doc-a", "listed")
    (tmp_path / f"{KEY_A}.meta.json").write_text("{not json", encoding="utf-8")
    reborn = LocalDocumentStore(cache_dir=tmp_path)
    monkeypatch.setattr(corpus, "local_document_cache", reborn)

    entries = corpus.list_converted_documents()

    assert len(entries) == 1
    assert entries[0].document_key == KEY_A
    assert entries[0].stored_at is not None


def test_restored_document_gives_clean_error_in_generation_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from collections import defaultdict

    import docling_mcp.tools.generation as generation

    seed = LocalDocumentStore(cache_dir=tmp_path)
    seed[KEY_A] = make_document("doc-a", "restored")

    reborn = LocalDocumentStore(cache_dir=tmp_path)
    monkeypatch.setattr(generation, "local_document_cache", reborn)
    monkeypatch.setattr(generation, "local_stack_cache", defaultdict(list))

    with pytest.raises(ValueError, match="Stack size is zero"):
        generation.add_paragraph_to_docling_document(
            document_key=KEY_A, paragraph="new paragraph"
        )
