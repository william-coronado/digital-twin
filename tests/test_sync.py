"""Incremental indexing: manifest handling, rebuilds, failure recovery and the watcher."""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import ingest
from conftest import chunk_count, sources_in_db


def sync():
    return ingest.sync_vector_store(verbose=False)


def manifest_on_disk():
    return json.load(open(ingest.MANIFEST_PATH))


def test_initial_sync_indexes_every_file_with_stable_ids(store):
    m = sync()
    assert set(m["files"]) == {"project_aurora.md", "employees.csv"}
    assert m["total_chunks"] == chunk_count() > 0
    assert m["index_settings"]["embedding_model"] == "test-embed"
    assert all("#" in i for i in ingest.get_db()._collection.get()["ids"])
    assert not os.path.exists(ingest.MANIFEST_PATH + ".tmp")


def test_unchanged_files_are_not_reindexed(store, monkeypatch):
    sync()
    calls = []
    monkeypatch.setattr(ingest, "parse_file", lambda *a: calls.append(a) or [])
    sync()
    assert calls == []


def test_modified_file_is_reindexed_without_duplicates(store):
    sync()
    n = chunk_count()
    (store.src / "project_aurora.md").write_text("# Project Aurora\nThe budget was revised to $3.1 million.\n")
    sync()
    docs = ingest.get_db()._collection.get(where={"source": "project_aurora.md"})["documents"]
    assert any("$3.1 million" in d for d in docs) and not any("$2.4 million" in d for d in docs)
    assert chunk_count() <= n


def test_file_replaced_by_older_copy_counts_as_modified(store):
    sync()
    path = store.src / "project_aurora.md"
    old = manifest_on_disk()["files"]["project_aurora.md"]["mtime"]
    os.utime(path, (old - 1000, old - 1000))
    sync()
    assert manifest_on_disk()["files"]["project_aurora.md"]["mtime"] == old - 1000


def test_deleted_file_is_removed(store):
    sync()
    (store.src / "employees.csv").unlink()
    m = sync()
    assert "employees.csv" not in m["files"] and "employees.csv" not in sources_in_db()


def test_corrupt_manifest_rebuilds_without_duplicates(store):
    sync()
    n = chunk_count()
    open(ingest.MANIFEST_PATH, "w").write("{trunc")
    sync()
    assert chunk_count() == n


def test_malformed_manifest_rebuilds(store):
    sync()
    n = chunk_count()
    json.dump({"files": []}, open(ingest.MANIFEST_PATH, "w"))
    m = sync()
    assert isinstance(m["files"], dict) and chunk_count() == n


def test_lost_manifest_also_drops_chunks_of_files_deleted_meanwhile(store):
    sync()
    os.remove(ingest.MANIFEST_PATH)
    (store.src / "employees.csv").unlink()
    sync()
    assert sources_in_db() == {"project_aurora.md"}


def test_manifest_without_settings_is_rebuilt_not_trusted(store):
    sync()
    m = manifest_on_disk()
    del m["index_settings"]
    json.dump(m, open(ingest.MANIFEST_PATH, "w"))
    ingest.get_db()._collection.add(ids=["stray#0"], embeddings=[[0.125] * 64], documents=["stray"], metadatas=[{"source": "stray.md"}])
    sync()
    assert "stray.md" not in sources_in_db()


def test_chunk_settings_change_rebuilds_index(store):
    sync()
    n = chunk_count()
    store.cfg.update(chunk_size=120, chunk_overlap=20)
    m = sync()
    assert m["index_settings"]["chunk_size"] == 120
    assert chunk_count() == m["total_chunks"] > n


def test_rebuild_is_deferred_while_embeddings_are_unavailable(store):
    sync()
    n, before = chunk_count(), manifest_on_disk()
    store.cfg.update(chunk_size=120, chunk_overlap=20)
    store.embeddings.broken = True
    m = sync()
    assert chunk_count() == n, "existing index must stay searchable"
    assert "(index rebuild)" in m["failed"]
    assert manifest_on_disk() == before, "manifest kept so the rebuild is retried"
    store.embeddings.broken = False
    assert sync()["index_settings"]["chunk_size"] == 120


def test_parse_failure_is_reported_and_retried(store):
    sync()
    (store.src / "broken.pdf").write_bytes(b"not a pdf")
    m = sync()
    assert "broken.pdf" in m["failed"] and "broken.pdf" not in m["files"]
    (store.src / "broken.pdf").unlink()
    assert not sync()["failed"]


def test_failed_delete_keeps_old_entry_for_retry(store):
    sync()
    db = ingest.get_db()
    old = manifest_on_disk()["files"]["project_aurora.md"]
    (store.src / "project_aurora.md").write_text("Rewritten.\n")
    (store.src / "employees.csv").unlink()

    def failing_delete(*args, **kwargs):
        raise RuntimeError("delete failed")
    # Patched and restored by hand: monkeypatch.undo() would also undo the
    # store fixture and point ingest back at the configured (real) store
    real_delete = db.delete
    db.delete = failing_delete
    try:
        m = sync()
    finally:
        db.delete = real_delete
    assert m["files"]["project_aurora.md"] == old and "project_aurora.md" in m["failed"]
    assert "employees.csv" in m["files"] and "employees.csv" in m["failed"]

    m = sync()
    assert not m["failed"] and "employees.csv" not in sources_in_db()
    assert m["files"]["project_aurora.md"] != old


def test_overlapping_syncs_agree(store):
    sync()
    (store.src / "new.md").write_text("Widgets are blue.\n")
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(sync) for _ in range(2)]
    for future in futures:
        future.result()  # re-raises if either sync failed
    assert manifest_on_disk()["total_chunks"] == chunk_count()
    assert "new.md" in manifest_on_disk()["files"]


def test_watcher_syncs_once_after_a_burst_settles(monkeypatch):
    monkeypatch.setattr(ingest, "WATCHER_QUIET_SECONDS", 0.3)
    calls = []
    handler = ingest.SourceFolderHandler(lambda: calls.append(time.monotonic()))

    class Event:
        is_directory = False
        src_path = "report.pdf"
    start = time.monotonic()
    for _ in range(5):
        handler.on_any_event(Event())
        time.sleep(0.1)
    time.sleep(0.6)
    assert len(calls) == 1
    assert calls[0] - start >= 0.4 + 0.3  # after the last event, not the first


def test_watcher_ignores_unsupported_files(monkeypatch):
    monkeypatch.setattr(ingest, "WATCHER_QUIET_SECONDS", 0.05)
    calls = []
    handler = ingest.SourceFolderHandler(lambda: calls.append(1))

    class Event:
        is_directory = False
        src_path = "photo.jpg"
    handler.on_any_event(Event())
    time.sleep(0.2)
    assert calls == []
