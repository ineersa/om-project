import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from om_dataset import collection
from om_dataset.cli import main


def pi_session(path, with_memory=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = [
        {"type": "session", "id": "session", "version": 3, "cwd": "/synthetic"},
        {"type": "message", "id": "a", "parentId": None,
         "message": {"role": "user", "content": "Keep approval separate."}},
    ]
    if with_memory:
        entries.append({"type": "custom", "id": "b", "parentId": "a",
                        "customType": "om.observations.recorded", "data": {
                            "coversUpToId": "a", "observations": [{
                                "id": "abcdef123456", "content": "Separate approval is required.",
                                "timestamp": "2026-10-13 09:00", "relevance": "critical",
                                "sourceEntryIds": ["a"], "tokenCount": 8}]}})
    path.write_text("".join(json.dumps(e) + "\n" for e in entries))
    return path


def test_private_collection_round_trip_retains_original_bytes(tmp_path):
    source = pi_session(tmp_path / "live" / "session.jsonl")
    original = source.read_bytes()
    sources = collection.discover(source.parent, [])
    output = tmp_path / "dataset"
    result = collection.collect(sources, output)
    assert source.read_bytes() == original
    assert result["session_files"] == 1
    assert result["memory_candidates"] == 1
    assert result["structurally_valid_candidates"] == 1
    assert result["training_ready"] is False
    assert collection.validate_collection(output)["status"] == "pass"
    row = json.loads((output / "memory-candidates.jsonl").read_text())
    assert row["observation"]["source_refs"] == [{"run_id": "pi:session", "seq": 2}]
    assert row["candidate_status"] == "unreviewed"
    assert (output / row["source_archive_ref"]["path"]).read_bytes() == original
    assert output.stat().st_mode & 0o777 == 0o700
    assert (output / "memory-candidates.jsonl").stat().st_mode & 0o777 == 0o600
    assert (output / row["source_archive_ref"]["path"]).stat().st_mode & 0o777 == 0o400


def test_collect_all_sessions_without_inventing_no_op(tmp_path):
    source = pi_session(tmp_path / "live" / "session.jsonl", False)
    result = collection.collect(collection.discover(source.parent, []), tmp_path / "dataset")
    assert result["session_files"] == 1
    assert result["memory_candidates"] == 0
    parsed = json.loads(next((tmp_path / "dataset/parsed").glob("*.json")).read_text())
    assert parsed["stats"]["memory_outcome"] == "unknown"


def test_malformed_line_preserved_and_reported(tmp_path):
    source = pi_session(tmp_path / "live" / "session.jsonl", False)
    with source.open("ab") as stream:
        stream.write(b"not-json\n")
    original = source.read_bytes()
    output = tmp_path / "dataset"
    result = collection.collect(collection.discover(source.parent, []), output)
    assert result["issues_by_code"]["malformed_json"] == 1
    assert next((output / "archives").glob("*.jsonl")).read_bytes() == original


def test_bad_candidate_still_collected(tmp_path):
    source = pi_session(tmp_path / "live" / "session.jsonl")
    text = source.read_text().replace('"sourceEntryIds": ["a"]', '"sourceEntryIds": ["missing"]')
    source.write_text(text)
    result = collection.collect(collection.discover(source.parent, []), tmp_path / "dataset")
    assert result["memory_candidates"] == 1
    assert result["invalid_candidates"] == 1


def test_sqlite_backup_sees_committed_wal_and_does_not_migrate(tmp_path):
    source = tmp_path / "live.sqlite"
    with closing(sqlite3.connect(source)) as live:
        live.execute("PRAGMA journal_mode = WAL")
        live.execute("CREATE TABLE state (value TEXT)")
        live.execute("INSERT INTO state VALUES ('in WAL')")
        live.commit()
        before = source.read_bytes()
        wal_before = source.with_name(source.name + "-wal").read_bytes()
        output = tmp_path / "dataset"
        (output / "archives").mkdir(parents=True)
        ref, capture = collection.snapshot_database(source, output)
        with closing(sqlite3.connect((output / ref["path"]).as_uri() + "?mode=ro", uri=True)) as snapshot:
            assert snapshot.execute("SELECT value FROM state").fetchall() == [("in WAL",)]
        assert source.read_bytes() == before
        assert source.with_name(source.name + "-wal").read_bytes() == wal_before
        assert capture["journal_and_database_atomic"] is False


def test_snapshot_bounded_before_appends(tmp_path, monkeypatch):
    source = tmp_path / "live.jsonl"
    source.write_bytes(b"first\n")
    original_fstat = collection.os.fstat
    appended = False

    def append_after_stat(fd):
        nonlocal appended
        result = original_fstat(fd)
        if not appended:
            with source.open("ab") as stream:
                stream.write(b"second\n")
            appended = True
        return result

    monkeypatch.setattr(collection.os, "fstat", append_after_stat)
    output = tmp_path / "dataset"
    (output / "archives").mkdir(parents=True)
    ref, capture = collection.snapshot_file(source, output)
    assert (output / ref["path"]).read_bytes() == b"first\n"
    assert capture["source_changed_during_capture"] is True
    assert source.read_bytes() == b"first\nsecond\n"


def test_refuse_existing_output_or_output_inside_source(tmp_path):
    source = pi_session(tmp_path / "live" / "session.jsonl")
    sources = collection.discover(source.parent, [])
    existing = tmp_path / "dataset"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        collection.collect(sources, existing)
    with pytest.raises(ValueError, match="outside live"):
        collection.collect(sources, source.parent / "output")


def test_resume_preserves_completed_job_snapshot(tmp_path, monkeypatch):
    root = tmp_path / "live"
    first = pi_session(root / "a.jsonl")
    second = pi_session(root / "b.jsonl", False)
    sources = collection.discover(root, [])
    real_parser = collection.parse_pi
    calls = 0

    def interrupt(path):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic interruption")
        return real_parser(path)

    monkeypatch.setattr(collection, "parse_pi", interrupt)
    output = tmp_path / "dataset"
    with pytest.raises(RuntimeError):
        collection.collect(sources, output)
    assert not (output / "manifest.json").exists()
    first.write_bytes(b"changed live session\n")
    monkeypatch.setattr(collection, "parse_pi", real_parser)
    report = collection.collect(sources, output, resume=True)
    assert report["session_files"] == 2
    assert report["memory_candidates"] == 1
    assert collection.validate_collection(output)["status"] == "pass"
    with pytest.raises(ValueError, match="incomplete collection"):
        collection.collect(sources, output, resume=True)


def test_reject_tampered_artifact_and_path_escape(tmp_path):
    source = pi_session(tmp_path / "live" / "session.jsonl")
    output = tmp_path / "dataset"
    collection.collect(collection.discover(source.parent, []), output)
    path = output / "memory-candidates.jsonl"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="integrity"):
        collection.validate_collection(output)
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"{}")
    with pytest.raises(ValueError, match="escapes"):
        collection.resolve_artifact(output, {"path": "../outside.json", "bytes": 2,
                                             "sha256": hashlib.sha256(b"{}").hexdigest()})


def test_inventory_cli_and_failure_do_not_print_session_text(tmp_path, capsys):
    source = pi_session(tmp_path / "live" / "session.jsonl")
    assert main(["inventory", "--pi-root", str(source.parent)]) == 0
    assert "Keep approval separate" not in capsys.readouterr().out
    assert main(["collect", "--pi-root", str(source.parent), "--output", str(source.parent)]) == 1
    output = capsys.readouterr()
    assert "Collection failed" in output.err
    assert "Keep approval separate" not in output.err


def test_discovery_pairs_hatfield_journals_and_om_store(tmp_path):
    project = tmp_path / "project"
    store = project / ".hatfield/extensions-data/observational-memory"
    store.mkdir(parents=True)
    (store / "om.sqlite").write_bytes(b"")
    journal = project / ".hatfield/sessions/run/events.jsonl"
    journal.parent.mkdir(parents=True)
    journal.write_text("{}\n")
    sources = collection.discover(None, [store])
    assert [s["kind"] for s in sources] == ["hatfield", "hatfield_db"]
    assert all(s["om_dir"] == str(store) for s in sources)


def test_resume_captured_source_without_live_file_and_cleans_finalization(tmp_path, monkeypatch, capsys):
    source = pi_session(tmp_path / "live" / "a.jsonl")
    sources = collection.discover(source.parent, [])
    real_parser = collection.parse_pi

    def interrupt(path):
        raise RuntimeError("synthetic interruption after capture")

    monkeypatch.setattr(collection, "parse_pi", interrupt)
    output = tmp_path / "dataset"
    with pytest.raises(RuntimeError):
        collection.collect(sources, output)
    source.unlink()
    source.parent.rmdir()
    (output / "report.json.partial").write_text("interrupted")
    (output / "manifest.json.partial").write_text("interrupted")
    monkeypatch.setattr(collection, "parse_pi", real_parser)
    assert main(["collect", "--resume", "--output", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["memory_candidates"] == 1
    assert collection.validate_collection(output)["status"] == "pass"


def test_resume_ignores_new_live_sessions(tmp_path, monkeypatch, capsys):
    source = pi_session(tmp_path / "live" / "a.jsonl")
    sources = collection.discover(source.parent, [])
    real_parser = collection.parse_pi

    def interrupt(path):
        raise RuntimeError("synthetic interruption")

    monkeypatch.setattr(collection, "parse_pi", interrupt)
    output = tmp_path / "dataset"
    with pytest.raises(RuntimeError):
        collection.collect(sources, output)
    pi_session(source.parent / "new.jsonl")
    monkeypatch.setattr(collection, "parse_pi", real_parser)
    assert main(["collect", "--resume", "--output", str(output), "--pi-root", str(source.parent)]) == 0
    assert json.loads(capsys.readouterr().out)["session_files"] == 1


def test_invalid_numeric_or_unicode_pi_records_remain_in_archive(tmp_path):
    source = pi_session(tmp_path / "live" / "session.jsonl", False)
    with source.open("ab") as stream:
        stream.write(b'{"type":"custom","id":"bad","data":NaN}\n')
        stream.write(b'{"type":"message","id":"unicode","message":{"content":"\\ud800"}}\n')
    output = tmp_path / "dataset"
    result = collection.collect(collection.discover(source.parent, []), output)
    assert result["issues_by_code"]["malformed_json"] == 2
    assert next((output / "archives").glob("*.jsonl")).read_bytes() == source.read_bytes()
    assert collection.validate_collection(output)["status"] == "pass"


def test_normalization_uses_frozen_sources_and_preserves_prior_release(tmp_path):
    source = pi_session(tmp_path / "live" / "session.jsonl")
    output = tmp_path / "dataset"
    collection.collect(collection.discover(source.parent, []), output)
    original_manifest = (output / "manifest.json").read_bytes()
    source.unlink()
    normalized = tmp_path / "normalized"
    result = collection.normalize_collection(output, normalized)
    assert result["session_files"] == 1
    assert result["memory_candidates"] == 1
    assert (output / "manifest.json").read_bytes() == original_manifest
    config = json.loads((normalized / "collection-config.json").read_text())
    assert config["sources"][0]["upstream_source"]["path"] == str(source)
    assert config["sources"][0]["upstream_manifest_sha256"] == collection.file_hash(output / "manifest.json")
    assert collection.validate_collection(normalized)["status"] == "pass"
