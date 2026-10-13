"""Snapshot native histories and build a private, unlabeled corpus."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
import time
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .adapters.hatfield import parse_hatfield, parse_hatfield_db
from .adapters.pi import parse_pi

SCHEMA_VERSION = "om-corpus/1"
TARGET_REVISION = "5ea2fd920d04eccee75094afe033302c5596393c"


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("xb") as stream:
        os.chmod(temporary, 0o600)
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def artifact(root: Path, path: Path) -> dict:
    return {"path": str(path.relative_to(root)), "sha256": file_hash(path),
            "bytes": path.stat().st_size}


def resolve_artifact(root: Path, ref: dict) -> Path:
    if Path(ref["path"]).is_absolute():
        raise ValueError("Absolute artifact path")
    path = (root / ref["path"]).resolve(strict=True)
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("Artifact escapes collection")
    if path.stat().st_size != ref["bytes"] or file_hash(path) != ref["sha256"]:
        raise ValueError("Artifact integrity failure: " + ref["path"])
    return path


def snapshot_file(source: Path, root: Path) -> tuple[dict, dict]:
    """Copy a bounded byte prefix, including malformed or incomplete final lines."""
    temporary = root / "archives" / "capture.partial"
    with source.open("rb") as origin, temporary.open("xb") as destination:
        os.chmod(temporary, 0o600)
        before = os.fstat(origin.fileno())
        remaining = before.st_size
        while remaining:
            block = origin.read(min(remaining, 1024 * 1024))
            if not block:
                break
            destination.write(block)
            remaining -= len(block)
        destination.flush()
        os.fsync(destination.fileno())
        after = os.fstat(origin.fileno())
    digest = file_hash(temporary)
    final = root / "archives" / (digest + source.suffix)
    if final.exists():
        if file_hash(final) != digest:
            raise ValueError("Existing archive hash mismatch")
        temporary.unlink()
    else:
        temporary.replace(final)
    os.chmod(final, 0o400)
    metadata = {"source_path": str(source), "captured_at": now(),
                "source_size_before": before.st_size, "source_size_after": after.st_size,
                "source_mtime_ns_before": before.st_mtime_ns,
                "source_mtime_ns_after": after.st_mtime_ns,
                "source_changed_during_capture": (before.st_size, before.st_mtime_ns) !=
                                                 (after.st_size, after.st_mtime_ns),
                "source_shrank_during_capture": remaining != 0,
                "captured_bytes": final.stat().st_size,
                "capture_boundary": "initial_byte_length"}
    return artifact(root, final), metadata


def snapshot_database(source: Path, root: Path, timeout: float = 60) -> tuple[dict, dict]:
    """SQLite backup includes committed WAL data without running migrations."""
    temporary = root / "archives" / "database.partial"
    before = source.stat()
    start = time.monotonic()

    def progress(status, remaining, total):
        if time.monotonic() - start > timeout:
            raise TimeoutError("SQLite backup exceeded deadline")

    # Do not use immutable=1 here: that would ignore live WAL transactions.
    with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True, timeout=10)) as origin:
        origin.execute("PRAGMA query_only = ON")
        with closing(sqlite3.connect(temporary)) as destination:
            os.chmod(temporary, 0o600)
            origin.backup(destination, pages=256, progress=progress, sleep=0.01)
            result = destination.execute("PRAGMA quick_check").fetchall()
            if result != [("ok",)]:
                raise ValueError("SQLite snapshot failed quick_check")
    digest = file_hash(temporary)
    final = root / "archives" / (digest + ".sqlite")
    if final.exists():
        if file_hash(final) != digest:
            raise ValueError("Existing database archive hash mismatch")
        temporary.unlink()
    else:
        temporary.replace(final)
    os.chmod(final, 0o400)
    after = source.stat()
    return artifact(root, final), {"source_path": str(source), "captured_at": now(),
                                  "capture_method": "sqlite_backup_api",
                                  "source_size_before": before.st_size,
                                  "source_size_after": after.st_size,
                                  "journal_and_database_atomic": False}


def discover(pi_root: Path | None, hatfield_om_dirs: list[Path],
             hatfield_session_roots: list[Path] | None = None) -> list[dict]:
    """Explicit sources only. Do not search unrelated projects or auth files."""
    roots = hatfield_session_roots or []
    if roots and len(roots) != len(hatfield_om_dirs):
        raise ValueError("Provide one session root per Hatfield OM directory")
    sources = []
    if pi_root is not None:
        pi_root = pi_root.expanduser().resolve(strict=True)
        if not pi_root.is_dir():
            raise ValueError("Pi root must be a directory")
        sources.extend({"kind": "pi", "path": str(p.resolve(strict=True))}
                       for p in sorted(pi_root.rglob("*.jsonl")) if p.is_file())
    for i, om in enumerate(hatfield_om_dirs):
        om = om.expanduser().resolve(strict=True)
        if not om.is_dir():
            raise ValueError("Hatfield OM path must be a directory")
        if roots:
            sessions = roots[i].expanduser().resolve(strict=True)
        elif om.parent.name == "extensions-data" and om.parent.parent.name == ".hatfield":
            sessions = om.parent.parent / "sessions"
        else:
            raise ValueError("Nonstandard OM layout requires explicit session roots")
        if not sessions.is_dir():
            raise ValueError("Hatfield session root is missing; provide --hatfield-session-root")
        sources.extend({"kind": "hatfield", "path": str(p.resolve(strict=True)),
                        "om_dir": str(om)} for p in sorted(sessions.rglob("events.jsonl")))
        databases = sorted(om.glob("*.sqlite"))
        if not databases:
            raise ValueError("No SQLite OM store found in " + str(om))
        sources.extend({"kind": "hatfield_db", "path": str(p.resolve(strict=True)),
                        "om_dir": str(om)} for p in databases)
    if not sources:
        raise ValueError("No source files discovered")
    deduplicated = {s["path"]: s for s in sources}
    if len(deduplicated) != len(sources):
        raise ValueError("Overlapping source roots")
    return sorted(sources, key=lambda s: (s["kind"] == "hatfield_db", s["kind"], s["path"]))


def inventory(sources: list[dict]) -> dict:
    counts, sizes = Counter(), Counter()
    for source in sources:
        counts[source["kind"]] += 1
        sizes[source["kind"]] += Path(source["path"]).stat().st_size
    return {"source_files": dict(counts), "source_bytes": dict(sizes),
            "sqlite_bytes_exclude_live_wal": True, "model_calls": 0}


def collector_files() -> list[Path]:
    package = Path(__file__).parent
    project = package.parent
    return sorted(package.rglob("*.py")) + [package / "corpus-manifest.schema.json",
                                            project / "pyproject.toml", project / "uv.lock"]


def collect(sources: list[dict], output: Path, resume: bool = False) -> dict:
    output = output.expanduser().absolute()
    source_paths = [Path(s["path"]).resolve() for s in sources]
    code_files = collector_files()
    project = Path(__file__).parent.parent
    fingerprint = {str(p.relative_to(project)): file_hash(p) for p in code_files}
    if any(output.resolve() == p or output.resolve().is_relative_to(p.parent) for p in source_paths):
        raise ValueError("Collection output must be outside live source directories")
    if resume:
        if not output.is_dir() or (output / "manifest.json").exists():
            raise ValueError("Resume requires an interrupted, incomplete collection")
        configuration = json.loads((output / "collection-config.json").read_text())
        if configuration["sources"] != sources:
            raise ValueError("Resume source inventory differs; start a new collection")
        if configuration["collector_fingerprint"] != fingerprint:
            raise ValueError("Collector changed; do not mix parser versions during resume")
        # Only interrupted staging files are disposable. Complete archives remain unchanged.
        for directory in (output / "archives", output / "parsed", output / "jobs"):
            for p in directory.glob("*.partial"):
                p.unlink()
        for name in ("report.json.partial", "manifest.json.partial"):
            (output / name).unlink(missing_ok=True)
    else:
        output.mkdir(mode=0o700, parents=True, exist_ok=False)
        for directory in ("archives", "parsed", "jobs", "collector"):
            (output / directory).mkdir(mode=0o700)
        for source_file in code_files:
            copy = output / "collector" / source_file.relative_to(project)
            copy.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            copy.write_bytes(source_file.read_bytes())
            os.chmod(copy, 0o400)
        write_json(output / "collection-config.json", {"sources": sources, "started_at": now(),
                                                      "collector_fingerprint": fingerprint,
                                                      "python_version": sys.version})
    os.chmod(output, 0o700)
    artifacts = [artifact(output, output / "collection-config.json")]
    for source_file in code_files:
        copy = output / "collector" / source_file.relative_to(project)
        ref = artifact(output, copy)
        if ref["sha256"] != fingerprint[str(source_file.relative_to(project))]:
            raise ValueError("Collector snapshot integrity failure")
        artifacts.append(ref)
    candidates_by_task, adapters, evidence, issues = Counter(), Counter(), Counter(), Counter()
    session_count, entries, candidate_count, valid_count = 0, 0, 0, 0
    archive_counts = Counter()
    source_counts, source_sizes, archive_sizes = Counter(), Counter(), Counter()
    source_maps: dict[str, dict[str, set[int]]] = {}
    session_runs: dict[str, set[str]] = {}
    session_output = output / "sessions.jsonl"
    candidate_output = output / "memory-candidates.jsonl"
    issue_output = output / "issues.jsonl"
    with session_output.open("wb") as sessions, candidate_output.open("wb") as memory, issue_output.open("wb") as errors:
        for path in (session_output, candidate_output, issue_output):
            os.chmod(path, 0o600)
        for source in sources:
            kind = source["kind"]
            key = hashlib.sha256(canonical(source)).hexdigest()
            job_path = output / "jobs" / (key + ".json")
            if job_path.exists():
                job = json.loads(job_path.read_text())
                snapshot = resolve_artifact(output, job["archive_ref"])
            else:
                if kind == "hatfield_db":
                    archive_ref, capture = snapshot_database(Path(source["path"]), output)
                else:
                    archive_ref, capture = snapshot_file(Path(source["path"]), output)
                snapshot = output / archive_ref["path"]
                job = {"source": source, "archive_ref": archive_ref, "capture": capture}
                # Persist the source boundary before parsing. Resume must never
                # recapture a failed parse against changed live session data.
                write_json(job_path, job)
            if "parsed_ref" in job:
                parsed_path = resolve_artifact(output, job["parsed_ref"])
                parsed = json.loads(parsed_path.read_text())
            else:
                parsed = (parse_hatfield_db(snapshot, source_maps.get(source["om_dir"], {}))
                          if kind == "hatfield_db" else parse_pi(snapshot) if kind == "pi"
                          else parse_hatfield(snapshot))
                parsed_path = output / "parsed" / (key + ".json")
                write_json(parsed_path, parsed)
                job["parsed_ref"] = artifact(output, parsed_path)
                write_json(job_path, job)
            artifacts.extend([job["archive_ref"], job["parsed_ref"], artifact(output, job_path)])
            archive_counts[job["archive_ref"]["sha256"]] += 1
            source_counts[kind] += 1
            source_sizes[kind] += job["capture"]["source_size_before"]
            archive_sizes[kind] += job["archive_ref"]["bytes"]
            if kind == "hatfield":
                group = source_maps.setdefault(source["om_dir"], {})
                runs = session_runs.setdefault(source["om_dir"], set())
                if parsed["session_id"]:
                    runs.add(parsed["session_id"])
                for entry in parsed["sources"]:
                    ref = entry.get("source_ref")
                    if ref and entry.get("valid"):
                        group.setdefault(ref["run_id"], set()).add(ref["seq"])
            if kind != "hatfield_db":
                session_count += 1
                entries += len(parsed["sources"])
                sessions.write(canonical({"source_kind": kind, "session_id": parsed["session_id"],
                                          "archive_ref": job["archive_ref"], "parsed_ref": job["parsed_ref"],
                                          "capture": job["capture"], "split": "unassigned",
                                          "reconstruction_class": "unclassified",
                                          "source_entry_count": len(parsed["sources"]),
                                          "memory_candidate_count": len(parsed["memory_candidates"]),
                                          "issue_count": len(parsed["issues"]),
                                          "historical_input_identity_established": False}) + b"\n")
            else:
                for run in parsed["run_ids"]:
                    if run not in session_runs.get(source["om_dir"], set()):
                        sessions.write(canonical({"source_kind": "hatfield_db_only", "session_id": run,
                                                  "archive_ref": job["archive_ref"], "parsed_ref": job["parsed_ref"],
                                                  "reconstruction_class": "summary_only", "split": "unassigned",
                                                  "historical_input_identity_established": False}) + b"\n")
            for candidate in parsed["memory_candidates"]:
                candidate_count += 1
                valid_count += bool(candidate["valid"])
                candidates_by_task[candidate["task"]] += 1
                adapters[candidate["adapter"]] += 1
                evidence[candidate["evidence_status"]] += 1
                native = candidate.get("native_record")
                candidate = {**candidate, "source_archive_ref": job["archive_ref"],
                             "source_kind": kind, "session_id": parsed.get("session_id") or
                             (native.get("run_id") if isinstance(native, dict) else None),
                             "candidate_status": "unreviewed", "split": "unassigned",
                             "historical_input_identity_established": False}
                candidate["candidate_id"] = hashlib.sha256(canonical(candidate)).hexdigest()
                memory.write(canonical(candidate) + b"\n")
            source_issues = list(parsed["issues"])
            if job["capture"].get("source_changed_during_capture"):
                source_issues.append({"code": "source_changed_during_capture"})
            if job["capture"].get("source_shrank_during_capture"):
                source_issues.append({"code": "source_shrank_during_capture"})
            for issue in source_issues:
                issues[issue["code"]] += 1
                errors.write(canonical({**issue, "source_kind": kind,
                                        "archive_ref": job["archive_ref"]}) + b"\n")
    report = {"source_files": dict(source_counts), "source_bytes_at_capture": dict(source_sizes),
              "snapshot_bytes": dict(archive_sizes), "sqlite_snapshot_includes_committed_wal": True,
              "model_calls": 0, "session_files": session_count,
              "source_entries": entries, "memory_candidates": candidate_count,
              "structurally_valid_candidates": valid_count,
              "invalid_candidates": candidate_count - valid_count,
              "candidates_by_task": dict(candidates_by_task), "candidates_by_adapter": dict(adapters),
              "evidence_status": dict(evidence), "issues_by_code": dict(issues),
              "exact_duplicate_archive_count": sum(n - 1 for n in archive_counts.values()),
              "independent_session_families": None, "token_counts": None,
              "split_status": "unassigned", "training_ready": False, "redacted": False}
    write_json(output / "report.json", report)
    artifacts.extend(artifact(output, p) for p in (session_output, candidate_output, issue_output,
                                                   output / "report.json"))
    unique = {ref["path"]: ref for ref in artifacts}
    manifest = {"schema_version": SCHEMA_VERSION, "dataset_kind": "private_source_corpus",
                "completed_at": now(), "training_ready": False, "redacted": False,
                "model_calls": 0, "target": {"runtime": "hatfield", "revision": TARGET_REVISION},
                "pi_seq_mapping": "1-based physical JSONL line number, including header and malformed lines",
                "source_visibility": "native ancestry and compaction require replay; no prompts claimed captured",
                "split_status": "unassigned", "artifacts": sorted(unique.values(), key=lambda r: r["path"]),
                "report_ref": artifact(output, output / "report.json")}
    write_json(output / "manifest.json", manifest)
    return report


def validate_collection(root: Path) -> dict:
    """Verify a completed portable collection without consulting live sources."""
    from jsonschema import Draft202012Validator, FormatChecker

    root = root.expanduser().resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_text())
    schema = json.loads((Path(__file__).parent / "corpus-manifest.schema.json").read_text())
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(manifest)
    for ref in manifest["artifacts"]:
        resolve_artifact(root, ref)
    report = json.loads(resolve_artifact(root, manifest["report_ref"]).read_text())
    total = 0
    with (root / "memory-candidates.jsonl").open("rb") as stream:
        for line in stream:
            candidate = json.loads(line)
            expected = candidate.pop("candidate_id")
            if expected != hashlib.sha256(canonical(candidate)).hexdigest():
                raise ValueError("Candidate identity mismatch")
            if candidate["candidate_status"] != "unreviewed" or candidate["split"] != "unassigned":
                raise ValueError("Collection must not claim label acceptance or frozen splits")
            total += 1
    if total != report["memory_candidates"]:
        raise ValueError("Candidate count mismatch")
    return {"status": "pass", "verified_artifacts": len(manifest["artifacts"]),
            "memory_candidates": total, "training_ready": False}


def normalize_collection(source_collection: Path, output: Path) -> dict:
    """Reparse frozen archives into a new release without touching live sessions."""
    source_collection = source_collection.expanduser().resolve(strict=True)
    validate_collection(source_collection)
    configuration = json.loads((source_collection / "collection-config.json").read_text())
    manifest_hash = file_hash(source_collection / "manifest.json")
    sources = []
    for source in configuration["sources"]:
        key = hashlib.sha256(canonical(source)).hexdigest()
        job = json.loads((source_collection / "jobs" / (key + ".json")).read_text())
        path = resolve_artifact(source_collection, job["archive_ref"])
        sources.append({**source, "path": str(path), "upstream_source": source,
                        "upstream_capture": job["capture"],
                        "upstream_manifest_sha256": manifest_hash})
    return collect(sources, output)
