"""Prepare two small chronological Observer replays from clean frozen ranges."""
from collections import defaultdict
from datetime import datetime
import json
import os
from pathlib import Path
import secrets

from .collection import canonical, file_hash, resolve_artifact, write_json
from .hatfield_source import build_blocks
from .ranges import _entries, match_hatfield, pi_range
from .redaction import Redactor
from .teacher import JOBS_PATH, OUTPUT_PATH, POLICIES_PATH, PROVIDER, PROVIDERS, REQUEST_OPTIONS, STUDENT_CONTEXT_WINDOW, slice_job

RANGES = Path("data/ranges-2026-10-13")
CORPUS = Path("data/collection-2026-10-13-v2")


def _time(value):
    return datetime.fromisoformat(value).strftime("%Y-%m-%d %H:%M")


def _pi_source(line, entry):
    message = entry.get("message", {})
    content = message.get("content", entry.get("content", entry.get("summary", "")))
    if isinstance(content, list):
        parts = []
        for part in content:
            if part.get("type") == "thinking":
                continue
            if part.get("type") == "text":
                parts.append(part["text"])
            elif part.get("type") == "toolCall":
                parts.append("Tool call " + part["name"] + " [" + part["id"] + "]:\n" +
                             json.dumps(part["arguments"], ensure_ascii=False))
            else:
                raise ValueError("Pilot selection requires text/tool-only sources")
        content = "\n".join(parts)
    if not isinstance(content, str):
        raise ValueError("Unsupported Pi source content")
    role = {"user": "User", "toolResult": "Tool result"}.get(message.get("role"), "Assistant")
    if entry["type"] == "branch_summary":
        content = "Branch summary:\n" + content
    return {"seq": line, "role": role, "timestamp": _time(entry["timestamp"]), "text": content,
            **({"tool_name": message["toolName"]} if role == "Tool result" else {})}


def _jobs(rows):
    first = rows[0]
    entries = _entries(resolve_artifact(CORPUS, first["source_archive_ref"]))
    by_line = dict(entries)
    by_id = {entry.get("id"): entry for _, entry in entries}
    previous_marker = None
    if first["source_kind"] == "hatfield":
        events = list(by_line.values())
        global_blocks = {ratio: build_blocks(events, ratio) for ratio in (2.35, 3.25)}
        timestamps = {e["seq"]: _time(e["ts"]) for e in events}
    jobs = []
    for row in rows:
        run = ("pi:" if row["source_kind"] == "pi" else "") + row["session_id"]
        job = {"run_id": run, "source_kind": row["source_kind"], "historical_range_id": row["range_id"],
               "source_archive_ref": row["source_archive_ref"],
               "cutoff_seq": max(ref["seq"] for ref in row["source_refs"]),
               "state_scope": row["native_identity"].get("store_sha256", first["source_archive_ref"]["sha256"]),
               "prior_memory_status": "fresh_replay_empty_not_historical",
               "fallback_time_status": "recording_time_not_original_clock"}
        if row["source_kind"] == "pi":
            marker = by_line[row["native_identity"]["line_no"]]
            if previous_marker is not None:
                parent, visited = marker.get("parentId"), set()
                while parent is not None and parent != previous_marker and parent not in visited:
                    visited.add(parent)
                    parent = by_id.get(parent, {}).get("parentId")
                if parent != previous_marker:
                    raise ValueError("Pi replay markers must share an ancestry chain")
            previous_marker = marker["id"]
            bounds = pi_range(entries, row["native_identity"]["line_no"])
            if bounds["source_seqs"] != [r["seq"] for r in row["source_refs"]]:
                raise ValueError("Pi source reconstruction changed")
            job.update(local_time=_time(by_line[row["native_identity"]["line_no"]]["timestamp"]),
                       sources=[_pi_source(ref["seq"], by_line[ref["seq"]]) for ref in row["source_refs"]],
                       renderer="pi-visible-text-and-tool-calls/1")
        else:
            match = match_hatfield(row["coverage"], events, global_blocks)
            if match is None:
                raise ValueError("Hatfield source reconstruction changed")
            blocks, _ = match
            job.update(local_time=_time(row["coverage"]["covered_at"]),
                       historical_source_digest=row["coverage"]["source_digest"],
                       historical_part_digest=row["coverage"]["part_digest"],
                       sources=[{"seq": b["seq"], "timestamp": timestamps[b["seq"]],
                                 "role": {"user": "User", "tool": "Tool result"}.get(b["segments"][0]["role"], "Assistant"),
                                 "text": b["rendered_text"], "segments": b["segments"],
                                 "source_refs": b["source_refs"], "native_rendered": True} for b in blocks])
        jobs.append(job)
    return jobs


def prepare():
    rows = [json.loads(line) for line in (RANGES / "ranges.jsonl").open()]
    excluded = {(r["source_kind"], r["session_id"]) for line in (RANGES / "exclusions.jsonl").open()
                if (r := json.loads(line))}
    groups = defaultdict(list)
    for row in rows:
        if not row["issues"] and row["range_replay_eligible"]:
            groups[(row["source_kind"], row["session_id"], row["source_archive_ref"]["sha256"])].append(row)
    redactor = Redactor()
    if not redactor.catalog_loaded:
        raise ValueError("Private credential catalog is required")
    selected, scopes = [], []
    for kind in ("pi", "hatfield"):
        candidates = []
        for (source_kind, run, _), group in groups.items():
            if source_kind != kind or (kind, run) in excluded or not 2 <= len(group) <= 4:
                continue
            if kind == "pi":
                group.sort(key=lambda r: r["source_end_seq"])
                if group[0]["after_entry_id"] is not None or any(b["after_entry_id"] != a["covers_up_to_id"] for a, b in zip(group, group[1:])):
                    continue
            else:
                group.sort(key=lambda r: r["coverage"]["source_end_seq"])
                if group[0]["coverage"]["source_start_seq"] != 1 or any(r["source_verification"]["method"] != "interval" for r in group):
                    continue
                if any(b["coverage"]["source_start_seq"] != a["coverage"]["source_end_seq"] + 1 for a, b in zip(group, group[1:])):
                    continue
            candidates.append(group)
        candidates.sort(key=lambda g: (sum(len(r["source_refs"]) for r in g), g[0]["session_id"]))
        for group in candidates:
            try:
                jobs = [redactor.job(job) for job in _jobs(group)]
            except ValueError:
                continue
            if any(b["cutoff_seq"] <= a["cutoff_seq"] for a, b in zip(jobs, jobs[1:])):
                continue
            if sum(len(s["text"]) for j in jobs for s in j["sources"]) > 120_000:
                continue
            selected.extend(jobs)
            scopes.append({"source_kind": kind, "run_id": jobs[0]["run_id"], "ranges": len(jobs),
                           "state_scope": jobs[0]["state_scope"], "source_archive_ref": group[0]["source_archive_ref"]})
            break
        else:
            raise ValueError("No small clean chronological session found for " + kind)
    bundle = json.loads((POLICIES_PATH / "hatfield-target/definitions.json").read_text())
    policy = next(a for a in bundle["agents"] if a["role"] == "observer")
    config = {**PROVIDERS[PROVIDER], "provider": PROVIDER, "request_options": REQUEST_OPTIONS,
              "context_window": STUDENT_CONTEXT_WINDOW}
    chunks = sum(len(slice_job(job, policy, config)) for job in selected)
    JOBS_PATH.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(JOBS_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as destination:
        for job in selected:
            redactor.require_clean(job)
            destination.write(canonical(job) + b"\n")
        destination.flush()
        os.fsync(destination.fileno())
    report = {"sessions": scopes, "jobs": len(selected), "initial_chunks": chunks, "model_calls": 0,
              "redaction_id": redactor.fingerprint, "jobs_sha256": file_hash(JOBS_PATH),
              "ranges_manifest_sha256": file_hash(RANGES / "manifest.json"),
              "scope": "All retained historical Observer ranges for two small clean sessions; uncovered tails are not invented."}
    write_json(JOBS_PATH.with_suffix(".manifest.json"), report)
    return {"sessions": len(scopes), "jobs": len(selected), "initial_chunks": chunks, "model_calls": 0}


def prepare_review():
    """Export fixed blinded content comparisons, not an identical-input benchmark."""
    output, private = Path("data/review/cases.jsonl"), Path("data/review-private/identity-map.json")
    if output.exists() or private.exists():
        raise FileExistsError("Review pack already exists; never reshuffle an ongoing review")
    attempts = [json.loads(line) for line in OUTPUT_PATH.open()]
    if not attempts or len({r["job_id"] for r in attempts}) != len(attempts) or any(r["status"] != "completed" for r in attempts):
        raise ValueError("Review export requires unique completed attempts")
    ranges = {r["range_id"]: r for line in (RANGES / "ranges.jsonl").open() if (r := json.loads(line))}
    wanted = {cid for result in attempts for cid in ranges[result["job"]["historical_range_id"]]["candidate_ids"]}
    original = {}
    for line in Path("data/curated-2026-10-13-v2/memory-candidates.jsonl").open():
        candidate = json.loads(line)
        if candidate["candidate_id"] in wanted:
            original[candidate["candidate_id"]] = candidate["observation"]
    if set(original) != wanted:
        raise ValueError("Historical candidates are missing")
    redactor, cases, mappings = Redactor(), [], []

    def display(observations):
        return "\n\n".join(f"[{o['relevance']}] {o['timestamp']}\n{o['content']}\nEvidence: " +
                            ", ".join(f"{r['run_id']}:{r['seq']}" for r in o["source_refs"]) for o in observations)

    def evidence_text(source):
        if source.get("native_rendered"):
            return source["text"]
        role = source["role"] + (" for " + source["tool_name"] if source["role"] == "Tool result" else "")
        return f"[{role} @ {source['timestamp']}]:\n{source['text']}"

    for number, result in enumerate(attempts, 1):
        redactor.require_clean(result)
        job = result["job"]
        if job["chunk_count"] != 1:
            raise ValueError("Subdivided historical ranges require explicit baseline alignment")
        baseline = [original[cid] for cid in ranges[job["historical_range_id"]]["candidate_ids"]]
        choices = [("historical_original", display(baseline)), ("fresh_teacher", display(result["observations"]))]
        if secrets.randbelow(2):
            choices.reverse()
        case = {"schema_version": "om-review-case/1", "case_id": f"case-{number:03d}", "task": "observer",
                "redaction_id": redactor.fingerprint,
                "comparison_scope": "Source-grounded content comparison, not an identical-input model benchmark. "
                    "Matched prior context is unavailable. Do not assume fewer observations imply omissions or duplication; skip if missing context prevents a decision.",
                "prior_memory": "Matched prior memory for both candidates was not recovered.",
                "evidence": [{"source_id": f"{job['run_id']}:{s['seq']}", "text": evidence_text(s),
                              "source_refs": s.get("source_refs", [{"run_id": job["run_id"], "seq": s["seq"]}])} for s in job["sources"]],
                "candidates": {label: choice[1] for label, choice in zip(("A", "B"), choices)}}
        cases.append(redactor.tree(case))
        mappings.append({"case_id": case["case_id"], "historical_range_id": job["historical_range_id"],
                         "teacher_job_id": result["job_id"], "A": choices[0][0], "B": choices[1][0]})
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    private.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Persist the mapping first. An interrupted preparation must not silently reshuffle.
    write_json(private, {"schema_version": "om-review-map/1", "cases": mappings,
                        "teacher_config": attempts[0]["config"], "historical_model_identity": "not_verified"})
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as destination:
        for case in cases:
            redactor.require_clean(case)
            destination.write(canonical(case) + b"\n")
        destination.flush()
        os.fsync(destination.fileno())
    return {"cases": len(cases), "historical_observations": len(original),
            "fresh_observations": sum(len(r["observations"]) for r in attempts), "model_calls": 0}


if __name__ == "__main__":
    print(json.dumps(prepare()))
