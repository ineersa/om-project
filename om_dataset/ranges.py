"""Recover historical Observer boundaries from frozen archives, never live sessions."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime
from pathlib import Path

from .collection import artifact, canonical, file_hash, now, resolve_artifact, write_json
from .curation import EvidenceIndex, _rows
from .hatfield_source import build_blocks, part_digest, source_digest


HATFIELD_RENDERER_REVISION = "5ea2fd920d04eccee75094afe033302c5596393c"
ESTIMATOR_CHANGE_REVISION = "db3376f6507cbf575b7453dcf01e5fe3a78cc39b"


def _entries(path):
    result = []
    with path.open("rb") as stream:
        for line, raw in enumerate(stream, 1):
            try:
                entry = json.loads(raw)
                if isinstance(entry, dict):
                    result.append((line, entry))
            except (ValueError, UnicodeError):
                pass  # Archives retain malformed lines; missing IDs cannot resolve below.
    return result


def _renderable(entry):
    if entry["type"] == "branch_summary":
        return isinstance(entry.get("summary"), str)
    if entry["type"] == "custom_message":
        return True
    message = entry.get("message")
    if not isinstance(message, dict):
        return False
    if message.get("role") != "assistant":
        return True
    content = message.get("content")
    if isinstance(content, list):
        parts = [b["text"] if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str)
                 else "" if isinstance(b, dict) and b.get("type") == "thinking" and b.get("redacted") is True
                 else "[non-text placeholder]" for b in content]
        content = "\n".join(parts)
    return any(content.split("\n")) if isinstance(content, str) else True


def pi_range(entries, marker_line):
    """Use ancestry, not file adjacency; current bounds follow native progress.ts."""
    marker = next(entry for line, entry in entries if line == marker_line)
    counts = Counter(e.get("id") for _, e in entries if isinstance(e.get("id"), str))
    nodes = {e["id"]: (line, e) for line, e in entries if isinstance(e.get("id"), str)}
    branch, seen, node = [], set(), marker.get("id")
    while node is not None:
        if not isinstance(node, str) or counts[node] != 1 or node in seen or node not in nodes:
            return {"status": "unresolved", "issues": ["invalid_native_ancestry"]}
        line, entry = nodes[node]
        seen.add(node)
        branch.append((line, entry))
        parent = entry.get("parentId")
        if parent is not None and (not isinstance(parent, str) or parent not in nodes or nodes[parent][0] >= line):
            return {"status": "unresolved", "issues": ["invalid_native_ancestry"]}
        node = parent
    branch.reverse()
    positions = {e["id"]: i for i, (_, e) in enumerate(branch)}
    data = marker.get("data", {})
    end_id = data.get("coversUpToId")
    if end_id not in positions or positions[end_id] >= len(branch) - 1:
        reason = "coverage_end_off_append_branch" if end_id in nodes and end_id not in positions else "unavailable_coverage_end"
        return {"status": "unresolved", "issues": [reason]}
    end = positions[end_id]
    current = marker.get("customType") == "om.observations.recorded"
    if current:
        lower = -1
        for _, e in branch[:-1]:
            d = e.get("data", {})
            if (e.get("type") == "custom" and e.get("customType") == "om.observations.recorded"
                    and isinstance(d, dict) and isinstance(d.get("observations"), list) and d["observations"]
                    and isinstance(d.get("coversUpToId"), str)):
                lower = max(lower, positions.get(d["coversUpToId"], -1))
        start = lower + 1
        status, issues = "range_inferred", []
    else:
        start_id = data.get("coversFromId")
        if start_id not in positions:
            return {"status": "unresolved", "issues": ["unavailable_coverage_start"]}
        start = positions[start_id]
        lower = start - 1
        status, issues = "range_only", ["legacy_producer_unproven"]
    if start > end:
        return {"status": "unresolved", "issues": ["non_forward_coverage"]}
    raw = branch[start:end + 1]
    selected = [(line, e) for line, e in raw if e.get("type") in
                ({"message", "custom_message", "branch_summary"} if current else {"message"}) and _renderable(e)]
    return {"status": status, "issues": issues, "after_entry_id": branch[lower][1]["id"] if lower >= 0 else None,
            "covers_up_to_id": end_id, "branch_entry_ids": [e["id"] for _, e in raw],
            "source_entry_ids": [e["id"] for _, e in selected], "source_seqs": [line for line, _ in selected],
            "source_start_seq": raw[0][0], "source_end_seq": branch[end][0],
            "source_selection": "pi_current_source_types" if current else "known_v2_message_rule_unverified_for_enriched_format"}


def match_hatfield(coverage, events, global_blocks):
    """Accept only exact source AND part hashes, including historically supported rendering."""
    if coverage.get("part_count") != 1 or coverage.get("part_index") != 1:
        return None  # Multipart needs historical width recovery; never fabricate parts.
    if (coverage.get("chunk_key") == coverage.get("coverage_key")
            and coverage.get("part_digest") == coverage.get("source_digest")):
        return None  # Migration placeholders are not rendered-source commitments.
    start, end = coverage["source_start_seq"], coverage["source_end_seq"]
    interval = [e for e in events if start <= e["seq"] <= end]

    def verified(blocks):
        return bool(blocks) and source_digest(blocks) == coverage["source_digest"] and part_digest(blocks) == coverage["part_digest"]

    for ratio in (2.35, 3.25):
        try:
            blocks = build_blocks(interval, ratio)
            if verified(blocks):
                return blocks, {"method": "interval", "chars_per_token": ratio}
        except (ValueError, TypeError, KeyError, OverflowError):
            continue
        blocks = global_blocks.get(ratio, [])
        selected = [b for b in blocks if start <= b["seq"] <= end]
        if verified(selected):
            return selected, {"method": "global_anchors", "chars_per_token": ratio}
        indexes = [i for i, b in enumerate(blocks) if start <= b["seq"] <= end]
        if not indexes:
            continue
        left, right = min(indexes), max(indexes) + 1
        # Native grouping can cut between blocks sharing an anchor or emitted out of seq order.
        # A bounded endpoint search is only a candidate generator; hashes decide acceptance.
        for a in range(max(0, left - 3), min(len(blocks), left + 3) + 1):
            for b in range(max(a + 1, right - 3), min(len(blocks), right + 3) + 1):
                candidate = blocks[a:b]
                if verified(candidate):
                    return candidate, {"method": "contiguous_group", "chars_per_token": ratio,
                                       "endpoint_offsets": [a - left, b - right]}
    return None


def reconstruct_ranges(curated: Path, output: Path) -> dict:
    curated, output = curated.resolve(strict=True), output.resolve()
    manifest = json.loads((curated / "manifest.json").read_text())
    if manifest["schema_version"] != "om-curated-corpus/1":
        raise ValueError("Expected a curated corpus")
    upstream = (curated / manifest["upstream_collection"]).resolve(strict=True)
    if file_hash(upstream / "manifest.json") != manifest["upstream_manifest_sha256"]:
        raise ValueError("Upstream collection changed")
    for ref in manifest["artifacts"]:
        resolve_artifact(curated, ref)
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in (curated, upstream)):
        raise ValueError("Output overlaps upstream corpus")
    index = EvidenceIndex(upstream)
    eligible = {(r["archive_ref"]["sha256"], r["session_id"]) for r in _rows(curated / "sessions.jsonl")}
    groups = defaultdict(list)
    expected = set()
    for c in _rows(curated / "memory-candidates.jsonl"):
        if c["task"] == "observer":
            key = (c["source_archive_ref"]["sha256"], c["session_id"], c["native_entry_id"], c["line_no"])
            groups[key].append(c)
            expected.add(c["candidate_id"])
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    counts, excluded_counts, methods = Counter(), Counter(), Counter()
    assigned, retained = set(), set()
    rows_path = output / "ranges.jsonl"
    exclusions_path = output / "exclusions.jsonl"
    with rows_path.open("xb") as destination, exclusions_path.open("xb") as exclusions:
        os.chmod(rows_path, 0o600)
        os.chmod(exclusions_path, 0o600)

        def save(row, candidates):
            row.update(candidate_ids=[c["candidate_id"] for c in candidates],
                       historical_request_identity_established=False)
            refs = {(r["run_id"], r["seq"]) for r in row.get("source_refs", [])}
            claimed = {(r["run_id"], r["seq"]) for c in candidates for r in c["observation"]["source_refs"]}
            row["target_source_refs_covered"] = bool(refs) and claimed <= refs
            if claimed and refs and not claimed <= refs:
                issue = "legacy_reference_types_not_known_v2_message_rule" if row["source_kind"] == "pi" and row["status"] == "range_only" else "target_refs_outside_reconstructed_sources"
                row["issues"].append(issue)
            row["range_replay_eligible"] = row["status"] in {"source_verified", "range_inferred"} and row["target_source_refs_covered"]
            row["range_id"] = hashlib.sha256(canonical(row["native_identity"])).hexdigest()
            assigned.update(row["candidate_ids"])
            key = row["source_kind"] + ":" + row["status"]
            if not row["range_replay_eligible"] or row["issues"]:
                exclusions.write(canonical(row) + b"\n")
                excluded_counts[key] += 1
                return
            destination.write(canonical(row) + b"\n")
            counts[key] += 1
            retained.update(row["candidate_ids"])
            if "source_verification" in row:
                proof = row["source_verification"]
                methods[proof["method"] + ":" + str(proof["chars_per_token"])] += 1

        pi_groups = defaultdict(list)
        for key, candidates in groups.items():
            if candidates[0]["source_kind"] == "pi":
                pi_groups[key[:2]].append((key, candidates))
        for scope, batches in pi_groups.items():
            job = index.pi_sources[scope]
            entries = _entries(resolve_artifact(upstream, job["archive_ref"]))
            for key, candidates in batches:
                row = pi_range(entries, key[3])
                row.update(source_kind="pi", session_id=scope[1], source_archive_ref=job["archive_ref"],
                           native_identity={"archive_sha256": scope[0], "entry_id": key[2], "line_no": key[3]})
                row["source_refs"] = [{"run_id": "pi:" + scope[1], "seq": s} for s in row.pop("source_seqs", [])]
                save(row, candidates)

        config = json.loads((upstream / "collection-config.json").read_text())
        for source in config["sources"]:
            if source["kind"] != "hatfield_db":
                continue
            source_key = hashlib.sha256(canonical(source)).hexdigest()
            job = json.loads((upstream / "jobs" / (source_key + ".json")).read_text())
            store = job["archive_ref"]["sha256"]
            database = resolve_artifact(upstream, job["archive_ref"])
            with closing(sqlite3.connect(database.as_uri() + "?mode=ro&immutable=1", uri=True)) as db:
                db.row_factory = sqlite3.Row
                coverages = list(db.execute("SELECT * FROM om_coverage ORDER BY run_id, source_start_seq, coverage_key"))
            by_run = defaultdict(list)
            for coverage in coverages:
                by_run[coverage["run_id"]].append(dict(coverage))
            for run, run_coverages in by_run.items():
                journal = index.journals.get((source["om_dir"], run))
                if not journal or (journal["archive_ref"]["sha256"], run) not in eligible:
                    continue
                events = [e for _, e in _entries(resolve_artifact(upstream, journal["archive_ref"]))]
                global_blocks = {}
                for ratio in (2.35, 3.25):
                    try:
                        global_blocks[ratio] = build_blocks(events, ratio)
                    except (ValueError, TypeError, KeyError, OverflowError):
                        global_blocks[ratio] = []
                for coverage in run_coverages:
                    candidates = groups.get((store, run, coverage["boundary_key"], None), [])
                    match = match_hatfield(coverage, events, global_blocks)
                    row = {"source_kind": "hatfield", "session_id": run, "coverage": coverage, "issues": [],
                           "source_archive_ref": journal["archive_ref"], "memory_store_archive_ref": job["archive_ref"],
                           "native_identity": {"store_sha256": store, "coverage_key": coverage["coverage_key"]},
                           "historical_noop": coverage["observation_count"] == 0,
                           "status": "source_verified" if match else "range_only"}
                    if match:
                        blocks, proof = match
                        row.update(source_verification=proof, source_refs=[{"run_id": r, "seq": s} for r, s in sorted({
                            (r["run_id"], r["seq"]) for b in blocks for r in b["source_refs"]})],
                            block_anchors=[{"seq": b["seq"], "kind": b["kind"]} for b in blocks])
                    else:
                        row["issues"].append("source_hashes_not_reconstructed")
                    save(row, candidates)
        for key, candidates in groups.items():
            if not set(c["candidate_id"] for c in candidates) <= assigned:
                save({"source_kind": candidates[0]["source_kind"], "session_id": key[1],
                      "native_identity": {"archive_sha256": key[0], "entry_id": key[2], "line_no": key[3]},
                      "status": "unresolved", "issues": ["missing_historical_coverage"]}, candidates)
    if assigned != expected:
        raise ValueError("Candidate coverage accounting failed")
    report = {"schema_version": "om-observer-ranges/1", "model_calls": 0, "training_ready": False,
              "upstream_curated": str(curated), "upstream_collection": str(upstream),
              "upstream_curated_manifest_sha256": file_hash(curated / "manifest.json"),
              "upstream_collection_manifest_sha256": file_hash(upstream / "manifest.json"),
              "observer_candidates_accounted": len(assigned), "range_counts": dict(counts),
              "excluded_range_counts": dict(excluded_counts), "observer_candidates_retained": len(retained),
              "observer_candidates_excluded": len(assigned - retained),
              "source_verification_methods": dict(methods), "completed_at": now(),
              "hatfield_renderer_revision": HATFIELD_RENDERER_REVISION,
              "estimator_change_revision": ESTIMATOR_CHANGE_REVISION,
              "code_sha256": {p.name: file_hash(p) for p in (Path(__file__), Path(__file__).with_name("hatfield_source.py"))}}
    report["pilot_jobs"] = prepare_pilot(output, upstream)
    write_json(output / "report.json", report)
    write_json(output / "manifest.json", {**report, "artifacts": [artifact(output, p) for p in
               (rows_path, exclusions_path, output / "report.json", output / "pilot-jobs.jsonl")]})
    return report


def prepare_pilot(output: Path, upstream: Path, jobs_path: Path | None = None) -> int:
    """Three independent first ranges; fresh memory, verified source bytes, no labeling."""
    seen = set()
    with (jobs_path or output / "pilot-jobs.jsonl").open("xb") as destination:
        os.chmod(destination.name, 0o600)
        for row in _rows(output / "ranges.jsonl"):
            if (row["status"] != "source_verified" or row["coverage"]["source_start_seq"] != 1
                    or not row["target_source_refs_covered"]):
                continue
            scope = (row["source_archive_ref"]["sha256"], row["session_id"])
            if scope in seen:
                continue
            events = [e for _, e in _entries(resolve_artifact(upstream, row["source_archive_ref"]))]
            ratio = row["source_verification"]["chars_per_token"]
            match = match_hatfield(row["coverage"], events, {ratio: build_blocks(events, ratio)})
            if match is None:
                raise ValueError("Pilot source verification changed")
            blocks, _ = match
            timestamps = {e["seq"]: datetime.fromisoformat(e["ts"]).strftime("%Y-%m-%d %H:%M") for e in events}
            job = {"run_id": row["session_id"], "cutoff_seq": max(r["seq"] for r in row["source_refs"]),
                   "state_scope": row["native_identity"]["store_sha256"],
                   "source_archive_ref": row["source_archive_ref"],
                   "local_time": datetime.fromisoformat(row["coverage"]["covered_at"]).strftime("%Y-%m-%d %H:%M"),
                   "sources": [{"seq": b["seq"], "role": {"user": "User", "tool": "Tool result"}.get(b["segments"][0]["role"], "Assistant"),
                                "timestamp": timestamps[b["seq"]], "segments": b["segments"],
                                "text": b["rendered_text"], "source_refs": b["source_refs"], "native_rendered": True} for b in blocks],
                   "historical_range_id": row["range_id"], "historical_source_digest": row["coverage"]["source_digest"],
                   "historical_part_digest": row["coverage"]["part_digest"],
                   "prior_memory_status": "fresh_replay_empty_not_historical", "fallback_time_status": "coverage_time_not_original_clock"}
            destination.write(canonical(job) + b"\n")
            seen.add(scope)
            if len(seen) == 3:
                break
    return len(seen)


def validate_ranges(output: Path) -> dict:
    output = output.resolve(strict=True)
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["schema_version"] != "om-observer-ranges/1":
        raise ValueError("Unexpected range manifest")
    curated, upstream = Path(manifest["upstream_curated"]), Path(manifest["upstream_collection"])
    for root, key in ((curated, "upstream_curated_manifest_sha256"), (upstream, "upstream_collection_manifest_sha256")):
        if file_hash(root / "manifest.json") != manifest[key]:
            raise ValueError("Upstream manifest changed")
    for ref in manifest["artifacts"]:
        resolve_artifact(output, ref)
    expected = {c["candidate_id"] for c in _rows(curated / "memory-candidates.jsonl") if c["task"] == "observer"}
    retained, excluded, identities = set(), set(), set()
    counts, excluded_counts = Counter(), Counter()
    for filename, candidates, counter in (("ranges.jsonl", retained, counts), ("exclusions.jsonl", excluded, excluded_counts)):
        for row in _rows(output / filename):
            identity = hashlib.sha256(canonical(row["native_identity"])).hexdigest()
            clean = row["status"] in {"source_verified", "range_inferred"} and row["target_source_refs_covered"] and not row["issues"]
            if (identity != row["range_id"] or identity in identities or row["historical_request_identity_established"]
                    or clean != (filename == "ranges.jsonl")):
                raise ValueError("Invalid range identity or clean-range selection")
            identities.add(identity)
            candidates.update(row["candidate_ids"])
            counter[row["source_kind"] + ":" + row["status"]] += 1
    if (retained & excluded or retained | excluded != expected or len(expected) != manifest["observer_candidates_accounted"]
            or len(retained) != manifest["observer_candidates_retained"] or len(excluded) != manifest["observer_candidates_excluded"]
            or dict(counts) != manifest["range_counts"] or dict(excluded_counts) != manifest["excluded_range_counts"]):
        raise ValueError("Incomplete range accounting")
    return {"status": "pass", "range_counts": dict(counts), "excluded_range_counts": dict(excluded_counts),
            "observer_candidates_retained": len(retained), "observer_candidates_excluded": len(excluded), "model_calls": 0}
