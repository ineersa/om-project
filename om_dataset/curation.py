"""Build an OM-only, source-resolved view without altering frozen archives."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

from .collection import artifact, canonical, file_hash, now, resolve_artifact, validate_collection, write_json


def _scope(candidate: dict) -> tuple[str, str]:
    return candidate["source_archive_ref"]["sha256"], candidate["session_id"]


def _observation_key(candidate: dict) -> tuple:
    return (*_scope(candidate), candidate["native_memory_id"])


class EvidenceIndex:
    def __init__(self, upstream: Path):
        self.upstream = upstream
        self.journals = {}
        self.pi_sources = {}
        self.db_projects = {}
        self.parents = {}
        self.positions = {}
        self.ambiguous = {}
        self._cached_ancestry = None
        configuration = json.loads((upstream / "collection-config.json").read_text())
        for source in configuration["sources"]:
            key = hashlib.sha256(canonical(source)).hexdigest()
            job = json.loads((upstream / "jobs" / (key + ".json")).read_text())
            if source["kind"] == "hatfield_db":
                self.db_projects[job["archive_ref"]["sha256"]] = source["om_dir"]
            elif source["kind"] in {"pi", "hatfield"}:
                parsed = json.loads((upstream / job["parsed_ref"]["path"]).read_text())
                run = parsed["session_id"]
                if source["kind"] == "pi":
                    self.pi_sources[(job["archive_ref"]["sha256"], run)] = job
                    if parsed["memory_candidates"]:
                        scope = (job["archive_ref"]["sha256"], run)
                        self.parents[scope] = parsed["stats"]["ancestry_lookup"]
                        self.positions[scope] = {s["native_id"]: s["line_no"] for s in parsed["sources"]
                                                 if isinstance(s["native_id"], str)}
                        self.ambiguous[scope] = set(parsed["stats"]["ambiguous_native_ids"])
                else:
                    self.journals[(source["om_dir"], run)] = job

    def source_job(self, candidate: dict) -> dict | None:
        if candidate["source_kind"] == "pi":
            return self.pi_sources.get(_scope(candidate))
        project = self.db_projects.get(candidate["source_archive_ref"]["sha256"])
        return self.journals.get((project, candidate["session_id"]))

    def ancestors(self, candidate: dict) -> set[str] | None:
        if candidate["source_kind"] != "pi":
            return None
        scope = _scope(candidate)
        key = (scope, candidate["native_entry_id"], candidate["line_no"])
        if self._cached_ancestry and self._cached_ancestry[0] == key:
            return self._cached_ancestry[1]
        parents = self.parents.get(scope, {})
        positions = self.positions.get(scope, {})
        ambiguous = self.ambiguous.get(scope, set())
        node = candidate["native_entry_id"]
        result = set()
        while node is not None:
            if not isinstance(node, str) or node in ambiguous or node in result or node not in parents:
                return set()
            result.add(node)
            parent = parents[node]
            if parent is not None and (not isinstance(parent, str) or parent not in positions
                                       or positions[parent] >= positions[node]):
                return set()
            node = parent
        self._cached_ancestry = (key, result)
        return result

    def visible(self, observation: dict, candidate: dict) -> bool:
        ancestors = self.ancestors(candidate)
        return ancestors is None or (observation["native_entry_id"] in ancestors
                                     and observation["line_no"] < candidate["line_no"])


def _rows(path: Path):
    with path.open("rb") as stream:
        for line in stream:
            yield json.loads(line)


def _observer_reason(candidate: dict, index: EvidenceIndex) -> str | None:
    if not index.source_job(candidate):
        return "missing_session_journal"
    if not candidate.get("valid"):
        return "invalid_candidate"
    observation = candidate.get("observation", {})
    if candidate.get("evidence_status") != "resolved" or not observation.get("source_refs"):
        return "unresolved_content_reference"
    if not candidate.get("native_memory_id"):
        return "missing_native_memory_id"
    if candidate["source_kind"] == "pi" and not index.ancestors(candidate):
        return "invalid_native_ancestry"
    return None


def _native_current_observation(record) -> bool:
    # Pi V3 fold validation is intentionally distinct from target eligibility.
    # For example, a nonempty but invalid calendar timestamp can win a native ID.
    if not isinstance(record, dict):
        return False
    def text(value):
        return isinstance(value, str) and len(value) > 0
    tokens = record.get("tokenCount")
    refs = record.get("sourceEntryIds")
    if not isinstance(tokens, (int, float)) or isinstance(tokens, bool):
        return False
    try:
        finite = math.isfinite(tokens)
    except OverflowError:
        return False
    return (text(record.get("id")) and re.fullmatch(r"[a-f0-9]{12}", record["id"]) is not None
            and text(record.get("content")) and text(record.get("timestamp"))
            and record.get("relevance") in ("low", "medium", "high", "critical")
            and isinstance(refs, list) and bool(refs) and all(text(ref) for ref in refs)
            and finite and tokens >= 0)


def _selections(upstream: Path, index: EvidenceIndex):
    input_path = upstream / "memory-candidates.jsonl"
    observations = defaultdict(list)
    batches = defaultdict(list)
    for candidate in _rows(input_path):
        if candidate["source_kind"] == "pi" and not isinstance(candidate["native_entry_id"], str):
            continue
        if candidate["task"] == "observer" and isinstance(candidate.get("native_memory_id"), str):
            observations[_observation_key(candidate)].append({
                "candidate_id": candidate["candidate_id"], "native_entry_id": candidate["native_entry_id"],
                "line_no": candidate["line_no"], "native_record": candidate["native_record"],
                "adapter": candidate["adapter"], "eligible": _observer_reason(candidate, index) is None,
                "batch": (*_scope(candidate), candidate["native_entry_id"])})
        if candidate["adapter"] == "pi-current" and candidate["task"] == "observer":
            coverage = candidate.get("coverage", {}).get("coversUpToId")
            batches[(*_scope(candidate), candidate["native_entry_id"])].append(
                _native_current_observation(candidate["native_record"])
                and isinstance(coverage, str) and bool(coverage))

    def select_observation(candidate, native_id):
        aliases = [o for o in observations.get((*_scope(candidate), native_id), [])
                   if index.visible(o, candidate)]
        # V3 folds an entire valid batch, then retains the first valid record for
        # each native ID. Never let filtering promote a later alias over it.
        aliases = [o for o in aliases if o["adapter"] != "pi-current" or all(batches[o["batch"]])]
        aliases.sort(key=lambda o: o["line_no"] if o["line_no"] is not None else -1)
        if not aliases:
            return None
        if aliases[0]["adapter"] == "pi-legacy" and len({
            hashlib.sha256(canonical(o["native_record"])).hexdigest() for o in aliases
        }) > 1:
            # The legacy name covers several formats. Do not guess an ID winner
            # when the archived values disagree and native selection is unknown.
            return None
        winner = aliases[0]
        return winner if winner["eligible"] else None

    def supports(candidate, ids):
        if not isinstance(ids, list) or not ids or not all(isinstance(i, str) and i for i in ids):
            return None
        linked = []
        for native_id in ids:
            match = select_observation(candidate, native_id)
            if match is None:
                return None
            linked.append(match["candidate_id"])
        return linked

    def eligibility(candidate):
        if candidate["task"] == "observer":
            return _observer_reason(candidate, index), []
        if not index.source_job(candidate):
            return "missing_session_journal", []
        if not candidate.get("valid"):
            return "invalid_candidate", []
        if candidate["source_kind"] == "pi" and not index.ancestors(candidate):
            return "invalid_native_ancestry", []
        task = candidate["task"]
        if task in {"reflector", "dropper"}:
            ids = candidate.get("native_supporting_observation_ids" if task == "reflector"
                                else "native_observation_ids")
            linked = supports(candidate, ids)
            return ("unresolved_dependent_content", []) if linked is None else (None, linked)
        if task == "snapshot":
            snapshot = candidate["native_record"]
            linked = []
            for observation in snapshot.get("observations", []):
                if not isinstance(observation, dict) or not observation.get("id"):
                    return "summary_only_snapshot", []
                # A content-hash ID alone does not prove which text, timestamp,
                # relevance or sources the snapshot actually retained.
                fields = ("id", "content", "timestamp", "relevance", "sourceEntryIds")
                match = select_observation(candidate, observation["id"])
                if match is None or not all(match["native_record"].get(k) == observation.get(k) for k in fields):
                    return "unresolved_snapshot_content", []
                linked.append(match["candidate_id"])
            for reflection in snapshot.get("reflections", []):
                proof = supports(candidate, reflection.get("supportingObservationIds"))
                if proof is None:
                    return "unresolved_snapshot_support", []
                linked.extend(proof)
            # Empty snapshots provide no content-grounded example.
            return (None, sorted(set(linked))) if linked else ("empty_snapshot", [])
        return "unsupported_task", []

    seen = defaultdict(list)
    for candidate in _rows(input_path):
        reason, links = eligibility(candidate)
        duplicate = None
        if reason is None and candidate["task"] == "observer":
            key = hashlib.sha256(canonical({"scope": _scope(candidate),
                                            "native_record": candidate["native_record"]})).hexdigest()
            ancestors = index.ancestors(candidate)
            duplicate = next((old for old in seen[key] if ancestors is None or
                              old["native_entry_id"] in ancestors), None)
            if duplicate is not None:
                reason = "exact_same_evidence_duplicate"
            else:
                seen[key].append({"candidate_id": candidate["candidate_id"],
                                  "native_entry_id": candidate["native_entry_id"]})
        yield candidate, reason, links, duplicate


def curate_collection(upstream: Path, output: Path) -> dict:
    upstream = upstream.expanduser().resolve(strict=True)
    validate_collection(upstream)
    output = output.expanduser().absolute()
    if output.resolve().is_relative_to(upstream) or upstream.is_relative_to(output.resolve()):
        raise ValueError("Curated output must not contain or modify its source collection")
    index = EvidenceIndex(upstream)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    # Protect even outputs created outside the project's default data/ directory.
    (output / ".gitignore").write_text("*\n")
    os.chmod(output / ".gitignore", 0o600)
    kept, excluded, relevance = Counter(), Counter(), Counter()
    sessions = {}
    total = 0
    files = [output / name for name in ("memory-candidates.jsonl", "eligibility.jsonl", "exclusions.jsonl")]
    with files[0].open("wb") as accepted, files[1].open("wb") as proofs, files[2].open("wb") as rejected:
        for path in files:
            os.chmod(path, 0o600)
        for candidate, reason, links, duplicate in _selections(upstream, index):
            total += 1
            if reason:
                excluded[reason] += 1
                rejection = {"candidate_id": candidate["candidate_id"], "reason": reason,
                             "task": candidate["task"], "adapter": candidate["adapter"]}
                if duplicate:
                    rejection["kept_candidate_id"] = duplicate["candidate_id"]
                rejected.write(canonical(rejection) + b"\n")
                continue
            job = index.source_job(candidate)
            session_key = job["archive_ref"]["sha256"]
            sessions[session_key] = {"session_id": candidate["session_id"],
                                     "source_kind": "pi" if candidate["source_kind"] == "pi" else "hatfield",
                                     "archive_ref": job["archive_ref"], "parsed_ref": job["parsed_ref"],
                                     "split": "unassigned"}
            accepted.write(canonical(candidate) + b"\n")
            proofs.write(canonical({"candidate_id": candidate["candidate_id"],
                                    "status": "content_reference_resolved",
                                    "supporting_candidate_ids": sorted(set(links))}) + b"\n")
            kept[(candidate["adapter"], candidate["task"])] += 1
            if candidate["task"] == "observer":
                relevance[candidate["observation"]["relevance"]] += 1
    session_path = output / "sessions.jsonl"
    with session_path.open("wb") as stream:
        os.chmod(session_path, 0o600)
        for key in sorted(sessions):
            stream.write(canonical(sessions[key]) + b"\n")
    report = {"input_candidates": total, "retained_candidates": sum(kept.values()),
              "excluded_candidates": sum(excluded.values()), "exclusions_by_reason": dict(excluded),
              "retained_by_adapter_and_task": {f"{a}/{t}": n for (a, t), n in kept.items()},
              "retained_session_files": len(sessions),
              "sessions_by_platform": dict(Counter(s["source_kind"] for s in sessions.values())),
              "observer_relevance": dict(relevance), "training_ready": False,
              "agent_requests_captured": False, "model_calls": 0, "split_status": "unassigned"}
    write_json(output / "report.json", report)
    code_path = output / "curation-code.py"
    code_path.write_bytes(Path(__file__).read_bytes())
    os.chmod(code_path, 0o400)
    artifact_paths = [*files, session_path, output / "report.json", output / ".gitignore", code_path]
    # This is a dependent view. Native archive references resolve in the frozen
    # upstream release, not in this output directory or any live session store.
    manifest = {"schema_version": "om-curated-corpus/1", "completed_at": now(),
                "upstream_collection": os.path.relpath(upstream, output),
                "upstream_manifest_sha256": file_hash(upstream / "manifest.json"),
                "training_ready": False, "redacted": False, "model_calls": 0,
                "target": json.loads((upstream / "manifest.json").read_text())["target"],
                "agent_definition_status": "production bundles and exact requests not exported",
                "artifacts": [artifact(output, p) for p in artifact_paths],
                "deduplication": "exact native record, same source archive, same native branch only",
                "eligibility": "OM-bearing sessions; direct or transitive resolved content evidence"}
    write_json(output / "manifest.json", manifest)
    return report


def validate_curated_collection(root: Path) -> dict:
    root = root.expanduser().resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema_version") != "om-curated-corpus/1" or manifest.get("training_ready") is not False:
        raise ValueError("Invalid curated corpus manifest")
    for ref in manifest["artifacts"]:
        resolve_artifact(root, ref)
    upstream = (root / manifest["upstream_collection"]).resolve(strict=True)
    if file_hash(upstream / "manifest.json") != manifest["upstream_manifest_sha256"]:
        raise ValueError("Upstream manifest mismatch")
    validate_collection(upstream)
    def unique_rows(path):
        result = {}
        for row in _rows(path):
            identity = row["candidate_id"]
            if identity in result:
                raise ValueError("Repeated candidate ID in curated output")
            result[identity] = row
        return result

    retained = unique_rows(root / "memory-candidates.jsonl")
    for identity, candidate in retained.items():
        body = {k: v for k, v in candidate.items() if k != "candidate_id"}
        if hashlib.sha256(canonical(body)).hexdigest() != identity:
            raise ValueError("Curated candidate identity mismatch")
    exclusions = unique_rows(root / "exclusions.jsonl")
    proofs = unique_rows(root / "eligibility.jsonl")
    if proofs.keys() != retained.keys() or retained.keys() & exclusions.keys():
        raise ValueError("Curated eligibility partition mismatch")
    index = EvidenceIndex(upstream)
    seen = set()
    for original, reason, links, duplicate in _selections(upstream, index):
        identity = original["candidate_id"]
        if identity in seen:
            raise ValueError("Repeated upstream candidate ID")
        seen.add(identity)
        if reason is None:
            if retained.get(identity) != original:
                raise ValueError("Curated retained row differs from upstream selection")
            expected = {"candidate_id": identity, "status": "content_reference_resolved",
                        "supporting_candidate_ids": sorted(set(links))}
            if proofs.get(identity) != expected:
                raise ValueError("Curated content proof differs from source evidence")
        else:
            expected = {"candidate_id": identity, "reason": reason,
                        "task": original["task"], "adapter": original["adapter"]}
            if duplicate:
                expected["kept_candidate_id"] = duplicate["candidate_id"]
            if exclusions.get(identity) != expected:
                raise ValueError("Curated exclusion differs from upstream selection")
    if seen != retained.keys() | exclusions.keys():
        raise ValueError("Curated view does not partition upstream candidates")
    for proof in proofs.values():
        for dependency in proof["supporting_candidate_ids"]:
            if dependency in retained:
                continue
            if exclusions.get(dependency, {}).get("kept_candidate_id") not in retained:
                raise ValueError("Dependent content has no retained evidence")
    report = json.loads((root / "report.json").read_text())
    if (len(retained) != report["retained_candidates"] or len(exclusions) != report["excluded_candidates"]
            or len(seen) != report["input_candidates"]):
        raise ValueError("Curated counts mismatch")
    return {"status": "pass", "retained_candidates": len(retained),
            "excluded_candidates": len(exclusions), "training_ready": False}
