"""Prepare and run substantial Pi/Hatfield replays without touching the small review pack."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path

from . import maintenance
from .collection import canonical, file_hash, write_json
from .pilot import CORPUS, RANGES, _jobs
from .policies import validate_policies
from .ranges import validate_ranges
from .redaction import Redactor
from .teacher import POLICIES_PATH, PROVIDER, PROVIDERS, REQUEST_OPTIONS, STUDENT_CONTEXT_WINDOW, run_file, slice_job

JOBS_PATH = Path("data/teacher-long-pilot/jobs.jsonl")
OUTPUT_PATH = Path("data/teacher-long-pilot/attempts.jsonl")


def _chain(kind, rows):
    """Take a clean prefix from the start, never skip missing coverage."""
    if kind == "hatfield":
        result, end = [], 0
        for row in sorted(rows, key=lambda r: r["coverage"]["source_end_seq"]):
            if row["coverage"]["source_start_seq"] != end + 1 or not row["source_refs"]:
                break
            result.append(row)
            end = row["coverage"]["source_end_seq"]
        return result
    children = defaultdict(list)
    for row in rows:
        children[row["after_entry_id"]].append(row)
    cached = {}
    def longest(end):
        if end not in cached:
            cached[end] = max(([row] + longest(row["covers_up_to_id"]) for row in children[end]
                               if row["covers_up_to_id"] != end), key=len, default=[])
        return cached[end]
    return longest(None)


def prepare(policies, config):
    if JOBS_PATH.exists():
        report = json.loads(JOBS_PATH.with_suffix(".manifest.json").read_text(encoding="utf-8"))
        if report["jobs_sha256"] != file_hash(JOBS_PATH):
            raise ValueError("Prepared replay inputs changed")
        return report
    validate_ranges(RANGES)
    groups = defaultdict(list)
    for line in (RANGES / "ranges.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        if row["issues"] or not row["range_replay_eligible"]:
            raise ValueError("Unclean range found in retained range release")
        groups[(row["source_kind"], row["session_id"], row["source_archive_ref"]["sha256"])].append(row)
    historical_tokens = {}
    for line in Path("data/curated-2026-10-13-v2/memory-candidates.jsonl").open(encoding="utf-8"):
        candidate = json.loads(line)
        if candidate["task"] == "observer":
            historical_tokens[candidate["candidate_id"]] = maintenance.pool_tokens([candidate["observation"]])
    redactor = Redactor()
    if not redactor.catalog_loaded:
        raise ValueError("The private credential catalog is required")
    selected, sessions = [], []
    for kind in ("pi", "hatfield"):
        candidates = []
        for (source_kind, _, _), rows in groups.items():
            if source_kind != kind:
                continue
            chain = _chain(kind, rows)
            tokens = sum(historical_tokens[cid] for row in chain for cid in row["candidate_ids"])
            if tokens > maintenance.REFLECT_AFTER_TOKENS:
                candidates.append((tokens, chain))
        # Prefer the smallest session with at least two threshold-sized historical
        # pools, or the largest available one. Counts select cohorts, not labels.
        goal = 2 * maintenance.REFLECT_AFTER_TOKENS
        candidates.sort(key=lambda c: (0, c[0]) if c[0] >= goal else (1, -c[0]))
        for tokens, chain in candidates:
            try:
                jobs = [redactor.job(job) for job in _jobs(chain)]
            except (ValueError, KeyError):
                continue  # Unsupported source modalities are not silently discarded.
            if any(b["cutoff_seq"] <= a["cutoff_seq"] for a, b in zip(jobs, jobs[1:])):
                continue
            if kind == "pi":
                scope = hashlib.sha256(canonical({"archive": chain[0]["source_archive_ref"]["sha256"],
                                                   "terminal_marker": chain[-1]["native_identity"]})).hexdigest()
                jobs = [{**job, "state_scope": scope} for job in jobs]
            chunks = sum(len(slice_job(job, policies["observer"], config)) for job in jobs)
            selected.extend(jobs)
            sessions.append({"source_kind": kind, "run_id": jobs[0]["run_id"], "state_scope": jobs[0]["state_scope"],
                             "historical_ranges": len(jobs), "initial_observer_chunks": chunks,
                             "source_refs": sum(len(s.get("source_refs", [None])) for job in jobs for s in job["sources"]),
                             "rendered_source_blocks": sum(len(job["sources"]) for job in jobs),
                             "source_characters": sum(len(s["text"]) for job in jobs for s in job["sources"]),
                             "historical_observation_tokens": tokens,
                             "source_archive_ref": chain[0]["source_archive_ref"],
                             "terminal_range_id": chain[-1]["range_id"]})
            break
        else:
            raise ValueError("No substantial clean renderable prefix for " + kind)
    JOBS_PATH.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = JOBS_PATH.with_name(JOBS_PATH.name + ".partial")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as output:
        for job in selected:
            redactor.require_clean(job)
            output.write(canonical(job) + b"\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, JOBS_PATH)
    report = {"sessions": sessions, "jobs_sha256": file_hash(JOBS_PATH), "redaction_id": redactor.fingerprint,
              "ranges_manifest_sha256": file_hash(RANGES / "manifest.json"),
              "scope": "Clean chronological prefixes from session start; excluded gaps and uncovered tails are not replayed.",
              "thresholds": {"reflect_after_tokens": maintenance.REFLECT_AFTER_TOKENS,
                             "observation_target_tokens": maintenance.OBSERVATIONS_TARGET_TOKENS},
              "historical_memory_used_for_teacher_inputs": False, "model_calls": 0}
    write_json(JOBS_PATH.with_suffix(".manifest.json"), report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Prepare and validate inputs without provider calls.")
    args = parser.parse_args()
    validate_policies(POLICIES_PATH)
    bundle = json.loads((POLICIES_PATH / "hatfield-target/definitions.json").read_text(encoding="utf-8"))
    policies = {p["role"]: p for p in bundle["agents"]}
    config = {**PROVIDERS[PROVIDER], "provider": PROVIDER, "request_options": REQUEST_OPTIONS,
              "context_window": STUDENT_CONTEXT_WINDOW}
    report = prepare(policies, config)
    redactor = Redactor()
    if redactor.fingerprint != report["redaction_id"]:
        raise ValueError("The redaction policy changed; refusing to re-label an existing replay implicitly")
    for line in JOBS_PATH.open(encoding="utf-8"):
        redactor.require_clean(json.loads(line))
    if args.dry_run:
        print(json.dumps(report))
        return 0
    if not os.environ.get(config["api_key_env"]):
        parser.error("Missing " + config["api_key_env"])
    pin = OUTPUT_PATH.with_name("run.json")
    contract = {"jobs_sha256": report["jobs_sha256"], "redaction_id": redactor.fingerprint, "config": config,
                "definitions_sha256": file_hash(POLICIES_PATH / "hatfield-target/definitions.json"),
                "code": {name: file_hash(Path(__file__).with_name(name))
                         for name in ("teacher.py", "maintenance.py", "replay.py", "long_pilot.py")}}
    if pin.exists():
        if json.loads(pin.read_text(encoding="utf-8")) != contract:
            raise ValueError("Replay code or settings changed; use a new output directory instead of implicitly re-labeling")
    else:
        write_json(pin, contract)
    summary = run_file(JOBS_PATH, OUTPUT_PATH, policies["observer"], config,
                       maintenance_policies={r: policies[r] for r in ("reflector", "dropper")})
    print(json.dumps(summary))
    return int(any(summary[s] for s in ("failed", "stalled", "budget_exhausted", "blocked")))


if __name__ == "__main__":
    raise SystemExit(main())
