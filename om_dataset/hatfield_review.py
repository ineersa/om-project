"""Replace Pi review cases with Hatfield cases while retaining existing Hatfield votes."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
from collections import Counter

from . import comparison as c
from .collection import canonical, file_hash, write_json

ROOT = Path("data/teacher-hatfield-comparison")
ACTIVE = Path("data/review")
PRIVATE = Path("data/review-private")
HISTORY = Path("data/review-history")


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def prepare(redactor):
    path = ROOT / "manifest.json"
    if path.exists():
        saved = _json(path)
        if saved["preparer_sha256"] != file_hash(Path(__file__)) or saved["redaction_id"] != redactor.fingerprint:
            raise ValueError("Pinned replacement preparation changed")
        return saved
    old = _json(c.OUTPUT / "manifest.json")
    public = [json.loads(s) for s in (ACTIVE / "cases.jsonl").read_bytes().splitlines()]
    by_id = {row["case_id"]: row for row in public}
    mapping = _json(PRIVATE / "identity-map.json")
    old_sha = file_hash(ACTIVE / "cases.jsonl")
    if mapping["cases_sha256"] != old_sha:
        raise ValueError("Current cases and identity mapping disagree")
    kept = [x for x in old["cases"] if x["source_kind"] == "hatfield"]
    retired = [x for x in old["cases"] if x["source_kind"] == "pi"]
    if len(kept) != 3 or len(retired) != 3 or set(by_id) != {x["case_id"] for x in old["cases"]}:
        raise ValueError("Expected the existing three Hatfield and three Pi cases")
    policies = {p["role"]: p for p in _json(c.POLICIES)["agents"]}
    for case in kept:
        c.exact_input(case["baseline"], policies[case["baseline"]["task"]], redactor)
        if case["baseline"]["job"]["run_id"].startswith("pi:"):
            raise ValueError("Retained case is not Hatfield-derived")
    excluded = set(old.get("excluded_preflight_source_jobs", [])) | {x["baseline"]["job_id"] for x in old["cases"]}
    rows, source_sha = c._rows(c.SOURCE)
    replacements = c.select_cases([r for r in rows if r["job_id"] not in excluded], policies, redactor)[:3]
    for n, case in enumerate(replacements, 1):
        case["case_id"] = f"hatfield-new-{n:03d}"
    kept_ids = {x["case_id"] for x in kept}
    saved = {"preparer_sha256": file_hash(Path(__file__)), "redaction_id": redactor.fingerprint,
             "old_cases_sha256": old_sha, "old_mapping_sha256": file_hash(PRIVATE / "identity-map.json"),
             "old_case_hashes": {x["case_id"]: c._hash(x) for x in public},
             "retired_case_ids": [x["case_id"] for x in retired],
             "retained_cases": [x for x in public if x["case_id"] in kept_ids],
             "retained_mapping": [x for x in mapping["cases"] if x["case_id"] in kept_ids],
             "retained_sources": {x["case_id"]: x["baseline"]["job_id"] for x in kept},
             "cases": replacements, "source_sha256": source_sha,
             "excluded_source_jobs": sorted(excluded), "source_kind": "hatfield"}
    redactor.require_clean(saved)
    c._private_dir(ROOT)
    write_json(path, saved)
    return saved


def publish(saved, replacements, replacement_mapping, redactor):
    """Publish by content-checked moves; resume a partially completed directory swap."""
    kept_ids = {x["case_id"] for x in saved["retained_cases"]}
    pack = replacements + saved["retained_cases"]
    if len(pack) != 6 or len({x["case_id"] for x in pack}) != 6 or Counter(x["task"] for x in pack) != {"observer": 2, "reflector": 2, "dropper": 2}:
        raise ValueError("Replacement pack must have six unique cases")
    if {x["case_id"] for x in replacements} != {x["case_id"] for x in saved["cases"]}:
        raise ValueError("Unexpected replacement IDs")
    for case in pack:
        redactor.require_clean(case)
    encoded = b"".join(canonical(x) + b"\n" for x in pack)
    new_sha = hashlib.sha256(encoded).hexdigest()
    archive = HISTORY / saved["old_cases_sha256"]
    transition = ROOT / "transition.json"
    public_tmp, private_tmp = ROOT / "next-review", ROOT / "next-private"

    def matches(directory, filename, digest):
        path = directory / filename
        return path.exists() and file_hash(path) == digest

    # A completed publication never overwrites subsequent human decisions.
    if matches(ACTIVE, "cases.jsonl", new_sha):
        tx = _json(transition)
        if not matches(PRIVATE, "identity-map.json", tx["new_mapping_sha256"]):
            raise ValueError("Published mapping changed")
        return {"active_cases": 6, "source_kind": "hatfield", "retained_cases": 3, "replacement_cases": 3}
    old_public = archive / "review" if (archive / "review").exists() else ACTIVE
    votes = old_public / "votes"
    votes.mkdir(exist_ok=True, mode=0o700)
    fd = os.open(votes / ".lock", os.O_RDWR | os.O_CREAT, 0o600)
    with os.fdopen(fd, "a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not transition.exists():
            if not matches(ACTIVE, "cases.jsonl", saved["old_cases_sha256"]) or not matches(PRIVATE, "identity-map.json", saved["old_mapping_sha256"]):
                raise ValueError("Active pack changed during replacement generation")
            for directory in (public_tmp, private_tmp):
                if directory.exists():
                    shutil.rmtree(directory)  # No publication has started yet.
                c._private_dir(directory)
            (public_tmp / "votes").mkdir(mode=0o700)
            with (public_tmp / "cases.jsonl").open("wb") as output:
                os.fchmod(output.fileno(), 0o600)
                output.write(encoded)
                output.flush()
                os.fsync(output.fileno())
            mapping = {"protocol": c.PROTOCOL, "cases": replacement_mapping + saved["retained_mapping"],
                       "cases_sha256": new_sha, "source_kinds": {x["case_id"]: "hatfield" for x in pack}}
            redactor.require_clean(mapping)
            c.teacher._checkpoint(private_tmp / "identity-map.json", mapping)
            c._private_dir(archive)
            c.teacher._checkpoint(transition, {"old_cases_sha256": saved["old_cases_sha256"], "new_cases_sha256": new_sha,
                                             "new_mapping_sha256": file_hash(private_tmp / "identity-map.json")})
        tx = _json(transition)
        if tx["new_cases_sha256"] != new_sha:
            raise ValueError("Pinned replacement pack changed")
        for path in votes.glob("*.json"):
            vote = _json(path)
            if vote["case_sha256"] != saved["old_case_hashes"].get(vote["case_id"]):
                raise ValueError("Existing vote does not match its original case")
            if vote["case_id"] in kept_ids:
                shutil.copy2(path, public_tmp / "votes" / path.name)

        def move(source, target, filename, digest):
            if matches(target, filename, digest):
                return
            if target.exists() or not matches(source, filename, digest):
                raise ValueError("Unexpected review publication state")
            os.replace(source, target)
        move(ACTIVE, archive / "review", "cases.jsonl", saved["old_cases_sha256"])
        move(PRIVATE, archive / "review-private", "identity-map.json", saved["old_mapping_sha256"])
        move(private_tmp, PRIVATE, "identity-map.json", tx["new_mapping_sha256"])
        move(public_tmp, ACTIVE, "cases.jsonl", new_sha)
        for directory in (ACTIVE.parent, archive, ROOT):
            fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            os.fsync(fd)
            os.close(fd)
    return {"active_cases": 6, "source_kind": "hatfield", "retained_cases": 3, "replacement_cases": 3}


def sync_staging(saved):
    """Retire the mixed-source staging pack as well as the active pack."""
    archive = HISTORY / saved["old_cases_sha256"]
    for target, source, previous, name in (
        (c.STAGING, ACTIVE / "cases.jsonl", saved["old_cases_sha256"], "staged-review"),
        (c.IDENTITIES, PRIVATE / "identity-map.json", saved["old_mapping_sha256"], "staged-private"),
    ):
        if target.exists():
            if file_hash(target) == file_hash(source):
                continue
            if file_hash(target) != previous:
                raise ValueError("Unexpected staging pack; refusing to overwrite it")
            os.replace(target.parent, archive / name)
        c._private_dir(target.parent)
        temporary = target.with_suffix(target.suffix + ".publish.partial")
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as output:
            output.write(source.read_bytes())
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target)
        fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        os.fsync(fd)
        os.close(fd)


def main():
    redactor = c._redactor()
    c._private_dir(ROOT)
    fd = os.open(ROOT / ".lock", os.O_RDWR | os.O_CREAT, 0o600)
    with os.fdopen(fd, "a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        saved = prepare(redactor)
        staged, identity = ROOT / "replacements.jsonl", ROOT / "replacement-identities.json"
        if not staged.exists():
            if identity.exists():
                raise ValueError("Incomplete staged publication; preserve its identity assignments")
            props = c.server_properties(redactor)
            props_path = ROOT / "server-properties.json"
            if props_path.exists() and _json(props_path) != props:
                raise ValueError("Local model settings changed")
            if not props_path.exists():
                write_json(props_path, props)
            results = c.run_cases(saved["cases"], ROOT / "results", redactor, properties=props)
            if any(x["status"] != "completed" for x in results):
                raise ValueError("Replacement invocations are incomplete; resume before publishing")
            c.export_cases(saved["cases"], results, redactor, staged, identity)
        replacements = [json.loads(s) for s in staged.read_bytes().splitlines()]
        if _json(identity)["cases_sha256"] != file_hash(staged):
            raise ValueError("Replacement mapping does not match its cases")
        summary = publish(saved, replacements, _json(identity)["cases"], redactor)
        sync_staging(saved)
        print(json.dumps(summary))


if __name__ == "__main__":
    main()
