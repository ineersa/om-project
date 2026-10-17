"""Restart failed empty-reflection invocations without changing prompts or completed work."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil

from .collection import canonical, write_json
from .redaction import Redactor

OUTPUT = Path("data/teacher-long-pilot/attempts.jsonl")


def reset_empty_reflections(path=OUTPUT):
    redactor, reset = Redactor(), []
    # Use the runner's lock, so a live generation cannot race checkpoint archival.
    with path.open("r+b") as log:
        fcntl.flock(log, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = [json.loads(line) for line in log if line.strip()]
        completed = {r["job_id"] for r in rows if r["status"] == "completed"}
        latest = {r["job_id"]: r for r in rows}
        for key, result in latest.items():
            if key in completed or result.get("task") != "reflector" or result["status"] != "failed":
                continue
            calls = [call for message in result.get("messages", []) for call in message.get("tool_calls") or []]
            if (result.get("reflections") or len(calls) != 1 or
                    calls[0]["function"]["name"] != "record_reflections" or
                    json.loads(calls[0]["function"]["arguments"]) != {"reflections": []}):
                continue
            checkpoint = path.with_name(path.name + ".checkpoints") / (key + ".json")
            if not checkpoint.exists():
                continue
            raw = checkpoint.read_bytes()
            saved = json.loads(raw)
            redactor.require_clean(saved)
            if saved["accepted"] or saved["finished"] is not None:
                raise ValueError("Refusing to discard accepted or finished checkpoint work")
            archive = path.parent / "recovery" / (key + "-" + hashlib.sha256(raw).hexdigest()[:16] + ".json")
            archive.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            event = {"schema_version": "om-recovery/1", "job_id": key,
                     "reason": "invalid_empty_reflection_call", "accepted_reflections": 0,
                     "checkpoint_sha256": hashlib.sha256(raw).hexdigest(), "prompt_changed": False,
                     "recorded_at": datetime.now(timezone.utc).isoformat()}
            if archive.exists():
                if archive.read_bytes() != raw:
                    raise ValueError("Conflicting recovery archive")
                checkpoint.unlink()
            else:
                os.replace(checkpoint, archive)
            os.chmod(archive, 0o600)
            event_path = archive.with_suffix(".event.json")
            if not event_path.exists():
                write_json(event_path, event)
            fd = os.open(archive.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            reset.append(key)
    return {"reset_failed_invocations": len(reset), "job_ids": reset, "completed_work_changed": False}


def continue_replay():
    """Import compatible cached traces into a pinned recovery run, then resume."""
    from . import maintenance, teacher
    from .collection import file_hash
    from .long_pilot import JOBS_PATH
    from .policies import validate_policies

    base, root = OUTPUT.parent, OUTPUT.parent / "recovered"
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(root / ".lock", os.O_RDWR | os.O_CREAT, 0o600)
    with os.fdopen(fd, "a+b") as lock, OUTPUT.open("rb") as source:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(source, fcntl.LOCK_EX | fcntl.LOCK_NB)
        original = json.loads((base / "run.json").read_text(encoding="utf-8"))
        validate_policies(teacher.POLICIES_PATH)
        definitions = teacher.POLICIES_PATH / "hatfield-target/definitions.json"
        if file_hash(definitions) != original["definitions_sha256"] or file_hash(JOBS_PATH) != original["jobs_sha256"]:
            raise ValueError("Original definitions or jobs changed")
        redactor = Redactor()
        if redactor.fingerprint != original["redaction_id"]:
            raise ValueError("Original redaction policy changed")
        policies = {p["role"]: p for p in json.loads(definitions.read_text(encoding="utf-8"))["agents"]}
        code = {name: file_hash(Path(teacher.__file__).with_name(name)) for name in original["code"]}
        for name, value in original["code"].items():
            if name != "teacher.py" and code[name] != value:
                raise ValueError("Only the maintenance no-action guard may change during this recovery")
        archived = base / "recovery" / ("runner-" + original["code"]["teacher.py"] + ".py")
        if file_hash(archived) != original["code"]["teacher.py"]:
            raise ValueError("Original runner snapshot changed")
        old = '                if rejected and not accepted:\n                    raise ValueError("All-rejected runs are not successful no-ops")\n'
        current = Path(teacher.__file__).read_text(encoding="utf-8")
        start = current.index('                if rejected and not accepted:\n')
        end = current.index('                student_messages =', start)
        # Prove every byte outside the narrowly scoped final-response guard is
        # unchanged, including rendering, handlers, state, budgets, and tools.
        if archived.read_text(encoding="utf-8").replace(old, current[start:end], 1) != current:
            raise ValueError("Runner changes are not compatible with cached traces")
        combine = lambda c: hashlib.sha256(canonical([c[n] for n in ("teacher.py", "maintenance.py", "replay.py")])).hexdigest()
        old_code, new_code = combine(original["code"]), combine(code)
        snapshot = source.read()
        contract = {"original_log_sha256": hashlib.sha256(snapshot).hexdigest(), "original_run": original,
                    "code": code, "recovery_sha256": file_hash(Path(__file__)),
                    "redaction_id": redactor.fingerprint}
        published = root / "state"
        pin, output = published / "run.json", published / "attempts.jsonl"
        if pin.exists():
            if json.loads(pin.read_text(encoding="utf-8")) != contract:
                raise ValueError("Pinned recovery changed")
        else:
            if published.exists():
                raise ValueError("Unpinned recovery output exists")
            staging = root / "initialization.partial"
            if staging.exists():
                shutil.rmtree(staging)  # Never contains live calls before publication.
            staging.mkdir(mode=0o700)
            latest = {r["job_id"]: r for r in (json.loads(line) for line in snapshot.splitlines())}
            temporary = staging / "attempts.jsonl"
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            imported = repaired = 0
            with os.fdopen(fd, "wb") as target:
                for previous_id, row in latest.items():
                    task = row.get("task", "observer")
                    if task not in policies or row["status"] not in {"completed", "failed"}:
                        continue
                    if row["runner_sha256"] != old_code or row["config"] != original["config"]:
                        raise ValueError("A trace is not from the original pinned runner/configuration")
                    policy = policies[task]
                    clean_policy, job = redactor.tree(policy), redactor.job(row["job"])
                    text, _ = teacher.render(job) if task == "observer" else maintenance.render(job)
                    expected = [{"role": "system", "content": clean_policy["instructions"]}, {"role": "user", "content": text}]
                    if row["messages"][:2] != expected or row["tools"] != [teacher._tool(clean_policy)]:
                        raise ValueError("Cached input/tools no longer reproduce exactly")
                    if row["status"] == "failed":
                        # Revalidate the *saved* provider replies through the fixed
                        # loop. No new response or accepted action is invented.
                        trace = row
                        steps = iter((i, m) for i, m in enumerate(trace["messages"]) if m["role"] == "assistant")
                        usage = iter(trace["usage"])
                        def replay(body, run):
                            i, message = next(steps)
                            if body["messages"] != trace["messages"][:i]:
                                raise ValueError("Saved trace does not replay exactly")
                            return {"choices": [{"finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
                                                  "message": message}], "usage": next(usage)}
                        row = teacher.run_job(job, policy, original["config"], replay, redactor=redactor)
                        if row["status"] != "completed" or not row.get("protocol_recovery") or row["messages"] != trace["messages"]:
                            raise ValueError("Failure is not a recoverable empty maintenance batch")
                        row.update(policy_id=policy["definition_id"], config=original["config"], runner_sha256=new_code)
                        row["record_origin"] = "saved_reply_revalidation_no_provider_call"
                        repaired += 1
                    else:
                        row["record_origin"] = "compatible_cached_trace"
                        imported += 1
                    row["job_id"] = hashlib.sha256(canonical({"job": job, "policy": policy,
                        "config": original["config"], "code": new_code})).hexdigest()
                    row["cache_import"] = {"original_job_id": previous_id, "original_runner_sha256": old_code,
                                           "cache_runner_sha256": new_code}
                    redactor.require_clean(row)
                    target.write(canonical(row) + b"\n")
                target.flush()
                os.fsync(target.fileno())
            # Recomputed generation commits verify the causal state while cached
            # role traces avoid repeating successful paid calls.
            teacher._checkpoint(staging / "run.json", contract)
            teacher._checkpoint(staging / "import.json", {"reused_traces": imported, "revalidated_no_actions": repaired,
                                                          "provider_calls_during_import": 0})
            os.replace(staging, published)
            fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        if not os.environ.get(original["config"]["api_key_env"]):
            raise ValueError("Provider credential is missing")
        return teacher.run_file(JOBS_PATH, output, policies["observer"], original["config"],
                                maintenance_policies={r: policies[r] for r in ("reflector", "dropper")})


if __name__ == "__main__":
    print(json.dumps(continue_replay()))
