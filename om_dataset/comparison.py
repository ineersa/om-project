"""Six fixed-input comparisons against captured long-pilot teacher invocations."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
from http.client import HTTPConnection, HTTPSConnection
import json
import os
from pathlib import Path
import re
import secrets
from urllib.error import HTTPError
from urllib.parse import urlparse

from . import maintenance, teacher
from .collection import canonical, file_hash, write_json
from .redaction import Redactor

MAIN = Path(__file__).resolve().parents[1]
SOURCE = MAIN / "data/teacher-long-pilot/attempts.jsonl"
POLICIES = MAIN / "data/policies-2026-10-13/hatfield-target/definitions.json"
OUTPUT = Path("data/teacher-comparison")
STAGING = Path("data/review-next/cases.jsonl")
IDENTITIES = Path("data/review-next-private/identity-map.json")
MODEL_PATH = "/home/ineersa/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
CONFIG = {"provider": "local-qwen", "model": "flash", "base_url": "http://10.0.0.89:8052/v1",
          "context_window": 131072, "request_options": {"max_tokens": 65536},
          "reasoning_control": "server defaults; no effort-level override"}
PROTOCOL = ("Hatfield histories only; two nonempty invocations per task. Maintenance must belong to a committed generation; "
            "drop proposals must intersect committed drops. Exclude every logical source task/range with any non-completed attempt. "
            "Choose smallest initial serialized input estimate, then cutoff, then source job ID. "
            "Fixed purposive six-case sample, not an unbiased win-rate benchmark.")


def _hash(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _rows(path):
    snapshot = path.read_bytes()
    if not snapshot.endswith(b"\n"):
        raise ValueError("Source has an incomplete append; retry after the pilot writer finishes")
    return [json.loads(line) for line in snapshot.splitlines()], hashlib.sha256(snapshot).hexdigest()


def _redactor():
    redactor = Redactor(catalog=MAIN / ".hatfield/security/redaction-values.json")
    if not redactor.catalog_loaded:
        raise ValueError("Private credential catalog required")
    return redactor


def _source_key(row):
    job = row["job"]
    # Deliberately exclude the whole source range/task, including other partitions
    # and changed runner/cache IDs, if any attempt at that source task failed.
    return (job["run_id"], job.get("state_scope"), job.get("historical_range_id") or job["cutoff_seq"], row.get("task", "observer"))


def exact_input(baseline, policy, redactor):
    """Fail before a call/cache use unless current rendering reproduces captured input."""
    redactor.require_clean(baseline)
    job, policy = redactor.job(baseline["job"]), redactor.tree(policy)
    task = baseline.get("task", "observer")
    if task != policy["role"] or baseline["policy_id"] != policy["definition_id"]:
        raise ValueError("Original policy mismatch")
    config = baseline["config"]
    if config["context_window"] != 131072 or config["request_options"]["max_tokens"] != 65536:
        raise ValueError("Baseline does not use the common student/output budget")
    text, _ = teacher.render(job) if task == "observer" else maintenance.render(job)
    messages = [{"role": "system", "content": policy["instructions"]}, {"role": "user", "content": text}]
    tools = [teacher._tool(policy)]
    if baseline["messages"][:2] != messages or baseline.get("tools") != tools:
        raise ValueError("Captured initial messages/tools differ from job + original policy")
    if teacher._context_tokens(messages, tools) + 65536 > 131072:
        raise ValueError("Initial input exceeds common student budget")
    return {"messages": messages, "tools": tools, "messages_sha256": _hash(messages),
            "tools_sha256": _hash(tools), "prior_state_sha256": _hash(job)}


def select_cases(rows, policies, redactor):
    failed = {_source_key(r) for r in rows if r["status"] != "completed"}
    commits = {(r["job"]["run_id"], r["job"].get("state_scope"), r["generation_id"]): r
               for r in rows if r.get("task") == "generation_commit" and r["status"] == "completed"}
    selected = []
    used_inputs = set()
    for _ in range(2):
        for task, field in (("observer", "observations"), ("reflector", "reflections"), ("dropper", "drop_candidates")):
            candidates = []
            for row in rows:
                if row.get("task", "observer") != task or row["status"] != "completed" or _source_key(row) in failed:
                    continue
                job = row["job"]
                if job["run_id"].startswith("pi:") or not row.get(field):
                    continue
                if task != "observer":
                    commit = commits.get((job["run_id"], job.get("state_scope"), job["generation_id"]))
                    if commit is None:
                        continue
                    if task == "reflector" and not {r["reflection_id"] for r in row[field]} <= set(commit["accepted_delta"]["reflection_ids"]):
                        continue
                    if task == "dropper" and not ({o["observation_id"] for o in row[field]} & set(commit["accepted_delta"]["dropped_ids"])):
                        continue
                captured = exact_input(row, policies[task], redactor)
                if (task, captured["messages_sha256"]) in used_inputs:
                    continue
                candidates.append((teacher._context_tokens(captured["messages"], captured["tools"]), job["cutoff_seq"], row["job_id"], row, captured))
            if not candidates:
                raise ValueError("No eligible nonempty committed Hatfield case for " + task)
            _, _, _, baseline, captured = min(candidates, key=lambda c: c[:3])
            used_inputs.add((task, captured["messages_sha256"]))
            selected.append({"case_id": f"hatfield-{len(selected) + 1:03d}", "source_kind": "hatfield",
                             "baseline": baseline, "policy": policies[task], "captured": captured})
    return selected


def _request(method, endpoint, body, redactor, config=CONFIG):
    redactor.require_clean(body)
    url = urlparse(config["base_url"])
    connection = (HTTPSConnection if url.scheme == "https" else HTTPConnection)(url.hostname, url.port, timeout=300)
    try:
        connection.request(method, endpoint, body=json.dumps(body).encode() if body is not None else None,
                           headers={"Authorization": "Bearer dummy", "Content-Type": "application/json"})
        response = connection.getresponse()
        if response.status != 200:
            raise HTTPError(endpoint, response.status, "Local provider request failed", None, None)
        value = json.load(response)
        redactor.require_clean(value)
        return value
    finally:
        connection.close()


def server_properties(redactor):
    props = _request("GET", "/props", None, redactor)
    def strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for item in value.values():
                yield from strings(item)
        elif isinstance(value, list):
            for item in value:
                yield from strings(item)
    if MODEL_PATH not in strings(props):
        raise ValueError("Local server does not identify the expected deployed model")
    # Slot occupancy/sleep state are volatile and must not invalidate a resume.
    return {"model_path": MODEL_PATH, "defaults": props.get("default_generation_settings", {}),
            "total_slots": props.get("total_slots"),
            "chat_template_sha256": _hash(props.get("chat_template", ""))}


def _private_dir(path):
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path, 0o700)


def _verify_result(result, baseline, captured, redactor):
    if (result.get("messages", [])[:2] != captured["messages"] or result.get("tools") != captured["tools"]
            or redactor.job(result["job"]) != redactor.job(baseline["job"])):
        raise ValueError("Result/checkpoint is not from the exact captured input and prior state")


def run_cases(cases, root, redactor, complete=None, properties=None):
    """Resume native tool-round checkpoints; retain every attempt, cache successes."""
    _private_dir(root)
    fd = os.open(root / ".lock", os.O_RDWR | os.O_CREAT, 0o600)
    with os.fdopen(fd, "a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_cases(cases, root, redactor, complete, properties)


def _run_cases(cases, root, redactor, complete, properties):
    if not redactor.catalog_loaded:
        raise ValueError("Private credential catalog required")
    _private_dir(root)
    results = []
    code = {m.__name__: file_hash(Path(m.__file__)) for m in (teacher, maintenance)}
    for case in cases:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", case["case_id"]):
            raise ValueError("Unsafe case ID")
        baseline, policy = case["baseline"], case["policy"]
        captured = exact_input(baseline, policy, redactor)
        if captured != case["captured"]:
            raise ValueError("Fixed case input changed")
        key = _hash({"job": baseline["job"], "policy": policy, "config": CONFIG, "code": code,
                     "comparison_code": file_hash(Path(__file__)),
                     "server_properties": properties, "redaction_id": redactor.fingerprint})
        directory = root / case["case_id"]
        _private_dir(directory)
        request = directory / "request.json"
        if request.exists():
            if json.loads(request.read_text(encoding="utf-8"))["comparison_key"] != key:
                raise ValueError("Comparison request changed; refusing to restart an unfinished case implicitly")
        else:
            write_json(request, {"comparison_key": key})
        cached, checkpoint = directory / "completed.json", directory / ("checkpoint-" + key + ".json")
        if cached.exists():
            result = json.loads(cached.read_text(encoding="utf-8"))
            redactor.require_clean(result)
            if result["comparison_key"] != key:
                raise ValueError("Cached comparison has changed inputs/configuration/runner")
        else:
            def generate(body, run):
                if body["messages"][:2] != captured["messages"] or body["tools"] != captured["tools"]:
                    raise ValueError("Provider body differs from captured initial input")
                redactor.require_clean(body)
                return complete(body, run) if complete else _request(
                    "POST", urlparse(CONFIG["base_url"]).path.rstrip("/") + "/chat/completions", body, redactor)
            result = teacher.run_job(baseline["job"], policy, CONFIG, generate, checkpoint, redactor)
            result.update(comparison_key=key, policy_id=policy["definition_id"], config=CONFIG, captured_input=captured,
                          server_properties_sha256=_hash(properties),
                          source_job_id=baseline["job_id"], recorded_at=datetime.now(timezone.utc).isoformat())
            redactor.require_clean(result)
            attempts = list(directory.glob("attempt-*.json"))
            write_json(directory / f"attempt-{len(attempts) + 1:04d}.json", result)
            if result["status"] == "completed":
                write_json(cached, result)
        _verify_result(result, baseline, captured, redactor)
        if result["status"] == "completed":
            checkpoint.unlink(missing_ok=True)
        results.append(result)
    return results


def _actions(result):
    task, status = result["task"], result["status"]
    field = {"observer": "observations", "reflector": "reflections", "dropper": "drop_candidates"}[task]
    actions = result.get(field, [])
    heading = "Completed with no recorded actions." if status == "completed" and not actions else "Status: " + status
    if status != "completed":
        heading += ". Failed/incomplete trace, NOT a valid no-op."
    if result.get("rejected_items"):
        heading += f"\nRecorded rejections: {result['rejected_items']}. This trace requires a recovery mask."
    if task == "dropper":
        heading += "\nThese are per-partition proposals, NOT final committed generation drops."
    lines = [heading]
    local, observations = {}, {}
    if task != "observer":
        _, allowed = maintenance.render(result["job"])
        local = {o["observation_id"]: key for key, o in allowed.items()}
        observations = {o["observation_id"]: o for o in result["job"]["observations"]}
    for number, action in enumerate(actions, 1):
        if task == "observer":
            refs = ", ".join(f"{r['run_id']}:{r['seq']}" for r in action["source_refs"])
            lines.append(f"{number}. [{action['relevance']}] {action['timestamp']}\n{action['content']}\nEvidence: {refs}")
        elif task == "reflector":
            support = ", ".join(local[oid] for oid in action["supporting_observation_ids"])
            lines.append(f"{number}. {action['content']}\nSupporting observations: {support}")
        else:
            oid = action["observation_id"]
            observation = observations[oid]
            lines.append(f"{number}. Propose dropping observation [{local[oid]}]: "
                         f"[{observation['relevance']}] {observation['timestamp']}\n{observation['content']}")
    if not actions:
        lines.append("No action recorded.")
        final = result.get("messages", [])[-1:]
        if final and final[0]["role"] == "assistant" and final[0].get("content"):
            lines.append("Final response: " + final[0]["content"])
    # Failed/rejected tool attempts remain visible without duplicating every
    # accepted action and its provider-generated call IDs in the review UI.
    calls = {}
    for message in result.get("messages", [])[2:]:
        for call in message.get("tool_calls") or []:
            calls[call["id"]] = call["function"]
        if message["role"] == "tool":
            receipt = json.loads(message["content"])
            if receipt["status"] != "rejected" and not receipt.get("rejected", receipt.get("missing", 0)):
                continue
            function = calls.get(message["tool_call_id"], {})
            try:
                arguments = json.loads(function.get("arguments", "null"))
            except (ValueError, TypeError):
                arguments = function.get("arguments")
            lines.append("Rejected tool attempt: " + json.dumps({"tool": function.get("name"), "arguments": arguments,
                                                                  "receipt": receipt}, ensure_ascii=False))
    if status != "completed":
        lines.append("Error: " + json.dumps(result.get("error", {})))
    return "\n\n".join(lines)


def export_cases(cases, results, redactor, staging=STAGING, identities=IDENTITIES):
    if staging.exists() or identities.exists():
        raise FileExistsError("Staged pack/mapping already exists; never reshuffle or overwrite")
    if not cases or len(results) != len(cases) or len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("Controlled pack requires paired invocations with unique case IDs")
    pack, mapping = [], []
    for case, result in zip(cases, results, strict=True):
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", case["case_id"]):
            raise ValueError("Unsafe case ID")
        baseline, job = case["baseline"], case["baseline"]["job"]
        if case["source_kind"] != "hatfield" or job["run_id"].startswith("pi:"):
            raise ValueError("Judge review cases must use Hatfield histories only")
        captured = exact_input(baseline, case["policy"], redactor)
        _verify_result(result, baseline, captured, redactor)
        task = baseline.get("task", "observer")
        choices = [("captured_deepseek", _actions(baseline)), ("local_qwen", _actions(result))]
        if secrets.randbelow(2):
            choices.reverse()
        evidence = [{"source_id": "captured-input", "text": captured["messages"][1]["content"]}]
        prior = "The captured input in Evidence includes the exact visible prior memory. Both candidates received that same state; it is not repeated here."
        scope = "Identical captured initial messages, tool schemas, and prior state; single invocations, not end-to-end replay or an equal-compute benchmark. Purposive six-case sample, not an unbiased win-rate benchmark."
        if task == "dropper":
            scope += " Compare proposals for this input partition only, not the entire committed generation."
        public = {"schema_version": "om-review-case/1", "case_id": case["case_id"], "task": task,
                  "evidence": evidence, "prior_memory": prior,
                  "candidates": dict(zip(("A", "B"), [choice[1] for choice in choices])),
                  "redaction_id": redactor.fingerprint, "comparison_scope": scope}
        redactor.require_clean(public)
        pack.append(public)
        mapping.append({"case_id": case["case_id"], "A": choices[0][0], "B": choices[1][0],
                        "source_job_id": baseline["job_id"], "messages_sha256": captured["messages_sha256"],
                        "tools_sha256": captured["tools_sha256"], "prior_state_sha256": captured["prior_state_sha256"],
                        "baseline_config": baseline["config"], "local_config": CONFIG})
    _private_dir(staging.parent)
    _private_dir(identities.parent)
    fd = os.open(identities, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    encoded = b"".join(canonical(row) + b"\n" for row in pack)
    with os.fdopen(fd, "wb") as output:
        output.write(canonical({"protocol": PROTOCOL, "cases": mapping,
                                "cases_sha256": hashlib.sha256(encoded).hexdigest()}) + b"\n")
        output.flush()
        os.fsync(output.fileno())
    # Mapping is durable before publication; interrupted publication fails closed.
    temporary = staging.with_suffix(".jsonl.partial")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as output:
        for row in pack:
            output.write(canonical(row) + b"\n")
        output.flush()
        os.fsync(output.fileno())
    os.link(temporary, staging)  # Atomic publication that cannot replace an existing pack.
    temporary.unlink()
    return {"cases": len(pack), "staging": str(staging)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    redactor = _redactor()
    policies = {p["role"]: p for p in json.loads(POLICIES.read_text(encoding="utf-8"))["agents"]}
    rows, source_hash = _rows(SOURCE)
    _private_dir(OUTPUT)
    manifest = OUTPUT / "manifest.json"
    if manifest.exists():
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        redactor.require_clean(saved)
        if saved["local_config"] != CONFIG:
            raise ValueError("Fixed local comparison configuration changed")
        cases = saved["cases"]
        if any(c["source_kind"] != "hatfield" for c in cases):
            raise ValueError("Legacy mixed-source pack; use python -m om_dataset.hatfield_review to replace Pi cases")
        failed = {_source_key(r) for r in rows if r["status"] != "completed"}
        if any(_source_key(c["baseline"]) in failed for c in cases):
            raise ValueError("A fixed source case now has a failed attempt; do not silently reselect")
        for case in cases:
            if case["policy"] != policies[case["baseline"].get("task", "observer")]:
                raise ValueError("Original policy changed")
            exact_input(case["baseline"], case["policy"], redactor)
    else:
        cases = select_cases(rows, policies, redactor)
        saved = {"protocol": PROTOCOL, "source_sha256": source_hash, "cases": cases,
                 "local_config": CONFIG, "redaction_id": redactor.fingerprint}
        redactor.require_clean(saved)
        write_json(manifest, saved)
    if args.dry_run:
        print(json.dumps({"cases": len(cases), "provider_calls": 0, "protocol": PROTOCOL}))
        return 0
    if STAGING.exists() or IDENTITIES.exists():
        raise FileExistsError("Staged pack/mapping already exists")
    props = server_properties(redactor)
    props_path = OUTPUT / ("server-properties-" + _hash(props) + ".json")
    if not props_path.exists():
        write_json(props_path, {"props": props, "config": CONFIG})
    results = run_cases(cases, OUTPUT / "results", redactor, properties=props)
    print(json.dumps(export_cases(cases, results, redactor)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
