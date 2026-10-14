"""Small, Python-only Observer runner for prepared development message bags."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from datetime import datetime
from http.client import HTTPSConnection
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlparse

from jsonschema import ValidationError, validate

from .collection import canonical, file_hash
from .policies import validate_policies


PROVIDER = "opencode-go"  # Change to "deepseek" to use the direct API.
JOBS_PATH = Path("data/ranges-2026-10-13/pilot-jobs.jsonl")
OUTPUT_PATH = Path("data/teacher-pilot/attempts.jsonl")
POLICIES_PATH = Path("data/policies-2026-10-13")
STUDENT_CONTEXT_WINDOW = 131_072  # LFM; the DeepSeek teacher supports 1M.
PROVIDERS = {
    "opencode-go": {"base_url": "https://opencode.ai/zen/go/v1", "model": "deepseek-v4.1-flash",
                    "api_key_env": "OPENCODE_API_KEY"},
    "deepseek": {"base_url": "https://api.deepseek.com", "model": "deepseek-flash",
                 "api_key_env": "DEEPSEEK_API_KEY"},
}
REQUEST_OPTIONS = {"max_tokens": 65536, "thinking": {"type": "enabled"}, "reasoning_effort": "low"}


def complete_request(body: dict, run_id: str, config: dict, timeout: int) -> dict:
    url = urlparse(config["base_url"])
    endpoint = url.path.rstrip("/") + "/chat/completions"
    headers = {"Authorization": "Bearer " + os.environ[config["api_key_env"]],
               "Content-Type": "application/json"}
    if config["provider"] == "opencode-go":
        # Same headers as Hatfield's OpenCodeSessionHttpClient, with this client's name.
        headers.update({"x-opencode-session": run_id, "x-opencode-client": "om-project",
                        "User-Agent": "om-project"})
    connection = HTTPSConnection(url.hostname, timeout=timeout)
    try:
        connection.request("POST", endpoint, body=json.dumps(body).encode(), headers=headers)
        response = connection.getresponse()
        if response.status != 200:
            raise HTTPError(endpoint, response.status, "Provider request failed", None, None)
        return json.load(response)
    finally:
        connection.close()


def render(job: dict) -> tuple[str, set[tuple[str, int]]]:
    """Render only the supplied source prefix; never load historical memory labels."""
    run = job["run_id"]
    sources = job["sources"]
    if not isinstance(run, str) or not run or not sources:
        raise ValueError("A run and a nonempty source chunk are required")
    datetime.strptime(job["local_time"], "%Y-%m-%d %H:%M")
    allowed, blocks = set(), []
    last_seq = 0
    for source in sources:
        seq = source["seq"]
        native = source.get("native_rendered") is True
        if type(seq) is not int or not 1 <= seq <= job["cutoff_seq"] or (not native and seq <= last_seq):
            raise ValueError("Sources must be ordered and within the declared cutoff")
        last_seq = seq
        datetime.strptime(source["timestamp"], "%Y-%m-%d %H:%M")
        if source["role"] not in {"User", "Assistant", "Tool result"} or not isinstance(source["text"], str):
            raise ValueError("Invalid source block")
        if native:
            # Native tool groups can emit out of seq order and contain both call/result refs.
            refs = source["source_refs"]
            if not refs or any(r["run_id"] != run or type(r["seq"]) is not int or
                               not 1 <= r["seq"] <= job["cutoff_seq"] for r in refs):
                raise ValueError("Invalid native source references")
            allowed.update((r["run_id"], r["seq"]) for r in refs)
            label = f"[Source entry id: {run}:{seq}]"
            blocks.append(source["text"] if source["text"].startswith(label) else label + "\n" + source["text"])
            continue
        allowed.add((run, seq))
        role = source["role"]
        if role == "Tool result":
            role += " for " + source.get("tool_name", "tool")
        blocks.append(f"[Source entry id: {run}:{seq}]\n[{role} @ {source['timestamp']}]: {source['text']}")
    reflections = "\n".join(job.get("prior_reflections", [])) or "(none yet)"
    observations = "\n".join(job.get("prior_observations", [])) or "(none yet)"
    text = (f"CURRENT REFLECTIONS:\n{reflections}\n\nCURRENT OBSERVATIONS:\n{observations}"
            + "\n\nNEW CONVERSATION CHUNK:\n" + "\n\n".join(blocks)
            + "\n\nCurrent local time fallback: " + job["local_time"])
    return text, allowed


def record(arguments: str, accepted: dict, run: str, allowed: set, schema: dict) -> dict:
    """Partial acceptance and progress receipts; stricter schema/calendar checks for labels."""
    added = duplicates = rejected = 0
    error = None
    message = "Observations accepted. Continue if uncovered content remains; stop when the chunk is fully covered."
    try:
        args = json.loads(arguments)
        if not isinstance(args, dict) or set(args) != {"observations"} or not isinstance(args["observations"], list):
            raise ValueError("Invalid arguments")
    except (ValueError, TypeError):
        args = {"observations": []}
        rejected, error = 1, "invalid_arguments"
        message = "record_observations requires observations as a list (array)."
    for raw in args["observations"]:
        try:
            validate(raw, schema["properties"]["observations"]["items"])
            content = raw["content"].strip(" \t\n\r\x00\x0b")
            if not content or "\n" in content or "\r" in content:
                raise ValueError("Invalid content")
            datetime.strptime(raw["timestamp"], "%Y-%m-%d %H:%M")
            refs = {(ref["run_id"], ref["seq"]) for ref in raw["source_refs"]}
            if not refs or not refs <= allowed:
                raise ValueError("Unknown evidence")
            refs = [{"run_id": r, "seq": s} for r, s in sorted(refs)]
            # Match the pinned identity formula and PHP's insertion-order JSON encoding.
            identity = {"type": "observation-v1", "run_id": run, "observer_schema_version": "1",
                        "timestamp": raw["timestamp"], "content": content, "source_refs": refs}
            encoded = json.dumps(identity, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            encoded = encoded.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
            oid = hashlib.sha256(encoded.encode()).hexdigest()
            if oid in accepted:
                duplicates += 1
            else:
                accepted[oid] = {**raw, "content": content, "source_refs": refs, "observation_id": oid}
                added += 1
        except (ValidationError, ValueError, TypeError, KeyError):
            rejected += 1
    if rejected and not (added or duplicates):
        error = error or "all_rejected"
        if error == "all_rejected":
            message = "All observations in this call were rejected. Fix timestamps/relevance/source_refs and retry."
    elif rejected:
        message = "Some observations were rejected; accepted ones were kept. Fix rejected items and continue."
    receipt = {"status": "rejected" if error else "accepted", "added": added,
               "duplicates": duplicates, "rejected": rejected, "total": len(accepted), "message": message}
    if error:
        receipt["error"] = error
    return receipt


def slice_job(job: dict, policy: dict, config: dict) -> list[dict]:
    """Pack all sources using Hatfield's context ratio and token estimate, without truncation."""
    try:
        render(job)
    except (ValueError, KeyError, TypeError):
        return [job]  # Let run_job save the validation failure without calling the provider.
    estimate = lambda text: math.ceil(len(text) / 2.35)
    envelope = math.floor(config["context_window"] * 0.65)
    envelope = min(envelope, config["context_window"] - config["request_options"].get("max_tokens", 0))
    overhead = estimate(policy["instructions"]) + estimate(json.dumps(policy["tools"], ensure_ascii=False)) + 32
    reflections = list(job.get("prior_reflections", []))
    observations = list(job.get("prior_observations", []))
    memory_budget = math.floor((envelope - overhead) * 0.55)
    # Prior memory arrives as rendered lines, not native timestamp/rating objects.
    # Trim in supplied observation order, then reverse reflection order; keep raw job unchanged.
    while estimate("\n".join(reflections + observations)) > memory_budget:
        if observations:
            observations.pop(0)
        elif reflections:
            reflections.pop()
        else:
            break
    base = {**job, "prior_reflections": reflections, "prior_observations": observations}

    def fits(sources):
        return overhead + estimate(render({**base, "sources": sources})[0]) <= envelope

    chunks, group = [], []
    for source in job["sources"]:
        if fits(group + [source]):
            group.append(source)
            continue
        if group:
            chunks.append({**base, "sources": group})
            group = []
        if fits([source]):
            group = [source]
            continue
        empty = {**source, "text": ""}
        framing = estimate(render({**base, "sources": [empty]})[0])
        chars = math.floor((envelope - overhead - framing) * 2.35)
        if chars < 1:
            raise ValueError("Fixed prompt cannot fit the model context")
        for start in range(0, len(source["text"]), chars):
            end = min(start + chars, len(source["text"]))
            part = {**source, "text": source["text"][start:end], "text_span": [start, end]}
            chunks.append({**base, "sources": [part]})
    if group:
        chunks.append({**base, "sources": group})
    return [{**chunk, "chunk_index": index, "chunk_count": len(chunks)}
            for index, chunk in enumerate(chunks)]


def run_job(job: dict, policy: dict, config: dict, complete) -> dict:
    messages, accepted, usage = [], {}, []
    rejected = calls = 0
    result = {"status": "failed", "quality_status": "unreviewed", "training_ready": False,
              "reconstruction": "fresh_replay", "renderer": "python-observer-pilot/1", "job": job}
    tool = {"type": "function", "function": {k: policy["tools"][0][k] for k in ("name", "description", "parameters")}}
    result["tools"] = [tool]
    try:
        text, allowed = render(job)
        messages = [{"role": "system", "content": policy["instructions"]}, {"role": "user", "content": text}]
        seen_ids = set()
        while True:
            response = complete({**config["request_options"], "model": config["model"], "messages": messages,
                                 "tools": [tool], "tool_choice": "auto"}, job["run_id"])
            usage.append(response.get("usage", {}))
            choice = response["choices"][0]
            if choice["finish_reason"] not in {"stop", "tool_calls"}:
                raise ValueError("Incomplete provider response")
            raw = choice["message"]
            if raw.get("role") != "assistant":
                raise ValueError("Expected an assistant response")
            message = {k: raw[k] for k in ("role", "content", "tool_calls", "reasoning_content") if k in raw}
            messages.append(message)
            proposed = raw.get("tool_calls") or []
            if not proposed:
                if choice["finish_reason"] != "stop" or not isinstance(raw.get("content"), str) or not raw["content"].strip():
                    raise ValueError("Missing final completion")
                if rejected and not accepted:
                    raise ValueError("All-rejected runs are not successful no-ops")
                result["status"] = "completed"
                break
            if choice["finish_reason"] != "tool_calls":
                raise ValueError("Inconsistent tool-call finish reason")
            for call in proposed:
                calls += 1
                cid = call["id"]
                if not isinstance(cid, str) or not cid or cid in seen_ids:
                    raise ValueError("Invalid call identity")
                seen_ids.add(cid)
                function = call["function"]
                if call.get("type") != "function" or function["name"] != tool["function"]["name"]:
                    raise ValueError("Unauthorized tool")
                receipt = record(function["arguments"], accepted, job["run_id"], allowed, tool["function"]["parameters"])
                rejected += receipt["rejected"]
                messages.append({"role": "tool", "tool_call_id": cid, "content": json.dumps(receipt)})
    except Exception as error:
        # Never persist provider exception text: it may contain credentials or request bodies.
        result["error"] = {"type": type(error).__name__}
        if isinstance(error, HTTPError):
            result["error"]["http_status"] = error.code
    # Some DeepSeek modes require reasoning_content on continuation. Keep it transient, not a training target.
    result.update(messages=[{k: v for k, v in m.items() if k != "reasoning_content"} for m in messages],
                  observations=list(accepted.values()), usage=usage, tool_calls=calls, rejected_items=rejected,
                  requires_recovery_mask=bool(rejected))
    return result


def run_file(input_path: Path, output_path: Path, policy: dict, config: dict, complete) -> dict:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output must differ")
    output_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(output_path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o600)
    summary = {"completed": 0, "failed": 0, "skipped": 0}
    with os.fdopen(fd, "a+") as output:
        fcntl.flock(output, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.fchmod(output.fileno(), 0o600)
        output.seek(0)
        seen = {r["job_id"] for line in output if (r := json.loads(line))["status"] == "completed"}
        code_hash = file_hash(Path(__file__))
        for line in input_path.open():
            if not line.strip():
                continue
            for job in slice_job(json.loads(line), policy, config):
                key = hashlib.sha256(canonical({"job": job, "policy": policy, "config": config, "code": code_hash})).hexdigest()
                if key in seen:
                    summary["skipped"] += 1
                    continue
                result = run_job(job, policy, config, complete)
                result.update(job_id=key, policy_id=policy["definition_id"], config=config, runner_sha256=code_hash)
                output.write(canonical(result).decode() + "\n")
                output.flush()
                os.fsync(output.fileno())
                summary[result["status"]] += 1
                if result["status"] == "completed":
                    seen.add(key)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run a small Python-only Observer pilot. Outputs require review.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and render jobs without credentials or network calls.")
    args = parser.parse_args(argv)
    try:
        validate_policies(POLICIES_PATH)
        bundle = json.loads((POLICIES_PATH / "hatfield-target/definitions.json").read_text())
        policy = next(p for p in bundle["agents"] if p["role"] == "observer")
        config = {**PROVIDERS[PROVIDER], "provider": PROVIDER, "request_options": REQUEST_OPTIONS,
                  "context_window": STUDENT_CONTEXT_WINDOW}
        if args.dry_run:
            count = chunks = 0
            for line in JOBS_PATH.open():
                if line.strip():
                    job = json.loads(line)
                    render(job)
                    chunks += len(slice_job(job, policy, config))
                    count += 1
            print(json.dumps({"validated_jobs": count, "chunks": chunks, "model_calls": 0}))
            return 0
        if not os.environ.get(config["api_key_env"]):
            parser.error(f"Missing {config['api_key_env']}")

        def complete(body, run_id):
            return complete_request(body, run_id, config, policy["call_options"]["maxDurationSeconds"])

        summary = run_file(JOBS_PATH, OUTPUT_PATH, policy, config, complete)
        print(json.dumps(summary))
        return 1 if summary["failed"] else 0
    except Exception as error:
        print(f"Observer pilot failed ({type(error).__name__}); no request text printed.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
