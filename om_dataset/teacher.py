"""Python-only OM tool loop and chronological Observer replay."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import time
from datetime import datetime
from http.client import HTTPSConnection
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlparse

from jsonschema import ValidationError, validate

from .collection import canonical, file_hash
from . import maintenance
from .policies import validate_policies
from .redaction import Redactor


PROVIDER = "opencode-go"  # Change to "deepseek" to use the direct API.
JOBS_PATH = Path("data/teacher-pilot/two-session-jobs.jsonl")
OUTPUT_PATH = Path("data/teacher-pilot/two-session-attempts.jsonl")
POLICIES_PATH = Path("data/policies-2026-10-13")
STUDENT_CONTEXT_WINDOW = 131_072  # LFM; the DeepSeek teacher supports 1M.
PROVIDERS = {
    "opencode-go": {"base_url": "https://opencode.ai/zen/go/v1", "model": "deepseek-v4.1-flash",
                    "api_key_env": "OPENCODE_API_KEY"},
    "deepseek": {"base_url": "https://api.deepseek.com", "model": "deepseek-flash",
                 "api_key_env": "DEEPSEEK_API_KEY"},
}
REQUEST_OPTIONS = {"max_tokens": 65536, "thinking": {"type": "enabled"}, "reasoning_effort": "low"}
RUN_LIMITS = {"max_duration_seconds": 900, "max_completion_tokens": 131_072, "max_no_progress_turns": 3}


class BudgetExceeded(Exception):
    """An attempt exhausted a declared resource budget, not a successful no-op."""


def _tool(policy):
    return {"type": "function", "function": {k: policy["tools"][0][k] for k in ("name", "description", "parameters")}}


def _tokens(value):
    return math.ceil(len(json.dumps(value, ensure_ascii=False)) / 2.35) + 32


def _context_tokens(messages, tools):
    return _tokens({"messages": messages, "tools": tools, "tool_choice": "auto"})


def complete_request(body: dict, run_id: str, config: dict, timeout: int) -> dict:
    Redactor().require_clean(body)
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
            if "continuation_frame" in source:
                start, end = source["text_span"]
                blocks.append(source["continuation_frame"] + f"\n[Source continuation: original chars {start}:{end}]\n" + source["text"])
            else:
                blocks.append(source["text"] if source["text"].startswith(label) else label + "\n" + source["text"])
            continue
        allowed.add((run, seq))
        role = source["role"]
        if role == "Tool result":
            role += " for " + source.get("tool_name", "tool")
        continuation = f"[Source continuation: original chars {source['text_span'][0]}:{source['text_span'][1]}]\n" if "text_span" in source else ""
        blocks.append(f"[Source entry id: {run}:{seq}]\n[{role} @ {source['timestamp']}]: {continuation}{source['text']}")
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


def _segments(source, run):
    """Use trusted producer spans; old single-event blocks can use their first header."""
    if not source.get("native_rendered"):
        return [{"start": 0, "end": len(source["text"])}]
    segments = source.get("segments")
    if segments is None:
        # Do not scan quoted content for headers or infer a multi-event tool group's framing.
        label = f"[Source entry id: {run}:{source['seq']}]\n"
        body = source["text"].removeprefix(label)
        header = re.match(r"^\[([^\n\]]+) @ (\d{4}-\d{2}-\d{2} \d{2}:\d{2})\](?: ERROR)?:[ \n]?", body)
        if not source["text"].startswith(label) or len(source["source_refs"]) != 1 or header is None:
            raise ValueError("Oversized native blocks require trusted segment metadata")
        segments = [{"start": 0, "end": len(source["text"]), "framing": label + header.group(0)}]
    position = 0
    for segment in segments:
        if (segment["start"] != position or not position < segment["end"] <= len(source["text"])
                or not isinstance(segment["framing"], str) or not segment["framing"].startswith(f"[Source entry id: {run}:{source['seq']}]")):
            raise ValueError("Invalid native segment metadata")
        position = segment["end"]
    if position != len(source["text"]):
        raise ValueError("Incomplete native segment metadata")
    return segments


def slice_job(job: dict, policy: dict, config: dict) -> list[dict]:
    """Pack source bytes and repeat trusted framing on continuation fragments."""
    try:
        render(job)
    except (ValueError, KeyError, TypeError):
        return [job]  # Let run_job save the validation failure without calling the provider.
    estimate = lambda text: math.ceil(len(text) / 2.35)
    # Keep continuation headroom inside the input allowance, after reserving output.
    envelope = math.floor((config["context_window"] - config["request_options"].get("max_tokens", 0)) * 0.65)
    overhead = _context_tokens([{"role": "system", "content": policy["instructions"]}], [_tool(policy)])
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
        messages = [{"role": "system", "content": policy["instructions"]},
                    {"role": "user", "content": render({**base, "sources": sources})[0]}]
        return _context_tokens(messages, [_tool(policy)]) <= envelope

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
        for segment in _segments(source, job["run_id"]):
            start = segment["start"]
            while start < segment["end"]:
                def fragment(end):
                    part = {**source, "text": source["text"][start:end], "text_span": [start, end]}
                    if source.get("native_rendered"):
                        part["continuation_frame"] = segment["framing"]
                    return part
                low, high = start, segment["end"]
                while low < high:
                    middle = (low + high + 1) // 2
                    if fits([fragment(middle)]):
                        low = middle
                    else:
                        high = middle - 1
                if low == start:
                    raise BudgetExceeded("initial_context")
                chunks.append({**base, "sources": [fragment(low)]})
                start = low
    if group:
        chunks.append({**base, "sources": group})
    return [{**chunk, "chunk_index": index, "chunk_count": len(chunks)}
            for index, chunk in enumerate(chunks)]


def _checkpoint(path, data):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(path.name + ".partial")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as output:
        json.dump(data, output, ensure_ascii=False, allow_nan=False)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def run_job(job: dict, policy: dict, config: dict, complete, checkpoint_path: Path | None = None, redactor=None) -> dict:
    redactor = redactor or Redactor()
    job = redactor.job(job)
    policy = redactor.tree(policy)
    messages, accepted, usage = [], {}, []
    rejected = calls = spent = no_progress = 0
    limits = {**RUN_LIMITS, **config.get("run_limits", {})}
    started = time.monotonic()
    elapsed = 0
    deadline = started + limits["max_duration_seconds"]
    result = {"status": "failed", "quality_status": "unreviewed", "training_ready": False,
              "reconstruction": "fresh_replay", "renderer": "python-observer-pilot/2", "job": job,
              "limits": limits, "request_contexts": [], "redaction_id": redactor.fingerprint}
    task = job.get("task", policy.get("role", "observer"))
    result["task"] = task
    tool = _tool(policy)
    result["tools"] = [tool]

    def save_progress(finished=None):
        if checkpoint_path is not None:
            # Private continuation state may contain reasoning required by the provider.
            # It is never a training record and is deleted after the result is durable.
            data = {"messages": messages, "accepted": accepted, "usage": usage,
                        "rejected": rejected, "calls": calls, "spent": spent, "no_progress": no_progress,
                        "elapsed": elapsed + time.monotonic() - started, "seen_ids": sorted(seen_ids),
                        "request_contexts": result["request_contexts"], "finished": finished}
            redactor.require_clean(data)
            _checkpoint(checkpoint_path, data)

    try:
        if "_slice_budget_error" in job:
            raise BudgetExceeded(job["_slice_budget_error"])
        if task not in {"observer", "reflector", "dropper"} or policy.get("role", task) != task:
            raise ValueError("Task and policy do not match")
        text, allowed = render(job) if task == "observer" else maintenance.render(job)
        messages = [{"role": "system", "content": policy["instructions"]}, {"role": "user", "content": text}]
        seen_ids = set()
        if checkpoint_path is not None and checkpoint_path.exists():
            saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            redactor.require_clean(saved)
            if saved["finished"] is not None:
                return {**saved["finished"], "resumed_from_checkpoint": True}
            messages, accepted, usage = saved["messages"], saved["accepted"], saved["usage"]
            rejected, calls, spent = saved["rejected"], saved["calls"], saved["spent"]
            no_progress, elapsed, seen_ids = saved["no_progress"], saved["elapsed"], set(saved["seen_ids"])
            result["request_contexts"] = saved["request_contexts"]
            result["resumed_from_checkpoint"] = True
            deadline = started + limits["max_duration_seconds"] - elapsed
        while True:
            if no_progress >= limits["max_no_progress_turns"]:
                result.update(status="stalled", error={"type": "NoProgress", "consecutive_turns": no_progress})
                break
            remaining = deadline - time.monotonic()
            reserve = config["request_options"].get("max_tokens", 0)
            context = _context_tokens(messages, [tool])
            if remaining <= 0:
                raise BudgetExceeded("deadline")
            if context + reserve > config["context_window"]:
                raise BudgetExceeded("context_window")
            if spent + reserve > limits["max_completion_tokens"]:
                raise BudgetExceeded("completion_tokens")
            result["request_contexts"].append({"estimated_input_tokens": context, "output_allowance": reserve})
            body = {**config["request_options"], "model": config["model"], "messages": messages,
                    "tools": [tool], "tool_choice": "auto"}
            redactor.require_clean(body)
            if complete is None:
                timeout = min(remaining, policy["call_options"]["maxDurationSeconds"])
                response = complete_request(body, job["run_id"], config, timeout)
            else:
                response = complete(body, job["run_id"])
            usage.append(response.get("usage", {}))
            choice = response["choices"][0]
            if choice["finish_reason"] not in {"stop", "tool_calls"}:
                raise ValueError("Incomplete provider response")
            raw = redactor.tree(choice["message"])
            if raw.get("role") != "assistant":
                raise ValueError("Expected an assistant response")
            message = {k: raw[k] for k in ("role", "content", "tool_calls", "reasoning_content") if k in raw}
            messages.append(message)
            completion_tokens = response.get("usage", {}).get("completion_tokens")
            spent += completion_tokens if type(completion_tokens) is int and completion_tokens >= 0 else _tokens(message)
            if time.monotonic() >= deadline:
                raise BudgetExceeded("deadline")
            if spent > limits["max_completion_tokens"]:
                raise BudgetExceeded("completion_tokens")
            proposed = raw.get("tool_calls") or []
            if not proposed:
                if choice["finish_reason"] != "stop" or not isinstance(raw.get("content"), str) or not raw["content"].strip():
                    raise ValueError("Missing final completion")
                if rejected and not accepted:
                    empty_field = {"reflector": "reflections", "dropper": "ids"}.get(task)
                    empty_only = empty_field is not None and all(
                        (args := json.loads(call["function"]["arguments"])).get(empty_field) == []
                        and set(args) <= ({empty_field, "reason"} if task == "dropper" else {empty_field})
                        and ("reason" not in args or isinstance(args["reason"], str))
                        for m in messages for call in (m.get("tool_calls") or []))
                    if not empty_only:
                        raise ValueError("All-rejected runs are not successful no-ops")
                    # No state changed. Preserve the rejection and require a
                    # recovery mask; this is not a clean training target.
                    result["protocol_recovery"] = "empty_maintenance_call_then_no_action"
                student_messages = [{k: v for k, v in m.items() if k != "reasoning_content"} for m in messages]
                if _context_tokens(student_messages, [tool]) > config["context_window"]:
                    raise BudgetExceeded("student_sequence")
                result["status"] = "completed"
                break
            if choice["finish_reason"] != "tool_calls":
                raise ValueError("Inconsistent tool-call finish reason")
            before = len(accepted)
            for call in proposed:
                calls += 1
                cid = call["id"]
                if not isinstance(cid, str) or not cid or cid in seen_ids:
                    raise ValueError("Invalid call identity")
                seen_ids.add(cid)
                function = call["function"]
                if call.get("type") != "function" or function["name"] != tool["function"]["name"]:
                    raise ValueError("Unauthorized tool")
                if task == "observer":
                    receipt = record(function["arguments"], accepted, job["run_id"], allowed, tool["function"]["parameters"])
                else:
                    receipt = maintenance.record(function["arguments"], accepted, job["run_id"], allowed,
                                                 tool["function"]["parameters"], job)
                rejected += receipt.get("rejected", receipt.get("missing", int(receipt["status"] == "rejected")))
                messages.append({"role": "tool", "tool_call_id": cid, "content": json.dumps(receipt)})
            no_progress = no_progress + 1 if len(accepted) == before else 0
            save_progress()
            if no_progress >= limits["max_no_progress_turns"]:
                result.update(status="stalled", error={"type": "NoProgress", "consecutive_turns": no_progress})
                break
    except BudgetExceeded as error:
        result.update(status="budget_exhausted", error={"type": "BudgetExceeded", "reason": str(error)})
    except Exception as error:
        # Never persist provider exception text: it may contain credentials or request bodies.
        result["error"] = {"type": type(error).__name__}
        if isinstance(error, HTTPError):
            result["error"]["http_status"] = error.code
    # Some DeepSeek modes require reasoning_content on continuation. Keep it transient, not a training target.
    result.update(messages=[{k: v for k, v in m.items() if k != "reasoning_content"} for m in messages],
                  observations=list(accepted.values()) if task == "observer" else [], usage=usage, tool_calls=calls, rejected_items=rejected,
                  requires_recovery_mask=bool(rejected), completion_tokens=spent)
    if task != "observer":
        result["reflections" if task == "reflector" else "drop_candidates"] = list(accepted.values())
    student_tokens = _context_tokens(result["messages"], [tool])
    result["student_context"] = {"estimated_tokens": student_tokens, "context_window": config["context_window"],
                                 "fits": student_tokens <= config["context_window"],
                                 "estimator": "serialized-json-characters/2.35+32", "exact_tokenizer_checked": False}
    if result["status"] == "completed":
        save_progress(finished=result)
    return result


def _resume(output, path, redactor):
    """Recover only a torn, unterminated tail; a corrupt complete line is an error."""
    completed, recovered = {}, 0
    output.seek(0)
    while True:
        start = output.tell()
        line = output.readline()
        if not line:
            break
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if line.endswith(b"\n") or output.read(1):
                raise ValueError("Corrupt attempt log record; refusing to skip it") from None
            quarantine = path.with_name(path.name + ".torn-" + hashlib.sha256(line).hexdigest()[:16])
            if quarantine.exists():
                if quarantine.read_bytes() != line:
                    raise ValueError("Conflicting tail quarantine")
            else:
                fd = os.open(quarantine, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, "wb") as tail:
                    tail.write(line)
                    tail.flush()
                    os.fsync(tail.fileno())
            output.truncate(start)
            output.flush()
            os.fsync(output.fileno())
            recovered = 1
            break
        redactor.require_clean(row)
        if row["status"] == "completed":
            completed[row["job_id"]] = row
        if not line.endswith(b"\n"):
            output.write(b"\n")
            output.flush()
            os.fsync(output.fileno())
    return completed, recovered


def _memory_line(observation):
    return f"[{observation['observation_id']}] {observation['timestamp']} [{observation['relevance']}] {observation['content']}"


def _advance(state, observations, cutoff):
    merged = {o["observation_id"]: o for o in state["observations"]}
    for observation in observations:
        merged.setdefault(observation["observation_id"], observation)
    return {**state, "observations": list(merged.values()), "cutoff_seq": cutoff}


def run_file(input_path: Path, output_path: Path, policy: dict, config: dict, complete=None,
             maintenance_policies: dict | None = None) -> dict:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output must differ")
    redactor = Redactor()
    if complete is None and not redactor.catalog_loaded:
        raise ValueError("The private credential-redaction catalog is required before live labeling")
    output_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(output_path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o600)
    summary = dict.fromkeys(("completed", "failed", "skipped", "stalled", "budget_exhausted", "blocked", "recovered_tails"), 0)
    states, blocked = {}, set()
    with os.fdopen(fd, "a+b") as output:
        fcntl.flock(output, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.fchmod(output.fileno(), 0o600)
        seen, summary["recovered_tails"] = _resume(output, output_path, redactor)
        code_hash = file_hash(Path(__file__))
        if maintenance_policies is not None:
            from .replay import maintain, _progress
            code_hash = hashlib.sha256(canonical([code_hash, file_hash(Path(maintenance.__file__)),
                                                 file_hash(Path(__file__).with_name("replay.py"))])).hexdigest()
        for line in input_path.open(encoding="utf-8"):
            if not line.strip():
                continue
            original = redactor.job(json.loads(line))
            if original.get("historical_range_id") and not original.get("state_scope"):
                raise ValueError("Historical replay requires an explicit archive/branch state_scope")
            scope = (original["run_id"], original.get("state_scope", ""))
            state = states.setdefault(scope, {"seed_observations": original.get("prior_observations", []),
                                             "reflections": original.get("prior_reflections", []),
                                             "observations": [], "cutoff_seq": 0})
            if maintenance_policies is not None and (state["seed_observations"] or original.get("prior_reflections")):
                raise ValueError("Full replay requires structured fresh state, not unlinked seed memory")
            if state["cutoff_seq"] >= original["cutoff_seq"]:
                raise ValueError("Replay jobs must advance chronologically within each state_scope")
            before_id = hashlib.sha256(canonical(state)).hexdigest()
            if original.get("state_parent_id", before_id) != before_id:
                raise ValueError("Replay state parent does not match")
            base = {**original, "state_before_id": before_id,
                    "prior_observations": state["seed_observations"] + [_memory_line(o) for o in state["observations"]],
                    "prior_reflections": [f"[{r['reflection_id']}] {r['content']}" for r in state["reflections"]]
                        if maintenance_policies is not None else list(state["reflections"])}
            if state["cutoff_seq"]:
                base["prior_memory_status"] = "teacher_generated_replay"
            try:
                chunks = slice_job(base, redactor.tree(policy), config)
            except BudgetExceeded as error:
                chunks = [{**base, "_slice_budget_error": str(error)}]
            pending, successful = [], scope not in blocked
            for index, job in enumerate(chunks):
                key = hashlib.sha256(canonical({"job": job, "policy": policy, "config": config, "code": code_hash})).hexdigest()
                checkpoint = output_path.with_name(output_path.name + ".checkpoints") / (key + ".json")
                if key in seen:
                    checkpoint.unlink(missing_ok=True)
                    pending.extend(seen[key]["observations"])
                    summary["skipped"] += 1
                    continue
                if successful:
                    result = run_job(job, policy, config, complete, checkpoint, redactor)
                else:
                    result = {"status": "blocked", "error": {"type": "PredecessorIncomplete"}, "job": job,
                              "observations": [], "quality_status": "unreviewed", "training_ready": False}
                result.update(job_id=key, policy_id=policy["definition_id"], config=config, runner_sha256=code_hash)
                if result["status"] == "completed":
                    pending.extend(result["observations"])
                    if index == len(chunks) - 1:
                        after = _advance(state, pending, original["cutoff_seq"])
                        result["state_after"] = {"state_id": hashlib.sha256(canonical(after)).hexdigest(),
                                                 "observation_ids": [o["observation_id"] for o in after["observations"]]}
                else:
                    successful = False
                redactor.require_clean(result)
                output.write(canonical(result) + b"\n")
                output.flush()
                os.fsync(output.fileno())
                if maintenance_policies is not None:
                    _progress(result)
                summary[result["status"]] += 1
                if result["status"] == "completed":
                    seen[key] = {"status": "completed", "observations": result["observations"]}
                    checkpoint.unlink(missing_ok=True)
            if successful:
                after = _advance(state, pending, original["cutoff_seq"])
                if maintenance_policies is not None:
                    after = maintain(after, original, maintenance_policies, config, complete, output,
                                     output_path, seen, summary, code_hash, redactor)
                if after is not None:
                    states[scope] = after
                else:
                    blocked.add(scope)
            else:
                blocked.add(scope)
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
            redactor = Redactor()
            for line in JOBS_PATH.open(encoding="utf-8"):
                if line.strip():
                    job = redactor.job(json.loads(line))
                    render(job)
                    chunks += len(slice_job(job, redactor.tree(policy), config))
                    count += 1
            print(json.dumps({"validated_jobs": count, "chunks": chunks, "model_calls": 0}))
            return 0
        if not os.environ.get(config["api_key_env"]):
            parser.error(f"Missing {config['api_key_env']}")

        summary = run_file(JOBS_PATH, OUTPUT_PATH, policy, config)
        print(json.dumps(summary))
        return 1 if any(summary[s] for s in ("failed", "stalled", "budget_exhausted", "blocked")) else 0
    except Exception as error:
        print(f"Observer pilot failed ({type(error).__name__}); no request text printed.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
