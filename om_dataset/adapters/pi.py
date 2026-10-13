"""Read-only Pi corpus metadata and memory candidates. No model calls."""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

SOURCE_TYPES = {"message", "custom_message", "branch_summary"}
RELEVANCE = {"low", "medium", "high", "critical"}


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError("Non-JSON numeric constant")


def _finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Non-finite numeric value")
    return result


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip()) and "\n" not in value and "\r" not in value


def _id(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _timestamp(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M").strftime("%Y-%m-%d %H:%M") == value
    except ValueError:
        return False


def _tokens(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0 and value < float("inf")


def _candidate(entry: dict, line: int, adapter: str, task: str, record: Any, index: int | None) -> dict:
    data = entry.get("data")
    coverage = {k: data[k] for k in ("coversFromId", "coversUpToId") if isinstance(data, dict) and k in data}
    return {"task": task, "adapter": adapter, "native_entry_id": entry.get("id"),
            "line_no": line, "native_memory_id": record.get("id") if isinstance(record, dict) else None,
            "record_index": index, "native_record": record, "coverage": coverage,
            "evidence_status": "unavailable", "valid": True, "validation_errors": []}


def _records(entry: dict, line: int, adapter: str, task: str, records: Any,
             current: bool, issue) -> list[dict]:
    if not isinstance(records, list):
        candidate = _candidate(entry, line, adapter, task, records, None)
        candidate["valid"] = False
        candidate["validation_errors"] = ["records_not_array"]
        issue("records_not_array", line)
        return [candidate]
    result = []
    for index, record in enumerate(records):
        c = _candidate(entry, line, adapter, task, record, index)
        errors = c["validation_errors"]
        if not isinstance(record, dict):
            errors.append("record_not_object")
        else:
            if not _id(record.get("id")):
                errors.append("invalid_memory_id")
            elif current and re.fullmatch(r"[a-f0-9]{12}", record["id"]) is None:
                errors.append("invalid_memory_id")
            if not _text(record.get("content")):
                errors.append("invalid_content")
            if task == "observer":
                if not current and not {"id", "timestamp", "content", "relevance", "sourceEntryIds"} <= record.keys():
                    c["summary_only"] = True
                else:
                    c["observation"] = {"timestamp": record.get("timestamp"), "content": record.get("content"),
                                        "relevance": record.get("relevance"), "source_refs": []}
                c["native_source_entry_ids"] = record.get("sourceEntryIds")
                if not _timestamp(record.get("timestamp")):
                    errors.append("invalid_timestamp")
                if not isinstance(record.get("relevance"), str) or record.get("relevance") not in RELEVANCE:
                    errors.append("invalid_relevance")
                refs = record.get("sourceEntryIds")
                if not isinstance(refs, list) or not refs or not all(_id(x) for x in refs):
                    errors.append("invalid_source_entry_ids")
            else:
                c["reflection"] = {"content": record.get("content")}
                c["native_supporting_observation_ids"] = record.get("supportingObservationIds")
                refs = record.get("supportingObservationIds")
                if not isinstance(refs, list) or not refs or not all(_id(x) for x in refs):
                    errors.append("invalid_supporting_observation_ids")
            if current or "tokenCount" in record:
                if not _tokens(record.get("tokenCount")):
                    errors.append("invalid_token_count")
        c["valid"] = not errors
        for error in errors:
            issue(error, line, record_index=index)
        result.append(c)
    return result


def parse_current_memory(entry: dict, line: int, issue) -> list[dict]:
    """Parse current ledger names, validating shape separately from detection."""
    kind = entry.get("customType")
    data = entry.get("data")
    if not isinstance(kind, str) or kind not in {"om.observations.recorded", "om.reflections.recorded", "om.observations.dropped"}:
        return []
    if not isinstance(data, dict):
        issue("invalid_current_data", line)
        c = _candidate(entry, line, "pi-current", "snapshot", data, None)
        c.update(valid=False, validation_errors=["invalid_current_data"])
        return [c]
    if kind == "om.observations.recorded":
        result = _records(entry, line, "pi-current", "observer", data.get("observations"), True, issue)
    elif kind == "om.reflections.recorded":
        result = _records(entry, line, "pi-current", "reflector", data.get("reflections"), True, issue)
    else:
        c = _candidate(entry, line, "pi-current", "dropper", data, None)
        c["native_observation_ids"] = data.get("observationIds")
        ids = data.get("observationIds")
        if not isinstance(ids, list) or not ids or not all(_id(x) for x in ids):
            c.update(valid=False, validation_errors=["invalid_drop_ids"])
            issue("invalid_drop_ids", line)
        result = [c]
    if not _id(data.get("coversUpToId")):
        issue("invalid_coverage", line)
        for c in result:
            c["valid"] = False
            c["validation_errors"].append("invalid_coverage")
    return result


def parse_legacy_memory(entry: dict, line: int, issue) -> list[dict]:
    """Legacy records and older prose summaries stay distinct."""
    if entry.get("customType") != "om.observation":
        return []
    data = entry.get("data")
    if isinstance(data, dict) and "records" in data:
        result = _records(entry, line, "pi-legacy", "observer", data["records"], False, issue)
    else:
        c = _candidate(entry, line, "pi-legacy", "observer", data, None)
        c["summary_only"] = True
        c["native_memory_id"] = None
        if not isinstance(data, dict) or not isinstance(data.get("content"), str) or not data["content"].strip():
            c.update(valid=False, validation_errors=["invalid_legacy_summary"])
            issue("invalid_legacy_summary", line)
        result = [c]
    if isinstance(data, dict):
        for key in ("coversFromId", "coversUpToId"):
            if not _id(data.get(key)):
                issue("invalid_coverage", line, field=key)
                for c in result:
                    c["valid"] = False
                    c["validation_errors"].append("invalid_coverage:" + key)
        if not _tokens(data.get("tokenCount")):
            issue("invalid_aggregate_token_count", line)
            for c in result:
                c["valid"] = False
                c["validation_errors"].append("invalid_aggregate_token_count")
    return result


def parse_pi(path: Path) -> dict:
    """Stream JSONL; byte pointers refer to this exact unmodified file."""
    header = None
    session_id = None
    sources: list[dict] = []
    candidates: list[dict] = []
    issues: list[dict] = []
    types: Counter = Counter()
    nodes: dict[str, dict] = {}
    duplicate_ids: set[str] = set()
    physical_lines = 0
    offset = 0

    def issue(code, line_no, **extra):
        issues.append({"code": code, "line_no": line_no, **extra})

    with Path(path).open("rb") as stream:
        for line_no, raw in enumerate(stream, 1):
            physical_lines = line_no
            start = offset
            offset += len(raw)
            if not raw.endswith(b"\n"):
                issue("incomplete_line", line_no, byte_offset=start)
            try:
                entry = json.loads(raw, parse_constant=_reject_constant, parse_float=_finite_float)
                # JSON allows escaped lone surrogates; UTF-8 model exports do not.
                if b"\\ud" in raw.lower():
                    json.dumps(entry, ensure_ascii=False).encode("utf-8")
            except (ValueError, UnicodeError):
                issue("malformed_json", line_no, byte_offset=start, byte_length=len(raw))
                continue
            if not isinstance(entry, dict) or not isinstance(entry.get("type"), str):
                issue("invalid_entry_shape", line_no, byte_offset=start)
                continue
            if entry["type"] == "session":
                if header is not None:
                    issue("multiple_headers", line_no)
                else:
                    header = entry
                    session_id = entry.get("id") if _id(entry.get("id")) else None
                    if session_id is None:
                        issue("invalid_session_id", line_no)
                    if entry.get("version") != 3:
                        issue("unsupported_session_version", line_no)
                continue
            native_type = entry["type"]
            types[native_type] += 1
            node = {"line_no": line_no, "byte_offset": start, "byte_length": len(raw),
                    "native_id": entry.get("id"), "parent_id": entry.get("parentId"),
                    "native_type": native_type, "custom_type": entry.get("customType"),
                    "role": entry.get("role"), "content_sha256": None, "source_ref": None}
            message = entry.get("message")
            if isinstance(message, dict):
                node["role"] = message.get("role")
            if native_type in SOURCE_TYPES:
                value = entry.get("message", entry.get("content", entry.get("summary")))
                node["content_sha256"] = _hash(value)
            sources.append(node)
            native_id = node["native_id"]
            if not _id(native_id):
                issue("invalid_entry_id", line_no)
            elif native_id in nodes:
                duplicate_ids.add(native_id)
                issue("duplicate_entry_id", line_no, native_id=native_id)
            else:
                nodes[native_id] = node
            if native_type == "custom":
                candidates.extend(parse_current_memory(entry, line_no, issue))
                candidates.extend(parse_legacy_memory(entry, line_no, issue))
                custom_type = entry.get("customType")
                if isinstance(custom_type, str) and custom_type.startswith("om.") and custom_type not in {
                    "om.observations.recorded", "om.reflections.recorded", "om.observations.dropped", "om.observation"
                }:
                    issue("unsupported_om_custom_type", line_no, custom_type=custom_type)
            details = entry.get("details")
            if native_type == "compaction" and isinstance(details, dict) and isinstance(details.get("type"), str) and details.get("type") in {"om.folded", "observational-memory"}:
                adapter = "pi-current" if details["type"] == "om.folded" else "pi-legacy"
                c = _candidate(entry, line_no, adapter, "snapshot", details, None)
                supported = {1} if adapter == "pi-current" else {2, 4}
                if details.get("version") not in supported or not isinstance(details.get("observations"), list) or not isinstance(details.get("reflections"), list) or (adapter == "pi-current" and not isinstance(details.get("fullFold"), bool)):
                    c.update(valid=False, validation_errors=["invalid_snapshot_shape"])
                    issue("invalid_snapshot_shape", line_no)
                elif adapter == "pi-current" or details["version"] == 4:
                    for task, key in (("observer", "observations"), ("reflector", "reflections")):
                        checks = _records(entry, line_no, adapter, task, details[key], adapter == "pi-current", issue)
                        for checked in checks:
                            c["validation_errors"].extend(checked["validation_errors"])
                    c["valid"] = not c["validation_errors"]
                else:
                    for key in ("observations", "reflections"):
                        for index, record in enumerate(details[key]):
                            if not isinstance(record, dict) or not isinstance(record.get("content"), str) or not record["content"].strip() or not _tokens(record.get("tokenCount")):
                                c["valid"] = False
                                c["validation_errors"].append("invalid_legacy_snapshot_record")
                                issue("invalid_legacy_snapshot_record", line_no, record_index=index)
                candidates.append(c)
    if header is None:
        issue("missing_header", 0)
    for node in sources:
        node["source_ref"] = {"run_id": "pi:" + session_id, "seq": node["line_no"]} if session_id else None
        parent = node["parent_id"]
        if parent is None:
            continue
        if not _id(parent):
            issue("invalid_parent_id", node["line_no"])
        elif parent in duplicate_ids:
            issue("ambiguous_parent", node["line_no"], native_id=parent)
        elif parent not in nodes:
            issue("missing_parent", node["line_no"], native_id=parent)
        elif nodes[parent]["line_no"] >= node["line_no"]:
            issue("forward_parent", node["line_no"], native_id=parent)

    def ancestry(native_id):
        chain = []
        seen = set()
        valid = True
        while native_id is not None:
            if not _id(native_id) or native_id not in nodes or native_id in duplicate_ids:
                valid = False
                break
            if native_id in seen:
                valid = False
                break
            seen.add(native_id)
            node = nodes[native_id]
            chain.append(native_id)
            parent = node["parent_id"]
            if _id(parent) and parent in nodes and nodes[parent]["line_no"] >= node["line_no"]:
                valid = False
            native_id = parent
        return list(reversed(chain)), valid

    # Validate every component, including abandoned branches. Avoid recursive traversal.
    completed = set()
    for native_id in nodes:
        if native_id in completed:
            continue
        seen = set()
        trail = []
        current = native_id
        while _id(current) and current in nodes and current not in completed:
            if current in seen:
                issue("parent_cycle", nodes[current]["line_no"], native_id=current)
                break
            seen.add(current)
            trail.append(current)
            current = nodes[current]["parent_id"]
        completed.update(trail)
    for c in candidates:
        if c["task"] != "observer" or "observation" not in c:
            continue
        refs = c.get("native_source_entry_ids")
        if not isinstance(refs, list) or not refs:
            continue
        branch, branch_valid = ancestry(c["native_entry_id"])
        branch_ids = set(branch)
        statuses = []
        evidence = []
        for ref in refs:
            if not _id(ref) or ref not in nodes or ref in duplicate_ids or nodes[ref]["native_type"] not in SOURCE_TYPES:
                code, status = "unresolved_source_ref", "missing"
            elif nodes[ref]["line_no"] >= c["line_no"]:
                code, status = "future_source_ref", "missing"
            elif not branch_valid:
                code, status = "invalid_evidence_branch", "unavailable"
            elif ref not in branch_ids:
                code, status = "off_branch_source_ref", "off_branch"
            elif session_id is None:
                code, status = "missing_source_namespace", "unavailable"
            else:
                code, status = None, "resolved"
            statuses.append(status)
            if code:
                evidence.append({"code": code, "native_id": ref})
                issue(code, c["line_no"], native_id=ref, record_index=c["record_index"])
            else:
                c["observation"]["source_refs"].append(nodes[ref]["source_ref"])
        c["evidence_issues"] = evidence
        c["evidence_status"] = next((x for x in ("missing", "unavailable", "off_branch") if x in statuses), "resolved")
        if evidence:
            c["valid"] = False
            c["validation_errors"].extend(item["code"] for item in evidence)
    parents = {n["parent_id"] for n in sources if _id(n["parent_id"])}
    leaf_ids = [native_id for native_id in nodes if native_id not in parents and native_id not in duplicate_ids]
    default_leaf = sources[-1]["native_id"] if sources else None
    branch, branch_valid = ancestry(default_leaf)
    if not branch_valid:
        branch = []
    return {"session_id": session_id, "header": header, "sources": sources,
            "memory_candidates": candidates, "issues": issues,
            "stats": {"physical_lines": physical_lines, "byte_length": offset, "entry_count": len(sources),
                      "native_type_counts": dict(types), "memory_candidate_count": len(candidates),
                      "default_leaf_id": default_leaf, "leaf_ids": leaf_ids,
                      "default_branch_ids": branch, "default_branch_valid": branch_valid,
                      "ancestry_lookup": {key: node["parent_id"] for key, node in nodes.items()},
                      "ambiguous_native_ids": sorted(duplicate_ids), "memory_outcome": "unknown"}}
