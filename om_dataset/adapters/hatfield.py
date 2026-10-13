"""Read-only Hatfield archive adapters. Candidate records are not training labels."""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import Counter
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any

KNOWN_VERSIONS = {
    "20260722_001_domain", "20260725_002_reflection_multi_and_indexes",
    "20260726_003_active_generation_and_relevance_text",
    "20260729_004_current_activity", "20260922_005_semantic_health",
}
RELEVANCE = {"low", "medium", "high", "critical"}


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError("Non-JSON numeric constant")


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _issue(code: str, **location: Any) -> dict:
    return {"code": code, **location}


def _semantic(event: dict) -> tuple[str | None, Any]:
    payload = event.get("payload")
    if not isinstance(payload, dict):
        return None, payload
    kind = event.get("type")
    if kind == "llm_step_completed" and "assistant_message" in payload:
        return "assistant", _message_content(payload["assistant_message"])
    if kind == "tool_execution_end" and "tool_result" in payload:
        return "tool", _message_content(payload["tool_result"])
    if kind == "agent_command_applied":
        command = payload.get("command")
        if isinstance(command, dict) and "message" in command:
            message = command["message"]
        elif payload.get("kind") in {"steer", "follow_up", "append_message"} and "message" in payload:
            message = payload["message"]
        else:
            message = None
        if message is not None:
            role = message.get("role") if isinstance(message, dict) else None
            return role if role in {"user", "assistant", "tool"} else None, _message_content(message)
    # Queued commands must not masquerade as applied user messages.
    # run_started input remains unprojected until its runtime shape is verified.
    return None, _message_content(payload)


def _message_content(message: Any) -> Any:
    if isinstance(message, dict):
        return {key: value for key, value in message.items()
                if key not in {"seq", "ts", "timestamp", "created_at", "usage", "source_model"}}
    return message


def parse_hatfield(path: Path) -> dict:
    """Index every journal line without returning conversation text.

    A complete boundary requires a newline-terminated JSON object with integer
    seq. A valid boundary additionally requires a supported, ordered envelope.
    Sequence gaps are allowed. Byte pointers address the supplied raw archive.
    """
    sources, issues = [], []
    session_id = None
    seen: set[tuple[str, int]] = set()
    previous_seq = None
    last_complete_seq = last_valid_seq = None
    counts: Counter = Counter()
    offset = 0
    with Path(path).open("rb") as stream:
        for line_no, raw in enumerate(stream, 1):
            item = {"line_no": line_no, "byte_offset": offset, "byte_length": len(raw),
                    "native_id": None, "parent_id": None, "native_type": None,
                    "role": None, "source_ref": None, "content_sha256": None, "valid": False}
            offset += len(raw)
            sources.append(item)
            counts["lines"] += 1
            errors = []
            if not raw.endswith(b"\n"):
                errors.append("incomplete_line")
            try:
                event = json.loads(raw, parse_constant=_reject_constant)
            except (UnicodeDecodeError, ValueError):
                errors.append("malformed_json")
                event = None
            if not isinstance(event, dict):
                if event is not None:
                    errors.append("event_not_object")
                issues.extend(_issue(code, line_no=line_no) for code in errors)
                counts["invalid_lines"] += 1
                continue
            seq, run = event.get("seq"), event.get("run_id")
            item["native_id"] = seq
            item["native_type"] = event.get("type")
            item["role"], semantic = _semantic(event)
            try:
                item["content_sha256"] = _hash(semantic)
            except UnicodeEncodeError:
                errors.append("invalid_unicode_content")
            item["semantic_hash_kind"] = (
                "message" if item["role"] is not None else "unprojected_payload")
            if event.get("schema_version") != "1.0":
                errors.append("unknown_event_schema")
            if not _text(event.get("type")) or not isinstance(event.get("payload"), dict):
                errors.append("invalid_event_payload")
            if not _text(run):
                errors.append("missing_run_id")
            else:
                if session_id is None:
                    session_id = run
                if run != session_id:
                    errors.append("mixed_run_id")
            if not _integer(seq):
                errors.append("invalid_seq")
            else:
                if raw.endswith(b"\n"):
                    last_complete_seq = seq
                if previous_seq is not None and seq <= previous_seq:
                    errors.append("non_monotonic_seq")
                previous_seq = seq if previous_seq is None else max(previous_seq, seq)
                if _text(run):
                    item["source_ref"] = {"run_id": run, "seq": seq}
                    if (run, seq) in seen:
                        errors.append("duplicate_source_ref")
                    seen.add((run, seq))
            item["valid"] = not errors
            if not errors:
                last_valid_seq = seq
                counts["valid_events"] += 1
            else:
                counts["invalid_lines"] += 1
                issues.extend(_issue(code, line_no=line_no) for code in errors)
    return {"session_id": session_id, "header": {}, "sources": sources,
            "memory_candidates": [], "issues": issues,
            "stats": {**dict(counts), "bytes": offset,
                      "last_complete_seq": last_complete_seq,
                      "last_valid_seq": last_valid_seq}}


def _refs(value: Any, errors: list[str]) -> list[dict]:
    try:
        result = json.loads(value) if isinstance(value, str) else None
    except (ValueError, TypeError):
        result = None
    if not isinstance(result, list) or not result:
        errors.append("invalid_source_refs_json")
        return []
    if any(not isinstance(ref, dict) or set(ref) != {"run_id", "seq"}
           or not _text(ref.get("run_id")) or not _integer(ref.get("seq"))
           for ref in result):
        errors.append("invalid_source_ref")
        return []
    if len({(r["run_id"], r["seq"]) for r in result}) != len(result):
        errors.append("duplicate_source_ref")
    return result


def _strings(value: Any, errors: list[str]) -> list[str]:
    try:
        result = json.loads(value) if isinstance(value, str) else None
    except (ValueError, TypeError):
        result = None
    if not isinstance(result, list) or not result or any(not _text(x) for x in result):
        errors.append("invalid_supporting_observation_ids_json")
        return []
    if len(set(result)) != len(result):
        errors.append("duplicate_support_id")
    return result


def _content(row: dict, errors: list[str]) -> None:
    value = row.get("content")
    if not _text(value):
        errors.append("invalid_content")
    elif any(c in value for c in ("\n", "\r")):
        errors.append("multiline_content")
    if not _integer(row.get("token_count")):
        errors.append("invalid_token_count")


def _observation(row: dict, index: int, sources: dict[str, set[int]]) -> dict:
    errors: list[str] = []
    _content(row, errors)
    for name in ("observation_id", "run_id", "boundary_key", "observer_model",
                 "observer_schema_version", "created_at"):
        if not _text(row.get(name)):
            errors.append("invalid_" + name)
    start, end = row.get("source_start_seq"), row.get("source_end_seq")
    if not _integer(start) or not _integer(end) or start > end:
        errors.append("invalid_source_range")
    timestamp = row.get("timestamp")
    if not isinstance(timestamp, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", timestamp):
        errors.append("invalid_timestamp")
    else:
        try:
            datetime.strptime(timestamp, "%Y-%m-%d %H:%M")
        except ValueError:
            errors.append("invalid_timestamp")
    relevance = row.get("relevance")
    transformations = []
    if _integer(relevance):
        relevance = ("low" if relevance <= 24 else "medium" if relevance <= 49
                     else "high" if relevance <= 74 else "critical")
        transformations.append({"field": "relevance", "policy": "hatfield_20260726_integer_bins",
                                "from": row["relevance"], "to": relevance})
    if not isinstance(relevance, str) or relevance not in RELEVANCE:
        errors.append("invalid_relevance")
    content = row.get("content")
    if isinstance(content, str):
        try:
            if row.get("content_hash") != hashlib.sha256(content.encode()).hexdigest():
                errors.append("content_hash_mismatch")
        except UnicodeEncodeError:
            errors.append("invalid_unicode_content")
    else:
        errors.append("invalid_content_hash")
    refs = _refs(row.get("source_refs_json"), errors)
    missing = [ref for ref in refs if ref["seq"] not in sources.get(ref["run_id"], set())]
    if missing:
        errors.append("unresolved_source_refs")
    status = ("unavailable" if not refs or all(r["run_id"] not in sources for r in refs)
              else "missing" if missing else "resolved")
    return {"task": "observer", "adapter": "hatfield",
            "native_memory_id": row.get("observation_id"),
            "native_entry_id": row.get("boundary_key"), "line_no": None,
            "record_index": index, "native_record": row,
            "coverage": {"source_start_seq": start, "source_end_seq": end,
                         "boundary_key": row.get("boundary_key")},
            "observation": {"timestamp": timestamp, "content": content,
                            "relevance": relevance, "source_refs": refs},
            "transformations": transformations, "unresolved_source_refs": missing,
            "evidence_status": status, "valid": not errors, "validation_errors": errors}


def _reflection(row: dict, index: int, observations: set[tuple[str, str]]) -> dict:
    errors: list[str] = []
    _content(row, errors)
    # Threshold-triggered reflections use an empty compaction request ID.
    # Preserve it; do not claim that every reflection came from compaction.
    for name in ("reflection_id", "run_id",
                 "reflector_model", "reflector_schema_version", "created_at"):
        if not _text(row.get(name)):
            errors.append("invalid_" + name)
    if row.get("compaction_request_id") is not None and not isinstance(row["compaction_request_id"], str):
        errors.append("invalid_compaction_request_id")
    supports = _strings(row.get("supporting_observation_ids_json"), errors)
    missing = [oid for oid in supports if (row.get("run_id"), oid) not in observations]
    if missing:
        errors.append("unresolved_support_ids")
    return {"task": "reflector", "adapter": "hatfield",
            "native_memory_id": row.get("reflection_id"),
            "native_entry_id": row.get("compaction_request_id"), "line_no": None,
            "record_index": index, "native_record": row,
            "reflection": {"content": row.get("content")},
            "native_supporting_observation_ids": supports,
            "unresolved_support_ids": missing,
            "evidence_status": "unavailable" if not supports else "missing" if missing else "resolved",
            "valid": not errors, "validation_errors": errors}


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def parse_hatfield_db(path: Path, sources_by_run: dict[str, set[int]]) -> dict:
    """Parse an already archived SQLite snapshot without modifying it."""
    candidates, issues, run_ids = [], [], set()
    tables, versions = {}, []
    stats: dict = {"tables": tables, "schema_versions": versions}
    # immutable is safe only because the caller supplies an archived snapshot.
    with closing(sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro&immutable=1", uri=True)) as db:
        db.execute("PRAGMA query_only = ON")
        db.row_factory = sqlite3.Row
        names = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
                 if r[0].startswith("om_")]
        for name in names:
            columns = [r[1] for r in db.execute("PRAGMA table_info(" + _quote(name) + ")")]
            count = db.execute("SELECT count(*) FROM " + _quote(name)).fetchone()[0]
            tables[name] = {"columns": columns, "row_count": count}
            if "run_id" in columns:
                run_ids.update(r[0] for r in db.execute("SELECT DISTINCT run_id FROM " + _quote(name))
                               if _text(r[0]))
        if "om_schema_version" in names:
            for row in db.execute("SELECT * FROM om_schema_version"):
                version = {k: row[k] for k in ("version", "description", "checksum", "applied_at")
                           if k in row.keys()}
                versions.append(version)
                if version.get("version") not in KNOWN_VERSIONS:
                    issues.append(_issue("unknown_db_schema_version", version=version.get("version")))
        else:
            issues.append(_issue("missing_schema_version_table"))
        required = {
            "om_observation": {"observation_id", "run_id", "boundary_key", "source_start_seq",
                               "source_end_seq", "source_refs_json", "content", "content_hash",
                               "relevance", "token_count", "observer_model", "observer_schema_version",
                               "created_at"},
            "om_reflection": {"reflection_id", "run_id", "compaction_request_id", "content",
                              "supporting_observation_ids_json", "token_count", "reflector_model",
                              "reflector_schema_version", "created_at"},
        }
        observation_ids: set[tuple[str, str]] = set()
        if "om_observation" in names and {"run_id", "observation_id"} <= set(tables["om_observation"]["columns"]):
            observation_ids = {(r[0], r[1]) for r in db.execute("SELECT run_id, observation_id FROM om_observation")}
        for table, columns in required.items():
            if table not in names:
                issues.append(_issue("missing_memory_table", table=table))
                continue
            missing = sorted(columns - set(tables[table]["columns"]))
            if missing:
                issues.append(_issue("missing_required_columns", table=table, columns=missing))
            if table == "om_observation" and "timestamp" not in tables[table]["columns"]:
                issues.append(_issue("missing_observation_timestamp_column", table=table))
            # Read every row even for an unknown schema; the archive preserves it.
            # Missing fields become visible per-row validation failures.
            order_column = "observation_id" if table == "om_observation" else "reflection_id"
            ordering = " ORDER BY " + _quote(order_column) if order_column in tables[table]["columns"] else ""
            for index, sqlrow in enumerate(db.execute("SELECT * FROM " + _quote(table) + ordering)):
                row = dict(sqlrow)
                candidate = (_observation(row, index, sources_by_run) if table == "om_observation"
                             else _reflection(row, index, observation_ids))
                if missing:
                    candidate["validation_errors"].append("missing_required_columns")
                    candidate["valid"] = False
                candidates.append(candidate)
                for code in candidate["validation_errors"]:
                    issues.append(_issue(code, table=table, record_index=index,
                                         native_memory_id=candidate["native_memory_id"]))
        for run_id in sorted(run_ids - sources_by_run.keys()):
            issues.append(_issue("missing_source_archive", run_id=run_id))
    stats["candidate_count"] = len(candidates)
    stats["valid_candidate_count"] = sum(c["valid"] for c in candidates)
    stats["db_only_run_ids"] = sorted(run_ids - sources_by_run.keys())
    return {"memory_candidates": candidates, "issues": issues, "stats": stats,
            "run_ids": sorted(run_ids)}
