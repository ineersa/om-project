"""Source rendering for Hatfield 5ea2fd920d04eccee75094afe033302c5596393c.

Digest verification remains mandatory: this is not a general PHP serializer.
Missing/non-ISO timestamps are rejected instead of reproducing PHP's current-time fallback.
PHP float formatting and numeric-object-key conversions can still differ.
"""
from datetime import datetime
import hashlib
import json
import math


def _get(value, key, default=None):
    item = value.get(key) if isinstance(value, dict) else None
    return default if item is None else item


def _string(value):
    if isinstance(value, (dict, list)):
        raise ValueError("Non-scalar PHP string conversion is unsupported")
    return "" if value is None or value is False else "1" if value is True else str(value)


def _json(value):
    # PHP decodes objects as associative arrays; an empty object re-encodes as [].
    def arrays(item):
        if isinstance(item, dict):
            if not item:
                return []
            return {k: arrays(v) for k, v in item.items()}
        if isinstance(item, list):
            return [arrays(v) for v in item]
        return item
    return json.dumps(arrays(value), ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace(
        "\u2028", "\\u2028").replace("\u2029", "\\u2029")


def _text(message):
    content = _get(message, "content")
    if isinstance(content, str):
        return content
    if isinstance(content, (list, dict)):
        parts = content.values() if isinstance(content, dict) else content
        return "\n".join(part if isinstance(part, str) else _get(part, "text")
                         for part in parts if isinstance(part, str) or isinstance(_get(part, "text"), str))
    text = _get(message, "text")
    return text if isinstance(text, str) else _json(message)


def _timestamp(event):
    # EventPayloadNormalizer parses ts, then SessionEventDTO receives ATOM format.
    # Preserve the offset's wall-clock time; do not convert to UTC or slice raw ts.
    ts = event.get("ts")
    if not isinstance(ts, str) or not ts:
        raise ValueError("Cannot reconstruct a current-time timestamp fallback")
    try:
        parsed = datetime.fromisoformat(ts)
    except ValueError:
        raise ValueError("Unsupported historical timestamp") from None
    return parsed.isoformat(timespec="seconds").replace("T", " ")[:16]


def _block(event, kind, body, refs=None):
    return {"run_id": event["run_id"], "seq": event["seq"], "kind": kind,
            "rendered_text": f"[Source entry id: {event['run_id']}:{event['seq']}]\n" + body.rstrip(" \t\n\r\0\x0b"),
            "source_refs": refs if refs is not None else [{"run_id": event["run_id"], "seq": event["seq"]}]}


def build_blocks(events: list[dict], chars_per_token: float = 2.35) -> list[dict]:
    """Render events in native emission order, including delayed/unmatched tool calls."""
    pending, blocks = {}, []
    for event in sorted(events, key=lambda e: e["seq"]):
        timestamp, p, kind = _timestamp(event), event["payload"], event["type"]
        def role_line(role, text):
            role = _string(role)
            label = {"user": "User", "assistant": "Assistant", "tool": "Tool"}.get(role.lower())
            if label is None:
                label = (role[:1].upper() if role[:1] in "abcdefghijklmnopqrstuvwxyz" else role[:1]) + role[1:]
            return f"[{label} @ {timestamp}]: {text}"
        if kind == "run_started":
            messages = _get(_get(p, "payload"), "messages", _get(p, "messages"))
            if isinstance(messages, (dict, list)):
                for message in messages.values() if isinstance(messages, dict) else messages:
                    if isinstance(message, (dict, list)):
                        role = _string(_get(message, "role", "user"))
                        if role not in {"system", "developer", "user-context"}:
                            blocks.append(_block(event, "message", role_line(role, _text(message))))
        elif kind == "agent_command_applied":
            text = _string(_get(p, "text", ""))
            message = _get(p, "message")
            if not text and isinstance(message, (dict, list)):
                text = _text(message)
            blocks.append(_block(event, "user", role_line("User", text)))
        elif kind == "llm_step_completed":
            assistant, parts = _get(p, "assistant_message"), []
            if isinstance(assistant, (dict, list)):
                parts.append(role_line("Assistant", _text(assistant)))
                calls = _get(assistant, "tool_calls", [])
                if isinstance(calls, (dict, list)):
                    for call in calls.values() if isinstance(calls, dict) else calls:
                        if not isinstance(call, (dict, list)):
                            continue
                        cid = _string(_get(call, "id", _get(call, "tool_call_id", "")))
                        function = _get(call, "function")
                        name = _string(_get(call, "name", _get(function, "name", "tool")))
                        args = _get(call, "arguments", _get(function, "arguments", []))
                        parts.append(f"[Tool call {cid} name={name} args={_json(args)}]")
                        if cid:
                            pending[cid] = (event, parts)
                            parts = []
            elif isinstance(_get(p, "text"), str):
                parts.append(role_line("Assistant", p["text"]))
            if parts:
                blocks.append(_block(event, "assistant", "\n".join(parts)))
        elif kind == "tool_execution_end":
            typed = _get(p, "tool_result")
            if not isinstance(typed, (dict, list)):
                raise ValueError("ToolExecutionEnd requires an array tool_result")
            cid, result = _string(_get(typed, "tool_call_id", "")), _get(typed, "result", [])
            result = result if isinstance(result, (dict, list)) else []
            text, name = _text(result), _string(_get(result, "tool_name", "tool"))
            if math.ceil(len(text) / chars_per_token) > 5000:
                text = text[:400] + f"\n...[tool output digest sha256={_sha(text)} chars={len(text)}]...\n" + text[-200:]
            error = " ERROR" if _get(typed, "is_error") is True else ""
            line = f"[Tool result for {name} @ {timestamp}]{error}:\n{text}"
            if cid and cid in pending:
                call_event, parts = pending.pop(cid)
                refs = {(e["run_id"], e["seq"]) for e in (call_event, event)}
                normalized = [{"run_id": run, "seq": seq} for run, seq in sorted(refs) if run and seq >= 1]
                blocks.append(_block(call_event, "tool_group", "\n".join(parts) + "\n" + line, normalized))
            else:
                blocks.append(_block(event, "tool_result", line))
        elif kind == "llm_step_failed":
            retryable = _get(p, "retryable", False) not in (None, False, 0, "", "0", [], {})
            blocks.append(_block(event, "outcome", f"[Outcome @ {timestamp}]: llm_step_failed retryable={str(retryable).lower()}"))
        elif kind == "agent_end":
            blocks.append(_block(event, "outcome", f"[Outcome @ {timestamp}]: agent_end reason={_string(_get(p, 'reason', 'completed'))}"))
    for event, parts in pending.values():
        blocks.append(_block(event, "tool_call", "\n".join(parts)))
    return blocks


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source_digest(blocks) -> str:
    """Hash native insertion-ordered JSON, excluding source_refs."""
    return _sha(_json([{k: block[k] for k in ("run_id", "seq", "kind", "rendered_text")} for block in blocks]))


def part_digest(blocks) -> str:
    """Hash an unsplit group's rendered UTF-8 bytes joined by two newlines."""
    return _sha("\n\n".join(block["rendered_text"] for block in blocks))
