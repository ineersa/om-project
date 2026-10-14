import hashlib

import pytest

from om_dataset.hatfield_source import build_blocks, part_digest, source_digest


def event(seq, kind, payload):
    return {"run_id": "r", "seq": seq, "type": kind, "payload": payload,
            "turn_no": 1, "ts": "20261013T120000+0200"}


def sample():
    return [
        event(1, "run_started", {"payload": {"messages": [
            {"role": "system", "content": "hidden"}, {"role": "user-context", "content": "hidden"},
            {"role": "user", "content": [{"text": "Hi"}, "é", {"type": "image"}]}]}}),
        event(2, "llm_step_completed", {"assistant_message": {"content": "A", "tool_calls": [
            {"id": "first", "name": "read", "arguments": {"z": "/é\u2028", "a": {}}},
            {"id": "second", "function": {"name": "read", "arguments": "{}"}}]}}),
        event(3, "llm_step_failed", {"retryable": "0"}),
        event(4, "tool_execution_end", {"tool_result": {"tool_call_id": "first", "is_error": True,
            "result": {"tool_name": "read", "content": "done \n"}}}),
        event(5, "agent_end", {}),
    ]


def test_native_order_refs_timestamp_and_json():
    blocks = build_blocks(list(reversed(sample())))
    assert [b["seq"] for b in blocks] == [1, 3, 2, 5, 2]
    assert [b["kind"] for b in blocks] == ["message", "outcome", "tool_group", "outcome", "tool_call"]
    assert blocks[0]["rendered_text"] == "[Source entry id: r:1]\n[User @ 2026-10-13 12:00]: Hi\né"
    assert blocks[2]["source_refs"] == [{"run_id": "r", "seq": 2}, {"run_id": "r", "seq": 4}]
    assert 'args={"z":"/é\\u2028","a":[]}' in blocks[2]["rendered_text"]
    assert blocks[2]["rendered_text"].endswith("[Tool result for read @ 2026-10-13 12:00] ERROR:\ndone")
    assert "retryable=false" in blocks[1]["rendered_text"]
    assert "reason=completed" in blocks[3]["rendered_text"]
    # Golden hashes from the pinned PHP builder/identity, not the Python implementation.
    assert source_digest(blocks) == "01ecc7a8dce71c34be11c6ce26439b1e06f8773aad2ad7df27ca653dd8c9e6ad"
    assert part_digest(blocks) == "dbafc49366e19d7f60ce40c7e27ac4c9397c0a310b6e7af6bdc856d4056d6ba7"


def test_unicode_tool_digest_boundary_and_immutable_input():
    text = "😀" * 11750 + "Ω"
    e = event(1, "tool_execution_end", {"tool_result": {"result": {"content": text}}})
    body = build_blocks([e])[0]["rendered_text"]
    digest = hashlib.sha256(text.encode()).hexdigest()
    assert body.endswith(text[:400] + f"\n...[tool output digest sha256={digest} chars=11751]...\n" + text[-200:])
    assert e["payload"]["tool_result"]["result"]["content"] == text
    e["payload"]["tool_result"]["result"]["content"] = text[:-1]
    assert "tool output digest" not in build_blocks([e])[0]["rendered_text"]


def test_rejects_unrecoverable_timestamp_and_invalid_result():
    e = event(1, "agent_command_applied", {"message": {"text": "hello"}})
    assert build_blocks([e])[0]["rendered_text"].endswith("hello")
    e["ts"] = None
    with pytest.raises(ValueError):
        build_blocks([e])
    with pytest.raises(ValueError):
        build_blocks([event(1, "tool_execution_end", {})])
    assert source_digest([]) == hashlib.sha256(b"[]").hexdigest()
    assert part_digest([]) == hashlib.sha256(b"").hexdigest()
