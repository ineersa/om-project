import json

from om_dataset.collection import collect
from om_dataset.curation import curate_collection
from om_dataset.hatfield_source import build_blocks, part_digest, source_digest
from om_dataset.ranges import match_hatfield, pi_range, reconstruct_ranges, validate_ranges
from om_dataset.teacher import render


def node(line, id, parent, type="message", **extra):
    return line, {"id": id, "parentId": parent, "type": type, "message": {"role": "user", "content": "Remember this."}, **extra}


def test_pi_range_follows_branch_and_previous_coverage_not_append_adjacency():
    entries = [node(1, "a", None), node(2, "b", "a"), node(3, "late", "b"),
               node(4, "old", "late", "custom", customType="om.observations.recorded",
                    data={"coversUpToId": "b", "observations": [{}]}),
               node(5, "abandoned", "a"), node(6, "new", "old", "custom",
                    customType="om.observations.recorded", data={"coversUpToId": "late", "observations": [{}]})]
    result = pi_range(entries, 6)
    assert result["source_entry_ids"] == ["late"] and result["after_entry_id"] == "b"
    assert result["status"] == "range_inferred"
    entries.append(node(7, "late", "a"))
    assert pi_range(entries, 6)["status"] == "unresolved"


def test_legacy_endpoints_are_inclusive_but_producer_is_not_assumed():
    entries = [node(1, "a", None), node(2, "b", "a"), node(3, "old", "b", "custom",
               customType="om.observation", data={"coversFromId": "a", "coversUpToId": "b"})]
    result = pi_range(entries, 3)
    assert result["source_entry_ids"] == ["a", "b"]
    assert result["status"] == "range_only" and "legacy_producer_unproven" in result["issues"]


def test_hatfield_recovers_atomic_tool_group_and_requires_both_hashes():
    events = [{"run_id": "r", "seq": 1, "type": "llm_step_completed", "ts": "2026-10-13T12:00:00Z",
               "payload": {"assistant_message": {"content": "Checking.", "tool_calls": [{"id": "c", "name": "read", "arguments": {}}]}}},
              {"run_id": "r", "seq": 2, "type": "agent_command_applied", "ts": "2026-10-13T12:00:01Z", "payload": {"text": "While you read."}},
              {"run_id": "r", "seq": 3, "type": "tool_execution_end", "ts": "2026-10-13T12:00:02Z",
               "payload": {"tool_result": {"tool_call_id": "c", "result": {"tool_name": "read", "content": "x" * 14000}}}}]
    old_blocks = build_blocks(events, 3.25)
    coverage = {"source_start_seq": 1, "source_end_seq": 2, "part_count": 1, "part_index": 1,
                "source_digest": source_digest(old_blocks), "part_digest": part_digest(old_blocks)}
    match = match_hatfield(coverage, events, {r: build_blocks(events, r) for r in (2.35, 3.25)})
    assert match[1] == {"method": "global_anchors", "chars_per_token": 3.25}
    assert [b["seq"] for b in match[0]] == [2, 1]
    job = {"run_id": "r", "cutoff_seq": 3, "local_time": "2026-10-13 12:00", "sources": [
        {"seq": b["seq"], "role": "Assistant", "timestamp": "2026-10-13 12:00", "text": b["rendered_text"],
         "native_rendered": True, "source_refs": b["source_refs"]} for b in match[0]]}
    text, allowed = render(job)
    source_section = text.split("NEW CONVERSATION CHUNK:\n", 1)[1].rsplit("\n\nCurrent local time fallback:", 1)[0]
    assert source_section == "\n\n".join(b["rendered_text"] for b in match[0])
    assert allowed == {("r", 1), ("r", 2), ("r", 3)}
    coverage["part_digest"] = "0" * 64
    assert match_hatfield(coverage, events, {3.25: old_blocks}) is None


def test_frozen_range_export_accounts_for_candidates_and_detects_tampering(tmp_path):
    source = tmp_path / "live/session.jsonl"
    source.parent.mkdir()
    rows = [{"type": "session", "id": "s", "version": 3}, node(2, "a", None)[1],
            node(3, "o", "a", "custom", customType="om.observations.recorded", data={"coversUpToId": "a",
                 "observations": [{"id": "a" * 12, "content": "Remember this.", "timestamp": "2026-10-13 12:00",
                                   "relevance": "low", "tokenCount": 5, "sourceEntryIds": ["a"]}]})[1],
            node(4, "b", "o")[1],
            node(5, "bad", "b", "custom", customType="om.observations.recorded", data={"coversUpToId": "b",
                 "observations": [{"id": "b" * 12, "content": "Another fact.", "timestamp": "2026-10-13 12:00",
                                   "relevance": "low", "tokenCount": 5, "sourceEntryIds": ["a"]}]})[1]]
    source.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    collection, curated, output = (tmp_path / name for name in ("collection", "curated", "ranges"))
    collect([{"kind": "pi", "path": str(source)}], collection)
    curate_collection(collection, curated)
    report = reconstruct_ranges(curated, output)
    assert report["observer_candidates_accounted"] == 2
    assert report["observer_candidates_retained"] == report["observer_candidates_excluded"] == 1
    assert report["range_counts"] == {"pi:range_inferred": 1}
    excluded = json.loads((output / "exclusions.jsonl").read_text())
    assert excluded["issues"] == ["target_refs_outside_reconstructed_sources"]
    assert validate_ranges(output)["status"] == "pass"
    (output / "ranges.jsonl").write_text("{}\n")
    import pytest
    with pytest.raises(ValueError):
        validate_ranges(output)
