import json
import tempfile
import unittest
from pathlib import Path

from om_dataset.adapters.pi import parse_pi


def message(id, parent=None, text="private text"):
    return {"type": "message", "id": id, "parentId": parent,
            "message": {"role": "user", "content": [{"type": "text", "text": text}]}}


def observation(refs=None, **changes):
    record = {"id": "012345abcdef", "content": "A fact", "timestamp": "2026-10-13 12:00",
              "relevance": "high", "sourceEntryIds": refs or ["a"], "tokenCount": 3}
    record.update(changes)
    return record


def ledger(record=None, id="om", parent="a"):
    return {"type": "custom", "id": id, "parentId": parent, "customType": "om.observations.recorded",
            "data": {"coversUpToId": parent, "observations": [record or observation()]}}


class PiTests(unittest.TestCase):
    def parse(self, entries, header=True):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "session.jsonl"
            rows = [{"type": "session", "id": "s", "version": 3}] if header else []
            rows += entries
            raw = b"".join((json.dumps(r, ensure_ascii=False).encode() + b"\n") if not isinstance(r, bytes) else r for r in rows)
            path.write_bytes(raw)
            parsed = parse_pi(path)
            for source in parsed["sources"]:
                segment = raw[source["byte_offset"]:source["byte_offset"] + source["byte_length"]]
                self.assertEqual(json.loads(segment)["id"], source["native_id"])
            return parsed

    def test_current_normal_and_controls(self):
        result = self.parse([message("a", text="秘密"), ledger(),
                             {"type": "custom", "id": "control", "parentId": "om", "customType": "rewind-turn", "data": {}}])
        c = result["memory_candidates"][0]
        self.assertTrue(c["valid"])
        self.assertEqual(c["evidence_status"], "resolved")
        self.assertEqual(c["observation"]["source_refs"], [{"run_id": "pi:s", "seq": 2}])
        self.assertEqual(result["stats"]["default_branch_ids"], ["a", "om", "control"])
        self.assertEqual(len(result["sources"]), 3)
        self.assertNotIn("秘密", json.dumps(result["sources"], ensure_ascii=False))
        self.assertIsNone(result["sources"][-1]["content_sha256"])

    def test_no_om_unknown(self):
        result = self.parse([message("a")])
        self.assertEqual(result["memory_candidates"], [])
        self.assertEqual(result["stats"]["memory_outcome"], "unknown")

    def test_legacy_structured_and_prose(self):
        structured = {"type": "custom", "id": "old", "parentId": "a", "customType": "om.observation",
                      "data": {"coversFromId": "a", "coversUpToId": "a", "records": [observation(tokenCount=4)], "tokenCount": 4}}
        del structured["data"]["records"][0]["tokenCount"]
        prose = {"type": "custom", "id": "older", "parentId": "old", "customType": "om.observation",
                 "data": {"content": "2026-10-13 12:00 Some historical prose\nAnother line", "tokenCount": 8,
                          "coversFromId": "a", "coversUpToId": "a"}}
        result = self.parse([message("a"), structured, prose])
        first, second = result["memory_candidates"]
        self.assertEqual(first["adapter"], "pi-legacy")
        self.assertEqual(first["evidence_status"], "resolved")
        self.assertTrue(first["valid"])
        self.assertNotIn("tokenCount", first["native_record"])
        self.assertTrue(second["summary_only"])
        self.assertNotIn("observation", second)
        self.assertIsNone(second["native_memory_id"])
        self.assertTrue(second["valid"])

    def test_off_branch_future_missing_and_wrong_type(self):
        result = self.parse([message("a"), message("b", "a"), message("c", "a"),
                             ledger(observation(["b", "future", "absent", "control"]), parent="c"),
                             message("future", "om"),
                             {"type": "custom", "id": "control", "parentId": "future", "customType": "whatever"}])
        c = result["memory_candidates"][0]
        self.assertEqual(c["evidence_status"], "missing")
        self.assertEqual(c["native_source_entry_ids"], ["b", "future", "absent", "control"])
        codes = {i["code"] for i in c["evidence_issues"]}
        self.assertEqual(codes, {"off_branch_source_ref", "future_source_ref", "unresolved_source_ref"})
        self.assertEqual(c["observation"]["source_refs"], [])
        self.assertEqual(set(result["stats"]["leaf_ids"]), {"b", "control"})

    def test_malformed_byte_line(self):
        result = self.parse([message("a"), b'{bad\n', ledger()])
        self.assertEqual(result["sources"][-1]["line_no"], 4)
        self.assertIn("malformed_json", {i["code"] for i in result["issues"]})
        self.assertEqual(result["stats"]["physical_lines"], 4)

    def test_cycles_missing_and_forward_parent(self):
        result = self.parse([message("a", "b"), message("b", "a"), message("c", "absent"), ledger(parent="c")])
        self.assertFalse(result["stats"]["default_branch_valid"])
        self.assertEqual(result["stats"]["default_branch_ids"], [])
        codes = {i["code"] for i in result["issues"]}
        self.assertTrue({"parent_cycle", "missing_parent", "forward_parent"} <= codes)
        self.assertEqual(result["memory_candidates"][0]["evidence_status"], "unavailable")

    def test_duplicate_parent_and_source_ambiguity(self):
        result = self.parse([message("a"), message("a"), ledger()])
        self.assertIn("a", result["stats"]["ambiguous_native_ids"])
        self.assertEqual(result["memory_candidates"][0]["evidence_status"], "missing")
        self.assertFalse(result["stats"]["default_branch_valid"])

    def test_validation_preserves_invalid_original(self):
        record = observation(timestamp="2026-02-30 12:00", content="bad\nline", relevance=[], tokenCount=-1)
        result = self.parse([message("a"), ledger(record)])
        c = result["memory_candidates"][0]
        self.assertFalse(c["valid"])
        self.assertEqual(c["native_record"], record)
        self.assertTrue({"invalid_timestamp", "invalid_content", "invalid_relevance", "invalid_token_count"} <= set(c["validation_errors"]))

    def test_reflection_drop_and_snapshots(self):
        reflection = {"id": "abcdef012345", "content": "Durable fact", "supportingObservationIds": ["012345abcdef"], "tokenCount": 3}
        result = self.parse([message("a"), ledger(),
                             {"type": "custom", "id": "r", "parentId": "om", "customType": "om.reflections.recorded",
                              "data": {"coversUpToId": "om", "reflections": [reflection]}},
                             {"type": "custom", "id": "d", "parentId": "r", "customType": "om.observations.dropped",
                              "data": {"coversUpToId": "r", "observationIds": ["012345abcdef"]}},
                             {"type": "compaction", "id": "fold", "parentId": "d", "summary": "context",
                              "details": {"type": "om.folded", "version": 1, "fullFold": True,
                                          "observations": [observation()], "reflections": [reflection]}},
                             {"type": "compaction", "id": "oldfold", "parentId": "fold", "summary": "context",
                              "details": {"type": "observational-memory", "version": 2,
                                          "observations": [{"content": "prose", "tokenCount": 2}], "reflections": []}}])
        self.assertEqual([c["task"] for c in result["memory_candidates"]], ["observer", "reflector", "dropper", "snapshot", "snapshot"])
        self.assertTrue(all(c["valid"] for c in result["memory_candidates"]))
        self.assertEqual(result["memory_candidates"][1]["native_supporting_observation_ids"], ["012345abcdef"])
        self.assertEqual(result["memory_candidates"][2]["native_observation_ids"], ["012345abcdef"])

    def test_missing_header_does_not_invent_namespace(self):
        result = self.parse([message("a"), ledger()], header=False)
        self.assertIsNone(result["session_id"])
        self.assertIsNone(result["sources"][0]["source_ref"])
        self.assertEqual(result["memory_candidates"][0]["evidence_status"], "unavailable")

    def test_incomplete_legacy_record_is_summary_only(self):
        record = {"content": "Only prose remains"}
        result = self.parse([message("a"), {"type": "custom", "id": "old", "parentId": "a",
                            "customType": "om.observation", "data": {"coversFromId": "a", "coversUpToId": "a",
                            "records": [record], "tokenCount": 2}}])
        c = result["memory_candidates"][0]
        self.assertTrue(c["summary_only"])
        self.assertNotIn("observation", c)
        self.assertIsNone(c["native_memory_id"])
        self.assertFalse(c["valid"])
        self.assertEqual(c["native_record"], record)

    def test_legacy_version_four_snapshot_has_records_without_token_counts(self):
        record = observation()
        del record["tokenCount"]
        result = self.parse([message("a"), {"type": "compaction", "id": "f", "parentId": "a",
                            "details": {"type": "observational-memory", "version": 4,
                                        "observations": [record], "reflections": [
                                            {"id": "fedcba654321", "content": "A durable conclusion",
                                             "supportingObservationIds": [record["id"]]}]}}])
        candidate = result["memory_candidates"][0]
        self.assertEqual(candidate["task"], "snapshot")
        self.assertTrue(candidate["valid"])
        self.assertEqual(candidate["evidence_status"], "unavailable")
        self.assertNotIn("tokenCount", candidate["native_record"]["observations"][0])

    def test_non_string_discriminators_do_not_crash(self):
        result = self.parse([message("a"), {"type": "custom", "id": "c", "parentId": "a", "customType": []},
                             {"type": "compaction", "id": "f", "parentId": "c", "details": {"type": []}}])
        self.assertEqual(len(result["sources"]), 3)
        self.assertEqual(result["memory_candidates"], [])


if __name__ == "__main__":
    unittest.main()
