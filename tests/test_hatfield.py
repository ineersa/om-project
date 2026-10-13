import hashlib
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from om_dataset.adapters.hatfield import parse_hatfield, parse_hatfield_db


class AdapterTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def event(self, seq=1, **changes):
        event = {"schema_version": "1.0", "run_id": "run-a", "seq": seq,
                 "turn_no": 1, "type": "llm_step_completed", "ts": "2026-10-13T11:00:00Z",
                 "payload": {"assistant_message": {"role": "assistant", "content": "Synthetic answer"}}}
        event.update(changes)
        return event

    def journal(self, events):
        path = self.root / "events.jsonl"
        path.write_bytes(b"".join(json.dumps(e, ensure_ascii=False).encode() + b"\n" for e in events))
        return path

    def db(self, observations=None, reflections=None, version="20260726_003_active_generation_and_relevance_text"):
        path = self.root / "om.sqlite"
        with closing(sqlite3.connect(path)) as db, db:
            db.execute("CREATE TABLE om_schema_version(version TEXT, description TEXT, checksum TEXT, applied_at TEXT)")
            db.execute("INSERT INTO om_schema_version VALUES(?,?,?,?)", (version, "Synthetic schema", "hash", "2026-10-13"))
            db.execute("""CREATE TABLE om_observation(
                observation_id TEXT, run_id TEXT, boundary_key TEXT,
                source_start_seq INTEGER, source_end_seq INTEGER, source_refs_json TEXT,
                content TEXT, content_hash TEXT, relevance, timestamp TEXT, token_count INTEGER,
                observer_model TEXT, observer_schema_version TEXT, created_at TEXT)""")
            db.execute("""CREATE TABLE om_reflection(
                reflection_id TEXT, run_id TEXT, compaction_request_id TEXT, content TEXT,
                supporting_observation_ids_json TEXT, token_count INTEGER,
                reflector_model TEXT, reflector_schema_version TEXT, created_at TEXT)""")
            db.execute("CREATE TABLE om_coverage(run_id TEXT, source_start_seq INTEGER)")
            db.execute("INSERT INTO om_coverage VALUES('coverage-only-run', 2)")
            db.execute("CREATE TABLE om_generation(run_id TEXT, generation_id TEXT)")
            db.execute("INSERT INTO om_generation VALUES('generation-only-run', 'g1')")
            db.execute("CREATE TABLE secret_table(secret TEXT)")
            db.execute("INSERT INTO secret_table VALUES('not for export')")
            for row in observations or [self.observation()]:
                keys = list(row)
                db.execute('INSERT INTO om_observation (' + ','.join(keys) + ') VALUES (' + ','.join('?' for _ in keys) + ')', list(row.values()))
            for row in reflections or [self.reflection()]:
                keys = list(row)
                db.execute('INSERT INTO om_reflection (' + ','.join(keys) + ') VALUES (' + ','.join('?' for _ in keys) + ')', list(row.values()))
        return path

    def observation(self, **changes):
        content = "The synthetic workflow requires a local archive."
        row = {"observation_id": "o1", "run_id": "run-a", "boundary_key": "b1",
               "source_start_seq": 1, "source_end_seq": 3,
               "source_refs_json": '[{"run_id":"run-a","seq":1}]',
               "content": content, "content_hash": hashlib.sha256(content.encode()).hexdigest(),
               "relevance": "high", "timestamp": "2026-10-13 11:00", "token_count": 10,
               "observer_model": "fixture-model", "observer_schema_version": "fixture-v1",
               "created_at": "2026-10-13T11:01:00Z"}
        row.update(changes)
        return row

    def reflection(self, **changes):
        row = {"reflection_id": "r1", "run_id": "run-a", "compaction_request_id": "c1",
               "content": "Archive locally before labeling.", "supporting_observation_ids_json": '["o1"]',
               "token_count": 6, "reflector_model": "fixture-model", "reflector_schema_version": "fixture-v1",
               "created_at": "2026-10-13T11:02:00Z"}
        row.update(changes)
        return row

    def test_threshold_reflection_without_compaction_request(self):
        path = self.db(reflections=[self.reflection(compaction_request_id="")])
        result = parse_hatfield_db(path, {"run-a": {1, 2, 3}})
        reflection = next(c for c in result["memory_candidates"] if c["task"] == "reflector")
        self.assertTrue(reflection["valid"])
        self.assertEqual(reflection["native_record"]["compaction_request_id"], "")

    def test_stream_byte_pointers_and_semantic_hash(self):
        first = self.event(1)
        first["payload"]["assistant_message"]["content"] = "Unicode ✓"
        second = self.event(4)
        second["payload"]["assistant_message"]["content"] = "Unicode ✓"
        second["payload"]["assistant_message"]["timestamp"] = 45
        result = parse_hatfield(self.journal([first, second]))
        self.assertEqual(result["issues"], [])
        self.assertEqual(result["stats"]["last_complete_seq"], 4)
        self.assertEqual(result["stats"]["last_valid_seq"], 4)
        self.assertEqual(result["sources"][0]["content_sha256"], result["sources"][1]["content_sha256"])
        raw = (self.root / "events.jsonl").read_bytes()
        for source in result["sources"]:
            data = raw[source["byte_offset"]:source["byte_offset"] + source["byte_length"]]
            self.assertEqual(json.loads(data)["seq"], source["native_id"])
        self.assertNotIn("Unicode", json.dumps(result))

    def test_event_errors_and_complete_boundary(self):
        path = self.journal([self.event(), self.event(1), self.event(3, run_id="other"),
                             self.event(4, schema_version="future"), self.event(None)])
        with path.open("ab") as stream:
            stream.write(b"{not-json}\n")
            stream.write(json.dumps(self.event(5)).encode())
        result = parse_hatfield(path)
        codes = {i["code"] for i in result["issues"]}
        self.assertTrue({"duplicate_source_ref", "non_monotonic_seq", "mixed_run_id",
                         "unknown_event_schema", "invalid_seq", "malformed_json", "incomplete_line"} <= codes)
        self.assertEqual(result["stats"]["last_complete_seq"], 4)
        self.assertEqual(result["stats"]["last_valid_seq"], 1)
        self.assertEqual(len(result["sources"]), 7)

    def test_roles_do_not_project_queued_command(self):
        message = {"role": "user", "content": "Synthetic task"}
        events = [self.event(1, type="agent_command_queued", payload={"command": {"message": message}}),
                  self.event(2, type="agent_command_applied", payload={"command": {"message": message}}),
                  self.event(3, type="agent_command_applied", payload={"kind": "steer", "message": message}),
                  self.event(4, type="tool_execution_end", payload={"tool_result": {"content": "Synthetic result"}}),
                  self.event(5, type="run_started", payload={"input": "Synthetic input"})]
        result = parse_hatfield(self.journal(events))
        self.assertEqual([s["role"] for s in result["sources"]], [None, "user", "user", "tool", None])
        self.assertEqual(result["sources"][1]["content_sha256"], result["sources"][2]["content_sha256"])
        self.assertEqual(result["memory_candidates"], [])

    def test_no_invented_run_id(self):
        path = self.root / "imaginary-run"
        path.mkdir()
        raw = path / "events.jsonl"
        raw.write_bytes(b"broken\n")
        self.assertIsNone(parse_hatfield(raw)["session_id"])

    def test_normal_database_and_no_writes(self):
        path = self.db()
        before = path.read_bytes()
        result = parse_hatfield_db(path, {"run-a": {1, 3}})
        self.assertEqual({i["code"] for i in result["issues"]}, {"missing_source_archive"})
        observer, reflector = result["memory_candidates"]
        self.assertTrue(observer["valid"])
        self.assertTrue(reflector["valid"])
        self.assertEqual(observer["evidence_status"], "resolved")
        self.assertEqual(reflector["native_supporting_observation_ids"], ["o1"])
        self.assertEqual(result["stats"]["db_only_run_ids"], ["coverage-only-run", "generation-only-run"])
        self.assertEqual(result["run_ids"], ["coverage-only-run", "generation-only-run", "run-a"])
        self.assertNotIn("secret_table", result["stats"]["tables"])
        self.assertEqual(path.read_bytes(), before)
        self.assertFalse(path.with_name(path.name + "-wal").exists())
        self.assertEqual(observer["native_record"]["content"], self.observation()["content"])

    def test_missing_journal_is_explicit(self):
        result = parse_hatfield_db(self.db(), {})
        candidate = result["memory_candidates"][0]
        self.assertFalse(candidate["valid"])
        self.assertEqual(candidate["evidence_status"], "unavailable")
        self.assertIn("unresolved_source_refs", candidate["validation_errors"])
        self.assertIn("run-a", result["stats"]["db_only_run_ids"])

    def test_malformed_json_reference(self):
        result = parse_hatfield_db(self.db([self.observation(source_refs_json="broken")]), {"run-a": {1}})
        candidate = result["memory_candidates"][0]
        self.assertFalse(candidate["valid"])
        self.assertIn("invalid_source_refs_json", candidate["validation_errors"])
        self.assertEqual(candidate["native_record"]["source_refs_json"], "broken")

    def test_integer_relevance_published_bins(self):
        values = [0, 24, 25, 49, 50, 74, 75, 1000]
        expected = ["low", "low", "medium", "medium", "high", "high", "critical", "critical"]
        rows = [self.observation(observation_id="o" + str(n), relevance=n) for n in values]
        result = parse_hatfield_db(self.db(rows), {"run-a": {1}})
        candidates = [c for c in result["memory_candidates"] if c["task"] == "observer"]
        by_native = {c["native_record"]["relevance"]: c for c in candidates}
        self.assertEqual([by_native[n]["observation"]["relevance"] for n in values], expected)
        self.assertTrue(all(by_native[n]["valid"] for n in values))
        self.assertTrue(all(by_native[n]["transformations"] for n in values))

    def test_bad_hash_and_bad_calendar(self):
        result = parse_hatfield_db(self.db([self.observation(content_hash="bad", timestamp="2026-02-30 11:00")]), {"run-a": {1}})
        errors = result["memory_candidates"][0]["validation_errors"]
        self.assertIn("content_hash_mismatch", errors)
        self.assertIn("invalid_timestamp", errors)

    def test_unknown_schema_retained(self):
        result = parse_hatfield_db(self.db(version="future-schema"), {"run-a": {1}})
        self.assertIn("unknown_db_schema_version", {i["code"] for i in result["issues"]})
        self.assertEqual(len(result["memory_candidates"]), 2)
        self.assertEqual(result["stats"]["schema_versions"][0]["version"], "future-schema")

    def test_missing_columns_and_missing_timestamp_not_backfilled(self):
        path = self.root / "old.sqlite"
        with closing(sqlite3.connect(path)) as db, db:
            db.execute("CREATE TABLE om_observation(observation_id TEXT, content TEXT, created_at TEXT)")
            db.execute("INSERT INTO om_observation VALUES('o1', 'Synthetic old record', '2026-10-13T11:01:00Z')")
        result = parse_hatfield_db(path, {})
        self.assertIn("missing_required_columns", {i["code"] for i in result["issues"]})
        candidate = result["memory_candidates"][0]
        self.assertIsNone(candidate["observation"]["timestamp"])
        self.assertFalse(candidate["valid"])
        self.assertEqual(candidate["native_record"]["created_at"], "2026-10-13T11:01:00Z")

    def test_cross_run_reflection_support_rejected(self):
        result = parse_hatfield_db(self.db(reflections=[self.reflection(run_id="other")]), {"run-a": {1}})
        reflector = result["memory_candidates"][1]
        self.assertFalse(reflector["valid"])
        self.assertEqual(reflector["evidence_status"], "missing")
        self.assertIn("unresolved_support_ids", reflector["validation_errors"])

    def test_partial_missing_and_multiline(self):
        row = self.observation(source_refs_json='[{"run_id":"run-a","seq":1},{"run_id":"run-a","seq":3}]',
                               content="First\nsecond", relevance="unknown")
        result = parse_hatfield_db(self.db([row]), {"run-a": {1}})
        candidate = result["memory_candidates"][0]
        self.assertEqual(candidate["evidence_status"], "missing")
        self.assertTrue({"multiline_content", "invalid_relevance", "unresolved_source_refs"} <= set(candidate["validation_errors"]))

    def test_sequence_retreat_does_not_restore_validity_below_high_water(self):
        result = parse_hatfield(self.journal([self.event(10), self.event(5), self.event(8)]))
        self.assertEqual(result["stats"]["last_valid_seq"], 10)
        self.assertEqual(result["stats"]["last_complete_seq"], 8)
        self.assertEqual([s["valid"] for s in result["sources"]], [True, False, False])


if __name__ == "__main__":
    unittest.main()
