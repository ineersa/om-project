import json

import pytest

from om_dataset.teacher import run_file


def _policy(task, name):
    return {"role": task, "definition_id": task, "instructions": "Use the supplied tool.",
            "call_options": {"maxDurationSeconds": 300},
            "tools": [{"name": name, "description": name,
                       "parameters": {"properties": {"observations": {"items": {"type": "object"}}}}}]}


def _reply(name=None, arguments=None):
    message = {"role": "assistant", "content": "Done."}
    if name:
        message.update(content=None, tool_calls=[{"id": "call-1", "type": "function", "function": {
            "name": name, "arguments": json.dumps(arguments)}}])
    return {"choices": [{"finish_reason": "tool_calls" if name else "stop", "message": message}]}


def test_generation_resume_keeps_reflection_and_drop_atomic_and_causal(tmp_path, monkeypatch):
    source, output = tmp_path / "jobs.jsonl", tmp_path / "attempts.jsonl"
    jobs = [{"run_id": "run", "cutoff_seq": i, "local_time": "2026-10-15 12:00",
             "sources": [{"seq": i, "role": "User", "timestamp": "2026-10-15 12:00", "text": f"Source {i}."}]}
            for i in (1, 2)]
    source.write_text("".join(json.dumps(job) + "\n" for job in jobs))
    observer = _policy("observer", "record_observations")
    policies = {r: _policy(r, n) for r, n in (("reflector", "record_reflections"), ("dropper", "drop_observations"))}
    config = {"model": "test", "request_options": {"max_tokens": 65536}, "context_window": 131072}
    interrupted = False

    def complete(body, run):
        nonlocal interrupted
        name = body["tools"][0]["function"]["name"]
        if body["messages"][-1]["role"] == "tool":
            if name == "drop_observations" and not interrupted:
                interrupted = True
                raise KeyboardInterrupt()
            return _reply()
        if name == "record_reflections":
            assert not interrupted, "Completed reflection must be reused"
            return _reply(name, {"reflections": [{"content": "Durable consolidated fact.", "supporting_observation_ids": ["1"]}]})
        if name == "drop_observations":
            assert not interrupted, "Completed drop tool round must be reused"
            return _reply(name, {"ids": ["1"]})
        second = "Source 2." in body["messages"][1]["content"]
        if second:
            assert "Durable consolidated fact." in body["messages"][1]["content"]
            assert "A" * 100 not in body["messages"][1]["content"], "Dropped observation leaked into next input"
        else:
            assert not interrupted, "Completed Observer must be reused"
        return _reply(name, {"observations": [{"timestamp": "2026-10-15 12:00", "content": "Next fact." if second else "A" * 95000,
                                              "relevance": "high", "source_refs": [{"run_id": run, "seq": 2 if second else 1}]}]})

    with pytest.raises(KeyboardInterrupt):
        run_file(source, output, observer, config, complete, policies)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [r["task"] for r in rows] == ["observer", "reflector"]
    assert rows[-1]["generation_status"] == "pending_commit"
    assert list(output.with_name(output.name + ".checkpoints").glob("*.json"))
    summary = run_file(source, output, observer, config, complete, policies)
    assert summary["completed"] == 2 and summary["skipped"] == 2
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [r["task"] for r in rows] == ["observer", "reflector", "dropper", "generation_commit", "observer"]
    commit = rows[3]
    assert len(commit["accepted_delta"]["dropped_ids"]) == 1
    assert commit["state_after"]["observation_ids"] == []
    assert rows[4]["job"]["state_before_id"] == commit["state_after"]["state_id"]
    assert rows[2]["resumed_from_checkpoint"]
    saved = output.read_bytes()
    from om_dataset.redaction import Redactor
    fingerprint = Redactor().fingerprint
    secret = "rotated-provider-value-123456789"
    monkeypatch.setenv("OPENCODE_API_KEY", secret)
    monkeypatch.setenv("XDG_ACTIVATION_TOKEN", "new-desktop-login-token-12345")
    assert Redactor().fingerprint == fingerprint
    assert secret not in Redactor().text(secret)
    run_file(source, output, observer, config, lambda *_: pytest.fail("Completed replay contacted provider"), policies)
    assert output.read_bytes() == saved


def test_failed_maintenance_blocks_later_observer_without_committing(tmp_path, monkeypatch):
    from om_dataset import maintenance
    monkeypatch.setattr(maintenance, "REFLECT_AFTER_TOKENS", 1)
    source, output = tmp_path / "jobs.jsonl", tmp_path / "attempts.jsonl"
    jobs = [{"run_id": "run", "cutoff_seq": i, "local_time": "2026-10-15 12:00",
             "sources": [{"seq": i, "role": "User", "timestamp": "2026-10-15 12:00", "text": "A fact."}]} for i in (1, 2)]
    source.write_text("".join(json.dumps(j) + "\n" for j in jobs))
    observer = _policy("observer", "record_observations")
    reflector = _policy("reflector", "record_reflections")
    def complete(body, run):
        if body["tools"][0]["function"]["name"] != "record_observations":
            raise ConnectionError("Do not persist provider text")
        if body["messages"][-1]["role"] == "tool":
            return _reply()
        return _reply("record_observations", {"observations": [{"timestamp": "2026-10-15 12:00", "content": "A durable fact.",
                        "relevance": "low", "source_refs": [{"run_id": run, "seq": 1}]}]})
    summary = run_file(source, output, observer, {"model": "test", "request_options": {}, "context_window": 131072},
                       complete, {"reflector": reflector})
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert summary["failed"] == 1 and summary["blocked"] == 1
    assert not any(r.get("task") == "generation_commit" for r in rows)
    assert "Do not persist provider text" not in output.read_text()


def test_partitioned_pools_keep_all_observations_and_apply_one_global_drop_cap(tmp_path):
    source, output = tmp_path / "jobs.jsonl", tmp_path / "attempts.jsonl"
    source.write_text(json.dumps({"run_id": "run", "cutoff_seq": 1, "local_time": "2026-10-15 12:00",
                                 "sources": [{"seq": 1, "role": "User", "timestamp": "2026-10-15 12:00", "text": "Facts."}]}) + "\n")
    observer = _policy("observer", "record_observations")
    policies = {r: _policy(r, n) for r, n in (("reflector", "record_reflections"), ("dropper", "drop_observations"))}
    def complete(body, run):
        if body["messages"][-1]["role"] == "tool":
            return _reply()
        name = body["tools"][0]["function"]["name"]
        if name == "record_observations":
            return _reply(name, {"observations": [{"timestamp": "2026-10-15 12:00", "content": letter * 40000,
                 "relevance": "medium", "source_refs": [{"run_id": run, "seq": 1}]} for letter in "ABC"]})
        if name == "record_reflections":
            return _reply(name, {"reflections": [{"content": "Durable fact.", "supporting_observation_ids": ["1"]}]})
        import re
        ids = re.findall(r"^\[(\d+)\]", body["messages"][1]["content"], re.M)
        return _reply(name, {"ids": ids})
    summary = run_file(source, output, observer,
                       {"model": "test", "request_options": {"max_tokens": 65536}, "context_window": 131072}, complete, policies)
    assert summary["failed"] == summary["budget_exhausted"] == 0
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    original = {o["observation_id"] for o in rows[0]["observations"]}
    for task in ("reflector", "dropper"):
        parts = [r for r in rows if r.get("task") == task]
        assert len(parts) == 2
        assert {o["observation_id"] for r in parts for o in r["job"]["observations"]} == original
        assert all(r["student_context"]["fits"] for r in parts)
    commit = rows[-1]
    assert len(commit["accepted_delta"]["dropped_ids"]) == 2
    assert len(commit["state_after"]["observation_ids"]) == 1


@pytest.mark.parametrize("task,name,arguments", [
    ("reflector", "record_reflections", {"reflections": []}),
    ("dropper", "drop_observations", {"ids": [], "reason": "Nothing safe to drop."}),
    ("reflector", "record_reflections", {"reflections": [{"content": "Unsupported", "supporting_observation_ids": ["99"]}]}),
    ("dropper", "drop_observations", {"ids": ["99"]}),
])
def test_only_empty_maintenance_batches_can_recover_as_flagged_no_action(task, name, arguments):
    from om_dataset.teacher import run_job
    job = {"task": task, "run_id": "run", "cutoff_seq": 1,
           "observations": [{"observation_id": "obs", "content": "An active constraint.",
                             "timestamp": "2026-10-15 12:00", "relevance": "high"}],
           "reflections": [], "pool_tokens": 40001, "target_tokens": 30000, "max_drops": 1}
    responses = iter([_reply(name, arguments), _reply()])
    result = run_job(job, _policy(task, name),
                     {"model": "test", "request_options": {"max_tokens": 65536}, "context_window": 131072},
                     lambda body, run: next(responses))
    empty = arguments.get("reflections", arguments.get("ids")) == []
    assert result["status"] == ("completed" if empty else "failed")
    assert bool(result.get("protocol_recovery")) == empty
    assert result["rejected_items"] and result["requires_recovery_mask"]
    assert result["training_ready"] is False
    assert not result.get("reflections", result.get("drop_candidates"))
