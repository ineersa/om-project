from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
from urllib.error import HTTPError

import pytest

from om_dataset import comparison as c, maintenance, teacher
from om_dataset.redaction import Redactor


@pytest.fixture
def setup(tmp_path):
    catalog = tmp_path / "catalog.json"
    catalog.write_text('{"values": []}')
    redactor = Redactor(catalog=catalog)
    policies = {task: {"role": task, "definition_id": "policy-" + task, "instructions": "Use the supplied evidence.",
                      "call_options": {"maxDurationSeconds": 300},
                      "tools": [{"name": {"observer": "record_observations", "reflector": "record_reflections", "dropper": "drop_observations"}[task],
                                 "description": "Record actions.", "parameters": {"type": "object"}}]}
                for task in ("observer", "reflector", "dropper")}
    rows = []
    for run in ("pi:session", "session", "session-2"):
        observation = dict(observation_id="canonical", timestamp="2026-01-01 00:00", relevance="high", content=f"Keep this for {run}.", source_refs=[])
        for task, policy in policies.items():
            job = dict(task=task, run_id=run, state_scope="scope", cutoff_seq=1, historical_range_id="range", generation_id="generation",
                       local_time="2026-01-01 00:00", observations=[observation], reflections=[], prior_observations=[], prior_reflections=[],
                       sources=[dict(seq=1, role="User", timestamp="2026-01-01 00:00", text="Keep this.")],
                       pool_partition={"start": 0, "end": 1, "total": 1}, max_drops=1, pool_tokens=40001, target_tokens=30000)
            text, _ = teacher.render(job) if task == "observer" else maintenance.render(job)
            rows.append(dict(task=task, job=job, job_id=run + task, policy_id=policy["definition_id"], status="completed",
                             config={"context_window": 131072, "request_options": {"max_tokens": 65536}},
                             messages=[{"role": "system", "content": policy["instructions"]}, {"role": "user", "content": text}],
                             tools=[teacher._tool(policy)], observations=[observation] if task == "observer" else [],
                             reflections=[dict(reflection_id="reflection", content="Keep this.", supporting_observation_ids=["canonical"])] if task == "reflector" else [],
                             drop_candidates=[{"observation_id": "canonical"}] if task == "dropper" else []))
        rows.append(dict(task="generation_commit", status="completed", job=job, generation_id="generation",
                         accepted_delta={"dropped_ids": ["canonical"], "reflection_ids": ["reflection"]}))
    return redactor, policies, rows


def test_exact_guard_selection_and_failed_source_exclusion(setup):
    redactor, policies, rows = setup
    cases = c.select_cases(rows, policies, redactor)
    assert len(cases) == 6
    assert all(c["source_kind"] == "hatfield" and not c["baseline"]["job"]["run_id"].startswith("pi:") for c in cases)
    assert len({(c["baseline"]["task"], c["captured"]["messages_sha256"]) for c in cases}) == 6
    baseline = cases[0]["baseline"]
    changed = deepcopy(baseline)
    changed["job"]["sources"][0]["text"] = "Different evidence."
    with pytest.raises(ValueError, match="Captured initial"):
        c.exact_input(changed, policies["observer"], redactor)
    changed = deepcopy(baseline)
    changed["tools"][0]["function"]["description"] = "Different schema contract."
    with pytest.raises(ValueError, match="Captured initial"):
        c.exact_input(changed, policies["observer"], redactor)
    failed = deepcopy(baseline)
    failed.update(status="failed", job_id="different-cache-id")
    with pytest.raises(ValueError, match="No eligible"):
        c.select_cases(rows + [failed], policies, redactor)


def test_native_loop_checkpoint_resume_and_completed_cache(tmp_path, setup):
    redactor, policies, rows = setup
    case = c.select_cases(rows, policies, redactor)[1]  # reflector exercises local/canonical support conversion
    calls = []
    def interrupted(body, run):
        calls.append(deepcopy(body))
        assert body["messages"][:2] == case["captured"]["messages"] and body["tools"] == case["captured"]["tools"]
        assert body["max_tokens"] == 65536 and "reasoning_effort" not in body and "thinking" not in body
        if len(calls) == 2:
            raise KeyboardInterrupt()
        return {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": None,
                "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "record_reflections", "arguments":
                json.dumps({"reflections": [{"content": "A durable fact.", "supporting_observation_ids": ["1"]}]})}}]}}]}
    root = tmp_path / "results"
    with pytest.raises(KeyboardInterrupt):
        c.run_cases([case], root, redactor, interrupted)
    assert list(root.rglob("checkpoint-*.json"))
    def finish(body, run):
        assert body["messages"][-1]["role"] == "tool"
        return {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "Done."}}]}
    result = c.run_cases([case], root, redactor, finish)[0]
    assert result["status"] == "completed" and result["reflections"][0]["supporting_observation_ids"] == ["canonical"]
    assert not list(root.rglob("checkpoint-*.json"))
    def forbidden(*args):
        raise AssertionError("Completed cache must not generate")
    assert c.run_cases([case], root, redactor, forbidden)[0] == result
    changed = deepcopy(case)
    changed["baseline"]["job"]["observations"][0]["content"] = "Changed"
    with pytest.raises(ValueError):
        c.run_cases([changed], root, redactor, forbidden)


def test_fixed_blinded_staging_and_failed_versus_noop(tmp_path, setup):
    redactor, policies, rows = setup
    cases = c.select_cases(rows, policies, redactor)
    results = [deepcopy(case["baseline"]) for case in cases]
    results[0].update(status="failed", observations=[])
    results[1].update(reflections=[])
    staging, private = tmp_path / "review-next/cases.jsonl", tmp_path / "private/identity-map.json"
    changed = deepcopy(results)
    changed[1]["job"]["active_reflection_ids"] = ["changed hidden prior state"]
    with pytest.raises(ValueError, match="prior state"):
        c.export_cases(cases, changed, redactor, staging, private)
    rejected = deepcopy(cases)
    rejected[0]["source_kind"] = "pi"
    with pytest.raises(ValueError, match="Hatfield histories only"):
        c.export_cases(rejected, results, redactor, staging, private)
    assert c.export_cases(cases, results, redactor, staging, private)["cases"] == 6
    pack = [json.loads(line) for line in staging.read_text().splitlines()]
    assert len({case["case_id"] for case in pack}) == 6
    for case in pack:
        assert case["schema_version"] == "om-review-case/1" and set(case["candidates"]) == {"A", "B"}
        assert case["evidence"] and case["prior_memory"] and case["redaction_id"] == redactor.fingerprint
        assert "local_qwen" not in json.dumps(case) and "captured_deepseek" not in json.dumps(case)
    assert any("NOT a valid no-op" in value for value in pack[0]["candidates"].values())
    assert any("Completed with no recorded actions" in value for value in pack[1]["candidates"].values())
    assert "this input partition only" in pack[2]["comparison_scope"]
    assert staging.stat().st_mode & 0o777 == private.stat().st_mode & 0o777 == 0o600
    assert staging.parent.stat().st_mode & 0o777 == private.parent.stat().st_mode & 0o777 == 0o700
    snapshot = staging.read_bytes()
    import hashlib
    assert json.loads(private.read_text())["cases_sha256"] == hashlib.sha256(snapshot).hexdigest()
    with pytest.raises(FileExistsError):
        c.export_cases(cases, results, redactor, staging, private)
    assert staging.read_bytes() == snapshot


def test_local_http_transport_props_and_safe_status_errors(monkeypatch, setup):
    redactor, _, _ = setup
    requests = []
    class Connection:
        status = 200
        def __init__(self, host, port, timeout):
            assert (host, port) == ("10.0.0.89", 8052)
        def request(self, method, endpoint, body, headers):
            requests.append((method, endpoint, body, headers))
        def getresponse(self):
            response = BytesIO(json.dumps({"model_path": c.MODEL_PATH}).encode())
            response.status = self.status
            return response
        def close(self):
            pass
    monkeypatch.setattr(c, "HTTPConnection", Connection)
    assert c.server_properties(redactor)["model_path"] == c.MODEL_PATH
    assert requests[-1][0:2] == ("GET", "/props") and requests[-1][3]["Authorization"] == "Bearer dummy"
    Connection.status = 503
    with pytest.raises(HTTPError) as error:
        c._request("POST", "/v1/chat/completions", {}, redactor)
    assert error.value.code == 503 and "model_path" not in str(error.value)
