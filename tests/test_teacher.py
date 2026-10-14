import json

import pytest

from om_dataset.teacher import run_file


@pytest.fixture
def pilot(tmp_path):
    job = {"run_id": "run-1", "cutoff_seq": 1, "local_time": "2026-10-13 12:00",
           "sources": [{"seq": 1, "role": "User", "timestamp": "2026-10-13 12:00",
                        "text": "Keep all relevance levels."}]}
    policy = {"definition_id": "test-policy", "instructions": "Record observations.",
              "call_options": {"maxToolCalls": 6},
              "tools": [{"name": "record_observations", "description": "Record observations.",
                         "parameters": {"properties": {"observations": {"items": {"type": "object"}}}}}]}
    observation = {"timestamp": "2026-10-13 12:00", "content": "User wants all relevance levels retained.",
                   "relevance": "high", "source_refs": [{"run_id": "run-1", "seq": 1}]}
    config = {"model": "test-model", "base_url": "https://example.invalid/v1",
              "request_options": {}, "context_window": 131_072}
    source, output = tmp_path / "jobs.jsonl", tmp_path / "attempts.jsonl"
    source.write_text(json.dumps(job) + "\n")
    return source, output, policy, config, observation


def reply(observations=None, cid="call-1"):
    message = {"role": "assistant", "content": "Done."}
    if observations is not None:
        message.update(content=None, reasoning_content="Do not save this reasoning.",
                       tool_calls=[{"id": cid, "type": "function", "function": {
                           "name": "record_observations", "arguments": json.dumps({"observations": observations})}}])
    return {"choices": [{"finish_reason": "stop" if observations is None else "tool_calls", "message": message}]}


def test_saves_real_receipts_and_resumes_without_duplicate_output(pilot):
    source, output, policy, config, observation = pilot
    responses = iter([reply([observation]), reply([observation], "call-2"), reply()])
    requests = []

    def complete(body, run_id):
        requests.append((json.loads(json.dumps(body)), run_id))
        return next(responses)

    assert run_file(source, output, policy, config, complete)["completed"] == 1
    assert [run_id for _, run_id in requests] == ["run-1"] * 3
    assert requests[1][0]["messages"][2]["reasoning_content"] == "Do not save this reasoning."
    saved = output.read_text()
    result = json.loads(saved)
    assert result["status"] == "completed" and result["quality_status"] == "unreviewed"
    assert len(result["observations"]) == 1
    assert len(result["observations"][0]["observation_id"]) == 64
    receipts = [json.loads(m["content"]) for m in result["messages"] if m["role"] == "tool"]
    assert receipts[0]["added"] == 1 and receipts[0]["total"] == 1
    assert receipts[1]["duplicates"] == 1 and receipts[1]["total"] == 1
    assert result["messages"][-1] == {"role": "assistant", "content": "Done."}
    assert "Do not save this reasoning" not in saved
    assert run_file(source, output, policy, config,
                    lambda body, run_id: pytest.fail("Completed work must not be rerun"))["skipped"] == 1
    assert output.read_text() == saved


@pytest.mark.parametrize("failure", ["invalid_reference", "provider_error"])
def test_failures_are_saved_but_never_become_successful_noops(pilot, failure):
    source, output, policy, config, observation = pilot
    observation["source_refs"][0]["seq"] = 99
    responses = iter([reply([observation]), reply()])

    def complete(body, run_id):
        if failure == "provider_error":
            raise ConnectionError("A provider exception could contain a secret")
        return next(responses)

    assert run_file(source, output, policy, config, complete)["failed"] == 1
    result = json.loads(output.read_text())
    assert result["status"] == "failed" and result["observations"] == []
    assert result["training_ready"] is False
    assert "could contain a secret" not in output.read_text()
    if failure == "invalid_reference":
        assert json.loads(result["messages"][3]["content"])["error"] == "all_rejected"


def test_future_source_is_rejected_before_generation(pilot):
    source, output, policy, config, _ = pilot
    job = json.loads(source.read_text())
    job["sources"][0]["seq"] = 2
    source.write_text(json.dumps(job) + "\n")
    assert run_file(source, output, policy, config,
                    lambda body, run_id: pytest.fail("Future input must not reach the provider"))["failed"] == 1
    result = json.loads(output.read_text())
    assert result["error"]["type"] == "ValueError" and result["messages"] == []


@pytest.mark.parametrize("provider", ["opencode-go", "deepseek"])
def test_provider_request_headers_and_defaults(pilot, monkeypatch, provider):
    from io import BytesIO
    from types import SimpleNamespace

    from om_dataset import teacher

    source, output, policy, config, observation = pilot
    assert teacher.STUDENT_CONTEXT_WINDOW == 131_072
    config.update(teacher.PROVIDERS[provider], provider=provider, request_options=teacher.REQUEST_OPTIONS)
    monkeypatch.setenv(config["api_key_env"], "test-key")
    responses = iter([reply([observation]), reply()])
    requests = []

    def connect(host, timeout):
        response = BytesIO(json.dumps(next(responses)).encode())
        response.status = 200
        return SimpleNamespace(request=lambda method, path, body, headers: requests.append(
            (host, path, json.loads(body), headers)), getresponse=lambda: response, close=lambda: None)

    monkeypatch.setattr(teacher, "HTTPSConnection", connect)
    complete = lambda body, run_id: teacher.complete_request(body, run_id, config, 60)
    assert run_file(source, output, policy, config, complete)["completed"] == 1
    for host, path, body, headers in requests:
        assert host == ("opencode.ai" if provider == "opencode-go" else "api.deepseek.com")
        assert path == ("/zen/go/v1/chat/completions" if provider == "opencode-go" else "/chat/completions")
        assert body["model"] == ("deepseek-v4.1-flash" if provider == "opencode-go" else "deepseek-flash")
        assert body["reasoning_effort"] == "low" and body["thinking"] == {"type": "enabled"}
        assert body["max_tokens"] == 65536
        assert headers["Authorization"] == "Bearer test-key"
        if provider == "opencode-go":
            assert headers["x-opencode-session"] == "run-1"
            assert headers["x-opencode-client"] == headers["User-Agent"] == "om-project"
        else:
            assert "x-opencode-session" not in headers
    assert "test-key" not in output.read_text()


def test_large_unicode_source_is_sliced_without_loss_and_more_than_six_calls_work(pilot):
    source, output, policy, config, observation = pilot
    job = json.loads(source.read_text())
    text = "Keep all relevance levels. 😀\n" * 2500
    job["sources"][0]["text"] = text
    source.write_text(json.dumps(job) + "\n")
    config["context_window"] = 20000  # Exercise multiple chunks with a small synthetic context.
    requests = 0

    def complete(body, run_id):
        nonlocal requests
        requests += 1
        return reply() if requests % 8 == 0 else reply([observation], f"call-{requests}")

    summary = run_file(source, output, policy, config, complete)
    chunks = [json.loads(line) for line in output.read_text().splitlines()]
    assert summary["completed"] == len(chunks) > 1 and summary["failed"] == 0
    assert "".join(c["job"]["sources"][0]["text"] for c in chunks) == text
    assert all(c["tool_calls"] == 7 for c in chunks)
    assert len({c["job_id"] for c in chunks}) == len(chunks)
    assert run_file(source, output, policy, config,
                    lambda body, run_id: pytest.fail("Completed chunks must not be repeated"))["skipped"] == len(chunks)
