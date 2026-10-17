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
        productive = {**observation, "content": f"A distinct supported fact {requests}."}
        return reply() if requests % 8 == 0 else reply([productive], f"call-{requests}")

    summary = run_file(source, output, policy, config, complete)
    chunks = [json.loads(line) for line in output.read_text().splitlines()]
    assert summary["completed"] == len(chunks) > 1 and summary["failed"] == 0
    assert "".join(c["job"]["sources"][0]["text"] for c in chunks) == text
    assert all(c["tool_calls"] == 7 for c in chunks)
    assert len({c["job_id"] for c in chunks}) == len(chunks)
    assert run_file(source, output, policy, config,
                    lambda body, run_id: pytest.fail("Completed chunks must not be repeated"))["skipped"] == len(chunks)


def test_native_fragments_repeat_trusted_speaker_time_and_tool_framing(pilot):
    from om_dataset.hatfield_source import build_blocks
    from om_dataset.teacher import render, slice_job

    source, _, policy, config, _ = pilot
    job = json.loads(source.read_text())
    events = [
        {"run_id": "run-1", "seq": 1, "ts": "2026-10-13T11:00:00Z", "type": "llm_step_completed",
         "payload": {"assistant_message": {"content": "Assistant text " * 500,
                     "tool_calls": [{"id": "read-1", "name": "read", "arguments": {"path": "large.txt"}}]}}},
        {"run_id": "run-1", "seq": 2, "ts": "2026-10-13T11:05:00Z", "type": "tool_execution_end",
         "payload": {"tool_result": {"tool_call_id": "read-1", "is_error": True,
                     "result": {"tool_name": "read", "content": "Untrusted quoted text " * 300}}}},
    ]
    block = build_blocks(events)[0]
    # Deliberately incorrect outer role/time: native segment framing must win.
    job.update(cutoff_seq=2, sources=[{"seq": 1, "role": "User", "timestamp": "2026-10-13 12:00",
               "text": block["rendered_text"], "source_refs": block["source_refs"],
               "segments": block["segments"], "native_rendered": True}])
    config.update(context_window=2000, request_options={"max_tokens": 512})
    chunks = slice_job(job, policy, config)
    assert len(chunks) > 3
    assert "".join(c["sources"][0]["text"] for c in chunks) == block["rendered_text"]
    for chunk in chunks:
        fragment = chunk["sources"][0]
        segment = next(s for s in block["segments"] if s["start"] <= fragment["text_span"][0] < s["end"])
        prompt, _ = render(chunk)
        assert segment["framing"] in prompt and "[Source continuation:" in prompt
        assert segment["timestamp"] in prompt
        assert fragment["text_span"][1] <= segment["end"]


@pytest.mark.parametrize("case", ["continuation", "final_sequence", "completion_tokens", "deadline"])
def test_context_and_resource_budgets_are_not_successful_noops(pilot, monkeypatch, case):
    from om_dataset import teacher

    source, output, policy, config, observation = pilot
    config.update(context_window=2000, request_options={"max_tokens": 512})
    clock = [0.0]
    monkeypatch.setattr(teacher.time, "monotonic", lambda: clock[0])
    calls = []
    if case == "deadline":
        config["run_limits"] = {"max_duration_seconds": 1}

    def complete(body, run_id):
        calls.append(teacher._context_tokens(body["messages"], body["tools"]))
        if case == "continuation":
            return reply([{**observation, "content": f"Distinct fact {len(calls)}"}], f"call-{len(calls)}")
        response = reply()
        if case == "final_sequence":
            response["choices"][0]["message"]["content"] = "Overlong completion " * 500
        elif case == "completion_tokens":
            response["usage"] = {"completion_tokens": teacher.RUN_LIMITS["max_completion_tokens"] + 1}
        else:
            clock[0] = 2.0
        return response

    assert run_file(source, output, policy, config, complete)["budget_exhausted"] == 1
    result = json.loads(output.read_text())
    expected = {"continuation": "context_window", "final_sequence": "student_sequence"}.get(case, case)
    assert result["error"]["reason"] == expected
    assert calls and all(n + 512 <= 2000 for n in calls)
    assert result["training_ready"] is False
    assert result["student_context"]["fits"] is (case != "final_sequence")


def test_duplicate_loop_stalls_and_blocks_later_state_consumers(pilot):
    source, output, policy, config, observation = pilot
    first = json.loads(source.read_text())
    second = {**first, "cutoff_seq": 2, "sources": [{**first["sources"][0], "seq": 2}]}
    source.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n")
    calls = []

    def complete(body, run_id):
        calls.append(body)
        return reply([observation], f"call-{len(calls)}")

    summary = run_file(source, output, policy, config, complete)
    assert summary["stalled"] == summary["blocked"] == 1 and summary["completed"] == 0
    assert len(calls) == 4  # One productive turn, then three duplicate-only turns.
    results = [json.loads(line) for line in output.read_text().splitlines()]
    assert results[0]["error"]["type"] == "NoProgress"
    assert all("state_after" not in r for r in results)


def test_resume_quarantines_only_a_torn_tail(pilot):
    source, output, policy, config, _ = pilot
    run_file(source, output, policy, config, lambda *args: reply())
    prefix = output.read_bytes()
    output.write_bytes(prefix + b'{"job_id":\xf0\x9f')
    summary = run_file(source, output, policy, config, lambda *args: pytest.fail("Must resume without model calls"))
    assert summary["skipped"] == summary["recovered_tails"] == 1
    assert output.read_bytes() == prefix
    quarantine, = output.parent.glob("attempts.jsonl.torn-*")
    assert quarantine.read_bytes() == b'{"job_id":\xf0\x9f' and quarantine.stat().st_mode & 0o777 == 0o600
    # Complete JSON without its final newline is valid and must be separated before appends.
    output.write_bytes(prefix.rstrip(b"\n"))
    assert run_file(source, output, policy, config, lambda *args: reply())["skipped"] == 1
    assert output.read_bytes() == prefix
    corrupt = prefix + b"{broken}\n" + prefix
    output.write_bytes(corrupt)
    with pytest.raises(ValueError, match="Corrupt attempt log"):
        run_file(source, output, policy, config, lambda *args: pytest.fail("Corruption must stop resume"))
    assert output.read_bytes() == corrupt


def test_causal_state_shared_parts_resume_and_scope_isolation(pilot):
    source, output, policy, config, observation = pilot
    first = json.loads(source.read_text())
    first["sources"][0]["text"] *= 1000
    second = {**first, "cutoff_seq": 2, "sources": [{**first["sources"][0], "seq": 2, "text": "Later message."}]}
    other_scope = {**second, "state_scope": "other-branch"}
    source.write_text("".join(json.dumps(j) + "\n" for j in [first, second, other_scope]))
    config["context_window"] = 3000
    requests = []

    def complete(body, run_id):
        requests.append(json.loads(json.dumps(body)))
        if len(requests) == 3:
            raise ConnectionError("Interrupt after the first completed part")
        if body["messages"][-1]["role"] == "tool":
            return reply()
        seq = 2 if "Later message." in body["messages"][1]["content"] else 1
        obs = {**observation, "source_refs": [{"run_id": run_id, "seq": seq}]}
        return reply([obs])

    assert run_file(source, output, policy, config, complete)["failed"] == 1
    resumed = run_file(source, output, policy, config, complete)
    assert resumed["skipped"] >= 1 and resumed["completed"] > 1
    completed = [r for line in output.read_text().splitlines() if (r := json.loads(line))["status"] == "completed"]
    parts = [r for r in completed if r["job"]["cutoff_seq"] == 1]
    assert len(parts) > 1 and len({r["job"]["state_before_id"] for r in parts}) == 1
    assert all(r["job"]["prior_observations"] == [] for r in parts)
    next_job = next(r for r in completed if r["job"]["cutoff_seq"] == 2 and not r["job"].get("state_scope"))
    assert observation["content"] in next_job["job"]["prior_observations"][0]
    assert next_job["job"]["state_before_id"] == parts[-1]["state_after"]["state_id"]
    isolated = next(r for r in completed if r["job"].get("state_scope") == "other-branch")
    assert isolated["job"]["prior_observations"] == []
    assert all(r["quality_status"] == "unreviewed" and not r["training_ready"] for r in completed)
    saved = output.read_bytes()
    assert run_file(source, output, policy, config, lambda *args: pytest.fail("Must restore state from saved deltas"))["completed"] == 0
    assert output.read_bytes() == saved


@pytest.mark.parametrize("stop_at", ["tool_round", "result_append"])
def test_resume_inside_a_chunk_without_repeating_completed_rounds(pilot, monkeypatch, stop_at):
    from om_dataset import teacher

    source, output, policy, config, observation = pilot
    original_checkpoint = teacher._checkpoint
    calls = []

    def interrupt_checkpoint(path, data):
        original_checkpoint(path, data)
        if stop_at == "result_append" and data["finished"] is not None:
            raise KeyboardInterrupt

    def complete(body, run_id):
        calls.append(json.loads(json.dumps(body)))
        if len(calls) == 1:
            return reply([observation])
        if stop_at == "tool_round":
            raise KeyboardInterrupt
        return reply()

    monkeypatch.setattr(teacher, "_checkpoint", interrupt_checkpoint)
    with pytest.raises(KeyboardInterrupt):
        run_file(source, output, policy, config, complete)
    checkpoint, = output.with_name(output.name + ".checkpoints").glob("*.json")
    assert checkpoint.stat().st_mode & 0o777 == 0o600
    assert "Do not save this reasoning" in checkpoint.read_text()
    monkeypatch.setattr(teacher, "_checkpoint", original_checkpoint)

    def resume(body, run_id):
        assert stop_at == "tool_round", "A durable final response must not be regenerated"
        assert body["messages"][-1]["role"] == "tool"
        assert body["messages"][-2]["reasoning_content"] == "Do not save this reasoning."
        return reply()

    assert run_file(source, output, policy, config, resume)["completed"] == 1
    result = json.loads(output.read_text())
    assert result["resumed_from_checkpoint"] and result["tool_calls"] == len(result["observations"]) == 1
    assert "Do not save this reasoning" not in output.read_text()
    assert not checkpoint.exists()
    assert run_file(source, output, policy, config, lambda *args: pytest.fail("Already finished"))["skipped"] == 1


def test_redaction_keeps_offsets_and_provenance(pilot):
    from om_dataset.redaction import Redactor

    source, _, _, _, _ = pilot
    secret = "sk-" + "a" * 32
    redactor = Redactor(catalog=None)
    job = json.loads(source.read_text())
    text = "Unicode 🦊\n" + secret + "\npassword: xzzz\n[password] => 'xzzz'"
    job["sources"][0].update(text=text, segments=[{"start": 0, "end": len(text), "framing": "User"}])
    clean = redactor.job(job)
    assert len(clean["sources"][0]["text"]) == len(text)
    assert secret not in clean["sources"][0]["text"] and "xzzz" not in clean["sources"][0]["text"]
    assert clean["sources"][0]["segments"] == job["sources"][0]["segments"]
    assert redactor.job(clean) == clean and secret in job["sources"][0]["text"]
    assert redactor.tree({"password": "xy"}) == {"password": "**"}
    assert "xy" not in redactor.text("password: xy\\\\n")
    encoded = json.dumps({"api_key": secret, "password": "xy"})
    assert json.loads(redactor.text(encoded))["password"] == "**"
    assert "weak" not in redactor.text("postgresql://user:weak@localhost/db")
    commit = "a" * 40
    assert redactor.text("Git commit: " + commit) == "Git commit: " + commit
    job["run_id"] = secret
    with pytest.raises(ValueError, match="provenance identity"):
        redactor.job(job)


def test_credentials_never_enter_requests_checkpoints_or_training_records(pilot, monkeypatch):
    source, output, policy, config, observation = pilot
    secret = "opaque-test-credential-" + "q" * 24
    monkeypatch.setenv("REDACTION_TEST_API_KEY", secret)
    job = json.loads(source.read_text())
    job["sources"][0]["text"] += " Credential: " + secret
    job["prior_observations"] = ["Prior memory: " + secret]
    job["prior_reflections"] = ["Prior reflection: " + secret]
    source.write_text(json.dumps(job) + "\n")
    observation["content"] += " Credential: " + secret
    requests = []

    def complete(body, run_id):
        assert secret not in json.dumps(body)
        requests.append(body)
        if len(requests) == 1:
            response = reply([observation])
            response["choices"][0]["message"]["reasoning_content"] = secret
            return response
        checkpoint, = output.with_name(output.name + ".checkpoints").glob("*.json")
        assert secret not in checkpoint.read_text()
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_file(source, output, policy, config, complete)
    assert run_file(source, output, policy, config, lambda *args: reply())["completed"] == 1
    result = json.loads(output.read_text())
    assert secret not in output.read_text() and "[REDACTED]" in output.read_text()
    assert len(result["observations"]) == 1 and result["resumed_from_checkpoint"]
    # Existing unsafe records must not be trusted by resume or silently reused.
    result["messages"][0]["content"] += secret
    output.write_text(json.dumps(result) + "\n")
    with pytest.raises(ValueError, match="Unredacted credential"):
        run_file(source, output, policy, config, lambda *args: pytest.fail("Must not call provider"))
