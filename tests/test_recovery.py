import json

from om_dataset.recover import reset_empty_reflections


def test_empty_reflection_retry_preserves_log_and_archives_only_failed_checkpoint(tmp_path):
    path = tmp_path / "attempts.jsonl"
    failed = {"job_id": "failed", "task": "reflector", "status": "failed", "reflections": [],
              "messages": [{"role": "assistant", "tool_calls": [{"function": {
                  "name": "record_reflections", "arguments": '{"reflections": []}'}}]}]}
    completed = {"job_id": "completed", "task": "reflector", "status": "completed"}
    path.write_text(json.dumps(failed) + "\n" + json.dumps(completed) + "\n")
    before = path.read_bytes()
    checkpoints = path.with_name(path.name + ".checkpoints")
    checkpoints.mkdir()
    saved = {"accepted": {}, "finished": None, "messages": []}
    (checkpoints / "failed.json").write_text(json.dumps(saved))
    (checkpoints / "completed.json").write_text("keep")
    assert reset_empty_reflections(path)["reset_failed_invocations"] == 1
    assert path.read_bytes() == before
    assert not (checkpoints / "failed.json").exists()
    assert (checkpoints / "completed.json").read_text() == "keep"
    archives = [p for p in (tmp_path / "recovery").glob("*.json") if not p.name.endswith(".event.json")]
    assert len(archives) == 1 and json.loads(archives[0].read_text()) == saved
    assert archives[0].stat().st_mode & 0o777 == 0o600
    assert reset_empty_reflections(path)["reset_failed_invocations"] == 0
