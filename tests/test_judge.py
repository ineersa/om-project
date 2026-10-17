from copy import deepcopy
import json
from pathlib import Path

import pytest
from jsonschema import ValidationError

from om_dataset import judge as j
from om_dataset.collection import write_json
from om_dataset.redaction import Redactor


@pytest.fixture
def setup(tmp_path):
    redactor = Redactor(catalog=None)
    control = json.loads(j.CONTROLS.read_text())[0]
    case = deepcopy(control["case"])
    case["redaction_id"] = redactor.fingerprint
    policy = {"instructions": "Use the permitted evidence.", "tools": []}
    root = tmp_path / "results"
    root.mkdir()
    return case, policy, redactor, root


def assessment(case, preference="A"):
    return {
        "preference": preference, "reason": "A preserves the proposal status; B falsely claims completion.",
        "candidates": {
            "A": {"verdict": "acceptable", "summary": "Preserves the supported plan.", "issues": []},
            "B": {"verdict": "needs_correction", "summary": "False completion.", "issues": [{
                "category": "status_or_scope", "severity": "critical",
                "candidate_excerpt": "The billing database migration completed successfully.",
                "explanation": "The source states that the migration has not run.",
                "evidence": [{"source_id": "synthetic-input", "quote": "The migration has not run."}],
            }]},
        }, "limitations": [],
    }


def response(value, finish="stop"):
    return {"choices": [{"finish_reason": finish, "message": {"role": "assistant", "content": json.dumps(value)}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 30}}


@pytest.mark.parametrize("control", json.loads(j.CONTROLS.read_text()),
                         ids=lambda control: control["case"]["case_id"])
def test_synthetic_controls_are_valid_and_blinded(control):
    case = control["case"]
    j.validate_case(case, Redactor(catalog=None), synthetic=True)
    assert set(control["expected"]) == {"preference", "A", "B"}
    assert control["expected"]["preference"] in j.CHOICES
    body = j.build_request(case, {"instructions": "Use the permitted evidence.", "tools": []},
                           j.PROMPT.read_text())
    payload = json.loads(body["messages"][1]["content"])
    assert "expected" not in payload and "case_id" not in payload["case"]
    assert body["messages"][0]["content"] == j.PROMPT.read_text()


@pytest.mark.parametrize("case_id", ["control-observer-compression", "control-reflector-compression"])
def test_compression_can_choose_between_two_acceptable_candidates(case_id):
    control = next(c for c in json.loads(j.CONTROLS.read_text()) if c["case"]["case_id"] == case_id)
    case = control["case"]
    value = {"preference": "A", "reason": "A preserves the core information in less memory text.",
             "candidates": {label: {"verdict": "acceptable", "summary": "Supported memory.", "issues": []}
                            for label in ("A", "B")}, "limitations": []}
    assert control["expected"] == {"preference": "A", "A": "acceptable", "B": "acceptable"}
    for swapped in (False, True):
        shown = deepcopy(value)
        shown["preference"] = "B" if swapped else "A"
        j.validate_assessment(shown, case, swapped=swapped)
        assert j.normalize(shown, swapped)["preference"] == "A"


def test_request_blinding_and_swapped_order(setup):
    case, policy, _, _ = setup
    case.update(human_vote="SECRET_HUMAN_VOTE", producer="SECRET_PRODUCER", expected="SECRET_EXPECTED")
    body = j.build_request(case, policy, "Judge fairly.", swapped=True)
    encoded = json.dumps(body)
    assert not any(marker in encoded for marker in ("SECRET_HUMAN_VOTE", "SECRET_PRODUCER", "SECRET_EXPECTED"))
    assert case["case_id"] not in encoded
    public = json.loads(body["messages"][1]["content"])["case"]
    assert public["candidates"] == {"A": case["candidates"]["B"], "B": case["candidates"]["A"]}
    assert body["thinking"] == {"type": "enabled"} and body["reasoning_effort"] == "high"
    assert "tools" not in body and "api_key_env" not in body


def test_validation_requires_real_quotes_and_consistent_verdicts(setup):
    case, _, redactor, _ = setup
    j.validate_case(case, redactor)
    value = assessment(case)
    assert j.validate_assessment(value, case) == value
    swapped = deepcopy(value)
    swapped["candidates"] = {"A": value["candidates"]["B"], "B": value["candidates"]["A"]}
    swapped["preference"] = "B"
    assert j.validate_assessment(swapped, case, swapped=True)
    assert j.normalize(swapped, True)["candidates"] == value["candidates"]
    for change in ("source", "evidence_quote", "candidate_quote", "no_issue", "inconsistent", "both", "extra"):
        broken = deepcopy(value)
        issue = broken["candidates"]["B"]["issues"][0]
        if change == "source":
            issue["evidence"][0]["source_id"] = "invented"
        elif change == "evidence_quote":
            issue["evidence"][0]["quote"] = "The migration failed."
        elif change == "candidate_quote":
            issue["candidate_excerpt"] = "Not in the candidate."
        elif change == "no_issue":
            broken["candidates"]["B"]["issues"] = []
        elif change == "inconsistent":
            broken["candidates"]["B"]["verdict"] = "acceptable"
        elif change == "both":
            broken["preference"] = "both"
        else:
            broken["automatic_training_approval"] = True
        with pytest.raises((ValueError, ValidationError)):
            j.validate_assessment(broken, case)
    identical = deepcopy(case)
    identical["candidates"]["B"] = identical["candidates"]["A"]
    same = deepcopy(value)
    same["candidates"]["B"] = deepcopy(same["candidates"]["A"])
    with pytest.raises(ValueError, match="Identical"):
        j.validate_assessment(same, identical)


def test_case_validation_rejects_secrets_and_stale_redaction(setup):
    case, _, redactor, _ = setup
    bad = deepcopy(case)
    bad["redaction_id"] = "stale"
    with pytest.raises(ValueError, match="redaction"):
        j.validate_case(bad, redactor)
    bad = deepcopy(case)
    bad["case_id"] = "../escape"
    with pytest.raises(ValueError, match="Unsafe"):
        j.validate_case(bad, redactor)
    bad = deepcopy(case)
    bad["evidence"].append(deepcopy(bad["evidence"][0]))
    with pytest.raises(ValueError, match="Duplicate"):
        j.validate_case(bad, redactor)
    bad = deepcopy(case)
    bad["candidates"]["A"] += " sk-" + "a" * 30
    with pytest.raises(ValueError, match="Unredacted"):
        j.validate_case(bad, redactor)


def test_format_repairs_preserve_decisions_and_verify_nested_evidence_scope(setup):
    case, _, _, _ = setup
    case["evidence"][0]["text"] = ("[Source entry id: run:1]\n[User @ 2026-10-16 10:00]: "
                                   "The migration has not run.\n\n[Source entry id: run:2]\n"
                                   "[Assistant @ 2026-10-16 10:01]: Unrelated follow-up.")
    raw = assessment(case)
    raw["candidates"]["A"]["summary_note"] = None
    raw["candidates"]["B"]["summary_extra"] = ""
    raw["candidates"]["B"]["issues"][0]["evidence"][0]["source_id"] = "run:1"
    repaired, changes = j.repair_format(raw, case)
    assert j.validate_assessment(repaired, case)
    assert len(changes) == 3
    assert repaired["preference"] == raw["preference"]
    assert raw["candidates"]["A"]["summary_note"] is None  # Original is untouched.
    for label in ("A", "B"):
        assert repaired["candidates"][label]["verdict"] == raw["candidates"][label]["verdict"]
    wrong_scope = deepcopy(raw)
    wrong_scope["candidates"]["B"]["issues"][0]["evidence"][0]["source_id"] = "run:2"
    repaired, _ = j.repair_format(wrong_scope, case)
    with pytest.raises(ValueError, match="Evidence"):
        j.validate_assessment(repaired, case)
    nonempty_note = deepcopy(raw)
    nonempty_note["candidates"]["A"]["summary_note"] = "A hidden extra assessment."
    repaired, _ = j.repair_format(nonempty_note, case)
    with pytest.raises(ValidationError):
        j.validate_assessment(repaired, case)
    case["evidence"].append({"source_id": "second-block", "text": case["evidence"][0]["text"]})
    repaired, _ = j.repair_format(raw, case)
    with pytest.raises(ValueError, match="Evidence"):
        j.validate_assessment(repaired, case)


def test_completed_cache_does_not_repeat_calls_or_accept_changed_inputs(setup):
    case, policy, redactor, root = setup
    calls = []
    def request(body):
        calls.append(body)
        return response(assessment(case))
    result = j.assess(case, policy, "Prompt v1", False, root, redactor, "run", request=request)
    assert result["status"] == "completed" and not result["training_ready"]
    assert len(calls) == 1
    def forbidden(body):
        raise AssertionError("Cached result must not generate")
    assert j.assess(case, policy, "Prompt v1", False, root, redactor, "run", request=forbidden) == result
    with pytest.raises(ValueError, match="changed"):
        j.assess(case, policy, "Prompt v2", False, root, redactor, "run", request=forbidden)
    assert next(root.glob("*.json")).stat().st_mode & 0o777 == 0o600


def test_reuse_revalidates_identical_recorded_reply_without_new_calls(setup, tmp_path):
    case, policy, redactor, root = setup
    source = tmp_path / "previous"
    (source / "results").mkdir(parents=True)
    value = assessment(case)
    value["candidates"]["A"]["verdict_reason_not_needed"] = None
    path = source / "results" / f"{case['case_id']}.original.json"
    old = {"case_id": case["case_id"], "case_sha256": j.digest(case), "swapped": False,
           "run_id": "old-run", "request_sha256": j.digest(j.build_request(case, policy, "Prompt")),
           "status": "failed", "raw_content": json.dumps(value), "finish_reason": "stop",
           "usage": {"completion_tokens": 20}}
    write_json(path, old)
    before = path.read_bytes()
    def forbidden(body):
        raise AssertionError("Reuse must not call a model")
    result = j.assess(case, policy, "Prompt", False, root, redactor, "new-run",
                      request=forbidden, reuse=(source, "old-run"))
    assert result["status"] == "completed" and result["format_repairs"]
    assert result["assessment"]["preference"] == "A"
    assert result["raw_content"] == old["raw_content"]
    assert path.read_bytes() == before
    assert result["reused_prediction"]["run_id"] == "old-run"
    different = tmp_path / "different"
    different.mkdir()
    with pytest.raises(ValueError, match="does not match"):
        j.assess(case, policy, "Changed prompt", False, different, redactor, "new-run",
                 request=forbidden, reuse=(source, "old-run"))


@pytest.mark.parametrize("kind", ["truncated", "malformed", "fabricated_quote", "provider", "secret"])
def test_bad_replies_are_failures_not_noops_or_acceptances(setup, kind):
    case, policy, redactor, root = setup
    def request(body):
        if kind == "provider":
            raise RuntimeError("secret provider exception must not be saved")
        value = assessment(case)
        reply = response(value, finish="length" if kind == "truncated" else "stop")
        if kind == "malformed":
            reply["choices"][0]["message"]["content"] = "not json"
        elif kind == "fabricated_quote":
            value["candidates"]["B"]["issues"][0]["evidence"][0]["quote"] = "Invented evidence."
            reply = response(value)
        elif kind == "secret":
            reply["choices"][0]["message"]["content"] = "sk-" + "b" * 30
        return reply
    result = j.assess(case, policy, "Prompt", False, root, redactor, "run", request=request)
    assert result["status"] == "failed" and "assessment" not in result
    saved = next(root.glob("*.json")).read_text()
    assert "secret provider exception" not in saved and "sk-" not in saved
    assert result["failure_stage"] in {"provider", "response", "parse", "validation"}


def test_votes_are_hash_checked_and_never_changed(setup, tmp_path):
    case, _, redactor, _ = setup
    votes = tmp_path / "votes"
    votes.mkdir()
    path = votes / (case["case_id"] + ".json")
    vote = {"schema_version": "om-review-vote/1", "answer_source": "human", "case_id": case["case_id"],
            "case_sha256": j.digest(case), "choice": "A", "reason": "Status matters."}
    write_json(path, vote)
    before = path.read_bytes()
    assert j.load_votes([case], votes, redactor)[case["case_id"]]["choice"] == "A"
    assert path.read_bytes() == before
    changed = deepcopy(case)
    changed["comparison_scope"] += " Changed."
    with pytest.raises(ValueError, match="does not match"):
        j.load_votes([changed], votes, redactor)


def test_report_normalizes_positions_and_separates_missing_or_bad_calls(setup):
    case, _, _, _ = setup
    good = assessment(case)
    flipped = deepcopy(good)
    flipped["preference"] = "B"
    flipped["candidates"] = {"A": good["candidates"]["B"], "B": good["candidates"]["A"]}
    results = [{"case_id": case["case_id"], "swapped": False, "status": "completed", "assessment": good},
               {"case_id": case["case_id"], "swapped": True, "status": "completed", "assessment": flipped}]
    votes = {case["case_id"]: {"choice": "A", "reason": "", "vote_sha256": "hash"}}
    summary = j.report([case], [], results, votes)
    assert summary["counts"]["agrees_both_orders"] == summary["counts"]["order_stable"] == 1
    assert not summary["calibrated"] and not summary["training_ready"]
    results[1]["status"] = "failed"
    summary = j.report([case], [], results, votes)
    assert summary["counts"]["failed_calls"] == 1
    assert summary["cases"][0]["agrees_both_orders"] is None
    assert summary["counts"]["agrees_both_orders"] == 0
    control = {"case": case, "expected": {"preference": "A", "A": "acceptable", "B": "needs_correction"}}
    results[1]["status"] = "completed"
    assert j.report([], [control], results, {})["counts"]["controls_pass_both_orders"] == 1


def test_dry_run_has_no_network_writes_or_vote_access(monkeypatch, setup, tmp_path, capsys):
    case, policy, redactor, _ = setup
    monkeypatch.setattr(j.comparison, "_redactor", lambda: redactor)
    monkeypatch.setattr(j, "load_inputs", lambda r: ([case], [], {case["task"]: policy}))
    def forbidden(*args, **kwargs):
        raise AssertionError("Dry-run must not access votes, write files, or call providers")
    monkeypatch.setattr(j, "load_votes", forbidden)
    monkeypatch.setattr(j, "write_json", forbidden)
    monkeypatch.setattr(j.teacher, "complete_request", forbidden)
    root = j.MAIN / "data/judge-pilot/nonexistent-test-output"
    assert not root.exists()
    assert j.main(["--dry-run", "--output", str(root)]) == 0
    assert json.loads(capsys.readouterr().out)["provider_calls"] == 0
    assert not root.exists()
    with pytest.raises(ValueError, match="versioned"):
        j.main(["--dry-run", "--output", str(j.MAIN / "data/review")])


def test_limited_run_resumes_without_vote_leakage_and_report_only_is_offline(monkeypatch, setup, tmp_path):
    case, policy, redactor, _ = setup
    monkeypatch.setattr(j, "MAIN", tmp_path)
    monkeypatch.setattr(j.comparison, "_redactor", lambda: redactor)
    monkeypatch.setattr(j, "load_inputs", lambda r: ([case], [], {case["task"]: policy}))
    monkeypatch.setattr(j, "file_hash", lambda p: "fixture-" + p.name)
    monkeypatch.setenv(j.CONFIG["api_key_env"], "test")
    root = tmp_path / "data/judge-pilot/test"
    events = []
    def request(body, run, config, timeout):
        events.append("prediction")
        assert "SECRET_VOTE" not in json.dumps(body)
        value = assessment(case)
        if len(events) > 1:
            value["preference"] = "B"
            value["candidates"] = {"A": value["candidates"]["B"], "B": value["candidates"]["A"]}
        return response(value)
    def votes(cases, path, r):
        events.append("votes")
        return {case["case_id"]: {"choice": "A", "reason": "SECRET_VOTE", "vote_sha256": "hash"}}
    monkeypatch.setattr(j.teacher, "complete_request", request)
    monkeypatch.setattr(j, "load_votes", votes)
    assert j.main(["--output", str(root), "--max-new-calls", "1"]) == 0
    assert events == ["prediction", "votes"]
    assert json.loads((root / "report.json").read_text())["counts"]["missing_calls"] == 1
    assert j.main(["--output", str(root), "--max-new-calls", "1"]) == 0
    assert events == ["prediction", "votes", "prediction", "votes"]
    summary = json.loads((root / "report.json").read_text())
    assert summary["counts"]["missing_calls"] == 0
    assert summary["counts"]["agrees_both_orders"] == 1
    before = {p.name: p.read_bytes() for p in (root / "results").glob("*.json")}
    def forbidden(*args, **kwargs):
        raise AssertionError("Report-only must not call the provider")
    monkeypatch.setattr(j.teacher, "complete_request", forbidden)
    assert j.main(["--output", str(root), "--report-only"]) == 0
    assert {p.name: p.read_bytes() for p in (root / "results").glob("*.json")} == before
    assert root.stat().st_mode & 0o777 == 0o700
    with pytest.raises(SystemExit):
        j.main(["--output", str(root), "--max-new-calls", "0"])
