from copy import deepcopy
import json
from urllib.error import HTTPError

import pytest

from om_dataset import jev_judge as j
from om_dataset.collection import write_json
from om_dataset.redaction import Redactor


POLICY = {"instructions": "Use the permitted evidence.", "tools": [{
    "name": "record_memory", "description": "Record memory.", "parameters": {"type": "object"},
    "handler_source": "SECRET_HANDLER", "handler_binding": "SECRET_BINDING",
}]}


@pytest.fixture
def setup(tmp_path):
    case = deepcopy(json.loads(j.judge.CONTROLS.read_text())[0]["case"])
    policy = deepcopy(POLICY)
    body = j.build_request(case, policy, j.PROMPT.read_text())
    root = tmp_path / "results"
    root.mkdir()
    return case, policy, body, Redactor(catalog=None), root


def response(body, **selections):
    selections = dict(selections)
    for key, choice in list(selections.items()):
        if choice == "needs_work" and key + "_location" in body["questions"]:
            selections.setdefault(key + "_location", next(option for option in
                body["questions"][key + "_location"]["criteria"] if option not in {"none", "uncertain"}))
    answers = {}
    for key, question in body["questions"].items():
        default = "A" if key == "preference" else "none" if key.endswith("_location") else "pass"
        choice = selections.get(key, default)
        answers[key] = {"type": "choice", "choice": choice, "confidence": 1.0,
                        "probabilities": {option: float(option == choice) for option in question["criteria"]}}
    return {"model": body["model"], "answers": answers, "usage": {"input_tokens": 100, "output_tokens": 20}}


def test_request_is_typed_blinded_and_swaps_metrics(setup):
    case, policy, _, _, _ = setup
    case.update(human_vote="SECRET_VOTE", producer="SECRET_PRODUCER", expected="SECRET_EXPECTED")
    body = j.build_request(case, policy, j.PROMPT.read_text(), swapped=True)
    encoded = json.dumps(body)
    assert not any(secret in encoded for secret in (
        "SECRET_VOTE", "SECRET_PRODUCER", "SECRET_EXPECTED", "SECRET_HANDLER", "SECRET_BINDING", case["case_id"]))
    assert set(body) == {"model", "state", "questions"}
    assert all(q["type"] == "choice" for q in body["questions"].values())
    assert body["state"]["case"]["candidates"]["A"] == case["candidates"]["B"]
    assert body["state"]["display_metrics"]["A"]["characters"] == len(case["candidates"]["B"])
    assert "Prefer three useful records over ten" in body["state"]["rubric"]["compression"]
    # Meaning must be in instructions, since Jev does not see question IDs.
    assert "case.candidates.A" in body["questions"]["A_useful_coverage"]["instructions"]
    assert body["state"]["worker_policy"] == {
        "instructions": policy["instructions"], "tools": [j.teacher._tool(policy)]}
    assert not any("severity" in key or "verdict" in key or "unsupported" in key for key in body["questions"])


ALL_CONTROLS = (json.loads(j.judge.CONTROLS.read_text()) + json.loads(j.EXTRA_CONTROLS.read_text())
                + json.loads(j.ROLE_CONTROLS.read_text()))


@pytest.mark.parametrize("control", ALL_CONTROLS, ids=lambda c: c["case"]["case_id"])
def test_all_roles_build_and_validate_typed_answers(control):
    case = control["case"]
    j.judge.validate_case(case, Redactor(catalog=None), synthetic=True)
    body = j.build_request(case, POLICY, j.PROMPT.read_text())
    statuses = {key.removeprefix("A_") for key in body["questions"]
                if key.startswith("A_") and not key.endswith("_location")}
    expected = {"observer": {"useful_coverage", "context_recoverability", "importance_assignment", "compression"},
                "reflector": {"useful_coverage", "context_recoverability", "compression"},
                "dropper": {"safe_removal"}}[case["task"]]
    assert statuses == expected
    assert all(len(q["criteria"]) <= 255 for q in body["questions"].values())
    assert all(set(body["questions"][f"A_{check}"]["criteria"]) == {"pass", "needs_work", "uncertain"} for check in statuses)
    assert j.validate_response(response(body), body)


@pytest.mark.parametrize("change", ["model", "missing", "extra", "type", "choice", "distribution", "sum", "boolean", "nan", "maximum", "confidence", "usage"])
def test_rejects_invalid_typed_responses(setup, change):
    _, _, body, _, _ = setup
    value = response(body)
    answer = value["answers"]["preference"]
    if change == "model":
        value["model"] = "jev-latest"
    elif change == "missing":
        del value["answers"]["A_useful_coverage"]
    elif change == "extra":
        value["answers"]["extra"] = answer
    elif change == "type":
        answer["type"] = "noul"
    elif change == "choice":
        answer["choice"] = "invented"
    elif change == "distribution":
        answer["probabilities"]["invented"] = 0
    elif change == "sum":
        answer["probabilities"]["A"] = 0.5
    elif change == "boolean":
        answer["probabilities"]["A"] = True
    elif change == "nan":
        answer["probabilities"]["A"] = float("nan")
    elif change == "maximum":
        answer["choice"] = "B"
    elif change == "confidence":
        answer["confidence"] = 2
    elif change == "usage":
        value["usage"]["input_tokens"] = -1
    with pytest.raises(ValueError):
        j.validate_response(value, body)


def test_compact_candidate_can_win_without_rejecting_other(setup):
    body = setup[2]
    value = j.assessment(response(body), body)
    assert value["preference"] == "A"
    assert all(c["verdict"] == "acceptable" for c in value["candidates"].values())
    assert all(c["review_status"] == "pass" for c in value["candidates"].values())
    assert not value["adjustments"] and not value["rationale_available"]


def test_live_api_probability_rounding_is_accepted_without_normalizing(setup):
    body = setup[2]
    value = response(body)
    value["answers"]["A_useful_coverage"] = {
        "type": "choice", "choice": "needs_work", "confidence": 0.72,
        "probabilities": {"pass": 0.17, "needs_work": 0.81, "uncertain": 0.01},
    }
    before = deepcopy(value)
    assert j.validate_response(value, body) == before


@pytest.mark.parametrize("check", ["useful_coverage", "context_recoverability", "importance_assignment"])
def test_failed_core_check_cannot_be_compensated_by_compression(setup, check):
    value = j.assessment(response(setup[2], **{f"A_{check}": "needs_work"}), setup[2])
    candidate = value["candidates"]["A"]
    assert candidate["verdict"] == "needs_correction" and candidate["review_status"] == "needs_work"
    assert candidate["checks"][check]["location_id"] is not None
    assert value["raw_preference"] == "A" and value["preference"] == "B"
    assert len(value["adjustments"]) == 1


def test_unsafe_drop_cannot_be_compensated_by_pairwise_preference(setup):
    case = next(c["case"] for c in ALL_CONTROLS if c["case"]["task"] == "dropper")
    body = j.build_request(case, POLICY, j.PROMPT.read_text())
    value = j.assessment(response(body, A_safe_removal="needs_work"), body)
    assert value["candidates"]["A"]["verdict"] == "needs_correction"
    assert value["preference"] == "B"


@pytest.mark.parametrize("selections", [
    {"A_useful_coverage": "uncertain"}, {"A_context_recoverability": "uncertain"},
    {"A_importance_assignment": "uncertain"},
    {"A_useful_coverage": "needs_work", "A_useful_coverage_location": "none"},
    {"A_useful_coverage": "needs_work", "A_useful_coverage_location": "uncertain"},
])
def test_missing_evidence_or_unlocated_failure_is_not_an_approval(setup, selections):
    value = j.assessment(response(setup[2], **selections), setup[2])
    assert value["candidates"]["A"]["verdict"] == "uncertain"
    assert value["preference"] == "uncertain"


def test_unused_speculative_locations_do_not_create_defects(setup):
    body = setup[2]
    value = j.assessment(response(body, A_useful_coverage_location="record_1"), body)
    candidate = value["candidates"]["A"]
    assert candidate["verdict"] == "acceptable"
    assert all(item["status"] == "pass" and item["location_id"] is None for item in candidate["checks"].values())


def test_compression_work_is_reviewed_separately_from_information_adequacy(setup):
    body = setup[2]
    value = j.assessment(response(body, A_compression="needs_work"), body)
    candidate = value["candidates"]["A"]
    assert candidate["verdict"] == "acceptable" and candidate["review_status"] == "needs_work"
    assert candidate["checks"]["compression"]["status"] == "needs_work"


def test_localization_options_identify_records_and_native_source_facts(setup):
    case, policy, _, _, _ = setup
    case["candidates"]["A"] = "Status: completed\n\n1. First claim.\nSupporting observations: 8\n\n2. Second claim."
    case["evidence"] = [{"source_id": "input", "text": "CURRENT OBSERVATIONS:\n[8] 2026-09-17 20:10 [high] New state.\n[3] 2026-09-17 17:50 [high] Old state."}]
    body = j.build_request(case, policy, j.PROMPT.read_text())
    options = body["questions"]["A_useful_coverage_location"]["criteria"]
    assert set(options) == {"none", "uncertain", "whole_candidate", "record_1", "record_2", "input:observation:8", "input:observation:3"}
    value = j.assessment(response(body, A_useful_coverage="needs_work", A_useful_coverage_location="record_2",
                                  A_context_recoverability="needs_work", A_context_recoverability_location="input:observation:8"), body)
    checks = value["candidates"]["A"]["checks"]
    assert checks["useful_coverage"]["location_id"] == "record_2"
    assert checks["context_recoverability"]["location_kind"] == "source_fact"


def test_context_estimates_do_not_truncate_and_long_context_is_explicit(setup):
    body = setup[2]
    body["state"]["long_evidence"] = "x" * 100000
    before = deepcopy(body)
    assert j.context_estimate(body)["exceeds_estimate"]
    assert not j.context_estimate(body, True)["exceeds_estimate"]
    assert body == before


def test_duplicate_and_empty_record_numbers_never_hide_or_misidentify_text():
    assert set(j.candidate_options("1. First\n1. Second", "A")) == {"whole_candidate"}
    assert set(j.candidate_options("1. \n2. Second", "A")) == {"record_1", "record_2"}


def test_report_distinguishes_status_stability_from_location_stability(setup):
    case, policy, _, _, _ = setup
    case["candidates"] = {"A": "1. First\n2. Second", "B": "1. Other"}
    rows = []
    for swapped in (False, True):
        body = j.build_request(case, policy, j.PROMPT.read_text(), swapped=swapped)
        label = "B" if swapped else "A"
        selections = {f"{label}_useful_coverage": "needs_work",
                      f"{label}_useful_coverage_location": "record_2" if swapped else "record_1"}
        rows.append({"case_id": case["case_id"], "swapped": swapped, "status": "completed",
                     "assessment": j.assessment(response(body, **selections), body)})
    row = j.report([case], [], rows, {})["cases"][0]
    assert row["check_statuses_stable"] and not row["check_locations_stable"]


def test_controls_have_blinded_check_targets_and_unique_ids(monkeypatch):
    base = json.loads(j.judge.CONTROLS.read_text())
    monkeypatch.setattr(j.judge, "load_inputs", lambda redactor: ([], base, {}))
    _, controls, _ = j.load_inputs(Redactor(catalog=None))
    assert len(controls) == len({c["case"]["case_id"] for c in controls}) == 12
    for control in controls:
        body = j.build_request(control["case"], POLICY, j.PROMPT.read_text())
        assert "expected_checks" not in json.dumps(body)
        assert set(control["expected_checks"]) == {"A", "B"}


def test_cache_is_private_reused_and_tampering_is_rejected(setup):
    case, _, body, redactor, root = setup
    calls = []
    def request(body):
        calls.append(body)
        return response(body)
    first = j.assess(case, body, root, redactor, "run", request=request)
    assert j.assess(case, body, root, redactor, "run", request=request) == first
    assert len(calls) == 1
    path = root / f"{case['case_id']}.original.json"
    assert path.stat().st_mode & 0o777 == 0o600
    first["assessment"]["preference"] = "B"
    write_json(path, first)
    with pytest.raises(ValueError, match="Cached assessment"):
        j.assess(case, body, root, redactor, "run", request=request)


def test_service_failures_and_interrupted_calls_do_not_retry(setup):
    case, _, body, redactor, root = setup
    calls = []
    def request(body):
        calls.append(body)
        raise HTTPError("/v1/systemone", 429, "must not persist secret error text", None, None)
    result = j.assess(case, body, root, redactor, "run", request=request)
    assert result["status"] == "failed" and result["http_status"] == 429
    assert "secret error text" not in json.dumps(result)
    assert j.assess(case, body, root, redactor, "run", request=request) == result
    assert len(calls) == 1
    path = root / f"{case['case_id']}.original.json"
    result["status"] = "in_progress"
    write_json(path, result)
    interrupted = j.assess(case, body, root, redactor, "run", request=request)
    assert interrupted["failure_stage"] == "interrupted" and len(calls) == 1


def test_identical_candidates_still_have_two_position_records(setup):
    case, policy, _, redactor, root = setup
    case["candidates"]["B"] = case["candidates"]["A"]
    for swapped in (False, True):
        body = j.build_request(case, policy, j.PROMPT.read_text(), swapped=swapped)
        result = j.assess(case, body, root, redactor, "run", swapped=swapped, request=response)
        assert result["assessment"]["preference"] == "both"
    assert len(list(root.glob('*.json'))) == 2


def test_report_maps_raw_and_composed_labels_back_to_original(setup):
    case, _, body, _, _ = setup
    rows = [{"case_id": case["case_id"], "swapped": swapped, "status": "completed",
             "assessment": j.assessment(response(body, preference="B" if swapped else "A"), body)}
            for swapped in (False, True)]
    before = deepcopy(rows)
    summary = j.report([case], [], rows, {case["case_id"]: {"choice": "A"}})
    row = summary["cases"][0]
    assert row["swapped_normalized"]["preference"] == row["swapped_normalized"]["raw_preference"] == "A"
    assert summary["counts"]["paired_valid_review_cases"] == summary["counts"]["agrees_both_orders"] == 1
    assert summary["raw_preference_counts"]["agrees_both_orders"] == 1
    assert row["check_statuses_stable"] and row["check_locations_stable"]
    assert not summary["calibrated"] and not summary["training_ready"]
    assert rows == before


def test_swapped_adjustments_are_mapped_without_changing_raw_records(setup):
    case, _, body, _, _ = setup
    value = j.assessment(response(body, preference="B", B_useful_coverage="needs_work"), body)
    rows = [{"case_id": case["case_id"], "swapped": True, "status": "completed", "assessment": value}]
    before = deepcopy(rows)
    swapped = j.report([case], [], rows, {})["cases"][0]["swapped_normalized"]
    assert swapped["raw_preference"] == "A" and swapped["preference"] == "B"
    assert swapped["candidates"]["A"]["checks"]["useful_coverage"]["location_id"] is not None
    assert swapped["adjustments"][0]["from"] == "A" and swapped["adjustments"][0]["to"] == "B"
    assert rows == before


def test_read_only_cannot_call_provider_or_modify_interrupted_records(setup):
    case, _, body, redactor, root = setup
    with pytest.raises(ValueError, match="Missing prediction"):
        j.assess(case, body, root, redactor, "run", read_only=True, request=lambda body: pytest.fail("network"))
    result = j.assess(case, body, root, redactor, "run", request=response)
    path = root / f"{case['case_id']}.original.json"
    result["status"] = "in_progress"
    write_json(path, result)
    before = path.read_bytes()
    observed = j.assess(case, body, root, redactor, "run", read_only=True, request=lambda body: pytest.fail("network"))
    assert observed["failure_stage"] == "interrupted" and path.read_bytes() == before


def test_http_client_uses_user_key_and_explicit_long_context(monkeypatch, setup):
    captured = {}
    class Reply:
        status = 200
        def read(self, *args):
            return json.dumps(response(setup[2])).encode()
    class Connection:
        def __init__(self, host, timeout):
            captured.update(host=host, timeout=timeout)
        def request(self, method, path, **kwargs):
            captured.update(method=method, path=path, **kwargs)
        def getresponse(self):
            return Reply()
        def close(self):
            captured["closed"] = True
    monkeypatch.setenv("JEV_API_KEY", "test-key")
    monkeypatch.setattr(j, "HTTPSConnection", Connection)
    assert j.complete_request(setup[2], long_context=True)
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["headers"]["TypeSafe-Danger-Zone-Long-Context"] == "true"
    assert captured["path"] == "/v1/systemone" and captured["closed"]


def test_cli_dry_run_resume_report_only_and_pins(monkeypatch, setup, capsys):
    case, policy, _, redactor, root = setup
    monkeypatch.setattr(j.comparison, "_redactor", lambda: redactor)
    monkeypatch.setattr(j, "load_inputs", lambda redactor: ([case], [], {case["task"]: policy}))
    monkeypatch.setattr(j.judge, "load_votes", lambda *args: {})
    monkeypatch.setenv("JEV_API_KEY", "test-key")
    calls = []
    def request(body, **kwargs):
        calls.append(body)
        return response(body)
    monkeypatch.setattr(j, "complete_request", request)
    # The CLI restricts outputs to the project's private run tree.
    monkeypatch.setattr(j, "RUNS_ROOT", root / 'data/jev-judge')
    target = root / 'data/jev-judge/test-v1'
    args = ['--output', str(target)]
    assert j.main(args + ['--dry-run']) == 0 and not calls and not target.exists()
    assert j.main(args + ['--max-new-calls', '1']) == 0 and len(calls) == 1
    assert j.main(args) == 0 and len(calls) == 2
    assert j.main(args + ['--report-only']) == 0 and len(calls) == 2
    with pytest.raises(ValueError, match='Pinned Jev run changed'):
        j.main(args + ['--long-context'])
    with pytest.raises(ValueError, match='Pinned Jev run changed'):
        j.main(args + ['--min-confidence', '0.90'])


def test_strict_judge_criteria_are_role_specific():
    prompt = j.PROMPT.read_text()
    for task in ("reflector", "dropper"):
        case = next(c["case"] for c in ALL_CONTROLS if c["case"]["task"] == task)
        body = j.build_request(case, POLICY, prompt)
        assert body["state"]["worker_policy"]["instructions"] == POLICY["instructions"]
        if task == "dropper":
            assert "ALL remaining memory" in body["questions"]["A_safe_removal"]["criteria"]["needs_work"]
            assert "Another observation scheduled for deletion" in body["state"]["rubric"]["role"]
        else:
            assert "unreflected" in body["questions"]["A_useful_coverage"]["criteria"]["needs_work"]
            assert "coverage problem, not this problem" in body["questions"]["A_context_recoverability"]["criteria"]["needs_work"]


@pytest.mark.parametrize("confidence, expected", [(0.8499, "uncertain"), (0.85, "pass"), (0.90, "pass")])
def test_confidence_cutoff_is_inclusive_and_uses_confidence_not_probability(setup, confidence, expected):
    body = setup[2]
    value = response(body)
    value["answers"]["A_useful_coverage"]["confidence"] = confidence
    before = deepcopy(value)
    raw = j.assessment(value, body)
    gated = j.assessment(value, body, min_confidence=.85)
    assert raw["candidates"]["A"]["checks"]["useful_coverage"]["status"] == "pass"
    assert gated["candidates"]["A"]["checks"]["useful_coverage"]["status"] == expected
    assert value == before


def test_low_confidence_failure_is_review_not_rejection_and_ignores_its_location(setup):
    body = setup[2]
    value = response(body, A_useful_coverage="needs_work")
    value["answers"]["A_useful_coverage"]["confidence"] = .84
    result = j.assessment(value, body, min_confidence=.85)
    candidate = result["candidates"]["A"]
    assert candidate["review_status"] == candidate["verdict"] == "uncertain"
    assert candidate["checks"]["useful_coverage"]["location_id"] is None
    assert result["preference"] == "uncertain"


def test_low_confidence_pairwise_preference_abstains(setup):
    body = setup[2]
    value = response(body)
    value["answers"]["preference"]["confidence"] = .84
    result = j.assessment(value, body, min_confidence=.85)
    assert result["raw_preference"] == "A" and result["preference"] == "uncertain"
    assert result["adjustments"][0]["rule"] == "preference_confidence_below_cutoff"


def test_confidence_cache_is_revalidated_and_cutoff_is_pinned(setup):
    case, _, body, redactor, root = setup
    result = j.assess(case, body, root, redactor, "run", request=response)
    with pytest.raises(ValueError, match="Pinned Jev request changed"):
        j.assess(case, body, root, redactor, "run", min_confidence=.9, request=lambda body: pytest.fail("network"))
    result["confidence_assessment"]["candidates"]["A"]["checks"]["useful_coverage"]["status"] = "uncertain"
    write_json(root / f"{case['case_id']}.original.json", result)
    with pytest.raises(ValueError, match="Cached confidence assessment"):
        j.assess(case, body, root, redactor, "run", request=lambda body: pytest.fail("network"))


def test_cli_task_filter_does_not_send_other_roles(monkeypatch, setup, capsys):
    case, policy, _, redactor, root = setup
    reflector = next(c["case"] for c in ALL_CONTROLS if c["case"]["task"] == "reflector")
    monkeypatch.setattr(j.comparison, "_redactor", lambda: redactor)
    monkeypatch.setattr(j, "load_inputs", lambda redactor: ([case, reflector], [], {"observer": policy, "reflector": policy}))
    monkeypatch.setattr(j.judge, "load_votes", lambda *args: {})
    monkeypatch.setenv("JEV_API_KEY", "test-key")
    calls = []
    def request(body, **kwargs):
        calls.append(body["state"]["case"]["task"])
        return response(body)
    monkeypatch.setattr(j, "complete_request", request)
    monkeypatch.setattr(j, "RUNS_ROOT", root / "runs")
    args = ["--tasks", "reflector", "--min-confidence", "0.85", "--output", str(root / "runs/filtered")]
    j.main(args + ["--dry-run"])
    assert json.loads(capsys.readouterr().out)["planned_calls"] == 2
    j.main(args)
    assert calls == ["reflector", "reflector"]
    saved = json.loads((root / "runs/filtered/report.json").read_text())
    assert saved["min_confidence"] == .85
    assert saved["counts"]["review_cases"] == 1


def test_report_compares_complete_gated_and_ungated_counts_without_mutation(setup):
    case, _, body, redactor, root = setup
    def request(body):
        value = response(body)
        value["answers"]["A_useful_coverage"]["confidence"] = .84
        return value
    result = j.assess(case, body, root, redactor, "run", request=request)
    before = deepcopy(result)
    summary = j.report([case], [], [result], {}, min_confidence=.85)
    assert set(summary["counts"]) == set(summary["ungated_counts"])
    assert summary["ungated_check_status_counts"]["pass"] == summary["check_status_counts"]["pass"] + 1
    assert summary["check_status_counts"]["uncertain"] == 1
    assert result == before
    # Report-only threshold changes recompute rather than reuse the saved .85 assessment.
    lower = j.report([case], [], [result], {}, min_confidence=.8)
    assert lower["check_status_counts"] == lower["ungated_check_status_counts"]


def test_invalid_confidence_cutoff_cannot_call_provider(setup):
    case, _, body, redactor, root = setup
    with pytest.raises(ValueError, match="Confidence cutoff"):
        j.assess(case, body, root, redactor, "run", min_confidence=float("nan"),
                 request=lambda body: pytest.fail("network"))
    assert not list(root.glob("*.json"))


def test_identical_candidate_override_has_explicit_audit_rule(setup):
    case, policy, _, _, _ = setup
    case["candidates"]["B"] = case["candidates"]["A"]
    body = j.build_request(case, policy, j.PROMPT.read_text())
    value = response(body)
    value["answers"]["preference"]["confidence"] = .5
    result = j.assessment(value, body, min_confidence=.85)
    assert result["preference"] == "both"
    assert result["adjustments"][0]["rule"] == "identical_candidates"


@pytest.mark.parametrize("primary", ["pass", "needs_work"])
def test_unused_or_unlocalized_selection_metadata_does_not_invalidate_primary_checks(setup, primary):
    body = setup[2]
    value = response(body, A_useful_coverage=primary, A_useful_coverage_location="none")
    answer = value["answers"]["A_useful_coverage_location"]
    answer["probabilities"] = {key: 0.0 for key in answer["probabilities"]}
    answer["probabilities"].update(none=.25, record_1=.26, whole_candidate=.49)
    before = deepcopy(value)
    j.validate_response(value, body)
    result = j.assessment(value, body)
    assert value == before
    assert result["candidates"]["A"]["checks"]["useful_coverage"]["status"] == (
        "pass" if primary == "pass" else "uncertain")


def test_active_concrete_location_still_requires_maximum_probability(setup):
    body = setup[2]
    value = response(body, A_useful_coverage="needs_work", A_useful_coverage_location="record_1")
    answer = value["answers"]["A_useful_coverage_location"]
    answer["probabilities"] = {key: 0.0 for key in answer["probabilities"]}
    answer["probabilities"].update(none=.25, record_1=.26, whole_candidate=.49)
    with pytest.raises(ValueError, match="maximum-probability"):
        j.validate_response(value, body)
