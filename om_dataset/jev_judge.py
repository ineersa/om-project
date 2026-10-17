"""Typed Jev memory judgments against blinded reviews and synthetic controls."""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
from http.client import HTTPSConnection
import json
import math
import os
from pathlib import Path
import re
import time
from urllib.error import HTTPError

from . import comparison, judge, teacher
from .collection import file_hash, now, write_json

MAIN = judge.MAIN
OUTPUT = MAIN / "data/jev-judge/v4"
RUNS_ROOT = OUTPUT.parent
PROMPT = MAIN / "hatfield-om-dataset-spec/om-jev-quality-rubric.md"
EXTRA_CONTROLS = MAIN / "hatfield-om-dataset-spec/fixtures/jev-quality-controls.json"
ROLE_CONTROLS = MAIN / "hatfield-om-dataset-spec/fixtures/jev-reflector-dropper-controls.json"
MIN_CONFIDENCE = 0.85
MODEL = "jev-1.13.0"
HOST = "api.typesafe.ai"
ENDPOINT = "/v1/systemone"
CHECK_HEADINGS = {
    "useful_coverage": "Useful coverage",
    "context_recoverability": "Context recoverability",
    "importance_assignment": "Importance assignment",
    "compression": "Compression",
    "safe_removal": "Safe removal",
}
PREFERENCES = {
    "A": "A is meaningfully preferable. Substantially less text and fewer records can outweigh a small loss of secondary detail, not essential facts or correctness.",
    "B": "B is meaningfully preferable. Substantially less text and fewer records can outweigh a small loss of secondary detail, not essential facts or correctness.",
    "both": "Both are acceptable and there is no meaningful preference. Do not choose this merely because both are factually supported.",
    "neither": "Both need correction and neither is meaningfully preferable.",
    "uncertain": "The available evidence does not support a reliable comparison.",
}


def rubric(prompt, task):
    sections = dict(re.findall(r"(?ms)^## ([^\n]+)\n(.*?)(?=^## |\Z)", prompt))
    return {"scope": sections["Scope"].strip(), "role": sections[task.title()].strip(),
            **{name: sections[CHECK_HEADINGS[name]].strip() for name in checks_for(task)}}


def candidate_options(text, label):
    records = list(re.finditer(r"(?m)^(\d+)\.\s", text))
    if not records:
        return {"whole_candidate": f"The memory content or action in candidate {label}."}
    options = {}
    for i, match in enumerate(records):
        if f"record_{match[1]}" in options:
            return {"whole_candidate": f"Candidate {label}; duplicate numbering prevents an unambiguous individual record ID."}
        excerpt = text[match.end():records[i + 1].start() if i + 1 < len(records) else len(text)]
        preview = " ".join(excerpt.splitlines()[:1])[:120]
        options[f"record_{match[1]}"] = f"Candidate {label}, record {match[1]}: {preview}"
    return options


def evidence_options(evidence):
    options = {}
    for block in evidence:
        entries = list(re.finditer(r"(?m)^\[(\d+)\] (\d{4}-\d\d-\d\d \d\d:\d\d) ", block["text"]))
        if entries:
            for entry in entries:
                key = f"{block['source_id']}:observation:{entry[1]}"
                options[key] = f"Source `{block['source_id']}`, observation [{entry[1]}], {entry[2]}."
        else:
            options[block["source_id"]] = f"Source block `{block['source_id']}`."
    return options


def checks_for(task):
    if task == "dropper":
        return ("safe_removal",)
    checks = ("useful_coverage", "context_recoverability", "compression")
    return checks[:2] + ("importance_assignment",) + checks[2:] if task == "observer" else checks


def load_inputs(redactor):
    cases, controls, policies = judge.load_inputs(redactor)
    for control in controls:
        task = control["case"]["task"]
        target = ("compression" if control["case"]["case_id"] in {
            "control-observer-compression", "control-reflector-compression"} else
            "safe_removal" if task == "dropper" else "useful_coverage")
        control["expected_checks"] = {label: {target: "needs_work" if (
            control["expected"][label] == "needs_correction" or target == "compression" and label == "B")
            else "pass"} for label in ("A", "B")}
    controls += json.loads(EXTRA_CONTROLS.read_text(encoding="utf-8"))
    controls += json.loads(ROLE_CONTROLS.read_text(encoding="utf-8"))
    for control in controls:
        judge.validate_case(control["case"], redactor, synthetic=True)
        if set(control["expected"]) != {"A", "B", "preference"} or any(
                control["expected"][label] not in {"acceptable", "needs_correction", "uncertain"}
                for label in ("A", "B")) or control["expected"]["preference"] not in PREFERENCES:
            raise ValueError("Invalid control expectations")
        if set(control["expected_checks"]) != {"A", "B"} or any(
                check not in checks_for(control["case"]["task"]) or status not in {"pass", "needs_work", "uncertain"}
                for checks in control["expected_checks"].values() for check, status in checks.items()):
            raise ValueError("Invalid quality-check expectations")
    all_cases = cases + [c["case"] for c in controls]
    if len({case["case_id"] for case in all_cases}) != len(all_cases):
        raise ValueError("Duplicate case IDs")
    return cases, controls, policies


def build_request(case, policy, prompt, *, swapped=False):
    public = {key: deepcopy(case[key]) for key in
              ("task", "evidence", "prior_memory", "comparison_scope", "candidates")}
    if swapped:
        public["candidates"] = {"A": public["candidates"]["B"], "B": public["candidates"]["A"]}
    state = {
        "case": public,
        # Match the worker's actual model-facing schema, not snapshot metadata
        # such as PHP handler source and runtime bindings.
        "worker_policy": {"instructions": policy["instructions"], "tools": [teacher._tool(policy)]},
        "rubric": rubric(prompt, case["task"]),
        "evaluation_rules": (
            "Apply the original worker contract and the applicable quality checks. "
            "Questions are independent and cannot see one another's answers. "
            "Check-status questions establish pass, needs_work, or uncertain. Location questions "
            "are speculative: their answers are used ONLY when that check needs_work. "
            "Do not invent a problem just to select a location. Prefer compact memory after "
            "protecting accurate essentials and safe removal. Never average a factual error "
            "or unsafe removal away with good compression. Both candidates can be factually "
            "adequate while the shorter candidate is meaningfully preferable."
        ),
        "display_metrics": {label: {"characters": len(text),
                                    "numbered_records": len(re.findall(r"(?m)^\d+\.\s", text))}
                            for label, text in public["candidates"].items()},
    }
    questions = {}
    for label in ("A", "B"):
        for check in checks_for(case["task"]):
            key = f"{label}_{check}"
            questions[key] = {
                "type": "choice",
                "instructions": (
                    f"Assess {CHECK_HEADINGS[check].lower()} for candidate `case.candidates.{label}` "
                    f"using `rubric.{check}`, `rubric.role`, `rubric.scope`, and `worker_policy`. "
                    "Judge the entire set against the source and existing memory within the shown scope. "
                    "Assess only this check; a problem in another check does not make this one fail."
                ),
                "criteria": {
                    "pass": "Meets this check. Harmless paraphrases and minor stylistic alternatives are not problems.",
                    "needs_work": "A concrete source-grounded problem fails this check under its rubric.",
                    "uncertain": "Necessary evidence or metadata is unavailable or genuinely ambiguous; cannot judge this check.",
                },
            }
            if case["task"] == "dropper":
                questions[key]["criteria"].update({
                    "pass": "Every useful meaning removed by the proposal remains in a kept observation or existing reflection, or is genuinely obsolete.",
                    "needs_work": "A specific useful current fact disappears from ALL remaining memory. Check existing reflections and kept observations first; priority and partial-coverage tags are not proof of loss.",
                })
            elif case["task"] == "reflector":
                questions[key]["criteria"]["needs_work"] = {
                    "useful_coverage": "A proposed reflection invents or contradicts a fact, states a superseded design as current, or causes a concrete essential loss. Merely leaving an observation unreflected does not establish loss.",
                    "context_recoverability": "A needed identifier or source reference is missing or wrong, preventing reconstruction. An otherwise identifiable but stale claim is a coverage problem, not this problem.",
                    "compression": "Substantially less text or fewer records would retain the useful meaning. Identify avoidable wording or repetition, not a factual error.",
                }[check]
            options = candidate_options(public["candidates"][label], label)
            options.setdefault("whole_candidate", f"Candidate {label} as a set; the problem spans multiple records.")
            if check in {"useful_coverage", "context_recoverability"}:
                options.update(evidence_options(public["evidence"]))
            if len(options) > 253:
                raise ValueError("Localize fewer source facts or candidate records per case; Choice allows 255 options")
            questions[f"{key}_location"] = {
                "type": "choice", "instructions": (
                    f"ONLY IF candidate `case.candidates.{label}` needs work on {CHECK_HEADINGS[check].lower()} "
                    f"under `rubric.{check}`, locate one concrete problem. Select a defective candidate "
                    "record, or the source fact whose essential meaning or identifier is missing. "
                    "For safe removal, select the proposed drop whose meaning is NOT preserved in kept "
                    "observations or existing reflections. Prefer a specific location; whole_candidate "
                    "is for a problem spanning records. Select none if the check passes. "
                    "This answer does not establish failure; it is unused unless the status is needs_work."
                ),
                "criteria": {"none": "No concrete problem for this check.",
                             "uncertain": "Cannot locate a specific source-grounded problem for this check.", **options},
            }
    questions["preference"] = {
        "type": "choice",
        "instructions": "Compare `case.candidates.A` and `case.candidates.B` under the applicable `rubric` and `evaluation_rules`. Strongly prefer less text and fewer records when useful meaning survives. Never trade factual correctness or safe removal for compression.",
        "criteria": deepcopy(PREFERENCES),
    }
    return {"model": MODEL, "state": state, "questions": questions}


def context_estimate(body, long_context=False):
    # This is the project's character-based estimate, not Jev's tokenizer.
    per_question = max(teacher._tokens({"state": body["state"], "question": q})
                       for q in body["questions"].values())
    total = teacher._tokens(body)
    question_limit, request_limit = (262144, 262144) if long_context else (32768, 65536)
    return {"per_question": per_question, "per_request": total,
            "exceeds_estimate": per_question > question_limit or total > request_limit}


def complete_request(body, *, long_context=False):
    key = os.environ.get("JEV_API_KEY") or os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise ValueError("Set JEV_API_KEY or TYPESAFE_API_KEY before a live run")
    headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
    if long_context:
        headers["TypeSafe-Danger-Zone-Long-Context"] = "true"
    connection = HTTPSConnection(HOST, timeout=120)
    try:
        connection.request("POST", ENDPOINT, body=json.dumps(body, allow_nan=False).encode(), headers=headers)
        response = connection.getresponse()
        if response.status != 200:
            raise HTTPError(ENDPOINT, response.status, "TypeSafe request failed", None, None)
        return json.load(response)
    finally:
        connection.close()


def probability(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def validate_response(response, body):
    if response.get("model") != body["model"]:
        raise ValueError("Unexpected Jev model version")
    answers = response.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(body["questions"]):
        raise ValueError("Missing or unexpected typed answers")
    for key, question in body["questions"].items():
        answer = answers[key]
        if not isinstance(answer, dict) or answer.get("type") != "choice":
            raise ValueError("Expected a Choice answer")
        choices = question["criteria"]
        probabilities = answer.get("probabilities")
        if (not isinstance(probabilities, dict) or set(probabilities) != set(choices)
                or not all(probability(p) for p in probabilities.values())
                # The live API rounds probabilities to two decimal places.
                or not math.isclose(sum(probabilities.values()), 1, abs_tol=0.005 * len(choices) + 1e-9)
                or answer.get("choice") not in choices or not probability(answer.get("confidence"))):
            raise ValueError("Invalid Choice distribution")
        # An unused speculative location cannot affect the decision. Keep its
        # returned metadata unchanged, but do not let a non-maximal selection
        # invalidate all primary answers. Active, concrete selections remain strict.
        unused_location = key.endswith("_location") and (
            answers[key.removesuffix("_location")].get("choice") != "needs_work"
            or answer["choice"] in {"none", "uncertain"})
        if not unused_location and probabilities[answer["choice"]] < max(probabilities.values()) - 1e-5:
            raise ValueError("Choice does not select a maximum-probability option")
    usage = response.get("usage", {})
    if not isinstance(usage, dict) or any(usage.get(k) is not None and
            (type(usage[k]) is not int or usage[k] < 0) for k in ("input_tokens", "output_tokens")):
        raise ValueError("Invalid token usage")
    return response


def assessment(response, body, *, min_confidence=None):
    if min_confidence is not None and not probability(min_confidence):
        raise ValueError("Confidence cutoff must be between zero and one")
    answers = response["answers"]
    candidates, adjustments = {}, []
    for label in ("A", "B"):
        checks = {}
        for check in checks_for(body["state"]["case"]["task"]):
            key = f"{label}_{check}"
            model_status = answers[key]["choice"]
            status = ("uncertain" if min_confidence is not None and answers[key]["confidence"] < min_confidence
                      else model_status)
            item = {"status": status, "location_id": None, "location_kind": None}
            if min_confidence is not None:
                item["confidence"] = answers[key]["confidence"]
                if status != model_status:
                    item.update(model_status=model_status, review_reason="confidence_below_cutoff")
            if status == "needs_work":
                location = answers[f"{key}_location"]["choice"]
                if location in {"none", "uncertain"}:
                    item.update(status="uncertain", model_status=status)
                else:
                    item.update(location_id=location, location_kind=(
                        "candidate_set" if location == "whole_candidate" else
                        "candidate_record" if location.startswith("record_") else "source_fact"))
                if min_confidence is not None:
                    item["location_confidence"] = answers[f"{key}_location"]["confidence"]
            checks[check] = item
        def status_of(items):
            values = {item["status"] for item in items}
            return "needs_work" if "needs_work" in values else "uncertain" if "uncertain" in values else "pass"
        # Keep information adequacy separate from compression for the existing
        # control expectations: a longer candidate can still be factually adequate.
        core = status_of(item for check, item in checks.items() if check != "compression")
        candidates[label] = {
            "checks": checks, "review_status": status_of(checks.values()),
            "verdict": {"pass": "acceptable", "needs_work": "needs_correction", "uncertain": "uncertain"}[core],
            "verdict_source": "quality_checks_excluding_compression",
        }
    preference = raw_preference = answers["preference"]["choice"]
    if min_confidence is not None and answers["preference"]["confidence"] < min_confidence:
        preference = "uncertain"
    acceptable = [label for label in ("A", "B") if candidates[label]["verdict"] == "acceptable"]
    if len(acceptable) == 1 and candidates["B" if acceptable[0] == "A" else "A"]["verdict"] == "needs_correction":
        if preference != "uncertain":
            preference = acceptable[0]
    elif (preference == "both" and len(acceptable) != 2) or (preference == "neither" and any(
            c["verdict"] != "needs_correction" for c in candidates.values())):
        preference = "uncertain"
    if preference in {"A", "B"}:
        other = "B" if preference == "A" else "A"
        if (candidates[preference]["verdict"] == "uncertain" or
                (candidates[preference]["verdict"] == "needs_correction" and candidates[other]["verdict"] == "uncertain")):
            preference = "uncertain"
    if body["state"]["case"]["candidates"]["A"] == body["state"]["case"]["candidates"]["B"]:
        verdicts = {c["verdict"] for c in candidates.values()}
        preference = "both" if verdicts == {"acceptable"} else "neither" if verdicts == {"needs_correction"} else "uncertain"
    if preference != raw_preference:
        rule = ("preference_confidence_below_cutoff" if min_confidence is not None
                and answers["preference"]["confidence"] < min_confidence and preference == "uncertain"
                else "verdict_preference_consistency")
        if min_confidence is not None and body["state"]["case"]["candidates"]["A"] == body["state"]["case"]["candidates"]["B"]:
            rule = "identical_candidates"
        adjustments.append({"from": raw_preference, "to": preference, "rule": rule})
    return {"preference": preference, "raw_preference": raw_preference, "candidates": candidates,
            "adjustments": adjustments, "rationale_available": False}


def assess(case, body, root, redactor, run_id, *, swapped=False, long_context=False,
           min_confidence=MIN_CONFIDENCE, request=None, read_only=False):
    if not probability(min_confidence):
        raise ValueError("Confidence cutoff must be between zero and one")
    redactor.require_clean(body)
    path = root / f"{case['case_id']}.{'swapped' if swapped else 'original'}.json"
    request_hash = judge.digest(body)
    if path.exists():
        result = json.loads(path.read_text(encoding="utf-8"))
        redactor.require_clean(result)
        if (result["run_id"] != run_id or result["request_sha256"] != request_hash
                or result["case_sha256"] != judge.digest(case) or result["swapped"] is not swapped
                or result["request"] != body or result["long_context"] is not long_context
                or result["min_confidence"] != min_confidence):
            raise ValueError("Pinned Jev request changed; use a new output directory")
        if result["status"] == "completed":
            validate_response(result["response"], body)
            if assessment(result["response"], body) != result["assessment"]:
                raise ValueError("Cached assessment does not match typed answers")
            if assessment(result["response"], body, min_confidence=min_confidence) != result["confidence_assessment"]:
                raise ValueError("Cached confidence assessment does not match typed answers")
        elif result["status"] == "in_progress":
            result.update(status="failed", failure_stage="interrupted", error_type="InterruptedRequest")
            if not read_only:
                write_json(path, result)
        elif result["status"] != "failed":
            raise ValueError("Unknown cached prediction status")
        return result
    if read_only:
        raise ValueError("Missing prediction in report-only mode")
    result = {"case_id": case["case_id"], "case_sha256": judge.digest(case), "swapped": swapped,
              "run_id": run_id, "request_sha256": request_hash, "recorded_at": now(),
              "status": "in_progress", "training_ready": False, "long_context": long_context,
              "min_confidence": min_confidence, "request": body}
    redactor.require_clean(result)
    write_json(path, result)
    started, stage = time.monotonic(), "provider"
    try:
        response = request(body) if request else complete_request(body, long_context=long_context)
        redactor.require_clean(response)
        result["response"] = response
        stage = "validation"
        validate_response(response, body)
        result.update(status="completed", assessment=assessment(response, body),
                      confidence_assessment=assessment(response, body, min_confidence=min_confidence),
                      usage=response.get("usage", {}))
    except Exception as error:
        result.update(status="failed", failure_stage=stage, error_type=type(error).__name__)
        if isinstance(error, HTTPError):
            result["http_status"] = error.code
    result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    redactor.require_clean(result)
    write_json(path, result)
    return result


def report(cases, controls, results, votes, *, min_confidence=None):
    raw_results = results
    if min_confidence is not None:
        results = [{**r, "assessment": assessment(r["response"], r["request"], min_confidence=min_confidence)}
                   if r["status"] == "completed" else r
                   for r in results]
    summary = judge.report(cases, controls, results, votes)
    summary["schema_version"] = "om-jev-judge-calibration/4"
    summary["min_confidence"] = min_confidence
    summary["rationale_available"] = False
    for row in summary["cases"] + summary["controls"]:
        normalized = row["swapped_normalized"]
        if normalized:
            flip = lambda value: "B" if value == "A" else "A" if value == "B" else value
            normalized["raw_preference"] = flip(normalized["raw_preference"])
            for adjustment in normalized["adjustments"]:
                for key in ("candidate", "from", "to"):
                    if key in adjustment:
                        adjustment[key] = flip(adjustment[key])
        original = row["original"]
        for name, keys in (("check_statuses_stable", ("status",)),
                           ("check_locations_stable", ("status", "location_id"))):
            def signature(value):
                return {label: sorted((check,) + tuple(str(item[k]) for k in keys)
                                      for check, item in candidate["checks"].items())
                        for label, candidate in value["candidates"].items()}
            row[name] = signature(original) == signature(normalized) if original and normalized else None
        if row in summary["controls"]:
            expected = next(c["expected_checks"] for c in controls if c["case"]["case_id"] == row["case_id"])
            row["expected_checks"] = expected
            row["target_checks_pass_both_orders"] = (all(
                value["candidates"][label]["checks"][check]["status"] == status
                for value in (original, normalized) for label, checks in expected.items()
                for check, status in checks.items()) if original and normalized else None)
    paired_cases = [row for row in summary["cases"] if row["original"] and row["swapped_normalized"]]
    paired_controls = [row for row in summary["controls"] if row["original"] and row["swapped_normalized"]]
    summary["verdict_source"] = "quality_checks_excluding_compression; no broad model verdict is requested"
    summary["review_status_source"] = "all_applicable_quality_checks_including_compression"
    summary["check_status_counts"] = {status: sum(
        item["status"] == status for r in results if r["status"] == "completed"
        for candidate in r["assessment"]["candidates"].values() for item in candidate["checks"].values())
        for status in ("pass", "needs_work", "uncertain")}
    summary["raw_preference_counts"] = {
        "paired_review_cases": len(paired_cases), "paired_controls": len(paired_controls),
        "order_stable": sum(row["original"]["raw_preference"] == row["swapped_normalized"]["raw_preference"] for row in paired_cases),
        "agrees_both_orders": sum(all(row[key]["raw_preference"] == (row.get("human") or {}).get("choice")
                                     for key in ("original", "swapped_normalized")) for row in paired_cases),
        "controls_preferences_pass_both_orders": sum(all(row[key]["raw_preference"] == row["expected"]["preference"]
                                                          for key in ("original", "swapped_normalized")) for row in paired_controls),
    }
    summary["counts"].update(
        paired_valid_review_cases=sum(r["original"] is not None and r["swapped_normalized"] is not None for r in summary["cases"]),
        paired_valid_controls=sum(r["original"] is not None and r["swapped_normalized"] is not None for r in summary["controls"]),
        adjusted_calls=sum(bool(r.get("assessment", {}).get("adjustments")) for r in results),
        controls_target_checks_pass_both_orders=sum(r["target_checks_pass_both_orders"] is True for r in summary["controls"]),
    )
    ungated = report(cases, controls, raw_results, votes) if min_confidence is not None else summary
    summary["ungated_counts"] = deepcopy(ungated["counts"])
    summary["ungated_check_status_counts"] = deepcopy(ungated["check_status_counts"])
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--controls-only", action="store_true")
    parser.add_argument("--tasks", nargs="+", choices=sorted(judge.TASKS), help="Run only these worker-output judging tasks")
    parser.add_argument("--min-confidence", type=float, default=MIN_CONFIDENCE,
                        help="Abstain below this Choice confidence on primary checks and pairwise preference")
    parser.add_argument("--long-context", action="store_true", help="Opt in to TypeSafe's slower, less accurate 262k mode")
    parser.add_argument("--max-new-calls", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if (args.dry_run and args.report_only) or (args.max_new_calls is not None and (args.max_new_calls < 1 or args.report_only)):
        parser.error("Choose dry-run or report-only; a positive call limit applies only to prediction runs")
    if not probability(args.min_confidence):
        parser.error("Confidence cutoff must be between zero and one")
    root = (args.output or (OUTPUT.parent / f"controls-{OUTPUT.name}" if args.controls_only else OUTPUT)).resolve()
    base = RUNS_ROOT.resolve()
    if not root.is_relative_to(base) or root == base:
        raise ValueError("Use a versioned output directory under data/jev-judge")
    redactor = comparison._redactor()
    cases, controls, policies = load_inputs(redactor)
    if args.tasks:
        cases = [case for case in cases if case["task"] in args.tasks]
        controls = [control for control in controls if control["case"]["task"] in args.tasks]
    if args.controls_only:
        cases = []
    prompt = PROMPT.read_text(encoding="utf-8")
    work = [(case, swapped, build_request(case, policies[case["task"]], prompt, swapped=swapped))
            for case in cases + [c["case"] for c in controls] for swapped in (False, True)]
    estimates = {case["case_id"]: context_estimate(body, args.long_context) for case, _, body in work}
    for _, _, body in work:
        redactor.require_clean(body)
    if args.dry_run:
        print(json.dumps({"model": MODEL, "review_cases": len(cases), "synthetic_controls": len(controls),
                          "planned_calls": len(work), "provider_calls": 0, "output": str(root),
                          "tasks": sorted(set(args.tasks or judge.TASKS)),
                          "long_context": args.long_context, "min_confidence": args.min_confidence,
                          "context_estimates": estimates}))
        return 0
    if not args.report_only and any(e["exceeds_estimate"] for e in estimates.values()):
        print(json.dumps({"warning": "Character-based context estimates exceed the selected limits. The service validates actual token counts; evidence is never truncated or retried in long-context mode automatically.",
                          "cases": [key for key, value in estimates.items() if value["exceeds_estimate"]]}), flush=True)
    if not args.report_only and not (os.environ.get("JEV_API_KEY") or os.environ.get("TYPESAFE_API_KEY")):
        raise ValueError("Set JEV_API_KEY or TYPESAFE_API_KEY before a live run")
    pin = {"schema_version": "om-jev-judge-run/4", "model": MODEL, "endpoint": f"https://{HOST}{ENDPOINT}",
           "min_confidence": args.min_confidence, "tasks": sorted(set(args.tasks or judge.TASKS)),
           "long_context": args.long_context, "prompt_sha256": file_hash(PROMPT),
           "controls_sha256": file_hash(judge.CONTROLS), "extra_controls_sha256": file_hash(EXTRA_CONTROLS),
           "role_controls_sha256": file_hash(ROLE_CONTROLS),
           "redaction_id": redactor.fingerprint,
           "requests": {f"{case['case_id']}:{swapped}": judge.digest(body) for case, swapped, body in work},
           "code": {p.name: file_hash(p) for p in (Path(__file__), Path(judge.__file__), Path(comparison.__file__),
                                                    Path(teacher.__file__), MAIN / "om_dataset/redaction.py")}}
    comparison._private_dir(root)
    with os.fdopen(os.open(root / ".lock", os.O_RDWR | os.O_CREAT, 0o600), "a+b") as lock:
        os.fchmod(lock.fileno(), 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = root / "run.json"
        saved = {"pin": pin}
        if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != saved:
            raise ValueError("Pinned Jev run changed; use a new output directory")
        if not manifest.exists():
            if args.report_only:
                raise ValueError("Missing Jev run manifest")
            write_json(manifest, saved)
        run_id, results, new_calls = judge.digest(saved), [], 0
        result_dir = root / "results"
        comparison._private_dir(result_dir)
        for case, swapped, body in work:
            path = result_dir / f"{case['case_id']}.{'swapped' if swapped else 'original'}.json"
            if not path.exists():
                if args.report_only:
                    raise ValueError("Incomplete run; report-only cannot create predictions")
                if args.max_new_calls is not None and new_calls >= args.max_new_calls:
                    continue
                new_calls += 1
            result = assess(case, body, result_dir, redactor, run_id, swapped=swapped,
                            long_context=args.long_context, min_confidence=args.min_confidence,
                            read_only=args.report_only)
            results.append(result)
            print(json.dumps({"case_id": case["case_id"], "swapped": swapped, "status": result["status"]}), flush=True)
            if result.get("failure_stage") == "provider":
                break  # Stop on service errors rather than spending the remaining request budget.
        votes = judge.load_votes(cases, judge.VOTES, redactor)
        summary = report(cases, controls, results, votes, min_confidence=args.min_confidence)
        redactor.require_clean(summary)
        write_json(root / "report.json", summary)
        print(json.dumps({"report": str(root / 'report.json'), **summary["counts"], "calibrated": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
