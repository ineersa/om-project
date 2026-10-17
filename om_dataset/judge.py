"""Blinded DeepSeek judge trial against human reviews and synthetic controls."""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import re
import time

from jsonschema import validate

from . import comparison, teacher
from .collection import canonical, file_hash, now, write_json

MAIN = Path(__file__).resolve().parents[1]
CASES = MAIN / "data/review/cases.jsonl"
VOTES = MAIN / "data/review/votes"
PROMPT = MAIN / "hatfield-om-dataset-spec/om-judge-prompt.md"
CONTROLS = MAIN / "hatfield-om-dataset-spec/fixtures/judge-controls.json"
OUTPUT = MAIN / "data/judge-pilot/v5"
CONFIG = {**teacher.PROVIDERS[teacher.PROVIDER], "provider": teacher.PROVIDER,
          "context_window": 131072, "request_options": {
    "max_tokens": 32768, "thinking": {"type": "enabled"}, "reasoning_effort": "high",
    "response_format": {"type": "json_object"},
}}
TASKS = {"observer", "reflector", "dropper"}
CHOICES = {"A", "B", "both", "neither", "uncertain"}


def obj(properties, required=None):
    return {"type": "object", "properties": properties,
            "required": list(properties) if required is None else required, "additionalProperties": False}


def text(maximum=1200, minimum=1):
    return {"type": "string", "minLength": minimum, "maxLength": maximum}


QUOTE_SCHEMA = obj({"source_id": text(128), "quote": text(500)})
ISSUE_SCHEMA = obj({
    "category": {"enum": ["unsupported", "omission", "status_or_scope", "redundancy_or_verbosity",
                           "unsafe_drop", "protocol", "uncertain_evidence"]},
    "severity": {"enum": ["minor", "major", "critical"]},
    "candidate_excerpt": text(500, 0), "explanation": text(900),
    "evidence": {"type": "array", "items": QUOTE_SCHEMA, "minItems": 1, "maxItems": 3},
})
ASSESSMENT_SCHEMA = obj({
    "verdict": {"enum": ["acceptable", "needs_correction", "uncertain"]},
    "summary": text(), "issues": {"type": "array", "items": ISSUE_SCHEMA, "maxItems": 4},
})
RESPONSE_SCHEMA = obj({
    "preference": {"enum": sorted(CHOICES)}, "reason": text(1600),
    "candidates": obj({label: ASSESSMENT_SCHEMA for label in ("A", "B")}),
    "limitations": {"type": "array", "items": text(400), "maxItems": 5},
})


def digest(value):
    return comparison._hash(value)


def validate_case(case, redactor, *, synthetic=False):
    redactor.require_clean(case)
    if case.get("schema_version") != "om-review-case/1" or case.get("task") not in TASKS:
        raise ValueError("Unsupported review case")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", case.get("case_id", "")):
        raise ValueError("Unsafe case ID")
    if not case.get("redaction_id") or (not synthetic and case["redaction_id"] != redactor.fingerprint):
        raise ValueError("Review case redaction policy changed")
    if set(case.get("candidates", {})) != {"A", "B"} or any(
            not isinstance(value, str) or not value.strip() for value in case["candidates"].values()):
        raise ValueError("Expected two nonempty candidate texts")
    evidence = case.get("evidence", [])
    if not evidence or any(not isinstance(e, dict) or not isinstance(e.get("source_id"), str)
                           or not e["source_id"] or not isinstance(e.get("text"), str) or not e["text"] for e in evidence):
        raise ValueError("Missing evidence blocks")
    if len({e["source_id"] for e in evidence}) != len(evidence):
        raise ValueError("Duplicate evidence IDs")
    for key in ("prior_memory", "comparison_scope"):
        if not isinstance(case.get(key), str) or not case[key]:
            raise ValueError("Missing comparison context")


def load_inputs(redactor):
    raw = CASES.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("Incomplete review pack")
    cases = [json.loads(line) for line in raw.splitlines()]
    if not cases:
        raise ValueError("Empty review pack")
    for case in cases:
        validate_case(case, redactor)
    controls = json.loads(CONTROLS.read_text(encoding="utf-8"))
    for control in controls:
        validate_case(control["case"], redactor, synthetic=True)
        validate(control["expected"], obj({"preference": {"enum": sorted(CHOICES)},
                                           "A": {"enum": ["acceptable", "needs_correction", "uncertain"]},
                                           "B": {"enum": ["acceptable", "needs_correction", "uncertain"]}}))
    all_cases = cases + [control["case"] for control in controls]
    if len({case["case_id"] for case in all_cases}) != len(all_cases):
        raise ValueError("Duplicate case IDs")
    agents = json.loads(comparison.POLICIES.read_text(encoding="utf-8"))["agents"]
    policies = {p["role"]: p for p in agents}
    if not TASKS <= policies.keys():
        raise ValueError("Missing task policies")
    redactor.require_clean(policies)
    return cases, controls, policies


def build_request(case, policy, prompt, *, swapped=False):
    # Explicit projection: votes, expected control answers, and producer mappings
    # are never read here and cannot enter a request through extra case fields.
    public = {key: deepcopy(case[key]) for key in
              ("task", "evidence", "prior_memory", "comparison_scope", "candidates")}
    if swapped:
        public["candidates"] = {"A": public["candidates"]["B"], "B": public["candidates"]["A"]}
    payload = {"worker_policy": {"instructions": policy["instructions"], "tools": policy["tools"]},
               "case": public, "allowed_evidence_ids": [e["source_id"] for e in case["evidence"]],
               "response_contract": RESPONSE_SCHEMA}
    return {"model": CONFIG["model"], "messages": [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ], **deepcopy(CONFIG["request_options"])}


def repair_format(value, case):
    """Remove empty note fields and resolve verifiable nested evidence labels.

    These repairs cannot change a verdict, preference, explanation, or quotation.
    Preserve the original response and record every repair outside the assessment.
    """
    value, repairs = deepcopy(value), []
    if not isinstance(value, dict) or not isinstance(value.get("candidates"), dict):
        return value, repairs
    blocks = {e["source_id"]: e["text"] for e in case["evidence"]}
    for label, candidate in value["candidates"].items():
        if not isinstance(candidate, dict):
            continue
        for field in sorted(set(candidate) - set(ASSESSMENT_SCHEMA["properties"])):
            if candidate[field] in (None, ""):
                del candidate[field]
                repairs.append({"kind": "empty_note", "candidate": label, "field": field})
        for issue in candidate.get("issues", []) if isinstance(candidate.get("issues"), list) else []:
            if not isinstance(issue, dict) or not isinstance(issue.get("evidence"), list):
                continue
            for citation in issue["evidence"]:
                if not isinstance(citation, dict):
                    continue
                alias, quote = citation.get("source_id"), citation.get("quote")
                if not isinstance(alias, str) or not isinstance(quote, str) or not quote or alias in blocks:
                    continue
                matches = set()
                for source_id, content in blocks.items():
                    for pattern in (r"(?m)^\[Source entry id: ([^\]\n]+)\]\n", r"(?m)^\[([^\]\n]+)\] "):
                        markers = list(re.finditer(pattern, content))
                        for index, marker in enumerate(markers):
                            end = markers[index + 1].start() if index + 1 < len(markers) else len(content)
                            if marker[1] == alias and quote in content[marker.end():end]:
                                matches.add(source_id)
                if len(matches) == 1:
                    citation["source_id"] = matches.pop()
                    repairs.append({"kind": "nested_evidence_id", "candidate": label,
                                    "original": alias, "source_id": citation["source_id"]})
    return value, repairs


def validate_assessment(value, case, *, swapped=False):
    validate(value, RESPONSE_SCHEMA)
    evidence = {e["source_id"]: e["text"] for e in case["evidence"]}
    for label, assessment in value["candidates"].items():
        original = ("B" if label == "A" else "A") if swapped else label
        if assessment["verdict"] == "needs_correction" and not assessment["issues"]:
            raise ValueError("A correction verdict requires a supported issue")
        if assessment["verdict"] == "acceptable" and any(i["severity"] != "minor" for i in assessment["issues"]):
            raise ValueError("Acceptable verdict conflicts with material issues")
        for issue in assessment["issues"]:
            excerpt = issue["candidate_excerpt"]
            if excerpt and excerpt not in case["candidates"][original]:
                raise ValueError("Candidate quotation does not match")
            for quote in issue["evidence"]:
                if quote["source_id"] not in evidence or quote["quote"] not in evidence[quote["source_id"]]:
                    raise ValueError("Evidence quotation does not match")
    verdicts = [a["verdict"] for a in value["candidates"].values()]
    if value["preference"] == "both" and verdicts != ["acceptable", "acceptable"]:
        raise ValueError("Both requires two acceptable verdicts")
    if value["preference"] == "neither" and verdicts != ["needs_correction", "needs_correction"]:
        raise ValueError("Neither requires two correction verdicts")
    if case["candidates"]["A"] == case["candidates"]["B"] and value["preference"] in {"A", "B"}:
        raise ValueError("Identical candidates cannot justify a position preference")
    return value


def normalize(value, swapped):
    value = deepcopy(value)
    value["explanation_label_order"] = "swapped" if swapped else "original"
    if swapped:
        if value["preference"] in {"A", "B"}:
            value["preference"] = "B" if value["preference"] == "A" else "A"
        value["candidates"] = {"A": value["candidates"]["B"], "B": value["candidates"]["A"]}
    return value


def assess(case, policy, prompt, swapped, root, redactor, run_id, request=None, reuse=None):
    body = build_request(case, policy, prompt, swapped=swapped)
    redactor.require_clean(body)
    if teacher._context_tokens(body["messages"], []) + CONFIG["request_options"]["max_tokens"] > CONFIG["context_window"]:
        raise ValueError("Judge input exceeds the declared context estimate; do not truncate")
    request_hash = digest(body)
    path = root / f"{case['case_id']}.{'swapped' if swapped else 'original'}.json"
    if path.exists():
        result = json.loads(path.read_text(encoding="utf-8"))
        redactor.require_clean(result)
        if (result["run_id"] != run_id or result["request_sha256"] != request_hash
                or result["case_id"] != case["case_id"] or result["case_sha256"] != digest(case)
                or result["swapped"] is not swapped):
            raise ValueError("Cached judge request changed")
        if result["status"] == "completed":
            validate_assessment(result["assessment"], case, swapped=swapped)
        return result
    result = {"case_id": case["case_id"], "case_sha256": digest(case), "swapped": swapped,
              "run_id": run_id, "request_sha256": request_hash, "recorded_at": now(),
              "status": "failed", "training_ready": False}
    started = time.monotonic()
    stage = "provider"
    inherited = None
    if reuse and (source := reuse[0] / "results" / path.name).exists():
        inherited = json.loads(source.read_text(encoding="utf-8"))
        redactor.require_clean(inherited)
        if (inherited["run_id"] != reuse[1] or inherited["request_sha256"] != request_hash
                or inherited["case_id"] != case["case_id"] or inherited["case_sha256"] != digest(case)
                or inherited["swapped"] is not swapped):
            raise ValueError("Reusable prediction does not match this request")
        result["reused_prediction"] = {"source": str(source), "sha256": file_hash(source), "run_id": reuse[1]}
    try:
        if inherited is not None:
            result["usage"] = inherited.get("usage", {})
            if "raw_content" not in inherited:
                raise ValueError("Prior failed call has no response to revalidate; do not retry automatically")
            response = {"choices": [{"finish_reason": inherited["finish_reason"],
                                      "message": {"content": inherited["raw_content"]}}],
                        "usage": inherited.get("usage", {})}
        else:
            response = (request(body) if request else teacher.complete_request(
                body, "om-judge-" + run_id[:24], CONFIG, timeout=900))
        redactor.require_clean(response)
        stage = "response"
        choice, = response["choices"]
        result.update(usage=response.get("usage", {}), finish_reason=choice["finish_reason"])
        if choice["finish_reason"] != "stop" or choice["message"].get("tool_calls"):
            raise ValueError("Incomplete or tool-calling judge response")
        stage = "parse"
        result["raw_content"] = choice["message"]["content"]
        value, repairs = repair_format(json.loads(choice["message"]["content"]), case)
        stage = "validation"
        validate_assessment(value, case, swapped=swapped)
        result.update(status="completed", assessment=value, format_repairs=repairs)
    except Exception as exc:
        # Exceptions may contain request text or credentials. Do not persist them.
        result["error_type"] = type(exc).__name__
        result["failure_stage"] = stage
    result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    redactor.require_clean(result)
    write_json(path, result)
    return result


def load_votes(cases, votes, redactor):
    values = {}
    for case in cases:
        path = votes / (case["case_id"] + ".json")
        if not path.exists():
            continue
        vote = json.loads(path.read_text(encoding="utf-8"))
        redactor.require_clean(vote)
        if (vote.get("schema_version") != "om-review-vote/1" or vote.get("answer_source") != "human"
                or vote.get("case_id") != case["case_id"] or vote.get("case_sha256") != digest(case)
                or vote.get("choice") not in {"A", "B", "both", "neither", "skip"}):
            raise ValueError("Human vote does not match the current case")
        values[case["case_id"]] = {"choice": vote["choice"], "reason": vote.get("reason", ""),
                                   "vote_sha256": file_hash(path)}
    return values


def report(cases, controls, results, votes):
    by_key = {(r["case_id"], r["swapped"]): r for r in results}
    rows, control_rows = [], []
    for case in cases + [control["case"] for control in controls]:
        normalized = []
        for swapped in (False, True):
            result = by_key.get((case["case_id"], swapped))
            normalized.append(normalize(result["assessment"], swapped)
                              if result and result["status"] == "completed" else None)
        first, second = normalized
        row = {"case_id": case["case_id"], "task": case["task"],
               "original": first, "swapped_normalized": second,
               "format_repairs": {"swapped" if swapped else "original": by_key.get(
                   (case["case_id"], swapped), {}).get("format_repairs", []) for swapped in (False, True)},
               "order_stable": first["preference"] == second["preference"] if first and second else None,
               "verdicts_stable": all(first["candidates"][k]["verdict"] == second["candidates"][k]["verdict"]
                                      for k in ("A", "B")) if first and second else None}
        if case["case_id"] in votes:
            row["human"] = votes[case["case_id"]]
            expected = row["human"]["choice"]
            row["agrees_both_orders"] = (first["preference"] == second["preference"] == expected
                                          if first and second and expected != "skip" else None)
        if case in cases:
            rows.append(row)
        else:
            expected = next(c["expected"] for c in controls if c["case"]["case_id"] == case["case_id"])
            row["expected"] = expected
            row["passes_both_orders"] = (all(a["preference"] == expected["preference"] and all(
                a["candidates"][label]["verdict"] == expected[label] for label in ("A", "B"))
                for a in normalized) if first and second else None)
            control_rows.append(row)
    audited = [r for r in rows if "human" in r and r["human"]["choice"] != "skip"]
    return {"schema_version": "om-judge-calibration/1", "calibrated": False, "training_ready": False,
            "scope": "Development pilot only. Pairwise human preferences do not establish absolute correctness. "
                     "No held-back judge audit has passed. A different judge model does not guarantee independence.",
            "counts": {"review_cases": len(cases), "human_comparisons": len(audited),
                       "agrees_both_orders": sum(r.get("agrees_both_orders") is True for r in rows),
                       "order_stable": sum(r["order_stable"] is True for r in rows),
                       "synthetic_controls": len(controls),
                       "controls_pass_both_orders": sum(r["passes_both_orders"] is True for r in control_rows),
                       "format_repaired_calls": sum(bool(r.get("format_repairs")) for r in results),
                       "reused_predictions": sum("reused_prediction" in r for r in results),
                       "missing_calls": 2 * (len(cases) + len(controls)) - len(results),
                       "failed_calls": sum(r["status"] != "completed" for r in results)},
            "by_task": {task: {"compared": sum(r["task"] == task for r in audited),
                               "agrees_both_orders": sum(r["task"] == task and r.get("agrees_both_orders") is True for r in rows)}
                        for task in sorted(TASKS)}, "cases": rows, "controls": control_rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--max-new-calls", type=int, help="Stop after this many new calls; preserve partial results")
    parser.add_argument("--reuse-predictions-from", type=Path,
                        help="Revalidate recorded replies from identical requests after a parser-only change")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    if args.dry_run and args.report_only:
        parser.error("Choose dry-run or report-only")
    if args.max_new_calls is not None and (args.max_new_calls < 1 or args.report_only):
        parser.error("A positive call limit is available only for prediction runs")
    root = args.output.resolve()
    if not root.is_relative_to((MAIN / "data/judge-pilot").resolve()) or root == (MAIN / "data/judge-pilot").resolve():
        raise ValueError("Use a versioned output directory under data/judge-pilot")
    redactor = comparison._redactor()
    cases, controls, policies = load_inputs(redactor)
    prompt = PROMPT.read_text(encoding="utf-8")
    redactor.require_clean(prompt)
    all_cases = cases + [c["case"] for c in controls]
    for case in all_cases:
        body = build_request(case, policies[case["task"]], prompt)
        redactor.require_clean(body)
        if teacher._context_tokens(body["messages"], []) + CONFIG["request_options"]["max_tokens"] > CONFIG["context_window"]:
            raise ValueError("Oversized judge input")
    pin = {"schema_version": "om-judge-run/1", "config": CONFIG, "prompt_sha256": file_hash(PROMPT),
           "controls_sha256": file_hash(CONTROLS), "cases": {c["case_id"]: digest(c) for c in all_cases},
           "policies_sha256": digest(policies), "redaction_id": redactor.fingerprint,
           "code": {p.name: file_hash(p) for p in (Path(__file__), Path(comparison.__file__),
                                                    Path(teacher.__file__), MAIN / "om_dataset/redaction.py")}}
    reuse = None
    if args.reuse_predictions_from:
        source = args.reuse_predictions_from.resolve()
        if source == root or not source.is_relative_to((MAIN / "data/judge-pilot").resolve()):
            raise ValueError("Reuse a different judge-pilot run directory")
        previous = json.loads((source / "run.json").read_text(encoding="utf-8"))
        redactor.require_clean(previous)
        if ({k: v for k, v in previous["pin"].items() if k != "code"}
                != {k: v for k, v in pin.items() if k != "code"}):
            raise ValueError("Only parser code may change when reusing predictions")
        reuse = source, digest(previous)
    if args.dry_run:
        print(json.dumps({"review_cases": len(cases), "synthetic_controls": len(controls),
                          "planned_calls": 2 * len(all_cases), "provider_calls": 0,
                          "destination": CONFIG["base_url"], "output": str(root)}))
        return 0
    comparison._private_dir(root)
    with (root / ".lock").open("a+b") as lock:
        os.fchmod(lock.fileno(), 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = root / "run.json"
        saved = {"pin": pin}
        if manifest.exists():
            saved = json.loads(manifest.read_text(encoding="utf-8"))
            if saved["pin"] != pin:
                raise ValueError("Pinned judge inputs changed; use a new output directory")
        elif args.report_only:
            raise ValueError("Missing judge run manifest")
        else:
            write_json(manifest, saved)
        if not args.report_only and not os.environ.get(CONFIG["api_key_env"]):
            raise ValueError("Configured judge provider credential is unavailable")
        run_id = digest(saved)
        results = []
        new_calls = 0
        result_dir = root / "results"
        comparison._private_dir(result_dir)
        for case in all_cases:
            for swapped in (False, True):
                path = result_dir / f"{case['case_id']}.{'swapped' if swapped else 'original'}.json"
                if args.report_only:
                    def no_network(body):
                        raise ValueError("Missing prediction in report-only mode")
                    # Do not create missing prediction records in report-only mode.
                    if not path.exists():
                        raise ValueError("Incomplete run; report-only requires all prediction files")
                    result = assess(case, policies[case["task"]], prompt, swapped, result_dir,
                                    redactor, run_id, request=no_network)
                else:
                    if not path.exists() and not (reuse and (reuse[0] / "results" / path.name).exists()):
                        if args.max_new_calls is not None and new_calls >= args.max_new_calls:
                            continue
                        new_calls += 1
                    result = assess(case, policies[case["task"]], prompt, swapped, result_dir, redactor, run_id, reuse=reuse)
                    print(json.dumps({"case_id": case["case_id"], "swapped": swapped, "status": result["status"]}), flush=True)
                results.append(result)
        # Votes are loaded after the inference phase and never enter a request.
        votes = load_votes(cases, VOTES, redactor)
        summary = report(cases, controls, results, votes)
        redactor.require_clean(summary)
        target = root / "report.json"
        if target.exists() and json.loads(target.read_text(encoding="utf-8")) == summary:
            pass
        else:
            write_json(target, summary)
        print(json.dumps({"report": str(target), **summary["counts"], "calibrated": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
