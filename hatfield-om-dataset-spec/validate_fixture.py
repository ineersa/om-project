#!/usr/bin/env python3
"""Validate the synthetic OM-DATA-01 fixture, not a production dataset.

Checks JSON schemas, relative artifact paths/hashes, identity hashes, source refs,
calendar timestamps, and basic call/receipt/state consistency. No model calls,
Hatfield execution, semantic judging, or writes are performed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

try:
    from jsonschema import Draft202012Validator, FormatChecker, ValidationError
except ImportError:
    raise SystemExit("Missing dependency: install jsonschema, then rerun this script.")


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def references(value: Any) -> Iterator[dict[str, str]]:
    if isinstance(value, dict):
        if set(value) == {"path", "sha256"}:
            yield value
        else:
            for child in value.values():
                yield from references(child)
    elif isinstance(value, list):
        for child in value:
            yield from references(child)


def resolve_artifact(root: Path, ref: dict[str, str]) -> Path:
    require(not Path(ref["path"]).is_absolute(), "Artifact path must be package-relative.")
    path = (root / ref["path"]).resolve(strict=True)
    require(path.is_relative_to(root), "Artifact reference escapes the package root.")
    require(path.is_file(), "Artifact reference is not a regular file.")
    require(hashlib.sha256(path.read_bytes()).hexdigest() == ref["sha256"],
            f"Artifact hash mismatch: {ref['path']}")
    return path


def check_observer_trace(record: dict[str, Any], argument_schema: dict[str, Any]) -> None:
    """Basic synthetic positive-trace checks; not a replacement for PHP handlers."""
    require(record["task"] == "observer", "Only the synthetic Observer fixture is supported.")
    allowed = {(r["run_id"], r["seq"]) for r in record["input"]["allowed_source_refs"]}
    messages = record["target"]["messages"]
    require(messages[-1]["role"] == "assistant" and not messages[-1].get("tool_calls"),
            "Trace must finish with an assistant stop, not an unanswered tool call.")
    validator = Draft202012Validator(argument_schema)
    calls: dict[str, dict[str, Any]] = {}
    receipts: set[str] = set()
    emitted: list[dict[str, Any]] = []
    for message in messages:
        for call in message.get("tool_calls", []):
            require(call["id"] not in calls, "Duplicate tool call ID.")
            require(call["function"]["name"] == "record_observations", "Wrong tool for Observer.")
            validator.validate(call["function"]["arguments"])
            calls[call["id"]] = call
            for observation in call["function"]["arguments"]["observations"]:
                require(bool(observation["content"].strip()) and "\n" not in observation["content"],
                        "Observation content must be nonempty and single-line.")
                datetime.strptime(observation["timestamp"], "%Y-%m-%d %H:%M")
                for ref in observation["source_refs"]:
                    require((ref["run_id"], ref["seq"]) in allowed, "Unknown source reference.")
                emitted.append(observation)
        if message["role"] == "tool":
            call_id = message["tool_call_id"]
            require(call_id in calls and call_id not in receipts, "Missing or repeated tool-call receipt.")
            require(message["name"] == calls[call_id]["function"]["name"], "Receipt tool-name mismatch.")
            receipt = json.loads(message["content"])
            require(receipt["status"] == "accepted" and receipt["rejected"] == 0,
                    "This positive fixture expects a fully accepted receipt.")
            receipts.add(call_id)
    require(set(calls) == receipts, "Not all calls have receipts.")
    require(len(calls) <= record["limits"]["max_tool_calls"], "Too many tool calls.")
    accepted = record["target"]["accepted_delta"]["observations"]
    require([{k: v for k, v in item.items() if k != "observation_id"} for item in accepted] == emitted,
            "Fixture accepted delta differs from emitted observations.")
    # Fixture has a single successful batch; do not generalize this to runtime deduplication.
    require(len(calls) == 1, "This fixture expects one batch.")
    receipt = json.loads(next(m["content"] for m in messages if m["role"] == "tool"))
    require(receipt["added"] == receipt["total"] == len(accepted) and receipt["duplicates"] == 0,
            "Fixture receipt counters do not match accepted data.")


def validate(root: Path) -> None:
    root = root.resolve(strict=True)
    schema = read_json(root / "example.schema.json")
    Draft202012Validator.check_schema(schema)
    record = read_json(root / "fixtures" / "observer-example.json")
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(record)
    require(record["record_kind"] == "fixture" and record["quality"]["status"] == "fixture",
            "Synthetic record must remain explicitly tagged as a fixture.")
    require(record["split"] == "unassigned", "Fixture must not enter a production split.")
    require(record["example_id"] == digest({k: v for k, v in record.items() if k != "example_id"}),
            "Example identity hash mismatch.")
    require(record["quality"]["checked_input_hash"] == digest(record["input"]),
            "Checked-input hash mismatch.")
    refs = list(references(record))
    for ref in refs:
        resolve_artifact(root, ref)
    prompt = resolve_artifact(root, record["policy"]["prompt_ref"]).read_text(encoding="utf-8")
    require(record["input"]["messages"][0]["content"] == prompt, "Fixture prompt differs from its blob.")
    tools = read_json(resolve_artifact(root, record["policy"]["tools_ref"]))
    require(record["input"]["tools"] == tools, "Fixture tools differ from their blob.")
    before = read_json(resolve_artifact(root, record["input"]["state_before_ref"]))
    after = read_json(resolve_artifact(root, record["target"]["state_after_ref"]))
    require(before["observations"] == [] and before["reflections"] == [], "Expected empty fixture seed.")
    require(after["observations"] == record["target"]["accepted_delta"]["observations"],
            "Fixture state transition mismatch.")
    argument_schemas = {task: read_json(root / "schemas" / f"{task}.arguments.schema.json")
                        for task in ("observer", "reflector", "dropper")}
    for arguments in argument_schemas.values():
        Draft202012Validator.check_schema(arguments)
    check_observer_trace(record, argument_schemas["observer"])
    Draft202012Validator(argument_schemas["observer"]).validate({"observations": []})
    for task, empty in (("reflector", {"reflections": []}), ("dropper", {"ids": []})):
        require(not Draft202012Validator(argument_schemas[task]).is_valid(empty),
                f"{task} should use no call, not an empty-array call.")
    broken = copy.deepcopy(record)
    broken["target"]["messages"][0]["tool_calls"][0]["function"]["arguments"]["observations"][0]["source_refs"][0]["seq"] = 999
    try:
        check_observer_trace(broken, argument_schemas["observer"])
    except ValueError as error:
        require(str(error) == "Unknown source reference.", "Negative test failed for an unexpected reason.")
    else:
        raise ValueError("Negative unknown-source-ref test did not reject.")
    print(f"PASS: draft schema, {len(refs)} artifact references, hashes, and synthetic Observer trace.")
    print("PASS: empty-array task contracts and unknown-source-reference negative check.")
    print("Not checked: actual Hatfield execution, model compatibility, tokenization, semantic quality, or a real corpus.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    try:
        validate(args.root)
    except (OSError, ValueError, KeyError, TypeError, ValidationError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
