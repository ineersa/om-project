import copy
import hashlib
import json
from pathlib import Path

import pytest

from om_dataset.cli import main
from om_dataset.collection import artifact, canonical, collect, discover
from om_dataset.curation import curate_collection, validate_curated_collection


def observation(source="a", **changes):
    result = {"id": "012345abcdef", "content": "User asked about an available tool.",
              "timestamp": "2026-10-13 09:00", "relevance": "low",
              "sourceEntryIds": [source], "tokenCount": 8}
    result.update(changes)
    return result


def ledger(identity="obs", parent="a", record=None):
    return {"type": "custom", "id": identity, "parentId": parent,
            "customType": "om.observations.recorded", "data": {
                "coversUpToId": "a", "observations": [record or observation()]}}


def session(path, extra=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"type": "session", "id": path.stem, "version": 3},
            {"type": "message", "id": "a", "parentId": None,
             "message": {"role": "user", "content": "Is that tool available?"}}]
    rows.extend(extra or [])
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def corpus(tmp_path, extra, include_empty=False):
    session(tmp_path / "live/session.jsonl", extra)
    if include_empty:
        session(tmp_path / "live/no-om.jsonl")
    output = tmp_path / "corpus"
    collect(discover(tmp_path / "live", []), output)
    return output


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_only_om_sessions_and_low_relevance_remain(tmp_path):
    upstream = corpus(tmp_path, [ledger()], include_empty=True)
    manifest_before = (upstream / "manifest.json").read_bytes()
    output = tmp_path / "curated"
    report = curate_collection(upstream, output)
    assert report["retained_session_files"] == 1
    assert report["observer_relevance"] == {"low": 1}
    assert report["retained_candidates"] == 1
    assert rows(output / "sessions.jsonl")[0]["session_id"] == "session"
    candidate = rows(output / "memory-candidates.jsonl")[0]
    assert candidate["native_record"]["sourceEntryIds"] == ["a"]
    assert candidate["observation"]["source_refs"] == [{"run_id": "pi:session", "seq": 2}]
    assert candidate["candidate_status"] == "unreviewed"
    assert report["agent_requests_captured"] is False
    assert (upstream / "manifest.json").read_bytes() == manifest_before
    assert output.stat().st_mode & 0o777 == 0o700
    assert validate_curated_collection(output)["status"] == "pass"


def test_missing_source_and_dependents_are_excluded(tmp_path):
    bad = observation(source="missing")
    reflection = {"type": "custom", "id": "reflection", "parentId": "obs",
                  "customType": "om.reflections.recorded", "data": {
                      "coversUpToId": "a", "reflections": [{"id": "fedcba654321",
                          "content": "A conclusion.", "supportingObservationIds": [bad["id"]], "tokenCount": 5}]}}
    drop = {"type": "custom", "id": "drop", "parentId": "reflection",
            "customType": "om.observations.dropped", "data": {
                "coversUpToId": "a", "observationIds": [bad["id"]]}}
    upstream = corpus(tmp_path, [ledger(record=bad), reflection, drop])
    output = tmp_path / "curated"
    report = curate_collection(upstream, output)
    assert report["retained_candidates"] == 0
    assert report["retained_session_files"] == 0
    assert report["exclusions_by_reason"]["unresolved_dependent_content"] == 2
    assert validate_curated_collection(output)["status"] == "pass"


def test_reflector_dropper_and_snapshot_keep_resolved_indirect_evidence(tmp_path):
    record = observation()
    reflection_record = {"id": "fedcba654321", "content": "A conclusion.",
                         "supportingObservationIds": [record["id"]], "tokenCount": 5}
    reflection = {"type": "custom", "id": "reflection", "parentId": "obs",
                  "customType": "om.reflections.recorded", "data": {
                      "coversUpToId": "a", "reflections": [reflection_record]}}
    snapshot = {"type": "compaction", "id": "fold", "parentId": "reflection", "details": {
        "type": "om.folded", "version": 1, "fullFold": True,
        "observations": [record], "reflections": [reflection_record]}}
    drop = {"type": "custom", "id": "drop", "parentId": "fold",
            "customType": "om.observations.dropped", "data": {
                "coversUpToId": "a", "observationIds": [record["id"]]}}
    upstream = corpus(tmp_path, [ledger(), reflection, snapshot, drop])
    output = tmp_path / "curated"
    report = curate_collection(upstream, output)
    assert report["retained_candidates"] == 4
    proofs = rows(output / "eligibility.jsonl")
    assert all(p["supporting_candidate_ids"] for p in proofs[1:])
    assert validate_curated_collection(output)["status"] == "pass"


def test_same_evidence_duplicates_removed_without_merging_different_branches(tmp_path):
    upstream = corpus(tmp_path, [ledger(), ledger("repeat", "obs"), ledger("other-branch", "a")])
    output = tmp_path / "curated"
    report = curate_collection(upstream, output)
    assert report["retained_candidates"] == 2
    assert report["exclusions_by_reason"] == {"exact_same_evidence_duplicate": 1}
    exclusion = rows(output / "exclusions.jsonl")[0]
    assert exclusion["kept_candidate_id"] in {c["candidate_id"] for c in rows(output / "memory-candidates.jsonl")}
    assert validate_curated_collection(output)["status"] == "pass"


def test_same_text_with_different_evidence_is_not_deduplicated(tmp_path):
    second_source = {"type": "message", "id": "b", "parentId": "obs",
                     "message": {"role": "user", "content": "Again: is that tool available?"}}
    upstream = corpus(tmp_path, [ledger(), second_source, ledger("second", "b", observation(source="b"))])
    report = curate_collection(upstream, tmp_path / "curated")
    assert report["retained_candidates"] == 2
    assert report["excluded_candidates"] == 0


def test_future_or_off_branch_support_is_not_accepted(tmp_path):
    reflection = {"type": "custom", "id": "reflection", "parentId": "a",
                  "customType": "om.reflections.recorded", "data": {
                      "coversUpToId": "a", "reflections": [{"id": "fedcba654321", "content": "Conclusion.",
                          "supportingObservationIds": [observation()["id"]], "tokenCount": 5}]}}
    upstream = corpus(tmp_path, [reflection, ledger("future", "a")])
    report = curate_collection(upstream, tmp_path / "curated")
    assert report["retained_candidates"] == 1
    assert report["exclusions_by_reason"] == {"unresolved_dependent_content": 1}


def test_snapshot_with_same_content_id_but_changed_refs_is_excluded(tmp_path):
    modified = observation(source="missing")
    snapshot = {"type": "compaction", "id": "fold", "parentId": "obs", "details": {
        "type": "om.folded", "version": 1, "fullFold": True,
        "observations": [modified], "reflections": []}}
    upstream = corpus(tmp_path, [ledger(), snapshot])
    report = curate_collection(upstream, tmp_path / "curated")
    assert report["retained_candidates"] == 1
    assert report["exclusions_by_reason"] == {"unresolved_snapshot_content": 1}


def test_curation_cli_validation_and_artifact_tampering(tmp_path, capsys):
    upstream = corpus(tmp_path, [ledger()])
    output = tmp_path / "curated"
    assert main(["curate", str(upstream), "--output", str(output)]) == 0
    assert main(["validate", str(output)]) == 0
    assert "Is that tool available?" not in capsys.readouterr().out
    with (output / "memory-candidates.jsonl").open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="integrity"):
        validate_curated_collection(output)


def test_refuse_existing_output_and_source_overlap(tmp_path):
    upstream = corpus(tmp_path, [ledger()])
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        curate_collection(upstream, existing)
    with pytest.raises(ValueError, match="source collection"):
        curate_collection(upstream, upstream / "nested")


def reflection(parent="obs"):
    return {"type": "custom", "id": "reflection", "parentId": parent,
            "customType": "om.reflections.recorded", "data": {
                "coversUpToId": "a", "reflections": [{"id": "fedcba654321", "content": "Conclusion.",
                    "supportingObservationIds": [observation()["id"]], "tokenCount": 5}]}}


def test_unresolved_first_native_id_winner_cannot_be_replaced_by_later_good_alias(tmp_path):
    snapshot = {"type": "compaction", "id": "fold", "parentId": "reflection", "details": {
        "type": "om.folded", "version": 1, "fullFold": True,
        "observations": [observation()], "reflections": []}}
    upstream = corpus(tmp_path, [ledger(record=observation(source="missing")),
                                ledger("later", "obs"), reflection("later"), snapshot])
    report = curate_collection(upstream, tmp_path / "curated")
    assert report["retained_candidates"] == 1
    assert report["exclusions_by_reason"]["unresolved_dependent_content"] == 1
    assert report["exclusions_by_reason"]["unresolved_snapshot_content"] == 1


def test_snapshot_cannot_choose_later_native_id_alias_with_different_fields(tmp_path):
    modified = observation(relevance="high")
    snapshot = {"type": "compaction", "id": "fold", "parentId": "later", "details": {
        "type": "om.folded", "version": 1, "fullFold": True,
        "observations": [modified], "reflections": []}}
    upstream = corpus(tmp_path, [ledger(), ledger("later", "obs", modified), snapshot])
    report = curate_collection(upstream, tmp_path / "curated")
    assert report["retained_candidates"] == 2
    assert report["exclusions_by_reason"] == {"unresolved_snapshot_content": 1}


def test_native_invalid_batch_does_not_win_an_id(tmp_path):
    invalid_batch = ledger()
    invalid_batch["data"]["observations"].append(observation(id="not-a-memory-id"))
    upstream = corpus(tmp_path, [invalid_batch, ledger("later", "obs"), reflection("later")])
    output = tmp_path / "curated"
    report = curate_collection(upstream, output)
    assert report["retained_by_adapter_and_task"]["pi-current/reflector"] == 1
    assert validate_curated_collection(output)["status"] == "pass"


@pytest.mark.parametrize("defect", ["duplicate", "forward"])
def test_invalid_intermediate_ancestry_blocks_dependent_evidence(tmp_path, defect):
    intermediate = {"type": "custom", "id": "p", "parentId": "obs", "customType": "control"}
    sequence = [ledger()]
    if defect == "duplicate":
        sequence += [intermediate, reflection("p"), copy.deepcopy(intermediate)]
    else:
        sequence += [reflection("p"), intermediate]
    upstream = corpus(tmp_path, sequence)
    report = curate_collection(upstream, tmp_path / "curated")
    assert report["retained_candidates"] == 1
    assert report["exclusions_by_reason"] == {"invalid_native_ancestry": 1}


def rewrite_rows(path, values):
    path.write_bytes(b"".join(canonical(value) + b"\n" for value in values))


def refresh_artifact_hashes(output):
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["artifacts"] = [artifact(output, output / ref["path"]) for ref in manifest["artifacts"]]
    path.write_bytes(canonical(manifest) + b"\n")


def test_validator_rejects_self_consistent_silent_omission(tmp_path):
    upstream = corpus(tmp_path, [ledger()])
    output = tmp_path / "curated"
    curate_collection(upstream, output)
    for name in ("memory-candidates.jsonl", "eligibility.jsonl"):
        rewrite_rows(output / name, [])
    report = json.loads((output / "report.json").read_text())
    report.update(input_candidates=0, retained_candidates=0)
    (output / "report.json").write_bytes(canonical(report))
    refresh_artifact_hashes(output)
    with pytest.raises(ValueError, match="differs from upstream selection"):
        validate_curated_collection(output)


def test_validator_rejects_repeated_ids_even_with_recomputed_artifact_hashes(tmp_path):
    upstream = corpus(tmp_path, [ledger()])
    output = tmp_path / "curated"
    curate_collection(upstream, output)
    for name in ("memory-candidates.jsonl", "eligibility.jsonl"):
        values = rows(output / name)
        rewrite_rows(output / name, values + values)
    refresh_artifact_hashes(output)
    with pytest.raises(ValueError, match="Repeated candidate ID"):
        validate_curated_collection(output)


def test_validator_rejects_substituted_candidate_with_valid_self_hash(tmp_path):
    upstream = corpus(tmp_path, [ledger()])
    output = tmp_path / "curated"
    curate_collection(upstream, output)
    candidate = rows(output / "memory-candidates.jsonl")[0]
    candidate["observation"]["content"] = "Not the saved candidate."
    candidate["candidate_id"] = hashlib.sha256(canonical({k: v for k, v in candidate.items()
                                                         if k != "candidate_id"})).hexdigest()
    rewrite_rows(output / "memory-candidates.jsonl", [candidate])
    proof = rows(output / "eligibility.jsonl")[0]
    proof["candidate_id"] = candidate["candidate_id"]
    rewrite_rows(output / "eligibility.jsonl", [proof])
    refresh_artifact_hashes(output)
    with pytest.raises(ValueError, match="differs from upstream selection"):
        validate_curated_collection(output)
