import hashlib
import json
from pathlib import Path

import pytest

from om_dataset import hatfield_review as h
from om_dataset.collection import canonical
from om_dataset.redaction import Redactor


def test_publication_retires_pi_preserves_hatfield_cases_and_votes_and_resumes(tmp_path, monkeypatch):
    for name, value in (("ROOT", "campaign"), ("ACTIVE", "review"), ("PRIVATE", "private"), ("HISTORY", "history")):
        monkeypatch.setattr(h, name, tmp_path / value)
        getattr(h, name).mkdir()
    catalog = tmp_path / "catalog.json"
    catalog.write_text('{"values": []}')
    redactor = Redactor(catalog=catalog)
    def case(cid, task):
        return {"case_id": cid, "task": task, "schema_version": "om-review-case/1", "redaction_id": redactor.fingerprint,
                "evidence": [{"source_id": "source", "text": "A supported fact."}], "prior_memory": "None.",
                "candidates": {"A": "First wording.", "B": "Second wording."}}
    tasks = ("observer", "reflector", "dropper")
    old = [case("pi-" + t, t) for t in tasks] + [case("hf-" + t, t) for t in tasks]
    old_bytes = b"".join(canonical(x) + b"\n" for x in old)
    (h.ACTIVE / "cases.jsonl").write_bytes(old_bytes)
    (h.PRIVATE / "identity-map.json").write_text('{"original": true}')
    (h.ACTIVE / "votes").mkdir()
    for x in (old[0], old[3]):
        (h.ACTIVE / "votes" / (x["case_id"] + ".json")).write_bytes(canonical({
            "case_id": x["case_id"], "case_sha256": h.c._hash(x), "choice": "A", "answer_source": "human"}))
    kept_vote = (h.ACTIVE / "votes/hf-observer.json").read_bytes()
    new = [case("new-" + t, t) for t in tasks]
    identities = [{"case_id": x["case_id"], "A": "producer-1", "B": "producer-2"} for x in new]
    saved = {"old_cases_sha256": hashlib.sha256(old_bytes).hexdigest(),
             "old_mapping_sha256": h.file_hash(h.PRIVATE / "identity-map.json"),
             "old_case_hashes": {x["case_id"]: h.c._hash(x) for x in old}, "retained_cases": old[3:],
             "retained_mapping": [{"case_id": x["case_id"], "A": "old-1", "B": "old-2"} for x in old[3:]],
             "cases": [{"case_id": x["case_id"]} for x in new]}
    replace = h.os.replace
    def interrupt_after_archiving(source, target):
        result = replace(source, target)
        if Path(source) == h.ACTIVE:
            raise InterruptedError("Stopped after archiving the old public pack")
        return result
    with monkeypatch.context() as patch:
        patch.setattr(h.os, "replace", interrupt_after_archiving)
        with pytest.raises(InterruptedError):
            h.publish(saved, new, identities, redactor)
    assert not h.ACTIVE.exists()
    assert h.publish(saved, new, identities, redactor)["source_kind"] == "hatfield"
    active = [json.loads(s) for s in (h.ACTIVE / "cases.jsonl").read_bytes().splitlines()]
    assert active == new + old[3:]
    assert (h.ACTIVE / "votes/hf-observer.json").read_bytes() == kept_vote
    assert not (h.ACTIVE / "votes/pi-observer.json").exists()
    assert (h.HISTORY / saved["old_cases_sha256"] / "review/votes/pi-observer.json").exists()
    # A vote made after publication must survive a repeated resume.
    vote = h.ACTIVE / "votes/new-observer.json"
    vote.write_text('{"choice": "B"}')
    h.publish(saved, new, identities, redactor)
    assert vote.read_text() == '{"choice": "B"}'
