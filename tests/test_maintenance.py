from copy import deepcopy

from om_dataset.maintenance import (REFLECT_AFTER_TOKENS, OBSERVATIONS_TARGET_TOKENS,
                                    render, record, pool_tokens, max_drops, select_drops,
                                    observation_set_hash, generation_id)


def obs(oid, relevance="low", timestamp="2026-01-01 00:00", content="Fact."):
    return dict(observation_id=oid, relevance=relevance, timestamp=timestamp, content=content, source_refs=[])


def test_native_prompts_and_local_ids_preserve_distinct_display_orders():
    observations = [obs("z", timestamp="2026-02-01 00:00"), obs("a")]
    reflections = [dict(reflection_id="r2", content="Second.", supporting_observation_ids=["a", "gone"], position=1),
                   dict(reflection_id="r1", content="First.", supporting_observation_ids=["a"], position=0)]
    job = dict(task="reflector", observations=observations, reflections=reflections)
    text, allowed = render(job)
    assert list(allowed) == ["1", "2"] and allowed["1"]["observation_id"] == "a"
    assert text == ("CURRENT REFLECTIONS:\n[r1] First.\n[r2] Second.\n\nCURRENT OBSERVATIONS:\n"
                    "[1] 2026-01-01 00:00 [low] [coverage: strong] Fact.\n"
                    "[2] 2026-02-01 00:00 [low] [coverage: none] Fact.\n\n"
                    "Crystallize any missing durable facts or patterns into new reflections. If nothing is stable enough, do not call the tool.")
    job.update(task="dropper", max_drops=1, pool_tokens=1, target_tokens=40)
    text, same = render(job)
    assert same == allowed and text.index("[r2]") < text.index("[r1]")
    assert text.index("[2] 2026") < text.index("[1] 2026")
    assert text.endswith("Active observation pool: ~1 tokens; target: ~40 tokens; fullness against target: ~3%; over target by ~0 tokens.\n"
                         "Maximum drops allowed this run: 1 observation. This maximum is sized to move the active pool toward the target if every proposed drop is clearly safe.\n"
                         "This maximum is a hard upper bound, not a target. Drop fewer or none if fewer observations are clearly safe.")
    assert reflections[0]["supporting_observation_ids"] == ["a", "gone"]


def test_reflector_partial_acceptance_native_receipts_and_existing_duplicates():
    job = dict(task="reflector", observations=[obs("z"), obs("a")], reflections=[])
    _, allowed = render(job)
    good = dict(content=" Fact. ", supporting_observation_ids=["2", " 1 ", "2"])
    bad = dict(content="Unsupported.", supporting_observation_ids=["a"])
    secret = dict(content="password=example", supporting_observation_ids=["1"])
    accepted = {}
    receipt = record({"reflections": [good, good, bad, secret]}, accepted, "run", allowed, {}, job)
    assert receipt == dict(status="accepted", added=1, duplicates=1, rejected=2, total_this_run=1,
                           guidance="Recorded 1 reflection; 1 duplicate; 2 rejected. Total this run: 1.")
    rid = "ffe1fa878bd59d46e6ecef9646a95e7f458acc2951ab2db7e0a95de85918338f"
    assert accepted == {rid: dict(reflection_id=rid, content="Fact.", supporting_observation_ids=["a", "z"])}
    assert job["reflections"] == []  # caller appends the delta, never replaces prior memory
    job["reflections"] = list(accepted.values())
    assert record({"reflections": [good]}, {}, "run", allowed, {}, job)["duplicates"] == 1
    assert record({"reflections": [bad]}, {}, "run", allowed, {}, job)["status"] == "accepted"
    assert record('{"reflections":[]}', {}, "run", allowed, {}, job) == dict(
        status="rejected", error="invalid_arguments",
        message="record_reflections requires at least one new reflection per call, or do not call the tool.")


def test_drop_proposals_are_not_capped_and_duplicate_counts_are_native():
    job = dict(task="dropper", observations=[obs("a"), obs("z")], reflections=[], max_drops=1)
    allowed = {"1": job["observations"][0], "2": job["observations"][1]}
    accepted = {}
    result = record({"ids": ["1", "1", "z", 2, "2"], "reason": "not persisted"}, accepted, "run", allowed, {}, job)
    assert result == dict(status="accepted", added=2, missing=2, duplicate_in_request=1, duplicate_in_run=0,
                          total_candidates=2, max_drops_allowed=1,
                          guidance="Queued 2 drop candidates. Candidates this run: 2. Maximum drops allowed: 1.")
    assert accepted == {"a": {"observation_id": "a"}, "z": {"observation_id": "z"}}
    result = record({"ids": ["1", "1"]}, accepted, "run", allowed, {}, job)
    assert result["duplicate_in_run"] == result["duplicate_in_request"] == 1
    assert record({}, accepted, "run", allowed, {}, job)["message"] == "drop_observations requires a non-empty ids list."


def test_pool_cap_and_server_ranking_keep_dangling_reflection_support():
    assert (REFLECT_AFTER_TOKENS, OBSERVATIONS_TARGET_TOKENS) == (40000, 30000)
    observations = [obs("strong", "critical"), obs("partial", "low"), obs("none", "low"),
                    obs("tie1"), obs("tie2"), obs("old", timestamp="2025-12-31 00:00")]
    reflections = [dict(reflection_id="r1", supporting_observation_ids=["strong", "strong", "partial", "gone"]),
                   dict(reflection_id="r2", supporting_observation_ids=["strong"])]
    before = deepcopy(reflections)
    assert pool_tokens([obs("a", content="🦉é"), obs("z", content="")]) == 1
    assert max_drops(observations, pool_tokens(observations)) == 0
    assert max_drops(observations, 0) == len(observations)
    assert max_drops(observations, pool_tokens(observations) - 1) == 1
    assert max_drops([], 0) == max_drops(observations, -1) == 0
    proposed = ["tie2", "none", "tie1", "old", "partial", "strong", "gone", "tie2"]
    assert select_drops(observations, reflections, proposed, 6) == ["strong", "partial", "old", "tie2", "none", "tie1"]
    assert select_drops(observations, reflections, proposed, 1) == ["strong"]
    assert reflections == before


def test_set_and_generation_identity_formulas():
    set_hash = observation_set_hash("run", ["z", "a", "a"])
    assert set_hash == "a8f64b4fb2fad75ab83cb0d444e709e6282bb0aeb13b18fb71f2405e3196de1f"
    assert generation_id("run", None, set_hash, "teacher") == generation_id("run", None, set_hash, "teacher", "1")
    assert generation_id("run", None, set_hash, "teacher") != generation_id("run", "prior", set_hash, "teacher")
