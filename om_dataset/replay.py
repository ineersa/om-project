"""Causal Reflector/Dropper generations over teacher-created observation pools."""
import hashlib
import json
import os
from pathlib import Path
import sys

from . import maintenance
from .collection import canonical


def _id(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _progress(result):
    print(json.dumps({"task": result.get("task", "observer"), "status": result["status"],
                      "run_id": result["job"]["run_id"], "cutoff_seq": result["job"]["cutoff_seq"],
                      "observations_added": len(result.get("observations", [])),
                      "reflections_added": len(result.get("reflections", [])),
                      "drops_committed": len(result.get("accepted_delta", {}).get("dropped_ids", []))}),
          file=sys.stderr, flush=True)


def _append(output, result, redactor):
    redactor.require_clean(result)
    output.write(canonical(result) + b"\n")
    output.flush()
    os.fsync(output.fileno())
    _progress(result)


def maintain(state, original, policies, config, complete, output, output_path, seen, summary, code_hash, redactor):
    """Commit reflection deltas and final drops together, or block the stream.

    Tool traces are durable independently. A generation_commit record is the
    commit point; interrupted generations reconstruct from cached tool deltas.
    """
    from .teacher import _context_tokens, _tool, run_job

    observations = state["observations"]
    tokens = maintenance.pool_tokens(observations)
    set_hash = maintenance.observation_set_hash(original["run_id"], [o["observation_id"] for o in observations])
    if tokens <= maintenance.REFLECT_AFTER_TOKENS or state.get("last_reflected_set") == set_hash:
        return state
    before = _id(state)
    gid = maintenance.generation_id(original["run_id"], state.get("generation_id"), set_hash,
                                    config.get("provider", "test") + "/" + config["model"])
    common = {"run_id": original["run_id"], "cutoff_seq": original["cutoff_seq"],
              "state_scope": original.get("state_scope", ""), "historical_range_id": original.get("historical_range_id"),
              "generation_id": gid, "state_before_id": before, "reflector_schema_version": "1"}
    reflections = list(state["reflections"])
    new_reflections, proposals = [], []
    envelope = int((config["context_window"] - config["request_options"].get("max_tokens", 0)) * 0.65)

    def phase(task):
        policy = policies[task]
        start = 0
        while start < len(observations):
            # Prior reflection visibility can shrink, but the persisted pool and
            # coverage/duplicate checks always retain every active reflection.
            visible = list(reflections)
            coverage = dict(maintenance._coverage(reflections))

            def job_at(end):
                return {**common, "task": task, "observations": observations[start:end], "reflections": visible,
                        "active_reflection_ids": [r["reflection_id"] for r in reflections],
                        "coverage_counts": coverage,
                        "pool_partition": {"start": start, "end": end, "total": len(observations)},
                        "pool_tokens": tokens, "target_tokens": maintenance.OBSERVATIONS_TARGET_TOKENS,
                        "max_drops": maintenance.max_drops(observations, maintenance.OBSERVATIONS_TARGET_TOKENS),
                        "omitted_reflection_ids": [r["reflection_id"] for r in reflections[len(visible):]]}

            def fits(end):
                text, _ = maintenance.render(job_at(end))
                return _context_tokens([{"role": "system", "content": policy["instructions"]},
                                        {"role": "user", "content": text}], [_tool(policy)]) <= envelope

            while visible and not fits(start + 1):
                visible.pop()
            low, high = start, len(observations)
            while low < high:
                middle = (low + high + 1) // 2
                if fits(middle):
                    low = middle
                else:
                    high = middle - 1
            job = job_at(max(start + 1, low))
            if low == start:
                job["_slice_budget_error"] = "maintenance_item_context"
            job = redactor.job(job)
            key = _id({"job": job, "policy": policy, "config": config, "code": code_hash})
            checkpoint = output_path.with_name(output_path.name + ".checkpoints") / (key + ".json")
            if key in seen:
                result = seen[key]
                checkpoint.unlink(missing_ok=True)
                summary["skipped"] += 1
            else:
                result = run_job(job, policy, config, complete, checkpoint, redactor)
                result.update(job_id=key, policy_id=policy["definition_id"], config=config, runner_sha256=code_hash,
                              generation_status="pending_commit")
                _append(output, result, redactor)
                summary[result["status"]] += 1
                if result["status"] == "completed":
                    seen[key] = result
                    checkpoint.unlink(missing_ok=True)
            if result["status"] != "completed":
                return False
            if task == "reflector":
                new_reflections.extend(result["reflections"])
                position = len(reflections)
                reflections.extend({**r, "position": position + i} for i, r in enumerate(result["reflections"]))
            else:
                proposals.extend(r["observation_id"] for r in result["drop_candidates"])
            start = low
        return True

    if not phase("reflector"):
        return None
    if new_reflections and tokens > maintenance.OBSERVATIONS_TARGET_TOKENS and not phase("dropper"):
        return None
    dropped = maintenance.select_drops(observations, reflections, proposals,
                                       maintenance.max_drops(observations, maintenance.OBSERVATIONS_TARGET_TOKENS))
    drop_set = set(dropped)
    after = {**state, "observations": [o for o in observations if o["observation_id"] not in drop_set],
             "reflections": reflections, "generation_id": gid, "last_reflected_set": set_hash}
    commit = {"task": "generation_commit", "status": "completed", "job": common,
              "job_id": _id({"generation": gid, "before": before, "policies": policies, "config": config, "code": code_hash}),
              "generation_id": gid, "state_before_id": before, "observations": [],
              "accepted_delta": {"reflection_ids": [r["reflection_id"] for r in new_reflections], "dropped_ids": dropped},
              "state_after": {"state_id": _id(after), "observation_ids": [o["observation_id"] for o in after["observations"]],
                              "reflection_ids": [r["reflection_id"] for r in reflections]},
              "pool_tokens_before": tokens, "pool_tokens_after": maintenance.pool_tokens(after["observations"]),
              "reflect_after_tokens": maintenance.REFLECT_AFTER_TOKENS,
              "target_tokens": maintenance.OBSERVATIONS_TARGET_TOKENS,
              "quality_status": "unreviewed", "training_ready": False, "redaction_id": redactor.fingerprint}
    if commit["job_id"] in seen:
        if seen[commit["job_id"]] != commit:
            raise ValueError("Generation commit does not match reconstructed state")
    else:
        _append(output, commit, redactor)
        seen[commit["job_id"]] = commit
    return after
