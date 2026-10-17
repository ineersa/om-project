"""Pinned Hatfield maintenance contracts (5ea2fd92), without provider or storage code.

Callers append accepted reflections and apply only select_drops()'s final IDs.
Never prune reflection support when observations drop. ``schema`` is the exported
JSON schema for API compatibility; native handlers validate fields themselves.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import re

REFLECT_AFTER_TOKENS = 40_000
OBSERVATIONS_TARGET_TOKENS = 30_000
_TRIM = " \t\n\r\x00\x0b"
_SECRET = re.compile(
    r"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY|"
    r"\b(?:api[_-]?key|password|client[_-]?secret|access[_-]?token|auth[_-]?token|bearer\s+token|private[_-]?key)\b\s*[:=]|"
    r"\b(?:api[_-]?key|password|client[_-]?secret|access[_-]?token|auth[_-]?token)\b\s+[\"']?\S+",
    re.I | re.ASCII,
)


def _hash(value):
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    text = text.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def observation_set_hash(run, observation_ids):
    return _hash({"type": "observation-set-v1", "run_id": run,
                  "observation_ids": sorted(set(observation_ids))})


def generation_id(run, prior_active_generation_id, set_hash, reflector_model, reflector_schema_version="1"):
    return _hash({"type": "threshold-generation-v1", "run_id": run,
                  "prior_active_generation_id": prior_active_generation_id,
                  "observation_set_hash": set_hash, "reflector_model": reflector_model,
                  "reflector_schema_version": reflector_schema_version})


def pool_tokens(observations):
    return sum(math.ceil(len(o["content"]) / 2.35) for o in observations)


def max_drops(observations, target_tokens=OBSERVATIONS_TARGET_TOKENS):
    count, tokens = len(observations), pool_tokens(observations)
    if count <= 0 or tokens <= 0 or target_tokens < 0 or tokens <= target_tokens:
        return 0
    return min(count, max(1, math.ceil((tokens - target_tokens) / (tokens / count))))


def _coverage(reflections, unique=True):
    counts = Counter()
    for reflection in reflections:
        support = reflection["supporting_observation_ids"]
        counts.update(set(support) if unique else support)
    return counts


def render(job):
    """Return native user input and decimal-local-ID -> canonical observation map.

    Reflector sorts its display; Dropper preserves supplied repository order, even
    though both assign local IDs in timestamp/canonical-ID order.
    """
    task = job["task"]
    if task not in ("reflector", "dropper"):
        raise ValueError("Unknown maintenance task")
    observations, reflections = job["observations"], job["reflections"]
    allowed, seen = {}, set()
    for observation in sorted(observations, key=lambda o: (o.get("timestamp", ""), o["observation_id"])):
        oid = observation["observation_id"]
        if oid and oid not in seen:
            allowed[str(len(allowed) + 1)] = observation
            seen.add(oid)
    local = {o["observation_id"]: key for key, o in allowed.items()}
    counts = job.get("coverage_counts", _coverage(reflections, unique=task == "dropper"))
    if task == "reflector":
        reflections = sorted(reflections, key=lambda r: (r.get("position") or 0, r["reflection_id"]))
        observations = sorted(observations, key=lambda o: (o["timestamp"], o["observation_id"]))
    lines = ["CURRENT REFLECTIONS:"]
    lines.extend([f"[{r['reflection_id']}] {r['content']}" for r in reflections] or ["(none yet)"])
    lines.extend(["", "CURRENT OBSERVATIONS:"])
    for observation in observations:
        count = counts.get(observation["observation_id"], 0)
        tier = "strong" if count >= 2 else "partial" if count == 1 else "none"
        lines.append(f"[{local[observation['observation_id']]}] {observation['timestamp']} "
                     f"[{observation['relevance']}] [coverage: {tier}] {observation['content']}")
    if not observations:
        lines.append("(none yet)")
    lines.append("")
    if task == "reflector":
        lines.append("Crystallize any missing durable facts or patterns into new reflections. If nothing is stable enough, do not call the tool.")
    else:
        tokens, target, maximum = job["pool_tokens"], job["target_tokens"], job["max_drops"]
        fullness = math.floor(tokens / target * 100 + 0.5) if target > 0 else 0
        lines.extend([
            f"Active observation pool: ~{tokens:,} tokens; target: ~{target:,} tokens; fullness against target: ~{fullness}%; over target by ~{max(0, tokens - target):,} tokens.",
            f"Maximum drops allowed this run: {maximum:,} observation{'s' if maximum != 1 else ''}. This maximum is sized to move the active pool toward the target if every proposed drop is clearly safe.",
            "This maximum is a hard upper bound, not a target. Drop fewer or none if fewer observations are clearly safe.",
        ])
    if partition := job.get("pool_partition"):
        lines.append(f"This request shows observations {partition['start'] + 1}:{partition['end']} of {partition['total']} in the active pool. "
                     "Other entries are handled in separate requests; use only the listed local observation ids.")
    return "\n".join(lines), allowed


def _values(value):
    # PHP arrays represent both JSON arrays and JSON objects.
    return list(value.values()) if isinstance(value, dict) else value


def record(arguments, accepted, run, allowed, schema, job):
    """Accumulate native proposals/deltas. allowed is render()'s local-ID map.

    Existing reflections in job.reflections are duplicates, not replacement
    targets. Drop proposals are not capped until deterministic final selection.
    """
    task = job["task"]
    if task not in ("reflector", "dropper"):
        raise ValueError("Unknown maintenance task")
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else arguments
    except (ValueError, TypeError):
        args = None
    key = "reflections" if task == "reflector" else "ids"
    values = args.get(key) if isinstance(args, dict) else None
    if not isinstance(values, (list, dict)) or not values:
        message = "drop_observations requires a non-empty ids list."
        if task == "reflector":
            message = ("record_reflections requires at least one new reflection per call, or do not call the tool."
                       if isinstance(values, (list, dict)) else "record_reflections requires reflections as a list (array).")
        return {"status": "rejected", "error": "invalid_arguments", "message": message}
    added = duplicates = rejected = duplicate_request = 0
    seen = set()
    existing = set(job.get("active_reflection_ids", [r["reflection_id"] for r in job["reflections"]]))
    for raw in _values(values):
        if task == "dropper":
            observation = allowed.get(raw.strip(_TRIM)) if isinstance(raw, str) else None
            if observation is None:
                rejected += 1
                continue
            oid = observation["observation_id"]
            if oid in seen:
                duplicate_request += 1
            elif oid in accepted:
                duplicates += 1
            else:
                accepted[oid] = {"observation_id": oid}
                added += 1
            seen.add(oid)
            continue
        try:
            if not isinstance(raw, dict) or raw.get("retain_id") is not None or raw.get("retained_observation_ids") is not None:
                raise ValueError("Not a delta reflection")
            if not isinstance(raw.get("content"), str):
                raise ValueError("Invalid content")
            content = raw["content"].strip(_TRIM)
            if not content or "\n" in content or "\r" in content or _SECRET.search(content):
                raise ValueError("Invalid content")
            support = raw.get("supporting_observation_ids")
            if not isinstance(support, (list, dict)) or not support:
                raise ValueError("Missing support")
            support = sorted({allowed[s.strip(_TRIM)]["observation_id"] for s in _values(support)
                              if isinstance(s, str) and s.strip(_TRIM) in allowed})
            if any(not isinstance(s, str) or s.strip(_TRIM) not in allowed for s in _values(raw["supporting_observation_ids"])):
                raise ValueError("Unknown local support")
            rid = _hash({"type": "reflection-v1", "run_id": run,
                         "reflector_schema_version": job.get("reflector_schema_version", "1"),
                         "content": content, "supporting_observation_ids": support})
            if rid in existing or rid in accepted:
                duplicates += 1
            else:
                accepted[rid] = {"reflection_id": rid, "content": content, "supporting_observation_ids": support}
                added += 1
        except (ValueError, TypeError, KeyError):
            rejected += 1
    if task == "dropper":
        maximum = job["max_drops"]
        return {"status": "accepted", "added": added, "missing": rejected,
                "duplicate_in_request": duplicate_request, "duplicate_in_run": duplicates,
                "total_candidates": len(accepted), "max_drops_allowed": maximum,
                "guidance": f"Queued {added} drop candidate{'s' if added != 1 else ''}. Candidates this run: {len(accepted)}. Maximum drops allowed: {maximum}."}
    return {"status": "accepted", "added": added, "duplicates": duplicates, "rejected": rejected,
            "total_this_run": len(accepted),
            "guidance": f"Recorded {added} reflection{'s' if added != 1 else ''}; {duplicates} duplicate{'s' if duplicates != 1 else ''}; {rejected} rejected. Total this run: {len(accepted)}."}


def _timestamp(value):
    # Canonical minute stamps and ATOM only; not PHP's relative-date language.
    try:
        parsed = datetime.fromisoformat(value.replace(" ", "T"))
        return (parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)).timestamp()
    except (ValueError, OverflowError):
        return math.inf


def select_drops(observations, reflections, proposed_ids, maximum):
    if maximum <= 0 or not proposed_ids:
        return []
    by_id, counts = {o["observation_id"]: o for o in observations}, _coverage(reflections)
    candidates = [oid for oid in dict.fromkeys(proposed_ids) if oid in by_id]
    def rank(oid):
        observation = by_id[oid]
        return (0 if counts[oid] >= 2 else 1 if counts[oid] else 2,
                {"low": 0, "medium": 1, "high": 2, "critical": 3}.get(observation["relevance"], 1),
                _timestamp(observation["timestamp"]))
    return sorted(candidates, key=rank)[:max(0, maximum)]
