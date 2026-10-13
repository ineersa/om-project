"""Export source-pinned memory-agent policies without executing application code."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

from jsonschema import Draft202012Validator

from .collection import TARGET_REVISION, artifact, canonical, file_hash, now, resolve_artifact, write_json

SCHEMA_VERSION = "om-agent-policies/1"
PI_REVISION = "58f05fa0a857bc4fcd3dda0b3e845e8bfe069adf"
PI_LEGACY_REVISION = "00d5975"
HATFIELD_PREFIX = ".hatfield/extensions/observational-memory/src/"
API_PREFIX = ".hatfield/extensions/extension-api/src/Agent/"
GROUPS = {
    "hatfield-target": {"roles": {"observer", "reflector", "dropper"}, "language": "php"},
    "pi-current-native": {"roles": {"observer", "reflector", "dropper"}, "language": "ts"},
    "pi-legacy-v2-reference": {"roles": {"observer", "reflector", "pruner"}, "language": "ts"},
}


def _git(repo: Path, *args: str) -> bytes:
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, timeout=30)
    if result.returncode:
        raise ValueError("Cannot read pinned Git sources")
    return result.stdout


def _revision(repo: Path, revision: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{7,40}", revision):
        raise ValueError("Use a hexadecimal commit ID, not a moving branch")
    return _git(repo, "rev-parse", "--verify", revision + "^{commit}").decode().strip()


def _source_paths(repo: Path, revision: str, family: str) -> list[str]:
    paths = []
    for entry in _git(repo, "ls-tree", "-r", "-z", revision).split(b"\0"):
        if not entry:
            continue
        metadata, raw_path = entry.split(b"\t", 1)
        mode, kind, _ = metadata.split()
        path = raw_path.decode("utf-8")
        if family == "hatfield-target":
            wanted = ((path.startswith(HATFIELD_PREFIX) or path.startswith(API_PREFIX))
                      and path.endswith(".php")) or path in {"AGENTS.md", "LICENSE", "composer.json", "composer.lock"}
        else:
            prefix = "src/" if family == "pi-current-native" else "src_v2/"
            wanted = (path.startswith(prefix) and path.endswith(".ts")) or path in {"LICENSE", "package.json"}
        if not wanted:
            continue
        if mode not in {b"100644", b"100755"} or kind != b"blob":
            raise ValueError("Source must be a regular Git blob")
        if Path(path).is_absolute() or ".." in Path(path).parts:
            raise ValueError("Unsafe source path")
        paths.append(path)
    if not paths:
        raise ValueError("No policy sources at pinned revision")
    return sorted(paths)


def _freeze(repo: Path, revision: str, family: str, root: Path) -> None:
    for path in _source_paths(repo, revision, family):
        destination = root / path
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with destination.open("xb") as stream:
            os.chmod(destination, 0o600)
            stream.write(_git(repo, "show", revision + ":" + path))


def _extract(root: Path, group: str, helpers: Path, pi_runtime_repo: Path,
             php_parser_autoload: Path) -> dict:
    command = (["php", str(helpers / "extract_hatfield.php"), str(root), str(php_parser_autoload)]
               if group == "hatfield-target" else
               ["node", str(helpers / "extract_pi.mjs"), str(root), str(pi_runtime_repo),
                "legacy" if group == "pi-legacy-v2-reference" else "current"])
    result = subprocess.run(command, capture_output=True, timeout=60)
    if result.returncode:
        raise ValueError("Static policy extraction failed")
    return json.loads(result.stdout)


def _identity(agent: dict) -> str:
    return hashlib.sha256(canonical({k: v for k, v in agent.items() if k != "definition_id"})).hexdigest()


def _provenance(value, source_root: Path, source_paths: set[Path]) -> None:
    if isinstance(value, list):
        for item in value:
            _provenance(item, source_root, source_paths)
    elif isinstance(value, dict):
        if {"path", "start_line", "end_line"} <= value.keys():
            relative = Path(value["path"])
            source = (source_root / relative).resolve(strict=True)
            if relative.is_absolute() or source not in source_paths:
                raise ValueError("Definition refers to unarchived source")
            lines = source.read_text().splitlines(keepends=True)
            start, end = value["start_line"], value["end_line"]
            if type(start) is not int or type(end) is not int or not 1 <= start <= end <= len(lines):
                raise ValueError("Invalid source range")
            if value.get("kind") == "source_expression":
                expression = value.get("expression")
                if not isinstance(expression, str) or expression not in "".join(lines[start - 1:end]):
                    raise ValueError("Source expression differs from frozen source")
        for item in value.values():
            _provenance(item, source_root, source_paths)


def _check_bundle(bundle: dict, group: str, revision: str, source_root: Path,
                  source_paths: set[Path]) -> None:
    agents = bundle["agents"]
    if {a["role"] for a in agents} != GROUPS[group]["roles"] or len(agents) != 3:
        raise ValueError("Missing or repeated memory agent")
    if not isinstance(bundle.get("extractor_runtime"), dict) or not bundle["extractor_runtime"]:
        raise ValueError("Missing parser runtime versions")
    for agent in agents:
        if agent["family"] != group or agent["revision"] != revision or agent["definition_id"] != _identity(agent):
            raise ValueError("Policy identity or revision mismatch")
        if not isinstance(agent["instructions"], str) or not agent["instructions"].strip():
            raise ValueError("Missing instructions")
        tools = agent["tools"]
        if group == "pi-legacy-v2-reference":
            if tools or agent.get("native_call_mode") != "text_completion":
                raise ValueError("Legacy V2 has no worker tools")
        elif len(tools) != 1:
            raise ValueError("Expected one worker tool per agent")
        for tool in tools:
            if not isinstance(tool["name"], str) or not tool["name"]:
                raise ValueError("Missing tool name")
            if not isinstance(tool["description"], str) or not tool["description"]:
                raise ValueError("Missing tool description")
            Draft202012Validator.check_schema(tool["parameters"])
        if not agent.get("input_builders"):
            raise ValueError("Missing input-builder source")
        _provenance(agent, source_root, source_paths)


def export_policies(collection: Path, output: Path, hatfield_repo: Path, pi_repo: Path,
                    pi_runtime_repo: Path, php_parser_autoload: Path,
                    pi_revision: str = PI_REVISION, pi_legacy_revision: str = PI_LEGACY_REVISION) -> dict:
    collection = collection.expanduser().resolve(strict=True)
    output = output.expanduser().resolve()
    hatfield_repo = hatfield_repo.expanduser().resolve(strict=True)
    pi_repo = pi_repo.expanduser().resolve(strict=True)
    pi_runtime_repo = pi_runtime_repo.expanduser().resolve(strict=True)
    php_parser_autoload = php_parser_autoload.expanduser().resolve(strict=True)
    corpus_manifest = json.loads((collection / "manifest.json").read_text())
    if corpus_manifest["schema_version"] not in {"om-curated-corpus/1", "om-corpus/1"}:
        raise ValueError("Expected a frozen source or curated corpus")
    target_revision = corpus_manifest["target"]["revision"]
    revisions = {"hatfield-target": _revision(hatfield_repo, target_revision),
                 "pi-current-native": _revision(pi_repo, pi_revision),
                 "pi-legacy-v2-reference": _revision(pi_repo, pi_legacy_revision)}
    if revisions["hatfield-target"] != TARGET_REVISION:
        raise ValueError("Corpus target differs from supported Hatfield pin")
    if output.exists():
        raise FileExistsError("Use a new policy release directory")
    for source in (collection, hatfield_repo, pi_repo, pi_runtime_repo, php_parser_autoload.parent):
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("Policy output overlaps a source")
    output.mkdir(parents=True, mode=0o700)
    (output / ".gitignore").write_text("*\n")
    os.chmod(output / ".gitignore", 0o600)
    helpers = output / "exporter"
    helpers.mkdir(mode=0o700)
    for name in ("policies.py", "collection.py", "extract_hatfield.php", "extract_pi.mjs"):
        (helpers / name).write_bytes(Path(__file__).with_name(name).read_bytes())
        os.chmod(helpers / name, 0o400)
    for name in ("pyproject.toml", "uv.lock"):
        (helpers / name).write_bytes(Path(__file__).parent.parent.joinpath(name).read_bytes())
        os.chmod(helpers / name, 0o400)
    groups = {}
    for group, revision in revisions.items():
        root = output / group / "sources"
        repo = hatfield_repo if group == "hatfield-target" else pi_repo
        _freeze(repo, revision, group, root)
        bundle = _extract(root, group, helpers, pi_runtime_repo, php_parser_autoload)
        runtime_sources = []
        reexport = bundle["extractor_runtime"].get("type_reexport")
        if reexport:
            origin = Path(reexport["path"]).resolve(strict=True)
            if not origin.is_relative_to(pi_runtime_repo) or file_hash(origin) != reexport["sha256"]:
                raise ValueError("Pi-AI factory re-export source changed or escaped runtime repository")
            destination = output / group / "runtime-sources/pi-ai-index.ts"
            destination.parent.mkdir(mode=0o700)
            destination.write_bytes(origin.read_bytes())
            os.chmod(destination, 0o400)
            if file_hash(destination) != reexport["sha256"]:
                raise ValueError("Pi-AI factory re-export source changed during capture")
            runtime_sources.append(str(destination.relative_to(output)))
        for agent in bundle["agents"]:
            agent.update(family=group, revision=revision)
            agent["definition_id"] = _identity(agent)
        source_files = {p.resolve() for p in root.rglob("*") if p.is_file()}
        _check_bundle(bundle, group, revision, root, source_files)
        path = output / group / "definitions.json"
        write_json(path, bundle)
        groups[group] = {"revision": revision, "definitions": str(path.relative_to(output)),
                         "source_root": str(root.relative_to(output)),
                         "runtime_sources": runtime_sources,
                         "extractor_runtime": bundle["extractor_runtime"]}
    report = {"agent_definitions": 9, "worker_tool_definitions": 6,
              "groups": {group: {"revision": data["revision"], "roles": sorted(GROUPS[group]["roles"])}
                         for group, data in groups.items()},
              "model_calls": 0, "training_ready": False, "historical_request_identity_proven": False,
              "handlers_executed": False,
              "legacy_attribution": "V2 reference only; enriched om.observation and version-4 snapshot producer unproven",
              "runtime_inputs": "Archived builders, not reconstructed invocation inputs"}
    write_json(output / "report.json", report)
    files = sorted(p for p in output.rglob("*") if p.is_file())
    manifest = {"schema_version": SCHEMA_VERSION, "completed_at": now(), "groups": groups,
                "target_revision": target_revision,
                "corpus": {"path": os.path.relpath(collection, output),
                           "manifest_sha256": file_hash(collection / "manifest.json")},
                "training_ready": False, "model_calls": 0, "historical_request_identity_proven": False,
                "handlers_executed": False, "artifacts": [artifact(output, p) for p in files]}
    write_json(output / "manifest.json", manifest)
    return report


def validate_policies(root: Path) -> dict:
    root = root.expanduser().resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_text())
    if (manifest["schema_version"] != SCHEMA_VERSION or manifest["training_ready"] is not False
            or manifest["model_calls"] != 0 or manifest["historical_request_identity_proven"] is not False
            or manifest["handlers_executed"] is not False or manifest["target_revision"] != TARGET_REVISION):
        raise ValueError("Invalid policy manifest")
    if set(manifest["groups"]) != set(GROUPS):
        raise ValueError("Missing policy family")
    refs = manifest["artifacts"]
    paths = [ref["path"] for ref in refs]
    if len(paths) != len(set(paths)):
        raise ValueError("Repeated artifact path")
    resolved = {resolve_artifact(root, ref) for ref in refs}
    corpus = (root / manifest["corpus"]["path"]).resolve(strict=True)
    if file_hash(corpus / "manifest.json") != manifest["corpus"]["manifest_sha256"]:
        raise ValueError("Corpus manifest changed")
    if json.loads((corpus / "manifest.json").read_text())["target"]["revision"] != manifest["target_revision"]:
        raise ValueError("Corpus target mismatch")
    for group, data in manifest["groups"].items():
        source_root = (root / data["source_root"]).resolve(strict=True)
        definition = (root / data["definitions"]).resolve(strict=True)
        if not source_root.is_relative_to(root) or definition not in resolved:
            raise ValueError("Unarchived policy definition")
        if not re.fullmatch(r"[0-9a-f]{40}", data["revision"]):
            raise ValueError("Policy revision is not a full commit ID")
        if group == "hatfield-target" and data["revision"] != manifest["target_revision"]:
            raise ValueError("Hatfield policy target mismatch")
        source_files = {p for p in resolved if p.is_relative_to(source_root)}
        bundle = json.loads(definition.read_text())
        if bundle["extractor_runtime"] != data["extractor_runtime"]:
            raise ValueError("Parser runtime versions differ")
        reexport = bundle["extractor_runtime"].get("type_reexport")
        runtime_sources = [(root / p).resolve(strict=True) for p in data.get("runtime_sources", [])]
        if any(p not in resolved for p in runtime_sources):
            raise ValueError("Unarchived parser runtime source")
        if reexport:
            if len(runtime_sources) != 1 or file_hash(runtime_sources[0]) != reexport["sha256"]:
                raise ValueError("Pi-AI factory re-export proof mismatch")
        elif runtime_sources:
            raise ValueError("Unexpected parser runtime source")
        _check_bundle(bundle, group, data["revision"], source_root, source_files)
    return {"valid": True, "verified_artifacts": len(refs), "agent_definitions": 9,
            "worker_tool_definitions": 6, "training_ready": False, "model_calls": 0}
