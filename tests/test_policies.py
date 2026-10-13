from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from om_dataset.cli import main
from om_dataset.collection import TARGET_REVISION, artifact, file_hash
from om_dataset.policies import (
    HATFIELD_PREFIX, PI_REVISION, _git, _identity, _revision,
    export_policies, validate_policies,
)

HATFIELD = Path.home() / "projects/agent-core"
PI = Path.home() / "claw/pi-observational-memory"
RUNTIME = Path.home() / "claw/pi-mono"
AUTOLOAD = HATFIELD / "vendor/autoload.php"


def _write(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture(scope="module")
def release(tmp_path_factory):
    required = [HATFIELD / ".git", PI / ".git", AUTOLOAD,
                RUNTIME / "node_modules/typescript", RUNTIME / "node_modules/typebox"]
    if not all(p.exists() for p in required) or not shutil.which("php") or not shutil.which("node"):
        pytest.skip("Pinned external source repositories and AST tooling are unavailable")
    base = tmp_path_factory.mktemp("policy-release")
    corpus = base / "corpus"
    corpus.mkdir()
    _write(corpus / "manifest.json", {"schema_version": "om-corpus/1",
                                     "target": {"revision": TARGET_REVISION}})
    output = base / "policies"
    report = export_policies(corpus, output, HATFIELD, PI, RUNTIME, AUTOLOAD)
    return corpus, output, report


def _copy_release(release, tmp_path):
    corpus, output, _ = release
    shutil.copytree(corpus, tmp_path / "corpus")
    destination = tmp_path / "policies"
    shutil.copytree(output, destination)
    return destination


def _resign(root, path):
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["artifacts"] = [artifact(root, path) if r["path"] == str(path.relative_to(root)) else r
                             for r in manifest["artifacts"]]
    _write(root / "manifest.json", manifest)


def test_export_contains_exact_pinned_sources_and_private_artifacts(release):
    _, output, report = release
    assert report["agent_definitions"] == 9
    assert report["worker_tool_definitions"] == 6
    assert report["model_calls"] == 0
    assert report["handlers_executed"] is False
    assert report["historical_request_identity_proven"] is False
    assert validate_policies(output)["valid"]
    assert (output / ".gitignore").read_text() == "*\n"
    source = HATFIELD_PREFIX + "Observer/ObserverSystemPrompt.php"
    assert (output / "hatfield-target/sources" / source).read_bytes() == _git(
        HATFIELD, "show", TARGET_REVISION + ":" + source)
    assert (output / "pi-current-native/sources/src/agents/observer/agent.ts").read_bytes() == _git(
        PI, "show", PI_REVISION + ":src/agents/observer/agent.ts")
    assert (output.stat().st_mode & 0o777) == 0o700
    assert ((output / "hatfield-target/definitions.json").stat().st_mode & 0o777) == 0o600


def test_hatfield_schema_limits_and_runtime_sources(release):
    _, root, _ = release
    bundle = json.loads((root / "hatfield-target/definitions.json").read_text())
    agents = {a["role"]: a for a in bundle["agents"]}
    observer = agents["observer"]
    assert observer["tools"][0]["name"] == "record_observations"
    observation = observer["tools"][0]["parameters"]["properties"]["observations"]["items"]
    assert observation["required"] == ["timestamp", "content", "relevance", "source_refs"]
    assert observation["properties"]["relevance"]["enum"] == ["low", "medium", "high", "critical"]
    assert observer["call_options"] == {"maxToolCalls": 6, "maxDurationSeconds": 300}
    assert agents["reflector"]["call_options"] == {"maxToolCalls": 16, "maxDurationSeconds": 300}
    assert agents["dropper"]["call_options"] == {
        "maxToolCalls": 16, "maxDurationSeconds": 300, "thinkingLevel": "off"}
    assert "RequestLocalObservationIdMap" in agents["reflector"]["tools"][0]["handler_source"]["expression"]
    assert all(a["tools"][0]["handler_source"]["kind"] == "source_expression" for a in agents.values())
    assert all(a["input_builders"] for a in agents.values())
    assert bundle["extractor_runtime"]["php_parser"]
    assert "recall" in bundle["omitted_tools"]


def test_legacy_is_not_misrepresented_as_current_tool_contract(release):
    _, output, report = release
    legacy = json.loads((output / "pi-legacy-v2-reference/definitions.json").read_text())
    assert {a["role"] for a in legacy["agents"]} == {"observer", "reflector", "pruner"}
    assert all(a["tools"] == [] and a["native_call_mode"] == "text_completion" for a in legacy["agents"])
    assert "producer unproven" in report["legacy_attribution"]
    native = json.loads((output / "pi-current-native/definitions.json").read_text())
    schema = native["agents"][0]["tools"][0]["parameters"]
    assert "sourceEntryIds" in schema["properties"]["observations"]["items"]["properties"]
    assert native["extractor_runtime"]["typebox"]


def test_verified_pi_ai_reexport_source_is_archived(release):
    _, root, _ = release
    manifest = json.loads((root / "manifest.json").read_text())
    group = manifest["groups"]["pi-current-native"]
    assert len(group["runtime_sources"]) == 1
    assert file_hash(root / group["runtime_sources"][0]) == group["extractor_runtime"]["type_reexport"]["sha256"]
    assert (root / "exporter/uv.lock").exists()
    assert (root / "exporter/collection.py").exists()


def test_validator_requires_reexport_proof_even_after_manifest_edit(release, tmp_path):
    root = _copy_release(release, tmp_path)
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["groups"]["pi-current-native"]["runtime_sources"] = []
    _write(path, manifest)
    with pytest.raises(ValueError, match="re-export proof"):
        validate_policies(root)


def test_validation_is_offline_and_does_not_reopen_source_repositories(release, monkeypatch):
    _, output, _ = release
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("Validation must not run application tools"))
    assert validate_policies(output)["valid"]


def test_cli_validation_dispatches_without_printing_instructions(release, capsys):
    _, output, _ = release
    assert main(["validate", str(output)]) == 0
    text = capsys.readouterr().out
    assert json.loads(text)["worker_tool_definitions"] == 6
    assert "You are" not in text


def test_refuses_existing_output_and_keeps_original_release(release):
    corpus, output, _ = release
    before = file_hash(output / "manifest.json")
    with pytest.raises(FileExistsError):
        export_policies(corpus, output, HATFIELD, PI, RUNTIME, AUTOLOAD)
    assert file_hash(output / "manifest.json") == before


def test_rejects_moving_revision_names_before_using_git(tmp_path):
    with pytest.raises(ValueError, match="commit ID"):
        _revision(tmp_path, "main")


def test_rejects_modified_source_artifact(release, tmp_path):
    root = _copy_release(release, tmp_path)
    source = root / "pi-current-native/sources/src/agents/observer/agent.ts"
    source.write_text(source.read_text() + "\n// changed\n")
    with pytest.raises(ValueError, match="integrity"):
        validate_policies(root)


def test_rejects_repeated_artifact_paths(release, tmp_path):
    root = _copy_release(release, tmp_path)
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["artifacts"].append(manifest["artifacts"][0])
    _write(path, manifest)
    with pytest.raises(ValueError, match="Repeated artifact"):
        validate_policies(root)


def test_rejects_corpus_manifest_change(release, tmp_path):
    root = _copy_release(release, tmp_path)
    (tmp_path / "corpus/manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="Corpus manifest"):
        validate_policies(root)


def test_rejects_modified_source_expression_even_with_rehashed_definition(release, tmp_path):
    root = _copy_release(release, tmp_path)
    path = root / "hatfield-target/definitions.json"
    bundle = json.loads(path.read_text())
    agent = bundle["agents"][0]
    agent["input_builders"][0]["expression"] = "not the source expression"
    agent["definition_id"] = _identity(agent)
    _write(path, bundle)
    _resign(root, path)
    with pytest.raises(ValueError, match="expression differs"):
        validate_policies(root)


def test_rejects_unarchived_provenance_with_rehashed_definition(release, tmp_path):
    root = _copy_release(release, tmp_path)
    path = root / "hatfield-target/definitions.json"
    bundle = json.loads(path.read_text())
    agent = bundle["agents"][0]
    agent["source"]["instructions"]["path"] = "../../manifest.json"
    agent["definition_id"] = _identity(agent)
    _write(path, bundle)
    _resign(root, path)
    with pytest.raises((ValueError, FileNotFoundError)):
        validate_policies(root)


def test_php_ast_does_not_execute_source_and_rejects_dynamic_schema(release, tmp_path):
    _, output, _ = release
    root = tmp_path / "sources"
    shutil.copytree(output / "hatfield-target/sources", root)
    prompt = root / HATFIELD_PREFIX / "Observer/ObserverSystemPrompt.php"
    marker = tmp_path / "executed"
    text = prompt.read_text().replace("final class ObserverSystemPrompt",
                                      "file_put_contents(" + repr(str(marker)) + ", 'bad');\n\nfinal class ObserverSystemPrompt", 1)
    prompt.write_text(text)
    command = ["php", "om_dataset/extract_hatfield.php", str(root), str(AUTOLOAD)]
    result = subprocess.run(command, capture_output=True)
    # The parser reads the side-effect statement but never evaluates application code.
    assert result.returncode == 0, result.stderr.decode()
    assert not marker.exists()
    pipeline = root / HATFIELD_PREFIX / "Observer/ObserverPipeline.php"
    pipeline.write_text(pipeline.read_text().replace("'required' => ['observations']",
                                                   "'required' => dynamic_schema()", 1))
    result = subprocess.run(command, capture_output=True)
    assert result.returncode != 0
    assert not result.stdout
    assert not marker.exists()
