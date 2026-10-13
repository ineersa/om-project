"""Offline extraction tests. Application modules are never loaded."""
import io
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

import pytest

EXTRACTOR = Path(__file__).resolve().parents[1] / "om_dataset" / "extract_pi.mjs"
RUNTIME = Path(os.environ.get("PI_RUNTIME_REPO", "/home/ineersa/claw/pi-mono"))
SOURCE_REPO = Path(os.environ.get("PI_OM_REPO", "/home/ineersa/claw/pi-observational-memory"))


@pytest.fixture
def runtime():
    if not shutil.which("node") or not (RUNTIME / "node_modules/typescript").exists() or not (RUNTIME / "node_modules/typebox").exists():
        pytest.skip("Node, TypeScript, and native TypeBox runtime are required")
    return RUNTIME


def extract(root, runtime, family="current", *, success=True):
    result = subprocess.run(
        ["node", str(EXTRACTOR), str(root), str(runtime), family],
        capture_output=True, text=True, timeout=60,
    )
    if success:
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)
    assert result.returncode != 0
    assert not result.stdout
    return result.stderr


def write(root, relative, text):
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)


def current_fixture(root, *, schema="Type.Object({ids: Type.Array(Type.String({minLength: 1})), reason: Type.Optional(Type.String({description: 'optional note'}))})", description="'Record ' + 'exactly.'"):
    for role in ("observer", "reflector", "dropper"):
        write(root, f"src/agents/{role}/prompts.ts", f'''const PREFIX = "Line one\\n";
export const {role.upper()}_SYSTEM = `${{PREFIX}}Line two: \\`quoted\\` and ${{"literal"}}`;
throw new Error("Application prompt executed");
''')
        write(root, f"src/agents/{role}/agent.ts", f'''import {{Type}} from 'typebox';
const Schema = {schema};
throw new Error("Application agent executed");
export function run{role.title()}(args: unknown) {{
  const tool: AgentTool<typeof Schema> = {{
    name: 'record_{role}', description: {description}, parameters: Schema,
    execute: async () => dangerousHandler()
  }};
  const userText = `Source: ${{args.chunk}}; clock: ${{Date.now()}}`;
  const context = {{systemPrompt: {role.upper()}_SYSTEM, tools: [tool]}};
  const thinkingLevel = args.thinkingLevel ?? 'low';
  const config = {{maxTurns: args.maxTurns, maxTokens: boundedMaxTokens(args.model)}};
}}
''')


def test_literals_native_schema_and_unrendered_inputs(tmp_path, runtime):
    current_fixture(tmp_path)
    bundle = extract(tmp_path, runtime)
    assert [agent["role"] for agent in bundle["agents"]] == ["observer", "reflector", "dropper"]
    for agent in bundle["agents"]:
        assert agent["instructions"] == 'Line one\nLine two: `quoted` and literal'
        tool = agent["tools"][0]
        assert tool["description"] == "Record exactly."
        assert tool["parameters"]["required"] == ["ids"]
        assert tool["parameters"]["properties"]["reason"]["description"] == "optional note"
        assert tool["parameters"]["properties"]["ids"]["items"]["minLength"] == 1
        assert tool["execute_source"]["expression"] == "async () => dangerousHandler()"
        builder = agent["input_builders"][0]
        assert builder["expression"] == '`Source: ${args.chunk}; clock: ${Date.now()}`'
        assert builder["kind"] == "source_expression"
        assert builder["enclosing_function"]["name"] == f"run{agent['role'].title()}"
        assert builder["start_line"] == builder["end_line"]
        assert any(item["name"] == "config" and "boundedMaxTokens" in item["expression"] for item in agent["runtime_expressions"])
        assert agent["source"]["instructions"]["path"].endswith("prompts.ts")
    assert bundle["extractor_runtime"]["typebox"] == json.loads((runtime / "node_modules/typebox/package.json").read_text())["version"]
    assert bundle["extractor_runtime"]["typescript"] == json.loads((runtime / "node_modules/typescript/package.json").read_text())["version"]


def test_schema_matches_native_typebox_output(tmp_path, runtime):
    current_fixture(tmp_path)
    bundle = extract(tmp_path, runtime)
    script = '''
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
const require = createRequire(process.argv[1] + '/package.json');
const {Type} = await import(pathToFileURL(require.resolve('typebox')).href);
console.log(JSON.stringify(Type.Object({
  ids: Type.Array(Type.String({minLength: 1})),
  reason: Type.Optional(Type.String({description: 'optional note'}))
})));
'''
    result = subprocess.run(["node", "--input-type=module", "-e", script, str(runtime)], capture_output=True, text=True, check=True)
    assert bundle["agents"][0]["tools"][0]["parameters"] == json.loads(result.stdout)


def test_shadowed_dynamic_const_is_not_replaced_by_module_const(tmp_path, runtime):
    current_fixture(tmp_path)
    target = tmp_path / "src/agents/observer/agent.ts"
    text = target.read_text().replace("const tool: AgentTool", "const Schema = getDynamicSchema();\n  const tool: AgentTool")
    target.write_text(text)
    assert "Unsupported definition expression" in extract(tmp_path, runtime, success=False)


@pytest.mark.parametrize("replacement,insert", [
    ("{Schema}: unknown", ""),
    ("{nested: {Schema}}: unknown", ""),
    ("[Schema]: unknown", ""),
    ("{Type}: unknown", "const Local = Type.Object({value: Type.String()});"),
    ("args: unknown", "const {Type} = args; const Local = Type.Object({value: Type.String()});"),
    ("args: unknown", "let {factory: Type} = args; const Local = Type.Object({value: Type.String()});"),
    ("args: unknown", "const [Type] = args; const Local = Type.Object({value: Type.String()});"),
    ("args: unknown", "const {Schema} = args;"),
    ("args: unknown", "let {schema: Schema} = args;"),
    ("args: unknown", "const Schema = Type.Object({}); const Schema = Type.Object({});"),
])
def test_binding_patterns_and_duplicates_fail_closed(tmp_path, runtime, replacement, insert):
    current_fixture(tmp_path)
    target = tmp_path / "src/agents/observer/agent.ts"
    text = target.read_text().replace("args: unknown", replacement)
    text = text.replace("  const tool:", f"  {insert}\n  const tool:")
    if "Local" in insert:
        text = text.replace("parameters: Schema", "parameters: Local")
    target.write_text(text)
    assert "Unsupported definition expression" in extract(tmp_path, runtime, success=False)


@pytest.mark.parametrize("binding", ["Schema", "{Schema}", "{alias: Schema}", "{Type}"])
def test_catch_bindings_fail_closed(tmp_path, runtime, binding):
    current_fixture(tmp_path)
    target = tmp_path / "src/agents/observer/agent.ts"
    text = target.read_text().replace("  const tool:", f"  try {{ throw null; }} catch ({binding}) {{\n  const tool:")
    text = text.replace("  const userText", "  }\n  const userText")
    if "Type" in binding:
        text = text.replace("parameters: Schema", "parameters: Type.Object({})")
    target.write_text(text)
    assert "Unsupported definition expression" in extract(tmp_path, runtime, success=False)


@pytest.mark.parametrize("declaration", [
    "const {Schema} = external;", "let {Schema} = external;",
    "const {Type} = external;", "let {Type} = external;",
    "const Schema = Type.Object({});", "import {Type} from 'typebox';",
])
def test_duplicate_module_bindings_fail_closed(tmp_path, runtime, declaration):
    current_fixture(tmp_path)
    target = tmp_path / "src/agents/observer/agent.ts"
    target.write_text(target.read_text() + "\n" + declaration)
    error = extract(tmp_path, runtime, success=False)
    assert "Unsupported definition expression" in error or "Duplicate or shadowed declaration" in error


@pytest.mark.parametrize("import_line", [
    "import {Type} from 'not-installed-application';",
    "import {Other as Type} from 'typebox';",
    "import type {Type} from 'typebox';",
])
def test_untrusted_type_imports_fail_closed(tmp_path, runtime, import_line):
    current_fixture(tmp_path)
    target = tmp_path / "src/agents/observer/agent.ts"
    target.write_text(target.read_text().replace("import {Type} from 'typebox';", import_line))
    assert "Unsupported definition expression" in extract(tmp_path, runtime, success=False)


@pytest.mark.parametrize("export_line,success", [
    ("export {Type} from 'typebox';", True),
    ("export {Type} from 'arbitrary-module';", False),
    ("export {Other as Type} from 'typebox';", False),
    ("export type {Type} from 'typebox';", False),
    ("export {Type} from 'typebox'; export {Type} from 'typebox';", False),
    ("export {Type} from 'typebox'; export const Type = fake;", False),
])
def test_pi_ai_reexport_verified_from_source(tmp_path, runtime, export_line, success):
    source = tmp_path / "source"
    current_fixture(source)
    target = source / "src/agents/observer/agent.ts"
    target.write_text(target.read_text().replace("from 'typebox'", "from '@earendil-works/pi-ai'"))
    isolated_runtime = tmp_path / "runtime"
    isolated_runtime.mkdir()
    (isolated_runtime / "node_modules").symlink_to(runtime / "node_modules", target_is_directory=True)
    write(isolated_runtime, "packages/ai/src/index.ts", export_line + '\nthrow new Error("Do not execute re-export module");')
    result = extract(source, isolated_runtime, success=success)
    if success:
        reexport = isolated_runtime / "packages/ai/src/index.ts"
        assert result["extractor_runtime"]["type_reexport"] == {
            "path": str(reexport), "sha256": hashlib.sha256(reexport.read_bytes()).hexdigest(),
        }
    else:
        assert "Unverified pi-ai Type re-export" in result


@pytest.mark.parametrize("schema,description", [
    ("Type.Object({value: Type.Number()})", "'Record'"),
    ("getSchema()", "'Record'"),
    ("Type.Object({...external})", "'Record'"),
    ("Type.Object({value: Type.String()})", "dynamicDescription()"),
    ("Type.Object({value: Type.String()})", "`${Date.now()}`"),
])
def test_unsupported_definitions_fail_closed(tmp_path, runtime, schema, description):
    current_fixture(tmp_path, schema=schema, description=description)
    assert "Unsupported definition expression" in extract(tmp_path, runtime, success=False)


def test_unsupported_instruction_fails_closed(tmp_path, runtime):
    current_fixture(tmp_path)
    write(tmp_path, "src/agents/observer/prompts.ts", "export const OBSERVER_SYSTEM = loadPrompt();")
    assert "Unsupported definition expression" in extract(tmp_path, runtime, success=False)


def test_legacy_completion_agents_and_function_provenance(tmp_path, runtime):
    write(tmp_path, "src_v2/prompts.ts", '''const RULE = 'Exact punctuation: !';
export const OBSERVER_SYSTEM = `Observer\n${RULE}`;
export const REFLECTOR_SYSTEM = 'Reflector ' + RULE;
export const PRUNER_SYSTEM = 'Pruner ' + RULE;
''')
    write(tmp_path, "src_v2/observer.ts", '''export function runObserver(args) {
const userText = `Observation ${args.chunk}`;
completeSimple(model, {systemPrompt: OBSERVER_SYSTEM}, options);
}''')
    write(tmp_path, "src_v2/compaction.ts", '''export function runReflector(args) {
const userText = `Reflection ${args.pool}`;
completeSimple(model, {systemPrompt: REFLECTOR_SYSTEM}, options);
}
export function runPruner(args) {
const userText = `Prune ${args.pool}`;
completeSimple(model, {systemPrompt: PRUNER_SYSTEM}, options);
}''')
    bundle = extract(tmp_path, runtime, "legacy")
    assert [agent["role"] for agent in bundle["agents"]] == ["observer", "reflector", "pruner"]
    for agent in bundle["agents"]:
        assert agent["tools"] == []
        assert agent["native_call_mode"] == "text_completion"
        assert len(agent["input_builders"]) == 1
        builder = agent["input_builders"][0]
        assert "${args." in builder["expression"]
        assert builder["enclosing_function"]["name"] == f"run{agent['role'].title()}"
        assert all(item["enclosing_function"]["name"] == builder["enclosing_function"]["name"] for item in agent["runtime_expressions"])


@pytest.mark.parametrize("revision,family,roles", [
    ("58f05fa0", "current", ["observer", "reflector", "dropper"]),
    ("00d5975", "legacy", ["observer", "reflector", "pruner"]),
])
def test_frozen_real_git_revisions(tmp_path, runtime, revision, family, roles):
    if not shutil.which("git") or not (SOURCE_REPO / ".git").exists():
        pytest.skip("External Pi OM Git repository unavailable")
    archive = subprocess.run(["git", "-C", str(SOURCE_REPO), "archive", revision], capture_output=True)
    if archive.returncode:
        pytest.skip(f"External revision {revision} unavailable")
    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as source:
        source.extractall(tmp_path, filter="data")
    bundle = extract(tmp_path, runtime, family)
    assert [agent["role"] for agent in bundle["agents"]] == roles
    for agent in bundle["agents"]:
        assert agent["instructions"]
        assert len(agent["input_builders"]) == 1
        assert "${" in agent["input_builders"][0]["expression"]
        assert agent["source"]["instructions"]["start_line"] > 0
        assert agent["runtime_expressions"]
        if family == "legacy":
            assert agent["tools"] == []
            assert agent["native_call_mode"] == "text_completion"
    if family == "current":
        reexport = runtime / "packages/ai/src/index.ts"
        assert bundle["extractor_runtime"]["type_reexport"] == {
            "path": str(reexport.resolve()), "sha256": hashlib.sha256(reexport.read_bytes()).hexdigest(),
        }
        assert [agent["tools"][0]["name"] for agent in bundle["agents"]] == ["record_observations", "record_reflections", "drop_observations"]
        drop_schema = bundle["agents"][2]["tools"][0]["parameters"]
        assert drop_schema["required"] == ["ids"]
        assert drop_schema["properties"]["ids"]["minItems"] == 1
        observation = bundle["agents"][0]["tools"][0]["parameters"]["properties"]["observations"]["items"]
        assert observation["required"] == ["timestamp", "content", "relevance", "sourceEntryIds"]
        assert observation["properties"]["sourceEntryIds"]["minItems"] == 1
        assert "Exact source entry ids" in observation["properties"]["sourceEntryIds"]["description"]
        assert [item["const"] for item in observation["properties"]["relevance"]["anyOf"]] == ["low", "medium", "high", "critical"]
