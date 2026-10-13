#!/usr/bin/env node
// Parse frozen source only. Never import or execute application code.
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';

async function main() {
  const [rootArg, runtimeArg, family] = process.argv.slice(2);
  if (!rootArg || !runtimeArg || !['current', 'legacy'].includes(family) || process.argv.length !== 5) {
    throw new Error('Usage: node extract_pi.mjs SOURCE_ROOT RUNTIME_REPO current|legacy');
  }
  const root = path.resolve(rootArg);
  const require = createRequire(path.join(path.resolve(runtimeArg), 'package.json'));
  const ts = require('typescript');
  const typeboxEntry = require.resolve('typebox');
  const { Type } = await import(pathToFileURL(typeboxEntry).href);
  let packageDir = path.dirname(typeboxEntry);
  while (!fs.existsSync(path.join(packageDir, 'package.json'))) {
    const parent = path.dirname(packageDir);
    if (parent === packageDir) throw new Error('Cannot locate TypeBox package metadata');
    packageDir = parent;
  }
  const typeboxVersion = JSON.parse(fs.readFileSync(path.join(packageDir, 'package.json'), 'utf8')).version;
  const factories = new Set(['Object', 'Array', 'Union', 'Literal', 'String', 'Optional']);
  let typeReexport;

  function verifyTypeReexport() {
    if (typeReexport) return;
    const file = path.join(path.resolve(runtimeArg), 'packages/ai/src/index.ts');
    const bytes = fs.readFileSync(file);
    const source = ts.createSourceFile(file, bytes.toString('utf8'), ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);
    const exports = [];
    for (const statement of source.statements) {
      if (!ts.isExportDeclaration(statement) || !statement.exportClause || !ts.isNamedExports(statement.exportClause)) continue;
      for (const element of statement.exportClause.elements) {
        if (element.name.text === 'Type') exports.push({ statement, element });
      }
    }
    if (source.parseDiagnostics.length || exports.length !== 1 || scopeBindings(source, 'Type').length) throw new Error('Unverified pi-ai Type re-export');
    const { statement, element } = exports[0];
    if (statement.isTypeOnly || element.isTypeOnly || !statement.moduleSpecifier || !ts.isStringLiteral(statement.moduleSpecifier)
      || statement.moduleSpecifier.text !== 'typebox' || (element.propertyName && element.propertyName.text !== 'Type')) {
      throw new Error('Unverified pi-ai Type re-export');
    }
    typeReexport = { path: file, sha256: createHash('sha256').update(bytes).digest('hex') };
  }

  function parse(relative) {
    const text = fs.readFileSync(path.join(root, relative), 'utf8');
    const source = ts.createSourceFile(relative, text, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);
    if (source.parseDiagnostics.length) throw new Error(`Invalid TypeScript: ${relative}`);
    return source;
  }
  function walk(node, visit) {
    visit(node);
    ts.forEachChild(node, child => walk(child, visit));
  }
  function provenance(node) {
    const source = node.getSourceFile();
    return {
      path: source.fileName,
      start_line: source.getLineAndCharacterOfPosition(node.getStart(source)).line + 1,
      end_line: source.getLineAndCharacterOfPosition(node.end).line + 1,
    };
  }
  function enclosing(node) {
    for (let parent = node.parent; parent; parent = parent.parent) {
      if (ts.isFunctionDeclaration(parent) || ts.isArrowFunction(parent) || ts.isFunctionExpression(parent)) {
        return { name: parent.name?.getText() ?? null, ...provenance(parent) };
      }
    }
    return null;
  }
  function expression(node, name) {
    const owner = ts.isFunctionDeclaration(node) ? { name: node.name?.text ?? null, ...provenance(node) } : enclosing(node);
    return { ...provenance(node), expression: node.getText(), kind: 'source_expression', name, enclosing_function: owner };
  }
  function bindingNames(name) {
    if (ts.isIdentifier(name)) return [name.text];
    if (ts.isObjectBindingPattern(name) || ts.isArrayBindingPattern(name)) {
      return name.elements.flatMap(element => ts.isBindingElement(element) ? bindingNames(element.name) : []);
    }
    return [];
  }
  function scopeBindings(scope, name) {
    const result = [];
    function declaration(node, kind) {
      if (bindingNames(node.name).includes(name)) result.push({ node, kind });
    }
    function variables(list) {
      for (const node of list.declarations) declaration(node, list.flags & ts.NodeFlags.Const ? 'const' : 'mutable');
    }
    if (ts.isSourceFile(scope) || ts.isBlock(scope)) {
      for (const statement of scope.statements) {
        if (ts.isVariableStatement(statement)) variables(statement.declarationList);
        else if (ts.isImportDeclaration(statement) && statement.importClause) {
          if (statement.importClause.name?.text === name) result.push({ node: statement.importClause, kind: 'unsupported' });
          const named = statement.importClause.namedBindings;
          if (named && ts.isNamedImports(named)) {
            for (const node of named.elements) if (node.name.text === name) result.push({ node, kind: 'import', statement });
          } else if (named?.name.text === name) result.push({ node: named, kind: 'unsupported' });
        } else if (statement.name && bindingNames(statement.name).includes(name)) result.push({ node: statement, kind: 'unsupported' });
      }
    }
    if (ts.isFunctionLike(scope)) {
      for (const parameter of scope.parameters) declaration(parameter, 'unsupported');
      if (scope.name?.text === name) result.push({ node: scope, kind: 'unsupported' });
      // var declarations are function-scoped, including those in nested blocks.
      function hoisted(node) {
        if (node !== scope && ts.isFunctionLike(node)) return;
        if (ts.isVariableDeclarationList(node) && !(node.flags & ts.NodeFlags.BlockScoped)) variables(node);
        ts.forEachChild(node, hoisted);
      }
      hoisted(scope);
    }
    if (ts.isCatchClause(scope) && scope.variableDeclaration) declaration(scope.variableDeclaration, 'unsupported');
    if ((ts.isForStatement(scope) || ts.isForInStatement(scope) || ts.isForOfStatement(scope))
      && scope.initializer && ts.isVariableDeclarationList(scope.initializer)) variables(scope.initializer);
    return result;
  }
  function constants(source) {
    const result = new Map();
    // Only module-scope const declarations are eligible for evaluation.
    for (const statement of source.statements) {
      if (!ts.isVariableStatement(statement) || !(statement.declarationList.flags & ts.NodeFlags.Const)) continue;
      for (const declaration of statement.declarationList.declarations) {
        if (ts.isIdentifier(declaration.name) && declaration.initializer) {
          if (scopeBindings(source, declaration.name.text).length !== 1) throw new Error(`Duplicate or shadowed declaration: ${declaration.name.text}`);
          result.set(declaration.name.text, declaration.initializer);
        }
      }
    }
    return result;
  }
  function evaluator(source) {
    const active = new Set();
    function bindingFor(node, name = node.text) {
      for (let scope = node.parent; scope; scope = scope.parent) {
        const matches = scopeBindings(scope, name);
        if (matches.length > 1) return fail(node);
        if (matches.length) return matches[0];
        if (scope === source) break;
      }
      return null;
    }
    function constantFor(node) {
      const binding = bindingFor(node);
      if (!binding || binding.kind !== 'const' || !ts.isIdentifier(binding.node.name) || !binding.node.initializer) return fail(node);
      return binding.node.initializer;
    }
    function approveType(node) {
      const binding = bindingFor(node.expression.expression, 'Type');
      if (!binding || binding.kind !== 'import') return fail(node);
      const { statement, node: imported } = binding;
      if (statement.importClause.isTypeOnly || imported.isTypeOnly || (imported.propertyName && imported.propertyName.text !== 'Type')
        || !ts.isStringLiteral(statement.moduleSpecifier)) return fail(node);
      switch (statement.moduleSpecifier.text) {
        case 'typebox': return;
        case '@earendil-works/pi-ai': verifyTypeReexport(); return;
        default: return fail(node);
      }
    }
    function fail(node) {
      throw new Error(`Unsupported definition expression at ${source.fileName}:${provenance(node).start_line}: ${node.getText()}`);
    }
    function evaluate(node) {
      if (ts.isParenthesizedExpression(node)) return evaluate(node.expression);
      if (ts.isStringLiteral(node) || ts.isNoSubstitutionTemplateLiteral(node)) return node.text;
      if (ts.isNumericLiteral(node)) {
        const value = Number(node.text);
        if (!Number.isFinite(value)) return fail(node);
        return value;
      }
      if (node.kind === ts.SyntaxKind.TrueKeyword) return true;
      if (node.kind === ts.SyntaxKind.FalseKeyword) return false;
      if (node.kind === ts.SyntaxKind.NullKeyword) return null;
      if (ts.isIdentifier(node)) {
        if (active.has(node.text)) throw new Error(`Cyclic const reference: ${node.text}`);
        active.add(node.text);
        try { return evaluate(constantFor(node)); } finally { active.delete(node.text); }
      }
      if (ts.isBinaryExpression(node) && node.operatorToken.kind === ts.SyntaxKind.PlusToken) {
        const left = evaluate(node.left), right = evaluate(node.right);
        if (typeof left !== 'string' || typeof right !== 'string') return fail(node);
        return left + right;
      }
      if (ts.isTemplateExpression(node)) {
        let value = node.head.text;
        for (const span of node.templateSpans) {
          const part = evaluate(span.expression);
          if (typeof part !== 'string') return fail(span.expression);
          value += part + span.literal.text;
        }
        return value;
      }
      if (ts.isArrayLiteralExpression(node)) return node.elements.map(evaluate);
      if (ts.isObjectLiteralExpression(node)) {
        const result = {};
        for (const property of node.properties) {
          if (!ts.isPropertyAssignment(property) || !(ts.isIdentifier(property.name) || ts.isStringLiteral(property.name))) return fail(property);
          const key = property.name.text;
          if (Object.hasOwn(result, key)) return fail(property);
          Object.defineProperty(result, key, { value: evaluate(property.initializer), enumerable: true, configurable: true, writable: true });
        }
        return result;
      }
      if (ts.isCallExpression(node) && ts.isPropertyAccessExpression(node.expression)
          && ts.isIdentifier(node.expression.expression) && node.expression.expression.text === 'Type'
          && factories.has(node.expression.name.text) && !node.typeArguments?.length) {
        approveType(node);
        return Type[node.expression.name.text](...node.arguments.map(evaluate));
      }
      return fail(node);
    }
    return evaluate;
  }
  function requiredConstant(source, name) {
    const node = constants(source).get(name);
    if (!node) throw new Error(`Missing ${name} in ${source.fileName}`);
    const value = evaluator(source)(node);
    if (typeof value !== 'string') throw new Error(`${name} must be a string`);
    return { value, node };
  }
  function runtimeDeclarations(source, functionName = null) {
    const inputs = [], runtime = [];
    for (const statement of source.statements) {
      if (ts.isFunctionDeclaration(statement) && (!functionName || !statement.name?.text.startsWith('run') || statement.name.text === functionName)) {
        runtime.push(expression(statement, statement.name?.text ?? 'function'));
      }
    }
    walk(source, node => {
      if (functionName && enclosing(node)?.name !== functionName) return;
      if (ts.isVariableDeclaration(node) && node.initializer) {
        const name = node.name.getText();
        (name === 'userText' ? inputs : runtime).push(expression(node.initializer, name));
      }
      if (ts.isCallExpression(node)) {
        runtime.push(expression(node, 'call'));
      }
    });
    if (!inputs.length) throw new Error(`Missing userText in ${source.fileName}${functionName ? ':' + functionName : ''}`);
    return { input_builders: inputs, runtime_expressions: runtime };
  }
  const agents = [];
  for (const role of family === 'current' ? ['observer', 'reflector', 'dropper'] : ['observer', 'reflector', 'pruner']) {
    const promptSource = parse(family === 'current' ? `src/agents/${role}/prompts.ts` : 'src_v2/prompts.ts');
    const instruction = requiredConstant(promptSource, `${role.toUpperCase()}_SYSTEM`);
    const source = parse(family === 'current' ? `src/agents/${role}/agent.ts` : `src_v2/${role === 'observer' ? 'observer' : 'compaction'}.ts`);
    const tools = [], toolSources = [];
    if (family === 'current') {
      const evaluate = evaluator(source);
      walk(source, node => {
        if (!ts.isVariableDeclaration(node) || !node.type || !ts.isTypeReferenceNode(node.type) || node.type.typeName.getText() !== 'AgentTool') return;
        if (!node.initializer || !ts.isObjectLiteralExpression(node.initializer)) throw new Error('AgentTool must have an inline object initializer');
        const properties = new Map();
        for (const property of node.initializer.properties) {
          if (!ts.isPropertyAssignment(property) || !ts.isIdentifier(property.name)) throw new Error('Unsupported AgentTool property');
          if (properties.has(property.name.text)) throw new Error('Duplicate AgentTool property');
          properties.set(property.name.text, property.initializer);
        }
        const tool = {};
        for (const key of ['name', 'description', 'parameters']) {
          if (!properties.has(key)) throw new Error(`AgentTool missing ${key}`);
          tool[key] = evaluate(properties.get(key));
        }
        if (typeof tool.name !== 'string' || typeof tool.description !== 'string' || !tool.parameters || typeof tool.parameters !== 'object') throw new Error('Invalid AgentTool definition');
        tool.source = provenance(node);
        if (properties.has('execute')) tool.execute_source = expression(properties.get('execute'), 'execute');
        tools.push(tool);
        toolSources.push(provenance(node));
      });
      if (!tools.length) throw new Error(`Missing AgentTool in ${source.fileName}`);
    }
    agents.push({
      role, instructions: instruction.value, tools,
      source: { instructions: provenance(instruction.node), tools: toolSources },
      ...runtimeDeclarations(source, family === 'legacy' ? `run${role[0].toUpperCase()}${role.slice(1)}` : null),
      ...(family === 'legacy' ? { native_call_mode: 'text_completion' } : {}),
    });
  }
  process.stdout.write(JSON.stringify({ agents, extractor_runtime: { typescript: ts.version, typebox: typeboxVersion, ...(typeReexport ? { type_reexport: typeReexport } : {}) }, omitted: family === 'current' ? ['User-facing recall tool; only memory worker tools are extracted.'] : [] }) + '\n');
}
main().catch(error => { process.stderr.write(`${error.message}\n`); process.exitCode = 1; });
