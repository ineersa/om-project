<?php

declare(strict_types=1);

// Parse frozen PHP only. Do not load application classes or invoke private methods.
require $argv[2];

use PhpParser\Node;
use PhpParser\NodeFinder;
use PhpParser\ParserFactory;

function namedArguments(Node\Expr\New_ $node): array
{
    $result = [];
    foreach ($node->args as $arg) {
        if (null === $arg->name || $arg->unpack) {
            throw new RuntimeException('Only named constructor arguments are supported.');
        }
        $result[$arg->name->toString()] = $arg->value;
    }
    return $result;
}

function uniqueNode(array $nodes, string $description): Node
{
    if (count($nodes) !== 1) {
        throw new RuntimeException('Expected one '.$description.'.');
    }
    return $nodes[0];
}

function returnedExpression(Node\Stmt\ClassMethod $method): Node\Expr
{
    if (count($method->stmts ?? []) !== 1 || !$method->stmts[0] instanceof Node\Stmt\Return_
        || null === $method->stmts[0]->expr) {
        throw new RuntimeException('Expected a single literal return.');
    }
    return $method->stmts[0]->expr;
}

function literal(Node\Expr $node, string $owner, array $classes, array $seen = []): mixed
{
    if ($node instanceof Node\Scalar\String_ || $node instanceof Node\Scalar\Int_
        || $node instanceof Node\Scalar\Float_) {
        return $node->value;
    }
    if ($node instanceof Node\Expr\ConstFetch) {
        return match (strtolower($node->name->toString())) {
            'true' => true, 'false' => false, 'null' => null,
            default => throw new RuntimeException('Unsupported constant.'),
        };
    }
    if ($node instanceof Node\Expr\Array_) {
        $result = [];
        foreach ($node->items as $item) {
            if (null === $item || $item->unpack || $item->byRef) {
                throw new RuntimeException('Unsupported array item.');
            }
            $value = literal($item->value, $owner, $classes, $seen);
            if (null === $item->key) {
                $result[] = $value;
            } else {
                $key = literal($item->key, $owner, $classes, $seen);
                if ((!is_string($key) && !is_int($key)) || array_key_exists($key, $result)) {
                    throw new RuntimeException('Unsupported or repeated array key.');
                }
                $result[$key] = $value;
            }
        }
        return $result;
    }
    if ($node instanceof Node\Expr\BinaryOp\Concat) {
        $left = literal($node->left, $owner, $classes, $seen);
        $right = literal($node->right, $owner, $classes, $seen);
        if (!is_string($left) || !is_string($right)) {
            throw new RuntimeException('Expected literal string concatenation.');
        }
        return $left.$right;
    }
    if ($node instanceof Node\Expr\ClassConstFetch && $node->class instanceof Node\Name
        && $node->name instanceof Node\Identifier) {
        $class = $node->class->getLast();
        $class = $class === 'self' ? $owner : $class;
        $key = $class.'::'.$node->name->toString();
        if (isset($seen[$key]) || !isset($classes[$class])) {
            throw new RuntimeException('Unresolved or cyclic constant.');
        }
        $seen[$key] = true;
        foreach ($classes[$class]['node']->getConstants() as $statement) {
            foreach ($statement->consts as $constant) {
                if ($constant->name->toString() === $node->name->toString()) {
                    return literal($constant->value, $class, $classes, $seen);
                }
            }
        }
        throw new RuntimeException('Unresolved constant.');
    }
    throw new RuntimeException('Unsupported definition expression: '.$node->getType());
}

function provenance(Node $node, array $file): array
{
    return [
        'path' => $file['path'],
        'start_line' => $node->getStartLine(),
        'end_line' => $node->getEndLine(),
    ];
}

function sourceExpression(Node $node, array $file): array
{
    return provenance($node, $file) + [
        'kind' => 'source_expression',
        'expression' => substr($file['text'], $node->getStartFilePos(),
            $node->getEndFilePos() - $node->getStartFilePos() + 1),
    ];
}

try {
    $root = rtrim($argv[1], '/');
    $parser = (new ParserFactory())->createForNewestSupportedVersion();
    $finder = new NodeFinder();
    $classes = [];
    $iterator = new RecursiveIteratorIterator(new RecursiveDirectoryIterator($root,
        FilesystemIterator::SKIP_DOTS));
    foreach ($iterator as $file) {
        if ($file->getExtension() !== 'php') {
            continue;
        }
        $text = file_get_contents($file->getPathname());
        $ast = $parser->parse($text);
        foreach ($finder->findInstanceOf($ast, Node\Stmt\Class_::class) as $class) {
            if (null === $class->name) {
                continue;
            }
            $name = $class->name->toString();
            if (isset($classes[$name])) {
                throw new RuntimeException('Ambiguous class name.');
            }
            $classes[$name] = ['node' => $class, 'text' => $text,
                'path' => substr($file->getPathname(), strlen($root) + 1)];
        }
    }
    $agents = [];
    foreach (['observer', 'reflector', 'dropper'] as $role) {
        $prefix = ucfirst($role);
        $prompt = $classes[$prefix.'SystemPrompt'];
        $pipeline = $classes[$prefix.'Pipeline'];
        $expression = returnedExpression($prompt['node']->getMethod('text'));
        $tools = $finder->find($pipeline['node'], static fn(Node $n): bool =>
            $n instanceof Node\Expr\New_ && $n->class instanceof Node\Name
            && $n->class->getLast() === 'AgentToolDTO');
        $tool = uniqueNode($tools, 'worker tool');
        $args = namedArguments($tool);
        $schema = $args['parametersJsonSchema'];
        if ($role === 'observer') {
            if (!$schema instanceof Node\Expr\MethodCall || !$schema->var instanceof Node\Expr\Variable
                || $schema->var->name !== 'this' || !$schema->name instanceof Node\Identifier
                || $schema->name->toString() !== 'toolParametersSchema' || $schema->args !== []) {
                throw new RuntimeException('Unsupported observer schema builder.');
            }
            $schema = returnedExpression($pipeline['node']->getMethod('toolParametersSchema'));
        }
        $requests = $finder->find($pipeline['node'], static fn(Node $n): bool =>
            $n instanceof Node\Expr\New_ && $n->class instanceof Node\Name
            && $n->class->getLast() === 'AgentCallRequestDTO');
        $request = uniqueNode($requests, 'worker request');
        $requestArgs = namedArguments($request);
        $options = [];
        foreach (['maxToolCalls', 'maxDurationSeconds', 'thinkingLevel'] as $key) {
            if (isset($requestArgs[$key])) {
                $options[$key] = literal($requestArgs[$key], $prefix.'Pipeline', $classes);
            }
        }
        $runtime = [];
        foreach ($requestArgs as $key => $value) {
            if ($key !== 'tools') {
                $runtime[] = sourceExpression($value, $pipeline) + ['field' => $key];
            }
        }
        $builders = [];
        if (null !== $pipeline['node']->getMethod('buildUserInput')) {
            $builders[] = sourceExpression($pipeline['node']->getMethod('buildUserInput'), $pipeline);
        } else {
            foreach (['OmChunkPacker', 'OmSourceBlockBuilder'] as $class) {
                $builders[] = sourceExpression($classes[$class]['node'], $classes[$class]);
            }
        }
        $handlers = $finder->find($pipeline['node'], static fn(Node $n): bool =>
            $n instanceof Node\Expr\New_ && $n->class instanceof Node\Name
            && in_array($n->class->getLast(), ['RecordObservationsToolHandler',
                'RecordReflectionsToolHandler', 'DropObservationsToolHandler'], true));
        $binding = uniqueNode($handlers, 'handler construction');
        $handler = $classes[$binding->class->getLast()];
        $agents[] = [
            'role' => $role,
            'instructions' => literal($expression, $prefix.'SystemPrompt', $classes),
            'tools' => [[
                'name' => literal($args['name'], $prefix.'Pipeline', $classes),
                'description' => literal($args['description'], $prefix.'Pipeline', $classes),
                'parameters' => literal($schema, $prefix.'Pipeline', $classes),
                'handler_binding' => sourceExpression($binding, $pipeline),
                'handler_source' => sourceExpression($handler['node'], $handler),
            ]],
            'call_options' => $options,
            'source' => ['instructions' => provenance($expression, $prompt),
                'tools' => [provenance($tool, $pipeline)],
                'parameters' => provenance($schema, $pipeline)],
            'input_builders' => $builders,
            'runtime_expressions' => $runtime,
        ];
    }
    echo json_encode(['agents' => $agents, 'extractor_runtime' => [
        'php' => PHP_VERSION,
        'php_parser' => Composer\InstalledVersions::getPrettyVersion('nikic/php-parser'),
    ], 'omitted_tools' => ['recall', 'search']], JSON_THROW_ON_ERROR | JSON_UNESCAPED_UNICODE);
} catch (Throwable $error) {
    fwrite(STDERR, 'Policy extraction failed ('.get_class($error).').'.PHP_EOL);
    exit(1);
}
