# Causal Scenario Import Boundary Design

## Goal

Replace the brittle source-string boundary check around `trade_rl.workflows.causal_scenario` with executable dependency analysis that understands Python imports, bounded lazy-import data flow, and structured configuration.

## Scope

The protected runtime and training roots remain:

- `trade_rl/rl`
- `trade_rl/serving`
- `trade_rl/release`
- `trade_rl/workflows`
- `trade_rl/integrations`

Files beneath `trade_rl/workflows/causal_scenario` are excluded because they are the isolated implementation itself. Every other Python module beneath the protected roots must remain unable to depend on `trade_rl.workflows.causal_scenario` or any submodule.

The maintained walk-forward example `examples/binance-multitimeframe/walk-forward-full.json` must not contain a `causal_scenario_library` configuration key at any nesting depth.

## Chosen approach

Add a reusable test-only AST scanner under `tests/architecture` and use it from the existing causal-scenario boundary test.

The scanner emits typed import references from:

- `import package` and aliased imports;
- `from package import name`, including relative imports resolved against the scanned module;
- imports placed inside functions, methods, conditionals, or `TYPE_CHECKING` blocks;
- `importlib.import_module(...)`, including aliases imported from `importlib`;
- built-in `__import__(...)`, including aliases imported from `builtins`.

Comments, docstrings, and ordinary string literals are not dependencies and do not create edges.

## Finite lazy-import data flow

The repository intentionally uses lazy package exports so optional frameworks are not imported at package initialization time. Two maintained patterns compute the module name from immutable module-level maps before calling `import_module`.

Treating every non-literal call argument as unsafe produced false positives in `trade_rl.integrations.__init__` and `trade_rl.rl.__init__`. The scanner therefore performs a deliberately bounded string-set analysis.

It propagates finite string candidates through:

- literal assignments and annotated assignments;
- tuples, lists, sets, and dictionaries;
- tuple/list unpacking;
- dictionary `.get()`, `.items()`, `.keys()`, and `.values()` calls;
- dictionary, list, set, and generator comprehensions;
- subscripts, attributes, starred expressions, conditional expressions, and boolean expressions.

Each function, async function, and class receives a copy of the enclosing finite-value environment. The environment is restored when leaving that scope.

Candidate strings are filtered to syntactically valid module names. When dotted candidates exist, undotted export names are removed, so a lazy export map yields module edges rather than exported attribute names.

This is an over-approximation: every finite candidate module becomes an import reference. Calls whose target still depends on an arbitrary function result remain unresolved and fail closed.

## Why this approach

### Rejected: keep string matching

String matching treats comments and documentation as dependencies, misses aliases and computed import forms, and cannot distinguish JSON keys from values.

### Rejected: import every protected module at runtime

Import-time interception detects only code executed during module import. It misses lazy imports inside functions and can trigger framework initialization or other side effects.

### Rejected: allowlist the two existing lazy imports

Path-specific exceptions would preserve the current implementation but would not prove what modules the lazy maps can load. A future prohibited target could be hidden behind the same exception.

### Selected: AST extraction plus bounded finite data flow

AST analysis covers imports regardless of control-flow position without importing application modules. Finite string propagation resolves maintained lazy-export maps while still failing closed for unknown dynamic imports. The process is deterministic, cross-platform, and side-effect free.

## Scanner contract

`tests/architecture/import_references.py` provides:

- `ImportReference`: immutable record containing source path, line number, target module, import kind, and resolution status;
- `scan_import_references(path: Path, *, module_name: str) -> tuple[ImportReference, ...]`;
- `module_name_from_path(path: Path, *, package_root: Path, root_package: str) -> str`;
- `causal_scenario_dependency_violations(...) -> tuple[str, ...]`;
- `forbidden_json_key_paths(payload: object, *, key: str) -> tuple[str, ...]`.

Static relative imports use normal Python package semantics. Invalid relative imports are returned as unresolved instead of being silently ignored.

Recognized dynamic-import functions accept literal targets or targets reducible to a finite string set. Relative `importlib.import_module` calls also require a literal or finite package candidate. A target that cannot be reduced remains unresolved.

## Boundary test contract

`tests/architecture/test_causal_scenario_library_boundary.py`:

1. keeps the existing public-API smoke assertions;
2. scans every protected Python file except the causal-scenario package;
3. rejects any resolved target equal to `trade_rl.workflows.causal_scenario` or beginning with that prefix plus `.`;
4. rejects every unresolved recognized dynamic import in protected roots;
5. parses the maintained JSON example with `json.loads`;
6. recursively rejects a mapping key exactly equal to `causal_scenario_library`;
7. reports path, line, import kind, and target in a stable sorted violation list.

## Error handling

A Python syntax error fails through `ast.parse`.

Malformed JSON fails through `json.loads` instead of being treated as a harmless absence.

Unknown dynamic imports fail closed because their dependency cannot be proven safe.

Finite candidate sets are expanded rather than selecting one candidate, preventing a prohibited member from being hidden among allowed modules.

## Test strategy

Focused scanner tests cover:

- ordinary and aliased imports;
- lazy imports inside functions;
- relative imports;
- literal and aliased `importlib.import_module` and `__import__` calls;
- assigned literal targets;
- maintained finite lazy-export map patterns;
- unresolved function-derived targets;
- comments, docstrings, and ordinary strings producing no references.

Boundary regression tests prove that:

- comments and ordinary strings containing the prohibited module do not fail;
- static lazy imports and literal dynamic imports of the prohibited module fail;
- unresolved recognized dynamic imports fail closed;
- the causal-scenario implementation package itself is excluded;
- JSON values containing the phrase are allowed while the exact key is rejected;
- nested JSON-key paths are stable and deterministic.

## Non-goals

- No production package or runtime behavior changes.
- No general-purpose Python security scanner.
- No arbitrary symbolic execution.
- No execution of protected application modules during analysis.
- No change to the causal-scenario public API.
- Production remains `NO-GO`.
