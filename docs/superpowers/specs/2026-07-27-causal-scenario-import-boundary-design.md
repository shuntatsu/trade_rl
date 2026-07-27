# Causal Scenario Import Boundary Design

## Goal

Replace the brittle source-string boundary check around `trade_rl.workflows.causal_scenario` with an executable dependency analysis that understands Python imports and structured configuration.

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

Add a small reusable AST scanner under `tests/architecture` and use it from the existing causal-scenario boundary test.

The scanner emits typed import references from:

- `import package` and aliased imports;
- `from package import name`, including relative imports resolved against the scanned module;
- imports placed inside functions, methods, conditionals, or `TYPE_CHECKING` blocks;
- `importlib.import_module(...)`, including aliases imported from `importlib`;
- built-in `__import__(...)`, including aliases imported from `builtins`.

Comments, docstrings, and ordinary string literals are not dependencies and must not create edges.

For recognized dynamic-import functions, the target must be a literal string. A non-literal target is reported as unresolved and fails the protected-root test. Relative `importlib.import_module` calls must also provide a literal package name so the target can be resolved deterministically.

## Why this approach

### Rejected: keep string matching

String matching is simple but treats comments and documentation as dependencies, misses aliases and computed import forms, and cannot distinguish JSON keys from values.

### Rejected: import every protected module at runtime

Import-time interception would detect only code executed during module import. It would miss lazy imports inside functions and could trigger framework initialization or other side effects.

### Selected: static AST dependency extraction

AST analysis covers imports regardless of control-flow position without importing application modules. It is deterministic, cross-platform, side-effect free, and can fail closed for recognized unresolved dynamic-import calls.

## Scanner contract

Create `tests/architecture/import_references.py` with:

- `ImportReference`: immutable record containing source path, line number, target module, import kind, and whether resolution failed;
- `scan_import_references(path: Path, *, module_name: str) -> tuple[ImportReference, ...]`;
- `module_name_from_path(path: Path, *, package_root: Path, root_package: str) -> str`.

The scanner tracks aliases introduced by `import importlib as ...`, `from importlib import import_module as ...`, `import builtins as ...`, and `from builtins import __import__ as ...` before classifying calls.

Static relative imports are resolved with normal Python package semantics. An invalid relative import is returned as unresolved instead of being silently ignored.

## Boundary test contract

Rewrite `tests/architecture/test_causal_scenario_library_boundary.py` so it:

1. keeps the existing public-API smoke assertions;
2. scans every protected Python file except the causal-scenario package;
3. rejects any resolved target equal to `trade_rl.workflows.causal_scenario` or beginning with that prefix plus `.`;
4. rejects every unresolved recognized dynamic import in the protected roots;
5. parses the maintained JSON example with `json.loads` and recursively rejects a mapping key exactly equal to `causal_scenario_library`;
6. reports path, line, kind, and target in a stable sorted violation list.

## Error handling

A Python syntax error in a scanned source file fails the test naturally through `ast.parse`.

Malformed JSON fails the test through `json.loads` instead of being treated as a harmless absence.

Dynamic imports with non-literal targets fail closed because their dependency cannot be proven safe.

## Test strategy

Add focused unit tests for the scanner covering:

- ordinary and aliased imports;
- lazy imports inside functions;
- relative imports;
- literal `importlib.import_module` and `__import__` calls;
- aliased dynamic-import functions;
- unresolved non-literal dynamic imports;
- comments, docstrings, and ordinary strings producing no references.

Add boundary-test regression fixtures proving that:

- a comment containing the prohibited module no longer fails;
- a lazy import and a dynamic import of the prohibited module do fail;
- a JSON value containing the phrase is allowed while the exact key is rejected.

## Non-goals

- No production package or runtime behavior changes.
- No general-purpose Python security scanner.
- No execution of protected application modules during analysis.
- No change to the causal-scenario public API.
- Production remains `NO-GO`.
