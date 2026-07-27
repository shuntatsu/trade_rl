# Causal Scenario Import Boundary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for completed work.

**Goal:** Replace the causal-scenario source-string isolation test with deterministic AST dependency extraction, bounded lazy-import data flow, and structured JSON-key validation.

**Architecture:** A test-only `tests.architecture.import_references` module parses protected Python sources without importing them and emits typed static or dynamic import references. The boundary test consumes those references, rejects causal-scenario dependencies and unresolved recognized dynamic imports, and parses the maintained walk-forward JSON structurally.

**Tech Stack:** Python 3.12, `ast`, `importlib.util.resolve_name`, `json`, pytest, Ruff, Mypy, import-linter.

## Global Constraints

- Do not modify production runtime behavior or package APIs.
- Scan `trade_rl/rl`, `trade_rl/serving`, `trade_rl/release`, `trade_rl/workflows`, and `trade_rl/integrations`.
- Exclude `trade_rl/workflows/causal_scenario` from protected-source scanning.
- Reject `trade_rl.workflows.causal_scenario` and all submodules.
- Resolve recognized dynamic imports when their targets reduce to finite constant candidates.
- Reject recognized dynamic imports whose targets remain unknown.
- Parse `examples/binance-multitimeframe/walk-forward-full.json` as JSON and reject the exact key `causal_scenario_library` at any depth.
- Preserve the existing causal-scenario public-API smoke test.
- Production remains `NO-GO`.

---

### Task 1: AST import reference extractor

**Files:**
- Create: `tests/architecture/__init__.py`
- Create: `tests/architecture/import_references.py`
- Create: `tests/architecture/test_import_references.py`

**Interfaces:**
- Produces: `ImportReference(path: Path, line: int, kind: str, target: str | None, unresolved: bool)`.
- Produces: `module_name_from_path(path: Path, *, package_root: Path, root_package: str) -> str`.
- Produces: `scan_import_references(path: Path, *, module_name: str) -> tuple[ImportReference, ...]`.

- [x] **Step 1: Write failing extractor tests**

Covered ordinary imports, aliases, function-local imports, relative imports, literal and aliased `importlib.import_module`, literal and aliased `__import__`, unresolved dynamic targets, and ignored comments/docstrings/ordinary strings.

- [x] **Step 2: Verify RED**

The focused contracts failed because the scanner module did not exist. An initial repository-root placement was rejected after CI showed that pytest did not expose that module path; the helper was moved into the explicit `tests.architecture` package.

- [x] **Step 3: Implement typed AST extraction**

Implemented `ast.NodeVisitor` extraction for `ast.Import`, `ast.ImportFrom`, relative import resolution, dynamic import aliases, source locations, and unresolved-reference reporting.

- [x] **Step 4: Verify scanner GREEN**

Scanner contracts, Ruff, format, Mypy, import architecture, both compatibility jobs, Training image, full tests, coverage, and CLI gates passed before boundary integration.

### Task 2: Executable causal-scenario boundary

**Files:**
- Modify: `tests/architecture/test_causal_scenario_library_boundary.py`
- Create: `tests/architecture/test_causal_scenario_boundary_contract.py`
- Extend: `tests/architecture/import_references.py`

**Interfaces:**
- Produces: `causal_scenario_dependency_violations(...) -> tuple[str, ...]`.
- Produces: `forbidden_json_key_paths(payload: object, *, key: str) -> tuple[str, ...]`.

- [x] **Step 1: Write RED boundary-contract tests**

Created temporary protected trees proving comments and ordinary strings are ignored, static lazy imports and literal dynamic imports are rejected, unresolved recognized dynamic imports fail closed, the isolated package is excluded, and exact nested JSON keys are detected.

- [x] **Step 2: Verify boundary RED**

The focused boundary test failed because the two boundary functions did not exist.

- [x] **Step 3: Implement structural boundary helpers**

Implemented protected-root scanning, stable path/line/kind/target diagnostics, exact prefix matching, unresolved dynamic-import rejection, and deterministic recursive JSON-key paths.

- [x] **Step 4: Replace raw source and JSON substring matching**

The maintained boundary test now uses AST dependency references and `json.loads`. The causal-scenario public API smoke remains unchanged.

- [x] **Step 5: Verify focused boundary GREEN**

Focused scanner and boundary contracts passed with repository static and runtime smoke gates.

### Task 3: Resolve maintained finite lazy-export maps

**Files:**
- Extend: `tests/architecture/import_references.py`
- Extend: `tests/architecture/test_import_references.py`

- [x] **Step 1: Capture real-repository RED evidence**

The structural boundary correctly reached the repository but initially reported two unresolved imports in `trade_rl.integrations.__init__` and `trade_rl.rl.__init__`. Both calls derive module names from finite immutable lazy-export maps.

- [x] **Step 2: Add focused finite-data-flow contracts**

Added RED tests for assigned literal targets, tuple-valued export maps, dictionary comprehensions that flatten module maps, tuple unpacking, `.get()`, and truly unknown function-derived targets.

- [x] **Step 3: Implement bounded string-set propagation**

Propagated finite strings through literals, collections, mappings, comprehensions, selected mapping methods, subscripts, attributes, starred values, conditionals, boolean expressions, assignments, named expressions, and scoped function/class bodies.

Module-looking candidates are over-approximated and expanded into separate edges. Arbitrary function results remain unresolved.

- [x] **Step 4: Preserve fail-closed behavior**

No path-specific allowlist was added. A prohibited module added to either lazy-export map would become a resolved prohibited edge. An unknown dynamic target remains an unresolved violation.

### Task 4: Repository-wide verification and PR completion

**Final diff:**
- `docs/superpowers/specs/2026-07-27-causal-scenario-import-boundary-design.md`
- `docs/superpowers/plans/2026-07-27-causal-scenario-import-boundary.md`
- `tests/architecture/__init__.py`
- `tests/architecture/import_references.py`
- `tests/architecture/test_import_references.py`
- `tests/architecture/test_causal_scenario_boundary_contract.py`
- `tests/architecture/test_causal_scenario_library_boundary.py`

- [x] **Step 1: Confirm no production runtime change**

All implementation changes are test-only. No `trade_rl` production source or example configuration is modified.

- [x] **Step 2: Remove temporary/misplaced files**

The temporary repository-root scanner was deleted. No temporary workflow or script is part of the final diff.

- [x] **Step 3: Run exact-head static and environment gates**

Ruff, format, Mypy, import architecture, dead-code report, workflow security, Studio frontend verification, fixed viewport verification, Recovery/Serving smoke, Ubuntu compatibility, Windows compatibility, and the complete Training image packaged non-root probe passed on implementation head `f3cc859027a293833ac36f93c72db4144b66d3c6`.

- [x] **Step 4: Run exact-head tests and coverage**

The implementation head completed `2031 passed, 2 skipped` in 99.54 seconds. Total coverage was `85.89%`; branch coverage was `74.32%`. Critical branch coverage and CLI smoke passed.

- [x] **Step 5: Confirm branch relationship and final diff**

The implementation head was behind main by `0`, and the comparison contained exactly the seven files listed above.

- [x] **Step 6: Record evidence and merge**

Update PR #222 with the final docs-only head, repeat exact-head CI, mark ready, and Squash merge only after every check succeeds. Production remains `NO-GO`.
