# Causal Scenario Import Boundary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the causal-scenario source-string isolation test with deterministic AST dependency extraction and structured JSON-key validation.

**Architecture:** A test-only `import_references` module parses protected Python sources without importing them and emits typed static or dynamic import references. The existing boundary test consumes those references, rejects causal-scenario dependencies and unresolved recognized dynamic imports, and parses the maintained walk-forward JSON structurally.

**Tech Stack:** Python 3.12, `ast`, `importlib.util.resolve_name`, `json`, pytest, Ruff, Mypy, import-linter.

## Global Constraints

- Do not modify production runtime behavior or package APIs.
- Scan `trade_rl/rl`, `trade_rl/serving`, `trade_rl/release`, `trade_rl/workflows`, and `trade_rl/integrations`.
- Exclude `trade_rl/workflows/causal_scenario` from protected-source scanning.
- Reject `trade_rl.workflows.causal_scenario` and all submodules.
- Reject recognized dynamic imports whose target cannot be resolved from literal arguments.
- Parse `examples/binance-multitimeframe/walk-forward-full.json` as JSON and reject the exact key `causal_scenario_library` at any depth.
- Preserve the existing causal-scenario public-API smoke test.
- Production remains `NO-GO`.

---

### Task 1: AST import reference extractor

**Files:**
- Create: `tests/architecture/import_references.py`
- Create: `tests/architecture/test_import_references.py`

**Interfaces:**
- Produces: `ImportReference(path: Path, line: int, kind: str, target: str | None, unresolved: bool)`.
- Produces: `module_name_from_path(path: Path, *, package_root: Path, root_package: str) -> str`.
- Produces: `scan_import_references(path: Path, *, module_name: str) -> tuple[ImportReference, ...]`.

- [ ] **Step 1: Write failing extractor tests**

Cover ordinary imports, aliases, function-local imports, relative imports, literal and aliased `importlib.import_module`, literal and aliased `__import__`, non-literal dynamic targets, and ignored comments/docstrings/ordinary strings.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
uv run pytest -q tests/architecture/test_import_references.py
```

Expected: collection fails because `tests.architecture.import_references` does not exist.

- [ ] **Step 3: Implement typed AST extraction**

Use `ast.NodeVisitor`. Record `ast.Import` and `ast.ImportFrom` nodes at any nesting level. Resolve relative imports with `importlib.util.resolve_name` using the scanned module package. Track the four supported dynamic-import alias forms and classify calls. Literal targets produce resolved references; non-literal targets and invalid relative calls produce unresolved references.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```bash
uv run pytest -q tests/architecture/test_import_references.py
uv run ruff check tests/architecture/import_references.py tests/architecture/test_import_references.py
uv run ruff format --check tests/architecture/import_references.py tests/architecture/test_import_references.py
```

Expected: all commands pass.

- [ ] **Step 5: Commit extractor**

```bash
git add tests/architecture/import_references.py tests/architecture/test_import_references.py
git commit -m "test: extract executable Python import references"
```

### Task 2: Executable causal-scenario boundary

**Files:**
- Modify: `tests/architecture/test_causal_scenario_library_boundary.py`
- Create: `tests/architecture/test_causal_scenario_boundary_contract.py`

**Interfaces:**
- Consumes: `module_name_from_path` and `scan_import_references` from Task 1.
- Produces: stable violation formatting containing path, line, import kind, and target.

- [ ] **Step 1: Write RED boundary-contract tests**

Create temporary Python trees proving that comments and ordinary strings are ignored, while static lazy imports, literal dynamic imports, and unresolved recognized dynamic imports are rejected. Create JSON fixtures proving an ordinary value containing the phrase is accepted and the exact nested key is rejected.

- [ ] **Step 2: Run boundary tests and verify RED**

Run:

```bash
uv run pytest -q tests/architecture/test_causal_scenario_boundary_contract.py
```

Expected: failures because the existing test still performs raw substring matching and exposes no executable boundary helper.

- [ ] **Step 3: Replace the raw source scan**

Refactor the boundary test around small test-only functions:

```python
def causal_scenario_dependency_violations(
    *,
    protected_roots: tuple[Path, ...],
    excluded_root: Path,
) -> tuple[str, ...]: ...


def forbidden_json_key_paths(payload: object, *, key: str) -> tuple[str, ...]: ...
```

Scan all protected `.py` files except the excluded package. Reject prohibited resolved targets and all unresolved recognized dynamic imports. Parse the maintained JSON with `json.loads`; recursively inspect mapping keys only.

- [ ] **Step 4: Run architecture tests and verify GREEN**

Run:

```bash
uv run pytest -q tests/architecture/test_import_references.py tests/architecture/test_causal_scenario_library_boundary.py tests/architecture/test_causal_scenario_boundary_contract.py
uv run ruff check tests/architecture
uv run ruff format --check tests/architecture
```

Expected: all commands pass.

- [ ] **Step 5: Commit boundary replacement**

```bash
git add tests/architecture/test_causal_scenario_library_boundary.py tests/architecture/test_causal_scenario_boundary_contract.py
git commit -m "test: enforce causal scenario import boundary structurally"
```

### Task 3: Repository-wide verification and PR completion

**Files:**
- Modify only if verification exposes an issue in the files from Tasks 1–2.

**Interfaces:**
- Consumes: final branch exact head.
- Produces: CI evidence and a merge-ready PR.

- [ ] **Step 1: Run repository static gates**

```bash
uv run ruff check .
uv run ruff format --check --diff .
uv run mypy .
uv run lint-imports
uv run vulture trade_rl tests --min-confidence 100
```

Expected: all commands pass.

- [ ] **Step 2: Run full tests and coverage**

```bash
uv run pytest -q --cov=trade_rl --cov-branch --cov-report=term-missing --cov-report=json:coverage.json
uv run python .github/check_critical_coverage.py coverage.json pyproject.toml
uv run trade-rl --version
```

Expected: all commands pass and repository coverage remains above configured thresholds.

- [ ] **Step 3: Confirm final diff**

Confirm that only the two documents and three test-only architecture files are present. Confirm no temporary workflow or script remains.

- [ ] **Step 4: Update PR evidence and merge**

Record the exact head SHA, RED failure evidence, full test count, coverage, compatibility, Training image, and import architecture status. Mark the PR ready and Squash merge only after all exact-head checks succeed.
