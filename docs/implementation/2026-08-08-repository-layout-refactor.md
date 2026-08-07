# Repository Layout Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the repository into explicit source, application, and automation roots without changing the public Python import namespace or trading semantics.

**Architecture:** Keep the installed Python package name and imports as `trade_rl.*`, but move its physical source root to `src/trade_rl/`. Move the React/Vite frontend from the ambiguous root `studio/` directory to `apps/studio-web/`, and move repository automation from `tools/` to `scripts/ci/`. Preserve internal package boundaries for this PR; deeper package renames such as `simulation -> execution` and `workflows -> application` are intentionally deferred to a follow-up refactor after this structural move is verified.

**Tech Stack:** Python 3.12, setuptools, uv, pytest, Ruff, mypy, import-linter, GitHub Actions, React/Vite/TypeScript.

## Global Constraints

- Do not change trading, accounting, reward, execution, evaluation, or policy semantics in this refactor.
- Preserve the Python import namespace `trade_rl.*` and the CLI entry point `trade-rl`.
- Preserve read compatibility for existing artifacts and checkpoints; no schema version changes belong in this refactor.
- Do not add compatibility shim packages at the old repository paths.
- Do not lower coverage, architecture, lint, type, or CI thresholds to make the move pass.
- Keep `tests/`, `docs/`, `.github/`, `examples/`, license files, and root deployment manifests at repository level for this PR.
- Production remains NO-GO.

---

## Target repository layout

```text
.
├── src/
│   └── trade_rl/              # installed Python package; import name remains trade_rl
├── apps/
│   └── studio-web/            # React/Vite frontend
├── scripts/
│   └── ci/                    # repository/CI-only executable scripts
├── tests/                     # Python tests; internal test regrouping is a follow-up
├── docs/
├── examples/
├── .github/
├── pyproject.toml
├── Dockerfile.training
├── compose.yaml
└── compose.training.yaml
```

### Explicitly deferred package-internal target

After this PR is fully Green, a separate refactor may evaluate the following package-internal rename without mixing it into the physical-layout move:

```text
src/trade_rl/
├── domain/
├── data/
├── simulation/        # candidate later rename: execution/
├── learning/          # candidate later regrouping under training/contracts/
├── rl/                # candidate later regrouping under training/rl/
├── workflows/         # candidate later rename: application/
├── integrations/      # candidate later regrouping under infrastructure/
├── artifacts/
├── catalog/
├── serving/
├── studio/            # Python Studio API/backend, not frontend
└── ...
```

---

### Task 1: Add the repository-layout contract

**Files:**
- Create: `tests/architecture/test_repository_layout.py`
- Modify: `docs/ARCHITECTURE.md`

**Interfaces:**
- Consumes: repository root filesystem.
- Produces: a fail-closed contract forbidding the old top-level `trade_rl/`, `studio/`, and `tools/` directories and requiring `src/trade_rl/`, `apps/studio-web/`, and `scripts/ci/`.

- [ ] **Step 1: Write the failing architecture test**

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_repository_uses_explicit_source_app_and_script_roots() -> None:
    required = (
        ROOT / "src" / "trade_rl",
        ROOT / "apps" / "studio-web",
        ROOT / "scripts" / "ci",
    )
    forbidden = (
        ROOT / "trade_rl",
        ROOT / "studio",
        ROOT / "tools",
    )
    assert all(path.is_dir() for path in required)
    assert all(not path.exists() for path in forbidden)
```

- [ ] **Step 2: Run the architecture test and verify RED**

Run: `uv run pytest -q tests/architecture/test_repository_layout.py`

Expected: FAIL because the repository still uses the old physical roots.

- [ ] **Step 3: Document the physical-root contract**

Add a concise repository-layout section to `docs/ARCHITECTURE.md` explaining that physical repository roots and Python package namespaces are separate concerns.

- [ ] **Step 4: Commit the RED contract**

Commit: `test: define explicit repository layout contract`

---

### Task 2: Move the Python package to src layout

**Files:**
- Move: `trade_rl/**` -> `src/trade_rl/**`
- Modify: `pyproject.toml`
- Modify: `.github/check_critical_coverage.py` only if it contains hard-coded old physical paths.
- Modify: `.importlinter` only if physical-path assumptions exist; keep module names unchanged.
- Modify: `Dockerfile.training`
- Modify: `compose.yaml`
- Modify: `compose.training.yaml`
- Modify: workflows and scripts containing physical `trade_rl/` paths.
- Modify: architecture/documentation tests that inspect physical repository paths.

**Interfaces:**
- Consumes: existing package `trade_rl.*`.
- Produces: the same import namespace installed from `src/`.

- [ ] **Step 1: Move every package blob without changing file contents**

Move `trade_rl/` to `src/trade_rl/`; do not leave an old-path shim.

- [ ] **Step 2: Update setuptools discovery**

Use:

```toml
[tool.setuptools.packages.find]
where = ["src"]
include = ["trade_rl*"]
```

Keep:

```toml
[project.scripts]
trade-rl = "trade_rl.cli:main"

[tool.setuptools.dynamic]
version = {attr = "trade_rl._version.__version__"}
```

- [ ] **Step 3: Update Python/test/tool lookup roots**

Set mypy to `files = ["src/trade_rl"]` and pytest to include both source and repository roots:

```toml
[tool.pytest.ini_options]
pythonpath = ["src", "."]
```

Update critical-coverage physical file paths from `trade_rl/...` to `src/trade_rl/...` if the coverage JSON uses physical paths after the move. Do not change thresholds.

- [ ] **Step 4: Update build/runtime manifests**

Replace physical repository references such as `COPY trade_rl`, bind mounts, package-path probes, and workflow `paths:` filters with `src/trade_rl` equivalents. Do not change module imports such as `python -m trade_rl...`.

- [ ] **Step 5: Run focused source-layout verification**

Run:

```bash
uv sync --extra dev --extra train-sb3 --frozen
uv run python -c "import trade_rl; print(trade_rl.__file__)"
uv run ruff check src/trade_rl tests
uv run ruff format --check src/trade_rl tests
uv run mypy src/trade_rl
uv run lint-imports
uv run pytest -q tests/architecture
```

Expected: imports resolve from `src/trade_rl`, all checks pass.

- [ ] **Step 6: Commit**

Commit: `refactor: move Python package under src`

---

### Task 3: Move the Studio frontend under apps

**Files:**
- Move: `studio/**` -> `apps/studio-web/**`
- Modify: root `README.md`
- Modify: `START.md`
- Modify: `docs/**` references to frontend paths.
- Modify: `.github/workflows/**` npm commands and path filters.
- Modify: `pyproject.toml` mypy excludes if necessary.
- Modify: Docker/compose manifests if they reference the frontend path.

**Interfaces:**
- Consumes: existing Vite application.
- Produces: unchanged frontend behavior under an explicit application root.

- [ ] **Step 1: Move the frontend tree verbatim**

Move the complete Vite project from `studio/` to `apps/studio-web/`, including lockfile, configs, design assets, scripts, and source.

- [ ] **Step 2: Update repository references**

Replace repository-path commands such as:

```bash
npm ci --prefix studio
npm test --prefix studio -- --run
npm run build --prefix studio
```

with:

```bash
npm ci --prefix apps/studio-web
npm test --prefix apps/studio-web -- --run
npm run build --prefix apps/studio-web
```

Do not rename the Python package `trade_rl.studio` in this task.

- [ ] **Step 3: Run frontend verification**

Run:

```bash
npm ci --prefix apps/studio-web
npm test --prefix apps/studio-web -- --run
npm run typecheck --prefix apps/studio-web
npm run build --prefix apps/studio-web
npm run check:layout --prefix apps/studio-web
```

Expected: all pass with behavior unchanged.

- [ ] **Step 4: Commit**

Commit: `refactor: move Studio web app under apps`

---

### Task 4: Replace tools with explicit CI scripts

**Files:**
- Move: `tools/nautilus_execution_probe_digest.py` -> `scripts/ci/nautilus_execution_probe_digest.py`
- Move: `tools/run_training_capability_audit.py` -> `scripts/ci/run_training_capability_audit.py`
- Modify: `.github/workflows/nautilus-capability.yml`
- Modify: `.github/workflows/full-training-capability-audit.yml`
- Modify: any docs/tests referencing `tools/`.

**Interfaces:**
- Consumes: existing script behavior.
- Produces: unchanged CI commands under a directory whose purpose is explicit.

- [ ] **Step 1: Move both scripts verbatim**
- [ ] **Step 2: Update every caller and path filter**
- [ ] **Step 3: Verify script entry points**

Run:

```bash
uv run python scripts/ci/nautilus_execution_probe_digest.py
uv run python scripts/ci/run_training_capability_audit.py --help
```

Expected: commands behave the same as before.

- [ ] **Step 4: Commit**

Commit: `refactor: move repository automation under scripts`

---

### Task 5: Remove obsolete path-specific code and duplicate tests

**Files:**
- Modify/delete only files proven obsolete by search, architecture tests, vulture, or duplicate-contract review.

**Interfaces:**
- Consumes: the new repository layout.
- Produces: no stale compatibility aliases, path constants, or tests whose sole purpose is an old physical path.

- [ ] **Step 1: Search for old roots**

Run repository searches for exact physical-path references:

```text
trade_rl/
studio/
tools/
```

Classify each match as Python import namespace, physical repository path, documentation prose, generated lock content, or obsolete compatibility code. Never replace `trade_rl.` imports merely because the physical directory moved.

- [ ] **Step 2: Remove only proven obsolete compatibility/path helpers**

Delete code only when all callers are gone and public/runtime contracts do not depend on it.

- [ ] **Step 3: Deduplicate structural tests**

Keep one authoritative repository-layout contract. Remove redundant tests that only assert old directory names after their replacement contract is Green. Do not remove behavior, accounting, migration, or evidence tests solely to reduce test count.

- [ ] **Step 4: Run dead-code and architecture checks**

Run:

```bash
uv run vulture src/trade_rl --min-confidence 100
uv run lint-imports
uv run pytest -q tests/architecture
```

- [ ] **Step 5: Commit**

Commit: `refactor: remove obsolete repository-path code`

---

### Task 6: Update documentation and contributor entry points

**Files:**
- Modify: `README.md`
- Modify: `START.md`
- Modify: `docs/README.md`
- Modify: `docs/ARCHITECTURE.md`
- Modify: relevant operations docs and Studio docs.

**Interfaces:**
- Produces: a contributor can identify the Python engine, web app, CI scripts, tests, and docs without guessing directory semantics.

- [ ] **Step 1: Add a short repository map to README**

Document:

```text
src/trade_rl      Python engine/backend package
apps/studio-web   Research web application
scripts/ci        CI and verification utilities
tests             Python verification suites
docs              Architecture, operations, and research documentation
```

- [ ] **Step 2: Update all setup commands and links**
- [ ] **Step 3: Verify the documentation contract tests**
- [ ] **Step 4: Commit**

Commit: `docs: document repository responsibility roots`

---

### Task 7: Final verification and self-review

**Files:** none unless verification finds defects.

- [ ] **Step 1: Review the full rename diff**

Confirm that moved Python files are content-identical unless a physical-path reference required an update. Confirm that no functional trading changes are mixed in.

- [ ] **Step 2: Run complete Python verification**

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src/trade_rl
uv run lint-imports
uv run pytest --cov=trade_rl --cov-branch
```

- [ ] **Step 3: Run critical coverage without reducing thresholds**
- [ ] **Step 4: Run Studio verification**

```bash
npm test --prefix apps/studio-web -- --run
npm run typecheck --prefix apps/studio-web
npm run build --prefix apps/studio-web
npm run check:layout --prefix apps/studio-web
```

- [ ] **Step 5: Run package/build and optional capability CI**

Verify the training image, Ubuntu/Windows compatibility, Nautilus Capability, PostgreSQL Catalog, package identity, and any workflow path filters affected by the move.

- [ ] **Step 6: Final reviewer pass**

Check naming, dependency direction, duplicate code, stale paths, generated files, secrets, documentation links, test ownership, and CI path filters. Fix findings and rerun affected focused tests followed by the complete verification on one final HEAD.

- [ ] **Step 7: Keep the PR draft until every required check is Green**
