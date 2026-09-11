# evaluation/runs Capability Facade Implementation Plan

Status: Active

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `trade_rl.evaluation.runs` the single Tier-2 cross-capability entry point for Run Core contracts while preserving existing implementation owners, runtime behavior, persisted artifacts, CLI behavior, and top-level evaluation API.

**Architecture:** Keep `runs/artifact.py`, `candidate_suite.py`, `config.py`, `execute.py`, and `provenance.py` as implementation owners. Add a pure re-export facade in `runs/__init__.py`, migrate production callers outside `runs/` to that facade, and add a direct-import AST contract that is separate from the existing re-export-aware semantic dependency collector.

**Tech Stack:** Python 3.12, NumPy, pytest, Ruff, Mypy, `tests.architecture.imports.ImportCollector`, GitHub Actions, uv.

**Spec:** `docs/specs/2026-09-11-runs-capability-facade-design.md`

## Global Constraints

- Do not move, merge, rename, or delete existing `evaluation/runs/` owner modules.
- Do not add business logic, wrappers, error translation, or side effects to `runs/__init__.py`.
- Do not expand `trade_rl.evaluation.__all__`.
- Do not expose `load_candidate_run_config`, `runtime_environment_manifest`, `PROVENANCE_SCHEMA`, CLI `main`, or `run_candidate_artifact` through the Tier-2 facade.
- Preserve candidate result/artifact/provenance schemas, exact-file evidence, semantic artifact digest, CLI behavior, and historical readers.
- Preserve existing `ImportCollector.collect()` semantics; add `collect_direct()` separately.
- Production source outside `trade_rl/evaluation/runs/` must not directly import `runs.artifact`, `runs.candidate_suite`, `runs.config`, `runs.execute`, or `runs.provenance` after migration.
- Tests may still import internal owner modules for implementation-level verification.
- New source bytes legitimately change implementation provenance; do not weaken provenance to hide this.
- Use TDD and keep spec/plan Active only while this work is in progress.

---

### Task 1: Direct-import inspection contract

**Files:**
- Modify: `tests/architecture/imports.py`
- Modify: `tests/architecture/test_import_collector_contract.py`

**Interfaces:**
- Produces: `ImportCollector.collect_direct(path: Path) -> set[str]`.
- Preserves: `ImportCollector.collect(path: Path) -> set[str]` exactly.

- [ ] **Step 1: Add RED tests**

Add:

```python
from tests.architecture.imports import ImportCollector


def test_direct_collection_does_not_follow_symbol_reexports(source_tree: Path) -> None:
    path = _write(source_tree, CLIENT, "from ..facade import Ledger\n")
    direct = ImportCollector(source_tree / "trade_rl").collect_direct(path)
    assert direct == {"trade_rl.evaluation.facade"}
    assert SEALED not in direct


def test_direct_collection_detects_physical_child_module_import(source_tree: Path) -> None:
    path = _write(
        source_tree,
        CLIENT,
        "from ..robustness.walk_forward import sealed_test\n",
    )
    direct = ImportCollector(source_tree / "trade_rl").collect_direct(path)
    assert "trade_rl.evaluation.robustness.walk_forward" in direct
    assert SEALED in direct


def test_direct_collection_resolves_relative_child_import(source_tree: Path) -> None:
    path = _write(
        source_tree,
        "trade_rl/evaluation/robustness/walk_forward/client.py",
        "from . import sealed_test\n",
    )
    direct = ImportCollector(source_tree / "trade_rl").collect_direct(path)
    assert SEALED in direct


def test_direct_collection_includes_local_scope_imports(source_tree: Path) -> None:
    path = _write(
        source_tree,
        CLIENT,
        "def deferred():\n    from ..robustness.walk_forward import sealed_test\n",
    )
    direct = ImportCollector(source_tree / "trade_rl").collect_direct(path)
    assert SEALED in direct
```

- [ ] **Step 2: Verify RED**

```bash
uv run pytest -q \
  tests/architecture/test_import_collector_contract.py::test_direct_collection_does_not_follow_symbol_reexports \
  tests/architecture/test_import_collector_contract.py::test_direct_collection_detects_physical_child_module_import \
  tests/architecture/test_import_collector_contract.py::test_direct_collection_resolves_relative_child_import \
  tests/architecture/test_import_collector_contract.py::test_direct_collection_includes_local_scope_imports
```

Expected: failure because `collect_direct` is absent.

- [ ] **Step 3: Implement the minimal API**

Add to `ImportCollector`:

```python
    def collect_direct(self, path: Path) -> set[str]:
        result: set[str] = set()
        for node in ast.walk(self._tree(path)):
            if isinstance(node, ast.Import):
                result.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = self._base(path, node)
                result.add(base)
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    child = f"{base}.{alias.name}"
                    if child in self.modules:
                        result.add(child)
        return result
```

Do not modify `_export`, `_star_names`, or `collect`.

- [ ] **Step 4: Verify Task 1 GREEN**

```bash
uv run pytest -q tests/architecture/test_import_collector_contract.py
uv run mypy tests/architecture/imports.py
uv run ruff check tests/architecture/imports.py tests/architecture/test_import_collector_contract.py
uv run ruff format --check tests/architecture/imports.py tests/architecture/test_import_collector_contract.py
```

- [ ] **Step 5: Commit**

```bash
git add tests/architecture/imports.py tests/architecture/test_import_collector_contract.py
git commit -m "test: distinguish direct and semantic imports"
```

---

### Task 2: Run Core facade architecture RED

**Files:**
- Create: `tests/architecture/test_runs_capability_facade.py`

**Interfaces:**
- Consumes: `collect_direct()` from Task 1.
- Produces: exact Tier-2 surface, direct-owner dependency boundary, and Tier-1 API snapshot.

- [ ] **Step 1: Add exact Tier-2 and direct-import contracts**

Use:

```python
from __future__ import annotations

from pathlib import Path

from tests.architecture.imports import ImportCollector, within_module

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "trade_rl"
RUNS = PACKAGE / "evaluation" / "runs"

RUNS_EXPORTS = {
    "CandidateRunArtifactIdentity",
    "CandidateRunConfig",
    "CandidateRunResult",
    "LeanCandidateConfig",
    "LoadedCandidateRun",
    "PublishedCandidateRun",
    "ResolvedCandidateRunSpec",
    "build_candidate_run_provenance",
    "execute_candidate_run",
    "inspect_candidate_run_artifact",
    "load_candidate_run_artifact",
    "parse_candidate_run_config",
    "publish_candidate_run",
    "resolve_candidate_run_spec",
    "run_lean_candidate_suite",
}

FORBIDDEN_OWNER_MODULES = (
    "trade_rl.evaluation.runs.artifact",
    "trade_rl.evaluation.runs.candidate_suite",
    "trade_rl.evaluation.runs.config",
    "trade_rl.evaluation.runs.execute",
    "trade_rl.evaluation.runs.provenance",
)


def test_runs_facade_exports_exact_surface_and_owner_identity() -> None:
    import trade_rl.evaluation.runs as facade
    from trade_rl.evaluation.runs import artifact, candidate_suite, config, execute, provenance

    assert set(facade.__all__) == RUNS_EXPORTS
    assert facade.CandidateRunConfig is config.CandidateRunConfig
    assert facade.ResolvedCandidateRunSpec is config.ResolvedCandidateRunSpec
    assert facade.parse_candidate_run_config is config.parse_candidate_run_config
    assert facade.resolve_candidate_run_spec is config.resolve_candidate_run_spec
    assert facade.LeanCandidateConfig is candidate_suite.LeanCandidateConfig
    assert facade.run_lean_candidate_suite is candidate_suite.run_lean_candidate_suite
    assert facade.CandidateRunResult is execute.CandidateRunResult
    assert facade.execute_candidate_run is execute.execute_candidate_run
    assert facade.CandidateRunArtifactIdentity is artifact.CandidateRunArtifactIdentity
    assert facade.LoadedCandidateRun is artifact.LoadedCandidateRun
    assert facade.PublishedCandidateRun is artifact.PublishedCandidateRun
    assert facade.inspect_candidate_run_artifact is artifact.inspect_candidate_run_artifact
    assert facade.load_candidate_run_artifact is artifact.load_candidate_run_artifact
    assert facade.publish_candidate_run is artifact.publish_candidate_run
    assert facade.build_candidate_run_provenance is provenance.build_candidate_run_provenance
    assert not hasattr(facade, "load_candidate_run_config")
    assert not hasattr(facade, "PROVENANCE_SCHEMA")


def test_production_outside_runs_imports_run_core_through_facade() -> None:
    collector = ImportCollector(PACKAGE)
    offenders: list[tuple[str, str]] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if path.is_relative_to(RUNS):
            continue
        for imported in sorted(collector.collect_direct(path)):
            if any(within_module(imported, owner) for owner in FORBIDDEN_OWNER_MODULES):
                offenders.append((path.relative_to(ROOT).as_posix(), imported))
    assert offenders == []
```

- [ ] **Step 2: Snapshot current `trade_rl.evaluation.__all__`**

Use the exact current set from `trade_rl/evaluation/__init__.py`; it must continue to contain `LeanCandidateConfig` and `run_lean_candidate_suite` but gain no new names.

- [ ] **Step 3: Verify architecture RED**

```bash
uv run pytest -q tests/architecture/test_runs_capability_facade.py
```

Expected: facade-surface and direct-import tests fail; Tier-1 API snapshot passes.

- [ ] **Step 4: Commit RED**

```bash
git add tests/architecture/test_runs_capability_facade.py
git commit -m "test: define runs capability facade boundary"
```

---

### Task 3: Pure facade and caller migration

**Files:**
- Modify: `trade_rl/evaluation/runs/__init__.py`
- Modify: `trade_rl/evaluation/__init__.py`
- Modify: `trade_rl/evaluation/experiments/analysis.py`
- Modify: `trade_rl/evaluation/experiments/delta.py`
- Modify: `trade_rl/evaluation/experiments/evidence.py`
- Modify: `trade_rl/evaluation/experiments/workflow.py`
- Modify: `trade_rl/evaluation/experiments/codec.py`
- Modify: `trade_rl/evaluation/experiments/bootstrap/config.py`
- Modify: `trade_rl/evaluation/experiments/bootstrap/workflow.py`

**Interfaces:**
- Produces: the exact 15-name `RUNS_EXPORTS` contract from Task 2.

- [ ] **Step 1: Implement `runs/__init__.py` as direct re-exports only**

```python
"""Tier-2 public facade for immutable candidate-run contracts and execution."""

from trade_rl.evaluation.runs.artifact import (
    CandidateRunArtifactIdentity,
    LoadedCandidateRun,
    PublishedCandidateRun,
    inspect_candidate_run_artifact,
    load_candidate_run_artifact,
    publish_candidate_run,
)
from trade_rl.evaluation.runs.candidate_suite import (
    LeanCandidateConfig,
    run_lean_candidate_suite,
)
from trade_rl.evaluation.runs.config import (
    CandidateRunConfig,
    ResolvedCandidateRunSpec,
    parse_candidate_run_config,
    resolve_candidate_run_spec,
)
from trade_rl.evaluation.runs.execute import CandidateRunResult, execute_candidate_run
from trade_rl.evaluation.runs.provenance import build_candidate_run_provenance

__all__ = [
    "CandidateRunArtifactIdentity",
    "CandidateRunConfig",
    "CandidateRunResult",
    "LeanCandidateConfig",
    "LoadedCandidateRun",
    "PublishedCandidateRun",
    "ResolvedCandidateRunSpec",
    "build_candidate_run_provenance",
    "execute_candidate_run",
    "inspect_candidate_run_artifact",
    "load_candidate_run_artifact",
    "parse_candidate_run_config",
    "publish_candidate_run",
    "resolve_candidate_run_spec",
    "run_lean_candidate_suite",
]
```

Do not import `candidate.py`.

- [ ] **Step 2: Migrate only cross-capability production imports**

Replacement inventory:

```text
trade_rl/evaluation/__init__.py
  LeanCandidateConfig, run_lean_candidate_suite

trade_rl/evaluation/experiments/analysis.py
  LoadedCandidateRun

trade_rl/evaluation/experiments/delta.py
  LoadedCandidateRun

trade_rl/evaluation/experiments/codec.py
  CandidateRunConfig, ResolvedCandidateRunSpec

trade_rl/evaluation/experiments/evidence.py
  LoadedCandidateRun, inspect_candidate_run_artifact,
  load_candidate_run_artifact, publish_candidate_run,
  CandidateRunConfig, ResolvedCandidateRunSpec,
  resolve_candidate_run_spec, execute_candidate_run,
  build_candidate_run_provenance

trade_rl/evaluation/experiments/workflow.py
  CandidateRunConfig, resolve_candidate_run_spec,
  build_candidate_run_provenance

trade_rl/evaluation/experiments/bootstrap/config.py
  CandidateRunConfig, parse_candidate_run_config

trade_rl/evaluation/experiments/bootstrap/workflow.py
  ResolvedCandidateRunSpec, resolve_candidate_run_spec
```

Every listed file must import these names from `trade_rl.evaluation.runs`. Do not change call sites, arguments, returns, or exception handling. Do not change direct owner imports inside `trade_rl/evaluation/runs/`.

- [ ] **Step 3: Verify targeted GREEN**

```bash
uv run pytest -q \
  tests/architecture/test_import_collector_contract.py \
  tests/architecture/test_runs_capability_facade.py \
  tests/evaluation \
  tests/architecture/test_experiment_workflow_boundaries.py
uv run ruff check trade_rl/evaluation tests/architecture
uv run ruff format --check trade_rl/evaluation tests/architecture
uv run mypy trade_rl/evaluation
uv run mypy tests/architecture/imports.py
```

- [ ] **Step 4: Re-run production import inventory**

Search all `trade_rl.evaluation.runs.` occurrences. Production owner-submodule imports outside `trade_rl/evaluation/runs/` must be absent. CLI module path text and package-internal owner imports remain valid.

- [ ] **Step 5: Commit**

```bash
git add trade_rl/evaluation tests/architecture
git commit -m "refactor: expose candidate run capability facade"
```

---

### Task 4: Durable docs and Active-doc cleanup

**Files:**
- Modify: `docs/architecture/package-boundaries.md`
- Modify: `docs/README.md`
- Delete: `docs/specs/2026-09-11-runs-capability-facade-design.md`
- Delete: `docs/plans/2026-09-11-runs-capability-facade-implementation.md`

- [ ] **Step 1: Promote the durable rule**

Add to `package-boundaries.md`:

```text
- `trade_rl.evaluation.runs` is the Tier-2 public facade for candidate-run
  contracts, execution, artifact inspection/publication, and provenance construction.
- `runs/config.py`, `candidate_suite.py`, `execute.py`, `artifact.py`, and
  `provenance.py` remain implementation owners.
- production code outside `evaluation/runs/` imports Run Core through the facade;
  package-internal code may import owner modules directly.
- `trade_rl.evaluation` remains the existing Tier-1 surface and is not expanded.
- persisted candidate-run schemas are independent compatibility contracts and are
  not inferred from Python import paths.
- architecture tooling distinguishes direct import spelling from semantic
  re-export ownership.
```

- [ ] **Step 2: Remove completed Active docs and restore README**

Delete the spec and this plan. Set the README current-state sentence to:

```text
現在Activeなspec/planはない。
```

Do not create `docs/history` or `docs/archive`.

- [ ] **Step 3: Verify docs policy**

```bash
uv run pytest -q tests/architecture/test_current_docs_layout.py
```

- [ ] **Step 4: Commit**

```bash
git add docs
git commit -m "docs: promote runs capability boundary"
```

---

### Task 5: Falsification and full hardened verification

- [ ] **Step 1: Execute falsification probes, restoring the tree after each**

Prove these mistakes are detected:

1. Reintroduce one external `runs.config` import; the direct-owner architecture test must fail.
2. Make `collect_direct()` follow `_export()` for named symbols; the facade-symbol direct-collection test must fail.
3. Replace one facade re-export with a wrapper; the owner-identity test must fail.
4. Add a new name to `trade_rl.evaluation.__all__`; the Tier-1 snapshot must fail.
5. Confirm a test-only direct owner import is not flagged by the production scan.

Record only probes actually executed.

- [ ] **Step 2: Run full repository checks**

```bash
uv run ruff check trade_rl tests
uv run ruff format --check trade_rl tests
uv run mypy trade_rl
uv run mypy tests/architecture/imports.py tests/architecture/distribution.py
uv run pytest -q tests
uv build
uv run python -m tests.architecture.distribution dist/*.tar.gz dist/*.whl
```

Rebuild a wheel from the sdist and run the same distribution closure check against it.

- [ ] **Step 3: Clean-installed smoke outside the checkout**

Install the built wheel into a fresh Python 3.12 venv, change to a directory outside the repository, and verify:

```python
import trade_rl
import trade_rl.evaluation
import trade_rl.evaluation.runs
from trade_rl.evaluation.runs import CandidateRunConfig, load_candidate_run_artifact
```

Then run:

```bash
python -I -m trade_rl.evaluation.runs.candidate --help
python -I -m trade_rl.evaluation.experiments.bootstrap.cli --help
```

- [ ] **Step 4: Final diff/self-review**

```bash
git diff --check main...HEAD
git status --short
git diff --name-status main...HEAD
```

Verify no temporary workflows, generated artifacts, wrappers, unrelated refactors, or completed spec/plan remain.

- [ ] **Step 5: Exact-head CI and merge gate**

Require final PR HEAD CI to pass Ruff, Format, production Mypy, architecture-tooling Mypy, full pytest, Build, distribution source closure/rebuild, clean installed smoke, and package identity. Check review comments/threads. Merge with an expected-head SHA guard only after all gates are green, then require the resulting `main` merge commit to pass its own push CI.
