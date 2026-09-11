# evaluation/runs Capability Facade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `trade_rl.evaluation.runs` the single Tier-2 cross-capability entry point for Run Core contracts while preserving existing implementation owners, runtime behavior, persisted artifacts, CLI behavior, and top-level evaluation API.

**Architecture:** Keep `runs/artifact.py`, `candidate_suite.py`, `config.py`, `execute.py`, and `provenance.py` as implementation owners. Add a pure re-export facade in `runs/__init__.py`, migrate production callers outside `runs/` to that facade, and add a direct-import AST contract that distinguishes import spelling from the existing re-export-aware semantic dependency collector.

**Tech Stack:** Python 3.12, dataclasses, NumPy, pytest, Ruff, Mypy, existing `tests.architecture.imports.ImportCollector`, GitHub Actions CI, uv packaging.

**Spec:** `docs/specs/2026-09-11-runs-capability-facade-design.md`

## Global Constraints

- Do not move, merge, rename, or delete existing `evaluation/runs/` owner modules.
- Do not add business logic, wrappers, adapters, error translation, or side effects to `runs/__init__.py`.
- Do not expand `trade_rl.evaluation.__all__`.
- Do not expose `load_candidate_run_config`, `runtime_environment_manifest`, `PROVENANCE_SCHEMA`, CLI `main`, or `run_candidate_artifact` through the Tier-2 facade.
- Preserve candidate result schema, candidate artifact identity schema, provenance schema, exact-file evidence, semantic artifact digest, CLI behavior, and historical readers.
- Preserve existing `ImportCollector.collect()` re-export-aware semantics exactly; add `collect_direct()` as a separate contract.
- Production source outside `trade_rl/evaluation/runs/` must not directly import `runs.artifact`, `runs.candidate_suite`, `runs.config`, `runs.execute`, or `runs.provenance` after migration.
- Tests may still import internal owner modules for implementation-level verification.
- New source bytes legitimately change candidate-run implementation provenance; do not normalize or weaken provenance to hide this.
- Use TDD: establish RED before production facade/import migration.
- `docs/specs/` and `docs/plans/` remain current only while this work is Active; remove both completed files after durable architecture documentation is updated.

---

### Task 1: Add a direct-import inspection contract without changing semantic import analysis

**Files:**
- Modify: `tests/architecture/imports.py`
- Modify: `tests/architecture/test_import_collector_contract.py`

**Interfaces:**
- Consumes: existing `ImportCollector._base()`, `ImportCollector.modules`, and `ImportCollector.collect()` semantics.
- Produces: `ImportCollector.collect_direct(path: Path) -> set[str]`.

- [ ] **Step 1: Add failing direct-import contract tests**

Append tests that import `ImportCollector` directly and prove direct spelling does not follow re-exports:

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
    relative = "trade_rl/evaluation/robustness/walk_forward/client.py"
    path = _write(source_tree, relative, "from . import sealed_test\n")
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

- [ ] **Step 2: Run the new tests to establish RED**

Run:

```bash
uv run pytest -q \
  tests/architecture/test_import_collector_contract.py::test_direct_collection_does_not_follow_symbol_reexports \
  tests/architecture/test_import_collector_contract.py::test_direct_collection_detects_physical_child_module_import \
  tests/architecture/test_import_collector_contract.py::test_direct_collection_resolves_relative_child_import \
  tests/architecture/test_import_collector_contract.py::test_direct_collection_includes_local_scope_imports
```

Expected: fail because `ImportCollector` has no `collect_direct` method.

- [ ] **Step 3: Implement the minimal `collect_direct()` API**

Add this method without modifying `_export()`, `_star_names()`, or `collect()`:

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

This deliberately records physical child modules while refusing to trace named symbol re-exports.

- [ ] **Step 4: Verify direct and legacy semantic collector contracts**

Run:

```bash
uv run pytest -q tests/architecture/test_import_collector_contract.py
uv run mypy tests/architecture/imports.py
uv run ruff check tests/architecture/imports.py tests/architecture/test_import_collector_contract.py
uv run ruff format --check tests/architecture/imports.py tests/architecture/test_import_collector_contract.py
```

Expected: all pass; existing re-export/star/cycle tests remain unchanged and green.

- [ ] **Step 5: Commit Task 1**

```bash
git add tests/architecture/imports.py tests/architecture/test_import_collector_contract.py
git commit -m "test: distinguish direct and semantic imports"
```

---

### Task 2: Define the Run Core facade and dependency boundary as RED architecture contracts

**Files:**
- Create: `tests/architecture/test_runs_capability_facade.py`
- No production files modified in this task.

**Interfaces:**
- Consumes: `ImportCollector.collect_direct()` from Task 1.
- Produces: exact Tier-2 facade surface, exact top-level evaluation API snapshot, and forbidden direct-owner dependency rule.

- [ ] **Step 1: Add the exact facade surface and owner-identity contract**

Create the following constants and test structure:

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
```

Use runtime imports only for identity checks:

```python
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
```

- [ ] **Step 2: Add the production direct-import boundary test**

Use `collect_direct()` and skip only the owner package itself:

```python
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

This intentionally ignores tests and allows internal `runs/` modules to keep direct owner imports.

- [ ] **Step 3: Snapshot the existing Tier-1 evaluation public API**

Hard-code the current `trade_rl.evaluation.__all__` set so a facade refactor cannot silently broaden Tier 1:

```python
EVALUATION_EXPORTS = {
    "BootstrapResult",
    "CapacityCurve",
    "CapacityPoint",
    "ClosedTradeDiagnostics",
    "ClosedTradeTracker",
    "ExecutionDiagnostics",
    "IndependentFoldSummary",
    "LeanCandidateConfig",
    "PERFECT_INFORMATION_BOUND_SCHEMA",
    "PairedComparison",
    "PerformanceMetrics",
    "PerfectInformationBoundConfig",
    "PerfectInformationBoundResult",
    "ReplayDecision",
    "ReturnKind",
    "ReturnSeries",
    "SeedEvaluation",
    "SeedResult",
    "SeedRobustnessSummary",
    "SingleSymbolReplayResult",
    "StrategyComparison",
    "StrategyComparisonEntry",
    "SymbolStrategyComparison",
    "UniversalStrategyComparison",
    "compare_paired_returns",
    "compare_strategies",
    "compare_strategies_by_symbol",
    "compound_return",
    "evaluate_capacity_grid",
    "evaluate_performance",
    "moving_block_mean_test",
    "resolve_gate",
    "run_lean_candidate_suite",
    "run_single_symbol_replay",
    "solve_perfect_information_bound",
    "summarize_independent_folds",
    "summarize_seed_robustness",
}


def test_evaluation_tier1_public_api_is_unchanged() -> None:
    import trade_rl.evaluation as evaluation

    assert set(evaluation.__all__) == EVALUATION_EXPORTS
```

- [ ] **Step 4: Run architecture RED**

Run:

```bash
uv run pytest -q tests/architecture/test_runs_capability_facade.py
```

Expected failures:

- `trade_rl.evaluation.runs` has no required Tier-2 `__all__`/exports yet.
- external production files still directly import owner submodules.
- Tier-1 snapshot should already pass.

If failures occur outside these expected contracts, stop and debug before production changes.

- [ ] **Step 5: Commit Task 2 RED**

```bash
git add tests/architecture/test_runs_capability_facade.py
git commit -m "test: define runs capability facade boundary"
```

---

### Task 3: Implement the pure facade and migrate cross-capability callers

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
- Consumes: existing owner symbols unchanged.
- Produces: the exact `RUNS_EXPORTS` surface from Task 2; all external production callers use `from trade_rl.evaluation.runs import ...`.

- [ ] **Step 1: Replace the empty facade with direct owner re-exports**

Use only re-exports; no wrappers:

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

Do not import `candidate.py`; importing the facade must not pull in the CLI adapter.

- [ ] **Step 2: Migrate each external production caller to the facade**

Apply only import-path changes, preserving every imported symbol and all executable code below the imports.

Replacement inventory:

```text
trade_rl/evaluation/__init__.py
  runs.candidate_suite -> runs
  symbols: LeanCandidateConfig, run_lean_candidate_suite

trade_rl/evaluation/experiments/analysis.py
  runs.artifact -> runs
  symbols: LoadedCandidateRun

trade_rl/evaluation/experiments/delta.py
  runs.artifact -> runs
  symbols: LoadedCandidateRun

trade_rl/evaluation/experiments/codec.py
  runs.config -> runs
  symbols: CandidateRunConfig, ResolvedCandidateRunSpec

trade_rl/evaluation/experiments/evidence.py
  runs.artifact + runs.config + runs.execute + runs.provenance -> one runs import
  symbols: LoadedCandidateRun, inspect_candidate_run_artifact,
           load_candidate_run_artifact, publish_candidate_run,
           CandidateRunConfig, ResolvedCandidateRunSpec,
           resolve_candidate_run_spec, execute_candidate_run,
           build_candidate_run_provenance

trade_rl/evaluation/experiments/workflow.py
  runs.config + runs.provenance -> one runs import
  symbols: CandidateRunConfig, resolve_candidate_run_spec,
           build_candidate_run_provenance

trade_rl/evaluation/experiments/bootstrap/config.py
  runs.config -> runs
  symbols: CandidateRunConfig, parse_candidate_run_config

trade_rl/evaluation/experiments/bootstrap/workflow.py
  runs.config -> runs
  symbols: ResolvedCandidateRunSpec, resolve_candidate_run_spec
```

Do not change direct owner imports inside `trade_rl/evaluation/runs/` itself.

- [ ] **Step 3: Run targeted GREEN checks**

Run:

```bash
uv run pytest -q \
  tests/architecture/test_import_collector_contract.py \
  tests/architecture/test_runs_capability_facade.py \
  tests/evaluation \
  tests/architecture/test_experiment_workflow_boundaries.py
uv run ruff check trade_rl/evaluation tests/architecture/test_import_collector_contract.py tests/architecture/test_runs_capability_facade.py
uv run ruff format --check trade_rl/evaluation tests/architecture/test_import_collector_contract.py tests/architecture/test_runs_capability_facade.py
uv run mypy trade_rl/evaluation
uv run mypy tests/architecture/imports.py
```

Expected: all green.

- [ ] **Step 4: Re-run repository-wide production import inventory**

Search and inspect every `trade_rl.evaluation.runs.` occurrence. Allowed production occurrences after migration are:

- inside `trade_rl/evaluation/runs/` owner modules;
- CLI module path text such as `python -m trade_rl.evaluation.runs.candidate` in docs/README;
- no direct owner-submodule import outside `runs/`.

The architecture test must independently enforce this; search output is a review signal, not the sole oracle.

- [ ] **Step 5: Commit Task 3**

```bash
git add trade_rl/evaluation tests/architecture
git commit -m "refactor: expose candidate run capability facade"
```

---

### Task 4: Promote the durable boundary and remove completed Active docs

**Files:**
- Modify: `docs/architecture/package-boundaries.md`
- Modify: `docs/README.md`
- Delete: `docs/specs/2026-09-11-runs-capability-facade-design.md`
- Delete: `docs/plans/2026-09-11-runs-capability-facade-implementation.md`
- Verify: `tests/architecture/test_current_docs_layout.py`

**Interfaces:**
- Consumes: verified implementation from Tasks 1-3.
- Produces: durable current architecture only; Git history retains completed spec/plan.

- [ ] **Step 1: Add the durable Run Core boundary to `package-boundaries.md`**

Document these current-state rules, not implementation history:

```text
- `trade_rl.evaluation.runs` is the Tier-2 public facade for candidate-run contracts,
  execution, artifact inspection/publication, and provenance construction.
- `runs/config.py`, `candidate_suite.py`, `execute.py`, `artifact.py`, and
  `provenance.py` remain implementation owners.
- production code outside `evaluation/runs/` imports Run Core through the facade;
  package-internal code may import owner modules directly to avoid facade cycles.
- `trade_rl.evaluation` remains the existing Tier-1 surface and is not expanded by
  this capability rule.
- persisted candidate-run schemas are independent compatibility contracts and are
  not inferred from Python import paths.
```

Also note that architecture tooling distinguishes direct import spelling from semantic re-export ownership.

- [ ] **Step 2: Remove completed spec/plan and restore README current-only state**

Delete both Active files after durable docs are updated. Change `docs/README.md` back to a truthful no-active-work statement:

```text
現在Activeなspec/planはない。
```

Do not create `docs/history` or `docs/archive`.

- [ ] **Step 3: Run docs retention/link tests**

Run:

```bash
uv run pytest -q tests/architecture/test_current_docs_layout.py
```

Expected: current authorities present, no inactive/completed ephemeral docs, no broken links, old bootstrap docs still absent.

- [ ] **Step 4: Commit Task 4**

```bash
git add docs tests/architecture/test_current_docs_layout.py
git commit -m "docs: promote runs capability boundary"
```

---

### Task 5: Falsification review and full hardened verification

**Files:**
- Review final diff only; no planned production files beyond Tasks 1-4.

**Interfaces:**
- Consumes: complete implementation.
- Produces: objective evidence for merge/closure.

- [ ] **Step 1: Run focused falsification probes**

Use temporary local/worktree edits or equivalent non-final changes and prove the tests kill these mistakes, restoring the tree after each probe:

1. Change one external facade import back to `from trade_rl.evaluation.runs.config import CandidateRunConfig`; `test_production_outside_runs_imports_run_core_through_facade` must fail.
2. Change `collect_direct()` to call `_export()` for named symbols; the facade-symbol direct-collection test must fail.
3. Replace one facade re-export with a wrapper function/class; owner-identity test must fail.
4. Add an extra name such as `CandidateRunConfig` to `trade_rl.evaluation.__all__`; Tier-1 snapshot must fail.
5. Verify a test-only direct internal import is not flagged by the production dependency scan.

Record which probes were actually executed and their observed failures; do not claim unexecuted probes.

- [ ] **Step 2: Run full repository checks**

Run exactly:

```bash
uv run ruff check trade_rl tests
uv run ruff format --check trade_rl tests
uv run mypy trade_rl
uv run mypy tests/architecture/imports.py tests/architecture/distribution.py
uv run pytest -q tests
uv build
uv run python -m tests.architecture.distribution dist/*.tar.gz dist/*.whl
```

Then rebuild a wheel from the produced sdist and run the distribution closure check against that rebuilt wheel, matching `.github/workflows/ci.yml`.

- [ ] **Step 3: Run clean-installed package smoke outside the checkout**

Create a fresh Python 3.12 virtual environment, install the built wheel, change working directory outside the repository, and verify:

```python
import trade_rl
import trade_rl.evaluation
import trade_rl.evaluation.runs
from trade_rl.evaluation.runs import CandidateRunConfig, load_candidate_run_artifact
```

Verify package version/installation origin and run:

```bash
python -I -m trade_rl.evaluation.runs.candidate --help
python -I -m trade_rl.evaluation.experiments.bootstrap.cli --help
```

- [ ] **Step 4: Final diff/self-review**

Check:

```bash
git diff --check main...HEAD
git status --short
git diff --name-status main...HEAD
```

Review requirement compliance, owner boundaries, cycles, public API, error behavior, artifact/provenance compatibility, dead helpers, temporary workflows, generated files, and unrelated changes.

Expected final source/doc intent:

- `tests/architecture/imports.py` + import contract tests;
- new facade architecture test;
- `runs/__init__.py` re-exports;
- cross-capability import-path migration only;
- durable `package-boundaries.md` update;
- completed Active spec/plan absent from final tree.

- [ ] **Step 5: Push final head and require exact-head CI**

Open/update the implementation PR as Draft until local/targeted gates are green. Then require the GitHub Actions CI for the exact final PR head to succeed through:

- Ruff
- Format
- production Mypy
- architecture-tooling Mypy
- full pytest
- Build
- distribution source closure and sdist rebuild
- clean installed core smoke
- package identity

Do not use a successful run from an older commit as evidence for the final head.

- [ ] **Step 6: Independent-equivalent review and merge**

If a separate reviewer/subagent is available, give it the original spec, Acceptance Criteria, final diff, tests, and exact CI evidence and ask it to search for violations rather than confirm the implementation. If no independent agent is available, reconstruct the checks from the spec and review the final diff without relying on implementation conclusions.

Only after this gate is green:

```bash
# Merge the PR using an expected-head SHA guard.
```

Then verify the resulting `main` merge commit has its own successful push CI before declaring the change complete.
