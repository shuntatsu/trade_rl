# Residual Core Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the mixed legacy `mars_lite` implementation with one typed, responsibility-separated `trade_rl` package for baseline-anchored residual RL research, evaluation, artifacts, and serving contracts.

**Architecture:** Build a standard-library-only domain core, then layer artifact codecs, evaluation primitives, residual policy components, workflows, CLI, and serving around it. Add migration tests for the supplied 2026-07-13 run, switch packaging and CI to `trade_rl`, and delete all maintained legacy Mars/direct execution paths.

**Tech Stack:** Python 3.12, dataclasses, NumPy, pandas, Gymnasium, Stable-Baselines3, FastAPI, pytest, Hypothesis, mypy, Ruff, Import Linter, Vulture.

## Global Constraints

- Production remains **NO-GO**.
- No legacy import or CLI compatibility layer.
- No maintained direct-action PPO mode.
- `trade_rl.domain` uses the Python standard library only.
- Workflows accept typed config objects, never `argparse.Namespace`.
- Dataset, signal, policy, evaluation, selection, and release artifacts are distinct.
- Failed mandatory gates prevent release creation.
- All artifact writes are canonical, content-addressed, validated, staged, and atomically published.
- The supplied 2026-07-13 report is a migration fixture and must classify as baseline-only analysis with production blocked.

---

## File structure

Create and maintain these authoritative areas:

```text
trade_rl/
  domain/
  artifacts/
  evaluation/
  data/
  strategies/
  signals/
  rl/
  simulation/
  workflows/
  serving/
  config/
  cli/
tests/
  domain/
  artifacts/
  evaluation/
  migration/
  workflows/
  serving/
  cli/
```

Do not create empty directories. Each package appears with its first concrete responsibility.

### Task 1: Package skeleton and domain invariants

**Files:**
- Create: `trade_rl/__init__.py`
- Create: `trade_rl/domain/__init__.py`
- Create: `trade_rl/domain/common.py`
- Create: `trade_rl/domain/datasets.py`
- Create: `trade_rl/domain/signals.py`
- Create: `trade_rl/domain/policies.py`
- Create: `trade_rl/domain/evaluation.py`
- Create: `trade_rl/domain/selection.py`
- Create: `trade_rl/domain/releases.py`
- Test: `tests/domain/test_artifact_invariants.py`

**Interfaces:**
- Produces immutable dataclasses and enums for all later tasks.
- Produces `validate_cross_artifact_identity(...) -> None`.

- [ ] Add failing tests for baseline-only selection without a policy digest, residual selection requiring ensemble members, rejected signal disabling alpha, and failed mandatory gates blocking releases.
- [ ] Run the domain test in CI and confirm failures are due to missing `trade_rl.domain`.
- [ ] Implement frozen dataclasses with validation in `__post_init__`.
- [ ] Run domain tests and mypy.
- [ ] Commit `feat: add typed residual domain model`.

### Task 2: Canonical artifacts and atomic publication

**Files:**
- Create: `trade_rl/artifacts/__init__.py`
- Create: `trade_rl/artifacts/codec.py`
- Create: `trade_rl/artifacts/hashing.py`
- Create: `trade_rl/artifacts/store.py`
- Create: `trade_rl/artifacts/validators.py`
- Test: `tests/artifacts/test_codec.py`
- Test: `tests/artifacts/test_store.py`

**Interfaces:**
- Produces `canonical_json_bytes(value: object) -> bytes`.
- Produces `content_digest(value: object) -> str`.
- Produces `ArtifactStore.stage_run`, `publish_run`, and `mark_failed`.

- [ ] Add failing tests for stable canonical JSON, digest sensitivity, validation-before-publication, atomic latest-pointer replacement, and failure isolation.
- [ ] Confirm RED in CI.
- [ ] Implement canonical serialization for dataclasses, enums, timestamps, mappings, tuples, and paths.
- [ ] Implement staged run directories and atomic publication using same-filesystem `os.replace`.
- [ ] Run artifact tests and commit `feat: add content-addressed artifact store`.

### Task 3: Unified evaluation primitives

**Files:**
- Create: `trade_rl/evaluation/__init__.py`
- Create: `trade_rl/evaluation/series.py`
- Create: `trade_rl/evaluation/metrics.py`
- Create: `trade_rl/evaluation/comparisons.py`
- Create: `trade_rl/evaluation/bootstrap.py`
- Create: `trade_rl/evaluation/gates.py`
- Test: `tests/evaluation/test_metrics.py`
- Test: `tests/evaluation/test_comparisons.py`
- Test: `tests/evaluation/test_gates.py`

**Interfaces:**
- Produces `ReturnSeries`, `PerformanceMetrics`, `PairedComparison`, and `GateDecision`.
- Consumes domain evaluation records.

- [ ] Add failing tests for total return, Sharpe, Sortino, drawdown, turnover, costs, funding, paired excess, annualization metadata, and gate fail-closed behavior.
- [ ] Confirm RED in CI.
- [ ] Implement metrics once and remove duplicated calculations from new code paths.
- [ ] Add deterministic block bootstrap with explicit seed.
- [ ] Run tests and commit `feat: centralize evaluation and gate logic`.

### Task 4: Walk-forward fold model and stitched OOS

**Files:**
- Create: `trade_rl/evaluation/walk_forward/__init__.py`
- Create: `trade_rl/evaluation/walk_forward/folds.py`
- Create: `trade_rl/evaluation/walk_forward/stitching.py`
- Create: `trade_rl/evaluation/walk_forward/reports.py`
- Test: `tests/evaluation/test_walk_forward_folds.py`
- Test: `tests/evaluation/test_walk_forward_stitching.py`

**Interfaces:**
- Produces `WalkForwardFold`, `build_folds`, `validate_fold`, and `stitch_oos`.

- [ ] Add Hypothesis tests proving purge separation, non-overlap, chronological OOS ordering, and rejection of insufficient executable folds.
- [ ] Confirm RED in CI.
- [ ] Implement pure fold construction and stitched aggregation without workflow imports.
- [ ] Run tests and commit `feat: add pure nested walk-forward primitives`.

### Task 5: Migrate the supplied real-data result

**Files:**
- Create: `tests/fixtures/legacy/realdata_20260713_report.json`
- Create: `tests/fixtures/legacy/realdata_20260713_manifest.json`
- Create: `tests/fixtures/legacy/realdata_20260713_signal_metadata.json`
- Create: `trade_rl/artifacts/legacy_migration.py`
- Test: `tests/migration/test_realdata_20260713.py`

**Interfaces:**
- Produces `migrate_legacy_research_run(...) -> MigratedResearchRun`.

- [ ] Add fixture data from the user-supplied report and manifests.
- [ ] Add failing tests asserting `COMPLETED`, `REJECTED`, `NOT_SELECTED`, `BASELINE_ONLY`, and `BLOCKED` classifications.
- [ ] Confirm the test rejects labeling feature/GBM metadata as a PPO ensemble.
- [ ] Implement narrow migration code solely for reading archived evidence; do not expose it as runtime compatibility.
- [ ] Run tests and commit `feat: classify archived residual research evidence`.

### Task 6: Residual policy, environment, and training boundaries

**Files:**
- Create: `trade_rl/rl/actions.py`
- Create: `trade_rl/rl/observations.py`
- Create: `trade_rl/rl/rewards.py`
- Create: `trade_rl/rl/environment.py`
- Create: `trade_rl/rl/checkpoints.py`
- Create: `trade_rl/rl/ensemble.py`
- Create: `trade_rl/rl/training.py`
- Create: `trade_rl/strategies/trend.py`
- Create: `trade_rl/simulation/execution.py`
- Create: `trade_rl/simulation/accounting.py`
- Test: `tests/rl/test_residual_action.py`
- Test: `tests/rl/test_environment_timing.py`
- Test: `tests/rl/test_ensemble_manifest.py`

**Interfaces:**
- Produces only baseline-anchored residual actions.
- Produces `ResidualTrainingConfig` and `train_residual_ensemble`.
- Consumes typed dataset/signal manifests and evaluation functions.

- [ ] Add failing tests for action schema, baseline identity action, decision interval reward attribution, ensemble member count, and dataset identity parity.
- [ ] Confirm RED in CI.
- [ ] Reimplement shared observation, execution, accounting, and residual environment behavior behind focused modules.
- [ ] Wrap Stable-Baselines3 only inside `training.py` and checkpoint adapters.
- [ ] Run tests and commit `feat: rebuild baseline anchored residual rl core`.

### Task 7: Workflows and one CLI

**Files:**
- Create: `trade_rl/config/schemas.py`
- Create: `trade_rl/config/loaders.py`
- Create: `trade_rl/workflows/train_signal.py`
- Create: `trade_rl/workflows/train_residual.py`
- Create: `trade_rl/workflows/walk_forward.py`
- Create: `trade_rl/workflows/evaluate.py`
- Create: `trade_rl/workflows/publish.py`
- Create: `trade_rl/cli/app.py`
- Create: `trade_rl/cli/commands/*.py`
- Test: `tests/workflows/test_walk_forward_workflow.py`
- Test: `tests/cli/test_cli.py`

**Interfaces:**
- Produces installed command `trade-rl`.
- Workflows receive frozen config dataclasses and collaborator protocols.

- [ ] Add failing tests for command discovery and typed workflow invocation.
- [ ] Confirm RED in CI.
- [ ] Implement `data`, `signal`, `train`, `walk-forward`, `evaluate`, `registry`, and `serve` command groups.
- [ ] Ensure no workflow accepts or mutates `argparse.Namespace`.
- [ ] Run tests and commit `feat: add typed workflows and unified cli`.

### Task 8: Serving, bundle, and registry contracts

**Files:**
- Create: `trade_rl/serving/bundle.py`
- Create: `trade_rl/serving/registry.py`
- Create: `trade_rl/serving/runtime.py`
- Create: `trade_rl/serving/audit.py`
- Create: `trade_rl/serving/api.py`
- Test: `tests/serving/test_bundle.py`
- Test: `tests/serving/test_registry.py`
- Test: `tests/serving/test_runtime.py`

**Interfaces:**
- Produces immutable bundle validation, atomic registry activation, and fail-closed runtime loading.
- Consumes typed release manifests and artifacts.

- [ ] Add failing tests for digest validation, release-gate enforcement, identity handshake, safe hot-swap, and previous-bundle preservation.
- [ ] Confirm RED in CI.
- [ ] Reimplement serving contracts without importing training workflows.
- [ ] Preserve read-only authenticated HTTP surface.
- [ ] Run tests and commit `feat: rebuild serving and registry boundaries`.

### Task 9: Packaging, dependency contracts, and legacy deletion

**Files:**
- Modify: `pyproject.toml`
- Modify: `.github/workflows/ci.yml`
- Create: `.importlinter`
- Rewrite: `README.md`
- Rewrite: `README.ja.md`
- Rewrite: `docs/ARCHITECTURE.md`
- Delete: `mars_lite/**`
- Delete: `legacy_tests/**`
- Delete: legacy `scripts/**` not replaced by the installed CLI
- Delete: obsolete direct/Mars tests and documentation
- Test: `tests/test_architecture_contract.py`

**Interfaces:**
- Packages only `trade_rl*`.
- Provides console script `trade-rl = trade_rl.cli.app:main`.

- [ ] Add architecture tests asserting no `mars_lite` package, no `action_mode="direct"`, no legacy script entry points, and no forbidden dependency direction.
- [ ] Confirm RED before deletion.
- [ ] Switch project metadata and CI to `trade_rl`.
- [ ] Enable Ruff F401/F841, strict mypy for `trade_rl`, Import Linter, Hypothesis, and Vulture advisory output.
- [ ] Delete all legacy execution paths and update documentation.
- [ ] Run full CI and commit `refactor: remove legacy mars and direct policy stacks`.

### Task 10: Final verification and pull request

**Files:**
- Modify only files required by verified failures.

- [ ] Run Ruff check and formatting.
- [ ] Run strict mypy.
- [ ] Run Import Linter.
- [ ] Run unit, property, migration, integration, and serving tests with coverage.
- [ ] Run Vulture and review each reported symbol.
- [ ] Search the final tree for `mars_lite`, `action_mode.*direct`, old script names, and disabled type/coverage rules.
- [ ] Confirm the supplied real-data fixture produces no release and no selected PPO artifact.
- [ ] Compare the branch to `main` and inspect deletions and dependency changes.
- [ ] Open a draft PR with architecture, migration result, deletions, checks, and known production non-guarantees.

## Self-review

- The plan covers all design requirements: typed artifacts, explicit baseline fallback, unified evaluation, split walk-forward, single CLI, serving separation, deletion, and quality gates.
- No runtime compatibility layer is planned.
- Every new behavior begins with a failing test.
- Production remains blocked regardless of historical holdout performance.
