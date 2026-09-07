# Training Capability Audit Operations Boundary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the full training-capability audit implementation from `scripts/run_training_capability_audit.py` into `trade_rl.operations` while preserving the current CLI, report schema, digest, output layout, and runtime probe semantics.

**Architecture:** Keep `scripts/run_training_capability_audit.py` as a thin executable adapter. Add `trade_rl.operations.training_capability_audit` as the public package facade and `trade_rl.operations._training_capability_audit_impl` as the sole owner of audit environments, synthetic data, SB3 probes, resume/export checks, and report assembly. Protect the boundary with AST/import tests and pin report serialization with deterministic filesystem tests.

**Tech Stack:** Python 3.12, pytest, Gymnasium, NumPy, PyTorch, Stable-Baselines3 / sb3-contrib, Ruff, Mypy, Import Linter, vulture, GitHub Actions.

**Spec:** `docs/implementation-plans/specs/2026-08-18-training-capability-audit-operations-design.md`

## Global Constraints

- Preserve `full_training_capability_audit_v1` exactly; do not add, remove, or rename report fields.
- Preserve report digest construction: compute `content_digest(report)` before inserting `report["digest"]`.
- Preserve persisted report bytes: `json.dumps(report, indent=2, sort_keys=True) + "\n"` encoded as UTF-8.
- Preserve stdout bytes apart from the terminal newline added by `print`: `json.dumps(report, sort_keys=True)`.
- Preserve the script CLI option `--output`, its `Path` type, and default `var/training-capability-audit`.
- Preserve output-root replacement semantics: an existing output root is removed and recreated before probes run.
- Preserve all current PPO/SAC/TD3/TQC hyperparameters, architecture assertions, replay/checkpoint requirements, export settings, residual-control probes, behavior-cloning probe, and hierarchical sequence probe.
- Do not modify production research, reward, risk, execution, selection, serving, or promotion semantics.
- Do not modify `.github/workflows/full-training-capability-audit.yml` unless a test proves the unchanged command cannot work.
- No lint ignore, test skip, assertion weakening, compatibility shim, or unrelated refactor may be used to obtain Green.

---

### Task 1: Lock the script/operations ownership boundary with a valid RED

**Files:**
- Create: `tests/architecture/test_training_capability_audit_owner.py`
- Test: `tests/architecture/test_training_capability_audit_owner.py`

**Interfaces:**
- Consumes: existing `scripts/run_training_capability_audit.py` and repository AST/import structure.
- Produces: architecture contract requiring public `trade_rl.operations.training_capability_audit.run_training_capability_audit(Path) -> dict[str, object]`, forbidding training implementation ownership in the script, and forbidding lower production layers from importing the audit operation.

- [ ] **Step 1: Write the failing architecture test**

Create `tests/architecture/test_training_capability_audit_owner.py` with direct AST checks so the intended failure does not depend on importing heavy optional training packages:

```python
from __future__ import annotations

import ast
from pathlib import Path

from tests.architecture.repository_paths import PYTHON_SOURCE_ROOT, REPOSITORY_ROOT

SCRIPT = REPOSITORY_ROOT / "scripts/run_training_capability_audit.py"
PUBLIC = PYTHON_SOURCE_ROOT / "operations/training_capability_audit.py"
PRIVATE = PYTHON_SOURCE_ROOT / "operations/_training_capability_audit_impl.py"

_FORBIDDEN_SCRIPT_IMPORT_PREFIXES = (
    "gymnasium",
    "numpy",
    "torch",
    "stable_baselines3",
    "sb3_contrib",
    "trade_rl.data",
    "trade_rl.integrations.binance",
    "trade_rl.integrations.sb3_training",
    "trade_rl.rl.actions",
    "trade_rl.rl.environment",
    "trade_rl.rl.export",
    "trade_rl.rl.observations",
    "trade_rl.rl.training",
    "trade_rl.simulation.execution",
    "trade_rl.strategies.trend",
)


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _import_modules(path: Path) -> frozenset[str]:
    modules: set[str] = set()
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return frozenset(modules)


def _top_level_definitions(path: Path) -> frozenset[str]:
    return frozenset(
        node.name
        for node in _tree(path).body
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
    )


def test_training_capability_audit_has_package_owned_public_boundary() -> None:
    assert PUBLIC.is_file()
    assert PRIVATE.is_file()
    public_source = PUBLIC.read_text(encoding="utf-8")
    assert "def run_training_capability_audit(" in public_source
    assert "_training_capability_audit_impl" in public_source
    assert _top_level_definitions(PUBLIC) == frozenset({"run_training_capability_audit"})


def test_training_capability_audit_script_is_a_thin_adapter() -> None:
    imports = _import_modules(SCRIPT)
    assert "trade_rl.operations.training_capability_audit" in imports
    for prefix in _FORBIDDEN_SCRIPT_IMPORT_PREFIXES:
        assert not any(
            module == prefix or module.startswith(f"{prefix}.") for module in imports
        ), prefix
    assert _top_level_definitions(SCRIPT) == frozenset({"main"})


def test_lower_production_layers_do_not_import_training_capability_audit() -> None:
    forbidden = {
        "trade_rl.operations.training_capability_audit",
        "trade_rl.operations._training_capability_audit_impl",
    }
    for path in sorted(PYTHON_SOURCE_ROOT.rglob("*.py")):
        if path in {PUBLIC, PRIVATE}:
            continue
        assert _import_modules(path).isdisjoint(forbidden), path
```

- [ ] **Step 2: Run the new architecture test and verify RED**

Run:

```bash
uv run pytest -q tests/architecture/test_training_capability_audit_owner.py
```

Expected on the pre-refactor implementation: FAIL because the public/private operations files do not exist and the script still owns/imports application-grade training logic. Record the exact failure count and commit SHA as RED evidence.

- [ ] **Step 3: Commit the RED architecture contract without production changes**

```bash
git add tests/architecture/test_training_capability_audit_owner.py
git commit -m "test: require package-owned training capability audit"
```

The commit must contain no `trade_rl/` production change.

---

### Task 2: Pin report and CLI behavior before moving implementation

**Files:**
- Create: `tests/operations/test_training_capability_audit.py`
- Create: `tests/scripts/test_run_training_capability_audit.py`
- Test: both files above

**Interfaces:**
- Consumes: planned public API `run_training_capability_audit(output_root: Path) -> dict[str, object]` and script symbol of the same name imported from the public facade.
- Produces: deterministic report/file/digest contract and a script adapter contract that can be tested without executing real training.

- [ ] **Step 1: Write the package report contract test**

Create `tests/operations/test_training_capability_audit.py`. Import the private implementation only to substitute expensive probe functions; exercise the public facade for the observable result:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from trade_rl.artifacts.hashing import content_digest
from trade_rl.operations import _training_capability_audit_impl as impl
from trade_rl.operations.training_capability_audit import run_training_capability_audit


class _TrainingResult:
    pass


def test_run_training_capability_audit_preserves_report_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "audit"
    root.mkdir()
    (root / "stale.txt").write_text("stale", encoding="utf-8")

    results = {name: _TrainingResult() for name in ("ppo", "sac", "td3", "tqc")}

    def train_algorithm(output_root: Path, algorithm: str):
        assert output_root == root
        return {"algorithm": algorithm, "status": "pass"}, results[algorithm]

    monkeypatch.setattr(impl, "_train_algorithm", train_algorithm)
    monkeypatch.setattr(impl, "_behavior_cloning_training", lambda _: {"status": "pass"})
    monkeypatch.setattr(impl, "_export_ppo", lambda _: {"status": "pass"})
    monkeypatch.setattr(
        impl,
        "_resume_replay",
        lambda output_root, source: {
            "source_matches_sac": source is results["sac"],
            "status": "pass",
        },
    )
    monkeypatch.setattr(impl, "_residual_feature_training", lambda _: {"status": "pass"})
    monkeypatch.setattr(impl, "_sequence_training", lambda _: {"status": "pass"})
    monkeypatch.setattr(impl, "_resume_ppo", lambda _: {"status": "pass"})

    report = run_training_capability_audit(root)

    assert not (root / "stale.txt").exists()
    assert report["schema_version"] == "full_training_capability_audit_v1"
    algorithms = cast(dict[str, object], report["algorithms"])
    replay_resume = cast(dict[str, object], report["replay_resume"])
    assert set(algorithms) == {"ppo", "sac", "td3", "tqc"}
    assert replay_resume["source_matches_sac"] is True

    unsigned = dict(report)
    digest = unsigned.pop("digest")
    assert digest == content_digest(unsigned)

    expected_bytes = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    assert (root / "audit-report.json").read_bytes() == expected_bytes
    assert json.loads(expected_bytes) == report
```

- [ ] **Step 2: Write the thin script CLI contract test**

Create `tests/scripts/test_run_training_capability_audit.py`:

```python
from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts import run_training_capability_audit as cli


def test_main_delegates_to_public_operation_and_preserves_stdout(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    output = tmp_path / "audit"
    report = {
        "digest": "d" * 64,
        "schema_version": "full_training_capability_audit_v1",
    }
    observed: list[Path] = []

    def run(path: Path) -> dict[str, object]:
        observed.append(path)
        return report

    monkeypatch.setattr(cli, "run_training_capability_audit", run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_training_capability_audit.py", "--output", str(output)],
    )

    assert cli.main() == 0
    assert observed == [output]
    assert capsys.readouterr().out == json.dumps(report, sort_keys=True) + "\n"


def test_main_preserves_default_output(monkeypatch, capsys) -> None:
    observed: list[Path] = []

    def run(path: Path) -> dict[str, object]:
        observed.append(path)
        return {"schema_version": "full_training_capability_audit_v1"}

    monkeypatch.setattr(cli, "run_training_capability_audit", run)
    monkeypatch.setattr(sys, "argv", ["run_training_capability_audit.py"])

    assert cli.main() == 0
    assert observed == [Path("var/training-capability-audit")]
    capsys.readouterr()
```

- [ ] **Step 3: Run the behavioral tests and verify they are RED for the intended missing boundary**

Run:

```bash
uv run pytest -q \
  tests/operations/test_training_capability_audit.py \
  tests/scripts/test_run_training_capability_audit.py
```

Expected before implementation: collection/import failure for `trade_rl.operations.training_capability_audit` and/or missing `cli.run_training_capability_audit`. The failure must be caused by the absent package boundary, not by a malformed fixture.

- [ ] **Step 4: Commit the RED behavioral contract**

```bash
git add tests/operations/test_training_capability_audit.py tests/scripts/test_run_training_capability_audit.py
git commit -m "test: pin training capability audit contracts"
```

Again, no production file belongs in this RED commit.

---

### Task 3: Move the audit implementation under `trade_rl.operations` and make the script thin

**Files:**
- Create: `trade_rl/operations/_training_capability_audit_impl.py`
- Create: `trade_rl/operations/training_capability_audit.py`
- Modify: `scripts/run_training_capability_audit.py`
- Test: Task 1 and Task 2 test files

**Interfaces:**
- Consumes: lower-level existing training/data/integration APIs already used by the script.
- Produces: `run_training_capability_audit(output_root: Path) -> dict[str, object]` as the only maintained package API for this audit.

- [ ] **Step 1: Move the implementation owner unchanged into the private operations module**

Create `trade_rl/operations/_training_capability_audit_impl.py` by moving all current audit implementation definitions from the script, excluding only `argparse`, `main`, the executable guard, and the shebang. Preserve the current imports needed by those definitions:

```python
from __future__ import annotations

import json
import math
import shutil
from dataclasses import asdict
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from torch import nn

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketDataset
from trade_rl.integrations.binance import binance_multitimeframe_feature_specs
from trade_rl.integrations.sb3_training import StableBaselines3Backend
from trade_rl.rl.actions import ActionSpec, AlphaContract
from trade_rl.rl.environment import ResidualMarketEnv, ResidualMarketEnvConfig
from trade_rl.rl.export import export_policy_actor
from trade_rl.rl.observations import ObservationLayout
from trade_rl.rl.training import PolicyTrainingResult, ResidualTrainingConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.trend import TrendConfig, TrendStrategy
```

Move these existing symbols with their bodies/constant values unchanged:

```text
_ENVIRONMENT_DIGEST
_ACTION_NAMES
_ACTION_SPEC_DIGEST
AuditEnv
_linear_widths
_architecture
_config
_train_algorithm
_resume_ppo
_resume_replay
_export_ppo
_market_dataset
AuditAlphaProvider
AuditFactorProvider
_residual_feature_training
_behavior_cloning_training
_sequence_dataset
_sequence_training
```

Rename only the script-local orchestration entry point from:

```python
def run_audit(output_root: Path) -> dict[str, object]:
```

to:

```python
def run_training_capability_audit(output_root: Path) -> dict[str, object]:
```

Its body must remain behaviorally identical:

```python
def run_training_capability_audit(output_root: Path) -> dict[str, object]:
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    algorithms: dict[str, object] = {}
    results: dict[str, PolicyTrainingResult] = {}
    for algorithm in ("ppo", "sac", "td3", "tqc"):
        record, result = _train_algorithm(output_root, algorithm)
        algorithms[algorithm] = record
        results[algorithm] = result
    report: dict[str, object] = {
        "algorithms": algorithms,
        "behavior_cloning": _behavior_cloning_training(output_root),
        "exports": _export_ppo(output_root),
        "replay_resume": _resume_replay(output_root, results["sac"]),
        "residual_controls": _residual_feature_training(output_root),
        "schema_version": "full_training_capability_audit_v1",
        "sequence": _sequence_training(output_root),
        "training_resume": _resume_ppo(output_root),
    }
    report["digest"] = content_digest(report)
    report_path = output_root / "audit-report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
```

Do not rename private helpers, change constants, reorder report sections, introduce abstractions, or rewrite the synthetic datasets during this move.

- [ ] **Step 2: Add the minimal public operations facade**

Create `trade_rl/operations/training_capability_audit.py`:

```python
from __future__ import annotations

from pathlib import Path

from trade_rl.operations._training_capability_audit_impl import (
    run_training_capability_audit as _run_training_capability_audit,
)


def run_training_capability_audit(output_root: Path) -> dict[str, object]:
    return _run_training_capability_audit(output_root)
```

Do not re-export `AuditEnv` or any private per-probe helper.

- [ ] **Step 3: Reduce the script to the exact CLI adapter**

Replace `scripts/run_training_capability_audit.py` with:

```python
#!/usr/bin/env python3
"""Execute short real-training probes for every maintained learning backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trade_rl.operations.training_capability_audit import (
    run_training_capability_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("var/training-capability-audit"),
    )
    args = parser.parse_args()
    report = run_training_capability_audit(args.output)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the focused tests to verify GREEN**

Run:

```bash
uv run pytest -q \
  tests/architecture/test_training_capability_audit_owner.py \
  tests/operations/test_training_capability_audit.py \
  tests/scripts/test_run_training_capability_audit.py
```

Expected: all focused tests PASS. If a test itself is wrong, change it only after comparing it to the pre-refactor implementation and the design contract; never adjust an expectation merely to match new implementation output.

- [ ] **Step 5: Run focused static checks**

Run:

```bash
uv run ruff check \
  trade_rl/operations/training_capability_audit.py \
  trade_rl/operations/_training_capability_audit_impl.py \
  scripts/run_training_capability_audit.py \
  tests/architecture/test_training_capability_audit_owner.py \
  tests/operations/test_training_capability_audit.py \
  tests/scripts/test_run_training_capability_audit.py
uv run ruff format --check \
  trade_rl/operations/training_capability_audit.py \
  trade_rl/operations/_training_capability_audit_impl.py \
  scripts/run_training_capability_audit.py \
  tests/architecture/test_training_capability_audit_owner.py \
  tests/operations/test_training_capability_audit.py \
  tests/scripts/test_run_training_capability_audit.py
uv run mypy \
  trade_rl/operations/training_capability_audit.py \
  trade_rl/operations/_training_capability_audit_impl.py \
  tests/operations/test_training_capability_audit.py \
  tests/scripts/test_run_training_capability_audit.py
```

Expected: PASS. Fix newly exposed typing errors with accurate annotations or narrowing; do not add broad suppressions to hide extraction defects.

- [ ] **Step 6: Commit the implementation move**

```bash
git add \
  trade_rl/operations/training_capability_audit.py \
  trade_rl/operations/_training_capability_audit_impl.py \
  scripts/run_training_capability_audit.py
git commit -m "refactor: move training capability audit into operations"
```

Keep test commits separate from the implementation commit so the RED→GREEN history remains inspectable.

---

### Task 4: Falsify the boundary and verify the exact final HEAD

**Files:**
- Modify tests only if falsification reveals a missing acceptance criterion.
- Do not modify `.github/workflows/full-training-capability-audit.yml` unless an actual contract failure proves it necessary.

**Interfaces:**
- Consumes: all previous tasks.
- Produces: exact-head evidence that architecture, runtime audit behavior, static gates, regression suite, and CI agree on one final commit.

- [ ] **Step 1: Run the exact repository static/architecture gates**

Run the commands currently defined in `.github/workflows/ci.yml`:

```bash
uv run python .github/check_workflow_security.py .
uv run ruff check --diff .
uv run ruff format --check --diff .
uv run mypy .
uv run lint-imports
uv run vulture trade_rl tests --min-confidence 100
uv run pytest -q tests/architecture
```

Record exact final Import Linter contract/file/dependency counts and Mypy source counts from this HEAD; do not copy counts from another PR.

- [ ] **Step 2: Execute the real training-capability audit command**

Install the same dependency surface as CI/audit first:

```bash
uv sync --extra dev --extra train-sb3 --extra export
```

Then run the exact workflow command:

```bash
rm -rf var/training-capability-audit
uv run python scripts/run_training_capability_audit.py \
  --output var/training-capability-audit \
  > var/training-capability-audit.stdout.json
```

Validate independently:

```python
import json
from pathlib import Path

path = Path("var/training-capability-audit/audit-report.json")
payload = json.loads(path.read_text(encoding="utf-8"))
assert payload["schema_version"] == "full_training_capability_audit_v1"
assert len(payload["digest"]) == 64
assert set(payload["algorithms"]) == {"ppo", "sac", "td3", "tqc"}
for record in payload["algorithms"].values():
    assert record["status"] == "pass"
    assert record["actual_timesteps"] >= 16
    assert record["checkpoint_count"] >= 1
for name in (
    "behavior_cloning",
    "exports",
    "replay_resume",
    "residual_controls",
    "sequence",
    "training_resume",
):
    assert payload[name]["status"] == "pass"
```

This is the primary Integration Test Oracle for proving the move did not omit a real probe. Mocked unit tests are insufficient for this gate.

- [ ] **Step 3: Run full regression and coverage gates exactly as CI defines them**

Run:

```bash
uv run pytest -q \
  tests/integrations/test_sb3_training.py::test_backend_resumes_ppo_checkpoint_to_requested_total \
  tests/serving/test_sb3_loader.py::test_structured_sb3_loader_rebuilds_native_sequence_observation
uv run pytest -q --cov=trade_rl --cov-branch --cov-report=term-missing --cov-report=json:coverage.json
uv run python .github/check_critical_coverage.py coverage.json pyproject.toml
```

The standard GitHub CI must additionally pass its frontend tests/typecheck/build/bundle/layout jobs, Ubuntu/Windows compatibility matrix, training-image build, image identity capture, packaged non-root runtime probe, and package/uv identity checks on the same exact final HEAD.

- [ ] **Step 4: Perform falsification review from the original spec**

Inspect the final diff and explicitly verify:

```text
scripts/run_training_capability_audit.py
  - no Gymnasium / NumPy / Torch / SB3 imports
  - no AuditEnv, synthetic dataset, per-algorithm config, resume, export, residual, BC, sequence logic
  - only argparse/json/Path + public operations API + main/guard

trade_rl/operations/training_capability_audit.py
  - only run_training_capability_audit is defined as public behavior
  - no private audit fixture surface is re-exported

trade_rl/operations/_training_capability_audit_impl.py
  - every original audit probe is present exactly once
  - report assembly still contains algorithms, behavior_cloning, exports,
    replay_resume, residual_controls, schema_version, sequence, training_resume
  - digest is inserted only after hashing the unsigned report
  - output-root deletion/recreation remains before probe execution

Repository
  - no lower production layer imports training_capability_audit
  - workflow command still points at scripts/run_training_capability_audit.py
  - no unrelated production/config/workflow files changed
```

Construct the specific counterexample mentally: a mocked unit implementation could return all status fields while omitting `_sequence_training` or `_resume_replay`. Confirm that the real audit command and unchanged workflow schema/status assertions execute those real probes and would fail if their artifacts/status were not produced.

- [ ] **Step 5: Verify GitHub Actions on the exact final HEAD**

Push the final HEAD and verify standard `CI` plus the manually dispatchable `Full training capability audit` workflow against that same SHA. The audit workflow must execute its unchanged script command and independent schema assertions. If any final change is made after a successful run, previous workflow evidence is stale and must be rerun before completion claims.

- [ ] **Step 6: Final repository-state check and PR preparation**

Before marking Ready, verify:

```bash
git status --short
git diff main...HEAD --check
git diff --stat main...HEAD
git log --oneline main..HEAD
```

Confirm the diff contains only the approved spec/plan, three tests, two operations modules, and the thin script adapter unless a separately justified test-only correction was required. Check for secrets, generated audit artifacts, debug code, untracked files, and accidental workflow changes.

Create/update the PR with these exact sections:

```text
What
Why
Acceptance Criteria
Design decisions
Scope / Non-goals
RED evidence
Tests / exact final verification
Failure modes / falsification review
CI state
Remaining limitations / empirical non-guarantees
```

Do not merge without explicit user authorization.
