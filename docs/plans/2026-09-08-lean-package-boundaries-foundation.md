# Lean Package Boundaries Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish executable architecture contracts and remove the transitional `trade_rl.domain` layer without changing any observable research, trading, artifact, risk, simulation, or evaluation behavior.

**Architecture:** This phase creates the architecture test harness first, records a complete production-file migration inventory, then moves only the cross-cutting foundations whose ownership is already unambiguous: validation helpers to `trade_rl/_validation.py`, canonical JSON to `trade_rl/artifacts/canonical.py`, and gate records/resolution to `trade_rl/evaluation/gates/`. It deletes `trade_rl/domain` and forwarding-only artifact compatibility once all repository imports are migrated.

**Tech Stack:** Python 3.12, pytest, Ruff, Mypy, GitHub Actions.

**Spec:** `docs/specs/2026-09-08-lean-package-boundaries-design.md`

## Global Constraints

- Implementation base must include PR #436 and PR #437; design-time exact base is `fbfe78553e414ae75674a81ab6bf606d969257e9`.
- Preserve all currently exported package-level APIs unless this plan explicitly removes a deprecated/private path.
- Do not alter `MarketExecutor`, `BookState`, strategy logic, fit scope, evaluation scope, candidate artifact schemas, thresholds, or numerical behavior.
- `trade_rl/_validation.py` must be standard-library-only.
- `trade_rl/artifacts` must not import `data`, `risk`, `simulation`, `strategies`, `evaluation`, or `integrations`.
- No `trade_rl.domain` compatibility package or forwarding module may survive.
- Same-head CI, not an older run, is the integration oracle.

---

## Task 1: Add migration inventory and architecture RED contracts

**Files:**
- Create: `docs/plans/2026-09-08-lean-package-file-inventory.md`
- Create: `tests/architecture/__init__.py`
- Create: `tests/architecture/test_lean_package_layout.py`
- Create: `tests/architecture/test_lean_dependency_boundaries.py`
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Produces a complete pre-refactor KEEP/MOVE/DELETE table.
- Produces `collect_trade_rl_imports(path: Path) -> set[str]` inside the test module for dependency assertions.

- [ ] **Step 1: Generate the exact production-file inventory before any move**

Run:

```bash
find trade_rl -type f -name '*.py' | sort
```

Copy every returned path into `docs/plans/2026-09-08-lean-package-file-inventory.md` and classify each line with one exact action:

```text
KEEP   trade_rl/__init__.py
KEEP   trade_rl/_version.py
DELETE trade_rl/_source_checkout.py : no maintained caller on exact base
MOVE   trade_rl/domain/common.py -> trade_rl/_validation.py
MOVE   trade_rl/domain/canonical_json.py -> trade_rl/artifacts/canonical.py
MOVE   trade_rl/domain/evaluation.py -> trade_rl/evaluation/gates/models.py
```

The inventory must contain the entire production tree, not only files changed in this phase. Later plans may update destinations only if new dependency evidence proves the design wrong.

- [ ] **Step 2: Write the layout RED test**

Create `tests/architecture/test_lean_package_layout.py` with:

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "trade_rl"


def test_transitional_domain_package_is_absent() -> None:
    assert not (PACKAGE / "domain").exists()


def test_validation_and_canonical_owners_exist() -> None:
    assert (PACKAGE / "_validation.py").is_file()
    assert (PACKAGE / "artifacts" / "canonical.py").is_file()


def test_evaluation_gate_package_exists() -> None:
    gates = PACKAGE / "evaluation" / "gates"
    assert (gates / "__init__.py").is_file()
    assert (gates / "models.py").is_file()
    assert (gates / "resolve.py").is_file()


def test_forwarding_artifact_codec_is_absent() -> None:
    assert not (PACKAGE / "artifacts" / "codec.py").exists()
```

- [ ] **Step 3: Write the AST dependency RED test**

Create `tests/architecture/test_lean_dependency_boundaries.py`:

```python
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "trade_rl"


def collect_trade_rl_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names if alias.name.startswith("trade_rl"))
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("trade_rl"):
                result.add(node.module)
    return result


def test_no_production_imports_retired_domain() -> None:
    offenders: list[str] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if any(name.startswith("trade_rl.domain") for name in collect_trade_rl_imports(path)):
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_artifacts_do_not_depend_on_upper_layers() -> None:
    forbidden = (
        "trade_rl.data",
        "trade_rl.risk",
        "trade_rl.simulation",
        "trade_rl.strategies",
        "trade_rl.evaluation",
        "trade_rl.integrations",
    )
    offenders: list[str] = []
    for path in sorted((PACKAGE / "artifacts").rglob("*.py")):
        imports = collect_trade_rl_imports(path)
        if any(name.startswith(forbidden) for name in imports):
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []
```

- [ ] **Step 4: Run RED**

```bash
uv run pytest -q tests/architecture
```

Expected: FAIL because `trade_rl/domain` exists, `_validation.py`/`artifacts/canonical.py`/`evaluation/gates/` do not yet exist, `artifacts/codec.py` still exists, and production imports still reference `trade_rl.domain`.

- [ ] **Step 5: Make CI exhaustive rather than folder-list based**

Replace the current source/test lists in `.github/workflows/ci.yml` with exactly:

```yaml
- name: Ruff
  run: uv run ruff check trade_rl tests

- name: Format
  run: uv run ruff format --check trade_rl tests

- name: Mypy
  run: uv run mypy trade_rl

- name: Tests
  run: uv run pytest -q tests
```

Keep the package-identity step unchanged.

- [ ] **Step 6: Commit RED contracts**

```bash
git add docs/plans/2026-09-08-lean-package-file-inventory.md tests/architecture .github/workflows/ci.yml
git commit -m "test: define lean package boundary foundation"
```

---

## Task 2: Move generic validation out of `domain`

**Files:**
- Create: `trade_rl/_validation.py`
- Modify every production file returned by `rg -l 'trade_rl\.domain\.common' trade_rl`
- Test: `tests/architecture/test_lean_dependency_boundaries.py`
- Test: existing data/risk/simulation/evaluation tests that consume the validators

**Interfaces:**
- `require_non_empty(value: str, *, field: str) -> str`
- `require_sha256(value: str, *, field: str) -> str`
- `require_git_sha(value: str, *, field: str = "git_commit") -> str`
- `require_aware_datetime(value: datetime, *, field: str) -> datetime`
- `require_unique_non_empty(values: tuple[str, ...], *, field: str) -> tuple[str, ...]`

- [ ] **Step 1: Record all current consumers**

Run:

```bash
rg -n 'trade_rl\.domain\.common' trade_rl tests
```

Expected production consumers include the maintained data, risk, artifact verification, simulation evidence, and walk-forward modules. Save no guessed list: the command output is the migration source of truth on the exact implementation head.

- [ ] **Step 2: Create `_validation.py` with only maintained validators**

Use:

```python
from __future__ import annotations

import re
from datetime import datetime
from typing import Final

_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE: Final = re.compile(r"^[0-9a-f]{40}$")


def require_non_empty(value: str, *, field: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field} must be non-empty")
    return normalized


def require_sha256(value: str, *, field: str) -> str:
    if not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def require_git_sha(value: str, *, field: str = "git_commit") -> str:
    if not _GIT_SHA_RE.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase 40-character Git SHA")
    return value


def require_aware_datetime(value: datetime, *, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value


def require_unique_non_empty(values: tuple[str, ...], *, field: str) -> tuple[str, ...]:
    if not values:
        raise ValueError(f"{field} must not be empty")
    normalized = tuple(require_non_empty(value, field=field) for value in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field} must contain unique values")
    return normalized
```

Do not copy `domain_content_digest()`.

- [ ] **Step 3: Rewrite consumers**

For every result from Step 1, replace imports from `trade_rl.domain.common` with `trade_rl._validation` and do not change call sites.

- [ ] **Step 4: Run nearest tests**

```bash
uv run pytest -q tests/data tests/risk tests/artifacts tests/evaluation tests/simulation tests/architecture
uv run ruff check trade_rl/_validation.py trade_rl tests/architecture
uv run ruff format --check trade_rl/_validation.py trade_rl tests/architecture
uv run mypy trade_rl
```

Expected: existing behavioral tests PASS; architecture still RED only for the not-yet-moved canonical/gate/domain files.

- [ ] **Step 5: Commit**

```bash
git add trade_rl/_validation.py trade_rl tests/architecture
git commit -m "refactor: centralize lean validation helpers"
```

---

## Task 3: Make canonical JSON artifact-owned and delete forwarding codec

**Files:**
- Create: `trade_rl/artifacts/canonical.py`
- Modify: `trade_rl/artifacts/hashing.py`
- Modify: `trade_rl/artifacts/__init__.py`
- Modify every consumer returned by `rg -l 'trade_rl\.(domain\.canonical_json|artifacts\.codec)' trade_rl tests`
- Delete: `trade_rl/artifacts/codec.py`
- Test: `tests/artifacts/test_canonical_json_shared.py`
- Test: `tests/artifacts/test_codec.py`
- Test: `tests/artifacts/test_codec_store_critical_coverage.py`

**Interfaces:**
- `JsonScalar`
- `JsonValue`
- `to_json_value(value: object) -> JsonValue`
- `canonical_json_bytes(value: object) -> bytes`
- `content_digest(value: object) -> str`

- [ ] **Step 1: Strengthen canonical ownership tests before production move**

Change `tests/artifacts/test_canonical_json_shared.py` so it imports only the new authority:

```python
from trade_rl.artifacts.canonical import canonical_json_bytes
```

Add:

```python
from pathlib import Path


def test_forwarding_codec_file_is_removed() -> None:
    root = Path(__file__).resolve().parents[2]
    assert not (root / "trade_rl" / "artifacts" / "codec.py").exists()
```

Run:

```bash
uv run pytest -q tests/artifacts/test_canonical_json_shared.py
```

Expected: FAIL because `trade_rl.artifacts.canonical` does not exist and `codec.py` still exists.

- [ ] **Step 2: Move the canonical implementation byte-for-byte in behavior**

Create `trade_rl/artifacts/canonical.py` from the current `domain/canonical_json.py` implementation. Preserve UTC `Z` datetime normalization, POSIX `Path` encoding, sorted keys, compact separators, finite-float rejection, and string-only mapping keys.

- [ ] **Step 3: Update hashing and all consumers**

`trade_rl/artifacts/hashing.py` must use:

```python
from hashlib import sha256
from trade_rl.artifacts.canonical import canonical_json_bytes


def content_digest(value: object) -> str:
    return sha256(canonical_json_bytes(value)).hexdigest()
```

Run:

```bash
rg -n 'trade_rl\.(domain\.canonical_json|artifacts\.codec)' trade_rl tests
```

Rewrite every maintained result to `trade_rl.artifacts.canonical` and require the command to return no results before deleting `codec.py`.

- [ ] **Step 4: Delete forwarding codec and verify exact encoding behavior**

```bash
uv run pytest -q tests/artifacts tests/architecture
uv run ruff check trade_rl/artifacts tests/artifacts tests/architecture
uv run ruff format --check trade_rl/artifacts tests/artifacts tests/architecture
uv run mypy trade_rl/artifacts
```

Expected: artifact tests PASS; architecture remains RED only for gate/domain migration.

- [ ] **Step 5: Commit**

```bash
git add trade_rl/artifacts trade_rl tests/artifacts tests/architecture
git commit -m "refactor: make canonical encoding artifact-owned"
```

---

## Task 4: Move gate models/resolution into `evaluation/gates`

**Files:**
- Create: `trade_rl/evaluation/gates/__init__.py`
- Create: `trade_rl/evaluation/gates/models.py`
- Create: `trade_rl/evaluation/gates/resolve.py`
- Delete: `trade_rl/evaluation/gates.py`
- Modify: `trade_rl/evaluation/__init__.py`
- Modify: `tests/evaluation/test_gates.py`
- Modify every consumer returned by `rg -n 'trade_rl\.domain\.evaluation|trade_rl\.evaluation\.gates' trade_rl tests`

**Interfaces:**
- `GateCheck`
- `GateDecision`
- `resolve_gate(...) -> GateDecision`
- Preserve `from trade_rl.evaluation import resolve_gate`.

- [ ] **Step 1: Write RED imports for the new package**

Change `tests/evaluation/test_gates.py` imports to:

```python
from trade_rl.evaluation.gates import GateCheck, GateDecision, resolve_gate
```

Add:

```python
def test_gate_types_are_exported_from_evaluation_package() -> None:
    from trade_rl import evaluation

    assert evaluation.resolve_gate is resolve_gate
```

Run:

```bash
uv run pytest -q tests/evaluation/test_gates.py
```

Expected: FAIL because `evaluation/gates` is still a single module and does not export the models.

- [ ] **Step 2: Move `GateCheck` and `GateDecision`**

Create `models.py` with the current comparator logic and dataclass invariants from `domain/evaluation.py`, changing only the helper import to:

```python
from trade_rl._validation import require_aware_datetime, require_non_empty, require_sha256
```

- [ ] **Step 3: Move resolver**

Create `resolve.py` from the existing `evaluation/gates.py` behavior and import models locally:

```python
from trade_rl.evaluation.gates.models import GateCheck, GateDecision
```

Create `evaluation/gates/__init__.py`:

```python
from trade_rl.evaluation.gates.models import GateCheck, GateDecision
from trade_rl.evaluation.gates.resolve import resolve_gate

__all__ = ["GateCheck", "GateDecision", "resolve_gate"]
```

Update `evaluation/__init__.py` to import `resolve_gate` from the package.

- [ ] **Step 4: Rewrite all repository gate-model imports and delete old module**

Run:

```bash
rg -n 'trade_rl\.domain\.evaluation|trade_rl\.evaluation\.gates' trade_rl tests
```

After rewriting intended imports, delete the old file `trade_rl/evaluation/gates.py` only after the `gates/` package exists.

- [ ] **Step 5: Run gate and evaluation regression tests**

```bash
uv run pytest -q tests/evaluation/test_gates.py tests/evaluation tests/architecture
uv run ruff check trade_rl/evaluation tests/evaluation tests/architecture
uv run ruff format --check trade_rl/evaluation tests/evaluation tests/architecture
uv run mypy trade_rl/evaluation
```

Expected: evaluation tests PASS; architecture still RED only because `trade_rl/domain` files have not yet been deleted.

- [ ] **Step 6: Commit**

```bash
git add trade_rl/evaluation tests/evaluation tests/architecture
git commit -m "refactor: colocate evaluation gate contracts"
```

---

## Task 5: Delete `domain`, remove unused source-checkout helper, and close Phase 1

**Files:**
- Delete: `trade_rl/domain/__init__.py`
- Delete: `trade_rl/domain/common.py`
- Delete: `trade_rl/domain/canonical_json.py`
- Delete: `trade_rl/domain/evaluation.py`
- Delete if exact-head search still returns no caller: `trade_rl/_source_checkout.py`
- Modify any last stale imports found by repository search
- Update: `docs/plans/2026-09-08-lean-package-file-inventory.md`

**Interfaces:** final Phase-1 package surface.

- [ ] **Step 1: Prove old implementations have no maintained consumers**

Run exactly:

```bash
rg -n 'trade_rl\.domain|domain_content_digest|source_checkout_root' trade_rl tests README.md docs
```

Expected before deletion: only the old `domain` definitions and, for `_source_checkout.py`, its own definition. If a real maintained caller appears, stop and classify it before deletion; do not add a shim.

- [ ] **Step 2: Delete the retired files**

Delete the four `domain` files. Delete `_source_checkout.py` only when Step 1 proves no maintained caller. Update the inventory action to its final evidence-backed state.

- [ ] **Step 3: Require architecture GREEN**

```bash
uv run pytest -q tests/architecture
```

Expected: PASS.

- [ ] **Step 4: Run Phase-1 full quality gate**

```bash
uv run ruff check trade_rl tests
uv run ruff format --check trade_rl tests
uv run mypy trade_rl
uv run pytest -q tests
```

Expected: all PASS. Record exact test count.

- [ ] **Step 5: Falsification review**

Verify all of the following independently from implementation intent:

```text
no trade_rl/domain directory
no import containing trade_rl.domain
no artifacts/codec.py forwarding module
canonical JSON bytes unchanged for existing fixtures
content_digest unchanged for existing fixtures
GateCheck/GateDecision validation semantics unchanged
from trade_rl.evaluation import resolve_gate still works
CI runs the full tests tree
no strategy/risk/execution numerical source changed except import lines
LICENSE and LICENSES/* unchanged
```

If a violation is found, first add a failing regression, then fix it and re-run Step 4.

- [ ] **Step 6: Commit Phase-1 final deletion**

```bash
git add trade_rl tests docs/plans/2026-09-08-lean-package-file-inventory.md .github/workflows/ci.yml
git commit -m "refactor: delete retired domain compatibility layer"
```

- [ ] **Step 7: Push exact HEAD and verify same-head GitHub Actions**

Confirm the workflow checks out the exact final Phase-1 head SHA. Required jobs/checks: Ruff PASS, Format PASS, Mypy PASS, full tests PASS, package identity PASS.

Do not claim the whole package-boundary cleanup is complete: this Phase proves only architecture contracts plus domain/artifact/gate ownership. Data/Binance, strategy/simulation, and evaluation/docs reorganization remain subsequent phases.
