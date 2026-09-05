# Universal Trade RL U0 Universe Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an immutable, artifact-bound Train / Development / Unseen Admission symbol-universe contract that prevents Admission-derived data or statistics from entering future Universal Base RL fitting, normalization, calibration, checkpoint selection, or architecture decisions.

**Architecture:** U0 adds a pure domain role contract, a workflow manifest bound to exact source-data identities, phase-aware access guards, Train-only fit provenance, and future RL run identities. Existing Causal Alpha V9/V10/V11 behavior and economics remain unchanged; later U1-U4 work must obtain symbol scopes from U0 rather than infer them from runtime availability.

**Tech Stack:** Python 3.12, dataclasses, Enum, canonical SHA-256 content digests, strict JSON loaders, pytest, Hypothesis, Ruff, MyPy, Import Linter.

**Spec:** `docs/implementation-plans/specs/2026-08-31-universal-trade-rl-zero-shot-transfer-design.md`

## Global Constraints

- Production operates one user-selected symbol with fixed initial capital; U0 does not introduce multi-symbol portfolio execution.
- Train, Development, and Admission roles are mutually disjoint, non-empty, deterministically sorted, immutable, and digest-bound.
- `BTCUSDT` remains in Train while the existing V4 market-proxy fit requires it.
- Admission symbols contribute no fit rows, labels, normalization, calibration, population thresholds, reward coefficients, architecture decisions, hyperparameters, or checkpoint selection before authorization.
- Admission data may be downloaded and integrity-checked; economic/fitted statistics derived from it remain forbidden before authorization.
- Every available but unused source symbol has an explicit non-empty exclusion reason; silent omission fails closed.
- U0 does not modify V9/V10/V11 targets, gates, rewards, fees, horizons, execution, or r21 artifacts.
- U0 artifacts remain research-only and `NO-GO`.

---

### Task 1: Immutable Symbol-Role Contract

**Files:**
- Create: `trade_rl/domain/universal_trade_rl_universe.py`
- Create: `tests/domain/test_universal_trade_rl_universe.py`

**Interfaces:**
- Produces `UniversalTradeRLSymbolRole`, `UniversalTradeRLSymbolExclusion`, `UniversalTradeRLUniverseConfig`, `role_for()`, and a content digest.
- Consumes only `trade_rl.artifacts.hashing.content_digest`.

- [ ] **Step 1: Write failing tests**

```python
from __future__ import annotations

import pytest

from trade_rl.domain.universal_trade_rl_universe import (
    UniversalTradeRLSymbolExclusion,
    UniversalTradeRLSymbolRole,
    UniversalTradeRLUniverseConfig,
)


def _config() -> UniversalTradeRLUniverseConfig:
    return UniversalTradeRLUniverseConfig(
        train_symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT"),
        development_symbols=("LINKUSDT",),
        admission_symbols=("AVAXUSDT",),
        exclusions=(
            UniversalTradeRLSymbolExclusion(
                symbol="LUNA2USDT",
                reason="insufficient_contiguous_history",
            ),
        ),
    )


def test_roles_are_disjoint_and_digest_bound() -> None:
    config = _config()
    assert config.role_for("BTCUSDT") is UniversalTradeRLSymbolRole.TRAIN
    assert config.role_for("LINKUSDT") is UniversalTradeRLSymbolRole.DEVELOPMENT
    assert config.role_for("AVAXUSDT") is UniversalTradeRLSymbolRole.ADMISSION
    assert config.role_for("LUNA2USDT") is None
    assert len(config.digest) == 64


def test_role_overlap_is_rejected() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        UniversalTradeRLUniverseConfig(
            train_symbols=("BTCUSDT", "ETHUSDT"),
            development_symbols=("ETHUSDT",),
            admission_symbols=("AVAXUSDT",),
        )


def test_btc_market_proxy_must_be_train() -> None:
    with pytest.raises(ValueError, match="BTCUSDT"):
        UniversalTradeRLUniverseConfig(
            train_symbols=("ETHUSDT",),
            development_symbols=("LINKUSDT",),
            admission_symbols=("AVAXUSDT",),
        )
```

- [ ] **Step 2: Verify RED**

```powershell
.venv/Scripts/python.exe -m pytest -q tests/domain/test_universal_trade_rl_universe.py
```

Expected: import failure.

- [ ] **Step 3: Implement the domain types**

```python
class UniversalTradeRLSymbolRole(str, Enum):
    TRAIN = "train"
    DEVELOPMENT = "development"
    ADMISSION = "admission"


@dataclass(frozen=True, slots=True)
class UniversalTradeRLSymbolExclusion:
    symbol: str
    reason: str


@dataclass(frozen=True, slots=True)
class UniversalTradeRLUniverseConfig:
    train_symbols: tuple[str, ...]
    development_symbols: tuple[str, ...]
    admission_symbols: tuple[str, ...]
    exclusions: tuple[UniversalTradeRLSymbolExclusion, ...] = ()
    schema_version: str = "universal_trade_rl_universe_config_v1"
    digest: str = ""
```

Validation must require uppercase canonical symbol syntax, non-empty role groups, sorted/unique groups, pairwise disjoint roles, no assigned/excluded overlap, unique exclusions, non-empty reasons, `BTCUSDT` in Train, exact schema, and digest recomputation. `to_payload()` uses tuples and sorted exclusion payloads.

- [ ] **Step 4: Run GREEN and static checks**

```powershell
.venv/Scripts/python.exe -m pytest -q tests/domain/test_universal_trade_rl_universe.py
.venv/Scripts/python.exe -m ruff check trade_rl/domain/universal_trade_rl_universe.py tests/domain/test_universal_trade_rl_universe.py
.venv/Scripts/python.exe -m ruff format --check trade_rl/domain/universal_trade_rl_universe.py tests/domain/test_universal_trade_rl_universe.py
.venv/Scripts/python.exe -m mypy trade_rl/domain/universal_trade_rl_universe.py
```

- [ ] **Step 5: Commit**

```powershell
git add trade_rl/domain/universal_trade_rl_universe.py tests/domain/test_universal_trade_rl_universe.py
git commit -m "feat: define universal trade rl universe roles"
```

---

### Task 2: Strict Config and Source-Catalog Inputs

**Files:**
- Create: `trade_rl/workflows/universal_trade_rl_universe_config.py`
- Create: `tests/workflows/test_universal_trade_rl_universe_config.py`
- Create: `examples/binance/universal-trade-rl-universe.example.json`
- Create: `examples/binance/universal-trade-rl-source-catalog.example.json`

**Interfaces:**
- Produces `UniversalTradeRLSymbolSource`, `load_universal_trade_rl_universe_config()`, `load_universal_trade_rl_source_catalog()`, and `universal_trade_rl_source_catalog_digest()`.

- [ ] **Step 1: Write failing strict-loader tests**

```python
def test_config_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "universe.json"
    path.write_text(json.dumps({
        "schema_version": "universal_trade_rl_universe_config_v1",
        "train_symbols": ["BTCUSDT"],
        "development_symbols": ["LINKUSDT"],
        "admission_symbols": ["AVAXUSDT"],
        "excluded_symbols": [],
        "unexpected": True,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="exact keys"):
        load_universal_trade_rl_universe_config(path)


def test_catalog_rejects_unsorted_symbols(tmp_path: Path) -> None:
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps({
        "schema_version": "universal_trade_rl_source_catalog_v1",
        "symbols": [
            {"symbol": "ETHUSDT", "dataset_digest": "a" * 64,
             "first_timestamp_ns": 1, "last_timestamp_ns": 100,
             "row_count": 100},
            {"symbol": "BTCUSDT", "dataset_digest": "b" * 64,
             "first_timestamp_ns": 1, "last_timestamp_ns": 100,
             "row_count": 100},
        ],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="sorted"):
        load_universal_trade_rl_source_catalog(path)
```

- [ ] **Step 2: Verify RED**

```powershell
.venv/Scripts/python.exe -m pytest -q tests/workflows/test_universal_trade_rl_universe_config.py
```

- [ ] **Step 3: Implement exact-key loaders and source records**

```python
@dataclass(frozen=True, slots=True)
class UniversalTradeRLSymbolSource:
    symbol: str
    dataset_digest: str
    first_timestamp_ns: int
    last_timestamp_ns: int
    row_count: int
```

Require SHA-256 dataset digest, non-negative first timestamp, strictly later last timestamp, positive row count, sorted unique records, no bool-as-int, and schemas `universal_trade_rl_universe_config_v1` / `universal_trade_rl_source_catalog_v1`. Config root keys are exactly `schema_version`, `train_symbols`, `development_symbols`, `admission_symbols`, `excluded_symbols`; exclusion keys are exactly `symbol`, `reason`.

- [ ] **Step 4: Add illustrative example JSON**

Universe example uses Train `BTCUSDT`, `ETHUSDT`, `SOLUSDT`; Development `LINKUSDT`; Admission `AVAXUSDT`; explicit exclusion `LUNA2USDT`. Source example lists all five assigned/excluded symbols in lexicographic order with distinct valid digests and increasing UTC-nanosecond bounds. These examples are not a production role decision.

- [ ] **Step 5: Verify examples and checks**

```powershell
.venv/Scripts/python.exe -m pytest -q tests/workflows/test_universal_trade_rl_universe_config.py
.venv/Scripts/python.exe -c "from pathlib import Path; from trade_rl.workflows.universal_trade_rl_universe_config import load_universal_trade_rl_universe_config, load_universal_trade_rl_source_catalog; load_universal_trade_rl_universe_config(Path('examples/binance/universal-trade-rl-universe.example.json')); load_universal_trade_rl_source_catalog(Path('examples/binance/universal-trade-rl-source-catalog.example.json'))"
.venv/Scripts/python.exe -m ruff check trade_rl/workflows/universal_trade_rl_universe_config.py tests/workflows/test_universal_trade_rl_universe_config.py
.venv/Scripts/python.exe -m mypy trade_rl/workflows/universal_trade_rl_universe_config.py
```

- [ ] **Step 6: Commit**

```powershell
git add trade_rl/workflows/universal_trade_rl_universe_config.py tests/workflows/test_universal_trade_rl_universe_config.py examples/binance/universal-trade-rl-universe.example.json examples/binance/universal-trade-rl-source-catalog.example.json
git commit -m "feat: load universal trade rl universe inputs"
```

---

### Task 3: Artifact-Bound Universe Manifest

**Files:**
- Create: `trade_rl/workflows/universal_trade_rl_universe_manifest.py`
- Create: `tests/workflows/test_universal_trade_rl_universe_manifest.py`

**Interfaces:**
- Produces `UniversalTradeRLUniverseEntry`, `UniversalTradeRLUniverseManifest`, `build_universal_trade_rl_universe_manifest()`, and `from_payload()`.

- [ ] **Step 1: Write RED tests for complete assignment**

```python
def test_manifest_rejects_unassigned_available_symbol() -> None:
    with pytest.raises(ValueError, match="unassigned"):
        build_universal_trade_rl_universe_manifest(
            config=_config_without_doge(),
            sources=(
                _source("AVAXUSDT", "a"),
                _source("BTCUSDT", "b"),
                _source("DOGEUSDT", "c"),
                _source("LINKUSDT", "d"),
            ),
        )


def test_manifest_rejects_missing_admission_source() -> None:
    with pytest.raises(ValueError, match="missing configured symbol"):
        build_universal_trade_rl_universe_manifest(
            config=_config(),
            sources=(_source("BTCUSDT", "b"), _source("LINKUSDT", "d")),
        )
```

- [ ] **Step 2: Verify RED**

```powershell
.venv/Scripts/python.exe -m pytest -q tests/workflows/test_universal_trade_rl_universe_manifest.py
```

- [ ] **Step 3: Implement entries and manifest**

Each entry carries `symbol`, role or exclusion, dataset digest, first/last timestamp, and row count. Assigned entries have role and no exclusion reason; excluded entries have no role and a reason. All available sources appear once, all configured symbols exist, no source remains unassigned, entries are sorted, and the manifest binds config digest, source-catalog digest, complete entries, schema `universal_trade_rl_universe_manifest_v1`, and artifact digest.

- [ ] **Step 4: Add round-trip and tamper tests**

Mutating any role, dataset digest, row count, timestamp, or exclusion reason in a serialized payload must make `UniversalTradeRLUniverseManifest.from_payload()` fail on digest or contract validation.

- [ ] **Step 5: Run GREEN/static checks and commit**

```powershell
.venv/Scripts/python.exe -m pytest -q tests/workflows/test_universal_trade_rl_universe_manifest.py
.venv/Scripts/python.exe -m ruff check trade_rl/workflows/universal_trade_rl_universe_manifest.py tests/workflows/test_universal_trade_rl_universe_manifest.py
.venv/Scripts/python.exe -m mypy trade_rl/workflows/universal_trade_rl_universe_manifest.py
git add trade_rl/workflows/universal_trade_rl_universe_manifest.py tests/workflows/test_universal_trade_rl_universe_manifest.py
git commit -m "feat: bind universal trade rl universe manifest"
```

---

### Task 4: Phase-Aware Access and Admission Authorization

**Files:**
- Create: `trade_rl/workflows/universal_trade_rl_universe_access.py`
- Create: `tests/workflows/test_universal_trade_rl_universe_access.py`

**Interfaces:**
- Produces `UniversalTradeRLAccessPhase`, `UniversalTradeRLAdmissionAuthorization`, `UniversalTradeRLUniverseAccess`, and scope validators.

- [ ] **Step 1: Write RED tests**

```python
def test_development_can_evaluate_but_not_fit() -> None:
    access = UniversalTradeRLUniverseAccess.for_phase(
        manifest=_manifest_fixture(),
        phase=UniversalTradeRLAccessPhase.DEVELOPMENT,
    )
    assert access.fit_symbols == ("BTCUSDT", "ETHUSDT")
    assert access.evaluation_symbols == ("LINKUSDT",)
    with pytest.raises(PermissionError, match="Train-only"):
        access.require_fit_scope(access.evaluation_symbols)


def test_admission_requires_matching_authorization() -> None:
    manifest = _manifest_fixture()
    with pytest.raises(PermissionError, match="authorization"):
        UniversalTradeRLUniverseAccess.for_phase(
            manifest=manifest,
            phase=UniversalTradeRLAccessPhase.ADMISSION,
        )


def test_admission_remains_illegal_for_fit_after_opening() -> None:
    access = _authorized_admission_access()
    with pytest.raises(PermissionError, match="Train-only"):
        access.require_fit_scope(access.admission_symbols)
```

- [ ] **Step 2: Verify RED**

```powershell
.venv/Scripts/python.exe -m pytest -q tests/workflows/test_universal_trade_rl_universe_access.py
```

- [ ] **Step 3: Implement immutable phase access**

Phases are `TRAIN`, `DEVELOPMENT`, `ADMISSION`. Admission authorization binds universe manifest digest, frozen generation digest, and Selection evidence digest using schema `universal_trade_rl_admission_authorization_v1`. TRAIN exposes Train fit symbols only. DEVELOPMENT exposes Train fit symbols and Development evaluation symbols. ADMISSION requires matching authorization, exposes no fit symbols, and exposes Admission evaluation symbols. No mutable `open()` method is permitted; opening creates a new immutable object.

- [ ] **Step 4: Add mismatch tests**

Reject wrong universe digest, authorization supplied to Train/Development, changed frozen generation digest, changed Selection digest, unsorted input scopes, and Development/Admission symbols in any fit/normalization/calibration scope.

- [ ] **Step 5: Run checks and commit**

```powershell
.venv/Scripts/python.exe -m pytest -q tests/workflows/test_universal_trade_rl_universe_access.py
.venv/Scripts/python.exe -m ruff check trade_rl/workflows/universal_trade_rl_universe_access.py tests/workflows/test_universal_trade_rl_universe_access.py
.venv/Scripts/python.exe -m mypy trade_rl/workflows/universal_trade_rl_universe_access.py
git add trade_rl/workflows/universal_trade_rl_universe_access.py tests/workflows/test_universal_trade_rl_universe_access.py
git commit -m "feat: enforce universal trade rl universe access"
```

---

### Task 5: Train-Only Fit/Statistics Provenance

**Files:**
- Create: `trade_rl/workflows/universal_trade_rl_data_provenance.py`
- Create: `tests/workflows/test_universal_trade_rl_data_provenance.py`

**Interfaces:**
- Produces `UniversalTradeRLFitPurpose`, `UniversalTradeRLFitProvenance`, `build_universal_trade_rl_fit_provenance()`, and `require_universal_trade_rl_train_only_provenance()`.

- [ ] **Step 1: Write RED tests**

```python
def test_normalization_provenance_is_train_only() -> None:
    evidence = build_universal_trade_rl_fit_provenance(
        manifest=_manifest_fixture(),
        access=_train_access(),
        purpose=UniversalTradeRLFitPurpose.FEATURE_NORMALIZATION,
        source_symbols=("BTCUSDT", "ETHUSDT"),
        knowledge_cutoff=10_000,
    )
    assert evidence.source_symbols == ("BTCUSDT", "ETHUSDT")


def test_calibration_rejects_development_symbol() -> None:
    with pytest.raises(PermissionError, match="Train-only"):
        build_universal_trade_rl_fit_provenance(
            manifest=_manifest_fixture(),
            access=_development_access(),
            purpose=UniversalTradeRLFitPurpose.CALIBRATION,
            source_symbols=("BTCUSDT", "LINKUSDT"),
            knowledge_cutoff=10_000,
        )
```

- [ ] **Step 2: Verify RED**

```powershell
.venv/Scripts/python.exe -m pytest -q tests/workflows/test_universal_trade_rl_data_provenance.py
```

- [ ] **Step 3: Implement purpose/provenance contracts**

Purposes are `FEATURE_NORMALIZATION`, `FORECAST_FIT`, `CALIBRATION`, `POPULATION_THRESHOLD_FIT`, `REWARD_COEFFICIENT_FIT`, `RL_TRAINING`. Provenance binds purpose, universe manifest digest, sorted source symbols, each symbol's source dataset digest, positive knowledge cutoff, schema `universal_trade_rl_fit_provenance_v1`, and artifact digest. Builders call `require_fit_scope()` before resolving source identities. Validation reconstructs expected source identities from the manifest rather than trusting payload claims.

- [ ] **Step 4: Add cross-generation/source-drift tests**

Provenance from manifest A must fail against manifest B even with the same symbol names. Changing any source dataset digest must change the provenance digest. Admission must fail for every purpose.

- [ ] **Step 5: Run checks and commit**

```powershell
.venv/Scripts/python.exe -m pytest -q tests/workflows/test_universal_trade_rl_data_provenance.py
.venv/Scripts/python.exe -m ruff check trade_rl/workflows/universal_trade_rl_data_provenance.py tests/workflows/test_universal_trade_rl_data_provenance.py
.venv/Scripts/python.exe -m mypy trade_rl/workflows/universal_trade_rl_data_provenance.py
git add trade_rl/workflows/universal_trade_rl_data_provenance.py tests/workflows/test_universal_trade_rl_data_provenance.py
git commit -m "feat: bind universal trade rl train provenance"
```

---

### Task 6: Future RL Run Identity

**Files:**
- Create: `trade_rl/workflows/universal_trade_rl_run_identity.py`
- Create: `tests/workflows/test_universal_trade_rl_run_identity.py`

**Interfaces:**
- Produces `UniversalTradeRLRunStage`, `UniversalTradeRLRunIdentity`, and `from_payload()`.

- [ ] **Step 1: Write RED tests**

```python
def test_admission_identity_requires_authorization() -> None:
    with pytest.raises(ValueError, match="authorization"):
        UniversalTradeRLRunIdentity(
            stage=UniversalTradeRLRunStage.ZERO_SHOT_ADMISSION,
            universe_manifest_digest="a" * 64,
            model_config_digest="b" * 64,
            fit_provenance_digests=("c" * 64,),
        )


def test_training_identity_forbids_admission_authorization() -> None:
    with pytest.raises(ValueError, match="forbid"):
        UniversalTradeRLRunIdentity(
            stage=UniversalTradeRLRunStage.BASE_TRAINING,
            universe_manifest_digest="a" * 64,
            model_config_digest="b" * 64,
            fit_provenance_digests=("c" * 64,),
            admission_authorization_digest="d" * 64,
        )
```

- [ ] **Step 2: Verify RED**

```powershell
.venv/Scripts/python.exe -m pytest -q tests/workflows/test_universal_trade_rl_run_identity.py
```

- [ ] **Step 3: Implement stage rules**

Stages are `UNIVERSE_MATERIALIZATION`, `BASE_TRAINING`, `DEVELOPMENT_SELECTION`, `ZERO_SHOT_ADMISSION`. Identity binds stage, universe digest, model-config digest, sorted unique fit-provenance digests, optional Admission authorization, schema `universal_trade_rl_run_identity_v1`, and artifact digest. Admission requires authorization; earlier stages forbid it. Fit provenance is required for all stages except materialization. Transfer fields are deliberately deferred to a versioned U4 identity.

- [ ] **Step 4: Run checks and commit**

```powershell
.venv/Scripts/python.exe -m pytest -q tests/workflows/test_universal_trade_rl_run_identity.py
.venv/Scripts/python.exe -m ruff check trade_rl/workflows/universal_trade_rl_run_identity.py tests/workflows/test_universal_trade_rl_run_identity.py
.venv/Scripts/python.exe -m mypy trade_rl/workflows/universal_trade_rl_run_identity.py
git add trade_rl/workflows/universal_trade_rl_run_identity.py tests/workflows/test_universal_trade_rl_run_identity.py
git commit -m "feat: bind universal trade rl run identity"
```

---

### Task 7: Atomic U0 Materialization CLI

**Files:**
- Create: `trade_rl/workflows/universal_trade_rl_universe_runner.py`
- Create: `tests/workflows/test_universal_trade_rl_universe_runner.py`
- Modify: `pyproject.toml` `[project.scripts]`

**Interfaces:**
- Produces `materialize_universal_trade_rl_universe()` and CLI `trade-rl-universe`.

- [ ] **Step 1: Write RED tests**

```python
def test_cli_materializes_bound_artifacts(tmp_path: Path, capsys) -> None:
    config_path, catalog_path = _write_valid_inputs(tmp_path)
    output = tmp_path / "output"
    result = cli_main([
        "--config", str(config_path),
        "--source-catalog", str(catalog_path),
        "--output-root", str(output),
    ])
    assert result == 0
    universe = json.loads((output / "universe.json").read_text())
    identity = json.loads((output / "identity.json").read_text())
    assert identity["universe_manifest_digest"] == universe["artifact_digest"]
    assert json.loads(capsys.readouterr().out)["status"] == "materialized"


def test_failure_leaves_no_partial_artifacts(tmp_path: Path) -> None:
    config_path, catalog_path = _write_inputs_with_unassigned_symbol(tmp_path)
    output = tmp_path / "output"
    assert cli_main([
        "--config", str(config_path),
        "--source-catalog", str(catalog_path),
        "--output-root", str(output),
    ]) == 5
    assert not (output / "universe.json").exists()
    assert not (output / "identity.json").exists()
```

- [ ] **Step 2: Verify RED**

```powershell
.venv/Scripts/python.exe -m pytest -q tests/workflows/test_universal_trade_rl_universe_runner.py
```

- [ ] **Step 3: Implement materialization**

`materialize_universal_trade_rl_universe(config_path, source_catalog_path, output_root)` loads strict inputs, builds manifest, creates an `UNIVERSE_MATERIALIZATION` identity, validates both, writes temporary canonical JSON, `flush`/`fsync`, then `os.replace`s `universe.json` and `identity.json`. Existing identical artifacts are an idempotent success; existing drift fails closed. CLI arguments are exactly `--config`, `--source-catalog`, `--output-root`; success code 0, invalid/I/O code 5. Terminal JSON includes status, both digests, and `production_status: "NO-GO"`.

- [ ] **Step 4: Register the CLI**

```toml
trade-rl-universe = "trade_rl.workflows.universal_trade_rl_universe_runner:cli_main"
```

- [ ] **Step 5: Add resume/tamper tests**

Verify byte-identical rerun, edited artifact rejection, role-config drift rejection, and source-digest drift rejection.

- [ ] **Step 6: Run checks/build and commit**

```powershell
.venv/Scripts/python.exe -m pytest -q tests/workflows/test_universal_trade_rl_universe_runner.py
.venv/Scripts/python.exe -m ruff check trade_rl/workflows/universal_trade_rl_universe_runner.py tests/workflows/test_universal_trade_rl_universe_runner.py
.venv/Scripts/python.exe -m mypy trade_rl/workflows/universal_trade_rl_universe_runner.py
uv build
git add trade_rl/workflows/universal_trade_rl_universe_runner.py tests/workflows/test_universal_trade_rl_universe_runner.py pyproject.toml
git commit -m "feat: materialize universal trade rl universe"
```

---

### Task 8: Leakage Falsification and Existing-Fit Compatibility

**Files:**
- Create: `tests/workflows/test_universal_trade_rl_universe_isolation.py`

**Interfaces:**
- Consumes all U0 public contracts and proves the combined boundary.

- [ ] **Step 1: Add end-to-end isolation tests**

```python
def test_admission_metadata_can_be_bound_but_not_fit() -> None:
    manifest = _manifest_fixture()
    admission = next(e for e in manifest.entries if e.symbol == "AVAXUSDT")
    assert len(admission.dataset_digest) == 64
    with pytest.raises(PermissionError, match="Train-only"):
        build_universal_trade_rl_fit_provenance(
            manifest=manifest,
            access=_train_access(manifest),
            purpose=UniversalTradeRLFitPurpose.FEATURE_NORMALIZATION,
            source_symbols=("AVAXUSDT",),
            knowledge_cutoff=10_000,
        )


def test_existing_fit_boundary_receives_only_train_symbols() -> None:
    access = _development_access(_manifest_fixture())
    observed: list[tuple[str, ...]] = []
    def fit_spy(*, train_symbols: tuple[str, ...]) -> None:
        observed.append(train_symbols)
    fit_spy(train_symbols=access.fit_symbols)
    assert observed == [("BTCUSDT", "ETHUSDT")]
    assert "LINKUSDT" not in observed[0]
    assert "AVAXUSDT" not in observed[0]
```

- [ ] **Step 2: Add bounded Hypothesis tests**

With `@settings(max_examples=100, deadline=None)`, generate canonical symbol sets and prove overlap rejection, unassigned-source rejection, digest mutation on role/source changes, and impossibility of Admission through `require_fit_scope()`.

- [ ] **Step 3: Run all U0 tests**

```powershell
.venv/Scripts/python.exe -m pytest -q tests/domain/test_universal_trade_rl_universe.py tests/workflows/test_universal_trade_rl_universe_config.py tests/workflows/test_universal_trade_rl_universe_manifest.py tests/workflows/test_universal_trade_rl_universe_access.py tests/workflows/test_universal_trade_rl_data_provenance.py tests/workflows/test_universal_trade_rl_run_identity.py tests/workflows/test_universal_trade_rl_universe_runner.py tests/workflows/test_universal_trade_rl_universe_isolation.py
```

- [ ] **Step 4: Run existing Causal Alpha compatibility**

```powershell
.venv/Scripts/python.exe -m pytest -q tests/workflows/test_universal_causal_alpha_v4_fitting.py tests/workflows/test_universal_causal_alpha_v5_calibration.py tests/workflows/test_universal_causal_alpha_v6_stage_entry.py tests/workflows/test_universal_causal_alpha_v7_stage_entry.py tests/workflows/test_universal_causal_alpha_v9_stage_entry.py tests/workflows/test_universal_causal_alpha_v10_stage_entry.py tests/workflows/test_universal_causal_alpha_v11_stage_entry.py
.venv/Scripts/python.exe -m importlinter
.venv/Scripts/python.exe -m ruff check trade_rl tests
.venv/Scripts/python.exe -m mypy trade_rl/domain/universal_trade_rl_universe.py trade_rl/workflows/universal_trade_rl_universe_config.py trade_rl/workflows/universal_trade_rl_universe_manifest.py trade_rl/workflows/universal_trade_rl_universe_access.py trade_rl/workflows/universal_trade_rl_data_provenance.py trade_rl/workflows/universal_trade_rl_run_identity.py trade_rl/workflows/universal_trade_rl_universe_runner.py
uv build
```

- [ ] **Step 5: Compare full suite against exact base**

Run `.venv/Scripts/python.exe -m pytest -q` with identical dependencies at feature HEAD and exact base. Record passed, failed, skipped, deselected, coverage, and critical coverage. Every additional failure must be fixed or reproduced at base before claiming no regression.

- [ ] **Step 6: Commit**

```powershell
git add tests/workflows/test_universal_trade_rl_universe_isolation.py
git commit -m "test: falsify universal trade rl universe leakage"
```

---

### Task 9: Documentation and U1 Handoff

**Files:**
- Create: `docs/UNIVERSAL_TRADE_RL.md`
- Modify: `README.md`
- Modify the approved spec only for precise implementation corrections, never to broaden U0.

- [ ] **Step 1: Document the contract**

State that deployment is one selected symbol/fixed capital; Base RL is symbol-independent and zero-shot-first; Train fits, Development guides design, Admission opens once; Admission metadata may be integrity-checked but not fitted; exclusions prevent silent cherry-picking; examples are illustrative; U0 does not train RL or prove profit; U1 starts only after a production role config/source catalog materializes to a stable digest.

- [ ] **Step 2: Add the reproducible command**

```powershell
trade-rl-universe `
  --config examples/binance/universal-trade-rl-universe.example.json `
  --source-catalog examples/binance/universal-trade-rl-source-catalog.example.json `
  --output-root var/runs/universal-trade-rl-u0-example
```

- [ ] **Step 3: Link from README without production claims**

- [ ] **Step 4: Run final checks**

```powershell
.venv/Scripts/python.exe -m pytest -q tests/domain/test_universal_trade_rl_universe.py tests/workflows/test_universal_trade_rl_universe_config.py tests/workflows/test_universal_trade_rl_universe_manifest.py tests/workflows/test_universal_trade_rl_universe_access.py tests/workflows/test_universal_trade_rl_data_provenance.py tests/workflows/test_universal_trade_rl_run_identity.py tests/workflows/test_universal_trade_rl_universe_runner.py tests/workflows/test_universal_trade_rl_universe_isolation.py
git diff --check
.venv/Scripts/python.exe -m ruff check trade_rl tests
.venv/Scripts/python.exe -m importlinter
uv build
```

- [ ] **Step 5: Commit**

```powershell
git add docs/UNIVERSAL_TRADE_RL.md README.md
git commit -m "docs: document universal trade rl universe isolation"
```

---

## Final Acceptance Criteria

1. Train, Development, Admission, and explicit exclusions are immutable, disjoint, sorted, and digest-bound.
2. Every source-catalog symbol is assigned or excluded; silent omission fails closed.
3. Assigned symbols carry bound source identity and range evidence.
4. Development and Admission are rejected for every fit/statistics purpose.
5. Development is available for evaluation without contributing fit statistics.
6. Admission requires matching frozen-generation authorization and remains illegal for fitting after opening.
7. Future run identities bind universe, phase, provenance, and authorization state.
8. Exact reruns are idempotent; role/source/artifact tampering fails closed.
9. Existing V4-V11 behavior and economics are unchanged.
10. Full-suite comparison has no unexplained regression relative to exact base.
11. Documentation states U0 is isolation infrastructure, not an economically validated RL model.

## Self-Review Record

- **Spec coverage:** Covers approved U0 role partition, immutable identity, Train-only statistical boundaries, Admission authorization, explicit exclusions, and U1 handoff. U1 observations/reward, U2 RL training, U3 zero-shot economics, and U4 transfer remain out of scope.
- **Placeholder scan:** No TBD/TODO, unnamed interface, or unspecified validation step remains.
- **Type consistency:** Domain, loader, manifest, access, provenance, run identity, and runner names are consistent across tasks.
- **Scope check:** Existing Causal Alpha is compatibility-tested but not retrofitted; U0 establishes the mandatory boundary for future Universal Trade RL work.
