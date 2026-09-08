Status: Active

# Candidate Run Core Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the existing candidate runner into shared config resolution, in-memory execution, provenance, and immutable artifact boundaries without changing candidate fit/replay semantics or the maintained CLI.

**Architecture:** `candidate.py` becomes a thin compatibility/CLI wrapper. `config.py` owns raw config parsing and dataset resolution, `execute.py` owns the single in-memory call into `run_lean_candidate_suite()`, `provenance.py` owns deterministic code/runtime provenance, and `artifact.py` owns publication/loading/semantic identity. `candidate_suite.py` remains the computation authority and is not behaviorally redesigned in this PR.

**Tech Stack:** Python 3.12, NumPy 1.26.x, standard library `hashlib/json/importlib.metadata/platform/pathlib`, existing `trade_rl.artifacts` canonical/atomic helpers, pytest, Ruff, Mypy.

**Spec:** `docs/specs/2026-09-09-controlled-experiment-loop-v1-design.md`

## Global Constraints

- Preserve the 5 candidates + 3 controls and all fit/evaluation semantics.
- Preserve `feature_available_time <= decision_time`, fit-symbol scope, per-symbol replay, and `MarketExecutor + BookState` economics.
- Preserve current CLI `python -m trade_rl.evaluation.runs.candidate`.
- Preserve `lean_candidate_result_v1` summary semantic payload and raw return arrays.
- Add exactly one evidence companion file: `provenance.json`.
- Candidate artifact roots contain exactly `summary.json`, `returns.npz`, `provenance.json`.
- No new dependency.
- Top-level `trade_rl.evaluation.__all__` remains unchanged.
- PR1 must leave `trade_rl/evaluation/experiments/` absent.
- No completed spec/plan removal in PR1; both remain `Status: Active` for PR2.

---

### Task 1: Establish Run Core architecture and config RED

**Files:**
- Create: `tests/evaluation/test_candidate_run_config.py`
- Modify: `tests/architecture/test_lean_evaluation_layout.py`
- No production files in this task.

**Interfaces:**
- Consumes: current candidate JSON contract from `trade_rl/evaluation/runs/candidate.py`.
- Produces test contract for `CandidateRunConfig`, `parse_candidate_run_config`, `load_candidate_run_config`, `ResolvedCandidateRunSpec`, and required new `runs/*.py` files.

- [ ] **Step 1: Add architecture RED**

Extend the required files in `test_evaluation_responsibility_packages_exist()` with:

```python
"runs/config.py",
"runs/execute.py",
"runs/provenance.py",
"runs/artifact.py",
```

Keep the existing `assert not (PACKAGE / "experiments").exists()` in PR1.

- [ ] **Step 2: Add config parser RED**

Create tests using the existing config shape:

```python
from trade_rl.evaluation.runs.config import (
    CandidateRunConfig,
    parse_candidate_run_config,
)


def raw_config() -> dict[str, object]:
    return {
        "signal_name": "signal",
        "feature_names": ["f0", "f1"],
        "fit_symbol_names": ["BTCUSDT", "ETHUSDT"],
        "fit_cutoff": "2026-01-01T04:00:00",
        "evaluation_start": "2026-01-01T04:00:00",
        "evaluation_stop_exclusive": "2026-01-01T07:00:00",
        "rule_entry_threshold": 0.10,
        "rule_exit_threshold": 0.02,
        "forecast_entry_threshold": 0.01,
        "forecast_exit_threshold": 0.002,
        "ppo_total_timesteps": 256,
        "ppo_seed": 7,
        "gross_budget": 0.5,
        "initial_capital": 1000.0,
    }


def test_parse_candidate_run_config_returns_frozen_semantic_config() -> None:
    config = parse_candidate_run_config(raw_config())
    assert isinstance(config, CandidateRunConfig)
    assert config.feature_names == ("f0", "f1")
    assert config.fit_symbol_names == ("BTCUSDT", "ETHUSDT")
    assert config.ppo_seed == 7
```

Add parameterized tests for unknown key, duplicate feature names, duplicate fit symbols, invalid threshold relation, non-finite numeric, invalid timestamp, non-positive timesteps, negative seed.

- [ ] **Step 3: Run RED**

Run:

```bash
uv run pytest -q tests/evaluation/test_candidate_run_config.py tests/architecture/test_lean_evaluation_layout.py
```

Expected: collection/import failure for missing `trade_rl.evaluation.runs.config` and architecture missing-file failure only. Existing unrelated tests must not be changed.

- [ ] **Step 4: Commit RED**

```bash
git add tests/evaluation/test_candidate_run_config.py tests/architecture/test_lean_evaluation_layout.py
git commit -m "test: define candidate run core boundaries"
```

---

### Task 2: Extract CandidateRunConfig and shared resolution

**Files:**
- Create: `trade_rl/evaluation/runs/config.py`
- Modify: `tests/evaluation/test_candidate_run_config.py`
- Test support may reuse the current `MarketDataset` fixture pattern from `tests/evaluation/test_candidate_run_artifact.py`.

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True, slots=True)
class CandidateRunConfig: ...

@dataclass(frozen=True, slots=True)
class ResolvedCandidateRunSpec:
    dataset_id: str
    dataset_artifact_schema: str
    dataset_artifact_digest: str
    config: CandidateRunConfig
    lean_config: LeanCandidateConfig
    evaluation_start_index: int
    evaluation_stop_index: int


def parse_candidate_run_config(raw: Mapping[str, object]) -> CandidateRunConfig: ...
def load_candidate_run_config(path: str | Path) -> CandidateRunConfig: ...
def resolve_candidate_run_spec(
    dataset: MarketDataset,
    *,
    dataset_artifact_schema: str,
    dataset_artifact_digest: str,
    config: CandidateRunConfig,
) -> ResolvedCandidateRunSpec: ...
```

- [ ] **Step 1: Add resolution RED**

Add a test dataset with `feature_names=("signal", "f1")`, `symbols=("BTCUSDT", "ETHUSDT")`, and exact hourly timestamps. Assert:

```python
spec = resolve_candidate_run_spec(
    dataset,
    dataset_artifact_schema="market_dataset_artifact_v3",
    dataset_artifact_digest="d" * 64,
    config=parse_candidate_run_config(raw),
)
assert spec.dataset_id == dataset.dataset_id
assert spec.lean_config.signal_index == 0
assert spec.lean_config.feature_indices == (0, 1)
assert spec.lean_config.fit_symbol_indices == (0, 1)
assert spec.evaluation_start_index == 4
assert spec.evaluation_stop_index == 7
```

Add fail-closed tests for unknown feature, unknown fit symbol, non-exact start/stop timestamp, and evaluation before fit cutoff.

- [ ] **Step 2: Run RED**

```bash
uv run pytest -q tests/evaluation/test_candidate_run_config.py
```

Expected: parser tests may pass after minimal parser code; new resolution tests fail until `resolve_candidate_run_spec` is implemented.

- [ ] **Step 3: Implement minimal shared config authority**

Move the current private validation/resolution behavior from `candidate.py` into `config.py`; do not create a second divergent implementation. Construct `LeanCandidateConfig` exactly once in `resolve_candidate_run_spec()`.

- [ ] **Step 4: Run targeted GREEN**

```bash
uv run pytest -q tests/evaluation/test_candidate_run_config.py tests/evaluation/test_candidate_suite.py
uv run mypy trade_rl/evaluation/runs/config.py
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add trade_rl/evaluation/runs/config.py tests/evaluation/test_candidate_run_config.py
git commit -m "refactor: extract candidate run config resolution"
```

---

### Task 3: Add in-memory execution boundary

**Files:**
- Create: `trade_rl/evaluation/runs/execute.py`
- Create: `tests/evaluation/test_candidate_run_execute.py`
- Do not change `candidate_suite.py` unless a test proves an import-only adjustment is necessary.

**Interfaces:**
- Consumes `ResolvedCandidateRunSpec`, `MarketDataset`, `run_lean_candidate_suite`.
- Produces:

```python
@dataclass(frozen=True, slots=True)
class CandidateRunResult:
    spec: ResolvedCandidateRunSpec
    symbols: tuple[str, ...]
    comparison: UniversalStrategyComparison


def execute_candidate_run(
    dataset: MarketDataset,
    spec: ResolvedCandidateRunSpec,
) -> CandidateRunResult: ...
```

- [ ] **Step 1: Write execution RED using monkeypatch**

Patch `trade_rl.evaluation.runs.execute.run_lean_candidate_suite` and assert exact arguments:

```python
result = execute_candidate_run(dataset, spec)
assert result.spec is spec
assert result.symbols == dataset.symbols
assert calls["kwargs"] == {
    "start_index": spec.evaluation_start_index,
    "stop_index": spec.evaluation_stop_index,
    "gross_budget": spec.config.gross_budget,
    "initial_capital": spec.config.initial_capital,
    "execution_cost": None,
    "risk": None,
}
```

Add a test that clones/replaces `spec.dataset_id` with another digest-safe string and expects `ValueError("dataset id does not match resolved candidate run spec")`.

- [ ] **Step 2: Run RED**

```bash
uv run pytest -q tests/evaluation/test_candidate_run_execute.py
```

Expected: missing module/API.

- [ ] **Step 3: Implement minimal execute layer**

The implementation body must be only validation + one call to `run_lean_candidate_suite()` + dataclass construction. Do not duplicate fit logic.

- [ ] **Step 4: Run GREEN**

```bash
uv run pytest -q tests/evaluation/test_candidate_run_execute.py tests/evaluation/test_candidate_suite.py
uv run mypy trade_rl/evaluation/runs/execute.py
```

- [ ] **Step 5: Commit**

```bash
git add trade_rl/evaluation/runs/execute.py tests/evaluation/test_candidate_run_execute.py
git commit -m "refactor: add in-memory candidate execution"
```

---

### Task 4: Add deterministic Run provenance

**Files:**
- Create: `trade_rl/evaluation/runs/provenance.py`
- Create: `tests/evaluation/test_candidate_run_provenance.py`

**Interfaces:**
- Produces versioned JSON-compatible provenance payload and digest helpers:

```python
PROVENANCE_SCHEMA = "candidate_run_provenance_v1"


def build_candidate_run_provenance() -> dict[str, object]: ...
def implementation_manifest() -> dict[str, object]: ...
def runtime_environment_manifest() -> dict[str, object]: ...
```

The payload contains `schema_version`, `implementation`, `implementation_digest`, `runtime_environment`, `runtime_environment_digest`.

- [ ] **Step 1: Write source-tree digest RED**

Tests must assert:

```python
payload = build_candidate_run_provenance()
assert payload["schema_version"] == "candidate_run_provenance_v1"
assert len(payload["implementation_digest"]) == 64
assert len(payload["runtime_environment_digest"]) == 64
```

Using a temporary fake package root or private helper input, prove that changing one `.py` byte changes implementation digest while absolute root path changes do not.

- [ ] **Step 2: Write runtime manifest RED**

Assert fixed keys exist for Python, OS, machine, trade-rl, numpy, gymnasium, lightgbm, stable-baselines3, torch. Optional missing package is represented as `None`, not omitted.

- [ ] **Step 3: Run RED**

```bash
uv run pytest -q tests/evaluation/test_candidate_run_provenance.py
```

- [ ] **Step 4: Implement provenance using canonical/content digest**

Use relative POSIX paths and SHA-256 exact source bytes. Never record absolute package root in the digest payload.

- [ ] **Step 5: Run GREEN/static**

```bash
uv run pytest -q tests/evaluation/test_candidate_run_provenance.py
uv run ruff check trade_rl/evaluation/runs/provenance.py tests/evaluation/test_candidate_run_provenance.py
uv run mypy trade_rl/evaluation/runs/provenance.py
```

- [ ] **Step 6: Commit**

```bash
git add trade_rl/evaluation/runs/provenance.py tests/evaluation/test_candidate_run_provenance.py
git commit -m "feat: bind candidate runs to runtime provenance"
```

---

### Task 5: Extract candidate artifact publication/load/semantic identity

**Files:**
- Create: `trade_rl/evaluation/runs/artifact.py`
- Create: `tests/evaluation/test_candidate_run_artifact_identity.py`
- Modify: `tests/evaluation/test_candidate_run_artifact.py`

**Interfaces:**
- Consumes `CandidateRunResult`, provenance payload, existing `atomic_write_bytes`.
- Produces:

```python
@dataclass(frozen=True, slots=True)
class PublishedCandidateRun:
    root: Path
    summary_path: Path
    returns_path: Path
    provenance_path: Path

@dataclass(frozen=True, slots=True)
class LoadedCandidateRun: ...

@dataclass(frozen=True, slots=True)
class CandidateRunArtifactIdentity:
    schema_version: str
    result_schema_version: str
    artifact_digest: str
    summary_file_sha256: str
    summary_file_size: int
    returns_file_sha256: str
    returns_file_size: int
    provenance_file_sha256: str
    provenance_file_size: int


def publish_candidate_run(
    output_root: str | Path,
    result: CandidateRunResult,
    provenance: Mapping[str, object],
) -> PublishedCandidateRun: ...

def load_candidate_run_artifact(root: str | Path) -> LoadedCandidateRun: ...
def inspect_candidate_run_artifact(root: str | Path) -> CandidateRunArtifactIdentity: ...
```

- [ ] **Step 1: Add publication RED**

Move the existing summary assertions from `test_candidate_run_artifact_writes_summary_and_raw_returns` to assert the same `lean_candidate_result_v1` semantic payload plus:

```python
assert artifact.provenance_path == output / "provenance.json"
assert set(path.name for path in output.iterdir()) == {
    "summary.json", "returns.npz", "provenance.json"
}
```

- [ ] **Step 2: Add semantic digest RED**

Create two artifact roots with equal parsed summary/provenance and equal NumPy arrays but differently repacked `returns.npz` bytes. Assert:

```python
assert first.returns_file_sha256 != second.returns_file_sha256
assert first.artifact_digest == second.artifact_digest
```

- [ ] **Step 3: Add malformed/tamper RED**

Tests must reject:

- extra file in root;
- missing provenance;
- symlink for any required file when platform supports symlink creation;
- object dtype array;
- 2-D array;
- NaN/inf return;
- unsupported summary/provenance schema.

- [ ] **Step 4: Run RED**

```bash
uv run pytest -q tests/evaluation/test_candidate_run_artifact_identity.py tests/evaluation/test_candidate_run_artifact.py
```

- [ ] **Step 5: Implement artifact layer**

Summary conversion must be copied/extracted from current behavior without changing strategy/metric field names. Semantic returns digest must sort keys and hash `dtype.str`, shape and contiguous C-order bytes; raw file hashes/sizes remain separate inspection evidence.

- [ ] **Step 6: Run GREEN**

```bash
uv run pytest -q tests/evaluation/test_candidate_run_artifact_identity.py tests/evaluation/test_candidate_run_artifact.py
uv run mypy trade_rl/evaluation/runs/artifact.py
```

- [ ] **Step 7: Commit**

```bash
git add trade_rl/evaluation/runs/artifact.py tests/evaluation/test_candidate_run_artifact_identity.py tests/evaluation/test_candidate_run_artifact.py
git commit -m "refactor: isolate immutable candidate run artifacts"
```

---

### Task 6: Make candidate.py a thin compatibility/CLI wrapper

**Files:**
- Modify: `trade_rl/evaluation/runs/candidate.py`
- Modify: `trade_rl/evaluation/runs/__init__.py`
- Modify: `tests/evaluation/test_candidate_run_artifact.py`
- Create or modify: `tests/evaluation/test_candidate_run_cli.py`

**Interfaces:**
- `run_candidate_artifact()` remains callable with the same keyword arguments.
- `runs.__init__` exports the new Run Core public API but `trade_rl.evaluation.__all__` stays unchanged.

- [ ] **Step 1: Add facade delegation RED**

Monkeypatch `inspect_published_market_dataset_artifact`, `load_market_dataset_artifact`, `load_candidate_run_config`, `resolve_candidate_run_spec`, `execute_candidate_run`, `build_candidate_run_provenance`, and `publish_candidate_run`; assert `run_candidate_artifact()` calls them in that order with the resolved dataset identity and returns the published object.

- [ ] **Step 2: Add CLI compatibility RED**

Patch `run_candidate_artifact()` and assert:

```python
assert main([
    "--dataset", "dataset",
    "--config", "config.json",
    "--output", "result",
]) == 0
```

and captured stdout is exactly the artifact root plus newline.

- [ ] **Step 3: Run RED**

```bash
uv run pytest -q tests/evaluation/test_candidate_run_cli.py tests/evaluation/test_candidate_run_artifact.py
```

- [ ] **Step 4: Refactor candidate.py**

Delete duplicated private parser/resolution/publication implementations after delegation works. Keep only CLI/facade orchestration.

- [ ] **Step 5: Export Run Core API**

`trade_rl.evaluation.runs.__all__` must include the intentional new config/execute/artifact/provenance contracts. Do not touch top-level evaluation exports.

- [ ] **Step 6: Run candidate/evaluation regression**

```bash
uv run pytest -q tests/evaluation tests/strategies tests/architecture/test_lean_evaluation_layout.py
uv run ruff check trade_rl/evaluation/runs tests/evaluation
uv run ruff format --check trade_rl/evaluation/runs tests/evaluation
uv run mypy trade_rl
```

- [ ] **Step 7: Commit**

```bash
git add trade_rl/evaluation/runs tests/evaluation tests/architecture/test_lean_evaluation_layout.py
git commit -m "refactor: make candidate runner a thin facade"
```

---

### Task 7: Documentation, falsification, and PR1 final gate

**Files:**
- Modify: `docs/architecture/package-boundaries.md`
- Modify: `docs/research/current-status.md`
- Keep active: `docs/specs/2026-09-09-controlled-experiment-loop-v1-design.md`
- Keep active: this plan and the CEL plan.

**Interfaces:**
- Documentation records new `runs/{config,execute,provenance,artifact}` ownership and three-file candidate evidence.

- [ ] **Step 1: Update current docs**

`package-boundaries.md` must show:

```text
evaluation/runs/{candidate.py,candidate_suite.py,config.py,execute.py,provenance.py,artifact.py}
```

`current-status.md` must keep the same research status but update candidate output to:

```text
summary.json
returns.npz
provenance.json
```

and state that CEL is still not implemented in PR1.

- [ ] **Step 2: Run semantic falsification**

Compare PR1 base `candidate.py` behavior against final Run Core with a deterministic fake `UniversalStrategyComparison`. Assert old-vs-new parsed summary trees and all raw return arrays are exactly equal; only `provenance.json` is additive. Also compare `candidate_suite.py` against PR1 base and require no non-import semantic change.

- [ ] **Step 3: Run full quality gate**

```bash
uv run ruff check trade_rl tests
uv run ruff format --check trade_rl tests
uv run mypy trade_rl
uv run pytest -q tests
```

Run package identity exactly as CI does:

```bash
expected="$(uv run python -c 'from importlib.metadata import version; print(version("trade-rl"))')"
module="$(uv run python -c 'import trade_rl; print(trade_rl.__version__)')"
test "$module" = "$expected"
```

- [ ] **Step 4: Final diff review**

Verify:

- no `evaluation/experiments` exists;
- no temporary workflow/script/debug file remains;
- `candidate_suite.py` computation contract is unchanged;
- top-level `evaluation.__all__` unchanged;
- only intentional additive provenance artifact shape changed.

- [ ] **Step 5: Commit docs/final permanent tests**

```bash
git add docs tests trade_rl/evaluation/runs
git commit -m "docs: document candidate run core evidence"
```

- [ ] **Step 6: Require exact-head CI**

Open/update Draft PR1 against `main`, verify the workflow checks out the exact final PR1 HEAD, and require Ruff / Format / Mypy / full tests / package identity success before PR1 is considered verified.
