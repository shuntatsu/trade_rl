# Canonical M2 Study Bootstrap v1 Implementation Plan

Status: Active

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fail-closed, reproducible Binance USD-M bootstrap that freezes exact source evidence, publishes one canonical market dataset, creates an immutable Controlled StudyPlan, and stops before baseline execution.

**Architecture:** Add a thin `evaluation.experiments.bootstrap` orchestration layer over existing Binance cache/metadata, data artifact publication, and Controlled Experiment Loop APIs. Strengthen the Binance transport with explicit cache-only mode and a dataset-builder-compatible frozen metadata facade. Build the complete bootstrap under a staging root and publish it with one final rename only after source, dataset, StudyPlan, and provenance identities all validate.

**Tech Stack:** Python 3.12, NumPy, existing `trade_rl` artifact/canonical digest utilities, Binance public REST/Vision integration, pytest/Hypothesis, Ruff, Mypy, uv.

**Spec:** `docs/specs/2026-09-09-canonical-m2-study-bootstrap-design.md`

## Global Constraints

- v1 accepts only Binance USD-M Futures: `market == "usds-m"`.
- `data_stop_exclusive` must be UTC month-start `00:00:00`.
- Bootstrap must not call `run_baseline`, define/run Experiments, select a winner, or access sealed unused-future authorization.
- Bootstrap input has no implicit research defaults and rejects unknown keys.
- `baseline.ppo_seed` is not accepted; `ppo_seeds[0]` is injected as the baseline run seed.
- All source bytes used by the dataset remain inside the final bootstrap root with content evidence.
- After source synchronization, dataset construction must be cache-only; any network attempt is an error.
- Existing candidate-run resolver and `create_study()` remain the sole baseline resolution/StudyPlan authorities.
- Bootstrap start/end implementation and runtime provenance digests must match; mismatch prevents final publication.
- `--output` must not exist; no overwrite, repair, append, or resume behavior in v1.
- Failure must not leave a valid-looking final output root.
- Current docs may contain `docs/specs`/`docs/plans` only while each file says `Status: Active`; completed spec/plan files are removed after durable docs are updated.
- No dependency from `integrations` or `evaluation/runs` to `evaluation/experiments.bootstrap`; no bootstrap dependency on sealed final-test authorization.

---

## File map

### Production files to modify

- `trade_rl/integrations/binance/vision.py` — expose one public interval-duration query used by strict bootstrap preflight alignment.
- `trade_rl/integrations/binance/transport.py` — add backward-compatible `allow_network: bool = True`; reject cache misses/REST calls when false.
- `trade_rl/integrations/binance/metadata.py` — make frozen metadata transport implement dataset builder's `load_exchange_information()` contract.
- `trade_rl/integrations/binance/__init__.py` — export maintained cache/metadata utilities required by the bootstrap boundary.
- `trade_rl/evaluation/experiments/__init__.py` — export only the four approved high-level bootstrap symbols.

### Production files to create

- `trade_rl/evaluation/experiments/bootstrap/__init__.py` — narrow public bootstrap facade.
- `trade_rl/evaluation/experiments/bootstrap/config.py` — strict JSON contract, normalization, preflight validation, canonical config digest.
- `trade_rl/evaluation/experiments/bootstrap/binance.py` — frozen Binance source composition, source roster/digest, metadata evidence payload.
- `trade_rl/evaluation/experiments/bootstrap/workflow.py` — staged end-to-end bootstrap, final graph validation, manifest, inspection.
- `trade_rl/evaluation/experiments/bootstrap/cli.py` — filesystem CLI facade only.

### Tests to create/modify

- `tests/architecture/test_current_docs_layout.py` — remove obsolete "specs directory must not exist" assertion while preserving Active-only ephemeral-doc contract.
- `tests/integrations/test_binance_transport.py` or existing Binance transport test owner — cache-only and frozen metadata compatibility.
- `tests/evaluation/experiments/bootstrap/test_config.py` — strict config contract/property tests.
- `tests/evaluation/experiments/bootstrap/test_binance.py` — source freeze, source roster, cache-only composition.
- `tests/evaluation/experiments/bootstrap/test_workflow.py` — staged success, atomic failure, provenance, dataset/Study binding.
- `tests/evaluation/experiments/bootstrap/test_tamper.py` — self-consistent and simple artifact tamper rejection.
- `tests/evaluation/experiments/bootstrap/test_cli.py` — CLI success/failure semantics with fake source transport seam.
- `tests/architecture/test_lean_dependency_boundaries.py` — dependency direction.
- `tests/architecture/test_lean_evaluation_layout.py` — permanent package layout after implementation.
- `tests/architecture/test_lean_binance_layout.py` — maintained Binance public surface if new public helpers are exported.

### Durable docs to modify at completion

- `docs/architecture/controlled-experiment-loop.md`
- `docs/architecture/package-boundaries.md`
- `docs/research/current-status.md`
- `docs/README.md`
- `docs/AGENTS.md` only if routing needs a new durable entry; otherwise leave unchanged.

### Ephemeral docs to delete at completion

- `docs/specs/2026-09-09-canonical-m2-study-bootstrap-design.md`
- `docs/plans/2026-09-10-canonical-m2-study-bootstrap-implementation.md`

---

### Task 1: Repair the current-docs retention oracle

**Files:**
- Modify: `tests/architecture/test_current_docs_layout.py`

**Interfaces:**
- Consumes: existing `REQUIRED_DOC_FILES`, `EPHEMERAL_DOC_ROOTS`, and `test_docs_tree_contains_current_authorities_and_only_active_ephemeral_docs()`.
- Produces: one non-contradictory docs retention contract: Active Markdown under `docs/specs`/`docs/plans` is allowed; inactive/history/archive docs are rejected.

- [ ] **Step 1: Preserve the observed RED as evidence**

Run the current design-head suite and retain the existing failure:

```bash
uv run pytest -q tests/architecture/test_current_docs_layout.py
```

Expected: the CEL-specific test fails only because `docs/specs` exists while the general Active-ephemeral test accepts it.

- [ ] **Step 2: Remove only the obsolete past-state assertions**

Change:

```python
assert not (DOCS / "specs").exists()
assert not (DOCS / "plans").exists()
```

inside `test_controlled_experiment_loop_is_durable_current_architecture()` to no assertions about ephemeral-directory existence. Do not weaken `test_docs_tree_contains_current_authorities_and_only_active_ephemeral_docs()`.

- [ ] **Step 3: Add an explicit inactive-spec rejection regression**

Add a pure helper or parameterized unit around the retention rule so a Markdown file under an ephemeral root without exact `Status: Active` is rejected. The assertion must inspect content, not merely suffix/path.

- [ ] **Step 4: Run targeted tests**

```bash
uv run pytest -q tests/architecture/test_current_docs_layout.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/architecture/test_current_docs_layout.py
git commit -m "test: align active docs retention contract"
```

---

### Task 2: Add fail-closed cache-only Binance transport primitives

**Files:**
- Modify: `trade_rl/integrations/binance/vision.py`
- Modify: `trade_rl/integrations/binance/transport.py`
- Modify: `trade_rl/integrations/binance/metadata.py`
- Modify: `trade_rl/integrations/binance/__init__.py`
- Test: `tests/integrations/test_binance.py`
- Test: `tests/integrations/test_binance_cache.py`
- Test: `tests/architecture/test_lean_binance_layout.py`

**Interfaces:**
- Produces: `binance_interval_milliseconds(interval: str) -> int`.
- Produces: `BinancePublicTransport(..., allow_network: bool = True)`.
- Produces: `FrozenBinanceExchangeInfoTransport.load_exchange_information(*, market, mode=REST) -> tuple[dict[str, object], str]`.
- Invariant: existing default transport behavior is byte/semantic compatible because `allow_network` defaults to `True`.

- [ ] **Step 1: Write RED tests for cache-only behavior**

Add tests proving:

```python
transport = BinancePublicTransport(cache_root=tmp_path, allow_network=False)
with pytest.raises(BinanceTransportError, match="network access is disabled"):
    transport._request_bytes("https://data.binance.vision/data/...missing.zip")
```

and that a valid cached Vision payload still loads when `allow_network=False` without calling `urllib.request.urlopen`.

Also assert REST-only metadata/funding/klines calls fail before HTTP when network is disabled.

- [ ] **Step 2: Write RED test for frozen metadata builder compatibility**

Freeze one snapshot with a fake delegate, reconstruct without a delegate, then assert:

```python
payload, source = frozen.load_exchange_information(
    market=BinanceMarket.USDS_M,
    mode=BinanceTransportMode.REST,
)
assert payload["symbols"][0]["symbol"] == "BTCUSDT"
assert source == "frozen:exchange-info"
```

Mutating `payload` must not mutate the frozen snapshot.

- [ ] **Step 3: Write RED test for public interval helper**

```python
assert binance_interval_milliseconds("15m") == 900_000
assert binance_interval_milliseconds("1h") == 3_600_000
with pytest.raises(ValueError, match="unsupported Binance interval"):
    binance_interval_milliseconds("7m")
```

- [ ] **Step 4: Run RED tests**

```bash
uv run pytest -q tests/integrations/test_binance.py tests/integrations/test_binance_cache.py tests/architecture/test_lean_binance_layout.py
```

Expected: failures because `allow_network`, compatibility method, and public helper do not exist.

- [ ] **Step 5: Implement minimal lower-layer changes**

In `vision.py`:

```python
def binance_interval_milliseconds(interval: str) -> int:
    return _interval_ms(interval)
```

In `BinancePublicTransport.__init__`:

```python
allow_network: bool = True
...
if not isinstance(allow_network, bool):
    raise ValueError("allow_network must be a boolean")
self.allow_network = allow_network
```

In `_request_bytes`, after a verified cache hit check and before constructing `urllib.request.Request`:

```python
if not self.allow_network:
    raise BinanceTransportError(f"network access is disabled for uncached source: {url}")
```

This single boundary must also block REST because REST routes eventually call `_request_bytes()`.

In frozen metadata transport add:

```python
def load_exchange_information(
    self,
    *,
    market: BinanceMarket | str,
    mode: BinanceTransportMode | str = BinanceTransportMode.REST,
) -> tuple[dict[str, object], str]:
    snapshot = self.load_exchange_information_snapshot(market=market, mode=mode)
    return _mutable_json_object(snapshot.payload), "frozen:exchange-info"
```

Export `FrozenBinanceExchangeInfoTransport`, Vision cache planning/sync/validation APIs used by bootstrap, and `binance_interval_milliseconds` from the maintained Binance facade.

- [ ] **Step 6: Run targeted static and unit gates**

```bash
uv run ruff check trade_rl/integrations/binance tests/integrations tests/architecture/test_lean_binance_layout.py
uv run ruff format --check trade_rl/integrations/binance tests/integrations tests/architecture/test_lean_binance_layout.py
uv run mypy trade_rl/integrations/binance
uv run pytest -q tests/integrations/test_binance.py tests/integrations/test_binance_cache.py tests/architecture/test_lean_binance_layout.py
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add trade_rl/integrations/binance tests/integrations tests/architecture/test_lean_binance_layout.py
git commit -m "feat: add cache-only Binance evidence transport"
```

---

### Task 3: Implement strict canonical bootstrap configuration

**Files:**
- Create: `trade_rl/evaluation/experiments/bootstrap/config.py`
- Create: `tests/evaluation/experiments/bootstrap/__init__.py`
- Create: `tests/evaluation/experiments/bootstrap/test_config.py`
- Modify: `trade_rl/evaluation/experiments/bootstrap/__init__.py` only after Task 6 public-facade work; keep config import direct during this task.

**Interfaces:**
- Produces: `CanonicalM2BootstrapConfig` immutable dataclass.
- Produces: `load_canonical_m2_bootstrap_config(path: str | Path) -> CanonicalM2BootstrapConfig`.
- `CanonicalM2BootstrapConfig.baseline` is a validated `CandidateRunConfig` whose `ppo_seed == ppo_seeds[0]`.
- `CanonicalM2BootstrapConfig.to_payload()` omits baseline `ppo_seed`, preserving a single seed-policy authority.
- `CanonicalM2BootstrapConfig.digest` is `content_digest(to_payload())`.

- [ ] **Step 1: Write strict-schema RED tests**

Cover exact top-level keys and exact nested baseline keys. Unknown and missing keys must fail. Explicit `baseline.ppo_seed` must fail as unknown.

Use one valid fixture with concrete values:

```python
{
  "schema_version": "canonical_m2_bootstrap_config_v1",
  "research_question": "Do maintained candidates beat controls on development data?",
  "market": "usds-m",
  "symbols": ["BTCUSDT", "ETHUSDT"],
  "base_timeframe": "1h",
  "feature_timeframes": ["4h", "1d"],
  "data_start": "2024-01-01T00:00:00+00:00",
  "data_stop_exclusive": "2025-01-01T00:00:00+00:00",
  "baseline": {
    "signal_name": "1h__log_return_24bar",
    "feature_names": ["1h__log_return_24bar", "1h__realized_volatility_24bar"],
    "fit_symbol_names": ["BTCUSDT", "ETHUSDT"],
    "fit_cutoff": "2024-07-01T00:00:00",
    "evaluation_start": "2024-07-01T00:00:00",
    "evaluation_stop_exclusive": "2025-01-01T00:00:00",
    "rule_entry_threshold": 0.01,
    "rule_exit_threshold": 0.005,
    "forecast_entry_threshold": 0.01,
    "forecast_exit_threshold": 0.005,
    "ppo_total_timesteps": 1000,
    "gross_budget": 0.5,
    "initial_capital": 100000.0
  },
  "ppo_seeds": [0, 1],
  "allowed_factors": ["FEATURE_SET"],
  "max_experiments": 8,
  "n_bootstrap": 500,
  "bootstrap_seed": 17
}
```

- [ ] **Step 2: Write temporal/preflight RED tests**

Reject:

- timezone-naive top-level `data_start`/`data_stop_exclusive`;
- non-UTC offsets after normalization policy if not exactly representable as UTC;
- `data_start >= fit_cutoff`;
- `fit_cutoff > evaluation_start`;
- `evaluation_start >= evaluation_stop_exclusive`;
- evaluation stop after data stop;
- non-month-boundary data stop;
- timestamp not aligned to all configured native clocks;
- duplicate feature timeframe/base timeframe duplication;
- non-USD-M market.

- [ ] **Step 3: Write identity/property RED tests**

Using Hypothesis or deterministic permutations, prove JSON object key order and whitespace do not change `config.digest`, while changing one semantic value does.

Assert:

```python
assert config.baseline.ppo_seed == config.ppo_seeds[0]
assert "ppo_seed" not in config.to_payload()["baseline"]
```

- [ ] **Step 4: Run RED**

```bash
uv run pytest -q tests/evaluation/experiments/bootstrap/test_config.py
```

Expected: import failure because `bootstrap/config.py` does not exist.

- [ ] **Step 5: Implement parser using existing validators**

Implementation rules:

- JSON file must be a regular file, not symlink.
- Parse object with `json.loads`; reject non-object.
- Validate exact key sets.
- Parse top-level source timestamps with `datetime.fromisoformat`, require timezone awareness, normalize to UTC.
- Use `binance_interval_milliseconds()` for alignment without duplicating interval tables.
- Convert baseline nested mapping to existing candidate-run config mapping by injecting only:

```python
candidate_payload = dict(baseline_payload)
candidate_payload["ppo_seed"] = ppo_seeds[0]
baseline = parse_candidate_run_config(candidate_payload)
```

- Convert allowed factor strings with `ControlledFactor(value)` and reject duplicates.
- Keep symbol/feature ordering exactly as authored after validation.

- [ ] **Step 6: Run targeted gates**

```bash
uv run ruff check trade_rl/evaluation/experiments/bootstrap/config.py tests/evaluation/experiments/bootstrap/test_config.py
uv run ruff format --check trade_rl/evaluation/experiments/bootstrap/config.py tests/evaluation/experiments/bootstrap/test_config.py
uv run mypy trade_rl/evaluation/experiments/bootstrap/config.py
uv run pytest -q tests/evaluation/experiments/bootstrap/test_config.py
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add trade_rl/evaluation/experiments/bootstrap/config.py tests/evaluation/experiments/bootstrap
git commit -m "feat: define canonical M2 bootstrap config"
```

---

### Task 4: Freeze Binance source evidence and derive source identity

**Files:**
- Create: `trade_rl/evaluation/experiments/bootstrap/binance.py`
- Create: `tests/evaluation/experiments/bootstrap/test_binance.py`

**Interfaces:**
- Produces internal `FrozenBinanceSource` with fields needed by workflow: cache transport, frozen metadata transport, Vision plan payload/digest, ordered raw source roster/digest, metadata evidence payload.
- Produces internal `_freeze_binance_source(config, staging_source_root) -> FrozenBinanceSource`.
- Produces internal `_inspect_frozen_binance_source(config, source_root) -> FrozenBinanceSource` that performs no network I/O.

- [ ] **Step 1: Write RED tests for exact plan and source roster**

With fake download bytes and a fake metadata delegate, assert:

- Vision URLs are exactly the output of `plan_binance_vision_cache()` in authored symbol/timeframe order;
- `source/vision-plan.json` contains full ordered `urls`, not only counts;
- each cache member validates through existing sidecar logic;
- source roster entry is exactly `{url, sha256, size_bytes}`;
- `raw_source_roster_digest == content_digest(list(roster))`;
- metadata evidence includes `schema_version`, `market`, `source_uri`, `raw_payload_sha256`, `retrieved_at`.

- [ ] **Step 2: Write RED tests for network cut composition**

After source freeze, replace `urllib.request.urlopen` with `pytest.fail`. Build-facing composite transport must load klines/funding/metadata only from frozen evidence. Delete one cache member and assert the next load raises `BinanceTransportError` rather than making HTTP.

- [ ] **Step 3: Write RED tests for tampered source evidence**

Modify raw archive bytes, sidecar digest, Vision plan URL order, and frozen exchange-info bytes independently. `_inspect_frozen_binance_source` must reject every case.

- [ ] **Step 4: Run RED**

```bash
uv run pytest -q tests/evaluation/experiments/bootstrap/test_binance.py
```

Expected: import failure because `bootstrap/binance.py` does not exist.

- [ ] **Step 5: Implement source composition**

Create a private composite class that exposes exactly dataset builder transport methods:

```python
class _FrozenDatasetTransport:
    def load_klines(self, **kwargs: object) -> tuple[list[list[object]], str]: ...
    def load_funding_rates(self, **kwargs: object) -> tuple[list[tuple[int, float]], str]: ...
    def load_exchange_information(self, **kwargs: object) -> tuple[dict[str, object], str]: ...
```

Delegate market data to `BinancePublicTransport(cache_root=..., allow_network=False)` and metadata to `FrozenBinanceExchangeInfoTransport(root=..., delegate=None)`.

Source acquisition order:

1. create frozen metadata transport with live delegate and load USD-M exchange-info once;
2. create Vision plan for `(base_timeframe, *feature_timeframes)`;
3. sync complete Vision cache using network-enabled transport;
4. call `require_complete_binance_vision_cache`;
5. reconstruct all source evidence using cache-only transports;
6. return only verified frozen composition.

- [ ] **Step 6: Run targeted gates**

```bash
uv run ruff check trade_rl/evaluation/experiments/bootstrap/binance.py tests/evaluation/experiments/bootstrap/test_binance.py
uv run ruff format --check trade_rl/evaluation/experiments/bootstrap/binance.py tests/evaluation/experiments/bootstrap/test_binance.py
uv run mypy trade_rl/evaluation/experiments/bootstrap/binance.py
uv run pytest -q tests/evaluation/experiments/bootstrap/test_binance.py
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add trade_rl/evaluation/experiments/bootstrap/binance.py tests/evaluation/experiments/bootstrap/test_binance.py
git commit -m "feat: freeze canonical Binance M2 source evidence"
```

---

### Task 5: Implement atomic bootstrap workflow and verified inspection

**Files:**
- Create: `trade_rl/evaluation/experiments/bootstrap/workflow.py`
- Create: `tests/evaluation/experiments/bootstrap/test_workflow.py`
- Create: `tests/evaluation/experiments/bootstrap/test_tamper.py`

**Interfaces:**
- Produces public immutable `CanonicalM2BootstrapResult`.
- Produces public `bootstrap_canonical_m2_study(config_path: str | Path, output_root: str | Path) -> CanonicalM2BootstrapResult`.
- Produces public `inspect_canonical_m2_bootstrap(root: str | Path) -> CanonicalM2BootstrapResult`.
- No public API accepts a prebuilt dataset or externally generated StudyPlan; v1 owns the full preparation chain.

- [ ] **Step 1: Write end-to-end RED with fake network transport seam**

Use a monkeypatched source-freeze function or injectable private transport factory so tests never require real Binance network. A successful bootstrap must produce exactly:

```text
bootstrap.json
bootstrap-manifest.json
source/exchange-info/exchange-info.raw.json
source/exchange-info/manifest.json
source/vision-plan.json
source/vision-cache/**
dataset/manifest.json
dataset/arrays.npz
study/plan.json
study/.mutation.lock
```

and must not contain `study/baseline`.

- [ ] **Step 2: Write identity binding RED assertions**

On success assert:

```python
result = inspect_canonical_m2_bootstrap(output)
assert result.config_digest == config.digest
assert result.dataset_id == load_market_dataset_artifact(output / "dataset").dataset_id
assert result.study_digest == inspect_study(output / "study").plan_digest
assert result.bootstrap_digest == expected_manifest_digest
```

The manifest must bind config, Vision plan, raw roster, metadata evidence, dataset id/schema/digest, StudyPlan digest, implementation digest, and runtime environment digest.

- [ ] **Step 3: Write atomicity/failure RED tests**

Inject failure independently at source sync, dataset publication, Study creation, end-provenance check, manifest write, and final validation. After every failure:

```python
assert not output.exists()
assert not list(output.parent.glob(f".{output.name}.staging-*"))
```

An already-existing output must raise `FileExistsError` without modifying it.

- [ ] **Step 4: Write provenance drift RED test**

Monkeypatch `build_candidate_run_provenance()` to return different start/end implementation digests. Bootstrap must raise and publish no final root. Repeat for runtime environment digest.

- [ ] **Step 5: Write tamper RED tests for inspection**

After creating a valid fake bootstrap, independently tamper:

- `bootstrap.json` semantic field;
- Vision plan URL order;
- raw source bytes and matching sidecar digest (self-consistent local tamper);
- exchange-info bytes and its manifest;
- dataset arrays;
- StudyPlan dataset digest;
- bootstrap manifest dataset/study/source digest fields;
- add a `study/baseline/` directory.

`inspect_canonical_m2_bootstrap()` must reject each case. For self-consistent source tamper, rejection must arise because the bootstrap manifest's frozen raw-source roster no longer matches, not merely because sidecar validation fails.

- [ ] **Step 6: Run RED**

```bash
uv run pytest -q tests/evaluation/experiments/bootstrap/test_workflow.py tests/evaluation/experiments/bootstrap/test_tamper.py
```

Expected: import failures for missing workflow APIs.

- [ ] **Step 7: Implement staged bootstrap**

Use:

```python
if output.exists():
    raise FileExistsError(...)
staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
try:
    ...
    staging.rename(output)
except BaseException:
    shutil.rmtree(staging, ignore_errors=True)
    raise
```

Workflow order must be exact:

1. parse config;
2. capture start provenance;
3. write normalized `bootstrap.json` inside staging;
4. freeze/inspect source;
5. build dataset with `build_binance_market_dataset(..., transport_mode=VISION, transport=frozen.composite_transport, metadata_evidence=frozen.metadata_evidence)`;
6. publish dataset with `publish_market_dataset_artifact()`;
7. re-load/inspect dataset and resolve baseline config;
8. call existing `create_study()`;
9. inspect Study and assert no baseline evidence;
10. capture end provenance and compare to start + StudyPlan;
11. write strict bootstrap manifest;
12. run the same internal graph validator used by public inspection against staging;
13. rename staging to final output;
14. inspect final output and return its result.

- [ ] **Step 8: Run targeted gates**

```bash
uv run ruff check trade_rl/evaluation/experiments/bootstrap tests/evaluation/experiments/bootstrap
uv run ruff format --check trade_rl/evaluation/experiments/bootstrap tests/evaluation/experiments/bootstrap
uv run mypy trade_rl/evaluation/experiments/bootstrap
uv run pytest -q tests/evaluation/experiments/bootstrap/test_workflow.py tests/evaluation/experiments/bootstrap/test_tamper.py
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add trade_rl/evaluation/experiments/bootstrap/workflow.py tests/evaluation/experiments/bootstrap
git commit -m "feat: publish atomic canonical M2 bootstrap"
```

---

### Task 6: Add narrow CLI and intentional public facade

**Files:**
- Create: `trade_rl/evaluation/experiments/bootstrap/__init__.py`
- Create: `trade_rl/evaluation/experiments/bootstrap/cli.py`
- Modify: `trade_rl/evaluation/experiments/__init__.py`
- Create: `tests/evaluation/experiments/bootstrap/test_cli.py`
- Modify/Create public API contract test as appropriate.

**Interfaces:**
- Public symbols from `trade_rl.evaluation.experiments`:
  - `CanonicalM2BootstrapConfig`
  - `CanonicalM2BootstrapResult`
  - `bootstrap_canonical_m2_study`
  - `inspect_canonical_m2_bootstrap`
- CLI arguments exactly: `--config PATH --output PATH`.

- [ ] **Step 1: Write RED public API snapshot**

Assert the four approved names are importable from `trade_rl.evaluation.experiments`, and no private source-freeze helpers are exported there.

- [ ] **Step 2: Write RED CLI tests**

Patch `bootstrap_canonical_m2_study` and assert:

```bash
python -m trade_rl.evaluation.experiments.bootstrap.cli --config config.json --output out
```

passes exact `Path` arguments, prints a stable JSON object containing bootstrap/dataset/study digests, and returns exit code 0. Missing arguments are argparse errors; an existing output or bootstrap exception exits nonzero and must not be converted into a success payload.

- [ ] **Step 3: Implement facade and CLI**

`bootstrap/__init__.py` exports only the four public names. Root experiment facade imports/re-exports those same names. CLI must contain no business logic beyond parse/call/serialize.

- [ ] **Step 4: Run targeted gates**

```bash
uv run ruff check trade_rl/evaluation/experiments tests/evaluation/experiments/bootstrap
uv run ruff format --check trade_rl/evaluation/experiments tests/evaluation/experiments/bootstrap
uv run mypy trade_rl/evaluation/experiments
uv run pytest -q tests/evaluation/experiments/bootstrap/test_cli.py tests/evaluation/experiments/bootstrap
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add trade_rl/evaluation/experiments tests/evaluation/experiments/bootstrap
git commit -m "feat: expose canonical M2 bootstrap workflow"
```

---

### Task 7: Architecture gates, durable docs, ephemeral cleanup, and full falsification

**Files:**
- Modify: `tests/architecture/test_lean_dependency_boundaries.py`
- Modify: `tests/architecture/test_lean_evaluation_layout.py`
- Modify: `tests/architecture/test_current_docs_layout.py`
- Modify: `docs/architecture/controlled-experiment-loop.md`
- Modify: `docs/architecture/package-boundaries.md`
- Modify: `docs/research/current-status.md`
- Modify: `docs/README.md`
- Delete: `docs/specs/2026-09-09-canonical-m2-study-bootstrap-design.md`
- Delete: `docs/plans/2026-09-10-canonical-m2-study-bootstrap-implementation.md`

**Interfaces:**
- Durable architecture says bootstrap is preparation-only and cannot run baseline/final tests.
- Current research status says canonical real M2 Study is still not run unless a real Study was actually executed; this implementation alone must not change that claim.
- Final docs tree contains no completed `docs/specs` or `docs/plans`.

- [ ] **Step 1: Write/extend architecture RED tests before docs cleanup**

Require permanent bootstrap package files:

```text
trade_rl/evaluation/experiments/bootstrap/__init__.py
trade_rl/evaluation/experiments/bootstrap/config.py
trade_rl/evaluation/experiments/bootstrap/binance.py
trade_rl/evaluation/experiments/bootstrap/workflow.py
trade_rl/evaluation/experiments/bootstrap/cli.py
```

Dependency test must reject imports from `trade_rl.integrations` into `trade_rl.evaluation`, reject `evaluation.runs -> evaluation.experiments`, and reject bootstrap imports of `robustness.walk_forward.sealed_test` or any sealed final authorization owner.

- [ ] **Step 2: Update durable docs from observed implementation**

Add to `controlled-experiment-loop.md` a preparation section describing strict config, exact source freeze, cache-only cut, dataset publication, StudyPlan creation, and baseline-not-run terminal state.

Add package boundary ownership for `evaluation/experiments/bootstrap` and public surface.

Update research status to state bootstrap tooling exists, but real canonical development Study remains unrun until actual immutable bootstrap + baseline/Experiments are executed.

- [ ] **Step 3: Delete completed spec and plan**

Remove the two Active ephemeral docs only after every durable requirement is reflected in architecture/research docs. Update `docs/README.md` back to "no Active spec/plan" and remove temporary routing.

- [ ] **Step 4: Run architecture/docs gates**

```bash
uv run pytest -q tests/architecture/test_current_docs_layout.py tests/architecture/test_lean_dependency_boundaries.py tests/architecture/test_lean_evaluation_layout.py tests/architecture/test_lean_binance_layout.py
```

Expected: PASS and no `docs/specs`/`docs/plans` remain.

- [ ] **Step 5: Run focused bootstrap/integration tests**

```bash
uv run pytest -q tests/evaluation/experiments/bootstrap tests/integrations/test_binance.py tests/integrations/test_binance_cache.py
```

Expected: PASS.

- [ ] **Step 6: Run complete quality gate**

```bash
uv run ruff check trade_rl tests
uv run ruff format --check trade_rl tests
uv run mypy trade_rl
uv run pytest -q tests
uv build
```

Also run the repository's package-identity verification exactly as permanent CI does.

- [ ] **Step 7: Falsification review**

Reconstruct requirements from the original spec, then explicitly attempt to find a currently-passing wrong implementation for each of these classes:

1. network secretly used after freeze;
2. a source archive can be replaced if its sidecar is also rewritten;
3. output root appears after a failed bootstrap;
4. bootstrap runs baseline despite spec;
5. StudyPlan points at a different dataset/config than bootstrap manifest;
6. provenance changes during bootstrap but success still publishes;
7. seed policy has two authorities;
8. final docs retain completed spec/plan;
9. bootstrap can reach sealed final authorization;
10. default Binance transport behavior changes for existing callers.

Add/fix regression tests for any actual gap found, then rerun targeted-to-full verification.

- [ ] **Step 8: Commit final cleanup**

```bash
git add -A
git commit -m "docs: promote canonical M2 bootstrap contract"
```

- [ ] **Step 9: Final exact-HEAD CI gate**

Push/verify the final implementation branch and require the same final HEAD to pass Ruff, Format, Mypy, full pytest, distribution build, package identity, and architecture tests before marking the PR Ready. Record actual run/job IDs and exact head SHA in the PR body.

## Acceptance Criteria traceability

- Strict pre-registration + one seed authority: Tasks 3, 5.
- Exact exchange metadata bytes: Tasks 2, 4, 5.
- Exact Vision archive bytes/evidence: Tasks 2, 4, 5.
- Cache-only network cut: Tasks 2, 4, 5.
- Canonical dataset publication/reload: Task 5.
- Immutable StudyPlan / no baseline: Task 5.
- Provenance start/end equality: Task 5.
- Atomic whole-root publication: Task 5.
- Tamper detection including self-consistent source-sidecar rewrite: Task 5.
- Narrow public API/CLI: Task 6.
- Dependency/final-test boundary: Task 7.
- Active-doc policy during work and current-only cleanup after completion: Tasks 1, 7.
- Full static/unit/integration/build/falsification verification: Task 7.
