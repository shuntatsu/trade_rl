# Local Validation Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the local end-to-end validation path accurately test the candidate timing configuration, identify exact market content, handle bar freshness correctly, isolate the legacy dashboard server, and run an exchange-free failure drill.

**Architecture:** Keep the current Control Plane, Serving Plane, and filesystem Registry boundaries. Add small pure helpers for canonical snapshot hashing and completed-bar resolution, wire them into `CsvFeatureProvider`, preserve the existing Registry implementation, and build a deterministic local GameDay around the real bundle, Registry, runtime, audit, and HTTP interfaces with injected test components instead of a real exchange or PPO model.

**Tech Stack:** Python 3.12, NumPy, FastAPI/TestClient, Stable-Baselines-compatible runtime protocols, pytest, Ruff, mypy, GitHub Actions.

## Global Constraints

- Python remains `>=3.12`.
- Add no new third-party dependency.
- Supported serving base timeframes remain exactly `15m`, `1h`, `4h`, and `1d`.
- Snapshot hashes must use SHA-256 and canonical little-endian numeric bytes.
- `TRADE_RL_ENABLE_LEGACY_METRICS_SERVER=1` is the only environment opt-in for the legacy dashboard server.
- The local GameDay must perform no network request and no exchange order submission.
- The existing filesystem `ModelRegistry` remains unchanged as the supported single-node local Registry.
- English documentation remains normative; Japanese documentation must describe the same behavior.
- Production status remains **NO-GO**.

---

### Task 1: Make P0 validate the candidate timing configuration

**Files:**
- Modify: `mars_lite/pipeline/cli.py`
- Modify: `mars_lite/pipeline/evaluator.py:90-194`
- Modify: `mars_lite/pipeline/production_pipeline.py:152-185`
- Modify: `tests/test_production_pipeline.py`
- Create: `tests/test_p0_configuration.py`

**Interfaces:**
- Consumes: existing `args.horizon`, `args.decision_every`, and `args.days` values.
- Produces: `--p0-days: int`, default `240`; `phase_p0()` report field `config`; helper `_build_p0_args(args: Any) -> Any` that copies arguments without changing candidate timing fields.

- [ ] **Step 1: Write failing Production Pipeline tests**

Add a spy test showing that candidate values are preserved and only the explicit P0 duration is changed:

```python
def test_p0_uses_candidate_timing_without_mutating_release_args(tmp_path, monkeypatch):
    seen = {}

    def phase_p0(args, output_dir):
        seen.update(
            horizon=args.horizon,
            decision_every=args.decision_every,
            days=args.days,
        )
        (output_dir / "p0_report.json").write_text(
            '{"gate":{"P0_PASSED":true}}', encoding="utf-8"
        )

    registered = []
    _stub_pipeline(monkeypatch, features=_Features(), registered=registered)
    monkeypatch.setattr(production_pipeline, "phase_p0", phase_p0)
    args = _run_args(
        tmp_path,
        horizon=12,
        decision_every=4,
        days=365,
        p0_days=90,
    )

    assert production_pipeline.run(args) == 0
    assert seen == {"horizon": 12, "decision_every": 4, "days": 90}
    assert (args.horizon, args.decision_every, args.days) == (12, 4, 365)
```

Add parameterized validation tests rejecting non-positive `horizon`, `decision_every`, or `p0_days` before P0 training.

- [ ] **Step 2: Run the tests and verify failure**

Run:

```bash
uv run pytest tests/test_production_pipeline.py::test_p0_uses_candidate_timing_without_mutating_release_args -q
```

Expected: FAIL because the pipeline still overwrites timing with `4`, `1`, and `240` and has no `p0_days` contract.

- [ ] **Step 3: Add the explicit CLI option**

Add to `build_parser()`:

```python
parser.add_argument(
    "--p0-days",
    type=int,
    default=240,
    help="P0 synthetic sample duration. Candidate horizon and decision interval are unchanged.",
)
```

Keep `--days` as the normal source duration for non-P0 pipeline data.

- [ ] **Step 4: Replace hidden argument mutation with a copied P0 argument set**

In `production_pipeline.py`, import `copy` and add:

```python
def _build_p0_args(args: Any) -> Any:
    horizon = int(args.horizon)
    decision_every = int(args.decision_every)
    p0_days = int(getattr(args, "p0_days", 240))
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if decision_every <= 0:
        raise ValueError("decision_every must be positive")
    if p0_days <= 0:
        raise ValueError("p0_days must be positive")
    p0_args = copy.copy(args)
    p0_args.days = p0_days
    return p0_args
```

Replace the tuple save/overwrite/finally block with:

```python
phase_p0(_build_p0_args(args), output_dir)
```

- [ ] **Step 5: Record the effective P0 configuration**

Initialize the P0 result payload in `phase_p0()` as:

```python
results = {
    "config": {
        "horizon": int(args.horizon),
        "decision_every": int(args.decision_every),
        "days": int(args.days),
    }
}
```

The existing positive/negative result keys and `gate` key remain unchanged.

- [ ] **Step 6: Add evaluator report tests**

In `tests/test_p0_configuration.py`, monkeypatch expensive training/evaluation dependencies and assert `p0_report.json` contains:

```python
assert report["config"] == {
    "horizon": 12,
    "decision_every": 4,
    "days": 90,
}
```

- [ ] **Step 7: Run focused tests**

Run:

```bash
uv run pytest tests/test_production_pipeline.py tests/test_p0_configuration.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add mars_lite/pipeline/cli.py mars_lite/pipeline/evaluator.py mars_lite/pipeline/production_pipeline.py tests/test_production_pipeline.py tests/test_p0_configuration.py
git commit -m "fix: align P0 with candidate timing"
```

---

### Task 2: Add canonical content-addressed snapshot identities

**Files:**
- Create: `mars_lite/serving/snapshot_identity.py`
- Modify: `mars_lite/serving/feature_provider.py`
- Create: `tests/test_snapshot_identity.py`
- Modify: `tests/test_feature_provider.py`

**Interfaces:**
- Produces:

```python
def compute_snapshot_id(
    *,
    bundle_digest: str,
    base_timeframe: str,
    timestamps: np.ndarray,
    symbols: Sequence[str],
    feature_names: Sequence[str],
    global_feature_names: Sequence[str],
    feature_history: np.ndarray,
    global_features: np.ndarray,
    close_history: np.ndarray,
) -> str:
    ...
```

- [ ] **Step 1: Write failing canonicalization tests**

Cover these invariants:

```python
def test_snapshot_id_is_stable_across_native_and_big_endian_arrays(): ...
def test_snapshot_id_changes_when_one_feature_value_changes(): ...
def test_snapshot_id_changes_when_close_history_changes(): ...
def test_snapshot_id_changes_when_ordered_schema_changes(): ...
def test_snapshot_id_rejects_non_finite_values(): ...
```

Use equivalent `float64`, `float32`, C-contiguous, non-contiguous, native-endian, and big-endian inputs and expect the same digest when numeric content and schema are equal.

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
uv run pytest tests/test_snapshot_identity.py -q
```

Expected: FAIL because `compute_snapshot_id` does not exist.

- [ ] **Step 3: Implement canonical byte framing**

Implement private helpers that hash each field with an unambiguous tag and 8-byte big-endian payload length:

```python
def _update_field(hasher, tag: str, payload: bytes) -> None:
    tag_bytes = tag.encode("utf-8")
    hasher.update(len(tag_bytes).to_bytes(8, "big"))
    hasher.update(tag_bytes)
    hasher.update(len(payload).to_bytes(8, "big"))
    hasher.update(payload)
```

Encode string sequences as canonical JSON with `ensure_ascii=False`, `separators=(",", ":")`, and `allow_nan=False`.

Normalize timestamps with:

```python
np.asarray(timestamps, dtype="datetime64[ns]").astype("<i8", copy=False)
```

Normalize numeric arrays with:

```python
np.ascontiguousarray(np.asarray(value, dtype="<f8"))
```

Hash a schema marker such as `trade-rl-feature-snapshot-v1`, dimensions, dtype marker, and raw bytes for each array. Reject empty bundle/timeframe strings and non-finite numeric arrays.

- [ ] **Step 4: Wire the helper into `CsvFeatureProvider`**

Replace the existing `bundle_digest:last_timestamp:n_bars` identity with `compute_snapshot_id(...)`, passing the exact selected timestamps and arrays used to build `FeatureSnapshot`.

- [ ] **Step 5: Add provider mutation test**

Build two otherwise identical feature sets with the same row count and timestamps but one changed feature value. Assert two uncached provider snapshots have different IDs.

- [ ] **Step 6: Run focused tests**

Run:

```bash
uv run pytest tests/test_snapshot_identity.py tests/test_feature_provider.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add mars_lite/serving/snapshot_identity.py mars_lite/serving/feature_provider.py tests/test_snapshot_identity.py tests/test_feature_provider.py
git commit -m "feat: content-address feature snapshots"
```

---

### Task 3: Select completed bars and calculate timeframe-aware staleness

**Files:**
- Create: `mars_lite/serving/market_time.py`
- Modify: `mars_lite/serving/feature_provider.py`
- Create: `tests/test_market_time.py`
- Modify: `tests/test_feature_provider.py`

**Interfaces:**
- Consumes: `mars_lite.data.data_utils.TF_TO_MINUTES`.
- Produces:

```python
@dataclass(frozen=True)
class CompletedBarEndpoint:
    end_exclusive: int
    latest_bar_close: np.datetime64
    data_age_hours: float


def resolve_completed_bar_endpoint(
    timestamps: np.ndarray,
    *,
    base_timeframe: str,
    now_utc: np.datetime64,
) -> CompletedBarEndpoint:
    ...
```

- [ ] **Step 1: Write failing pure time tests**

Add cases for `1h`, `4h`, and `1d`:

```python
@pytest.mark.parametrize(
    ("timeframe", "timestamps", "now", "expected_end", "expected_age"),
    [
        ("1h", ["2026-07-12T08:00", "2026-07-12T09:00"], "2026-07-12T09:30", 1, 0.5),
        ("4h", ["2026-07-12T00:00", "2026-07-12T04:00"], "2026-07-12T07:00", 1, 3.0),
        ("1d", ["2026-07-10T00:00", "2026-07-11T00:00"], "2026-07-12T12:00", 2, 12.0),
    ],
)
def test_completed_bar_endpoint(...): ...
```

Also test no completed bar, unsupported timeframe, unsorted timestamps, duplicate timestamps, and future `now_utc` normalization.

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
uv run pytest tests/test_market_time.py -q
```

Expected: FAIL because the helper does not exist.

- [ ] **Step 3: Implement completed-bar resolution**

Use `TF_TO_MINUTES[base_timeframe]` to construct a nanosecond duration. Convert timestamps and `now_utc` to `datetime64[ns]`. Require one-dimensional, strictly increasing timestamps. Compute closes and select:

```python
completed = np.flatnonzero(bar_closes <= now_ns)
```

Reject when `completed` is empty. Set `end_exclusive = completed[-1] + 1` and calculate age from the selected close. Clamp only sub-nanosecond floating noise; a materially negative age is an error.

- [ ] **Step 4: Inject a deterministic clock into the provider**

Change the constructor to:

```python
def __init__(
    self,
    *,
    runtime: ServingRuntime,
    data_dir: str | Path,
    cache_ttl_seconds: float = 30.0,
    clock: Callable[[], np.datetime64] | None = None,
) -> None:
```

Store `self._clock = clock or (lambda: np.datetime64("now", "ns"))`.

- [ ] **Step 5: Slice all snapshot inputs at the completed endpoint**

After building the FeatureSet:

```python
endpoint = resolve_completed_bar_endpoint(
    feature_set.timestamps,
    base_timeframe=base_timeframe,
    now_utc=self._clock(),
)
end = endpoint.end_exclusive
start = max(0, end - history_bars)
timestamps = feature_set.timestamps[start:end]
feature_history = feature_set.features[start:end]
close_history = feature_set.close[start:end]
global_features = feature_set.global_features[end - 1]
```

Set `data_age_hours=endpoint.data_age_hours`. Never include `feature_set` rows after `end` in either the snapshot arrays or hash.

- [ ] **Step 6: Add provider integration tests**

Use an injected clock and assert:

- the incomplete latest `4h` row is excluded;
- a completed `1d` row is aged from its close, not open;
- no completed row raises `ValueError("no completed bar")`;
- changing only an excluded incomplete row does not change the snapshot ID;
- changing a selected completed row does change the snapshot ID.

- [ ] **Step 7: Run focused tests**

Run:

```bash
uv run pytest tests/test_market_time.py tests/test_snapshot_identity.py tests/test_feature_provider.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add mars_lite/serving/market_time.py mars_lite/serving/feature_provider.py tests/test_market_time.py tests/test_feature_provider.py
git commit -m "fix: make serving freshness timeframe aware"
```

---

### Task 4: Fail closed on accidental legacy metrics-server startup

**Files:**
- Modify: `mars_lite/server/metrics_server.py`
- Modify: `scripts/train.py:579-594`
- Create: `tests/test_legacy_metrics_server.py`
- Modify: `tests/test_documentation_contract.py`

**Interfaces:**
- Produces:

```python
def create_app(
    metrics_history: MetricsHistory | None = None,
    output_dir: str = "./output",
    *,
    development_only: bool = False,
) -> FastAPI:
    ...
```

`run_server()` and `run_server_async()` receive the same keyword-only `development_only` argument.

- [ ] **Step 1: Write failing isolation tests**

Add:

```python
def test_legacy_metrics_server_refuses_start_without_opt_in(monkeypatch):
    monkeypatch.delenv("TRADE_RL_ENABLE_LEGACY_METRICS_SERVER", raising=False)
    with pytest.raises(RuntimeError, match="development-only"):
        create_app()


def test_legacy_metrics_server_accepts_explicit_factory_opt_in():
    app = create_app(development_only=True)
    assert app.title == "MarS Lite Training Server"


def test_legacy_metrics_server_accepts_exact_environment_opt_in(monkeypatch):
    monkeypatch.setenv("TRADE_RL_ENABLE_LEGACY_METRICS_SERVER", "1")
    assert create_app().title == "MarS Lite Training Server"
```

Also assert values such as `true`, `yes`, and `01` do not opt in.

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
uv run pytest tests/test_legacy_metrics_server.py -q
```

Expected: FAIL because `create_app()` starts without an opt-in.

- [ ] **Step 3: Implement the guard**

At module level, document that this is a legacy training dashboard, not the Serving Plane. Add:

```python
def _require_development_opt_in(development_only: bool) -> None:
    enabled = os.getenv("TRADE_RL_ENABLE_LEGACY_METRICS_SERVER") == "1"
    if not development_only and not enabled:
        raise RuntimeError(
            "legacy metrics server is development-only; set "
            "TRADE_RL_ENABLE_LEGACY_METRICS_SERVER=1 or pass development_only=True"
        )
```

Call it before constructing the FastAPI app.

- [ ] **Step 4: Propagate explicit opt-in through startup functions**

Add `development_only: bool = False` to `run_server()` and `run_server_async()` and pass it to `create_app()`.

For the module CLI, add `--development-only` and pass its value. Running `python -m mars_lite.server.metrics_server` without the flag or environment variable must fail closed.

- [ ] **Step 5: Preserve the intentional training-dashboard path**

In `scripts/train.py`, the existing explicit `--serve` option is itself a development-dashboard choice. Change the call to:

```python
run_server(
    host="0.0.0.0",
    port=args.port,
    output_dir=str(output_dir),
    development_only=True,
)
```

Do not change `scripts/run_server.py`; it must continue importing only `mars_lite.server.signal_server`.

- [ ] **Step 6: Run focused tests**

Run:

```bash
uv run pytest tests/test_legacy_metrics_server.py tests/test_documentation_contract.py tests/test_signal_server.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add mars_lite/server/metrics_server.py scripts/train.py tests/test_legacy_metrics_server.py tests/test_documentation_contract.py
git commit -m "fix: isolate legacy metrics server"
```

---

### Task 5: Add an exchange-free local GameDay harness

**Files:**
- Create: `mars_lite/serving/local_gameday.py`
- Create: `scripts/run_local_gameday.py`
- Create: `tests/test_local_gameday.py`
- Modify: `pyproject.toml` only if coverage omission must be removed for the new module; do not add dependencies.

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True)
class ScenarioResult:
    name: str
    passed: bool
    details: Mapping[str, Any]


def run_local_gameday(root: Path | None = None) -> dict[str, Any]:
    ...


def exit_code_for_summary(summary: Mapping[str, Any]) -> int:
    ...
```

The script prints one JSON object to stdout and returns `0` only when every scenario passes.

- [ ] **Step 1: Write failing summary and CLI tests**

Test that:

```python
def test_exit_code_is_nonzero_when_any_scenario_fails(): ...
def test_local_gameday_reports_all_required_scenarios(tmp_path): ...
def test_local_gameday_is_deterministic_for_fixed_inputs(tmp_path): ...
```

The expected scenario names are exactly:

```python
{
    "healthy_activation",
    "content_mutation_identity",
    "timeframe_freshness",
    "stale_data_fail_closed",
    "replay_rejection",
    "bundle_rejection_preserves_healthy_runtime",
    "rollback",
}
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
uv run pytest tests/test_local_gameday.py -q
```

Expected: FAIL because the harness does not exist.

- [ ] **Step 3: Build deterministic valid bundle fixtures inside the harness**

Use `create_candidate_bundle()` with:

- one symbol `BTCUSDT`;
- one feature `ret`;
- observation dimension matching one-symbol observation construction;
- complete eligible release metadata;
- complete finite risk policy;
- Git SHA `a` repeated 40 times for valid bundles;
- small placeholder model bytes because the GameDay injects a protocol-compatible component factory instead of loading PPO.

Create two valid versions and one Git-SHA-mismatched version. Register through the real `ModelRegistry`.

- [ ] **Step 4: Add deterministic runtime components**

Create an internal `_Policy` whose `predict()` returns a fixed one-symbol action. Construct `RuntimeComponents` with:

- a deterministic `decide()`;
- real `evaluate_guardrails()` and `apply_guardrails()` with a two-hour stale limit;
- a deterministic risk approval function;
- no exchange adapter.

Use the real `AuditStore`, `ServingRuntime`, and strict release binding.

- [ ] **Step 5: Implement the seven scenarios**

Each scenario catches its own exception and returns `ScenarioResult`.

1. `healthy_activation`: activate version 1, refresh, and compare readiness version, digest, and release SHA.
2. `content_mutation_identity`: call `compute_snapshot_id()` twice with the same timestamps/shape but one changed selected value and require different IDs.
3. `timeframe_freshness`: call `resolve_completed_bar_endpoint()` for `1h`, `4h`, and `1d`, and require the incomplete final row to be excluded.
4. `stale_data_fail_closed`: infer with a valid but stale `FeatureSnapshot`; require no non-zero target exposure and a stale-data guardrail reason.
5. `replay_rejection`: infer twice with the same request ID and payload hash; require the second response to be rejected and an audit event to exist.
6. `bundle_rejection_preserves_healthy_runtime`: activate the mismatched bundle; require `refresh()` to return false and readiness to remain `degraded` on version 1 and its digest.
7. `rollback`: activate valid version 2, refresh, call Registry rollback to version 1, refresh, and require the original identity.

- [ ] **Step 6: Exercise the HTTP boundary**

For at least the healthy and replay scenarios, create the real FastAPI app with `signal_server.create_app()`, a fixed in-memory feature provider, and `TestClient`. Send authenticated requests to `/ready` and `/api/signal/latest` rather than calling every runtime method directly.

- [ ] **Step 7: Implement machine-readable CLI behavior**

`scripts/run_local_gameday.py` must contain only argument parsing, a call to `run_local_gameday()`, `json.dumps(..., sort_keys=True)`, and `raise SystemExit(exit_code_for_summary(summary))`.

Support optional `--work-dir PATH`; when omitted use `TemporaryDirectory` and clean it automatically.

- [ ] **Step 8: Run focused tests and the command**

Run:

```bash
uv run pytest tests/test_local_gameday.py tests/test_serving_runtime.py tests/test_signal_server.py tests/test_serving_registry.py -q
uv run python scripts/run_local_gameday.py
```

Expected: tests PASS; command prints seven passing scenarios and exits `0`.

- [ ] **Step 9: Commit**

```bash
git add mars_lite/serving/local_gameday.py scripts/run_local_gameday.py tests/test_local_gameday.py pyproject.toml
git commit -m "feat: add offline local gameday"
```

---

### Task 6: Synchronize English and Japanese operating documentation

**Files:**
- Modify: `README.md`
- Modify: `README.ja.md`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/OPERATIONS.md`
- Modify: `docs/PRODUCTION_READINESS.md`
- Modify: `docs/ja/ARCHITECTURE.md`
- Modify: `docs/ja/OPERATIONS.md`
- Modify: `docs/ja/PRODUCTION_READINESS.md`
- Modify: `tests/test_documentation_contract.py`

**Interfaces:**
- Documents the commands and guarantees introduced by Tasks 1-5.

- [ ] **Step 1: Add failing documentation-contract assertions**

Require both language sets to mention:

- `--p0-days` and candidate-aligned P0 timing;
- content-addressed snapshot IDs;
- completed-bar freshness;
- `TRADE_RL_ENABLE_LEGACY_METRICS_SERVER=1`;
- `uv run python scripts/run_local_gameday.py`;
- single-node Registry scope;
- Production **NO-GO**.

- [ ] **Step 2: Run the documentation test and verify failure**

Run:

```bash
uv run pytest tests/test_documentation_contract.py -q
```

Expected: FAIL because the new contracts are not documented.

- [ ] **Step 3: Update architecture and operations documents**

Document that:

- P0 uses release candidate `horizon` and `decision_every`, while `--p0-days` controls only synthetic sample duration;
- snapshot IDs hash the exact selected inference content;
- a row is usable only after its timeframe close;
- the legacy metrics server is a development dashboard, not the Serving Plane;
- local GameDay is exchange-free and does not prove testnet or multi-node readiness;
- the filesystem Registry is supported for one local administrative filesystem domain only.

Add the local command:

```bash
uv run python scripts/run_local_gameday.py
```

- [ ] **Step 4: Keep readiness status honest**

Add repository-verifiable checklist items for the local GameDay, but do not check operational/testnet GameDay, real exchange execution, multi-node Registry, or GO approval. Keep the first line decision **NO-GO** in both languages.

- [ ] **Step 5: Run documentation tests**

Run:

```bash
uv run pytest tests/test_documentation_contract.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add README.md README.ja.md docs/ARCHITECTURE.md docs/OPERATIONS.md docs/PRODUCTION_READINESS.md docs/ja/ARCHITECTURE.md docs/ja/OPERATIONS.md docs/ja/PRODUCTION_READINESS.md tests/test_documentation_contract.py
git commit -m "docs: document trustworthy local validation"
```

---

### Task 7: Complete review, full verification, and publication

**Files:**
- Review all files changed from `main` to `agent/local-validation-hardening`.
- Modify only files required by actual review or verification failures.

**Interfaces:**
- Produces a Draft PR with exact verification evidence.

- [ ] **Step 1: Run focused verification**

```bash
uv run pytest \
  tests/test_production_pipeline.py \
  tests/test_p0_configuration.py \
  tests/test_snapshot_identity.py \
  tests/test_market_time.py \
  tests/test_feature_provider.py \
  tests/test_legacy_metrics_server.py \
  tests/test_local_gameday.py \
  tests/test_serving_runtime.py \
  tests/test_signal_server.py \
  tests/test_serving_registry.py \
  tests/test_documentation_contract.py -q
```

Expected: PASS.

- [ ] **Step 2: Run the local GameDay as an executable acceptance test**

```bash
uv run python scripts/run_local_gameday.py
```

Expected: JSON with `passed: true`, seven passing scenarios, and exit code `0`.

- [ ] **Step 3: Run repository static checks**

```bash
uv run ruff check .
uv run ruff format --check --diff .
uv run mypy mars_lite
```

Expected: all PASS.

- [ ] **Step 4: Run the complete test and coverage suite**

```bash
uv run pytest --cov=mars_lite --cov-report=term-missing --cov-fail-under=70 tests/
```

Expected: all tests PASS and coverage is at least 70%.

- [ ] **Step 5: Review the diff against the design**

```bash
git diff --check main...HEAD
git diff --stat main...HEAD
git diff main...HEAD
```

Confirm there is no distributed Registry, real exchange adapter, hidden candidate timing override, incomplete-bar inference, accidental legacy server startup, or Production GO claim.

- [ ] **Step 6: Request code review and fix every Critical or Important finding**

Review separately for:

- spec compliance;
- canonical hashing ambiguity;
- timezone and bar-boundary correctness;
- replay/audit correctness;
- accidental Production surface exposure;
- GameDay false positives.

Repeat focused verification after every fix.

- [ ] **Step 7: Open a Draft PR**

The PR body must list exact commands run, results, local GameDay scenario names, and explicitly state that real testnet GameDay and multi-node Registry remain out of scope and Production remains **NO-GO**.
