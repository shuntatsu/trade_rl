# Control Plane / Serving Plane Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the conflicting model lifecycle and mixed server architecture with one immutable serving-bundle registry, one authenticated read-only serving plane, stateful observation parity, correct guardrail/risk inputs, and a fully rewritten documentation set.

**Architecture:** The offline control plane owns training, evidence, bundle construction, registration, activation, rollback, and deployment. The online serving plane reads one atomic active pointer, loads and validates one immutable `ServingBundle`, caches it, and exposes only health/readiness plus an authenticated signal endpoint. Training evaluation and serving share one pure observation builder and one decision pipeline; the Trade Platform supplies authoritative account state on every request and enforces the returned pre-trade risk verdict before execution.

**Tech Stack:** Python 3.12, dataclasses, pathlib, hashlib, json, tempfile/os.replace, sqlite3, FastAPI, Stable-Baselines3 PPO, NumPy, pytest, Ruff, mypy, GitHub Actions.

## Global Constraints

- Production remains NO-GO until external owner actions in `docs/PRODUCTION_READINESS.md` have evidence.
- Serving authentication is `Authorization: Bearer <token>` with `TRADE_RL_SERVING_TOKEN` as the production secret source.
- The serving process is read-only and exposes no training, deletion, promotion, rollback, or registry mutation routes.
- The Trade Platform supplies current weights, portfolio value, day-start value, peak value, pending orders, consecutive-loss state, turnover statistics, request ID, and market snapshot ID on every signal request.
- SQLite is used only for audit events and replay/idempotency protection, never as the authoritative portfolio or market-state store.
- The Trade Platform is the final live execution and pre-trade-risk enforcement boundary; serving returns a deterministic risk verdict and actionable weights only when the request is valid.
- Registry versions are immutable. Activation changes only `active.json` through an atomic replace.
- Invalid bundle digest, schema, feature mask, symbol order, preprocessing configuration, model load, request state, or market data fails closed.
- Every task must preserve a runnable branch and add or update tests before implementation.
- Full acceptance requires `ruff check .`, `ruff format --check .`, `mypy mars_lite`, and `pytest --cov=mars_lite --cov-fail-under=70 tests/`.

---

## File map

### New files

- `mars_lite/serving/bundle.py` — immutable bundle manifest, canonical digest, schema validation, and bundle loading.
- `mars_lite/serving/registry.py` — the only registry implementation; immutable registration, atomic activation, rollback, active-pointer loading.
- `mars_lite/serving/contracts.py` — typed inference request/state, pending-order payloads, risk and response contracts.
- `mars_lite/env/observation.py` — pure observation construction shared by environment/evaluation/serving.
- `mars_lite/serving/audit_store.py` — SQLite audit log and replay/idempotency protection.
- `mars_lite/serving/runtime.py` — cached active bundle, readiness state, hot-swap, stateful inference orchestration.
- `mars_lite/server/auth.py` — bearer-token validation dependency.
- `scripts/manage_registry.py` — control-plane CLI for register, activate, rollback, inspect.
- `tests/test_serving_bundle.py`
- `tests/test_serving_registry.py`
- `tests/test_observation_parity_stateful.py`
- `tests/test_serving_audit_store.py`
- `tests/test_serving_runtime.py`
- `tests/test_serving_security.py`
- `tests/test_deployment_activation.py`

### Modified files

- `mars_lite/env/portfolio_env.py` — call `build_observation()` instead of constructing observations inline.
- `mars_lite/pipeline/phases.py` — write complete inference metadata and candidate bundle inputs.
- `mars_lite/server/signal_server.py` — become the read-only authenticated serving application around `ServingRuntime`.
- `mars_lite/trading/guardrails.py` — accept and validate real account state; preserve pure evaluation.
- `mars_lite/trading/pre_trade_risk.py` — add conversion from request pending-order contracts and deterministic result serialization.
- `scripts/run_server.py` — start only `signal_server` with registry, token, audit DB, and configured bind address.
- `.github/workflows/deploy.yml` — after evidence validation, register and atomically activate the exact approved bundle.
- `pyproject.toml` — only if a missing standard dependency must be declared; no new non-standard dependency is planned.
- Existing serving, registry, deployment, parity, guardrail, and risk tests — migrate to the new contracts.

### Removed after migration

- `mars_lite/server/model_registry.py`
- `mars_lite/serving/model_store.py`
- legacy management routes and tests in `mars_lite/server/metrics_server.py` that are not part of the serving plane
- obsolete Markdown files after their valid content is incorporated into the nine-document set

---

### Task 1: Immutable ServingBundle contract

**Files:**
- Create: `mars_lite/serving/bundle.py`
- Create: `tests/test_serving_bundle.py`

**Interfaces:**
- Produces: `ServingBundleManifest`, `ServingBundle`, `build_manifest(bundle_dir, metadata)`, `load_bundle(bundle_dir)`, `compute_bundle_digest(files)`.
- Later tasks consume: validated `ServingBundle.version`, `.bundle_digest`, `.model_path`, `.metadata`, `.preprocessing`, `.risk`.

- [ ] **Step 1: Write failing digest and tamper tests**

```python
from pathlib import Path

import pytest

from mars_lite.serving.bundle import build_manifest, load_bundle


def _candidate(tmp_path: Path) -> Path:
    root = tmp_path / "candidate"
    root.mkdir()
    (root / "model.zip").write_bytes(b"model-v1")
    (root / "metadata.json").write_text(
        '{"schema_version":1,"model_version":"v1","git_sha":"abc123",'
        '"symbols":["BTCUSDT"],"observation_schema_version":1}',
        encoding="utf-8",
    )
    (root / "preprocessing.json").write_text(
        '{"feature_names":["ret"],"feature_norm":"none",'
        '"feature_mask":[true],"post_mask_dim":1}',
        encoding="utf-8",
    )
    (root / "risk.json").write_text(
        '{"guardrails":{},"pre_trade":{}}', encoding="utf-8"
    )
    return root


def test_bundle_digest_is_deterministic_and_loadable(tmp_path: Path) -> None:
    root = _candidate(tmp_path)
    first = build_manifest(root)
    second = build_manifest(root)
    assert first.bundle_digest == second.bundle_digest
    loaded = load_bundle(root)
    assert loaded.version == "v1"
    assert loaded.bundle_digest == first.bundle_digest


def test_tampered_file_is_rejected(tmp_path: Path) -> None:
    root = _candidate(tmp_path)
    build_manifest(root)
    (root / "model.zip").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="digest mismatch"):
        load_bundle(root)
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `uv run pytest tests/test_serving_bundle.py -v`

Expected: collection error because `mars_lite.serving.bundle` does not exist.

- [ ] **Step 3: Implement the minimal bundle contract**

```python
@dataclass(frozen=True)
class ServingBundleManifest:
    schema_version: int
    model_version: str
    git_sha: str
    files: dict[str, str]
    bundle_digest: str


@dataclass(frozen=True)
class ServingBundle:
    root: Path
    manifest: ServingBundleManifest
    metadata: dict[str, Any]
    preprocessing: dict[str, Any]
    risk: dict[str, Any]

    @property
    def version(self) -> str:
        return self.manifest.model_version

    @property
    def bundle_digest(self) -> str:
        return self.manifest.bundle_digest

    @property
    def model_path(self) -> Path:
        return self.root / "model.zip"
```

Implement `_sha256_file`, canonical JSON using `json.dumps(..., sort_keys=True, separators=(",", ":"))`, path containment checks, required-file checks, finite/schema validation, `manifest.json` writing, and manifest verification. The bundle digest must hash the canonical `{path: digest}` mapping and must exclude `manifest.json` itself.

- [ ] **Step 4: Add schema and feature-mask failure tests**

```python
def test_feature_mask_dimension_mismatch_fails_closed(tmp_path: Path) -> None:
    root = _candidate(tmp_path)
    (root / "preprocessing.json").write_text(
        '{"feature_names":["a","b"],"feature_norm":"none",'
        '"feature_mask":[true,false],"post_mask_dim":2}',
        encoding="utf-8",
    )
    build_manifest(root)
    with pytest.raises(ValueError, match="post_mask_dim"):
        load_bundle(root)
```

- [ ] **Step 5: Run focused tests and commit**

Run: `uv run pytest tests/test_serving_bundle.py -v`

Expected: all tests pass.

Commit:

```bash
git add mars_lite/serving/bundle.py tests/test_serving_bundle.py
git commit -m "feat: add immutable serving bundle contract"
```

---

### Task 2: One authoritative atomic ModelRegistry

**Files:**
- Create: `mars_lite/serving/registry.py`
- Create: `scripts/manage_registry.py`
- Create: `tests/test_serving_registry.py`
- Modify: `scripts/run_pipeline.py`
- Remove after callers migrate: `mars_lite/server/model_registry.py`, `mars_lite/serving/model_store.py`

**Interfaces:**
- Consumes: `load_bundle(Path) -> ServingBundle`.
- Produces: `ModelRegistry.register(source_dir)`, `.activate(version, evidence_identity)`, `.rollback(target_version=None)`, `.get_active_bundle()`, `.list_versions()`.

- [ ] **Step 1: Write failing immutable registration and rollback tests**

```python
def test_register_activate_and_rollback_are_atomic(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    v1 = create_bundle(tmp_path / "v1", version="v1", payload=b"one")
    v2 = create_bundle(tmp_path / "v2", version="v2", payload=b"two")

    registry.register(v1)
    registry.activate("v1", evidence_identity="run-1")
    assert registry.get_active_bundle().version == "v1"

    registry.register(v2)
    registry.activate("v2", evidence_identity="run-2")
    assert registry.get_active_bundle().version == "v2"

    registry.rollback()
    assert registry.get_active_bundle().version == "v1"


def test_failed_activation_preserves_old_active(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    v1 = create_bundle(tmp_path / "v1", version="v1", payload=b"one")
    registry.register(v1)
    registry.activate("v1", evidence_identity="run-1")

    with pytest.raises(KeyError):
        registry.activate("missing", evidence_identity="run-x")
    assert registry.get_active_bundle().version == "v1"
```

- [ ] **Step 2: Run RED**

Run: `uv run pytest tests/test_serving_registry.py -v`

Expected: import failure for `mars_lite.serving.registry`.

- [ ] **Step 3: Implement registry layout and atomic active pointer**

Use:

```text
registry/
  versions/<version>/...
  active.json
  activation-history.jsonl
```

`register()` validates the source bundle, copies into a temporary sibling directory, validates the copy, then renames it into `versions/<version>`. Existing versions are rejected. `activate()` writes a temporary JSON file containing `version`, `bundle_digest`, `activated_at`, and `evidence_identity`, fsyncs it, and calls `os.replace()` onto `active.json`. History append occurs after successful replacement and records both previous and next identities.

- [ ] **Step 4: Add CLI tests**

Test `scripts/manage_registry.py register`, `activate`, `rollback`, `show-active`, and non-zero failure codes for unknown versions. The CLI imports only `mars_lite.serving.registry.ModelRegistry`.

- [ ] **Step 5: Migrate pipeline registration**

Replace `mars_lite.server.model_registry.ModelRegistry` usage in `scripts/run_pipeline.py` with the new registry. The pipeline registers a complete candidate bundle directory, not a raw model file. Registration must not activate automatically; activation remains deployment-controlled.

- [ ] **Step 6: Run tests and commit**

Run:

```bash
uv run pytest tests/test_serving_bundle.py tests/test_serving_registry.py -v
uv run mypy mars_lite/serving/bundle.py mars_lite/serving/registry.py
```

Commit:

```bash
git add mars_lite/serving/registry.py scripts/manage_registry.py scripts/run_pipeline.py tests/test_serving_registry.py
git commit -m "feat: unify model registry and atomic activation"
```

---

### Task 3: Pure stateful observation builder

**Files:**
- Create: `mars_lite/env/observation.py`
- Modify: `mars_lite/env/portfolio_env.py`
- Create: `tests/test_observation_parity_stateful.py`
- Modify: `tests/test_pipeline_parity.py`

**Interfaces:**
- Produces: `ObservationSchema`, `ObservationState`, `build_observation(...) -> np.ndarray`.
- Later tasks consume the same function before policy inference.

- [ ] **Step 1: Write a failing stateful parity test**

```python
def test_current_weights_are_present_before_policy_inference(feature_set) -> None:
    env = PortfolioTradingEnv(feature_set, episode_bars=20)
    env.reset(options={"start_idx": 10})
    env.weights = np.array([0.2, -0.1, 0.0, 0.1, 0.0, 0.0, 0.0])[: env.n_symbols]
    env.portfolio_value = 0.92
    env.peak_value = 1.0

    actual = env._obs()
    expected = build_observation(
        per_symbol_features=feature_set.features[env.t],
        global_features=feature_set.global_features[env.t],
        state=ObservationState(
            weights=env.weights,
            portfolio_value=0.92,
            peak_value=1.0,
            progress=(env.t - env.start_idx) / env.episode_bars,
        ),
        schema=env.observation_schema,
    )
    np.testing.assert_allclose(actual, expected)
```

- [ ] **Step 2: Run RED**

Run: `uv run pytest tests/test_observation_parity_stateful.py -v`

Expected: import failure for `mars_lite.env.observation`.

- [ ] **Step 3: Implement the pure builder**

```python
@dataclass(frozen=True)
class ObservationState:
    weights: np.ndarray
    portfolio_value: float
    peak_value: float
    progress: float
    vol_scale: float = 1.0
    dd_scale: float = 1.0
    disagreement_scale: float = 1.0
    est_port_vol: float = 0.0


@dataclass(frozen=True)
class ObservationSchema:
    include_risk_state: bool = False
    version: int = 1
```

`build_observation()` validates finite values, matching symbol dimensions, positive portfolio/peak values, and produces exactly the current environment layout: flattened per-symbol `[features..., current_weight]`, raw global features, then `[drawdown, gross, progress]`, plus four risk-state values when enabled.

- [ ] **Step 4: Replace inline `_obs()` construction**

`PortfolioTradingEnv._obs()` must only construct `ObservationState` and call `build_observation()`. Add `self.observation_schema = ObservationSchema(include_risk_state=self.obs_risk_state)` during initialization.

- [ ] **Step 5: Add a test proving different current positions change the policy input**

Create two observations from identical market features but different `weights`; assert the vectors differ only in weight/gross-related positions and are not equal.

- [ ] **Step 6: Run tests and commit**

Run:

```bash
uv run pytest tests/test_observation_parity_stateful.py tests/test_pipeline_parity.py tests/test_portfolio_env.py -v
```

Commit:

```bash
git add mars_lite/env/observation.py mars_lite/env/portfolio_env.py tests/test_observation_parity_stateful.py tests/test_pipeline_parity.py
git commit -m "refactor: share stateful observation construction"
```

---

### Task 4: Typed inference contracts, audit store, and replay protection

**Files:**
- Create: `mars_lite/serving/contracts.py`
- Create: `mars_lite/serving/audit_store.py`
- Create: `tests/test_serving_audit_store.py`

**Interfaces:**
- Produces: `InferenceRequest`, `InferenceState`, `PendingOrderInput`, `InferenceResponse`, `AuditStore.claim_request()`, `AuditStore.append_event()`.

- [ ] **Step 1: Write failing validation and replay tests**

```python
def test_inference_state_rejects_invalid_account_values() -> None:
    with pytest.raises(ValueError, match="day_start_value"):
        InferenceState(
            current_weights={"BTCUSDT": 0.1},
            portfolio_value=100.0,
            day_start_value=0.0,
            peak_value=110.0,
            consecutive_losses=0,
            turnover_mean=0.1,
            turnover_std=0.02,
            pending_orders=(),
        ).validate(("BTCUSDT",))


def test_duplicate_request_id_is_rejected(tmp_path: Path) -> None:
    store = AuditStore(tmp_path / "audit.sqlite3")
    assert store.claim_request("req-1", "hash-1") is True
    assert store.claim_request("req-1", "hash-1") is False
```

- [ ] **Step 2: Run RED**

Run: `uv run pytest tests/test_serving_audit_store.py -v`

- [ ] **Step 3: Implement immutable dataclasses and validation**

`InferenceRequest` contains `request_id`, `market_snapshot_id`, `state`, and optional `idempotency_key`. `InferenceState` contains the exact account fields in the design. Validation requires exact bundle-symbol coverage, finite numeric values, `portfolio_value > 0`, `day_start_value > 0`, `peak_value >= portfolio_value`, `turnover_std >= 0`, and well-formed pending orders.

- [ ] **Step 4: Implement SQLite store**

Create tables:

```sql
CREATE TABLE IF NOT EXISTS claimed_requests (
  request_id TEXT PRIMARY KEY,
  payload_hash TEXT NOT NULL,
  claimed_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_type TEXT NOT NULL,
  request_id TEXT,
  model_version TEXT,
  bundle_digest TEXT,
  payload_json TEXT NOT NULL,
  created_at REAL NOT NULL
);
```

Use transactions and `INSERT OR IGNORE` for replay protection. SQLite never stores authoritative portfolio state.

- [ ] **Step 5: Run tests and commit**

```bash
uv run pytest tests/test_serving_audit_store.py -v
uv run mypy mars_lite/serving/contracts.py mars_lite/serving/audit_store.py
git add mars_lite/serving/contracts.py mars_lite/serving/audit_store.py tests/test_serving_audit_store.py
git commit -m "feat: add inference contracts and audit replay store"
```

---

### Task 5: Cached ServingRuntime with safe hot-swap

**Files:**
- Create: `mars_lite/serving/runtime.py`
- Create: `tests/test_serving_runtime.py`
- Modify: `mars_lite/trading/pre_trade_risk.py`
- Modify: `mars_lite/trading/guardrails.py`

**Interfaces:**
- Consumes: registry, bundle, observation builder, inference contracts, audit store, decision pipeline.
- Produces: `ServingRuntime.refresh()`, `.readiness()`, `.infer(request, feature_snapshot)`.

- [ ] **Step 1: Write failing hot-swap preservation tests**

```python
def test_corrupt_new_bundle_keeps_old_runtime_active(runtime_fixture) -> None:
    runtime, registry, v1, v2 = runtime_fixture
    registry.activate("v1", evidence_identity="run-1")
    runtime.refresh()
    assert runtime.readiness().active_version == "v1"

    registry.register(v2)
    (registry.version_dir("v2") / "model.zip").write_bytes(b"corrupt")
    registry.activate("v2", evidence_identity="run-2")

    assert runtime.refresh() is False
    ready = runtime.readiness()
    assert ready.status == "degraded"
    assert ready.active_version == "v1"
```

- [ ] **Step 2: Write failing stateful inference test**

Use an injected fake model whose `predict(obs)` records the observation. Submit non-zero current weights and assert the recorded vector contains those weights before prediction. Also assert turnover passed to guardrails is `abs(target-current).sum()`.

- [ ] **Step 3: Run RED**

Run: `uv run pytest tests/test_serving_runtime.py -v`

- [ ] **Step 4: Implement runtime cache and loader injection**

`ServingRuntime` constructor accepts:

```python
def __init__(
    self,
    registry: ModelRegistry,
    audit_store: AuditStore,
    model_loader: Callable[[Path], PolicyLike] = load_ppo,
) -> None:
```

Maintain one immutable in-memory snapshot containing bundle, loaded model, post-processor, preprocessing config, risk config, and load timestamp. `refresh()` loads a candidate beside the current snapshot, runs bundle validation and a deterministic readiness observation/predict check, then swaps under a lock. Failure leaves the prior snapshot intact and records degraded status.

- [ ] **Step 5: Implement preprocessing restoration**

The runtime applies the exact bundle `feature_norm`, feature order, feature mask, and expected dimensions. Any mismatch raises a fail-closed validation error. Do not silently skip a mask.

- [ ] **Step 6: Implement stateful inference flow**

Order:

```text
claim request -> validate state -> preprocess snapshot -> build_observation
-> model.predict -> DecisionPipeline -> evaluate_guardrails with real state
-> PreTradeRiskVerifier with current weights and pending orders
-> response + audit event
```

A rejected guardrail/risk result returns `status="rejected"` and no actionable target weights. The response always includes version, digest, request ID, snapshot ID, reasons, and risk result.

- [ ] **Step 7: Run focused tests and commit**

```bash
uv run pytest tests/test_serving_runtime.py tests/test_guardrails.py tests/test_pre_trade_risk.py -v
uv run mypy mars_lite/serving/runtime.py mars_lite/trading/guardrails.py mars_lite/trading/pre_trade_risk.py
git add mars_lite/serving/runtime.py mars_lite/trading/guardrails.py mars_lite/trading/pre_trade_risk.py tests/test_serving_runtime.py
git commit -m "feat: add cached stateful serving runtime"
```

---

### Task 6: Read-only authenticated Serving Plane

**Files:**
- Create: `mars_lite/server/auth.py`
- Rewrite: `mars_lite/server/signal_server.py`
- Modify: `scripts/run_server.py`
- Create: `tests/test_serving_security.py`
- Rewrite/migrate: `tests/test_signal_server.py`

**Interfaces:**
- Produces public `GET /health`, public `GET /ready`, authenticated `POST /api/signal/latest`.
- No management routes are permitted.

- [ ] **Step 1: Write failing route-boundary tests**

```python
def test_serving_app_exposes_only_read_only_routes(client) -> None:
    paths = {route.path for route in client.app.routes}
    assert "/health" in paths
    assert "/ready" in paths
    assert "/api/signal/latest" in paths
    assert not any("delete" in path or "training" in path or "promote" in path for path in paths)


def test_signal_requires_bearer_token(client) -> None:
    assert client.post("/api/signal/latest", json=valid_request()).status_code == 401
    assert client.post(
        "/api/signal/latest",
        headers={"Authorization": "Bearer test-token"},
        json=valid_request(),
    ).status_code == 200
```

- [ ] **Step 2: Run RED**

Run: `uv run pytest tests/test_serving_security.py tests/test_signal_server.py -v`

- [ ] **Step 3: Implement bearer dependency**

`require_bearer_token(expected_token)` parses the Authorization header, uses `secrets.compare_digest`, returns 401 for missing/malformed credentials and 403 for an incorrect token. `/health` and `/ready` remain unauthenticated.

- [ ] **Step 4: Rewrite the app around ServingRuntime**

`create_app(runtime, feature_provider, auth_token, allowed_origins)` must not import training manager, expose model file paths, delete models, start training, or mutate the registry. CORS uses only configured origins; no wildcard when credentials are enabled.

- [ ] **Step 5: Replace the startup script**

`scripts/run_server.py` reads:

- `TRADE_RL_REGISTRY_DIR`
- `TRADE_RL_SERVING_TOKEN`
- `TRADE_RL_AUDIT_DB`
- `TRADE_RL_ALLOWED_ORIGINS`
- `TRADE_RL_HOST` default `127.0.0.1`
- `TRADE_RL_PORT` default `8001`

It starts `mars_lite.server.signal_server` only. Missing token is a startup error.

- [ ] **Step 6: Run tests and commit**

```bash
uv run pytest tests/test_serving_security.py tests/test_signal_server.py tests/test_serving_runtime.py -v
uv run mypy mars_lite/server/auth.py mars_lite/server/signal_server.py
git add mars_lite/server/auth.py mars_lite/server/signal_server.py scripts/run_server.py tests/test_serving_security.py tests/test_signal_server.py
git commit -m "feat: split authenticated read-only serving plane"
```

---

### Task 7: Candidate bundle production and deployment activation

**Files:**
- Modify: `mars_lite/pipeline/phases.py`
- Modify: `.github/workflows/deploy.yml`
- Create: `tests/test_deployment_activation.py`
- Modify: deployment-gate tests as required

**Interfaces:**
- Produces a complete candidate bundle directory from training.
- Deployment consumes the exact candidate and calls `scripts/manage_registry.py register` then `activate`.

- [ ] **Step 1: Write a failing candidate-content test**

Train or fake-save a small model and assert the candidate directory contains `model.zip`, `metadata.json`, `preprocessing.json`, `risk.json`, and `manifest.json`. Assert `feature_norm`, ordered symbols, feature names, feature mask, observation schema, run config, post-processing, risk config, Git SHA, and model version are present.

- [ ] **Step 2: Run RED**

Run: `uv run pytest tests/test_deployment_activation.py::test_training_writes_complete_candidate_bundle -v`

- [ ] **Step 3: Update training output**

Replace the old `save_bundle` metadata pair with complete candidate-bundle construction. Existing single-model behavior remains supported; ensemble candidates use an `ensemble/` directory declared by the manifest.

- [ ] **Step 4: Add deployment-to-served-identity integration test**

The test creates a candidate, evidence files bound to its digest, runs gate validation, registers/activates the candidate, refreshes runtime, and asserts the served version and digest equal the approved candidate.

- [ ] **Step 5: Modify deploy workflow**

After gate success, workflow steps must:

```bash
uv run python scripts/manage_registry.py --registry-dir "$REGISTRY_DIR" register deployment_bundle/candidate
uv run python scripts/manage_registry.py --registry-dir "$REGISTRY_DIR" activate "$MODEL_VERSION" --evidence-identity "run:${EVIDENCE_RUN_ID}"
```

The workflow requires a configured registry destination and must not activate if registration, digest verification, environment approval, or compatibility validation fails. Shadow performs validation without activation; Canary and Production use separate environment protections.

- [ ] **Step 6: Run tests and commit**

```bash
uv run pytest tests/test_deployment_gate.py tests/test_deployment_gate_adversarial.py tests/test_deployment_activation.py -v
python -c "import yaml; yaml.safe_load(open('.github/workflows/deploy.yml', encoding='utf-8'))"
git add mars_lite/pipeline/phases.py .github/workflows/deploy.yml tests/test_deployment_activation.py tests/test_deployment_gate.py tests/test_deployment_gate_adversarial.py
git commit -m "feat: connect approved bundles to atomic activation"
```

---

### Task 8: Remove competing lifecycle and mixed-server implementations

**Files:**
- Remove: `mars_lite/server/model_registry.py`
- Remove: `mars_lite/serving/model_store.py`
- Modify all imports found by repository search
- Remove or reduce: `mars_lite/server/metrics_server.py` serving/model-management responsibilities
- Modify tests that import removed modules

**Interfaces:**
- One registry remains: `mars_lite.serving.registry.ModelRegistry`.
- One serving app remains: `mars_lite.server.signal_server.create_app`.

- [ ] **Step 1: Add an architecture import test**

```python
def test_only_one_registry_and_one_serving_entrypoint_exist() -> None:
    assert importlib.util.find_spec("mars_lite.serving.registry") is not None
    assert importlib.util.find_spec("mars_lite.server.model_registry") is None
    assert importlib.util.find_spec("mars_lite.serving.model_store") is None
```

- [ ] **Step 2: Search callers before deletion**

Run:

```bash
rg "server\.model_registry|serving\.model_store|metrics_server.*signal|portfolio_model\.zip" mars_lite scripts tests
```

Every result must be migrated or deliberately removed.

- [ ] **Step 3: Delete obsolete implementations and migrate tests**

Do not leave compatibility shims that reintroduce two authorities. If legacy training-dashboard functionality is retained, it must be a separate control-plane process and must not expose `/api/signal/latest`.

- [ ] **Step 4: Run broad server/registry tests and commit**

```bash
uv run pytest tests/test_serving_registry.py tests/test_serving_runtime.py tests/test_signal_server.py tests/test_serving_security.py tests/test_pipeline.py -v
git add -A
git commit -m "refactor: remove competing registry and mixed serving paths"
```

---

### Task 9: Rewrite every project Markdown file into the normative set

**Files:**
- Rewrite: `README.md`
- Create/Rewrite: `docs/ARCHITECTURE.md`
- Create: `docs/OPERATIONS.md`
- Create: `docs/SECURITY.md`
- Create: `docs/MODEL_LIFECYCLE.md`
- Create: `docs/TESTING.md`
- Create: `docs/PRODUCTION_READINESS.md`
- Create: `docs/DECISIONS.md`
- Create: `docs/RESEARCH_HISTORY.md`
- Preserve: approved design and plan under `docs/superpowers/`
- Delete: every other project-owned `.md` after valid content is incorporated
- Review separately: `frontend/README.md`; replace with a short pointer to the root README or delete if the frontend is removed

**Interfaces:**
- Produces one current architecture authority: `docs/ARCHITECTURE.md`.
- `README.md` points to it and never claims Production readiness while checklist items remain open.

- [ ] **Step 1: Inventory Markdown files**

Run: `find . -name '*.md' -not -path './.git/*' -not -path './.venv/*' | sort`

Classify each as project-owned, generated/third-party, or required by tooling. Only project-owned files are consolidated.

- [ ] **Step 2: Write the nine-document set**

Each document must describe executable current behavior only. `RESEARCH_HISTORY.md` may retain dated experiment results only with dataset, configuration, scope, and limitations. Unverified profitability claims, fixed legal retention claims, placeholder contacts, and obsolete server/registry descriptions are removed.

- [ ] **Step 3: Add documentation contract tests**

Create `tests/test_documentation_contract.py` that asserts:

- all nine normative documents exist;
- root README links to `docs/ARCHITECTURE.md`;
- no project-owned Markdown outside the approved set and `docs/superpowers/` remains;
- forbidden stale paths (`mars_lite.server.model_registry`, `mars_lite.serving.model_store`, default `metrics_server` production entrypoint) do not appear in normative docs;
- Production status contains `NO-GO` until external checklist items are checked.

- [ ] **Step 4: Delete obsolete Markdown files**

Delete only after content migration and contract tests are written. Preserve license files and third-party/tool-required Markdown.

- [ ] **Step 5: Run tests and commit**

```bash
uv run pytest tests/test_documentation_contract.py -v
git add -A
git commit -m "docs: replace project documentation with one normative set"
```

---

### Task 10: Full verification, architecture review, and PR update

**Files:**
- Modify as required by verification failures only
- Update: PR #7 description

**Interfaces:**
- Produces a merge-ready code state while retaining Draft/NO-GO status for external Production actions.

- [ ] **Step 1: Run formatting and static checks**

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy mars_lite
```

Expected: all commands exit 0.

- [ ] **Step 2: Run focused architecture suites**

```bash
uv run pytest \
  tests/test_serving_bundle.py \
  tests/test_serving_registry.py \
  tests/test_observation_parity_stateful.py \
  tests/test_serving_audit_store.py \
  tests/test_serving_runtime.py \
  tests/test_serving_security.py \
  tests/test_deployment_activation.py \
  tests/test_documentation_contract.py -v
```

Expected: zero failures.

- [ ] **Step 3: Run the complete suite and coverage gate**

```bash
uv run pytest --cov=mars_lite --cov-fail-under=70 tests/
```

Expected: zero failures and coverage at least 70%.

- [ ] **Step 4: Perform architecture acceptance search**

```bash
rg "server\.model_registry|serving\.model_store|from mars_lite\.server\.metrics_server import run_server|allow_origins=\[\"\*\"\]" mars_lite scripts tests docs README.md
```

Expected: no production-path matches. Test fixtures or historical research references must not describe them as current behavior.

- [ ] **Step 5: Verify PR identity and CI**

Confirm the exact PR head SHA has a successful standard CI run. Update PR #7 body with:

- authoritative architecture summary;
- exact verification commands and results;
- served/active/deployed identity integration result;
- documentation consolidation result;
- remaining external Production blockers.

- [ ] **Step 6: Request final code review**

Review `main...HEAD` for Critical, Important, and Minor findings. Fix every Critical and Important issue before considering the branch implementation-complete. Keep the PR Draft and Production NO-GO until owner-operated items have evidence.

---

## Plan self-review

- Spec coverage: control/serving separation, one registry, immutable bundles, atomic activation, stateful observation, preprocessing parity, guardrail/risk state, authentication, replay protection, caching/hot-swap, deployment integration, tests, and full Markdown consolidation are each assigned to a task.
- Placeholder scan: the plan contains no `TBD`, `TODO`, “implement later”, or unspecified error-handling steps.
- Type consistency: `ServingBundle`, `ModelRegistry`, `ObservationState`, `ObservationSchema`, `InferenceRequest`, `InferenceState`, `AuditStore`, and `ServingRuntime` are introduced before later tasks consume them.
- Scope: tasks are ordered so each checkpoint produces independently testable behavior and the old production path is removed only after the replacement is covered.
