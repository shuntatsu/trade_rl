# Causal Alpha V3 Selection Diagnostics Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add provenance-bound per-replay diagnostics and rebuildable live selection progress to the maintained atomic-record Causal Alpha V3 runner without changing economic-selection behavior.

**Architecture:** Keep `CausalAlphaV3ReplayMetric` unchanged as the authoritative resume/economic record. Add a separate immutable diagnostics leaf bound to the replay metric and a derived `selection/progress.json` rebuilt from authoritative metrics plus optional diagnostics. Wire these into selection replay after each atomic metric write.

**Tech Stack:** Python 3.12, dataclasses, NumPy, existing `content_digest`, `canonical_json_bytes`, atomic artifact store, pytest, Ruff, Mypy, Import Linter, GitHub Actions.

## Global Constraints

- Canonical U6 reward/action/teacher/risk/execution contracts remain unchanged.
- V3 signal/economic/admission thresholds remain unchanged.
- Atomic replay records remain the only resume/economic source of truth.
- Diagnostics remain `research_only=true` and `promotion_eligible=false`.
- No holdout, validation/test symbol, or sealed-evaluation information enters selection/model tuning.
- No DAgger/BC/PPO work is started.

---

### Task 1: Replay diagnostics contract and target summarizer

**Files:**
- Create: `trade_rl/workflows/universal_causal_alpha_v3_diagnostics.py`
- Test: `tests/workflows/test_universal_causal_alpha_v3_selection_diagnostics.py`

**Interfaces:**
- Consumes: `CausalAlphaV3ReplayMetric`, `CausalAlphaV3ContractTargets`, target-path arrays/reasons.
- Produces: `CausalAlphaV3ReplayDiagnostics`, `summarize_causal_alpha_v3_targets(...)`.

- [ ] **Step 1: Write failing diagnostics tests**

```python
def test_target_diagnostics_count_direction_uncertainty_and_objective_margin():
    diagnostics = summarize_causal_alpha_v3_targets(
        run_manifest_digest=DIGEST_A,
        freeze_digest=DIGEST_B,
        candidate_digest=DIGEST_C,
        symbol="BTCUSDT",
        episode_index=3,
        contract_digest=DIGEST_D,
        replay_metric_digest=DIGEST_E,
        fit_digest=DIGEST_F,
        forecast_digest=DIGEST_G,
        target_path_digest=DIGEST_H,
        targets=np.asarray([-0.2, 0.0, 0.1, 0.0]),
        expected_returns=np.asarray([-0.03, -0.01, 0.02, 0.0]),
        uncertainties=np.asarray([0.02, 0.01, 0.04, 0.0]),
        liquidity_weight_caps=np.asarray([0.25, 0.25, 0.2, 0.2]),
        chosen_objectives=np.asarray([0.01, 0.0, 0.005, 0.0]),
        stay_objectives=np.asarray([0.0, 0.0, 0.001, 0.0]),
        reasons=("rebalance", "hold", "rebalance", "hold"),
    )
    assert diagnostics.long_target_count == 1
    assert diagnostics.short_target_count == 1
    assert diagnostics.flat_target_count == 2
    assert diagnostics.positive_forecast_count == 1
    assert diagnostics.negative_forecast_count == 2
    assert diagnostics.near_zero_forecast_count == 1
    assert diagnostics.mean_objective_improvement == pytest.approx(0.0035)
    assert diagnostics.target_reason_counts == (("hold", 2), ("rebalance", 2))
```

Also assert strict field/schema/digest tampering rejects on `from_payload`.

- [ ] **Step 2: Run targeted test and confirm RED**

Run: `pytest tests/workflows/test_universal_causal_alpha_v3_selection_diagnostics.py -q`
Expected: FAIL because diagnostics module/types do not exist.

- [ ] **Step 3: Implement immutable diagnostics contract**

Implement:

```python
@dataclass(frozen=True, slots=True)
class CausalAlphaV3ReplayDiagnostics:
    run_manifest_digest: str
    freeze_digest: str
    candidate_digest: str
    symbol: str
    episode_index: int
    contract_digest: str
    replay_metric_digest: str
    fit_digest: str
    forecast_digest: str
    target_path_digest: str
    decision_count: int
    long_target_count: int
    short_target_count: int
    flat_target_count: int
    positive_forecast_count: int
    negative_forecast_count: int
    near_zero_forecast_count: int
    mean_target: float
    mean_absolute_target: float
    maximum_absolute_target: float
    mean_expected_return: float
    mean_uncertainty: float
    p90_uncertainty: float
    mean_absolute_signal_to_uncertainty: float
    mean_liquidity_weight_cap: float
    mean_objective_improvement: float
    target_reason_counts: tuple[tuple[str, int], ...]
    research_only: bool = True
    promotion_eligible: bool = False
    schema_version: str = "causal_alpha_v3_replay_diagnostics_v1"
    digest: str = ""
```

Use exact-field validation in `from_payload`. `summarize_causal_alpha_v3_targets()` must reject shape mismatch, non-finite arrays, negative uncertainty/caps, empty reasons, or identity mismatch inputs. Near-zero forecast threshold is a fixed numerical epsilon (`1e-12`), not a tunable selection threshold.

- [ ] **Step 4: Run targeted tests and confirm GREEN**

Run: `pytest tests/workflows/test_universal_causal_alpha_v3_selection_diagnostics.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

Commit message: `feat: add v3 replay diagnostics contract`.

---

### Task 2: Artifact-store diagnostics persistence

**Files:**
- Modify: `trade_rl/workflows/universal_causal_alpha_v3_artifact_store.py`
- Test: `tests/workflows/test_universal_causal_alpha_v3_selection_diagnostics.py`

**Interfaces:**
- Consumes: `CausalAlphaV3ReplayDiagnostics`.
- Produces: `write_replay_diagnostics(...)`, `load_replay_diagnostics(...)` under `selection/diagnostics/<candidate>/<symbol>/<episode>.json`.

- [ ] **Step 1: Write failing persistence tests**

Test exact write/reload, conflicting second write rejection, path identity drift rejection, run/freeze mismatch rejection, and missing diagnostics returning an empty mapping rather than invalidating replay metrics.

- [ ] **Step 2: Run tests and confirm RED**

Run: `pytest tests/workflows/test_universal_causal_alpha_v3_selection_diagnostics.py -q`
Expected: FAIL because store methods do not exist.

- [ ] **Step 3: Implement store methods**

Diagnostics identity is `(candidate_digest, symbol, episode_index)`. Validate the loaded diagnostic's `replay_metric_digest` against a supplied mapping of authoritative replay identity -> replay metric digest. Never infer replay validity from diagnostics.

- [ ] **Step 4: Run targeted tests**

Run: `pytest tests/workflows/test_universal_causal_alpha_v3_selection_diagnostics.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

Commit message: `feat: persist v3 replay diagnostics`.

---

### Task 3: Derived selection progress

**Files:**
- Modify: `trade_rl/workflows/universal_causal_alpha_v3_diagnostics.py`
- Modify: `trade_rl/workflows/universal_causal_alpha_v3_artifact_store.py`
- Test: `tests/workflows/test_universal_causal_alpha_v3_selection_diagnostics.py`

**Interfaces:**
- Produces: `build_causal_alpha_v3_selection_progress(...) -> dict[str, object]` and `write_selection_progress(...)`.

- [ ] **Step 1: Write failing progress tests**

Given synthetic replay metrics for two candidates and two symbols, assert:

```python
assert payload["expected_replay_count"] == 8
assert payload["completed_replay_count"] == 4
assert payload["completion_fraction"] == pytest.approx(0.5)
assert payload["diagnostics_completed_count"] == 3
assert payload["candidates"][0]["irrecoverably_rejected"] is True
assert payload["symbols"]["BTCUSDT"]["mean_net_return"] == pytest.approx(...)
assert payload["promotion_eligible"] is False
```

Candidate ordering must follow authored candidate order; symbol ordering must follow `train_symbols`.

- [ ] **Step 2: Run and confirm RED**

Run: `pytest tests/workflows/test_universal_causal_alpha_v3_selection_diagnostics.py -q`
Expected: FAIL because progress builder does not exist.

- [ ] **Step 3: Implement deterministic progress builder**

Inputs: ordered candidates, ordered train symbols, expected replay count, persisted replay metrics, persisted diagnostics, thresholds, fit-cache count/hit count. Candidate aggregates use the same descriptive formulas as selection but MUST NOT select/rank a winner. Symbol aggregates are descriptive only. Write atomically to `selection/progress.json`; it may be overwritten because it is explicitly derived monitoring state, unlike immutable evidence leaves.

- [ ] **Step 4: Run targeted tests**

Run: `pytest tests/workflows/test_universal_causal_alpha_v3_selection_diagnostics.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

Commit message: `feat: add v3 selection progress artifact`.

---

### Task 4: Wire diagnostics/progress into hardened replay

**Files:**
- Modify: `trade_rl/workflows/universal_causal_alpha_v3_teacher.py`
- Modify: `trade_rl/workflows/universal_causal_alpha_v3_replay.py`
- Test: `tests/workflows/test_universal_causal_alpha_v3_runner_engine.py`
- Test: `tests/workflows/test_universal_causal_alpha_v3_selection_diagnostics.py`

**Interfaces:**
- `CausalAlphaV3ContractTargets` exposes deterministic target arrays needed by the summarizer without persisting full forecast arrays separately.

- [ ] **Step 1: Write failing integration test**

Run one fake selection scope and assert order/effects:

1. authoritative replay metric exists;
2. diagnostics leaf references that metric digest;
3. progress reports one completed scope and one diagnostics scope;
4. selection result is identical to the old path for the same metrics.

Add a resume case with an existing replay metric and no diagnostics: selection does not replay the economic scope; progress reports diagnostics coverage < replay coverage.

- [ ] **Step 2: Run and confirm RED**

Run: `pytest tests/workflows/test_universal_causal_alpha_v3_runner_engine.py tests/workflows/test_universal_causal_alpha_v3_selection_diagnostics.py -q`
Expected: FAIL on missing wiring.

- [ ] **Step 3: Implement wiring**

After `store.write_replay_metric(metric)` succeeds, build diagnostics from the already-computed target path and persist them. Refresh progress after loading resume state and after every new replay metric/diagnostic write. If process death happens after metric write but before diagnostics/progress, restart must skip the replay and rebuild progress; diagnostics remain explicitly missing rather than causing re-execution.

- [ ] **Step 4: Run related V3 tests**

Run: `pytest tests/workflows/test_universal_causal_alpha_v3_runner_engine.py tests/workflows/test_universal_causal_alpha_v3_runner_store.py tests/workflows/test_universal_causal_alpha_v3_runner_selection.py tests/workflows/test_universal_causal_alpha_v3_selection_diagnostics.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

Commit message: `feat: instrument hardened v3 selection`.

---

### Task 5: Documentation and regression boundary

**Files:**
- Modify: `docs/UNIVERSAL_TRAINING.md`
- Modify: `docs/RESEARCH_STATUS.md`
- Test: `tests/test_causal_alpha_v3_documentation_contract.py`

**Interfaces:** Documentation must distinguish legacy JSONL live-run evidence from maintained atomic-record evidence.

- [ ] **Step 1: Add failing documentation contract**

Require maintained docs to mention:

- `selection/records/...` as authoritative;
- `selection/diagnostics/...` as non-promotable diagnostic leaves;
- `selection/progress.json` as rebuildable monitoring state;
- legacy JSONL results are diagnostic-only and cannot be resumed/promoted into maintained V3.

- [ ] **Step 2: Run and confirm RED**

Run: `pytest tests/test_causal_alpha_v3_documentation_contract.py -q`
Expected: FAIL until docs are updated.

- [ ] **Step 3: Update maintained docs**

Document the root-cause decision table: gross-negative/low-turnover -> predictor; gross-positive/net-negative -> execution; tail+uncertainty -> calibration; directional imbalance -> asymmetric-threshold experiment; horizon disagreement -> horizon/rolling experiment.

- [ ] **Step 4: Run documentation test**

Run: `pytest tests/test_causal_alpha_v3_documentation_contract.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

Commit message: `docs: document v3 selection diagnostics`.

---

### Task 6: Full verification and falsification review

**Files:** all changed files.

- [ ] **Step 1: Run targeted V3 suite**

Run all `tests/workflows/test_universal_causal_alpha_v3_*` and script/documentation V3 tests. Expected: PASS.

- [ ] **Step 2: Run static checks**

Run repository-equivalent Ruff, `ruff format --check`, Mypy, import architecture, dead-code, and `git diff --check`. Expected: PASS.

- [ ] **Step 3: Run full test/coverage suite and build checks**

Use the repository's maintained CI commands. Expected: no failed tests, coverage gate >= 80%, critical coverage pass, frontend/build/training-image compatibility pass.

- [ ] **Step 4: Falsification review**

Explicitly verify:

- corrupt diagnostics cannot alter selection;
- missing diagnostics do not re-execute an already persisted replay;
- stale progress is rebuilt from authoritative records;
- diagnostic values cannot be used by rank/admission functions;
- no canonical U6 config/reward/risk files changed;
- final branch is based on integrated main and contains no unrelated files.

- [ ] **Step 5: Open Draft PR and wait for exact-head CI**

PR body must include What, Why, Acceptance Criteria, Tests, Verification, Risks, Remaining limitations, and note that economic improvement itself is not yet established.
