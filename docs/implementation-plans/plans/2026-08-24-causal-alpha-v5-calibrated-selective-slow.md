# Causal Alpha V5 Calibrated Selective Slow Anchor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a research-only V5 lane that calibrates V4 slow forecasts on a causal train suffix, abstains when slow direction confidence is insufficient, and evaluates fixed Signal, Selection, and Admission contracts before BC/RL.

**Architecture:** V5 is additive over immutable V4 forecast and execution contracts. Learning contracts own calibration, selective forecast, and target compilation; workflow modules own chronological fitting, Signal, replay, Selection, Admission, publication, and stage orchestration. V4 never imports V5.

**Tech Stack:** Python 3.12, NumPy, existing deterministic overlap-aware weighted ridge, existing simulator/risk/execution stack, pytest, Ruff, Mypy, Import Linter, GitHub Actions.

**Spec:** `docs/implementation-plans/specs/2026-08-24-causal-alpha-v5-calibrated-selective-slow-design.md`

## Global constraints

- V4 artifacts, schemas, thresholds, reward, execution, risk, and retained outcomes are immutable.
- V5 remains `research_only=true` and `promotion_eligible=false`.
- No symbol-ID feature, symbol-specific calibrator, symbol exclusion, episode exclusion, or result-driven parameter grid.
- Base labels require `label_end_72h < calibration_start`.
- Calibration labels require `decision >= calibration_start` and `label_end_72h < train_stop`.
- Calibrator ridge strength is exactly `1.0`; selective confidence threshold is exactly `1.0`.
- Overall active coverage is at least `0.25`; scope support is `max(3, ceil(0.20 * raw_direction_support))`.
- Existing fast 4h lane and maximum `±0.05` fast deviation are unchanged.
- Selection and Admission pass before BC or RL.
- No 1-minute data, skipped tests, weakened assertions, relaxed gates, or temporary workflow.

---

### Task 1: Freeze V5 learning contracts

**Files:**
- Create: `tests/learning/test_causal_alpha_v5_calibration.py`
- Create: `trade_rl/learning/causal_alpha_v5.py`
- Create: `examples/binance/universal-causal-alpha-v5-research.json`
- Modify: `trade_rl/learning/__init__.py`

**Produces:**
- `CausalAlphaV5CalibrationConfig`
- `CausalAlphaV5CalibrationFit`
- `CausalAlphaV5SelectiveForecast`
- `V5SelectiveState`
- `build_causal_alpha_v5_selective_forecast(...)`

- [ ] Write failing tests for every frozen config value, exact schema, strict integer/boolean handling, digest validation, feature names, and readonly arrays.
- [ ] Run `uv run pytest -q tests/learning/test_causal_alpha_v5_calibration.py`; confirm import failure.
- [ ] Implement config schemas:
  - `causal_alpha_v5_calibration_config_v1`
  - `causal_alpha_v5_calibration_fit_v1`
  - `causal_alpha_v5_selective_forecast_v1`
- [ ] Validate V4 fit/config/sample-scope identities, final and forward model digests, exactly three forward residual blocks, pooled/per-symbol/per-block support, non-negative residual/direction scales, and content digest.
- [ ] Add tests for raw slow fusion, independent direction fusion, calibrated return, monotonic uncertainty, confidence equality, hurdle equality, sign disagreement, and missing descriptor availability.
- [ ] Implement selective forecast with readonly raw/calibrated return, direction, uncertainty, confidence, hurdle, state, and active mask arrays.
- [ ] Freeze exact JSON config and export stable V5 API.
- [ ] Run focused pytest, Ruff, format, and Mypy.
- [ ] Commit: `feat: define causal alpha v5 calibration contracts`.

### Task 2: Implement causal chronological calibration

**Files:**
- Create: `tests/workflows/test_universal_causal_alpha_v5_calibration.py`
- Create: `trade_rl/workflows/universal_causal_alpha_v5_calibration.py`

**Produces:**
- `CausalAlphaV5CalibrationSplit`
- `fit_causal_alpha_v5_calibration(...)`
- `calibrate_causal_alpha_v5_forecast(...)`

- [ ] Write failing tests proving chronological 80/20 splitting, 72h purge, no Signal-label influence, feature-row cutoff, and fail-closed support.
- [ ] Implement split calculation from decision indices; callers cannot supply arbitrary masks.
- [ ] Write failing tests for exact forward sequence: `B1->B2`, `B1+B2->B3`, `B1+B2+B3->B4`.
- [ ] Implement fixed ridge fitting with strength `1.0`, normalized objective, overlap weights, and `working_memory_rows=4096`.
- [ ] Persist forward model/residual/weight digests, residual RMSE, direction RMSE, calibration boundary, and support.
- [ ] Add tests rejecting symbol identity, descriptor reorder, missing descriptor availability, and insufficient block/symbol support; prove an unseen symbol with valid descriptors can be calibrated.
- [ ] Run focused pytest, Ruff, format, and Mypy.
- [ ] Commit: `feat: fit v5 slow calibration on causal suffix`.

### Task 3: Implement selective target and action reasons

**Files:**
- Create: `tests/learning/test_causal_alpha_v5_target.py`
- Modify: `trade_rl/learning/causal_alpha_v5.py`

**Produces:**
- `CausalAlphaV5TargetPath`
- `causal_alpha_v5_target_path(...)`

- [ ] Write failing tests for `hold_flat`, `hold_position`, `entry`, `add`, `reduce`, `exit`, `flip`, `unactionable_hold`, `confidence_abstain`, `direction_disagreement_hold`, `edge_below_hurdle_hold`, `cadence_hold`, and `liquidity_deleverage`.
- [ ] Write failing tests proving inactive-from-flat cannot enter, inactive exposure cannot add/flip, inactive exposure may reduce/exit, and liquidity deleveraging overrides inactivity.
- [ ] Reuse V4 slow magnitudes, objective, cost, uncertainty, cadence, max delta, and deterministic tie-breaks; filter exposure-increasing candidates by V5 active state.
- [ ] Apply unchanged V4 fast impulse after slow anchor and assert deviation never exceeds `0.05`.
- [ ] Persist active mask and reason/count evidence.
- [ ] Run focused pytest, Ruff, format, and Mypy.
- [ ] Commit: `feat: add selective v5 slow target`.

### Task 4: Implement V5 Signal evidence

**Files:**
- Create: `tests/workflows/test_universal_causal_alpha_v5_signal.py`
- Create: `trade_rl/workflows/universal_causal_alpha_v5_signal.py`

**Produces:**
- `CausalAlphaV5SignalScopeMetric`
- `CausalAlphaV5SelectiveSlowEvidence`
- `CausalAlphaV5SignalEvidence`
- `build_causal_alpha_v5_signal_scope_metric(...)`
- `evaluate_causal_alpha_v5_signal_gate(...)`

- [ ] Write failing scope tests: unconditional rank/spread use all raw rows; selective direction uses active rows; inactive rows are fully reason-accounted; zero realized direction is excluded.
- [ ] Write failing gate tests for 71/72 scopes, 7/8 episodes, missing symbol/episode, coverage below `0.25`, insufficient scope support, and unaccounted abstention.
- [ ] Implement per-scope raw/active support, coverage, unconditional metrics, selective direction, active cohorts, reason counts, and upstream identities.
- [ ] Bootstrap eight episode clusters with `10000` resamples, seed `20260823`, block size `2`.
- [ ] Require unconditional Rank IC lower CI `>=0`, spread lower CI `>=0`, unconditional direction mean `>=0`, selective direction lower CI `>=0`, and all support/coverage rules.
- [ ] Bind unchanged V4 fast-lane evidence digest into combined V5 evidence.
- [ ] Add falsification test: positive rank/spread plus biased zero point cannot pass with one active row.
- [ ] Run focused pytest, Ruff, format, and Mypy.
- [ ] Commit: `feat: gate selective v5 slow signal`.

### Task 5: Implement replay attribution

**Files:**
- Create: `tests/workflows/test_universal_causal_alpha_v5_replay.py`
- Create: `trade_rl/workflows/universal_causal_alpha_v5_replay.py`

**Produces:**
- `CausalAlphaV5ReplayMetric`
- `build_causal_alpha_v5_replay_metric(...)`

- [ ] Write failing tests using real `ActionPathEvaluation` fixtures. Simulator values remain authoritative for gross/net return, cost, turnover, drawdown, and submitted/executed changes.
- [ ] Test action-reason counts, active coverage, flat-time fraction, time-weighted absolute exposure, completed holding duration, all-flat path, unclosed position, and flip semantics.
- [ ] Reject zero episode hours, malformed reasons, target/evaluation length mismatch, and executed changes without meaningful execution.
- [ ] Implement `causal_alpha_v5_replay_metric_v1` without duplicating PnL computation.
- [ ] Run focused pytest, Ruff, format, and Mypy.
- [ ] Commit: `feat: record v5 selective replay attribution`.

### Task 6: Implement balanced-wealth Selection

**Files:**
- Create: `tests/workflows/test_universal_causal_alpha_v5_selection.py`
- Create: `trade_rl/workflows/universal_causal_alpha_v5_selection.py`

**Produces:**
- `CausalAlphaV5SymbolSelectionSummary`
- `CausalAlphaV5SelectionEvidence`
- `evaluate_causal_alpha_v5_selection(...)`

- [ ] Write hand-computable failing tests for symbol gross/net wealth, symbol-balanced net wealth, median symbol wealth, positive-net scope fraction, worst scope, CVaR10, turnover p50/p95, cost, and net/gross retention.
- [ ] Write failing gates for balanced wealth `<=1`, median `<1`, positive fraction `<0.5`, missing authored symbol, no meaningful execution, hard-risk violation, unexplained rejection, and duplicate scope identity.
- [ ] Implement immutable ordered per-symbol summaries and aggregate evidence; compute wealth as `exp(log_return)` and fail on non-finite overflow.
- [ ] Run focused pytest, Ruff, format, and Mypy.
- [ ] Commit: `feat: select v5 by balanced net wealth`.

### Task 7: Implement untouched Admission and fail-closed pipeline

**Files:**
- Create: `tests/workflows/test_universal_causal_alpha_v5_admission.py`
- Create: `tests/workflows/test_universal_causal_alpha_v5_pipeline.py`
- Create: `trade_rl/workflows/universal_causal_alpha_v5_admission.py`
- Create: `trade_rl/workflows/universal_causal_alpha_v5_artifact_store.py`
- Create: `trade_rl/workflows/universal_causal_alpha_v5_pipeline.py`

- [ ] Write failing tests proving Signal blocks Selection/Admission, Selection blocks Admission, cutoff mismatch fails, records are unique by symbol, and rejection publishes no final package.
- [ ] Reuse V4 Admission economic semantics while binding V5 Signal, Selection, and calibration identities. Add no r15-derived threshold.
- [ ] Write filesystem-state tests for calibration failure, Signal rejection, Selection rejection, Admission rejection, and successful publication; later stages must not run.
- [ ] Implement staging/atomic publication and `causal_alpha_v5_research_package_v1`; package remains research-only and non-promotable.
- [ ] Run focused pytest, Ruff, format, and Mypy.
- [ ] Commit: `feat: add fail-closed v5 research stages`.

### Task 8: Integrate stage execution and CLI

**Files:**
- Create: `tests/workflows/test_universal_causal_alpha_v5_runner.py`
- Create: `trade_rl/workflows/universal_causal_alpha_v5_stage_execution.py`
- Create: `trade_rl/workflows/universal_causal_alpha_v5_runner.py`
- Create: `scripts/run_universal_causal_alpha_v5_research.py`
- Modify: `pyproject.toml`

- [ ] Write strict exact-config tests rejecting missing/unknown/reordered fields, boolean-as-integer, and changed frozen values.
- [ ] Write stage-order tests: prepare V4 -> fit V5 calibration -> selective forecasts/targets -> V5 Signal -> replay/Selection only after pass -> untouched Admission only after pass.
- [ ] Assert BC/RL functions are never imported or invoked.
- [ ] Reuse V4 data/context/forecast/replay adapters; do not copy V4 modules.
- [ ] Implement stable CLI exit codes: `0` admitted, `2` Signal, `3` Selection, `4` Admission, `5` calibration rejection.
- [ ] Run focused pytest, Ruff, format, and Mypy.
- [ ] Commit: `feat: wire causal alpha v5 research runner`.

### Task 9: Add architecture and compatibility gates

**Files:**
- Create: `tests/architecture/test_causal_alpha_v5_boundaries.py`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `.github/training_sensitive_paths.txt`
- Modify: `.github/workflows/ci.yml` only if the current classifier requires an explicit path.

- [ ] Write failing architecture tests proving V4 does not import V5, V5 uses declared V4 contracts, calibrator features contain no symbol identity, runner imports no BC/PPO/SB3/serving, V5 paths are training-sensitive, and V4 schemas/example JSON are unchanged.
- [ ] Document V5 responsibility boundaries and research-only stage order.
- [ ] Update the existing capability classifier without changing unrelated paths.
- [ ] Run architecture pytest, `lint-imports`, and workflow-security checks.
- [ ] Commit: `test: enforce causal alpha v5 boundaries`.

### Task 10: Full verification and falsification review

- [ ] Run targeted V4/V5 regression across V4 target/Signal/replay/Selection/Admission and all new V5 tests.
- [ ] Run `ruff check .`, `ruff format --check .`, `mypy .`, `lint-imports`, Vulture, and workflow security.
- [ ] Run full pytest with branch coverage and critical-coverage ratchet.
- [ ] Run Ubuntu and Windows compatibility, full training capability, training-image build, provenance recording, non-root probe, and package/version/uv identity.
- [ ] Falsify threshold drift, hidden symbol identity, Signal leakage, post-cutoff features, single-symbol coverage, HOLD-as-execution, cost double count, fast deviation over `0.05`, V4 mutation, stage bypass, early BC/RL, skipped tests, and temporary files/workflows.
- [ ] Fix every in-scope issue; rerun affected checks and then all required gates.
- [ ] Confirm `git diff --check`, clean status, final diff, one feature branch, one Draft PR, no competing PR, and exact-head CI.
- [ ] Update PR body with exact SHA, test counts, coverage, static checks, compatibility, capability/image checks, independent review findings, limitations, and explicit non-claims for profitability and Production GO.
- [ ] Do not merge to `main` without explicit user authorization.