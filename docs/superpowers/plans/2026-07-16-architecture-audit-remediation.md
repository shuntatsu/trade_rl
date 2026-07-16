# Architecture Audit Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close every architecture gap identified in the July 16 audit so the evaluated policy rule, final artifact, confirmation evidence, terminal accounting, market metadata, funding data, and release authorization share one fail-closed identity contract.

**Architecture:** Walk-forward will evaluate the same deterministic multi-seed mean ensemble rule used by final serving while retaining per-seed eligibility evidence. Confirmation and release evidence will use authenticated HMAC-SHA256 envelopes with trusted key IDs, complete immutable return/fill/order payloads, and derived metrics. Market ingestion will preserve point-in-time instrument filters and aggregate every funding event into its native bar.

**Tech Stack:** Python 3.12, NumPy, Stable-Baselines3, Gymnasium, pytest, GitHub Actions, standard-library `hmac`/`hashlib`.

## Global Constraints

- No production code without a failing test first.
- Existing public APIs remain compatible where safety permits; unsafe unsigned production activation fails closed.
- Direct exchange order routing remains out of scope and production status remains `NO-GO` until empirical gates pass.
- Every new identity is canonical and SHA-256 content-addressed.
- Full Ruff, formatting, MyPy, import-lint, pytest branch coverage, critical coverage, CLI smoke, and Docker smoke must pass.

---

### Task 1: Evaluate the deployable ensemble rule

**Files:**
- Modify: `trade_rl/workflows/fold_runner.py`
- Modify: `trade_rl/workflows/market_walk_forward.py`
- Test: `tests/workflows/test_fold_runner.py`
- Test: `tests/workflows/test_market_walk_forward.py`

**Interfaces:**
- Produces: `PolicyTrainingArtifact.ensemble_policy_digest: str`
- Produces: evaluator registry records containing ordered member policy digests and paths
- Consumes: deterministic mean action across all fixed seed finalists

- [ ] Write failing tests proving selection and outer-test requests use the ensemble digest, while per-seed scores still drive eligibility.
- [ ] Run the focused tests and verify they fail because the current representative single seed is selected.
- [ ] Add an immutable ensemble identity from ordered `(seed, policy_digest)` members and register a deterministic mean-prediction model.
- [ ] Replace median-single-seed selection with evaluation of the registered ensemble; preserve seed distribution diagnostics.
- [ ] Record selected member seeds/digests in fold evidence and require folds to agree on the same ensemble recipe.
- [ ] Run focused tests and commit.

### Task 2: Unify terminal accounting

**Files:**
- Modify: `trade_rl/workflows/market_walk_forward.py`
- Modify: `trade_rl/workflows/training_run.py`
- Test: `tests/workflows/test_market_walk_forward.py`
- Test: `tests/workflows/test_training_run.py`

**Interfaces:**
- Produces: one `terminal_accounting_mode` for training, BC, checkpoint validation, selection, and outer test.

- [ ] Write failing tests asserting maintained finite-horizon training uses `liquidate_on_end=True` when evaluation does.
- [ ] Verify RED.
- [ ] Resolve the maintained environment once and propagate identical terminal accounting through all factories and identity payloads.
- [ ] Verify GREEN and commit.

### Task 3: Authenticated evidence envelope

**Files:**
- Create: `trade_rl/release/signing.py`
- Create: `trade_rl/evaluation/confirmation.py`
- Modify: `trade_rl/release/attestation.py`
- Modify: `trade_rl/serving/bundle.py`
- Modify: `trade_rl/serving/registry.py`
- Modify: `trade_rl/serving/runtime.py`
- Modify: `examples/binance-multitimeframe/run_full_research.py`
- Modify: `examples/binance-multitimeframe/recheck_confirmation.py`
- Test: `tests/release/test_signing.py`
- Test: `tests/release/test_attestation.py`
- Test: `tests/evaluation/test_confirmation.py`
- Test: `tests/serving/test_registry.py`

**Interfaces:**
- Produces: `sign_payload(payload, key_id, key) -> AuthenticatedEnvelope`
- Produces: `verify_payload(envelope, trusted_keys) -> dict[str, object]`
- Produces: `FreshConfirmationEvidence.from_returns(...)` with metrics recomputed from immutable returns

- [ ] Write failing tamper, unknown-key, unsigned-release, and forged-summary tests.
- [ ] Verify RED.
- [ ] Implement canonical HMAC-SHA256 envelopes with key IDs and constant-time verification.
- [ ] Make confirmation evidence contain dataset/environment/policy/bundle/source/dependency identities, start/end, returns, orders digest, fills digest, reconciliation digest, and authenticated envelope; derive days, return, and drawdown from returns.
- [ ] Require trusted signed release attestations for registry/runtime activation unless an explicit test-only unreleased mode is enabled.
- [ ] Verify GREEN and commit.

### Task 4: Paired excess-return research gate

**Files:**
- Modify: `trade_rl/evaluation/research_gate.py`
- Modify: `examples/binance-multitimeframe/run_full_research.py`
- Modify: `examples/binance-multitimeframe/walk-forward-full.json`
- Test: `tests/evaluation/test_research_gate.py`
- Test: `tests/examples/test_binance_multitimeframe_full_assets.py`

**Interfaces:**
- Produces: `paired_block_bootstrap_excess_lower_bound(selected, baseline, ...) -> float`

- [ ] Write failing tests where selected return is positive but statistically indistinguishable from baseline.
- [ ] Verify RED.
- [ ] Bootstrap paired daily log excess returns and require a strictly positive lower bound plus configured material uplift.
- [ ] Enable non-zero seed success, worst-seed uplift, dispersion, turnover, cost, and drawdown limits in the maintained full config.
- [ ] Verify GREEN and commit.

### Task 5: Point-in-time Binance metadata and complete funding aggregation

**Files:**
- Modify: `trade_rl/data/contracts.py`
- Modify: `trade_rl/integrations/binance.py`
- Modify: `trade_rl/data/builder.py`
- Modify: `trade_rl/data/market.py`
- Modify: `examples/binance-multitimeframe/run_full_research.py`
- Test: `tests/integrations/test_binance.py`
- Test: `tests/data/test_builder.py`

**Interfaces:**
- Produces: effective-dated filter arrays for tick size, lot size, and minimum notional.
- Produces: native-bar funding aggregate `(sum_rate, event_count, available)`.

- [ ] Write failing tests for a historical filter change and three funding events inside one daily bar.
- [ ] Verify RED.
- [ ] Add effective-dated instrument rules and materialize per-bar arrays into dataset identity.
- [ ] Aggregate all funding events in `[bar_open, bar_close)` rather than exact timestamp matching.
- [ ] For maintained strict runs, reject fallback metadata and require a checked-in authenticated metadata history snapshot.
- [ ] Verify GREEN and commit.

### Task 6: Serving state reconciliation and documentation

**Files:**
- Create: `trade_rl/serving/state.py`
- Modify: `trade_rl/integrations/sb3_serving.py`
- Modify: `trade_rl/serving/runtime.py`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `README.md`
- Test: `tests/serving/test_state.py`
- Test: `tests/integrations/test_sb3_serving.py`

**Interfaces:**
- Produces: `ServingStateSnapshot` binding decision index, dataset ID, portfolio state digest, pending target, and observation digest.

- [ ] Write failing stale-state and mismatched-pending-target tests.
- [ ] Verify RED.
- [ ] Require an identity-bound state snapshot for structured prediction and reject stale/non-monotonic decisions.
- [ ] Remove contradictory structured-serving documentation and state the exact paper/live capability boundary.
- [ ] Verify GREEN and commit.

### Task 7: Repository-wide verification and integration

**Files:**
- Modify only as required by verification findings.

- [ ] Run Ruff, format check, MyPy, import-lint, vulture, focused tests, full branch coverage, critical coverage, CLI smoke, Windows-compatible tests, and Docker training image smoke.
- [ ] Fix only evidence-backed failures using RED-GREEN cycles.
- [ ] Create a PR to `main`, verify all checks, merge, and remove the feature branch.
