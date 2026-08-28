# Causal Alpha V10 Execution Contract Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make V10 hierarchical targets executable under the simulator's frozen no-trade-band contract and cost-aware without changing economic gates or timing hypotheses.

**Architecture:** Resolve `PreTradeRisk.no_trade_band` from each DB-backed replay environment, pass it into the V10 hierarchy compiler, and bind it into target identity. The compiler rejects sub-band flat entries, holds soft liquidity sizing between 4-hour evaluations, and reuses the V6 cost hurdle for entry eligibility.

**Tech Stack:** Python 3.12, NumPy, pytest, Ruff, Mypy, import-linter, GitHub Actions.

**Spec:** `docs/implementation-plans/specs/2026-08-29-causal-alpha-v10-execution-contract-design.md`

## Global Constraints

- Reward remains `100 * net_log_return`.
- No Selection/Admission numerical gate changes.
- No change to 72h/4h horizons or confirmation counts.
- No one-minute data, holdout tuning, BC, or RL changes.
- V8/V9 controls stay unchanged.
- The no-trade band comes from the actual environment; it is not a new V10 hyperparameter.
- `PreTradeRisk` strict boundary semantics are preserved: deltas `< band` are suppressed; equality is executable.

---

### Task 1: Lock the execution-band and after-cost compiler contract

**Files:**
- Modify: `tests/learning/test_causal_alpha_v10_hierarchy.py`
- Modify: `tests/risk/test_pretrade.py`
- Modify: `trade_rl/learning/causal_alpha_v6.py`
- Modify: `trade_rl/learning/causal_alpha_v10_hierarchy.py`

**Interfaces:**
- Consumes: `execution_no_trade_band: float` supplied by V10 workflow wiring.
- Produces: `causal_alpha_v10_hierarchical_target_path(..., execution_no_trade_band: float, ...) -> CausalAlphaV6TargetPath`.

- [ ] **Step 1: Write failing boundary tests**

Extend the V10 test helper so it can supply `liquidity_caps`, `risk_caps`, `costs`, and `execution_no_trade_band`. Add sub-band and equality boundary tests, plus a `PreTradeRisk` characterization test proving that equality is executable.

- [ ] **Step 2: Verify RED on the PR head**

Run through GitHub Actions after committing only the regression tests. Expected V10 failures are caused by the missing `execution_no_trade_band` compiler contract and/or current sub-band entry behavior. The PreTrade characterization test should already pass.

- [ ] **Step 3: Implement minimal execution-band handling**

Add `"execution_band_hold"` to `CAUSAL_ALPHA_V6_TARGET_REASONS`. Add `execution_no_trade_band` to the V10 hierarchy function, validate it as finite and non-negative, and allow a flat entry only when `abs(candidate_target) >= execution_no_trade_band`.

A blocked sub-band observation resets the pending entry confirmation and records `execution_band_hold`.

- [ ] **Step 4: Add and verify cost-aware entry test**

Use the existing V6 `execution_cost_multiplier = 1.5`; do not introduce a V10 tuning parameter. Entry must satisfy:

```python
abs(fast_mean[index]) > (
    fast_uncertainty[index]
    + 1.5 * costs[index]
    + config.edge_margin
)
```

Update V10 objective evidence with the same cost term.

- [ ] **Step 5: Add and verify cadence test**

Inject a lower liquidity cap between 4-hour decisions and assert the held target does not resize until cadence. Keep risk-cap projection fail-closed; restrict only soft liquidity-cap resizing to the 4-hour cadence.

- [ ] **Step 6: Bind compiler identity**

Build a V10 target-compiler digest from the V10 config digest, execution band, V6 execution-cost multiplier, and a fixed schema string. Store that digest as the V6-compatible path `config_digest`. Add a test compiling identical forecasts with two bands and assert path digests differ.

- [ ] **Step 7: Verify GREEN for focused tests**

```bash
uv run pytest -q tests/learning/test_causal_alpha_v10_hierarchy.py tests/risk/test_pretrade.py
```

---

### Task 2: Wire the simulator-authoritative band into V10 Selection

**Files:**
- Modify: `trade_rl/workflows/universal_causal_alpha_v10_stage_entry.py`
- Modify: `tests/workflows/test_universal_causal_alpha_v10_stage_entry.py`

**Interfaces:**
- Produces: `_execution_no_trade_band(prepared: Any, symbol: str) -> float`.
- `_target_paths(..., execution_no_trade_band: float, ...)` forwards the value to the hierarchy compiler only; V8/V9 controls are untouched.

- [ ] **Step 1: Write failing environment-resolution test**

Create a fake environment whose `pre_trade_risk.config.no_trade_band` is 0.05 and whose `close()` records invocation. Assert the resolver returns 0.05 and closes the environment.

- [ ] **Step 2: Verify RED**

Expected failure: `_execution_no_trade_band` does not exist.

- [ ] **Step 3: Implement minimal resolver and wiring**

Use existing `_environment(prepared, symbol)`, validate the resolved value, always close the environment in `finally`, resolve one band per symbol before replay loops, and pass it into `_target_paths`/the V10 hierarchy compiler.

- [ ] **Step 4: Verify focused workflow tests**

```bash
uv run pytest -q tests/workflows/test_universal_causal_alpha_v10_stage_entry.py tests/learning/test_causal_alpha_v10_hierarchy.py
```

---

### Task 3: Full verification and falsification review

- [ ] **Step 1: Run related V10 suites**

```bash
uv run pytest -q tests/learning/test_causal_alpha_v10_fit.py tests/learning/test_causal_alpha_v10_hierarchy.py tests/workflows/test_universal_causal_alpha_v10_gates.py tests/workflows/test_universal_causal_alpha_v10_stage_entry.py
```

- [ ] **Step 2: Run repository quality gates matching CI**

```bash
uv run ruff check --diff .
uv run ruff format --check --diff .
uv run mypy .
uv run lint-imports
uv run vulture trade_rl tests --min-confidence 100
uv run pytest -q --cov=trade_rl --cov-branch --cov-report=term-missing --cov-report=json:coverage.json
uv run python .github/check_critical_coverage.py coverage.json pyproject.toml
```

GitHub Actions additionally runs frontend verification, Ubuntu/Windows compatibility, training-image build, and the full training capability audit.

- [ ] **Step 3: Falsification review**

Re-read the original report and final diff without assuming the implementation is correct. Check specifically for: `<` versus `<=` band mismatch, confirmation accidentally advancing on non-executable observations, V8/V9 control changes, liquidity resizing outside cadence, hidden hyperparameter introduction, stale replay identity reuse, and cost evidence disagreeing with the actual entry gate.

- [ ] **Step 4: Inspect final PR HEAD and CI**

Require all applicable checks on the exact final head to finish successfully before marking the PR ready. Keep the PR draft if any required check fails or remains unverified.

- [ ] **Step 5: Do not merge**

Report the draft PR, exact final HEAD, verification evidence, unverified DB-backed Selection rerun, and residual risks. Merge to `main` only with explicit user authorization.
