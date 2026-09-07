# Causal Alpha V10 Execution Contract Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make V10 hierarchical targets executable under the simulator's frozen `PreTradeRisk` rebalance contract and cost-aware without changing economic gates or timing hypotheses.

**Architecture:** Resolve `(entry_threshold, no_trade_band)` from each DB-backed replay environment, pass the pair into the V10 hierarchy compiler, and bind both values into target identity. The compiler rejects flat entries below `max(entry_threshold, no_trade_band)`, treats liquidity cap as soft entry capacity while a position is held, and reuses the V6 cost hurdle for entry eligibility. Replay revalidates the same contract before evaluation.

**Tech Stack:** Python 3.12, NumPy, pytest, Ruff, Mypy, import-linter, GitHub Actions.

**Spec:** `docs/implementation-plans/specs/2026-08-29-causal-alpha-v10-execution-contract-design.md`

## Global Constraints

- Reward remains `100 * net_log_return`.
- No Selection/Admission numerical gate changes.
- No change to 72h/4h horizons or confirmation counts.
- No one-minute data, holdout tuning, BC, or RL changes.
- V8/V9 controls stay unchanged.
- Execution thresholds come from the actual environment; they are not new V10 hyperparameters.
- `PreTradeRisk` strict boundary semantics are preserved: `abs(target) < entry_threshold` blocks flat entry and deltas `< no_trade_band` are suppressed; equality is executable.

---

### Task 1: Lock the execution and after-cost compiler contract

**Files:**
- Modify: `tests/learning/test_causal_alpha_v10_hierarchy.py`
- Modify: `tests/risk/test_pretrade.py`
- Add: `tests/simulation/test_causal_alpha_v10_execution_contract.py`
- Modify: `trade_rl/learning/causal_alpha_v6.py`
- Modify: `trade_rl/learning/causal_alpha_v10_hierarchy.py`

**Interfaces:**
- Consumes: `execution_entry_threshold: float` and `execution_no_trade_band: float` supplied by V10 workflow wiring.
- Produces: `causal_alpha_v10_hierarchical_target_path(...) -> CausalAlphaV6TargetPath` whose identity includes those execution values.

- [ ] **Step 1: Write boundary and integration regressions**

Cover the maintained contract `entry_threshold=0.10`, `no_trade_band=0.05`: cap `0.099` remains flat; cap `0.10` is eligible. Separately pin that with `entry_threshold=0`, a target equal to a `0.05` band is executable. Compare the compiler result directly with `PreTradeRisk` in `tests/simulation`.

- [ ] **Step 2: Record RED evidence where the available CI path permits it**

The regression tests were authored before the production implementation. GitHub's Rebuilt Core currently has pre-existing formatting failures before its full pytest step, so record any unexecuted RED layer explicitly rather than claiming it ran.

- [ ] **Step 3: Implement full flat-entry eligibility**

Add `"execution_contract_hold"` to `CAUSAL_ALPHA_V6_TARGET_REASONS`. Validate both execution thresholds in the V10 hierarchy compiler and allow a flat entry only when:

```python
abs(candidate_target) >= max(entry_threshold, no_trade_band)
```

A blocked observation resets pending entry confirmation and records `execution_contract_hold`.

- [ ] **Step 4: Keep soft liquidity capacity out of held-position resizing**

A liquidity cap may reduce an entry candidate. Once the position is held, ordinary liquidity-cap reductions do not emit smaller intermediate targets; V10 keeps the previous strategic target. Existing risk-cap projection remains immediate and is not weakened.

- [ ] **Step 5: Add cost-aware entry**

Use the existing V6 `execution_cost_multiplier = 1.5`; do not introduce a V10 tuning parameter. The actual decision oracle is the existing `causal_alpha_v6_fast_objective`, so a coherent entry must have positive objective after uncertainty, execution cost, and edge margin. Update V10 objective evidence with the same function.

- [ ] **Step 6: Bind compiler identity**

Build a V10 compiler-contract digest from V10 config identity, `entry_threshold`, `no_trade_band`, the fixed V6 economic config digest, and a dedicated schema. Confirm that changing an execution threshold changes target-path identity without changing V8/V9 control identity.

- [ ] **Step 7: Verify focused compiler/risk tests**

```bash
uv run pytest -q \
  tests/learning/test_causal_alpha_v10_hierarchy.py \
  tests/risk/test_pretrade.py \
  tests/simulation/test_causal_alpha_v10_execution_contract.py
```

---

### Task 2: Wire the simulator-authoritative contract into V10 Selection

**Files:**
- Modify: `trade_rl/workflows/universal_causal_alpha_v10_stage_entry.py`
- Modify: `tests/workflows/test_universal_causal_alpha_v10_stage_entry.py`

**Interfaces:**
- Produces: `_execution_rebalance_contract(prepared: Any, symbol: str) -> tuple[float, float]` containing `(entry_threshold, no_trade_band)`.
- `_target_paths(..., execution_rebalance_contract=...)` forwards the pair to the hierarchy compiler only; V8/V9 controls are untouched.

- [ ] **Step 1: Test environment resolution and resource cleanup**

Create a fake environment with `entry_threshold=0.10` and `no_trade_band=0.05`; assert the resolver returns `(0.10, 0.05)` and closes the temporary environment.

- [ ] **Step 2: Test runtime drift rejection**

Create a replay environment whose contract differs from the compiler pair and assert fail-closed rejection before simulator evaluation.

- [ ] **Step 3: Implement resolver, wiring, and drift guard**

Use existing `_environment(prepared, symbol)`, validate both values, always close temporary resolver environments in `finally`, resolve one pair per symbol, pass it into V10 hierarchy compilation, and compare the pair again on every actual replay environment before evaluation.

- [ ] **Step 4: Verify focused workflow tests**

```bash
uv run pytest -q \
  tests/workflows/test_universal_causal_alpha_v10_stage_entry.py \
  tests/learning/test_causal_alpha_v10_hierarchy.py \
  tests/simulation/test_causal_alpha_v10_execution_contract.py
```

---

### Task 3: Full verification and falsification review

- [ ] **Step 1: Run related V10 suites**

```bash
uv run pytest -q \
  tests/learning/test_causal_alpha_v10_fit.py \
  tests/learning/test_causal_alpha_v10_hierarchy.py \
  tests/workflows/test_universal_causal_alpha_v10_gates.py \
  tests/workflows/test_universal_causal_alpha_v10_stage_entry.py \
  tests/risk/test_pretrade.py \
  tests/simulation/test_causal_alpha_v10_execution_contract.py
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

Re-read the original report and final diff from the public contracts rather than implementation intent. Check specifically for: entry-threshold versus no-trade-band precedence, strict `<` versus `<=` mismatches, confirmation advancing on non-executable observations, soft liquidity silently producing smaller targets, V8/V9 control changes, hidden hyperparameter introduction, stale replay identity reuse, runtime contract drift, and cost evidence disagreeing with the actual entry gate.

- [ ] **Step 4: Inspect final PR HEAD and CI**

Require all applicable checks on the exact final head to finish successfully before marking the PR ready. Classify pre-existing main failures separately; do not claim full CI green while they remain.

- [ ] **Step 5: Do not merge**

Report the draft PR, exact final HEAD, verification evidence, unverified DB-backed Selection rerun, baseline CI blockers, and residual risks. Merge to `main` only with explicit user authorization.
