# Causal Alpha V10 Closed-Loop Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the V10 hierarchical candidate decide from simulator-realized exposure on every replay step while preserving frozen economics, controls, and gate thresholds.

**Architecture:** Precompute immutable causal hierarchy inputs, execute only the hierarchical candidate through the existing `evaluate_action_path(model=...)` path, and build the final V10 target artifact from the closed-loop requested actions plus a V10-specific trace. Bind the full `PreTradeRiskConfig` into pre-replay identity and use absolute decision indices for 4-hour cadence. V8/V9 controls remain static target paths.

**Tech Stack:** Python 3.12, NumPy, pytest, Ruff, Mypy, import-linter, GitHub Actions.

**Spec:** `docs/implementation-plans/specs/2026-08-29-causal-alpha-v10-closed-loop-execution-design.md`

## Global Constraints

- Reward remains `100 * net_log_return`.
- No Selection/Admission numerical gate changes.
- No change to 4h/72h horizons or confirmation counts.
- No one-minute data, holdout tuning, BC, or RL opening.
- V8/V9 control target generation stays unchanged.
- PreTrade/simulator realized exposure remains authoritative.
- No direct position flip.
- New V10-only diagnostics do not expand the generic V6 reason vocabulary.
- Old open-loop V10 replay leaves must not be reusable as closed-loop leaves.

---

### Task 1: Lock closed-loop state and absolute cadence with failing tests

**Files:**
- Add: `tests/learning/test_causal_alpha_v10_closed_loop.py`
- Modify: `tests/learning/test_causal_alpha_v10_hierarchy.py`

**Interfaces:**
- Consumes: existing `trade_rl.learning.causal_alpha_v10_hierarchy` module.
- Produces: regression expectations for a callable hierarchy policy factory/class, realized-weight feedback, absolute cadence, hard-risk behavior, and split diagnostics.

- [ ] **Step 1: Add a failing API/feedback regression**

Create a test that imports the module, resolves the new API through `getattr` so collection succeeds before implementation, and fails by assertion when the API is absent:

```python
import trade_rl.learning.causal_alpha_v10_hierarchy as hierarchy


def test_v10_closed_loop_policy_uses_realized_weight_on_next_hold() -> None:
    factory = getattr(hierarchy, "prepare_causal_alpha_v10_hierarchy_policy", None)
    assert callable(factory), "closed-loop V10 hierarchy policy is not implemented"
```

Extend the same test after the API exists to feed an entry observation at `0.0`, then a next-step observation at realized `0.05` after a requested `0.10`; assert the next ordinary hold action is `0.05`, not `0.10`.

- [ ] **Step 2: Add failing hard-cap regressions**

Use the maintained hysteresis semantics:

```python
current = 0.10
risk_cap = 0.04
entry_threshold = 0.10
exit_threshold = 0.03
no_trade_band = 0.05
```

Assert the policy requests `0.0` and records `risk_cap_flatten`. Add a second case with a partial cap that is same-direction, at/above `entry_threshold`, and changes exposure by at least the band; assert the capped partial target is requested and trace reason is `risk_cap_projection`.

- [ ] **Step 3: Add failing absolute-cadence regression**

Build two hierarchy inputs with overlapping absolute decision indices but different first rows, for example `np.arange(1, 34)` and `np.arange(16, 49)`. Assert decision index `16` is classified as a fast cadence point in both paths. The old `offset % 16` implementation must fail this test.

- [ ] **Step 4: Add failing diagnostic-separation regression**

Assert a flat coherent candidate below the execution floor records `entry_floor_hold`, while an already-held position above a lower soft liquidity cap records `liquidity_capacity_hold`. The generic V6 reason should remain `hold_flat`/`hold_position` respectively.

- [ ] **Step 5: Run RED verification**

Run in an isolated GitHub verification branch whose parent is the test-only PR commit:

```bash
uv run pytest -q \
  tests/learning/test_causal_alpha_v10_closed_loop.py \
  tests/learning/test_causal_alpha_v10_hierarchy.py
```

Expected: FAIL for missing closed-loop policy/old cadence/old mixed diagnostic behavior, not syntax/import-collection errors.

---

### Task 2: Implement immutable execution contract and hierarchy input identity

**Files:**
- Modify: `trade_rl/learning/causal_alpha_v10_hierarchy.py`
- Modify: `trade_rl/workflows/universal_causal_alpha_v10_stage_entry.py`
- Modify: `tests/workflows/test_universal_causal_alpha_v10_stage_entry.py`

**Interfaces:**
- Produces: `CausalAlphaV10ExecutionContract` with full PreTrade identity and a digest.
- Produces: immutable `CausalAlphaV10HierarchyPolicyInput` (name may be shortened only if tests/docs are updated consistently) whose digest binds all pre-replay causal inputs and the execution contract.
- Produces: `_environment_rebalance_contract(environment) -> CausalAlphaV10ExecutionContract` and `_execution_rebalance_contract(prepared, symbol) -> CausalAlphaV10ExecutionContract`.

- [ ] **Step 1: Extend workflow contract tests before production changes**

Change the existing resolver test to use an actual `PreTradeRiskConfig` with non-default `exit_threshold`, drawdown thresholds, and turnover limit. Assert the resolved contract exposes the three policy thresholds and that changing any bound config field changes its digest.

Add drift tests for at least `exit_threshold` and `drawdown_start` so replay rejects a runtime whose full PreTrade identity differs despite equal entry/no-trade values.

- [ ] **Step 2: Implement the execution contract value**

In the learning layer, define a frozen value object that stores the maintained PreTrade fields without importing workflow code:

```python
@dataclass(frozen=True, slots=True)
class CausalAlphaV10ExecutionContract:
    max_gross: float
    max_abs_weight: float
    max_turnover: float | None
    entry_threshold: float
    exit_threshold: float
    no_trade_band: float
    drawdown_start: float
    drawdown_stop: float
    emergency_turnover_override: bool
    fail_closed_tolerance: float

    @property
    def digest(self) -> str:
        return content_digest({
            "schema_version": "causal_alpha_v10_execution_contract_v1",
            ...
        })
```

Validate finite/range-compatible values needed by the V10 policy and preserve `None` for unlimited turnover.

- [ ] **Step 3: Resolve the full contract from `PreTradeRiskConfig`**

Replace the tuple resolver with a typed contract built from the actual environment `pre_trade_risk.config`. Re-read it immediately before replay and compare contract digests, not only a two-float tuple.

- [ ] **Step 4: Implement immutable hierarchy input digest**

Move input validation/alignment out of the old monolithic path compiler into one immutable pre-replay object/factory. Its digest must include:

```text
execution_contract.digest
v10_config.digest
V6 economic config digest
source forecast digest
dual fit digest
all aligned decision/head/cost/cap/regime/actionable arrays
attribution boundary values
initial weight
```

Use `content_and_arrays_digest` for array identity. Changing a bound execution field must change this input digest even when predicted targets would otherwise be identical.

- [ ] **Step 5: Verify Task 2**

```bash
uv run pytest -q \
  tests/workflows/test_universal_causal_alpha_v10_stage_entry.py \
  tests/learning/test_causal_alpha_v10_closed_loop.py
uv run mypy \
  trade_rl/learning/causal_alpha_v10_hierarchy.py \
  trade_rl/workflows/universal_causal_alpha_v10_stage_entry.py
```

Expected: execution identity/drift tests PASS; closed-loop behavior tests may remain RED until Task 3.

---

### Task 3: Implement the sequential V10 policy with realized-state feedback

**Files:**
- Modify: `trade_rl/learning/causal_alpha_v10_hierarchy.py`
- Modify: `trade_rl/learning/causal_alpha_v10.py`
- Modify: `trade_rl/learning/causal_alpha_v6.py`
- Test: `tests/learning/test_causal_alpha_v10_closed_loop.py`
- Test: `tests/learning/test_causal_alpha_v10_hierarchy.py`

**Interfaces:**
- Produces: `prepare_causal_alpha_v10_hierarchy_policy(...)` returning a stateful model compatible with `evaluate_action_path(model=...)`.
- Policy method: `predict(observation: object, deterministic: bool = True) -> tuple[np.ndarray, object]`.
- Produces after complete replay: `target_path()` or equivalent finalizer returning a `CausalAlphaV10TargetPath` whose embedded V6 path is the actual requested action path and whose V10 trace covers every decision.

- [ ] **Step 1: Implement observation parsing and sequential guard**

Require mapping observations with one finite `current_weights` value. On the first step, require it to match the frozen initial weight within the existing action tolerance. Increment exactly one offset per `predict` call and reject calls after the input rows are exhausted.

- [ ] **Step 2: Make realized weight the decision current**

At each step:

```python
observed_current = float(current_weights[0])
requested = observed_current
```

Use `observed_current` for sign, objective delta, hold behavior, and transition decisions. Do not carry the previous requested target forward as the current portfolio.

If the prior realized sign was non-zero and the new realized position is flat without a policy-requested ordinary exit, reset entry/exit/slow confirmation state and record `realized_state_reset`. If the realized sign changes directly from positive to negative or vice versa without an allowed intervening flat state, raise a fail-closed error.

- [ ] **Step 3: Implement hard-risk projection semantics**

When `abs(observed_current) > risk_cap + epsilon`, compute `partial = sign(current) * risk_cap`.

A same-direction partial reduction is directly executable only when all are true:

```python
abs(partial) > execution.exit_threshold
abs(partial) >= execution.entry_threshold
abs(partial - observed_current) >= execution.no_trade_band
```

If executable, request `partial` and trace `risk_cap_projection`. Otherwise request `0.0`, trace `risk_cap_flatten`, and latch a flatten state until a later observation is flat.

Hard-risk logic runs before cadence/actionable signal logic.

- [ ] **Step 4: Use absolute decision cadence**

Replace every episode-offset cadence check with:

```python
cadence = (
    int(input.decision_indices[offset]) % config.fast_horizon_decisions == 0
)
```

Do not alter the 16-decision/4-hour value.

- [ ] **Step 5: Split V10 diagnostics from V6 reasons**

Revert `execution_contract_hold` from `CAUSAL_ALPHA_V6_TARGET_REASONS`. For V6-compatible reasons use the existing generic terms; attach per-decision V10 hierarchy reasons to `CausalAlphaV10TargetPath` (or a dedicated immutable trace contained by it).

For the hierarchical candidate the trace length must equal target-path decision count and values must belong to a fixed V10 trace vocabulary. For V8/V9 control candidates, the trace is empty. Include trace identity in the V10 target digest and bump the V10 target schema if serialization shape changes.

- [ ] **Step 6: Preserve the pure path function as a compatibility harness**

Keep `causal_alpha_v10_hierarchical_target_path(...)` available for unit compatibility by driving the new sequential policy with synthetic observations whose `current_weights` equal the previously requested target. This preserves old deterministic open-loop unit expectations except where absolute cadence/trace semantics intentionally changed, while production hierarchical replay uses true simulator observations.

- [ ] **Step 7: Verify Task 3**

```bash
uv run pytest -q \
  tests/learning/test_causal_alpha_v10_closed_loop.py \
  tests/learning/test_causal_alpha_v10_hierarchy.py \
  tests/learning/test_causal_alpha_v10_fit.py
```

Expected: all closed-loop, hard-risk, cadence, state-machine, no-direct-flip, and fit tests PASS.

---

### Task 4: Wire only hierarchical replay through the closed-loop model path

**Files:**
- Modify: `trade_rl/workflows/universal_causal_alpha_v10_stage_entry.py`
- Modify: `tests/workflows/test_universal_causal_alpha_v10_stage_entry.py`
- Add/modify: `tests/simulation/test_causal_alpha_v10_execution_contract.py`

**Interfaces:**
- V8/V9 controls: continue `evaluate_action_path(actions=...)` with precomputed target paths.
- Hierarchical V10: call `evaluate_action_path(model=hierarchy_policy, deterministic=True)`, then finalize the target artifact from policy trace and evaluation actions.

- [ ] **Step 1: Add workflow/integration RED tests**

Create a tiny evaluator environment or existing project fake whose first closed-loop hierarchy entry requests `0.10` but whose next observation reports `current_weights=0.05`. Assert the policy's next hold action is `0.05` and the evaluator records no repeated proposal solely from restoring `0.10`.

Add a control-path test asserting V8/V9 candidates are still evaluated with their unchanged static target arrays/digests.

- [ ] **Step 2: Separate static controls from hierarchical policy preparation**

Refactor `_target_paths` so the control targets are still computed exactly as before, while the hierarchy side returns/prepares the immutable hierarchy policy input rather than a final target path.

Do not change `_fit_v9_wave`, V8 target generation, candidate mapping, or gate code.

- [ ] **Step 3: Execute hierarchical candidate closed-loop**

In `_build_replay` (or a narrowly separated `_build_hierarchical_replay` helper), instantiate the policy, call the evaluator through `model=`, finalize the V10 target artifact, then build the same V6/V8-compatible replay economics and attribution from the finalized action path.

- [ ] **Step 4: Verify simulator boundary**

```bash
uv run pytest -q \
  tests/simulation/test_causal_alpha_v10_execution_contract.py \
  tests/workflows/test_universal_causal_alpha_v10_stage_entry.py \
  tests/learning/test_causal_alpha_v10_closed_loop.py
```

Expected: realized-state feedback and static control isolation PASS.

---

### Task 5: Make replay leaf resume identity closed-loop safe

**Files:**
- Modify: `trade_rl/workflows/universal_causal_alpha_v10_stage_entry.py`
- Modify: `tests/workflows/test_universal_causal_alpha_v10_stage_entry.py`

**Interfaces:**
- Bump `_REPLAY_LEAF_SCHEMA` to a new version.
- Leaf stores `candidate_input_digest`.
- Control input digest = existing static `CausalAlphaV10TargetPath.digest`.
- Hierarchical input digest = immutable hierarchy policy-input digest.

- [ ] **Step 1: Add stale-leaf RED regressions**

Construct/store a leaf with one execution contract, then resolve the same causal forecasts under a contract differing only in `exit_threshold` or drawdown threshold. Assert `_load_leaf` refuses reuse before replay.

Add a schema test proving an old v1 leaf is not accepted as a v2 closed-loop leaf.

- [ ] **Step 2: Store and compare pre-replay input identity**

Change leaf creation to include `candidate_input_digest`. Change leaf load validation to compare this digest plus the existing candidate/config/contract/symbol/episode/fit identities. For hierarchical candidates, do not require a precomputed final target digest because it does not exist until replay.

After load, continue validating the stored replay metric and final V10 target digest embedded in the metric/artifact payload.

- [ ] **Step 3: Verify restart behavior**

```bash
uv run pytest -q tests/workflows/test_universal_causal_alpha_v10_stage_entry.py
```

Expected: same-input resume reuses the leaf; any bound execution-contract change or schema-v1 leaf fails closed.

---

### Task 6: Targeted regression, static checks, and architecture review

**Files:** all changed production/tests.

- [ ] **Step 1: Run affected behavior suites**

```bash
uv run pytest -q \
  tests/learning/test_causal_alpha_v10_fit.py \
  tests/learning/test_causal_alpha_v10_hierarchy.py \
  tests/learning/test_causal_alpha_v10_closed_loop.py \
  tests/risk/test_pretrade.py \
  tests/simulation/test_causal_alpha_v10_execution_contract.py \
  tests/workflows/test_universal_causal_alpha_v10_gates.py \
  tests/workflows/test_universal_causal_alpha_v10_stage_entry.py
```

- [ ] **Step 2: Run V6/V8/V9 compatibility suites**

Run the existing target/replay/gate tests covering V6, V8, and V9 to prove the new V10 model path did not alter controls.

- [ ] **Step 3: Run static/architecture checks**

```bash
uv run ruff check <changed-python-files>
uv run ruff format --check <changed-python-files>
uv run mypy \
  trade_rl/learning/causal_alpha_v10.py \
  trade_rl/learning/causal_alpha_v10_hierarchy.py \
  trade_rl/workflows/universal_causal_alpha_v10_stage_entry.py
uv run lint-imports
uv run vulture trade_rl tests --min-confidence 100
```

- [ ] **Step 4: Falsification review**

Attempt to break the implementation with:

- drawdown-reduced realized exposure followed by supportive signals;
- partial fills leaving exposure above/below risk cap;
- external flatten between cadence points;
- direct observed sign flip;
- exact equality at entry/exit/band boundaries;
- episode slices shifted by one row;
- unchanged V8/V9 control inputs;
- resume after each PreTrade config field changes;
- a current weight that is absent, non-finite, or wrong dimensionality.

Fix any discovered in-scope defect and rerun affected tests.

---

### Task 7: Full verification against main and final PR review

- [ ] **Step 1: Run source-equivalent full suite**

Use a temporary verification branch whose parent is the exact final PR HEAD and whose only extra diff is a verification workflow. Run the repository full pytest/coverage/critical-coverage checks. Separately reproduce any global blocker on `main` before classifying it as baseline.

- [ ] **Step 2: Compare final source with main under identical baseline exclusions**

Run the same full-suite command on `main` and final source with only independently proven baseline blockers deselected and `--cov-fail-under=0`. Compare pass/fail counts and total/critical coverage.

- [ ] **Step 3: Inspect normal final-HEAD CI**

Record PostgreSQL catalog, compatibility, training image/capability, Ruff, formatting, typing, and tests that actually ran. Do not report ordinary CI as green if a baseline job remains red.

- [ ] **Step 4: Inspect final diff and PR state**

Require:

```text
base = intended main SHA/current main
head = exact reported PR HEAD
PR remains Draft/open/unmerged
no verification workflow files in PR diff
no V8/V9 strategy or gate-threshold changes
no debug/temp files
```

Update the PR body with the new closed-loop architecture, RED evidence, exact verification results, remaining CI baseline blockers, and explicit statement that DB-backed 216-leaf Selection is still unrun.

- [ ] **Step 5: Do not merge**

Keep PR #423 Draft. Merge only with explicit user authorization after the code quality gate and later research-stage evidence are dispositioned.
