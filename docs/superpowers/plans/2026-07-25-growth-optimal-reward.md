# Growth-Optimal Reward Implementation Plan

## Goal

Introduce an auditable terminal-growth training profile while preserving all existing reward and artifact contracts.

## Tasks

### 1. Add reward regression tests

Create `tests/rl/test_growth_optimal_reward.py` to prove:

- the configured reward is exactly scaled net interval log growth;
- interval rewards telescope to total episode net log growth;
- terminal-equity loss is not counted again when terminal shaping weights are zero;
- the legacy reward defaults remain unchanged.

### 2. Correct comparator termination semantics

Update `trade_rl/rl/transition.py` and add `tests/rl/test_transition_shadow_failure.py`.

- true termination depends on hybrid insolvency or terminal liquidation;
- shadow-only failure becomes truncation;
- diagnostics use `shadow_<reason>`;
- hybrid failure takes precedence when both books fail.

### 3. Add explicit research profiles

Create:

- `examples/binance-multitimeframe/training-growth-optimal.json`;
- `examples/binance-multitimeframe/walk-forward-growth-optimal.json`;
- `tests/examples/test_growth_optimal_reward_profiles.py`.

Both profiles use pure net log growth, `gamma = 1.0`, finite-horizon observation, realistic execution costs, hard risk constraints, and the existing walk-forward adoption gates.

### 4. Preserve compatibility

Do not modify:

- `trade_rl/rl/rewards.py`;
- the reward schema;
- legacy full-training or walk-forward profiles;
- existing reward diagnostics or fixtures.

### 5. Verify

Run exact-head CI and require:

- Ruff and format;
- Mypy;
- import architecture checks;
- full pytest and coverage;
- critical branch coverage;
- Windows and Ubuntu compatibility;
- training-image build;
- CLI smoke test.

Do not merge while any exact-head check is missing or failing.
