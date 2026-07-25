# Constrained PPO PR B TDD Status

Date: 2026-07-26
Branch: `agent/constrained-ppo-cost-critics`
Base implementation: PR #188 head `f1b5ced121da8ef3c557b4589560d6c7f522f232`

## Current unit

Tasks 1 and 2 are under test-driven implementation:

- typed canonical Cost Critic schema;
- explicit continuous and rare-event cost families;
- event-cost `gamma_c=1.0` as the canonical semantic setting;
- independent per-cost `gae_lambda`;
- pure vector-environment cost returns and GAE;
- explicit true-termination and time-limit-truncation behavior;
- no actor penalty or Lagrange multiplier state.

The current head contains the RED tests and minimal GREEN candidates for `cost_learning.py` and `cost_returns.py`. The PR base is temporarily pointed at `main` only to trigger the repository CI for the exact stacked head; after verification it returns to `agent/constrained-ppo-design`.
