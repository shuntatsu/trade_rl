# Reward / Boundary Reconciliation — Execution Correction

## Conclusion

The completed PR #369 boundary contract cannot be combined with PR #370's original three-candidate default in one `MarketWalkForwardConfig`.

`MarketWalkForwardConfig` deliberately requires every candidate in one selection catalog to share environment dynamics. After reconciliation:

- gamma-one PPO and gamma-one Lagrangian use `finite_horizon_termination` with time-to-go;
- discounted Lagrangian uses `external_truncation` without time-to-go.

Those are different MDP contracts. Treating them as one candidate set makes the workflow fail before training and would also weaken fair within-catalog selection.

## Corrected implementation

- Keep `walk-forward-target-weight-constrained-growth.json` as the default target-weight catalog.
- Limit that catalog to the two gamma-one candidates:
  1. `training-target-weight-growth-ppo.json`;
  2. `training-target-weight-constrained-growth.json`.
- Add `walk-forward-target-weight-constrained-growth-discounted.json` as a dedicated one-candidate time-preference ablation.
- Compare the two workflows only through common economic evidence such as after-cost net log growth, not raw training reward or direct candidate selection.
- Preserve `training-full.json` as an explicit legacy comparison.

## Evidence that triggered the correction

On reconciliation head `ac08f52e389eceaaecf32b2dac349779ac42f2ac`, both Ubuntu and Windows compatibility jobs reproduced the same failure while static checks and the training image passed:

```text
ValueError: walk-forward candidates must share environment dynamics,
action, risk, reward, and trend contracts
```

The failure occurred while loading the default target-weight walk-forward after the discounted candidate acquired its correct external-truncation boundary. The fix therefore changes the catalog boundary, not the validator or test strictness.
