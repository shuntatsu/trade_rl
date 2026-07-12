# Baseline-Anchored Residual RL

[日本語](ja/BASELINE_RESIDUAL_RL.md) | **English**

This document describes the implemented research path for the baseline-anchored residual policy. [`ARCHITECTURE.md`](ARCHITECTURE.md) remains the normative system architecture.

## Purpose

The direct-weight PPO must discover direction, allocation, turnover control, and risk behavior simultaneously. Under weak signal and realistic cost, its safe solution can become a completely flat portfolio. The residual path changes the identity action from flat to an executable trend baseline.

```text
identity action [0, 0]
  -> base_trend_v2
  -> HTF proposal constraint
  -> shared post-processing
  -> execution
```

The RL policy can only change two quantities:

- `action[0]`: continuous fast/base/slow trend mixing;
- `action[1]`: residual-alpha budget, bounded to 30% of gross.

## Main contracts

- `TrendFamily` depends on price history and UTC bar timestamps, not current account weights or slice-relative indices.
- One environment action advances one complete `decision_every` interval and returns one aggregated reward.
- The resolved `decision_every` value is passed explicitly to every residual environment and written back to the run configuration; invalid or zero values fail closed.
- Hybrid and shadow books use independent portfolio state but identical prices, funding, costs, HTF constraints, post-processing, and hard risk checks.
- Reward is the hybrid interval log return minus the shadow base-trend interval log return.
- Identity action must match the shadow weights, cost, and PnL within the tested numerical tolerances.
- The residual-alpha artifact is fitted only on development data whose labels are known. Holdout and Serving load the frozen artifact and never refit it.
- The residual-alpha dataset identity hashes the ordered schema, timestamps, training feature values, and target price history, rather than metadata alone.
- HTF operates on the desired proposal before stateful post-processing, so a neutral signal cannot repeatedly halve an already-constrained position.
- Perfect and noisy oracles remain diagnostic and never participate in a mandatory release gate.
- Sharpe and Sortino annualization use the active post-processor's base-timeframe factor rather than a fixed 1h constant.

## Development matrix

The research runner freezes the configuration on the development validation interval before evaluating the final test interval.

```text
A: pure base_trend_v2 identity policy
B: PPO fast/base/slow trend mixing, alpha disabled
C: fixed +15% residual-alpha budget, diagnostic only
D: PPO trend mixing plus PPO alpha budget
```

Selection rules:

- B is eligible only when it adds positive development excess return without exceeding the drawdown slack and remains non-negative under 2x development costs.
- C is never selected as a release policy; it measures whether the frozen alpha sleeve has standalone value. C must itself beat A and survive 2x development costs before alpha can be enabled in D.
- D is selected only when B/C/D eligibility conditions hold, D survives 2x development costs, and D strictly beats both B and C under normal development costs.
- Otherwise the runner falls back to B when trend mixing is eligible, or A (`baseline_only`) when it is not.

Only the frozen selected configuration is then evaluated once on the final test interval under 1x and 2x costs.

## Checkpoint selection

Residual PPO uses a zero-initialized action head, so the initial deterministic policy is exactly the identity action. Validation is aligned to rollout boundaries and uses the median excess log return across contiguous blocks. A checkpoint is eligible only when:

- median block excess is positive; and
- at least half of the blocks are positive.

When no checkpoint is eligible, that seed restores the identity snapshot instead of an arbitrary trained policy.

## Run tiers

`timesteps` count decision transitions, not base bars.

- `smoke`: at least 5 PPO updates;
- `research`: at least 50 updates and 3 seeds;
- `release`: at least 100 updates and 5 seeds.

For 8 environments and `n_steps=256`, these correspond to 10,240, 102,400, and 204,800 minimum timesteps.

## Running the research path

Use the dedicated runner:

```bash
uv run python scripts/run_baseline_residual.py \
  --source postgres \
  --pg-source binance \
  --base-timeframe 1h \
  --decision-every 4 \
  --timesteps 102400 \
  --ensemble 3 \
  --signal-model gbm \
  --run-tier research \
  --output output/baseline_residual_research
```

The general control-plane entrypoint also accepts the mode, but it must be explicitly research-only:

```bash
uv run python scripts/run_pipeline.py \
  --action-mode baseline-residual \
  --no-register \
  --source postgres \
  --timesteps 102400 \
  --ensemble 3 \
  --output output/baseline_residual_research
```

## Artifacts

The research runner writes:

- `residual_alpha.json` — frozen residual-alpha model and content-bound data identity;
- `B_trend_mix_model.zip` or `B_trend_mix_ensemble/`;
- `D_combined_model.zip` or `D_combined_ensemble/` when the alpha gate passes;
- `residual_train_report.json` — A/B/C/D development results at 1x and 2x costs, frozen selection, final relative evaluation, gates, and diagnostics;
- `residual_model_manifest.json` — dataset, training, selected policy mode, and selected alpha-activation identity.

## Serving schema

Residual ServingBundles use:

- action schema `baseline_residual_v1`;
- observation schema version 2;
- policy mode `ppo_residual_ensemble` or `baseline_only`;
- an absolute-time TrendFamily configuration;
- a frozen `residual_alpha.json` artifact;
- an explicit `residual_alpha_enabled` flag bound to the selected configuration;
- composer and HTF proposal-constraint configuration.

`baseline_only` bundles contain no PPO model and always disable residual alpha. B bundles also disable residual alpha even when the frozen artifact's research gate passed; only selected D bundles enable it. Serving supplies the identity action for `baseline_only` and runs the same trend, composer, HTF, post-processing, guardrail, and risk path. Schema mismatch, missing timestamps, missing alpha artifact, symbol or feature-order mismatch, an incompatible model layout, or a declaration that enables a gate-failing alpha artifact fails closed.

## Release boundary

Residual candidate construction and Serving validation are implemented, but the top-level registration path intentionally remains disabled. A residual invocation without `--no-register` raises an error until the sealed multi-fold residual walk-forward and release-evidence workflow is implemented and verified.

This boundary is deliberate. A single development/test run, successful CI, or a valid bundle shape does not establish profitability or Production readiness. Production remains **NO-GO**.
