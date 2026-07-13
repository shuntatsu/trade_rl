# Causal Training Hardening Design

## Goal

Make the maintained residual-RL path causal, reproducible, AUM-aware, evaluation-isolated, and fail-closed from environment step through production serving activation.

## Environment and execution

Training time limits are Gymnasium truncations and do not force liquidation. Stable-Baselines3 may therefore bootstrap the terminal observation. Explicit end-of-window liquidation is reserved for sealed evaluation, is terminal, and fails closed when liquidity leaves residual positions.

Policy observations contain current feature availability and current tradability, never synthetic episode progress or next-bar tradability. A decision after close `t` executes at open `t+1`; capacity uses completed volume from bar `t`, while actual tradability at `t+1` remains transition dynamics.

Initial capital is mandatory in quote-currency units. The environment digest includes dataset identity, resolved timing, trend configuration, risk limits, execution costs, reward settings, alpha mode, action and observation schemas, and AUM.

## Training

One immutable PPO configuration contains learning rate, rollout length, batch size, epochs, gamma, GAE, clip range, advantage normalization, entropy/value coefficients, gradient norm, policy type, device, timesteps, and seeds.

Policy artifacts distinguish requested from model-reported actual timesteps and store the resolved device. They also bind the training configuration digest, observation schema, environment digest, and AUM. Ensemble construction rejects inconsistent work, device, environment, or capital identities across seeds.

A small single-environment MLP can remain CPU/environment-bound. GPU utilization is diagnostic only; throughput, reproducibility, and sealed OOS performance are decision criteria.

## Nested Walk-Forward

The concrete fold runner passes only train and checkpoint-validation ranges to candidate trainers. Frozen candidates and the identity baseline are compared on configuration-selection data. Only the selected candidate and baseline are evaluated on the sealed outer-test range, exactly once. When no candidate clears the predeclared uplift threshold, baseline fallback is explicit.

Selected and baseline OOS returns are stitched independently. The final evaluation digest binds fold identities, selection evidence, policy identities, sealed-test evidence, and stitch mode.

## Gates and releases

Gate decisions bind the evaluated dataset, selected policy identity when applicable, and final evaluation digest. Release manifests preserve both selection-evaluation and gate-evaluation identities and reject mismatched dataset, signal, or policy identities.

## Serving

Serving bundle schema v2 binds action schema, observation schema, observation size, environment digest, AUM, dataset, signal, selection, policy, release, and artifact files. Runtime activation validates schemas before policy loading, and inference validates observation width before prediction.

Runtime and Registry require an approved release identity by default. Unreleased research bundles require the explicit `allow_unreleased=True` escape hatch.

## Testing

Regression coverage includes:

- bootstrap-compatible training truncations;
- terminal complete liquidation and fail-closed incomplete liquidation;
- causal observations and next-open capacity;
- explicit PPO settings, actual work, device, environment, and AUM identity;
- real Stable-Baselines3 train/save/load/predict;
- range-scoped nested Walk-Forward selection and sealed OOS execution;
- Gate and Release identity mismatches;
- serving observation-size/schema checks and release-gated activation.

## Remaining boundary

A project-specific real-data loader, fold-local preprocessing adapter, PPO candidate trainer, checkpoint selector, and evaluator must still be connected to the typed workflow requests. No profitability or production-readiness claim follows from this hardening work alone.
