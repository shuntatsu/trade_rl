# Hierarchical Gate-Target BC Actor Design

## Status

Approved direction: C. The policy keeps the existing continuous target-weight action contract and introduces a hierarchical actor internally.

## Problem

The current behavior-cloning trainer optimizes one unweighted mean-squared error over every sample and action dimension. In a dataset dominated by hold or small-change actions, the actor can reduce average MSE while collapsing to nearly constant target weights. The current mandatory gate only checks relative MSE improvement, so a model with zero causal holdout trades can pass.

The redesign must solve the collapse without changing the environment action space, execution semantics, PPO interface, checkpoint loading path, or serving output contract.

## Goals

1. Separate the decision to change a position from the target-weight proposal.
2. Keep one continuous target weight per asset as the only external action.
3. Use the same actor structure for BC, PPO, checkpoint reload, export, and serving.
4. Make hold-dominance and zero-trade collapse directly observable and fail closed.
5. Preserve asset-order identity, inactive-asset masking, architecture digesting, and deterministic reconstruction.
6. Keep Oracle loss-making intervals in trajectory order; never delete rows after observing outcomes.

## Non-goals

- No discrete or tuple Gym action space.
- No separate BC-only actor discarded before PPO.
- No post-hoc deletion of losing Oracle trades.
- No claim that the hindsight Oracle return is an achievable causal return.
- No increase to teacher period or BC epochs until the non-degenerate gate passes.

## External Contract

The environment continues to receive:

```text
Box(low=-1, high=1, shape=(n_symbols,))
```

Each component remains an absolute target weight in dataset symbol order. Pre-trade risk, no-trade bands, execution delay, partial fills, fees, funding, borrow, margin, and liquidation remain downstream responsibilities.

## Actor Architecture

For asset `i`, the shared actor produces:

- `gate_logit_i`: whether the target should materially differ from the current effective portfolio weight.
- `target_logit_i`: an unconstrained proposal converted to a bounded target with `tanh`.

The observation contract must expose the current effective portfolio weight as an explicit, unnormalized per-asset field. It must not be reconstructed from an encoded latent or inferred from the previous submitted action because partial fills and pending orders can make those values different.

The composed deterministic action is:

```text
gate_i = sigmoid(gate_logit_i / temperature)
proposal_i = tanh(target_logit_i)
composed_i = current_weight_i + gate_i * (proposal_i - current_weight_i)
```

`composed_i` is clamped to `[-1 + eps, 1 - eps]`. Because SB3's squashed Gaussian applies `tanh` to its Gaussian mean, the action head returns:

```text
mean_logit_i = atanh(composed_i)
```

Thus deterministic distribution mode remains exactly `composed_i`. Inactive dimensions remain masked through the existing masked distribution.

The gate is soft during BC and PPO. A hard threshold is not inserted into the policy because it would break differentiability and complicate PPO log-probability semantics. The existing downstream no-trade band remains responsible for turning small continuous changes into no execution.

## Observation Contract Extension

Structured sequence metadata gains an explicit `current_weight_column` identity. `SequenceAssetFeatureExtractor` appends the raw current-weight vector to its output after encoded asset, pooled, global, and active fields. `SharedAssetActorCriticExtractor` parses the vector and includes each asset's current weight in the actor context.

The current-weight field must be:

- derived from effective `BookState.weights` at the decision instant;
- unnormalized and bounded;
- included in observation schema and architecture digests;
- reconstructed identically by rollout compression and serving loaders;
- covered by observation parity tests.

## Teacher Labels

A new immutable label artifact is derived without deleting or reordering samples:

```text
gate_label_i = abs(teacher_target_i - current_weight_i) >= change_threshold
```

Labels are additionally masked by active/tradable identity. A transition to zero is a positive gate event when it materially changes current exposure. Reversals are one gate event with a target on the opposite side; they are not split into synthetic rows.

The label artifact stores:

- gate labels;
- current weights;
- target actions;
- active masks;
- event type per asset: hold, enter, resize, exit, reverse;
- action-distribution and run-length diagnostics;
- source teacher artifact digest and label-config digest.

## BC Objective

The trainer optimizes:

```text
L = gate_weight * L_gate
  + target_weight * L_target
  + composed_weight * L_composed
```

`L_gate` is class-balanced BCE-with-logits over active dimensions. The positive class weight is computed from the training partition only and capped to avoid unstable gradients.

`L_target` is masked Smooth L1 loss over positive gate events. It teaches where to move only when a meaningful change is required. The denominator is the number of positive active dimensions, not the full tensor size.

`L_composed` is Smooth L1 loss between the final composed deterministic action and the teacher target over all active dimensions. It preserves end-to-end target-weight correctness and keeps the two heads jointly identifiable.

Chronological validation remains. Early stopping uses a deterministic weighted validation score, while all component metrics are exported separately.

## Evaluation Split

Two distinct evaluations are required.

### Teacher reconstruction validation

This uses the chronological validation tail from the teacher-labelled range. It is mandatory for BC warm-start quality and reports:

- gate precision, recall, F1, and positive support;
- active-event target RMSE;
- composed action RMSE;
- activity ratio versus teacher;
- event-type recall for enter, resize, exit, and reverse;
- all-hold and all-trade collapse flags.

### Causal market holdout

This uses observations available at each decision and evaluates the policy path economically. It does not require high agreement with a future-dependent Oracle. It is mandatory only for non-degeneracy and catastrophic-risk checks:

- executed trade count;
- submitted and executed action-change rate;
- after-cost net return;
- maximum drawdown;
- regret versus configured causal baselines;
- zero-trade and constant-action collapse flags.

The report must clearly label Oracle metrics as hindsight diagnostics and causal holdout metrics as realizable-policy evidence.

## Mandatory Gates

BC warm-start fails closed when any configured mandatory condition fails. At minimum:

1. relative composed-loss improvement below threshold;
2. gate recall below threshold with sufficient positive support;
3. gate precision below threshold with sufficient predicted-positive support;
4. active-event target RMSE above threshold;
5. activity ratio outside configured lower and upper bounds;
6. all-hold, all-trade, or constant-action collapse;
7. causal holdout executed trade count below the configured minimum when teacher support exists;
8. causal holdout after-cost result breaches the configured catastrophic-regret limit.

Insufficient support is reported separately from pass. It cannot silently become pass.

## Configuration and Identity

The training schema is bumped to `training_run_config_v3`. The following fields are explicit and architecture-bound:

```text
policy_actor_head = hierarchical_gate_target_v1
hierarchical_gate_temperature
behavior_cloning_gate_loss_weight
behavior_cloning_target_loss_weight
behavior_cloning_composed_loss_weight
behavior_cloning_gate_change_threshold
behavior_cloning_max_positive_class_weight
behavior_cloning_min_gate_precision
behavior_cloning_min_gate_recall
behavior_cloning_max_active_target_rmse
behavior_cloning_min_activity_ratio
behavior_cloning_max_activity_ratio
behavior_cloning_min_causal_holdout_trades
behavior_cloning_max_causal_holdout_regret
```

No field is silently defaulted when loading v3. Existing v2 configs are rejected with a migration message. Actor-head identity and all structural parameters are included in architecture digest, checkpoint manifest, training report, export metadata, and serving validation.

## Compatibility

- Environment action shape and symbol order do not change.
- PPO, CostCriticPPO, and LagrangianPPO use the same hierarchical actor policy class.
- Existing checkpoints with the old action head do not load as the new architecture; mismatch fails before parameter loading.
- Serving emits the same target-weight vector and validates the new architecture digest.
- Oracle and baseline evaluation APIs remain path-level and retain complete chronological trajectories.

## Diagnostics

TensorBoard and structured reports expose:

- gate probability histograms;
- gate positive rate by asset and event type;
- proposal and composed target histograms;
- teacher/policy activity ratio;
- gate precision and recall;
- event target RMSE;
- downstream no-trade suppression rate;
- submitted versus executed action-change rate;
- causal trade count and after-cost return.

A zero-trade policy must be explainable as one of: gate closed, proposal too close to current weight, downstream no-trade suppression, execution rejection, or inactive/tradability masking.

## Rollout and Serving Parity

Index-backed sequence reconstruction must reproduce current weights at each rollout sample. The current weight is stateful and therefore cannot be recovered only from immutable market arrays; compact rollouts must store the minimum per-step current-weight vector or an equivalent deterministic state reference.

Training, evaluation, export, and serving parity tests compare:

- raw current weights;
- gate logits;
- proposal targets;
- composed actions;
- final deterministic target-weight action.

## Migration Sequence

1. Add label and metric contracts without changing the policy.
2. Extend observation and compact-rollout state with explicit current weights.
3. Add hierarchical head and deterministic composition tests.
4. Replace BC objective and expose component metrics.
5. Make reconstruction and causal-collapse gates mandatory.
6. Bind architecture identity through checkpoints and serving.
7. Migrate the full research config to v3.
8. Run short BC verification before increasing data range or epochs.
9. Run BC to PPO across folds and seeds only after all BC gates pass.

## Acceptance Criteria

- The full research config cannot pass BC with zero causal holdout trades when teacher change support is nonzero.
- Deterministic policy mode equals the mathematically composed action within numerical tolerance.
- BC and PPO use one serialized actor structure.
- Observation reconstruction and serving return identical current weights and actions.
- Architecture mismatch fails closed.
- Losing Oracle intervals remain in sequence and are evaluated as part of path-level objective evidence.
- Full tests, Ruff, MyPy, CPU BC-to-PPO smoke, and CUDA smoke pass before merge.
