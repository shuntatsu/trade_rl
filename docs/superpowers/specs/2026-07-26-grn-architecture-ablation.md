# Gated Residual Network Architecture Ablation

Date: 2026-07-26
Status: deferred independent experiment
Branch: `agent/constrained-ppo-design`

## 1. Decision

Gated Residual Networks (GRNs) are retained as a high-priority architecture candidate, but they must not be mixed into the constrained PPO foundation.

The constrained-growth work must first establish trustworthy action-path diagnostics, independent cost signals, cost critics, and stabilized Lagrange updates. Only after that baseline is green may GRN variants be compared in an independent PR and experiment identity.

The maintained economic reward remains unchanged:

\[
r_t = 100 \log\left(\frac{V^{net}_{t+1}}{V^{net}_t}\right)
\]

## 2. Motivation

The market value of a feature transformation is state dependent. Momentum, reversal, volatility, liquidity, funding, and portfolio-state information can be useful in one regime and harmful or irrelevant in another.

A GRN lets the model learn whether to apply a nonlinear transformation or preserve the residual input:

\[
z = F(\operatorname{LayerNorm}(x))
\]

\[
g = \sigma(G(\operatorname{LayerNorm}(x)))
\]

\[
y = \operatorname{LayerNorm}(x + g \odot z)
\]

The gate is diagnostic rather than a causal explanation. A high gate value only means that the trained model used that transformation strongly for the observed state.

## 3. Approved insertion points

The first ablation may introduce GRNs only at high-level representation boundaries.

### 3.1 Asset fusion

Replace the current plain fusion MLP with:

```text
multi-timeframe encodings + snapshot + asset state
    -> input projection to d_model
    -> GRN block
    -> GRN block
    -> per-asset token
```

The existing causal TCNs remain unchanged. This avoids simultaneously changing temporal receptive fields and feature fusion.

### 3.2 Purpose-specific adapters

After the shared cross-asset encoder, add small independent adapters:

```text
shared market representation
    |- actor GRN adapter -> action distribution
    |- reward GRN adapter -> reward value
    `- cost GRN adapter -> per-constraint values
```

The actor adapter is limited to one or two blocks. Reward and cost critics may use two or three blocks because long-horizon value estimation is the more likely capacity bottleneck.

### 3.3 Prohibited first-pass locations

The first ablation must not:

- replace the causal TCN blocks;
- add gates inside every temporal convolution;
- alter the cross-asset Transformer depth;
- add recurrent state;
- add variable-selection networks per raw feature;
- change state-dependent exploration;
- change reward or constraint definitions.

These changes would make attribution impossible.

## 4. GRN block contract

The maintained GRN block should use a fixed-width residual path:

```text
LayerNorm
 -> Linear(width, expansion_width)
 -> SiLU
 -> Dropout
 -> Linear(expansion_width, width)
 -> gated residual merge
 -> LayerNorm
```

The gate path is:

```text
LayerNorm(input)
 -> Linear(width, width)
 -> Sigmoid
```

Required properties:

- input and output shape are identical;
- finite-value checks fail closed in tests;
- dropout follows the existing architecture ceiling;
- gate bias is initialized near neutral rather than saturated;
- gate tensors are available to diagnostic hooks without entering observations or rewards;
- checkpoint architecture identity includes block count, width, expansion ratio, and gate initialization.

## 5. Controlled comparison

Three variants must be trained under identical data, seeds, folds, optimizer settings, rollout sizes, rewards, constraints, and execution assumptions.

### Variant A: current MLP

The existing fusion and critic MLPs are retained unchanged.

### Variant B: residual MLP

Plain residual blocks without learned gates are used. This separates the benefit of residual depth from the benefit of gating.

### Variant C: GRN

The approved fusion and purpose-specific adapters use gated residual blocks.

Parameter counts must be matched within approximately five percent where practical. If exact matching is impossible, the report must include parameter count, peak training memory, update time, and environment steps per second.

## 6. Initial capacity grid

The first comparison should remain small enough for the existing GPU workflow.

```text
Fusion blocks:        2
Actor adapter blocks: 1
Reward adapter blocks:2
Cost adapter blocks:  2
Expansion ratio:      2x
Dropout:               existing configured value
Gate bias:             0.0 baseline
```

Only after this comparison may the following be considered:

- actor adapter block count 2;
- critic adapter block count 3;
- expansion ratio 3x;
- gate bias favoring the residual path slightly.

## 7. Diagnostics

Each GRN location must report compact rollout-level statistics:

- gate mean;
- gate standard deviation;
- fraction below 0.05;
- fraction above 0.95;
- per-block activation norm;
- residual-update norm divided by input norm;
- gradient norm;
- non-finite count;
- actor, reward-critic, and cost-critic values separately.

These diagnostics must not retain full per-step tensors in ordinary training artifacts.

A model is considered gate-saturated when a large majority of values remain near zero or one across multiple folds and seeds. Saturation is not automatically failure, but it requires explicit interpretation and comparison with the residual MLP.

## 8. Evaluation protocol

The comparison uses the same maintained walk-forward folds and seed set. At minimum, report:

- selection and sealed out-of-sample net log growth;
- total return and worst-seed return;
- maximum drawdown;
- turnover per day;
- execution-cost fraction;
- drawdown-stop and forced-liquidation rates;
- all configured constraint values;
- reward-critic explained variance;
- each cost-critic explained variance;
- PPO approximate KL and clip fraction;
- action-path policy-to-filled distortion;
- training throughput and peak memory;
- gate saturation diagnostics.

Nominal and required adverse execution scenarios remain mandatory.

## 9. Adoption gate

GRN becomes the default architecture only if it satisfies all of the following:

1. all ordinary and constrained PPO regression tests remain green;
2. no required constraint gate regresses;
3. sealed out-of-sample growth improves across the aggregate comparison rather than one isolated seed;
4. worst-seed performance does not materially deteriorate;
5. execution cost and turnover do not increase without compensating net growth;
6. reward or cost value estimation improves, or an alternative measurable mechanism explains the gain;
7. action projection distortion does not increase materially;
8. memory and throughput remain within the maintained training budget;
9. the improvement survives the required adverse execution scenario;
10. the result is reproduced from a clean checkpoint and recorded experiment identity.

A higher training score alone is insufficient.

## 10. Rejection conditions

The GRN variant is rejected or returned for redesign when:

- gains appear only in training or one seed;
- gates collapse to a trivial always-open or always-closed pattern without benefit over the residual MLP;
- Critic losses improve while sealed trading performance worsens;
- policy-to-filled action distortion rises substantially;
- constraint violations increase;
- checkpoint or serving architecture identity becomes ambiguous;
- training throughput or memory exceeds the maintained budget without robust out-of-sample gain.

## 11. Delivery order

GRN work begins only after the constrained PPO foundation is complete:

1. action and cost contracts;
2. constrained rollout and cost critics;
3. stabilized Lagrangian PPO;
4. constrained experiment profiles and evaluation;
5. GRN unit and architecture tests;
6. current MLP versus residual MLP versus GRN ablation;
7. default adoption only after the adoption gate passes.

The GRN implementation must use an independent PR so it can be reverted or rejected without affecting the constrained-growth objective.
