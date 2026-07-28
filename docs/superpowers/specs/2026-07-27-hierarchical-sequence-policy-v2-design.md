# Hierarchical Sequence Policy v2 Design

## Status

Approved direction for implementation on `agent/cross-timeframe-attention`.

This design replaces the maintained sequence-policy architecture. It is not an optional compatibility mode. The current sequence generation is research-only and remains `NO-GO`, so preserving obsolete checkpoints or the concat-only fusion path is not a production requirement.

## Goal

Build one causal, inspectable, production-oriented sequence policy that learns dependencies along three separate axes:

1. within each native timeframe,
2. across 15m, 1h, 4h, and 1d representations for one asset,
3. across assets for one decision.

The policy must remain compatible with Oracle BC, PPO, constrained PPO variants, compact rollout reconstruction, checkpoint identity checks, structured serving, and deterministic evidence generation.

## Non-goals

- No Transformer-XL or recurrent memory across independently sampled episodes.
- No full attention over every candle from every timeframe and asset.
- No live-trading authorization.
- No runtime fallback to the old concat fusion.
- No silent loading of checkpoints produced by the previous architecture.

## Compatibility policy

Backward compatibility is retained only when at least one of the following is true:

- a production-authorized model depends on it,
- an external API or stored artifact contract promises it,
- it is required to reproduce historical evidence.

The current sequence policy satisfies none of those conditions as an active production model. Historical artifacts remain immutable and readable as evidence, but they are not resumable as v2 training checkpoints.

The comparison baseline is the immutable pre-v2 commit, not a second architecture branch inside production code.

## Configuration cleanup

### Replace mutually interacting booleans

Remove:

- `sequence_encoder`
- `asset_set_encoder`

Add one closed enum:

- `observation_encoder = "flat_mlp"`
- `observation_encoder = "asset_set"`
- `observation_encoder = "hierarchical_sequence_v2"`

This prevents impossible or ambiguous combinations and makes the active observation contract explicit.

### Rename ambiguous sequence settings

Remove:

- `sequence_capacity`
- `sequence_attention_heads`
- `sequence_attention_layers`

Add:

- `sequence_tcn_capacity`
- `sequence_timeframe_attention_heads`
- `sequence_timeframe_attention_layers`
- `sequence_timeframe_ffn_multiplier`
- `sequence_timeframe_gate_bias`
- `sequence_asset_attention_heads`
- `sequence_asset_attention_layers`
- `sequence_asset_ffn_multiplier`
- `sequence_asset_gate_bias`

Retain:

- `sequence_d_model`
- `sequence_dropout`
- compile and transfer controls
- parameter and rollout-memory ceilings

`sequence_tcn_capacity` remains because it selects a measured resource profile, not an obsolete architecture. `standard` is the maintained full-training profile; `compact` is limited to smoke and constrained hardware verification.

### Schema version

Bump the run configuration to `training_run_config_v2`. V1 files fail closed with a migration message rather than being interpreted heuristically.

All new architecture fields participate in the training digest, architecture identity, checkpoint manifest, run manifest, selection evidence, and serving bundle identity.

## Model architecture

### Stage 1: native-clock causal encoders

Keep one causal residual TCN per timeframe. Each TCN uses only completed, point-in-time-available bars, left padding, per-timestep normalization, and a receptive field covering the complete declared window.

The final available causal state becomes one native timeframe latent. The TCN projection remains after causal timestep selection so unavailable trailing positions do not create unnecessary projection work.

### Stage 2: timeframe token construction

For each asset, construct five tokens:

- asset-context token,
- 15m token,
- 1h token,
- 4h token,
- 1d token.

Each timeframe latent is projected to `sequence_d_model` and augmented with:

- learned timeframe identity,
- deterministic log-duration features,
- available-timestep fraction,
- latest-valid-position fraction,
- per-channel availability fraction at the selected timestep,
- log-scaled selected-timestep staleness summaries.

Quality features are derived from explicit availability and staleness planes, not inferred from normalized market values.

A timeframe with no usable timestep is masked from keys and values. Its token is zeroed before and after the attention stack. The context token is always available, so an asset with no valid timeframe history still has a finite fallback representation.

### Stage 3: asset-context token

Encode current snapshot and asset state independently, then fuse them into `sequence_d_model`. This token queries the four timeframe tokens and represents the decision-specific state: current market snapshot, holdings, portfolio state, tradability, and execution context.

### Stage 4: gated cross-timeframe attention

Use a dedicated pre-norm gated transformer stack over the five tokens.

Maintained full profile:

- model width: `sequence_d_model` (336 in the full preset),
- two layers,
- eight heads,
- FFN multiplier 3,
- GELU activation,
- dropout at most 0.05,
- per-channel residual gates initialized with a negative bias.

Each attention and FFN branch is applied as:

`output = residual + sigmoid(gate) * branch`

Negative gate initialization preserves the proven TCN/context representation at initialization and lets PPO open the new branch gradually.

The contextualized asset-context token is the single output of timeframe fusion. The old concatenation MLP is removed.

### Stage 5: gated cross-asset attention

Apply the same generic gated transformer block to one token per active asset. Learned symbol embeddings preserve ordered action identity. Inactive assets are masked and zeroed. An all-inactive batch produces exact zero asset tokens and pooled context.

Using one gated implementation for both axes removes the standard-transformer/gated-transformer split and gives both PPO attention stages the same numerical and initialization contract.

### Stage 6: actor and critic

Preserve the shared per-asset actor and portfolio critic boundaries:

- one shared action head is reused for every asset,
- actor input contains the contextual asset token, pooled market token, global state, and active flag,
- critic consumes pooled market and global state,
- inactive action dimensions remain exactly zero.

Oracle BC and PPO must use the same feature extractor and actor head.

## Architecture identity

Introduce an immutable `SequenceArchitectureIdentity` payload containing:

- architecture schema name,
- ordered timeframes,
- input channels,
- window lengths,
- TCN widths and dilation schedules,
- timeframe latent dimensions,
- d_model,
- timeframe attention settings,
- asset attention settings,
- snapshot and asset-state widths,
- ordered symbols,
- action identity.

Its content digest is checked before training, checkpoint resume, evaluation, export, and serving activation.

No loader may infer missing v2 fields from model weights.

## Diagnostics

The fast training forward does not retain attention matrices. A bounded diagnostic probe runs at configured intervals and records:

- context-to-timeframe attention share per timeframe,
- attention entropy and maximum share,
- unavailable-token ratio,
- attention and FFN gate means,
- gate saturation fractions,
- gradient norms for native TCN, timeframe stack, and asset stack,
- finite-value and exact-zero mask invariants.

Attention is treated as a diagnostic signal, not as a causal explanation of trades.

## Structured export and serving

The flat-vector export path remains invalid for sequence policies. Add a structured actor wrapper with a canonical ordered tuple of tensors corresponding to the sequence observation schema.

Export verification corpus must include:

- all-valid observations,
- partially unavailable channels,
- one fully unavailable timeframe,
- inactive assets,
- all assets inactive,
- high but valid staleness,
- deterministic boundary actions.

TorchScript and ONNX records are published only after parity with eager PyTorch is verified within the configured tolerance. Unsupported formats are recorded explicitly and cannot be reported as verified.

Serving activation checks architecture digest, observation schema, ordered features, ordered symbols, windows, normalizer, and action identity before swapping policy state.

## Testing requirements

### Unit invariants

- Prefix causality for every native encoder.
- Oldest declared history can influence the selected latent.
- Masked timeframe mutation cannot change output.
- Masked timeframe inputs and parameters receive no data-dependent gradient.
- Partially available timesteps remain usable.
- All-timeframe-missing assets produce finite context-derived tokens.
- Inactive assets and all-inactive batches produce exact zeros.
- Every maintained timeframe receives gradient under valid input.
- Gate initialization keeps v2 close to its residual path without making branches permanently inactive.
- Eager and compiled forward parity.
- FP32 and BF16 finite-output and gradient tolerances.
- Parameter ceiling enforcement.

### Integration invariants

- Oracle BC and PPO resolve the same architecture identity.
- PPO, CostCriticPPO, and LagrangianPPO construct and train one update.
- Compact rollout reconstruction preserves structured inputs.
- Checkpoint save/resume is exact for v2 and rejects v1/mismatched identities.
- TensorBoard diagnostics do not alter policy outputs or gradients.
- Structured serving smoke produces the same deterministic action as the training policy.

### Full verification

- Ruff format and lint.
- MyPy.
- Import architecture checks.
- Full pytest and critical coverage.
- CPU structured BC-to-PPO smoke.
- RTX/Docker CUDA smoke with four environments.
- Peak GPU memory and steps-per-second evidence.

## Evaluation and promotion

Compare three immutable candidates under identical folds, seeds, teacher, PPO budget, and execution assumptions:

- A: pre-v2 maintained sequence architecture at its fixed commit,
- B: hierarchical sequence policy v2,
- C: parameter-matched non-attention control.

Promotion requires improvement that survives seed and fold aggregation, not the best seed. Record growth, baseline uplift, regret, drawdown, turnover, cost fraction, worst fold, seed dispersion, throughput, memory, attention collapse, and gate saturation.

Code completion does not imply model promotion. The production decision remains `NO-GO` until the existing walk-forward and confirmation gates authorize v2.

## Delivery decomposition

1. PR A: configuration v2, generic gated transformer, hierarchical sequence model, architecture identity, unit tests.
2. PR B: BC/PPO/constrained-PPO integration, checkpointing, compile/BF16, diagnostics, CPU and CUDA smokes.
3. PR C: structured actor export, canonical serving loader, parity corpus, bundle identity.
4. PR D: immutable A/B/C evaluation, walk-forward evidence, promotion decision.
