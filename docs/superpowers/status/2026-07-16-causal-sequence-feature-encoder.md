# Causal Sequence Feature Encoder Status

Branch: `agent/causal-sequence-feature-encoder`

Production status: `NO-GO`

## Implemented

- Structured causal observations keep native 15m, 1h, 4h, and 1d sequences separate, with maintained windows of 96, 168, 120, and 60 completed bars.
- Native observations are selected by point-in-time availability. Incomplete higher-timeframe bars, backward fill, centered windows, and future-shifted Ichimoku values are rejected.
- Each sequence carries values, feature availability, and feature staleness. A timestep remains usable when at least one channel is available; individual unavailable channels stay masked instead of invalidating the entire timestep.
- Timeframe-specific residual causal TCNs use left padding and per-timestep LayerNorm. Prefix tests prove that changing later timesteps cannot change earlier representations.
- Per-asset encodings are fused by two layers of eight-head cross-asset attention. PPO and Oracle BC use the same `MultiInputPolicy` feature extractor.
- The maintained network uses `d_model=336`, actor layers 384/256/128, critic layers 512/384/256, and approximately 6,077,043 parameters. A hard 12M parameter ceiling is checked before training continues.
- The maintained point-in-time feature contract contains 226 ordered channels:
  - 15m: 59
  - 1h: 59
  - 4h: 55
  - 1d: 53
- The feature set includes candle geometry, range-based volatility, directional trend, volume and money-flow signals, funding transformations, and explicit causal cross-asset features: BTC-relative return, rolling BTC correlation, rolling BTC beta, cross-sectional momentum rank, and cross-asset dispersion.
- Cross-asset features retain the age of delayed native source events. Carried values become unavailable after their configured staleness limit.
- The per-decision 2% target slicing throttle is optional and disabled in the maintained direct target-weight preset. Gross exposure, concentration, liquidity, margin, tradability, and emergency deleveraging constraints remain.
- Walk-forward training views preserve both the longest sequence pre-roll and reward/baseline pre-roll. Normalizer fitting uses only the chronological fold train range and fits the flat current snapshot contract rather than attempting to coerce Dict observations into an array.
- Structured teacher artifacts are compact. They store decision indices and non-overlapping current state, then reconstruct sequence mini-batches from the immutable dataset. Overlapping 60-day sequences are not duplicated for every teacher sample or moved to the GPU as one tensor.
- BC validation is a chronological tail contained entirely inside the fold train range. Oracle construction, normalization, checkpoint selection, and feature filtering remain train-only.
- Exact observation keys, shapes, dtypes, ordered features, symbols, windows, action identity, environment identity, and architecture settings are content-addressed and fail closed on mismatch.
- PPO rollout memory is estimated before vector workers or the model are allocated. The maintained full configuration uses four environments, `n_steps=128`, batch size 128, and an estimated 473,122,816-byte Dict rollout buffer, below the 768 MiB limit.
- Docker and the maintained subprocess smoke set bounded OpenMP/MKL thread counts to avoid CPU thread oversubscription during preflight and fallback verification.

## Verification evidence

Local verification completed on the current source:

- Ruff format and lint: passed.
- MyPy: passed across 108 source files.
- Import architecture: 5 contracts kept, 0 broken.
- Vulture at 100% confidence: passed.
- Full non-Docker suite: 726 passed, 1 skipped.
- Overall branch coverage: 84.32%, above the configured 80% requirement.
- Critical branch coverage ratchet: all required modules/groups passed.
- CLI smoke and Python compileall: passed.
- `git diff --check`: passed.
- Full native-window CPU Oracle BC→PPO smoke: passed with 226 features, approximately 6.08M parameters, compact teacher arrays of about 15 KiB, and an 8-step PPO publish.

The connected execution environment does not expose an NVIDIA GPU or Docker daemon, so a real CUDA allocation measurement cannot be honestly claimed here. The maintained CUDA smoke now exercises the same structured sequence policy, feature contract, Oracle BC path, and four-environment PPO path and must pass on the user's RTX/Docker host before a full retraining generation is accepted.

## Research boundary

The opened July 2026 range remains development validation only. This implementation does not convert any previous result into confirmatory evidence and does not authorize live exchange trading. A new unused period is required for the next official model decision. Production status remains `NO-GO` regardless of code verification.
