# Causal Sequence Feature Encoder Status

Implementation commit: `b63249338d6151d0587a466ac36f52fd61a815cf`

Implemented and focused-verified:

- structured causal 15m/1h/4h/1d sequence observations with windows 96/168/120/60;
- completed-bar-only source indexing, availability and staleness tensors, ordered schema digests, and prefix-causality tests;
- causal residual TCN encoders with per-timestep LayerNorm, which prevents future timesteps from influencing past representations;
- structured SB3 `MultiInputPolicy` integration and shared sequence encoder support for pure PPO and Oracle BC→PPO;
- structured teacher artifacts with per-key shape, dtype and digest validation;
- chronological BC validation and early stopping inside the training range;
- optional per-decision turnover throttling, disabled in the maintained direct target-weight preset while hard portfolio and execution constraints remain;
- expanded trailing point-in-time indicators and native-clock presets: 15m=54, 1h=54, 4h=50, 1d=48, for 206 ordered current features;
- maintained PPO configuration updated to 4 environments, 1,024 rollout steps, batch size 128, lower learning rate, separate actor/value heads, and a 12M parameter hard ceiling.

Focused verification passed on GitHub Actions:

- Ruff format and lint;
- causal sequence, TCN, environment configuration, expanded indicator, structured teacher, behavior-cloning and pre-trade tests;
- MyPy across `trade_rl`;
- Python compileall;
- `git diff --check`.

The July range remains development validation only. Production status remains `NO-GO`. Full repository CI, complete walk-forward integration, Docker/CUDA smoke, rollout-memory measurement and fresh unused-period evaluation remain required before the implementation can be considered research-ready.
