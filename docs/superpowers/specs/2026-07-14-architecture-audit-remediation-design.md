# Architecture Audit Remediation Design

## Status

Approved by the user through the instruction to fix every issue identified in the post-merge architecture audit. This specification preserves the public research workflow while removing the documented architectural inconsistencies.

## Goals

1. Make the baseline-underperformance reward use a true fixed 30-day window.
2. Connect alpha and factor artifacts to the authoritative CLI training and walk-forward paths.
3. Move Stable-Baselines3 checkpoint loading behind an integration-layer protocol.
4. Remove the ambiguous duplicate market-dataset publication API.
5. Split oversized orchestration responsibilities without breaking public imports.
6. Add intermediate checkpoint selection during training.
7. Align documentation and schema identities.
8. Reduce broad type suppressions and raise confidence in financially critical modules.

## Non-goals

- No claim of profitability or production authorization.
- No change to the public `trade-rl` command name.
- No removal of existing compatibility imports in this release.
- No new model framework beyond PPO, SAC, TD3 and TQC.

## Reward window

`baseline_window_hours` remains 720 hours and `baseline_tolerance` remains 0.015. The default minimum history becomes the complete 720-hour window. Tolerance is never scaled by partial history. Baseline-underperformance shaping is zero until the complete window is present.

Training and walk-forward environments must provide a causal pre-roll range sufficient to initialize the reward tracker. A reset that requests baseline shaping without enough pre-roll fails closed rather than silently shortening the objective. Explicit legacy configurations may request a shorter minimum history, but the default and maintained CLI configuration use the full window.

The reward tracker accepts seeded historical hybrid and shadow log returns during reset. Seed data is strictly prior to the first rewarded decision and is bounded to the configured window.

## Signal artifact adapters

Introduce immutable alpha and factor artifact manifests containing schema version, dataset identity, feature/range identity, dimensions, file digests and artifact digest. Loading validates all identities and arrays before constructing providers.

The CLI run configuration accepts optional alpha and factor artifact paths. `ActionSpec.alpha_enabled` requires an alpha artifact; `ActionSpec.n_factors > 0` requires a factor artifact with the exact factor names and count. The environment, training manifest, serving bundle and walk-forward folds bind the artifact digests.

Artifacts used in nested walk-forward must be fold-local or explicitly immutable causal artifacts whose fit range ends before the fold evaluation range. Violations fail closed.

## Framework integration boundary

`workflows` consumes a `PolicyCheckpointLoader` protocol and never imports Stable-Baselines3 or sb3-contrib. The concrete loader lives in `trade_rl.integrations` and supports PPO, SAC, TD3 and TQC. Existing serving policy loading remains separate but shares validated algorithm-name parsing where useful.

## Dataset artifact API

`trade_rl.data.artifact_codec` remains the only codec implementation.

The maintained public API becomes:

- `write_market_dataset_files(root, dataset) -> DatasetArtifactFiles`
- `publish_market_dataset_artifact(root, dataset) -> PublishedDatasetArtifact`
- `load_market_dataset_artifact(root) -> MarketDataset`

`DatasetArtifactFiles` exposes manifest path, arrays path and artifact digest. `PublishedDatasetArtifact` exposes root, manifest path, arrays path and artifact digest.

The old `write_market_dataset_artifact` names remain deprecated compatibility wrappers in both prior modules for one release. They retain their previous return types and emit `DeprecationWarning`.

## Module decomposition

Public classes remain import-compatible. Internal responsibilities move to focused modules:

- `trade_rl.rl.episode` handles episode ranges, pre-roll validation and reset-state sampling.
- `trade_rl.rl.transition` handles one environment transition and terminal classification helpers.
- `trade_rl.workflows.checkpoints` owns framework-neutral checkpoint records and selection.
- `trade_rl.workflows.walk_forward_evaluation` owns range-scoped evaluation helpers.

The initial split targets cohesive helpers rather than mechanically moving every method. `ResidualMarketEnv` remains the Gymnasium facade.

## Intermediate checkpoints

Training configuration adds `checkpoint_interval_steps` and `max_checkpoints`. A positive interval saves model checkpoints during a single learning call through an SB3 callback. Each checkpoint records algorithm, seed, requested timestep, observed timestep, model file digest and environment identity.

Checkpoint selection evaluates only the checkpoint-validation range. The selected checkpoint is copied or referenced as the authoritative member `policy.zip`. The sealed outer-test range is never consulted during checkpoint creation or selection.

A zero interval disables intermediate checkpoints for compatibility. The maintained CLI default enables them with a conservative interval derived from total timesteps and bounded by `max_checkpoints`.

## Documentation and schema consistency

README, architecture documentation, research status and example configuration must all name reward schema v4. The repository must contain no maintained documentation claiming reward schema v3.

## Typing and verification

Remove file-wide `# mypy: disable-error-code="index"` from financially critical modules when local narrowing can express the indexing safely. Where third-party array typing is insufficient, use narrow line-level ignores with an explanation.

Keep the global branch-coverage floor at 80 percent to avoid meaningless test inflation. Add a separate critical-module coverage check requiring at least 90 percent branch coverage for:

- `simulation/accounting.py`
- `risk/pretrade.py`
- `rl/rewards.py`
- `evaluation/gates.py`
- `artifacts/`
- `serving/`

`simulation/execution.py` receives targeted scenario tests and an explicit measured threshold that can begin below 90 only when recorded in configuration with a ratchet preventing regression.

## Error handling

All identity, range, schema, shape and digest mismatches raise `ValueError` before training or evaluation starts. Missing optional artifact files raise `FileNotFoundError`. Partial checkpoint writes use a staging path and never become selectable. Existing economic terminal transitions remain MDP events; malformed data remains an exception.

## Acceptance criteria

- Default reward shaping activates only with a complete 720-hour history and never scales tolerance by partial history.
- CLI training and walk-forward can run with validated alpha/factor artifacts and reject mismatches.
- `trade_rl.workflows` has no Stable-Baselines3 or sb3-contrib imports.
- One canonical dataset API is documented and compatibility wrappers warn.
- Intermediate checkpoints are produced, validated and selected on checkpoint-validation data only.
- Reward schema documentation is consistently v4.
- Broad index suppressions are reduced in modified critical modules.
- Ruff, formatting, Mypy, Import Linter, dead-code report, full branch-coverage tests, critical-module coverage and CLI smoke all pass.
- Production status remains `NO-GO`.
