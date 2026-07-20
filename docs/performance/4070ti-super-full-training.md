# RTX 4070 Ti SUPER Full Training Contract

The maintained full research run trains only `oracle-bc-ppo-15m-target`. It intentionally preserves the full sequence-policy capacity rather than obtaining speed by shrinking the model or reducing PPO updates.

## Full model

- sequence capacity: `standard`
- sequence model width: `336`
- attention: 8 heads and 2 layers
- actor network: `[384, 256, 128]`
- value network: `[512, 384, 256]`
- maximum policy parameters: `12,000,000`
- PPO batch size: `128`
- PPO epochs per rollout: `10`
- ensemble seeds: `0`, `1`, and `2`

## Implementation-side speedups

Behavior cloning gathers all native-timeframe windows for a mini-batch through the precomputed `SequencePolicyPlane` instead of rebuilding each sample with a Python loop. The fold-scoped Stable-Baselines3 backend also caches the immutable oracle teacher rollout by dataset, train range, environment, action specification, and teacher configuration, so the three seeds reuse one supervised dataset.

These changes preserve causal sequence alignment and all dataset, normalizer, environment, action, and teacher-artifact identity checks.

## Verification

The focused Python 3.12 verification runs Ruff formatting and lint checks, Mypy on the changed production modules, and the example configuration, sequence normalization, structured teacher artifact, and Stable-Baselines3 integration test suites. The temporary source-transfer workflow is not part of the product branch.
