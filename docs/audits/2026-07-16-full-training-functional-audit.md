# Full Training Functional Audit — 2026-07-16

## Scope

This audit answers whether the reinforcement-learning system can execute its maintained learning paths to completion, whether enabled capabilities are consumed by the runtime rather than only recorded in configuration identity, and whether known causality, fold-isolation, selection, resume, export, and publication boundaries fail closed.

Profitability, strategy quality, direct exchange order routing, and live-account reconciliation are intentionally outside this audit.

## Verdict

The compact but real Python 3.12 training matrix completes across PPO, SAC, TD3, TQC, behavior cloning, checkpoint resume, replay-buffer resume, ONNX export, TorchScript export, residual alpha/factor/risk controls, and native 15m/1h/4h/1d structured sequence PPO.

The authoritative workflow matrix additionally exercises real nested walk-forward training, per-seed checkpoint selection, construction and selection evaluation of a fixed three-seed deployable ensemble, correct sealed outer-test evaluation of the selected mode, immutable ArtifactStore publication, signed serving-bundle activation, and deterministic inference. A short model that does not clear the selection gate correctly falls back to the baseline rather than bypassing the gate.

This proves functional completion and contract wiring. It does not prove convergence or completion of the maintained long-duration 524,288-timestep, multi-fold historical research recipe. A short audit run cannot be represented as a full-scale training run.

## Defects found and fixed

### 1. Silent actor/critic architecture no-ops

Before the audit, flat PPO consumed `policy_net_arch` for both actor and value paths, while SAC, TD3, and TQC consumed `policy_net_arch` for actor and critic paths. `value_net_arch` changed the configuration digest but did not change the constructed model.

The SB3 backend now binds:

- PPO: `pi=policy_net_arch`, `vf=value_net_arch`
- SAC/TD3/TQC: `pi=policy_net_arch`, `qf=value_net_arch`

Regression tests inspect the actual arguments passed to SB3 model constructors.

### 2. Algorithm-inactive fields were accepted

The common training configuration previously accepted non-default fields that the selected algorithm could not consume. The configuration identity could therefore change while the trained model remained unchanged.

The configuration now fails before training when non-default inactive fields are supplied:

- PPO rejects off-policy replay and update fields.
- SAC, TD3, and TQC reject PPO rollout, GAE, clipping, value-loss, and PPO-gradient fields.
- TD3 rejects SDE fields unsupported by SB3 TD3.
- A disabled sequence encoder rejects non-default sequence-only fields.
- A disabled asset-set encoder rejects non-default asset/global embedding fields.
- Disabled behavior cloning rejects non-default cloning-only fields.

Maintained JSON and GPU-smoke configuration builders were normalized to leave inactive fields at their canonical defaults.

### 3. Structured sequence PPO could not construct a model

The sequence backend passed actor-evidence metadata (`actor_head`, `actor_parameter_sharing`, and `actor_symbol_order`) into `SequenceAssetFeatureExtractor`, whose constructor does not accept those arguments. Real sequence training stopped during model construction.

Those values are now excluded from feature-extractor arguments and retained in `model-architecture.json`, where they belong. Real native-timeframe sequence PPO subsequently completed.

## Real training evidence

The capability harness uses real SB3/sb3-contrib/PyTorch models, real gradient updates, real checkpoints, and real serialization. The small timestep budgets are deliberately chosen to prove reachability and integrity rather than learning quality.

| Capability | Result | Runtime evidence |
|---|---:|---|
| PPO | PASS | 16 timesteps; distinct `[16, 8]` actor and `[24, 12]` value networks; two checkpoints |
| SAC | PASS | 16 timesteps; distinct actor/critic networks; SDE enabled; replay buffer persisted |
| TD3 | PASS | 16 timesteps; distinct actor/critic networks; SDE rejected/disabled; replay buffer persisted |
| TQC | PASS | 16 timesteps; distinct actor/critic networks; SDE enabled; replay buffer persisted |
| Behavior cloning → PPO | PASS | 111 teacher samples; MSE reduced before PPO rollout; PPO then completed |
| PPO checkpoint resume | PASS | interrupted checkpoint resumed to the requested total timestep count |
| SAC replay resume | PASS | model and replay state restored and training resumed to the requested total |
| ONNX export | PASS | exported model reloaded; maximum action error about `2.33e-9` |
| TorchScript export | PASS | exported model reloaded; maximum action error `0.0` |
| Residual controls | PASS | `fast_tilt`, `slow_tilt`, `risk_tilt`, `alpha_scale`, `factor_0`, and `factor_1` all bound to the action contract |
| Alpha artifact | PASS | dataset-bound artifact digest consumed by the environment and identity |
| Factor-basis artifact | PASS | factor artifact and ordered factor action dimensions consumed |
| Asset-set encoder | PASS | real PPO training with per-asset shared encoder and global feature input |
| Structured sequence encoder | PASS | real PPO training with 15m/1h/4h/1d native windows and `MultiTimeframeTCNEncoder`; about 1.78M parameters |
| Three-seed sequence workflow | PASS | real fold training for seeds 0/1/2, per-seed checkpoint selection, deployable-ensemble selection evaluation, and one sealed outer test of the selected residual ensemble or baseline fallback |
| Training → serving | PASS | real training publication, bundle validation, authenticated release activation, reload, and deterministic prediction |

The content-addressed capability report generated by the audit has schema `full_training_capability_audit_v1`. The retained workflow validates every record as `pass` before succeeding.

## Workflow completion checks

### Direct training workflow

`execute_training_run` is exercised through the repository's end-to-end research-to-serving test. The test verifies that:

1. A dataset artifact is loaded and identity-checked.
2. Training creates policy, checkpoints, normalizer/evidence, and ensemble manifests.
3. The staged run is atomically published.
4. A serving bundle is built from the published run.
5. A trusted signed release attestation is verified.
6. The runtime loads the policy and produces deterministic inference.

### Nested walk-forward workflow

The structured three-seed regression executes `execute_market_walk_forward` with real SB3 sequence models and verifies:

1. Train, checkpoint-validation, configuration-selection, purge, and sealed test ranges are distinct.
2. Fold-local sequence normalization is persisted.
3. Each of seeds 0, 1, and 2 produces checkpoint and selection evidence.
4. The deployable mean ensemble contains exactly one finalist for each configured seed and is evaluated during configuration selection.
5. The sealed outer test evaluates exactly the selected mode: the exact ensemble when eligible, otherwise the baseline fallback; candidate alternatives cannot repeatedly access the sealed range.
6. The completed run is published through `ArtifactStore`.

## Loophole and leakage matrix

The focused regression matrix covers the following rejection boundaries:

- Unknown, inactive, and unsupported training fields fail during configuration validation.
- PPO/off-policy/sequence/asset-set/BC mutually exclusive combinations cannot silently downgrade.
- Resume rejects mismatched dataset, environment, action, training, checkpoint, and replay identities.
- Signal artifacts reject changed dataset IDs, symbol order, feature identity, timestamps, or content digests.
- Sequence normalizers are fitted only on the declared training range and preserve source dataset/layout identities.
- Native-timeframe windows use availability, mask, and staleness evidence rather than future-filled values.
- Walk-forward purge and range boundaries are enforced.
- Outer-test access requires a finalized configuration selection and is sealed against repeated candidate access.
- Selected fold ensembles must agree with the final fixed seed recipe.
- Artifact publication remains atomic; incomplete runs do not advance the latest pointer.
- Export declarations are checked against produced files and numerical reload probes.
- Structured serving validates observation layout, dataset identity, normalizers, decision index, portfolio state, pending target, and policy/bundle identities.

## Feature compatibility rules

Not every feature can be enabled in one model, and treating mutually exclusive features as one giant configuration would itself be a loophole.

- PPO uses rollout/GAE/clipping/value-loss fields and may use behavior cloning.
- SAC, TD3, and TQC use replay-buffer/update fields.
- TD3 does not support SDE in the pinned SB3 path.
- Structured native-timeframe sequence training currently requires PPO and `MultiInputPolicy`.
- The asset-set flat encoder and structured sequence encoder are mutually exclusive alternatives.
- Behavior cloning currently requires PPO.
- Alpha and factor dimensions require their bound artifact providers.

Unsupported combinations now fail explicitly instead of selecting a fallback policy or ignoring fields.

## Repository-wide verification

The final branch must pass, on the same source tree:

- Ruff and format verification
- MyPy
- import architecture enforcement
- dead-code report
- recovery and structured-serving smoke
- complete pytest and branch coverage
- critical branch coverage
- CLI smoke
- Ubuntu compatibility
- Windows compatibility
- training image build and non-root runtime probe
- the real training capability matrix
- the targeted loophole, workflow, and serving matrix

## Remaining limitations

1. The audit uses short deterministic runs. It proves that optimization, serialization, resume, export, selection, publication, and serving paths execute; it does not prove that a long run converges.
2. The maintained full historical recipe—six folds, three candidates, three seeds, and 524,288 timesteps per training member—was not re-executed in this audit. Its compute completion remains a separate long-running verification.
3. GitHub-hosted audit runners are CPU-only. CUDA configuration, image packaging, and fail-closed device selection are tested, but this audit does not claim a fresh physical-GPU training completion.
4. Algorithm families and mutually exclusive encoder alternatives are validated in separate real runs, which is the only meaningful way to test them.

## Final interpretation

After the fixes, the repository has no known configuration field in the audited training groups that may be changed materially while being silently ignored. The compact real-training, nested walk-forward, publication, export, resume, and serving paths are executable to completion. Any claim about the maintained full-duration recipe must still be based on an actual full-duration run, not this functional audit.
