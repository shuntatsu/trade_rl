# Action-Head Ablation RED Evidence

The branch intentionally contains contract tests before production implementation.

Expected failing contracts:

- `shared_target_v1` is rejected for `hierarchical_sequence_v2`.
- `SharedPerAssetActorCriticPolicy.action_stage_outputs` does not exist.
- `sb3_policy_identity_v4` and direct-head identity are not implemented.
- direct-head action-stage TensorBoard metrics are absent.
- `evaluate_direct_behavior_cloning_gates` does not exist.
- paired action-head training and walk-forward profiles do not exist.

These failures define the implementation boundary. The subsequent commits must make these tests pass without changing the external target-weight, reward, risk, execution, PPO, or live-routing contracts.