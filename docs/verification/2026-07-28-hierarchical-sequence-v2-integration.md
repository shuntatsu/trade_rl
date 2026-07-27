# Hierarchical Sequence Policy v2 integration verification

Date: 2026-07-28

## Integrated contract

- `observation_encoder` is the single encoder selection contract.
- `training_run_config_v2` is required; legacy v1 input is rejected.
- Timeframe and cross-asset attention settings are independently named and identity-bound.
- The structured policy uses causal native-clock TCN encoders, quality-aware gated cross-timeframe attention, and gated cross-asset attention.
- The actual assembled sequence architecture, symbol order, and action order are bound into SB3 policy identity.
- Checkpoint identity composes policy architecture identity with algorithm-specific CostCriticPPO or LagrangianPPO identity.
- Checkpoint loading rebinds the sequence reconstruction runtime and rejects identity mismatch.

## Verification before this evidence commit

The policy-identity finalizer completed all of the following before publishing commit `8dd768baa8485604746aad642ff4c384fb4b1f22`:

- Ruff format and lint
- MyPy over the repository
- `tests/integrations/test_sb3_policy_identity_v2.py`
- `tests/integrations/test_sb3_checkpoint_assembly.py`
- `tests/integrations/test_sb3_model_assembly.py`
- `tests/integrations/test_sb3_model_construction.py`
- `tests/rl/test_cost_checkpoint_identity.py`
- `tests/rl/test_lagrangian_checkpoint_identity.py`

The current PR head must additionally pass the repository CI, sequence-projection stability, and PostgreSQL catalog workflows before merge.
