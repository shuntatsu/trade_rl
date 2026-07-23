# Environment Observation Contract Extraction Verification — 2026-07-23

## 1. Scope

This verification covers the behavior-preserving extraction of deterministic
observation-contract construction from `ResidualMarketEnv.__init__()` into the
typed `EnvironmentObservationContractBuilder`.

The extraction is limited to static construction and validation of:

- `ObservationBuilder` and `ObservationLayout`;
- flat and structured observation schemas and contract digests;
- normalizer and sequence-normalizer identity checks;
- structured sequence windows, policy plane, and layout metadata;
- Gymnasium observation and action spaces;
- the sequence-derived minimum start index.

Episode state, reward state, risk state, execution books, order books, provider
resolution, service wiring, and reset/step behavior remain owned by the environment
facade or their existing typed runtime services.

Production remains `NO-GO`. This change is an architecture-maintenance
remediation and does not establish profitability, exchange-equivalent execution,
operational authorization, or direct venue connectivity.

## 2. TDD evidence

The architecture and characterization tests were committed before production
implementation.

The clean RED commit was:

- commit: `b55b577748078f456af67860a1119f8233c8592e`;
- CI run: `29968955103`.

At that commit Ruff, formatting, Mypy, Import Linter, and the maintained smoke
checks passed. Test collection then failed only because
`trade_rl.rl.environment_observation_contract` did not yet exist. This established
that the new architecture contract was capable of detecting the missing ownership
boundary rather than merely confirming completed implementation.

## 3. Implemented boundary

`trade_rl/rl/environment_observation_contract.py` now owns:

- frozen, slotted `EnvironmentObservationContract`;
- `EnvironmentObservationContractBuilder`;
- existing flat-normalizer validation order and error strings;
- existing sequence-normalizer validation order and error strings;
- flat `OBSERVATION_SCHEMA` construction;
- structured `SEQUENCE_OBSERVATION_SCHEMA` construction;
- exact component dtypes, shapes, layout metadata, and digest payload;
- observation/action Gymnasium spaces;
- sequence minimum-index projection.

`ResidualMarketEnv.__init__()` performs one builder invocation and assigns the
returned contract fields. The constructor no longer directly references
`spaces.Box`, `spaces.Dict`, `SequenceWindowSpec`,
`build_sequence_policy_plane`, or `observation_passthrough_indices`.

The constructor source span is now 321 lines and remains protected by an
architecture limit of 360 lines. The extraction removed 219 lines of inline
observation construction from `trade_rl/rl/environment.py`.

## 4. Characterization coverage

The new tests cover:

- flat layout, schema, digest, action space, observation space, and minimum index;
- structured component spaces, dtypes, sequence policy plane, metadata, digest,
  and minimum index;
- normalizer size, dataset, schema, schema digest, action identity, alpha artifact,
  factor artifact, and passthrough failures;
- sequence-normalizer dataset and schema failures;
- sequence window length-type and feature-order failures;
- architecture ownership and constructor delegation.

The extracted module measured:

- 114 / 114 statements covered;
- 32 / 32 branches covered;
- 100.0% statement and branch coverage.

A permanent 100.0% critical branch-coverage ratchet is recorded in
`pyproject.toml` for the module.

## 5. Exact-head verification

The implementation and coverage-ratchet exact head was:

- commit: `507e5882b45a4b6947f5b198d3d3f1be111a4da2`;
- CI run: `29970680856`;
- PostgreSQL Catalog run: `29970680786`.

The CI run passed:

- Studio frontend and fixed-viewport verification;
- workflow security checks;
- Ruff and Ruff format check;
- Mypy across 189 source files;
- Import Linter;
- dead-code reporting;
- recovery and structured-serving smoke;
- complete pytest and coverage;
- critical branch-coverage ratchets;
- CLI smoke;
- Ubuntu compatibility;
- Windows compatibility;
- complete training-image build and non-root runtime probe.

The complete test result was:

- 1,253 passed;
- 2 skipped;
- 11 warnings;
- 83.90% total coverage;
- 70.90% total branch coverage.

The PostgreSQL Catalog run passed Compose validation, PostgreSQL startup and
readiness, migrations, catalog/unit/integration tests, and cleanup on the same
exact head.

## 6. Architecture disposition

`AUD-RL-001` remains a maintainability risk rather than a reproduced behavioral
defect.

Current status: **OPEN RISK, FURTHER REDUCED**.

The deterministic observation-contract seam is now typed, independently tested,
fully branch covered, and protected against returning to the environment
constructor. The remaining constructor responsibilities are provider validation,
identity binding, runtime-service wiring, and mutable Gymnasium-state
initialization. Those concerns should not be mechanically split without another
behavior-preserving seam and characterization evidence.

This item does not block causal research use. It remains an architecture
maintenance watchpoint, and production status remains `NO-GO` until all research,
evidence, operational, authorization, and profitability gates pass independently.
