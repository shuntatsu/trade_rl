# Environment Observation Contract Extraction Design

Date: 2026-07-23

## Problem

`ResidualMarketEnv.step()` is already a typed orchestration facade, but `ResidualMarketEnv.__init__()` still mixes several independent construction responsibilities. The largest behavior-preserving seam is observation-contract assembly:

- current-observation builder and layout creation;
- flat normalizer identity/schema/action/artifact validation;
- observation mask/activity passthrough validation;
- structured sequence-window builder creation;
- sequence normalizer identity/schema validation;
- sequence policy-plane construction;
- sequence minimum-index contribution;
- flat or structured Gymnasium observation-space construction;
- action-space construction;
- observation schema and canonical contract digest construction;
- structured layout metadata construction.

This code does not mutate episode books, order state, reward history, or step-time policy. Keeping it inline makes the environment constructor difficult to review and encourages future observation-policy additions to accumulate in the mutable environment facade.

This is a targeted reduction of the open `AUD-RL-001` construction-density risk. It does not claim a reproduced behavior defect or fully close the remaining facade risk.

## Goal

Extract observation-contract construction into a typed, deterministic builder while preserving all public environment fields, validation messages, schemas, digests, spaces, minimum start index, and runtime behavior.

## Chosen boundary

Create `trade_rl.rl.environment_observation_contract` with:

```python
@dataclass(frozen=True, slots=True)
class EnvironmentObservationContract:
    observation_builder: ObservationBuilder
    layout: ObservationLayout
    asset_active_column: int
    sequence_observation_builder: SequenceObservationBuilder | None
    sequence_policy_plane: SequencePolicyPlane | None
    sequence_layout_metadata: dict[str, object] | None
    observation_schema: str
    observation_contract_digest: str
    observation_space: spaces.Space[np.ndarray | dict[str, np.ndarray]]
    action_space: spaces.Box[np.ndarray]
    minimum_start_index: int


class EnvironmentObservationContractBuilder:
    def __init__(
        self,
        dataset: MarketDataset,
        config: ResidualMarketEnvConfig,
        *,
        action_spec: ActionSpec,
        normalizer: ObservationNormalizer | None,
        sequence_normalizer: SequenceFeatureNormalizer | None,
        alpha_artifact_digest: str | None,
        factor_artifact_digest: str | None,
        action_spec_digest: str,
    ) -> None: ...

    def build(self, *, minimum_start_index: int) -> EnvironmentObservationContract: ...
```

The exact `ObservationLayout` type imported from the maintained observation module must be used; no untyped dictionary substitutes the layout.

## Ownership and data flow

`ResidualMarketEnv.__init__()` retains ownership of:

- dataset and configuration;
- provider resolution and artifact identity;
- action specification;
- reward and execution objects;
- episode/runtime services;
- mutable episode state.

The builder owns only static observation-contract assembly. The environment calls it once, then assigns the returned fields without recomputation.

## Behavioral invariants

### Flat observation mode

The builder must preserve:

- `ObservationBuilder(action_size, n_factors, finite_horizon)` arguments;
- exact layout and `asset_active_column = 4 * dataset.n_features`;
- normalizer size, dataset identity, observation schema, schema digest, action identity, alpha/factor artifact identity, and passthrough-index validation;
- `OBSERVATION_SCHEMA`;
- `ObservationBuilder.schema_digest(dataset)` as the contract digest;
- flat `spaces.Box` bounds, shape, and `float32` dtype.

### Structured sequence mode

The builder must preserve:

- the configured ordered `SequenceWindowSpec` values;
- sequence normalizer dataset and layout-digest validation;
- `build_sequence_policy_plane()` inputs and output;
- maximum of caller minimum and sequence minimum index;
- exact Dict component names, shapes, bounds, and dtypes;
- strict integer window length and ordered feature-name validation messages;
- `sequence_layout_metadata` field names and values;
- `SEQUENCE_OBSERVATION_SCHEMA`;
- canonical `content_digest` payload and sorted component dtype mapping.

### Action space

The builder must preserve the `[-1, 1]`, `(action_spec.size,)`, `float32` action space for both modes.

### Environment public fields

The environment must continue exposing these names with identical values/types:

- `observation_builder`;
- `layout`;
- `asset_active_column`;
- `sequence_observation_builder`;
- `sequence_policy_plane`;
- `sequence_layout_metadata`;
- `_observation_schema`;
- `_observation_contract_digest`;
- `observation_space`;
- `action_space`;
- `_minimum_start_index`.

## Validation and error invariants

Every existing message from the extracted block remains exact, including:

- `normalizer size does not match observation layout`;
- normalizer dataset/schema/schema-digest/action/artifact identity messages;
- passthrough mask/activity preservation message;
- sequence normalizer dataset/schema messages;
- `sequence window length must be an integer`;
- `sequence feature names must be ordered`.

The builder does not catch or translate errors from maintained observation/sequence helpers.

## Architecture constraints

- `ResidualMarketEnv.__init__()` must delegate to `EnvironmentObservationContractBuilder`.
- The constructor must not directly call `spaces.Box`, `spaces.Dict`, `SequenceWindowSpec`, `build_sequence_policy_plane`, or `observation_passthrough_indices`.
- The constructor source span must be reduced to at most 360 lines.
- No observation contract logic is duplicated in `environment.py`.
- The builder may depend only on data/config/action/normalization/observation modules and Gymnasium spaces; it may not depend on the environment facade, execution, rewards, risk, or mutable book state.
- Existing Import Linter boundaries must remain unchanged.

## Testing strategy

1. Add an architecture RED contract requiring the module, typed result, delegation, source-span bound, and absence of low-level observation construction in the environment constructor.
2. Capture flat and structured environment contract payloads before extraction: schemas, digests, layouts, spaces, metadata, minimum index, and action space.
3. Add direct builder tests for flat/structured construction and all identity/error boundaries.
4. Run the existing environment, observation, sequence, rollout-memory, SB3 integration, Serving parity, and full repository suites.
5. Add a measured branch-coverage threshold supported by exact-head results; do not lower existing thresholds.

## Non-goals

- No change to observations, features, sequence windows, dtypes, masks, normalization, or policy inputs.
- No provider-resolution extraction in this PR.
- No action, reward, risk, execution, episode, or step-policy change.
- No public constructor signature change.
- No claim that `AUD-RL-001` is fully closed.
- No production-readiness or direct exchange capability change.

Production remains `NO-GO`.
