# RL Observation v2 Design

Date: 2026-09-12 JST
Status: Active
Issue: #489

## Conclusion

Before the first canonical real-data M2 comparison, freeze one small teacher-free PPO observation contract that closes obvious state-observability gaps without reopening the old large RL platform.

Observation v2 is fixed, not a user-tunable factor:

```text
local selected values
+ local selected availability
+ local selected normalized staleness
+ fixed ordered global regime values
+ fixed ordered global regime availability
+ current intent
+ current weight
```

The fixed global roster is:

```text
active_fraction
tradable_fraction
market_return_mean
market_return_dispersion
```

PPO remains a symbol-ID-free MLP with the same three discrete actions, reward, risk, execution/accounting path, training budget and seed policy.

## Objective

Make the first M2 PPO baseline a meaningful test of teacher-free RL rather than a test of an avoidably partial observation, while preserving anti-fishing and fair-comparison constraints.

## Non-goals

This change does not add or change:

- symbol/ticker/dataset/source identity channels;
- absolute-price identity channels;
- fee/spread/participation/borrow/funding policy inputs;
- history windows, stacking, LSTM, Transformer, recurrent PPO or memory;
- PPO architecture, optimizer, reward, action mapping, risk, execution or accounting;
- Ridge/LightGBM inputs;
- Development-derived feature selection/scaling;
- real-data M2 execution itself.

Execution economics remain a separate shared-environment responsibility under #481/#484.

## Contract ownership

`trade_rl.strategies.rl` owns the PPO observation contract.

The contract is fixed code, not candidate JSON configuration, so M2 cannot tune the global roster after observing results.

The owner exposes an immutable semantic contract containing at minimum:

- observation schema identifier;
- ordered fixed global feature names;
- explicit declaration that normalized local feature staleness is included;
- deterministic tensor ordering.

Run/Study persisted evidence records this contract explicitly. Old persisted v1 Run/Study readers remain supported and are not rewritten.

## Observation layout

For selected local feature indices `F` and fixed global indices `G` resolved by name:

```text
[
  local_values[F],
  local_available[F],
  local_staleness[F],
  global_values[G],
  global_available[G],
  current_intent,
  current_weight,
]
```

Observation size is:

```text
3 * len(F) + 2 * len(G) + 2
```

For the fixed four-channel global roster this is `3 * len(F) + 10`.

### Local values

Keep current semantics:

- selected local feature values only;
- value is emitted only when the feature is available and finite;
- otherwise policy value is exactly `0.0`;
- an independent availability mask is emitted.

### Local staleness

Use `MarketDataset.feature_staleness`, not raw age hours.

`MarketDataset` already enforces this channel as finite `[0, 1]` and requires unavailable features to have maximum staleness. The observation uses the selected indices in the same order as local values and availability.

For unavailable features, staleness remains `1.0`; raw unavailable feature values are still masked to zero.

### Global regime

Resolve the fixed names against `MarketDataset.global_feature_names` by exact name, preserving the contract order rather than dataset storage order.

All fixed names must exist exactly once. Missing, duplicate or otherwise unresolved global contract state fails closed before PPO fit/evaluation.

Global values are emitted only when the corresponding global availability flag is true and the value is finite; otherwise the value is `0.0`. Availability is emitted independently.

No symbol identifier is encoded.

## StrategyObservation boundary

`StrategyObservation` gains the normalized local `feature_staleness` vector because both training (`PPOTradingEnv`) and inference/replay (`PPOIntentStrategy` through canonical replay) must consume the same observation semantics.

Both maintained producers populate it from `dataset.resolved_array("feature_staleness")` at the same bar/symbol as `features` and `feature_available`.

The field is read-only, one-dimensional and shape-matched to `features`.

Generic rule/forecast strategies may ignore it; their decision semantics do not change.

## Identity and compatibility

The fixed observation contract must be explicit in new evidence without turning it into a tuning parameter.

### Candidate Run

New candidate-run summaries use a new result schema and include a `ppo_observation` object containing the exact contract schema and fixed ordered global roster. The semantic candidate artifact digest therefore binds the observation contract.

The loader continues accepting historical `lean_candidate_result_v1` artifacts. New writes use v2. Historical artifacts are never rewritten.

### Controlled Experiment / Study

New `ResolvedRunConfig` uses a new schema version and carries the exact fixed PPO observation schema/roster. `StudyPlan.to_payload()` already embeds the resolved baseline config, so the Study digest transitively and explicitly binds Observation v2.

The experiment codec continues decoding historical `resolved_run_config_v1` payloads and writes/round-trips them without adding v2 fields. New candidate specs project to v2.

No old Study artifact is mutated.

## Causality and fairness invariants

- symbol names/indices never enter the PPO tensor;
- future rows never influence an earlier observation;
- selected local market features remain the same precommitted M2 feature set used by forecast candidates;
- staleness describes the quality/age of those already-selected values rather than adding a new alpha source;
- global regime channels are symbol-independent, causal dataset channels fixed before Development results;
- one frozen policy remains applied independently to each evaluation symbol;
- fit-symbol scope remains explicit and round-robin;
- M2 does not change the roster after seeing PPO performance.

## Failure modes

The implementation must reject or detect:

- local staleness index/order mismatch;
- non-finite or out-of-range staleness reaching the tensor;
- unavailable raw value leakage;
- global feature storage order controlling tensor order;
- missing/duplicate global contract names being silently dropped;
- symbol identity entering tensor content;
- fit/evaluation tensor shape mismatch;
- training and replay using different observation encoders;
- result/Study identity omitting the observation contract;
- historical v1 artifact reader breakage;
- execution-cost channels or recurrent-model scope creeping into this change.

## Test oracle

Required observable evidence:

1. an exact hand-auditable encoded vector proving component order and shape;
2. local masking and selected staleness alignment;
3. fixed global name-order resolution independent of dataset storage order;
4. unknown/missing/duplicate global contract failure;
5. symbol-rename metamorphic invariance;
6. future-mutation metamorphic invariance for an earlier encoded observation;
7. round-robin explicit fit-symbol behavior unchanged;
8. reward/action/accounting parity regression unchanged;
9. new candidate artifact explicitly records Observation v2 and historical v1 artifact still loads;
10. new `ResolvedRunConfig` / `StudyPlan` payload explicitly binds Observation v2 and historical v1 codec still round-trips.

## Acceptance criteria

The design is complete when all Issue #489 acceptance criteria are satisfied and no execution-aware or sequence-model behavior appears in the final diff.

## Follow-up boundary

Only after one canonical real-data M2 comparison under this frozen observation may separate preregistered experiments consider:

- execution-aware policy state;
- short causal history/frame stacking;
- recurrent/sequence policies;
- additional global regime channels.
