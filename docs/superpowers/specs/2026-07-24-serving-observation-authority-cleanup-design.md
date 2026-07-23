# Serving Observation Authority Cleanup Design

Date: 2026-07-24

## Goal

Remove the unused `trade_rl.serving.observations` implementation so Serving has one maintained observation-validation and normalization path.

## Current authority

The maintained path is:

1. `load_serving_bundle()` loads and verifies the normalizer sidecar and digest;
2. `ServingRuntime._predict_action()` validates the active observation schema, shape, finiteness, and applies the loaded normalizer for flat observations;
3. structured observations are validated and passed to the structured policy without the flat normalizer.

`ServingObservationPipeline` duplicates part of that behavior, is not exported by `trade_rl.serving`, has no runtime callers, and has zero test coverage. Keeping it creates a second apparent authority that can drift from the actual runtime.

## Change

Delete `trade_rl/serving/observations.py`. Add an architecture contract that requires:

- the obsolete module to remain absent;
- no production source to import `trade_rl.serving.observations` or name `ServingObservationPipeline`;
- `trade_rl.serving.__init__` not to expose the obsolete symbols;
- the bundle loader to remain responsible for normalizer sidecar loading and digest validation;
- the runtime to remain responsible for observation validation and normalization.

Historical plans that mention the old component remain untouched as historical records.

## Compatibility

No public exported symbol is removed because the module was never exported by `trade_rl.serving.__init__`. Repository code search found no caller outside the obsolete module and a historical implementation plan.

## Safety

No bundle schema, runtime identity, inference output, normalizer artifact, or release gate changes. Production remains `NO-GO`.