# Causal Alpha V4 Sample Surface Plan Amendment

## Status and precedence

This implementation-plan amendment is authored before any V4 fit or forecast result has been observed. It corrects only the Task 6 sample-storage boundary so Task 7 cannot double-count the nine Universal instrument descriptors. The approved V4 design already distinguishes the maintained 206 target-local market features from the nine public continuous instrument descriptors.

This amendment follows the base V4 plan and the beta/source amendments. It overrides only the Task 6 `CausalAlphaV4SymbolSamples` surface and the Task 7 feature-matrix assembly details below. Reward, risk, execution, labels, beta, model strengths, Gate thresholds, and evaluation semantics are unchanged.

## Problem

`CausalAlphaSymbolSamples.features` in the existing V3 path is not the 206-channel target-local market block alone. `build_causal_alpha_symbol_samples` appends `UNIVERSAL_INSTRUMENT_DESCRIPTOR_NAMES`, so the stored feature matrix is:

```text
206 target-local market features
+ 9 instrument descriptors
= 215 columns
```

The approved V4 design requires the residual/direction surface to consume these as distinct semantic blocks:

```text
existing_target_local_features
local_cross_market_context
global_market_context
instrument_descriptors
causal_beta
```

Keeping all 215 V3 columns under `target_local_features` and then adding descriptors again in Task 7 would duplicate descriptor information and violate the authored feature semantics.

## Corrected Task 6 sample contract

`CausalAlphaV4SymbolSamples` stores the V3 source matrix as two immutable blocks:

```python
target_local_feature_names: tuple[str, ...]
target_local_features: np.ndarray
target_local_available: np.ndarray

instrument_descriptor_names: tuple[str, ...]
instrument_descriptors: np.ndarray
instrument_descriptor_available: np.ndarray
```

The exact descriptor names are `UNIVERSAL_INSTRUMENT_DESCRIPTOR_NAMES` in their maintained order.

The Task 6 builder must require that the source `CausalAlphaSymbolSamples.feature_names` ends with that exact nine-name descriptor tuple. It then splits the source arrays at `len(feature_names) - 9` without recomputation or reordering.

The source identity remains bound through `source_sample_digest`. The new V4 sample digest additionally binds both ordered name tuples and both value/availability arrays.

Fail closed when:

- the descriptor suffix is missing, duplicated, reordered, or incomplete;
- the split leaves no target-local market feature;
- descriptor/value/availability widths drift;
- a caller attempts to infer descriptors by symbol identity rather than the persisted source matrix.

## Corrected Task 7 assembly

Residual and direction fit matrices concatenate each semantic block exactly once in this order:

```text
target_local_features
local_cross_market_context
global_market_context
instrument_descriptors
causal_beta
```

Availability is preserved blockwise and is concatenated in the same order. The market-proxy head remains global-context-only.

No descriptor column is copied into `target_local_features`, and Task 7 must expose/test the resulting ordered fit feature names.

## Added Test Oracle

Task 6 must prove:

```text
source feature names = (market_names..., *UNIVERSAL_INSTRUMENT_DESCRIPTOR_NAMES)
=> V4 target_local_feature_names == market_names
=> V4 instrument_descriptor_names == UNIVERSAL_INSTRUMENT_DESCRIPTOR_NAMES
=> concatenating the two V4 source blocks reconstructs the original V3 source matrix exactly
```

A source sample with a reordered or missing descriptor suffix must be rejected before any V4 fit.

Task 7 must prove that the shared residual/direction matrix contains each descriptor name exactly once.
