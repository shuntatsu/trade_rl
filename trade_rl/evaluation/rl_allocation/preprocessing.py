"""Fit declared observable prefix rows without constructing an allocation env."""

from __future__ import annotations

import numpy as np

from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.data.market import MarketDataset
from trade_rl.strategies.dataset_scope import validated_training_scope
from trade_rl.strategies.forecasts.training_trace import _timestamp
from trade_rl.strategies.rl.allocation_preprocessing import (
    AllocationFeaturePreprocessing,
)
from trade_rl.strategies.rl.allocation_preprocessing_receipt import (
    preprocessing_fit_payload,
)
from trade_rl.strategies.rl.ppo_normalization import PPOFeatureNormalizer


def fit_allocation_feature_preprocessing(
    dataset: MarketDataset,
    *,
    feature_indices: tuple[int, ...],
    fit_symbol_indices: tuple[int, ...],
    fit_start: int,
    fit_stop: int,
    fit_as_of: np.datetime64,
    first_decision_index: int,
) -> AllocationFeaturePreprocessing:
    """Freeze equal-symbol statistics from jointly available earlier feature rows.

    Fit consumption binds masks/clocks across the selected prefix, but values only
    on admitted rows. Dataset lineage is separate; excluded magnitudes and future
    cells are not statistical inputs. This is no research or training authority.
    """
    if fit_symbol_indices is None:
        raise ValueError("fit symbol scope must be explicitly declared")
    indices, symbols = validated_training_scope(
        dataset, feature_indices=feature_indices, fit_symbol_indices=fit_symbol_indices
    )
    if (
        any(type(i) is not int for i in (fit_start, fit_stop, first_decision_index))
        or not 0 <= fit_start < fit_stop <= first_decision_index < dataset.n_bars
    ):
        raise ValueError("fit prefix must end before a valid policy decision")
    as_of = _timestamp(fit_as_of, field="fit_as_of")
    times = np.array(
        [
            _timestamp(t, field="prefix event")
            for t in dataset.timestamps[fit_start:fit_stop]
        ]
    )
    first = _timestamp(
        dataset.timestamps[first_decision_index], field="first policy decision"
    )
    if not times[-1] < as_of <= first:
        raise ValueError("prefix events must precede fit and fit precede policy")
    source = dataset.resolved_array("available_at")[fit_start:fit_stop, list(symbols)]
    source = np.array(
        [[_timestamp(t, field="prefix publication") for t in row] for row in source]
    )
    selected = np.asarray(
        dataset.features[fit_start:fit_stop][:, list(symbols)][:, :, list(indices)],
        dtype=np.float64,
    )
    available = dataset.feature_available[fit_start:fit_stop][:, list(symbols)][
        :, :, list(indices)
    ]
    finite = np.isfinite(selected)
    admitted = np.all(available & finite, axis=2) & (source <= times[:, None])
    per_symbol = [selected[admitted[:, row], row] for row in range(len(symbols))]
    if any(not len(values) for values in per_symbol):
        raise ValueError("every fit symbol needs jointly observable prefix values")
    rows = tuple(
        tuple(int(i) + fit_start for i in np.flatnonzero(admitted[:, row]))
        for row in range(len(symbols))
    )
    with np.errstate(over="raise", invalid="raise"):
        mean = np.mean([np.mean(values, axis=0) for values in per_symbol], axis=0)
        variance = np.mean(
            [np.mean((values - mean) ** 2, axis=0) for values in per_symbol], axis=0
        )
        scale = np.sqrt(variance)
    names = tuple(dataset.feature_names[i] for i in indices)
    normalizer = PPOFeatureNormalizer(
        source_dataset_id=dataset.dataset_id,
        feature_indices=indices,
        feature_names=names,
        fit_symbol_indices=symbols,
        start_index=fit_start,
        stop_index=fit_stop,
        mean=tuple(float(value) for value in mean),
        scale=tuple(float(value) if value > 1e-12 else 1.0 for value in scale),
        usable_counts=tuple((len(values),) * len(indices) for values in per_symbol),
    )
    consumption = {
        "schema": "allocation_observable_prefix_consumption_v1",
        "feature_config_digest": dataset.feature_config_digest,
        "source_normalization_digest": dataset.normalization_digest,
        "feature_indices": list(indices),
        "feature_names": list(names),
        "symbol_indices": list(symbols),
        "symbols": [dataset.symbols[i] for i in symbols],
        "fit_start": fit_start,
        "fit_stop": fit_stop,
        "timestamps_ns": times.astype(np.int64).tolist(),
        "source_available_ns": source.astype(np.int64).tolist(),
        "feature_available": available.tolist(),
        "finite": finite.tolist(),
        "admitted_rows": [list(row) for row in rows],
        "admitted_values": [values.tolist() for values in per_symbol],
    }
    return AllocationFeaturePreprocessing(
        normalizer=normalizer,
        feature_config_digest=dataset.feature_config_digest,
        source_normalization_digest=dataset.normalization_digest,
        fit_last_event_time_ns=int(times[-1].astype(np.int64)),
        fit_as_of_ns=int(as_of.astype(np.int64)),
        policy_start_index=first_decision_index,
        policy_start_time_ns=int(first.astype(np.int64)),
        admitted_row_indices=rows,
        fit_consumption_digest=content_digest(consumption),
    )


def validate_preprocessing_application(
    dataset: MarketDataset,
    declaration: AllocationFeaturePreprocessing,
    *,
    feature_indices: tuple[int, ...],
    decision_time: np.datetime64,
) -> None:
    """Actual inference semantics/time; fit Dataset ID and indices are provenance."""
    if type(declaration) is not AllocationFeaturePreprocessing or any(
        type(i) is not int or not 0 <= i < dataset.n_features for i in feature_indices
    ):
        raise ValueError("application requires a frozen preprocessing declaration")
    stamp = int(
        _timestamp(decision_time, field="application decision").astype(np.int64)
    )
    if (
        tuple(dataset.feature_names[i] for i in feature_indices)
        != declaration.normalizer.feature_names
        or dataset.feature_config_digest != declaration.feature_config_digest
        or dataset.normalization_digest != declaration.source_normalization_digest
        or stamp < max(declaration.fit_as_of_ns, declaration.policy_start_time_ns)
    ):
        raise ValueError("actual preprocessing application features/build/clock differ")


def validate_training_preprocessing(
    dataset: MarketDataset,
    declaration: AllocationFeaturePreprocessing,
    *,
    feature_indices: tuple[int, ...],
    symbol_index: int,
    first_decision_index: int,
) -> dict[str, object]:
    """Reconstruct the supplied frozen prefix; never replace its coefficients."""
    if (
        type(first_decision_index) is not int
        or not 0 <= first_decision_index < dataset.n_bars
    ):
        raise ValueError("preprocessing requires a valid first policy index")
    validate_preprocessing_application(
        dataset,
        declaration,
        feature_indices=feature_indices,
        decision_time=dataset.timestamps[first_decision_index],
    )
    fit = declaration.normalizer
    if (
        dataset.dataset_id != fit.source_dataset_id
        or feature_indices != fit.feature_indices
        or type(symbol_index) is not int
        or symbol_index not in fit.fit_symbol_indices
        or first_decision_index != declaration.policy_start_index
    ):
        raise ValueError("actual training preprocessing prefix/source scope differs")
    rebuilt = fit_allocation_feature_preprocessing(
        dataset,
        feature_indices=fit.feature_indices,
        fit_symbol_indices=fit.fit_symbol_indices,
        fit_start=fit.start_index,
        fit_stop=fit.stop_index,
        fit_as_of=np.datetime64(declaration.fit_as_of_ns, "ns"),
        first_decision_index=first_decision_index,
    )
    if canonical_json_bytes(rebuilt.payload()) != canonical_json_bytes(
        declaration.payload()
    ):
        raise ValueError("actual training preprocessing prefix differs from frozen fit")
    return preprocessing_fit_payload(rebuilt)


__all__ = [
    "fit_allocation_feature_preprocessing",
    "validate_preprocessing_application",
    "validate_training_preprocessing",
]
