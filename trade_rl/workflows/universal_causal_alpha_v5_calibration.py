"""Chronological train-only calibration for Causal Alpha V5."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.identity import content_and_arrays_digest
from trade_rl.data.universal_features import UNIVERSAL_INSTRUMENT_DESCRIPTOR_NAMES
from trade_rl.learning.causal_alpha_teacher import (
    CausalAlphaRidgeConfig,
    CausalAlphaRidgeModel,
    fit_causal_alpha_ridge,
)
from trade_rl.learning.causal_alpha_v3 import causal_alpha_overlap_uniqueness_weights
from trade_rl.learning.causal_alpha_v4 import (
    CausalAlphaV4Forecast,
    CausalAlphaV4SymbolSamples,
)
from trade_rl.learning.causal_alpha_v5 import (
    CAUSAL_ALPHA_V5_CALIBRATION_FEATURE_NAMES,
    CausalAlphaV5CalibrationConfig,
    CausalAlphaV5CalibrationFit,
    CausalAlphaV5SelectiveForecast,
    build_causal_alpha_v5_selective_forecast,
)
from trade_rl.workflows.universal_causal_alpha_v4_fitting import CausalAlphaV4Fit
from trade_rl.workflows.universal_causal_alpha_v4_runtime import (
    validate_causal_alpha_v4_train_sample_scope,
)

_SPLIT_SCHEMA: Final = "causal_alpha_v5_calibration_split_v1"
_WEIGHT_SCHEMA: Final = "causal_alpha_v5_calibration_weight_v1"
_FORWARD_RESIDUAL_SCHEMA: Final = "causal_alpha_v5_forward_residual_v1"
_EPSILON: Final = 1e-12


@dataclass(frozen=True, slots=True)
class CausalAlphaV5CalibrationSplit:
    """One authored chronological base/calibration boundary and four blocks."""

    train_symbols: tuple[str, ...]
    calibration_start: int
    train_stop: int
    block_boundaries: tuple[int, ...]
    schema_version: str = _SPLIT_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        symbols = tuple(self.train_symbols)
        boundaries = tuple(self.block_boundaries)
        if not symbols or len(set(symbols)) != len(symbols):
            raise ValueError("V5 calibration split symbols must be unique")
        if (
            isinstance(self.calibration_start, bool)
            or not isinstance(self.calibration_start, int)
            or isinstance(self.train_stop, bool)
            or not isinstance(self.train_stop, int)
            or self.calibration_start <= 0
            or self.train_stop <= self.calibration_start
        ):
            raise ValueError("V5 calibration split boundaries are invalid")
        if (
            len(boundaries) != 5
            or boundaries[0] != self.calibration_start
            or boundaries[-1] != self.train_stop
            or any(left >= right for left, right in zip(boundaries, boundaries[1:]))
        ):
            raise ValueError("V5 calibration block boundaries are invalid")
        if self.schema_version != _SPLIT_SCHEMA:
            raise ValueError("unsupported V5 calibration split schema")
        payload = {
            "block_boundaries": boundaries,
            "calibration_start": self.calibration_start,
            "schema_version": self.schema_version,
            "train_stop": self.train_stop,
            "train_symbols": symbols,
        }
        expected = content_digest(payload)
        if self.digest and self.digest != expected:
            raise ValueError("V5 calibration split digest mismatch")
        object.__setattr__(self, "train_symbols", symbols)
        object.__setattr__(self, "block_boundaries", boundaries)
        object.__setattr__(self, "digest", expected)

    @classmethod
    def from_samples(
        cls,
        *,
        train_symbols: tuple[str, ...],
        samples: Mapping[str, CausalAlphaV4SymbolSamples],
        train_stop: int,
        config: CausalAlphaV5CalibrationConfig,
    ) -> CausalAlphaV5CalibrationSplit:
        ordered = validate_causal_alpha_v4_train_sample_scope(
            train_symbols=train_symbols,
            samples=samples,
        )
        if isinstance(train_stop, bool) or not isinstance(train_stop, int):
            raise ValueError("V5 calibration train_stop must be an integer")
        if not isinstance(config, CausalAlphaV5CalibrationConfig):
            raise TypeError("V5 calibration split config is invalid")
        reference = ordered[train_symbols[0]]
        decisions = np.asarray(reference.decision_indices, dtype=np.int64)
        ends = np.asarray(reference.label_end_indices_72h, dtype=np.int64)
        eligible = (decisions < train_stop) & (ends >= decisions) & (ends < train_stop)
        eligible_decisions = decisions[eligible]
        base_count = int(
            math.floor(eligible_decisions.size * (1.0 - config.calibration_fraction))
        )
        if base_count <= 0 or base_count >= eligible_decisions.size:
            raise ValueError("V5 calibration split has insufficient causal decisions")
        calibration_start = int(eligible_decisions[base_count])
        calibration_eligible = (
            (decisions >= calibration_start)
            & (decisions < train_stop)
            & (ends >= decisions)
            & (ends < train_stop)
        )
        selected = np.flatnonzero(calibration_eligible)
        if selected.size < config.forward_block_count:
            raise ValueError("V5 calibration split has insufficient forward blocks")
        blocks = tuple(
            np.asarray(block, dtype=np.int64)
            for block in np.array_split(selected, config.forward_block_count)
        )
        if any(block.size == 0 for block in blocks):
            raise ValueError("V5 calibration split produced an empty block")
        boundaries = (
            *(int(decisions[block[0]]) for block in blocks),
            train_stop,
        )
        return cls(
            train_symbols=tuple(train_symbols),
            calibration_start=calibration_start,
            train_stop=train_stop,
            block_boundaries=boundaries,
        )


@dataclass(frozen=True, slots=True)
class _SymbolCalibrationRows:
    symbol: str
    decisions: np.ndarray
    label_ends: np.ndarray
    features: np.ndarray
    available: np.ndarray
    residual_labels: np.ndarray
    realized: np.ndarray
    raw_direction: np.ndarray
    block_rows: tuple[np.ndarray, ...]


def _slow_arrays(
    forecast: CausalAlphaV4Forecast,
) -> tuple[np.ndarray, np.ndarray]:
    raw_return = 0.5 * (
        np.asarray(forecast.final_predictions["24h"], dtype=np.float64)
        + np.asarray(forecast.final_predictions["72h"], dtype=np.float64) / 3.0
    )
    raw_direction = 0.5 * (
        np.asarray(forecast.direction_scores["24h"], dtype=np.float64)
        + np.asarray(forecast.direction_scores["72h"], dtype=np.float64)
    )
    return raw_return, raw_direction


def _symbol_rows(
    *,
    sample: CausalAlphaV4SymbolSamples,
    forecast: CausalAlphaV4Forecast,
    uncertainty: np.ndarray,
    split: CausalAlphaV5CalibrationSplit,
) -> _SymbolCalibrationRows:
    decisions = np.asarray(sample.decision_indices, dtype=np.int64)
    ends = np.asarray(sample.label_end_indices_72h, dtype=np.int64)
    rows = int(decisions.size)
    if forecast.symbol != sample.symbol or not np.array_equal(
        forecast.decision_indices, decisions
    ):
        raise ValueError("V5 calibration V4 forecast identity drifted")
    if uncertainty.shape != (rows,) or not np.isfinite(uncertainty).all():
        raise ValueError("V5 slow uncertainty must be aligned and finite")
    if np.any(uncertainty < 0.0):
        raise ValueError("V5 slow uncertainty must be non-negative")
    if sample.instrument_descriptor_names != UNIVERSAL_INSTRUMENT_DESCRIPTOR_NAMES:
        raise ValueError("V5 calibration descriptor order drifted")
    calibration_eligible = (
        (decisions >= split.calibration_start)
        & (decisions < split.train_stop)
        & (ends >= decisions)
        & (ends < split.train_stop)
    )
    if np.any(
        calibration_eligible & ~np.all(sample.instrument_descriptor_available, axis=1)
    ):
        raise ValueError("V5 calibration descriptor availability is incomplete")
    selected = np.flatnonzero(calibration_eligible)
    raw_return, raw_direction = _slow_arrays(forecast)
    realized = 0.5 * (
        np.asarray(sample.labels_24h, dtype=np.float64)
        + np.asarray(sample.labels_72h, dtype=np.float64) / 3.0
    )
    finite = (
        np.isfinite(raw_return[selected])
        & np.isfinite(raw_direction[selected])
        & np.isfinite(realized[selected])
    )
    selected = selected[finite]
    if selected.size == 0:
        raise ValueError(f"V5 {sample.symbol} has no calibration support")
    features = np.column_stack(
        (
            raw_return,
            raw_direction,
            np.log(np.maximum(uncertainty, _EPSILON)),
            sample.instrument_descriptors,
        )
    )
    available = np.column_stack(
        (
            np.ones((rows, 3), dtype=np.bool_),
            sample.instrument_descriptor_available,
        )
    ).astype(np.bool_, copy=False)
    residual = realized - raw_return
    block_rows = tuple(
        selected[(decisions[selected] >= start) & (decisions[selected] < stop)]
        for start, stop in zip(split.block_boundaries[:-1], split.block_boundaries[1:])
    )
    if any(block.size == 0 for block in block_rows):
        raise ValueError(f"V5 {sample.symbol} has an empty calibration block")
    return _SymbolCalibrationRows(
        symbol=sample.symbol,
        decisions=decisions,
        label_ends=ends,
        features=features,
        available=available,
        residual_labels=residual,
        realized=realized,
        raw_direction=raw_direction,
        block_rows=block_rows,
    )


def _pool(
    records: Mapping[str, _SymbolCalibrationRows],
    symbols: tuple[str, ...],
    block_indices: tuple[int, ...],
    *,
    knowledge_cutoff: int,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]
]:
    features: list[np.ndarray] = []
    available: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    directions: list[np.ndarray] = []
    weights_by_symbol: dict[str, np.ndarray] = {}
    for symbol in symbols:
        record = records[symbol]
        rows = np.concatenate(
            tuple(record.block_rows[index] for index in block_indices)
        )
        weights = causal_alpha_overlap_uniqueness_weights(
            record.decisions[rows],
            record.label_ends[rows],
            knowledge_cutoff=knowledge_cutoff,
        )
        if not np.any(weights > 0.0):
            raise ValueError(f"V5 {symbol} calibration weights have no support")
        features.append(record.features[rows])
        available.append(record.available[rows])
        labels.append(record.residual_labels[rows])
        ends.append(record.label_ends[rows])
        directions.append(record.raw_direction[rows])
        weights_by_symbol[symbol] = weights
    return (
        np.concatenate(features, axis=0),
        np.concatenate(available, axis=0),
        np.concatenate(labels),
        np.concatenate(ends),
        np.concatenate(directions),
        weights_by_symbol,
    )


def _weight_digest(
    *,
    kind: str,
    symbols: tuple[str, ...],
    weights: Mapping[str, np.ndarray],
    knowledge_cutoff: int,
) -> str:
    return content_and_arrays_digest(
        {
            "kind": kind,
            "knowledge_cutoff": knowledge_cutoff,
            "schema_version": _WEIGHT_SCHEMA,
            "symbols": symbols,
        },
        tuple((symbol, weights[symbol]) for symbol in symbols),
    )


def _fit_model(
    *,
    features: np.ndarray,
    available: np.ndarray,
    labels: np.ndarray,
    ends: np.ndarray,
    weights: np.ndarray,
    knowledge_cutoff: int,
    config: CausalAlphaV5CalibrationConfig,
) -> CausalAlphaRidgeModel:
    return fit_causal_alpha_ridge(
        features=features,
        labels=labels,
        feature_available=available,
        label_end_indices=ends,
        knowledge_cutoff=knowledge_cutoff,
        feature_names=CAUSAL_ALPHA_V5_CALIBRATION_FEATURE_NAMES,
        config=CausalAlphaRidgeConfig(ridge_strength=config.ridge_strength),
        sample_weights=weights,
        normalize_objective=True,
        working_memory_rows=4096,
    )


def fit_causal_alpha_v5_calibration(
    *,
    train_symbols: tuple[str, ...],
    samples: Mapping[str, CausalAlphaV4SymbolSamples],
    v4_fit: CausalAlphaV4Fit,
    slow_uncertainty: Mapping[str, object],
    train_stop: int,
    config: CausalAlphaV5CalibrationConfig,
) -> CausalAlphaV5CalibrationFit:
    """Fit one shared V5 residual calibrator on a purged chronological suffix."""

    ordered = validate_causal_alpha_v4_train_sample_scope(
        train_symbols=train_symbols,
        samples=samples,
    )
    symbols = tuple(train_symbols)
    if not isinstance(v4_fit, CausalAlphaV4Fit):
        raise TypeError("V5 calibration requires a V4 fit")
    if v4_fit.train_symbols != symbols:
        raise ValueError("V5 calibration V4 train symbol scope drifted")
    if not isinstance(config, CausalAlphaV5CalibrationConfig):
        raise TypeError("V5 calibration config is invalid")
    if set(slow_uncertainty) != set(symbols):
        raise ValueError("V5 slow uncertainty symbol scope drifted")
    split = CausalAlphaV5CalibrationSplit.from_samples(
        train_symbols=symbols,
        samples=ordered,
        train_stop=train_stop,
        config=config,
    )
    if v4_fit.knowledge_cutoff != split.calibration_start:
        raise ValueError("V5 V4 base-fit cutoff must equal calibration_start")

    records: dict[str, _SymbolCalibrationRows] = {}
    for symbol in symbols:
        sample = ordered[symbol]
        uncertainty = np.asarray(slow_uncertainty[symbol], dtype=np.float64).reshape(-1)
        records[symbol] = _symbol_rows(
            sample=sample,
            forecast=v4_fit.predict(sample),
            uncertainty=uncertainty,
            split=split,
        )
    per_symbol_support = tuple(
        sorted(
            (symbol, sum(block.size for block in records[symbol].block_rows))
            for symbol in symbols
        )
    )
    if any(count < config.minimum_symbol_support for _, count in per_symbol_support):
        raise ValueError("V5 calibration symbol support is insufficient")
    if sum(count for _, count in per_symbol_support) < config.minimum_pooled_support:
        raise ValueError("V5 calibration pooled support is insufficient")
    block_support = tuple(
        sum(records[symbol].block_rows[index].size for symbol in symbols)
        for index in range(config.forward_block_count)
    )

    forward_models: list[str] = []
    forward_residuals: list[str] = []
    forward_weight_digests: list[str] = []
    residual_values: list[np.ndarray] = []
    residual_weights: list[np.ndarray] = []
    direction_values: list[np.ndarray] = []
    direction_targets: list[np.ndarray] = []
    direction_weights: list[np.ndarray] = []
    forward_symbol_counts: list[int] = []
    for evaluation_block in range(1, config.forward_block_count):
        cutoff = split.block_boundaries[evaluation_block]
        train_pool = _pool(
            records,
            symbols,
            tuple(range(evaluation_block)),
            knowledge_cutoff=cutoff,
        )
        (
            train_features,
            train_available,
            train_labels,
            train_ends,
            _,
            train_weights_map,
        ) = train_pool
        train_weights = np.concatenate(
            tuple(train_weights_map[symbol] for symbol in symbols)
        )
        model = _fit_model(
            features=train_features,
            available=train_available,
            labels=train_labels,
            ends=train_ends,
            weights=train_weights,
            knowledge_cutoff=cutoff,
            config=config,
        )
        forward_models.append(model.digest)
        forward_weight_digests.append(
            _weight_digest(
                kind=f"forward_{evaluation_block}",
                symbols=symbols,
                weights=train_weights_map,
                knowledge_cutoff=cutoff,
            )
        )

        eval_features: list[np.ndarray] = []
        eval_available: list[np.ndarray] = []
        eval_labels: list[np.ndarray] = []
        eval_directions: list[np.ndarray] = []
        eval_realized: list[np.ndarray] = []
        eval_weights: list[np.ndarray] = []
        symbols_present = 0
        for symbol in symbols:
            record = records[symbol]
            rows = record.block_rows[evaluation_block]
            if rows.size:
                symbols_present += 1
            eval_features.append(record.features[rows])
            eval_available.append(record.available[rows])
            eval_labels.append(record.residual_labels[rows])
            eval_directions.append(record.raw_direction[rows])
            eval_realized.append(record.realized[rows])
            eval_weights.append(
                causal_alpha_overlap_uniqueness_weights(
                    record.decisions[rows],
                    record.label_ends[rows],
                    knowledge_cutoff=split.train_stop,
                )
            )
        forward_symbol_counts.append(symbols_present)
        x_eval = np.concatenate(eval_features, axis=0)
        a_eval = np.concatenate(eval_available, axis=0)
        y_eval = np.concatenate(eval_labels)
        raw_direction_eval = np.concatenate(eval_directions)
        realized_eval = np.concatenate(eval_realized)
        prediction = model.predict(x_eval, feature_available=a_eval)
        residual = y_eval - prediction
        eval_weight = np.concatenate(eval_weights)
        if not np.any(eval_weight > 0.0):
            raise ValueError("V5 forward evaluation weights have no support")
        forward_residuals.append(
            content_and_arrays_digest(
                {
                    "evaluation_block": evaluation_block,
                    "model_digest": model.digest,
                    "schema_version": _FORWARD_RESIDUAL_SCHEMA,
                },
                (
                    ("prediction", prediction),
                    ("residual", residual),
                    ("realized", realized_eval),
                    ("raw_direction", raw_direction_eval),
                ),
            )
        )
        residual_values.append(residual)
        residual_weights.append(eval_weight)
        nonzero = np.sign(realized_eval) != 0.0
        direction_values.append(raw_direction_eval[nonzero])
        direction_targets.append(np.sign(realized_eval[nonzero]))
        direction_weights.append(eval_weight[nonzero])

    all_residuals = np.concatenate(residual_values)
    all_residual_weights = np.concatenate(residual_weights)
    calibration_residual_rmse = math.sqrt(
        float(
            np.average(
                np.square(all_residuals),
                weights=all_residual_weights,
            )
        )
    )
    all_direction = np.concatenate(direction_values)
    all_direction_target = np.concatenate(direction_targets)
    all_direction_weights = np.concatenate(direction_weights)
    if all_direction.size == 0:
        raise ValueError("V5 calibration has no direction-score support")
    direction_score_rmse = math.sqrt(
        float(
            np.average(
                np.square(all_direction_target - all_direction),
                weights=all_direction_weights,
            )
        )
    )

    final_pool = _pool(
        records,
        symbols,
        tuple(range(config.forward_block_count)),
        knowledge_cutoff=train_stop,
    )
    final_features, final_available, final_labels, final_ends, _, final_weights_map = (
        final_pool
    )
    final_weights = np.concatenate(
        tuple(final_weights_map[symbol] for symbol in symbols)
    )
    final_model = _fit_model(
        features=final_features,
        available=final_available,
        labels=final_labels,
        ends=final_ends,
        weights=final_weights,
        knowledge_cutoff=train_stop,
        config=config,
    )
    return CausalAlphaV5CalibrationFit(
        v4_fit_digest=v4_fit.digest,
        v4_fit_config_digest=v4_fit.config.digest,
        v4_sample_scope_digest=v4_fit.sample_scope_digest,
        calibration_start=split.calibration_start,
        train_stop=split.train_stop,
        model=final_model,
        forward_model_digests=tuple(forward_models),
        forward_residual_digests=tuple(forward_residuals),
        final_weight_digest=_weight_digest(
            kind="final",
            symbols=symbols,
            weights=final_weights_map,
            knowledge_cutoff=train_stop,
        ),
        forward_weight_digests=tuple(forward_weight_digests),
        per_symbol_support=per_symbol_support,
        calibration_block_support=block_support,
        forward_block_symbol_counts=tuple(forward_symbol_counts),
        calibration_residual_rmse=calibration_residual_rmse,
        direction_score_rmse=direction_score_rmse,
        config=config,
    )


def calibrate_causal_alpha_v5_forecast(
    *,
    v4_forecast: CausalAlphaV4Forecast,
    sample: CausalAlphaV4SymbolSamples,
    slow_uncertainty: object,
    one_way_cost_rates: object,
    actionable_mask: object,
    calibration_fit: CausalAlphaV5CalibrationFit,
) -> CausalAlphaV5SelectiveForecast:
    """Apply one fitted V5 calibrator without symbol-specific dispatch."""

    if not isinstance(sample, CausalAlphaV4SymbolSamples):
        raise TypeError("V5 forecast calibration requires V4 symbol samples")
    if sample.symbol != v4_forecast.symbol or not np.array_equal(
        sample.decision_indices, v4_forecast.decision_indices
    ):
        raise ValueError("V5 forecast/sample identity drifted")
    if sample.instrument_descriptor_names != UNIVERSAL_INSTRUMENT_DESCRIPTOR_NAMES:
        raise ValueError("V5 forecast descriptor order drifted")
    return build_causal_alpha_v5_selective_forecast(
        v4_forecast=v4_forecast,
        slow_uncertainty=slow_uncertainty,
        instrument_descriptors=sample.instrument_descriptors,
        instrument_descriptor_available=sample.instrument_descriptor_available,
        one_way_cost_rates=one_way_cost_rates,
        actionable_mask=actionable_mask,
        calibration_fit=calibration_fit,
    )


__all__ = [
    "CausalAlphaV5CalibrationSplit",
    "calibrate_causal_alpha_v5_forecast",
    "fit_causal_alpha_v5_calibration",
]
