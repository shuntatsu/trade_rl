"""Fit direct-simple Ridge once per mature prefix and freeze the following block."""

from __future__ import annotations

import math

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.strategies.forecasts.simple_return import (
    SimpleReturnTrainingSet,
    _fit_simple_return_training,
)
from trade_rl.strategies.forecasts.simple_stream import (
    FrozenSimpleReturnStream,
    SimpleReturnPacket,
    SimpleReturnVintage,
)
from trade_rl.strategies.forecasts.stream import ForecastBlock, _after, _validate_blocks
from trade_rl.strategies.forecasts.supervised import build_causal_forecast_training_set
from trade_rl.strategies.forecasts.training_trace import _timestamp


def fit_prequential_simple_ridge(
    dataset: MarketDataset,
    *,
    blocks: tuple[ForecastBlock, ...],
    feature_indices: tuple[int, ...],
    fit_symbol_indices: tuple[int, ...] | None = None,
    horizon_hours: int = 24,
    alpha: float = 1.0,
) -> FrozenSimpleReturnStream:
    """Estimate raw close ratios, not wealth returns or calibrated conditional means.

    Corporate actions are not screened using the future Dataset. Recorded labels
    remain price returns; canonical execution owns realized corporate actions.
    """
    _validate_blocks(blocks)
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isfinite(alpha)
        or alpha <= 0
    ):
        raise ValueError("alpha must be a finite positive number")
    vintages: list[SimpleReturnVintage] = []
    packets: list[SimpleReturnPacket] = []
    available = dataset.resolved_array("available_at")
    information = dataset.resolved_array("information_available")
    for block in blocks:
        training = SimpleReturnTrainingSet(
            build_causal_forecast_training_set(
                dataset,
                feature_indices=feature_indices,
                fit_symbol_indices=fit_symbol_indices,
                fit_cutoff=block.fit_cutoff,
                horizon_hours=horizon_hours,
            )
        )
        model = _fit_simple_return_training(training, alpha=float(alpha))
        vintage = SimpleReturnVintage(block, training, model, dataset.symbols)
        vintages.append(vintage)
        for row, timestamp in enumerate(dataset.timestamps):
            as_of = _timestamp(timestamp, field="decision_time")
            if not block.prediction_start <= as_of < block.prediction_stop:
                continue
            for symbol_index, symbol in enumerate(dataset.symbols):
                if not information[row, symbol_index] or not np.all(
                    dataset.feature_available[
                        row, symbol_index, list(model.feature_indices)
                    ]
                ):
                    raise ValueError(
                        "simple forecast sources or selected inputs are unavailable"
                    )
                values = dataset.features[row, symbol_index]
                source_time = available[row, symbol_index]
                packets.append(
                    SimpleReturnPacket(
                        symbol=symbol,
                        as_of=as_of,
                        source_available_at=source_time,
                        forecast_available_at=_after(
                            max(as_of, block.model_fit_time, source_time),
                            block.inference_delay_seconds,
                        ),
                        horizon_end=_after(as_of, horizon_hours * 3600),
                        horizon_seconds=horizon_hours * 3600,
                        expected_simple_return=model.predict(values),
                        fit_prefix_marginal_variance=model.fit_prefix_marginal_variance,
                        vintage_digest=vintage.digest,
                        feature_values=tuple(
                            float(v) for v in values[list(model.feature_indices)]
                        ),
                        decision_close=float(dataset.close[row, symbol_index]),
                    )
                )
    return FrozenSimpleReturnStream(dataset.dataset_id, tuple(vintages), tuple(packets))


__all__ = ["fit_prequential_simple_ridge"]
