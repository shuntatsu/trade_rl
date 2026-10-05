"""Prefix-only Ridge generation and consumption of frozen forecast packets."""

from __future__ import annotations

import math
from bisect import bisect_right

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.strategies.forecasts.controller import (
    CostAwareForecastIntentController,
    ForecastIntentConfig,
)
from trade_rl.strategies.forecasts.ridge import _fit_ridge_training_set
from trade_rl.strategies.forecasts.stream import (
    ForecastBlock,
    ForecastPacket,
    FrozenForecastStream,
    RidgeForecastVintage,
    _after,
    _ns,
    _validate_blocks,
)
from trade_rl.strategies.forecasts.supervised import build_causal_forecast_training_set
from trade_rl.strategies.forecasts.training_trace import _timestamp
from trade_rl.strategies.interface import StrategyObservation
from trade_rl.strategies.position_intent import PositionIntent


def fit_prequential_ridge(
    dataset: MarketDataset,
    *,
    blocks: tuple[ForecastBlock, ...],
    feature_indices: tuple[int, ...],
    fit_symbol_indices: tuple[int, ...] | None = None,
    horizon_hours: int = 24,
    alpha: float = 1.0,
) -> FrozenForecastStream:
    """Fit each declared prefix once and publish forecasts only in its next block."""
    _validate_blocks(blocks)
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isfinite(alpha)
        or alpha <= 0
    ):
        raise ValueError("alpha must be a finite positive number")
    vintages: list[RidgeForecastVintage] = []
    packets: list[ForecastPacket] = []
    information_available = dataset.resolved_array("information_available")
    source_available_at = dataset.resolved_array("available_at")
    for block in blocks:
        training = build_causal_forecast_training_set(
            dataset,
            feature_indices=feature_indices,
            fit_symbol_indices=fit_symbol_indices,
            fit_cutoff=block.fit_cutoff,
            horizon_hours=horizon_hours,
        )
        model = _fit_ridge_training_set(training, alpha=float(alpha))
        vintage = RidgeForecastVintage(block, training, model, dataset.symbols)
        vintages.append(vintage)
        for row, as_of in enumerate(dataset.timestamps):
            if not block.prediction_start <= as_of < block.prediction_stop:
                continue
            for symbol_index, symbol in enumerate(dataset.symbols):
                if not information_available[row, symbol_index] or not np.all(
                    dataset.feature_available[
                        row, symbol_index, list(model.feature_indices)
                    ]
                ):
                    raise ValueError(
                        "forecast source or selected features are unavailable"
                    )
                values = dataset.features[row, symbol_index]
                available = source_available_at[row, symbol_index]
                packets.append(
                    ForecastPacket(
                        symbol=symbol,
                        as_of=as_of,
                        source_available_at=available,
                        forecast_available_at=_after(
                            max(as_of, block.model_fit_time, available),
                            block.inference_delay_seconds,
                        ),
                        horizon_end=_after(as_of, horizon_hours * 3600),
                        horizon_seconds=horizon_hours * 3600,
                        forecast_log_return=model.predict(values),
                        vintage_digest=vintage.digest,
                        feature_values=tuple(
                            float(v) for v in values[list(model.feature_indices)]
                        ),
                    )
                )
    return FrozenForecastStream(dataset.dataset_id, tuple(vintages), tuple(packets))


class PacketForecastStrategy:
    """Consume a frozen causal packet stream through the existing cost proxy."""

    def __init__(
        self,
        stream: FrozenForecastStream,
        *,
        entry_threshold: float,
        exit_threshold: float,
        one_way_switch_cost: float,
    ) -> None:
        if not isinstance(stream, FrozenForecastStream):
            raise ValueError("stream must be a FrozenForecastStream")
        self.stream = stream
        self.controller = CostAwareForecastIntentController(
            ForecastIntentConfig(entry_threshold, exit_threshold),
            one_way_switch_cost=one_way_switch_cost,
        )
        grouped: dict[tuple[str, str], list[ForecastPacket]] = {}
        for packet in stream.packets:
            grouped.setdefault((packet.vintage_digest, packet.symbol), []).append(
                packet
            )
        self._packets = {
            key: tuple(sorted(packets, key=lambda p: p.as_of))
            for key, packets in grouped.items()
        }
        # Within a block readiness is as_of + the fixed inference delay.
        self._ready_times = {
            key: tuple(_ns(p.forecast_available_at) for p in packets)
            for key, packets in self._packets.items()
        }
        self._block_starts = tuple(
            _ns(v.block.prediction_start) for v in stream.vintages
        )

    def decide(self, observation: StrategyObservation) -> PositionIntent:
        timestamp = _timestamp(observation.timestamp, field="decision_time")
        block_index = bisect_right(self._block_starts, _ns(timestamp)) - 1
        if block_index >= 0:
            owner = self.stream.vintages[block_index]
            if timestamp < owner.block.prediction_stop:
                key = (owner.digest, observation.symbol)
                ready_index = (
                    bisect_right(self._ready_times.get(key, ()), _ns(timestamp)) - 1
                )
                if ready_index >= 0:
                    newest = self._packets[key][ready_index]
                    if timestamp < newest.horizon_end:
                        return self.controller.decide(
                            newest.forecast_log_return,
                            current=observation.current_intent,
                        )
        raise ValueError(
            "no ready unexpired forecast in the active block for this symbol"
        )


__all__ = [
    "PacketForecastStrategy",
    "fit_prequential_ridge",
]
