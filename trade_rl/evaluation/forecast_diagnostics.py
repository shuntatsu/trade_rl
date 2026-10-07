"""Measure supplied frozen forecasts against mature same-close price returns.

This reader neither fits nor predicts. Supplied-source consistency does not
authenticate a historical fit, dataset provenance, or forecast calibration.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketDataset
from trade_rl.strategies.forecasts.simple_stream import (
    FrozenSimpleReturnStream,
    SimpleReturnPacket,
    SimpleReturnVintage,
)
from trade_rl.strategies.forecasts.stream import _after, _validate_blocks
from trade_rl.strategies.forecasts.training_trace import _timestamp


@dataclass(frozen=True, slots=True)
class SimpleReturnForecastDiagnostics:
    """Detached immutable diagnostic content; payload returns a fresh copy."""

    _payload_json: str

    def payload(self) -> dict[str, Any]:
        return dict(json.loads(self._payload_json))

    @property
    def digest(self) -> str:
        return content_digest(self.payload())


def _ns(value: np.datetime64) -> int:
    return int(_timestamp(value, field="source clock").astype(np.int64))


def _metrics(records: list[dict[str, Any]]) -> dict[str, int | float]:
    n = len(records)
    try:
        residuals = [r["residual"] for r in records]
        result = {
            "n": n,
            "mean_prediction": math.fsum(r["prediction"] for r in records) / n,
            "mean_realized": math.fsum(r["realized"] for r in records) / n,
            "bias": math.fsum(residuals) / n,
            "mae": math.fsum(abs(r) for r in residuals) / n,
            "rmse": math.sqrt(math.fsum(r * r for r in residuals) / n),
        }
    except (OverflowError, ValueError) as error:
        raise ValueError("forecast error arithmetic must be finite") from error
    if not all(math.isfinite(v) for v in result.values()):
        raise ValueError("forecast error arithmetic must be finite")
    return result


def _vintages(
    dataset: MarketDataset, stream: FrozenSimpleReturnStream
) -> dict[str, SimpleReturnVintage]:
    if not isinstance(stream.vintages, tuple) or not stream.vintages:
        raise ValueError("diagnostics require frozen vintage scopes")
    owners: dict[str, SimpleReturnVintage] = {}
    for vintage in stream.vintages:
        if not isinstance(vintage, SimpleReturnVintage):
            raise ValueError("diagnostics require frozen vintage scopes")
        # A vintage caches its digest. Recompute declared content without invoking
        # the stream constructor/reader, which would run model prediction.
        payload = vintage.payload()
        payload.pop("digest")
        if content_digest(payload) != vintage.digest or vintage.digest in owners:
            raise ValueError("vintage content or identity changed")
        replace(vintage)  # Validate declarations without fitting or predicting.
        block, training, model = vintage.block, vintage.training, vintage.model
        replace(block)  # Validate the declared clocks; never alter the source.
        if (
            model.fit_cutoff != training.fit_cutoff
            or block.fit_cutoff != training.fit_cutoff
            or model.horizon_hours != training.horizon_hours
            or model.n_samples != training.n_samples
            or model.feature_indices != training.feature_indices
            or model.fit_prefix_marginal_variance
            != training.fit_prefix_marginal_variance
            or any(i >= dataset.n_features for i in model.feature_indices)
            or training.selected_feature_names
            != tuple(dataset.feature_names[i] for i in model.feature_indices)
            or not set(vintage.prediction_symbols).issubset(dataset.symbols)
        ):
            raise ValueError(
                "vintage model, selected feature names or source scope differ"
            )
        trace = training.trace
        if (
            any(_ns(t) >= _ns(block.fit_cutoff) for t in trace.end_times)
            or any(_ns(t) >= _ns(block.fit_cutoff) for t in trace.label_available_times)
            or any(
                _ns(t) >= _ns(block.model_fit_time) for t in trace.label_available_times
            )
        ):
            raise ValueError(
                "vintage labels/publications must precede their causal fit"
            )
        owners[vintage.digest] = vintage
    _validate_blocks(tuple(v.block for v in stream.vintages))
    return owners


def evaluate_simple_return_forecasts(
    dataset: MarketDataset,
    stream: FrozenSimpleReturnStream,
    *,
    decision_indices: tuple[int, ...],
    evaluation_as_of: np.datetime64,
) -> SimpleReturnForecastDiagnostics:
    """Pair every requested owning-vintage symbol with its exact mature endpoint.

    Means and errors are packet-weighted, including across unequal vintage groups.
    The raw price label includes no fee, funding, wealth or conditional-risk claim.
    """
    if not isinstance(dataset, MarketDataset) or not isinstance(
        stream, FrozenSimpleReturnStream
    ):
        raise ValueError(
            "diagnostics require a Dataset and frozen simple-return stream"
        )
    if dataset.dataset_id != stream.dataset_id:
        raise ValueError("forecast Dataset identity differs")
    if (
        not isinstance(decision_indices, tuple)
        or not decision_indices
        or any(
            type(i) is not int or not 0 <= i < dataset.n_bars for i in decision_indices
        )
        or tuple(sorted(set(decision_indices))) != decision_indices
    ):
        raise ValueError(
            "decision_indices must be explicit sorted unique native integers"
        )
    cutoff = _timestamp(evaluation_as_of, field="evaluation_as_of")
    owners = _vintages(dataset, stream)
    timestamps = {_ns(t): row for row, t in enumerate(dataset.timestamps)}
    requested: dict[tuple[str, int], tuple[int, SimpleReturnVintage]] = {}
    for row in decision_indices:
        time = _timestamp(dataset.timestamps[row], field="decision time")
        matching = [
            v
            for v in owners.values()
            if v.block.prediction_start <= time < v.block.prediction_stop
        ]
        if len(matching) != 1:
            raise ValueError(
                "requested decision must belong to exactly one vintage block"
            )
        for symbol in matching[0].prediction_symbols:
            requested[(symbol, _ns(time))] = (row, matching[0])
    if not isinstance(stream.packets, tuple) or not stream.packets:
        raise ValueError("forecast coverage is empty")
    packets: dict[tuple[str, int], SimpleReturnPacket] = {}
    for packet in stream.packets:
        if not isinstance(packet, SimpleReturnPacket):
            raise ValueError("forecast coverage contains an unknown packet")
        replace(packet)  # Packet validation is numeric only, with no prediction.
        key = (packet.symbol, _ns(packet.as_of))
        if key[1] not in timestamps:
            raise ValueError("forecast coverage has an unknown Dataset timestamp")
        owner = owners.get(packet.vintage_digest)
        if key in packets:
            raise ValueError("duplicate forecast coverage")
        if owner is None or packet.symbol not in owner.prediction_symbols:
            raise ValueError("forecast coverage has an unknown vintage/symbol source")
        block, model = owner.block, owner.model
        if (
            not block.prediction_start <= packet.as_of < block.prediction_stop
            or packet.horizon_seconds != model.horizon_hours * 3600
            or packet.fit_prefix_marginal_variance != model.fit_prefix_marginal_variance
            or len(packet.feature_values) != len(model.feature_indices)
            or packet.forecast_available_at
            != _after(
                max(packet.as_of, block.model_fit_time, packet.source_available_at),
                block.inference_delay_seconds,
            )
        ):
            raise ValueError("forecast packet differs from its own causal vintage")
        packets[key] = packet
    if not requested.keys() <= packets.keys():
        raise ValueError("forecast coverage omits a requested decision/symbol")
    information = dataset.resolved_array("information_available")
    available = dataset.resolved_array("available_at")
    records: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for (symbol, time_ns), (row, owner) in requested.items():
        packet = packets[(symbol, time_ns)]
        if packet.vintage_digest != owner.digest:
            raise ValueError("forecast coverage uses a different owning vintage")
        symbol_index = dataset.symbols.index(symbol)
        feature_indices = list(owner.model.feature_indices)
        values = tuple(
            float(v) for v in dataset.features[row, symbol_index, feature_indices]
        )
        source_time = _timestamp(
            available[row, symbol_index], field="source publication"
        )
        if (
            not information[row, symbol_index]
            or not np.all(dataset.feature_available[row, symbol_index, feature_indices])
            or source_time > packet.as_of
            or source_time != packet.source_available_at
            or values != packet.feature_values
            or float(dataset.close[row, symbol_index]) != packet.decision_close
        ):
            raise ValueError("forecast snapshot differs from available Dataset sources")
        if packet.forecast_available_at > packet.as_of or packet.as_of > cutoff:
            raise ValueError("forecast is unavailable at its requested decision/cutoff")
        end_row = timestamps.get(_ns(packet.horizon_end))
        if end_row is None:
            raise ValueError("exact forecast horizon endpoint is missing")
        end_available = _timestamp(
            available[end_row, symbol_index], field="endpoint publication"
        )
        if (
            not information[end_row, symbol_index]
            or end_available < packet.horizon_end
            or max(packet.horizon_end, end_available) > cutoff
        ):
            raise ValueError(
                "forecast endpoint is not mature/published by evaluation_as_of"
            )
        end_close = float(dataset.close[end_row, symbol_index])
        realized = end_close / packet.decision_close - 1.0
        residual = packet.expected_simple_return - realized
        if end_close <= 0 or not all(
            math.isfinite(v) for v in (*values, end_close, realized, residual)
        ):
            raise ValueError("forecast source/error arithmetic must be finite")
        source = {
            "symbol": symbol,
            "decision_index": row,
            "endpoint_index": end_row,
            "as_of": time_ns,
            "horizon_end": _ns(packet.horizon_end),
            "feature_indices": feature_indices,
            "feature_names": list(owner.training.selected_feature_names),
            "feature_values": list(values),
            "feature_available": [True] * len(values),
            "decision_close": packet.decision_close,
            "source_available_at": _ns(source_time),
            "information_available": True,
            "endpoint_close": end_close,
            "endpoint_available_at": _ns(end_available),
            "endpoint_information_available": True,
        }
        record = {
            "symbol": symbol,
            "vintage_digest": owner.digest,
            "horizon_seconds": packet.horizon_seconds,
            "packet": packet.payload(),
            "source_projection_digest": content_digest(source),
            "prediction": packet.expected_simple_return,
            "realized": realized,
            "residual": residual,
        }
        records.append(record)
        sources.append(source)
        groups.setdefault((symbol, owner.digest, packet.horizon_seconds), []).append(
            record
        )
    payload = {
        "schema": "simple_return_forecast_diagnostics_v1",
        "consistency_scope": "supplied_sources_not_authenticated_historical_fit",
        "calibration_state": "uncalibrated",
        "valuation_basis": "same_close_price_return",
        "dataset_id": dataset.dataset_id,
        "stream_digest": stream.digest,
        "decision_indices": list(decision_indices),
        "evaluation_as_of": _ns(cutoff),
        "source_projection": sources,
        "source_projection_digest": content_digest(sources),
        "records": records,
        "metrics": _metrics(records),
        "groups": [
            {
                "symbol": key[0],
                "vintage_digest": key[1],
                "horizon_seconds": key[2],
                "metrics": _metrics(group),
            }
            for key, group in sorted(groups.items())
        ],
    }
    return SimpleReturnForecastDiagnostics(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    )


__all__ = ["SimpleReturnForecastDiagnostics", "evaluate_simple_return_forecasts"]
