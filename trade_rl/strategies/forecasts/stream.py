"""Frozen Ridge forecasts under an explicitly declared simulation availability plan."""

from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any

import numpy as np

from trade_rl._validation import require_sha256, require_unique_non_empty
from trade_rl.artifacts.hashing import content_digest
from trade_rl.strategies.forecasts.ridge import RidgeForecastModel
from trade_rl.strategies.forecasts.supervised import CausalForecastTrainingSet
from trade_rl.strategies.forecasts.training_trace import (
    ForecastTrainingTrace,
    _timestamp,
)

_NS_PER_SECOND = 1_000_000_000


def _ns(timestamp: np.datetime64) -> int:
    return int(timestamp.astype(np.int64))


def _from_ns(value: object) -> np.datetime64:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not -(2**63) < value < 2**63
    ):
        raise ValueError("timestamp must be a valid integer nanosecond epoch")
    return np.datetime64(value, "ns")


def _after(timestamp: np.datetime64, seconds: int) -> np.datetime64:
    return _from_ns(_ns(timestamp) + seconds * _NS_PER_SECOND)


@dataclass(frozen=True, slots=True)
class ForecastBlock:
    """Historical availability assumptions, not observed compute receipts."""

    fit_cutoff: np.datetime64
    model_fit_time: np.datetime64
    prediction_start: np.datetime64
    prediction_stop: np.datetime64
    inference_delay_seconds: int = 0

    def __post_init__(self) -> None:
        for field in (
            "fit_cutoff",
            "model_fit_time",
            "prediction_start",
            "prediction_stop",
        ):
            object.__setattr__(
                self, field, _timestamp(getattr(self, field), field=field)
            )
        if (
            not self.fit_cutoff
            <= self.model_fit_time
            <= self.prediction_start
            < self.prediction_stop
        ):
            raise ValueError(
                "fit cutoff/completion and prediction block must be ordered"
            )
        if self.fit_cutoff >= self.prediction_start:
            raise ValueError("prediction block must follow its fit boundary")
        delay = self.inference_delay_seconds
        if isinstance(delay, bool) or not isinstance(delay, int) or delay < 0:
            raise ValueError("inference delay must be a non-negative integer")
        _after(self.prediction_stop, delay)

    def payload(self) -> dict[str, object]:
        return {
            **{
                field: _ns(getattr(self, field))
                for field in (
                    "fit_cutoff",
                    "model_fit_time",
                    "prediction_start",
                    "prediction_stop",
                )
            },
            "inference_delay_seconds": self.inference_delay_seconds,
        }


def _validate_blocks(blocks: tuple[ForecastBlock, ...]) -> None:
    if (
        not isinstance(blocks, tuple)
        or not blocks
        or any(not isinstance(b, ForecastBlock) for b in blocks)
    ):
        raise ValueError("blocks must be a non-empty tuple of ForecastBlock")
    for previous, current in zip(blocks, blocks[1:]):
        if (
            current.prediction_start < previous.prediction_stop
            or current.fit_cutoff <= previous.fit_cutoff
            or current.model_fit_time <= previous.model_fit_time
        ):
            raise ValueError(
                "blocks must have disjoint ordered windows and increasing fits"
            )


def _model_payload(model: RidgeForecastModel) -> dict[str, object]:
    return {
        "feature_indices": list(model.feature_indices),
        "feature_mean": model.feature_mean.tolist(),
        "feature_scale": model.feature_scale.tolist(),
        "coefficients": model.coefficients.tolist(),
        "intercept": model.intercept,
        "horizon_hours": model.horizon_hours,
        "alpha": model.alpha,
        "n_samples": model.n_samples,
        "fit_cutoff": _ns(model.fit_cutoff),
    }


@dataclass(frozen=True, slots=True)
class RidgeForecastVintage:
    block: ForecastBlock
    training: CausalForecastTrainingSet
    model: RidgeForecastModel
    prediction_symbols: tuple[str, ...]
    _digest: str = dataclass_field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "prediction_symbols",
            require_unique_non_empty(
                self.prediction_symbols, field="prediction_symbols"
            ),
        )
        if (
            not isinstance(self.block, ForecastBlock)
            or not isinstance(self.training, CausalForecastTrainingSet)
            or not isinstance(self.model, RidgeForecastModel)
        ):
            raise ValueError(
                "vintage must bind block, actual training rows and Ridge model"
            )
        trace = self.training.trace
        if trace is None:
            raise ValueError("vintage requires actual source-maturity evidence")
        if (
            self.model.feature_indices != self.training.feature_indices
            or self.model.n_samples != self.training.n_samples
            or self.model.horizon_hours != self.training.horizon_hours
            or self.model.fit_cutoff != self.training.fit_cutoff
            or self.block.fit_cutoff != self.training.fit_cutoff
        ):
            raise ValueError("model and block must describe the same training scope")
        if np.max(trace.label_available_times) >= self.block.model_fit_time:
            raise ValueError("training source maturity must precede model completion")
        if not set(trace.row_symbols).issubset(self.prediction_symbols):
            raise ValueError("fit scope must belong to the declared symbol roster")
        object.__setattr__(self, "_digest", content_digest(self._content()))

    def _content(self) -> dict[str, object]:
        return {
            "schema": "ridge_forecast_vintage_v1",
            "return_unit": "log_return",
            "forecast_kind": "conditional_mean",
            "label_basis": "log_close_to_close",
            "calibration_state": "uncalibrated",
            "block": self.block.payload(),
            "training": self.training.scope_payload(),
            "model": _model_payload(self.model),
            "prediction_symbols": list(self.prediction_symbols),
        }

    @property
    def digest(self) -> str:
        return self._digest

    def payload(self) -> dict[str, object]:
        return {**self._content(), "digest": self.digest}


@dataclass(frozen=True, slots=True)
class ForecastPacket:
    symbol: str
    as_of: np.datetime64
    source_available_at: np.datetime64
    forecast_available_at: np.datetime64
    horizon_end: np.datetime64
    horizon_seconds: int
    forecast_log_return: float
    vintage_digest: str
    feature_values: tuple[float, ...]
    return_unit: str = "log_return"
    forecast_kind: str = "conditional_mean"

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not self.symbol.strip():
            raise ValueError("packet symbol must be non-empty")
        require_sha256(self.vintage_digest, field="vintage_digest")
        for field in (
            "as_of",
            "source_available_at",
            "forecast_available_at",
            "horizon_end",
        ):
            object.__setattr__(
                self, field, _timestamp(getattr(self, field), field=field)
            )
        if (
            self.source_available_at > self.as_of
            or self.forecast_available_at < self.as_of
        ):
            raise ValueError("packet source/forecast availability is inconsistent")
        horizon = self.horizon_seconds
        if (
            isinstance(horizon, bool)
            or not isinstance(horizon, int)
            or horizon <= 0
            or self.horizon_end != _after(self.as_of, horizon)
        ):
            raise ValueError("packet must bind its exact positive horizon")
        if self.return_unit != "log_return" or self.forecast_kind != "conditional_mean":
            raise ValueError("Ridge packets represent conditional mean log returns")
        values = (*self.feature_values, self.forecast_log_return)
        if (
            not isinstance(self.feature_values, tuple)
            or not self.feature_values
            or any(
                isinstance(v, bool)
                or not isinstance(v, (float, int))
                or not math.isfinite(v)
                for v in values
            )
        ):
            raise ValueError(
                "packet forecast and selected inputs must be finite numbers"
            )
        object.__setattr__(
            self, "feature_values", tuple(float(v) for v in self.feature_values)
        )
        object.__setattr__(self, "forecast_log_return", float(self.forecast_log_return))

    def _snapshot(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "as_of": _ns(self.as_of),
            "source_available_at": _ns(self.source_available_at),
            "feature_values": list(self.feature_values),
        }

    @property
    def snapshot_digest(self) -> str:
        return content_digest(self._snapshot())

    def _content(self) -> dict[str, object]:
        return {
            "schema": "forecast_packet_v1",
            **self._snapshot(),
            "snapshot_digest": self.snapshot_digest,
            "forecast_available_at": _ns(self.forecast_available_at),
            "horizon_end": _ns(self.horizon_end),
            "horizon_seconds": self.horizon_seconds,
            "forecast_log_return": self.forecast_log_return,
            "vintage_digest": self.vintage_digest,
            "return_unit": self.return_unit,
            "forecast_kind": self.forecast_kind,
        }

    @property
    def digest(self) -> str:
        return content_digest(self._content())

    def payload(self) -> dict[str, object]:
        return {**self._content(), "digest": self.digest}


def _packet_prediction(packet: ForecastPacket, model: RidgeForecastModel) -> float:
    return model.predict_selected(np.asarray(packet.feature_values, dtype=np.float64))


@dataclass(frozen=True, slots=True)
class FrozenForecastStream:
    dataset_id: str
    vintages: tuple[RidgeForecastVintage, ...]
    packets: tuple[ForecastPacket, ...]

    def __post_init__(self) -> None:
        require_sha256(self.dataset_id, field="dataset_id")
        if (
            not isinstance(self.vintages, tuple)
            or not self.vintages
            or any(not isinstance(v, RidgeForecastVintage) for v in self.vintages)
        ):
            raise ValueError("stream requires immutable Ridge vintages")
        if (
            not isinstance(self.packets, tuple)
            or not self.packets
            or any(not isinstance(p, ForecastPacket) for p in self.packets)
        ):
            raise ValueError("stream requires immutable packets")
        _validate_blocks(tuple(v.block for v in self.vintages))
        owners = {v.digest: v for v in self.vintages}
        seen: set[tuple[str, int]] = set()
        used: set[str] = set()
        for packet in self.packets:
            owner = owners.get(packet.vintage_digest)
            if owner is None or packet.symbol not in owner.prediction_symbols:
                raise ValueError(
                    "packet must belong to its declared model/symbol scope"
                )
            key = (packet.symbol, _ns(packet.as_of))
            if key in seen:
                raise ValueError("duplicate packet decision source")
            seen.add(key)
            used.add(owner.digest)
            block = owner.block
            if not block.prediction_start <= packet.as_of < block.prediction_stop:
                raise ValueError("packet must belong to its vintage's future block")
            if (
                packet.horizon_seconds != owner.model.horizon_hours * 3600
                or packet.forecast_available_at
                != _after(
                    max(packet.as_of, block.model_fit_time, packet.source_available_at),
                    block.inference_delay_seconds,
                )
            ):
                raise ValueError(
                    "packet horizon/availability must match its frozen recipe"
                )
            if not math.isclose(
                packet.forecast_log_return,
                _packet_prediction(packet, owner.model),
                rel_tol=1e-12,
                abs_tol=1e-15,
            ):
                raise ValueError(
                    "packet forecast does not match the frozen model and inputs"
                )
        if used != set(owners):
            raise ValueError("each declared block must have recorded predictions")

    def _causal_content(self) -> dict[str, object]:
        return {
            "vintages": [v.payload() for v in self.vintages],
            "packets": [p.payload() for p in self.packets],
        }

    @property
    def causal_scope_digest(self) -> str:
        return content_digest(self._causal_content())

    def payload(self) -> dict[str, object]:
        return {
            "schema": "prequential_ridge_stream_v1",
            "availability_semantics": "declared_simulation_v1",
            "dataset_id": self.dataset_id,
            **self._causal_content(),
            "causal_scope_digest": self.causal_scope_digest,
        }

    @property
    def digest(self) -> str:
        return content_digest(self.payload())

    @classmethod
    def from_payload(
        cls, payload: object, *, expected_digest: str
    ) -> FrozenForecastStream:
        require_sha256(expected_digest, field="expected_digest")
        if content_digest(payload) != expected_digest:
            raise ValueError("forecast stream content digest mismatch")
        try:
            data = _mapping(payload)
            vintages = tuple(_read_vintage(v) for v in data["vintages"])
            packets = tuple(_read_packet(p) for p in data["packets"])
            restored = cls(data["dataset_id"], vintages, packets)
            if restored.digest != expected_digest:
                raise ValueError(
                    "forecast stream contains inconsistent nested identities or schema"
                )
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError("invalid frozen forecast stream") from error
        return restored


def _mapping(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise ValueError("artifact records must be JSON objects")
    return value


def _read_vintage(value: object) -> RidgeForecastVintage:
    data = _mapping(value)
    block_data = _mapping(data["block"])
    block = ForecastBlock(
        **{
            field: _from_ns(block_data[field])
            for field in (
                "fit_cutoff",
                "model_fit_time",
                "prediction_start",
                "prediction_stop",
            )
        },
        inference_delay_seconds=block_data["inference_delay_seconds"],
    )
    training_data = _mapping(data["training"])
    trace_data = _mapping(training_data["trace"])
    trace = ForecastTrainingTrace(
        row_symbols=tuple(trace_data["row_symbols"]),
        **{
            field: np.array([_from_ns(t) for t in trace_data[field]])
            for field in (
                "start_times",
                "end_times",
                "start_available_at",
                "end_available_at",
            )
        },
        start_close=np.array(trace_data["start_close"]),
        end_close=np.array(trace_data["end_close"]),
    )
    training = CausalForecastTrainingSet(
        feature_indices=tuple(training_data["feature_indices"]),
        feature_names=tuple(training_data["feature_names"]),
        features=np.array(training_data["features"]),
        labels=np.array(training_data["labels"]),
        sample_weights=np.array(training_data["sample_weights"]),
        label_end_times=trace.end_times,
        fit_cutoff=_from_ns(training_data["fit_cutoff"]),
        horizon_hours=training_data["horizon_hours"],
        trace=trace,
    )
    model_data = _mapping(data["model"])
    model = RidgeForecastModel(
        feature_indices=tuple(model_data["feature_indices"]),
        **{
            field: np.array(model_data[field])
            for field in ("feature_mean", "feature_scale", "coefficients")
        },
        intercept=model_data["intercept"],
        horizon_hours=model_data["horizon_hours"],
        alpha=model_data["alpha"],
        n_samples=model_data["n_samples"],
        fit_cutoff=_from_ns(model_data["fit_cutoff"]),
    )
    return RidgeForecastVintage(
        block, training, model, tuple(data["prediction_symbols"])
    )


def _read_packet(value: object) -> ForecastPacket:
    data = _mapping(value)
    return ForecastPacket(
        symbol=data["symbol"],
        **{
            field: _from_ns(data[field])
            for field in (
                "as_of",
                "source_available_at",
                "forecast_available_at",
                "horizon_end",
            )
        },
        horizon_seconds=data["horizon_seconds"],
        forecast_log_return=data["forecast_log_return"],
        vintage_digest=data["vintage_digest"],
        feature_values=tuple(data["feature_values"]),
        return_unit=data["return_unit"],
        forecast_kind=data["forecast_kind"],
    )


__all__ = [
    "ForecastBlock",
    "ForecastPacket",
    "FrozenForecastStream",
    "RidgeForecastVintage",
]
