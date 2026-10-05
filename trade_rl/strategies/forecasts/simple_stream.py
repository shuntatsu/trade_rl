"""Separate, frozen simple-price-return projection records and strict reader."""

from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import field as dataclass_field

import numpy as np

from trade_rl._validation import require_sha256, require_unique_non_empty
from trade_rl.artifacts.hashing import content_digest
from trade_rl.strategies.forecasts.simple_return import (
    SimpleReturnRidgeModel,
    SimpleReturnTrainingSet,
)
from trade_rl.strategies.forecasts.stream import (
    ForecastBlock,
    _after,
    _from_ns,
    _mapping,
    _ns,
    _validate_blocks,
)
from trade_rl.strategies.forecasts.supervised import CausalForecastTrainingSet
from trade_rl.strategies.forecasts.training_trace import (
    ForecastTrainingTrace,
    _timestamp,
)


def _model_payload(model: SimpleReturnRidgeModel) -> dict[str, object]:
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
        "return_unit": model.return_unit,
        "valuation_basis": model.valuation_basis,
        "variance_kind": model.variance_kind,
        "fit_prefix_marginal_variance": model.fit_prefix_marginal_variance,
    }


@dataclass(frozen=True, slots=True)
class SimpleReturnVintage:
    block: ForecastBlock
    training: SimpleReturnTrainingSet
    model: SimpleReturnRidgeModel
    prediction_symbols: tuple[str, ...]
    _digest: str = dataclass_field(init=False, repr=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.block, ForecastBlock)
            or not isinstance(self.training, SimpleReturnTrainingSet)
            or not isinstance(self.model, SimpleReturnRidgeModel)
        ):
            raise ValueError("simple vintage requires traced direct labels and model")
        object.__setattr__(
            self,
            "prediction_symbols",
            require_unique_non_empty(
                self.prediction_symbols, field="prediction_symbols"
            ),
        )
        if (
            self.model.feature_indices != self.training.feature_indices
            or self.model.n_samples != self.training.n_samples
            or self.model.horizon_hours != self.training.horizon_hours
            or self.model.fit_cutoff != self.training.fit_cutoff
            or self.block.fit_cutoff != self.training.fit_cutoff
            or self.model.fit_prefix_marginal_variance
            != self.training.fit_prefix_marginal_variance
        ):
            raise ValueError("simple model and variance must bind their training scope")
        trace = self.training.trace
        if (
            trace is None
            or np.max(trace.label_available_times) >= self.block.model_fit_time
        ):
            raise ValueError("mature traced sources must precede model completion")
        if not set(trace.row_symbols).issubset(self.prediction_symbols):
            raise ValueError("fit symbols must belong to the prediction roster")
        object.__setattr__(self, "_digest", content_digest(self._content()))

    def _content(self) -> dict[str, object]:
        return {
            "schema": "simple_return_ridge_vintage_v1",
            "return_unit": "expected_simple_return",
            "forecast_kind": "regularized_linear_projection",
            "valuation_basis": "same_close_price_return",
            "calibration_state": "uncalibrated",
            "variance_kind": "fit_prefix_marginal_variance",
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
class SimpleReturnPacket:
    symbol: str
    as_of: np.datetime64
    source_available_at: np.datetime64
    forecast_available_at: np.datetime64
    horizon_end: np.datetime64
    horizon_seconds: int
    expected_simple_return: float
    fit_prefix_marginal_variance: float
    vintage_digest: str
    feature_values: tuple[float, ...]
    decision_close: float
    return_unit: str = "expected_simple_return"
    valuation_basis: str = "same_close_price_return"
    variance_kind: str = "fit_prefix_marginal_variance"
    forecast_kind: str = "regularized_linear_projection"

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
            raise ValueError(
                "simple forecast source/completion clocks are inconsistent"
            )
        if (
            isinstance(self.horizon_seconds, bool)
            or not isinstance(self.horizon_seconds, int)
            or self.horizon_seconds <= 0
            or self.horizon_end != _after(self.as_of, self.horizon_seconds)
        ):
            raise ValueError("packet must bind its exact positive horizon")
        if (
            self.return_unit != "expected_simple_return"
            or self.valuation_basis != "same_close_price_return"
            or self.variance_kind != "fit_prefix_marginal_variance"
            or self.forecast_kind != "regularized_linear_projection"
        ):
            raise ValueError(
                "simple packet unit, valuation or projection kind is invalid"
            )
        if not isinstance(self.feature_values, tuple) or not self.feature_values:
            raise ValueError("packet requires immutable selected inputs")
        for field, value in (
            *(("feature_values", v) for v in self.feature_values),
            ("expected_simple_return", self.expected_simple_return),
            ("fit_prefix_marginal_variance", self.fit_prefix_marginal_variance),
            ("decision_close", self.decision_close),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{field} must contain finite numbers")
        if (
            self.expected_simple_return < -1
            or self.fit_prefix_marginal_variance < 0
            or self.decision_close <= 0
        ):
            raise ValueError(
                "simple return, variance or decision price is outside its domain"
            )
        object.__setattr__(
            self, "feature_values", tuple(float(v) for v in self.feature_values)
        )
        for field in (
            "expected_simple_return",
            "fit_prefix_marginal_variance",
            "decision_close",
        ):
            object.__setattr__(self, field, float(getattr(self, field)))

    def _snapshot(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "as_of": _ns(self.as_of),
            "source_available_at": _ns(self.source_available_at),
            "feature_values": list(self.feature_values),
            "decision_close": self.decision_close,
        }

    @property
    def snapshot_digest(self) -> str:
        return content_digest(self._snapshot())

    def _content(self) -> dict[str, object]:
        return {
            "schema": "simple_return_packet_v1",
            **self._snapshot(),
            "snapshot_digest": self.snapshot_digest,
            "forecast_available_at": _ns(self.forecast_available_at),
            "horizon_end": _ns(self.horizon_end),
            "horizon_seconds": self.horizon_seconds,
            "expected_simple_return": self.expected_simple_return,
            "fit_prefix_marginal_variance": self.fit_prefix_marginal_variance,
            "vintage_digest": self.vintage_digest,
            "return_unit": self.return_unit,
            "valuation_basis": self.valuation_basis,
            "variance_kind": self.variance_kind,
            "forecast_kind": self.forecast_kind,
        }

    @property
    def digest(self) -> str:
        return content_digest(self._content())

    def payload(self) -> dict[str, object]:
        return {**self._content(), "digest": self.digest}


@dataclass(frozen=True, slots=True)
class FrozenSimpleReturnStream:
    dataset_id: str
    vintages: tuple[SimpleReturnVintage, ...]
    packets: tuple[SimpleReturnPacket, ...]

    def __post_init__(self) -> None:
        require_sha256(self.dataset_id, field="dataset_id")
        if (
            not isinstance(self.vintages, tuple)
            or not self.vintages
            or any(not isinstance(v, SimpleReturnVintage) for v in self.vintages)
        ):
            raise ValueError("simple stream requires immutable vintages")
        if (
            not isinstance(self.packets, tuple)
            or not self.packets
            or any(not isinstance(p, SimpleReturnPacket) for p in self.packets)
        ):
            raise ValueError("simple stream requires immutable packets")
        _validate_blocks(tuple(v.block for v in self.vintages))
        owners = {v.digest: v for v in self.vintages}
        seen: set[tuple[str, int]] = set()
        used: set[str] = set()
        for packet in self.packets:
            owner = owners.get(packet.vintage_digest)
            if owner is None or packet.symbol not in owner.prediction_symbols:
                raise ValueError("packet must belong to its model/symbol scope")
            key = (packet.symbol, _ns(packet.as_of))
            if key in seen:
                raise ValueError("duplicate simple packet decision source")
            seen.add(key)
            used.add(owner.digest)
            block = owner.block
            if not block.prediction_start <= packet.as_of < block.prediction_stop:
                raise ValueError("simple packet lies outside its following block")
            if (
                packet.horizon_seconds != owner.model.horizon_hours * 3600
                or packet.forecast_available_at
                != _after(
                    max(packet.as_of, block.model_fit_time, packet.source_available_at),
                    block.inference_delay_seconds,
                )
                or packet.fit_prefix_marginal_variance
                != owner.model.fit_prefix_marginal_variance
                or len(packet.feature_values) != len(owner.model.feature_indices)
            ):
                raise ValueError(
                    "simple packet horizon, variance or recipe is inconsistent"
                )
            predicted = owner.model.predict_selected(
                np.asarray(packet.feature_values, dtype=np.float64)
            )
            if (
                not math.isfinite(predicted)
                or predicted < -1
                or not math.isclose(
                    packet.expected_simple_return,
                    predicted,
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                )
            ):
                raise ValueError("simple packet disagrees with the frozen projection")
        if used != set(owners):
            raise ValueError("each simple vintage must have recorded predictions")

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
            "schema": "simple_return_ridge_stream_v1",
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
    ) -> FrozenSimpleReturnStream:
        require_sha256(expected_digest, field="expected_digest")
        try:
            if content_digest(payload) != expected_digest:
                raise ValueError("simple stream content digest mismatch")
            data = _mapping(payload)
            restored = cls(
                data["dataset_id"],
                tuple(_read_vintage(v) for v in data["vintages"]),
                tuple(_read_packet(p) for p in data["packets"]),
            )
            if restored.digest != expected_digest:
                raise ValueError(
                    "simple stream nested identities or schemas are inconsistent"
                )
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError("invalid frozen simple-return stream") from error
        return restored


def _read_vintage(value: object) -> SimpleReturnVintage:
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
    training_data = _mapping(_mapping(data["training"])["log_training"])
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
    training = SimpleReturnTrainingSet(
        CausalForecastTrainingSet(
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
    )
    model_data = _mapping(data["model"])
    model = SimpleReturnRidgeModel(
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
        fit_prefix_marginal_variance=model_data["fit_prefix_marginal_variance"],
        return_unit=model_data["return_unit"],
        valuation_basis=model_data["valuation_basis"],
        variance_kind=model_data["variance_kind"],
    )
    return SimpleReturnVintage(
        block, training, model, tuple(data["prediction_symbols"])
    )


def _read_packet(value: object) -> SimpleReturnPacket:
    data = _mapping(value)
    return SimpleReturnPacket(
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
        expected_simple_return=data["expected_simple_return"],
        fit_prefix_marginal_variance=data["fit_prefix_marginal_variance"],
        vintage_digest=data["vintage_digest"],
        feature_values=tuple(data["feature_values"]),
        decision_close=data["decision_close"],
        return_unit=data["return_unit"],
        valuation_basis=data["valuation_basis"],
        variance_kind=data["variance_kind"],
        forecast_kind=data["forecast_kind"],
    )


__all__ = ["SimpleReturnVintage", "SimpleReturnPacket", "FrozenSimpleReturnStream"]
