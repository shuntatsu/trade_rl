from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import cast

from tools.market_b4_oracle.allocation_choice import AllocationAssumptions

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MIN_NS = -(2**63)
_MAX_NS = 2**63 - 1

_STREAM_KEYS = {
    "schema",
    "availability_semantics",
    "dataset_id",
    "vintages",
    "packets",
    "causal_scope_digest",
}
_PACKET_KEYS = {
    "schema",
    "symbol",
    "as_of",
    "source_available_at",
    "feature_values",
    "decision_close",
    "snapshot_digest",
    "forecast_available_at",
    "horizon_end",
    "horizon_seconds",
    "expected_simple_return",
    "fit_prefix_marginal_variance",
    "vintage_digest",
    "return_unit",
    "valuation_basis",
    "variance_kind",
    "forecast_kind",
    "digest",
}
_VINTAGE_KEYS = {
    "schema",
    "return_unit",
    "forecast_kind",
    "valuation_basis",
    "calibration_state",
    "variance_kind",
    "block",
    "training",
    "model",
    "prediction_symbols",
    "digest",
}
_BLOCK_KEYS = {
    "fit_cutoff",
    "model_fit_time",
    "prediction_start",
    "prediction_stop",
    "inference_delay_seconds",
}
_TRAINING_KEYS = {
    "schema",
    "log_training",
    "labels",
    "return_unit",
    "valuation_basis",
    "variance_kind",
    "fit_prefix_marginal_variance",
}
_LOG_TRAINING_KEYS = {
    "schema",
    "feature_indices",
    "feature_names",
    "features",
    "labels",
    "sample_weights",
    "horizon_hours",
    "fit_cutoff",
    "trace",
}
_MODEL_KEYS = {
    "feature_indices",
    "feature_mean",
    "feature_scale",
    "coefficients",
    "intercept",
    "horizon_hours",
    "alpha",
    "n_samples",
    "fit_cutoff",
    "return_unit",
    "valuation_basis",
    "variance_kind",
    "fit_prefix_marginal_variance",
}


def _sha(value: object, *, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _ns(value: object, *, name: str) -> int:
    if type(value) is not int or not _MIN_NS <= value <= _MAX_NS:
        raise ValueError(f"{name} must be an ordinary signed nanosecond integer")
    return value


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an ordinary integer >= {minimum}")
    return value


def _finite(value: object, *, name: str, minimum: float | None = None) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be an ordinary finite number")
    try:
        result = float(cast(int | float, value))
    except OverflowError as error:
        raise ValueError(f"{name} exceeds finite binary64 range") from error
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{name} is outside its finite domain")
    return result


def _closed_dict(value: object, keys: set[str], *, name: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"{name} must have its exact closed field set")
    return cast(dict[str, object], value)


def _sequence(
    value: object, *, name: str, nonempty: bool = True
) -> list[object] | tuple[object, ...]:
    if type(value) not in (list, tuple):
        raise ValueError(f"{name} must be an already bounded list or tuple")
    result = cast(list[object] | tuple[object, ...], value)
    if nonempty and not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _names(value: object, *, name: str) -> tuple[str, ...]:
    items = _sequence(value, name=name)
    if any(type(item) is not str or not item.strip() for item in items):
        raise ValueError(f"{name} must contain nonempty strings")
    result = tuple(cast(str, item) for item in items)
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must be unique")
    return result


def _indices(value: object, *, name: str) -> tuple[int, ...]:
    items = _sequence(value, name=name)
    if any(type(item) is not int or item < 0 for item in items):
        raise ValueError(f"{name} must contain nonnegative ordinary integers")
    result = tuple(cast(int, item) for item in items)
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must be unique")
    return result


def _float_tuple(value: object, *, name: str) -> tuple[float, ...]:
    items = _sequence(value, name=name)
    return tuple(_finite(item, name=name) for item in items)


@dataclass(frozen=True, slots=True)
class OwnedDecisionSnapshot:
    """Caller-owned full Dataset row facts for a frozen forecast comparison."""

    dataset_id: str
    symbol: str
    decision_time_ns: int
    source_available_at_ns: int
    information_available: bool
    feature_names: tuple[str, ...]
    feature_values: tuple[float, ...]
    feature_available: tuple[bool, ...]
    dataset_close: float
    book_mark: float
    expected_horizon_seconds: int
    observation_feature_indices: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        _sha(self.dataset_id, name="dataset_id")
        if type(self.symbol) is not str or not self.symbol.strip():
            raise ValueError("symbol must be a nonempty string")
        decision = _ns(self.decision_time_ns, name="decision_time_ns")
        source_at = _ns(self.source_available_at_ns, name="source_available_at_ns")
        if source_at > decision:
            raise ValueError("source availability must not follow the decision clock")
        if type(self.information_available) is not bool:
            raise ValueError("information_available must be boolean")
        names = _names(self.feature_names, name="feature_names")
        values = _float_tuple(self.feature_values, name="feature_values")
        available = _sequence(self.feature_available, name="feature_available")
        if any(type(item) is not bool for item in available):
            raise ValueError("feature_available must contain booleans")
        if not (len(names) == len(values) == len(available)):
            raise ValueError(
                "full Dataset feature names, values and availability must align"
            )
        observation_indices = (
            tuple(range(len(names)))
            if self.observation_feature_indices is None
            else _indices(
                self.observation_feature_indices,
                name="observation_feature_indices",
            )
        )
        if any(index >= len(names) for index in observation_indices):
            raise ValueError(
                "observation feature index is outside the full Dataset row"
            )
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "feature_values", values)
        object.__setattr__(
            self, "feature_available", tuple(cast(bool, x) for x in available)
        )
        object.__setattr__(self, "observation_feature_indices", observation_indices)
        dataset_close = _finite(self.dataset_close, name="dataset_close", minimum=0.0)
        book_mark = _finite(self.book_mark, name="book_mark", minimum=0.0)
        if dataset_close <= 0 or book_mark <= 0:
            raise ValueError("Dataset close and book mark must be strictly positive")
        object.__setattr__(self, "dataset_close", dataset_close)
        object.__setattr__(self, "book_mark", book_mark)
        horizon = _integer(
            self.expected_horizon_seconds,
            name="expected_horizon_seconds",
            minimum=1,
        )
        if horizon > _MAX_NS // 1_000_000_000:
            raise ValueError("expected_horizon_seconds exceeds timestamp range")


def _packet_key(value: object) -> tuple[str, int]:
    if type(value) is not dict:
        raise ValueError("every packet row must be a parsed dictionary")
    row = cast(dict[str, object], value)
    symbol = row.get("symbol")
    if type(symbol) is not str or not symbol.strip():
        raise ValueError("packet symbol must be a nonempty string")
    return symbol, _ns(row.get("as_of"), name="packet.as_of")


def _validate_packet(value: object) -> dict[str, object]:
    packet = _closed_dict(value, _PACKET_KEYS, name="packet")
    if packet["schema"] != "simple_return_packet_v1":
        raise ValueError("unsupported packet schema")
    if packet["return_unit"] != "expected_simple_return":
        raise ValueError("packet return unit is not expected simple return")
    if packet["valuation_basis"] != "same_close_price_return":
        raise ValueError("packet valuation basis is not same-close")
    if packet["variance_kind"] != "fit_prefix_marginal_variance":
        raise ValueError("packet variance kind is unsupported")
    if packet["forecast_kind"] != "regularized_linear_projection":
        raise ValueError("packet forecast kind is unsupported")
    _sha(packet["digest"], name="packet.digest")
    _sha(packet["snapshot_digest"], name="packet.snapshot_digest")
    _sha(packet["vintage_digest"], name="packet.vintage_digest")
    for field in (
        "as_of",
        "source_available_at",
        "forecast_available_at",
        "horizon_end",
    ):
        _ns(packet[field], name=f"packet.{field}")
    _integer(packet["horizon_seconds"], name="packet.horizon_seconds", minimum=1)
    values = _float_tuple(packet["feature_values"], name="packet.feature_values")
    if not values:
        raise ValueError("packet selected features must not be empty")
    _finite(
        packet["expected_simple_return"],
        name="packet.expected_simple_return",
        minimum=-1.0,
    )
    _finite(
        packet["fit_prefix_marginal_variance"],
        name="packet.fit_prefix_marginal_variance",
        minimum=0.0,
    )
    decision_close = _finite(
        packet["decision_close"], name="packet.decision_close", minimum=0.0
    )
    if decision_close <= 0:
        raise ValueError("packet decision close must be strictly positive")
    return packet


def _validate_owner(
    value: object, *, source: OwnedDecisionSnapshot
) -> tuple[float, int, tuple[int, ...]]:
    vintage = _closed_dict(value, _VINTAGE_KEYS, name="vintage")
    if vintage["schema"] != "simple_return_ridge_vintage_v1":
        raise ValueError("unsupported vintage schema")
    if (
        vintage["return_unit"] != "expected_simple_return"
        or vintage["forecast_kind"] != "regularized_linear_projection"
        or vintage["valuation_basis"] != "same_close_price_return"
        or vintage["calibration_state"] != "uncalibrated"
        or vintage["variance_kind"] != "fit_prefix_marginal_variance"
    ):
        raise ValueError("vintage forecast declaration is unsupported")
    _sha(vintage["digest"], name="vintage.digest")

    block = _closed_dict(vintage["block"], _BLOCK_KEYS, name="vintage.block")
    cutoff = _ns(block["fit_cutoff"], name="block.fit_cutoff")
    model_fit = _ns(block["model_fit_time"], name="block.model_fit_time")
    prediction_start = _ns(block["prediction_start"], name="block.prediction_start")
    prediction_stop = _ns(block["prediction_stop"], name="block.prediction_stop")
    delay = _integer(
        block["inference_delay_seconds"], name="block.inference_delay_seconds"
    )
    if not cutoff <= model_fit <= prediction_start < prediction_stop:
        raise ValueError("vintage block clocks are not ordered")
    if delay != 0:
        raise ValueError("this adapter accepts only zero inference delay")

    training = _closed_dict(
        vintage["training"], _TRAINING_KEYS, name="vintage.training"
    )
    if (
        training["schema"] != "simple_return_training_scope_v1"
        or training["return_unit"] != "expected_simple_return"
        or training["valuation_basis"] != "same_close_price_return"
        or training["variance_kind"] != "fit_prefix_marginal_variance"
    ):
        raise ValueError("training scope declaration is unsupported")
    log_training = _closed_dict(
        training["log_training"], _LOG_TRAINING_KEYS, name="training.log_training"
    )
    if log_training["schema"] != "forecast_training_scope_v1":
        raise ValueError("log-training scope schema is unsupported")
    training_indices = _indices(
        log_training["feature_indices"], name="training.feature_indices"
    )
    training_names = _names(
        log_training["feature_names"], name="training.feature_names"
    )
    if len(training_indices) != len(training_names):
        raise ValueError("training feature indices and names do not align")
    training_horizon = _integer(
        log_training["horizon_hours"], name="training.horizon_hours", minimum=1
    )
    training_cutoff = _ns(log_training["fit_cutoff"], name="training.fit_cutoff")
    if training_cutoff != cutoff:
        raise ValueError("training and vintage cutoff differ")

    model = _closed_dict(vintage["model"], _MODEL_KEYS, name="vintage.model")
    model_indices = _indices(model["feature_indices"], name="model.feature_indices")
    if model_indices != training_indices:
        raise ValueError("model and training feature indices differ")
    if (
        _integer(model["horizon_hours"], name="model.horizon_hours", minimum=1)
        != training_horizon
    ):
        raise ValueError("model and training horizons differ")
    if _ns(model["fit_cutoff"], name="model.fit_cutoff") != cutoff:
        raise ValueError("model and vintage fit cutoff differ")
    means = _float_tuple(model["feature_mean"], name="model.feature_mean")
    scales = _float_tuple(model["feature_scale"], name="model.feature_scale")
    coefficients = _float_tuple(model["coefficients"], name="model.coefficients")
    if not (len(means) == len(scales) == len(coefficients) == len(model_indices)):
        raise ValueError("model vectors do not match its feature layout")
    if any(scale <= 0 for scale in scales):
        raise ValueError("model feature scales must be positive")
    _finite(model["intercept"], name="model.intercept")
    if _finite(model["alpha"], name="model.alpha", minimum=0.0) <= 0:
        raise ValueError("model alpha must be positive")
    _integer(model["n_samples"], name="model.n_samples", minimum=1)
    if (
        model["return_unit"] != "expected_simple_return"
        or model["valuation_basis"] != "same_close_price_return"
        or model["variance_kind"] != "fit_prefix_marginal_variance"
    ):
        raise ValueError("model unit/basis/variance declaration is unsupported")

    model_variance = _finite(
        model["fit_prefix_marginal_variance"],
        name="model.fit_prefix_marginal_variance",
        minimum=0.0,
    )
    training_variance = _finite(
        training["fit_prefix_marginal_variance"],
        name="training.fit_prefix_marginal_variance",
        minimum=0.0,
    )
    if model_variance != training_variance:
        raise ValueError("model and training variance differ")

    prediction_symbols = _names(
        vintage["prediction_symbols"], name="prediction_symbols"
    )
    if source.symbol not in prediction_symbols:
        raise ValueError("vintage does not cover the requested symbol")
    if any(index >= len(source.feature_names) for index in model_indices):
        raise ValueError("forecast feature index is outside the full Dataset row")
    expected_training_names = tuple(
        source.feature_names[index] for index in model_indices
    )
    if training_names != expected_training_names:
        raise ValueError(
            "training feature names differ from the indexed owned Dataset row"
        )
    return model_variance, training_horizon, model_indices


def bind_frozen_forecast_assumptions(
    stream_payload: object,
    *,
    source: OwnedDecisionSnapshot,
    cost_assumptions: AllocationAssumptions,
) -> AllocationAssumptions:
    """Join an already-parsed forecast to caller-owned decision-row facts.

    This narrow adapter maps only the packet simple return and fit-prefix
    marginal variance, preserving caller-provided cost fields. It neither parses
    nor pins stream bytes, recomputes/authenticates stream digests, fits/predicts
    a model, establishes Dataset provenance, or grants decision, execution,
    account, economic, research, or live authority.
    """
    if type(source) is not OwnedDecisionSnapshot:
        raise ValueError("requires an exact owned-decision snapshot")
    if type(cost_assumptions) is not AllocationAssumptions:
        raise ValueError("requires exact allocation cost assumptions")
    cost_assumptions.__post_init__()

    stream = _closed_dict(stream_payload, _STREAM_KEYS, name="stream")
    if stream["schema"] != "simple_return_ridge_stream_v1":
        raise ValueError("unsupported frozen stream schema")
    if stream["availability_semantics"] != "declared_simulation_v1":
        raise ValueError("unsupported stream availability semantics")
    if _sha(stream["dataset_id"], name="stream.dataset_id") != source.dataset_id:
        raise ValueError("stream and owned source Dataset identities differ")
    _sha(stream["causal_scope_digest"], name="stream.causal_scope_digest")

    packets = _sequence(stream["packets"], name="stream.packets")
    vintages = _sequence(stream["vintages"], name="stream.vintages")
    matches: list[dict[str, object]] = []
    for row in packets:
        symbol, as_of = _packet_key(row)
        if symbol == source.symbol and as_of == source.decision_time_ns:
            matches.append(cast(dict[str, object], row))
    if len(matches) != 1:
        raise ValueError("owned decision clock must resolve to exactly one packet")
    packet = _validate_packet(matches[0])

    owner_matches: list[dict[str, object]] = []
    for row in vintages:
        if type(row) is not dict:
            raise ValueError("every vintage row must be a parsed dictionary")
        digest = _sha(cast(dict[str, object], row).get("digest"), name="vintage.digest")
        if digest == packet["vintage_digest"]:
            owner_matches.append(cast(dict[str, object], row))
    if len(owner_matches) != 1:
        raise ValueError("packet must resolve to exactly one frozen vintage")
    owner_variance, owner_horizon_hours, forecast_indices = _validate_owner(
        owner_matches[0], source=source
    )

    decision = source.decision_time_ns
    owner_block = _closed_dict(
        owner_matches[0]["block"], _BLOCK_KEYS, name="vintage.block"
    )
    prediction_start = _ns(
        owner_block["prediction_start"], name="block.prediction_start"
    )
    prediction_stop = _ns(owner_block["prediction_stop"], name="block.prediction_stop")
    if not prediction_start <= decision < prediction_stop:
        raise ValueError(
            "owned decision clock is outside the vintage prediction window"
        )
    observation_indices = cast(tuple[int, ...], source.observation_feature_indices)
    required_feature_indices = set(forecast_indices) | set(observation_indices)
    if not source.information_available or any(
        not source.feature_available[index] for index in required_feature_indices
    ):
        raise ValueError(
            "owned source information or selected features are unavailable"
        )
    packet_as_of = _ns(packet["as_of"], name="packet.as_of")
    if packet_as_of != decision:
        raise ValueError("packet and owned decision clocks differ")
    if (
        _ns(packet["source_available_at"], name="packet.source_available_at")
        != source.source_available_at_ns
    ):
        raise ValueError("packet and owned source-availability clocks differ")
    if source.source_available_at_ns > decision:
        raise ValueError("owned source was not available by the decision clock")
    forecast_at = _ns(
        packet["forecast_available_at"], name="packet.forecast_available_at"
    )
    if forecast_at != decision:
        raise ValueError("forecast was not available at the owned decision clock")
    if (
        _integer(packet["horizon_seconds"], name="packet.horizon_seconds", minimum=1)
        != source.expected_horizon_seconds
        or owner_horizon_hours * 3600 != source.expected_horizon_seconds
    ):
        raise ValueError("packet, vintage and owned horizons differ")
    expected_end = decision + source.expected_horizon_seconds * 1_000_000_000
    if (
        expected_end > _MAX_NS
        or _ns(packet["horizon_end"], name="packet.horizon_end") != expected_end
    ):
        raise ValueError("packet horizon end is not the exact owned horizon")
    if (
        _ns(packet["source_available_at"], name="packet.source_available_at")
        > packet_as_of
        or forecast_at != packet_as_of
    ):
        raise ValueError("packet availability is not causal at the decision clock")

    packet_features = _float_tuple(
        packet["feature_values"], name="packet.feature_values"
    )
    expected_packet_features = tuple(
        source.feature_values[index] for index in forecast_indices
    )
    if packet_features != expected_packet_features:
        raise ValueError(
            "packet features differ from forecast-indexed values in the full Dataset row"
        )
    close = _finite(packet["decision_close"], name="packet.decision_close", minimum=0.0)
    if close != source.dataset_close or close != source.book_mark:
        raise ValueError("packet close differs from Dataset close or book mark")
    packet_variance = _finite(
        packet["fit_prefix_marginal_variance"],
        name="packet.fit_prefix_marginal_variance",
        minimum=0.0,
    )
    if packet_variance != owner_variance:
        raise ValueError("packet variance differs from its frozen vintage")
    expected = _finite(
        packet["expected_simple_return"],
        name="packet.expected_simple_return",
        minimum=-1.0,
    )
    return AllocationAssumptions(
        expected_simple_return=expected,
        return_variance=packet_variance,
        buy_cost=cost_assumptions.buy_cost,
        sell_cost=cost_assumptions.sell_cost,
        exit_cost=cost_assumptions.exit_cost,
        funding_return=cost_assumptions.funding_return,
        borrow_return=cost_assumptions.borrow_return,
        cash_return=cost_assumptions.cash_return,
    )


__all__ = ["OwnedDecisionSnapshot", "bind_frozen_forecast_assumptions"]
