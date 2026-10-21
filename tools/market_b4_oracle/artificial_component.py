"""Independent Q inputs for the fixed artificial H component comparison.

The module derives the component Dataset identity and fixed forecast from the
typed recipe. It never imports the artificial H operator or a native H builder.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, cast

import numpy as np

from tools.market_b4_oracle.artificial_spec import is_artificial_h_component_spec
from tools.market_b4_oracle.decision_identity import canonical_json_bytes
from tools.market_b4_oracle.frozen_market_replay import (
    FrozenMarketBinding,
    FrozenMarketPolicy,
    FrozenMarketWindow,
)
from tools.market_b4_oracle.output_index import NativeOutputIndex
from tools.market_replay_oracle import (
    ARRAY_FIELDS,
    VerifiedDatasetContent,
    canonical,
    content_id,
)

_SYMBOLS = ("BTCUSDT", "ETHUSDT")
_PULSE_INDICES = (2905, 3995, 5082, 5089, 6191, 7290, 7297, 8362, 9425, 9429)
_FEATURE_NAMES = (
    "fixture_pulse",
    "fixture_log_return",
    "fixture_body_return",
    "fixture_volume_change",
    "fixture_close_location",
    "fixture_lower_wick",
)
_GLOBAL_NAMES = (
    "active_fraction",
    "tradable_fraction",
    "market_return_mean",
    "market_return_dispersion",
)
_ARRAY_HASH_ORDER = (
    ("features", "features"),
    ("feature_available", "feature_available"),
    ("feature_age_hours", "feature_staleness_hours"),
    ("feature_staleness", "feature_staleness"),
)
_NANOSECONDS_PER_HOUR = 3_600_000_000_000
_FORECAST_MAXIMUM_BYTES = 64 * 1024**2


@dataclass(frozen=True, slots=True)
class ArtificialComponentInputs:
    """Root-derived fixed source and binding inputs for one artificial pair."""

    spec_sha256: str
    source_protocol_sha256: str
    content: VerifiedDatasetContent
    protocol_document_canonical: bytes
    forecast_payload_canonical: bytes

    @property
    def protocol_document(self) -> dict[str, object]:
        """Return a detached view; the canonical bytes remain the authority."""
        value = json.loads(self.protocol_document_canonical)
        if type(value) is not dict:
            raise ValueError("artificial protocol document is not an object")
        return cast(dict[str, object], value)

    @property
    def forecast_payload(self) -> dict[str, object]:
        """Return a detached view; the canonical bytes remain the authority."""
        value = json.loads(self.forecast_payload_canonical)
        if type(value) is not dict:
            raise ValueError("artificial forecast payload is not an object")
        return cast(dict[str, object], value)


def _sha(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _freeze_array(value: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(value)
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


def _freeze_metadata(value: object) -> object:
    """Match the nested immutable metadata returned by the pinned-v4 loader."""
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_metadata(child) for key, child in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_metadata(child) for child in value)
    return value


def _object_mapping(value: object, field: str) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(f"{field} must be an object")
    return cast(dict[str, object], value)


def _int_value(value: object, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field} must be an integer")
    return value


def _float_value(value: object, field: str) -> float:
    if type(value) is int:
        return float(value)
    if type(value) is float:
        return value
    raise ValueError(f"{field} must be numeric")


def _string_list(value: object, field: str) -> list[str]:
    if type(value) is not list or any(type(item) is not str for item in value):
        raise ValueError(f"{field} must be a string list")
    return cast(list[str], value)


def _integer_list(value: object, field: str) -> list[int]:
    if type(value) is not list or any(type(item) is not int for item in value):
        raise ValueError(f"{field} must be an integer list")
    return cast(list[int], value)


def _string_tuple(value: object, field: str) -> tuple[str, ...]:
    if type(value) not in (list, tuple) or any(
        type(item) is not str for item in cast(list[object] | tuple[object, ...], value)
    ):
        raise ValueError(f"{field} must be a string sequence")
    sequence = cast(list[str] | tuple[str, ...], value)
    return tuple(sequence)


def _named_arrays_digest(metadata: object, arrays: Mapping[str, np.ndarray]) -> str:
    """Match the Q identity byte protocol for the builder's normalization pin."""
    digest = hashlib.sha256(canonical(metadata))
    for name, source_name in _ARRAY_HASH_ORDER:
        source = arrays[source_name]
        if source.dtype.kind == "b":
            normalized = source.astype("|u1", order="C")
        elif source.dtype.kind == "f":
            normalized = source.astype("<f8", order="C")
        elif source.dtype.kind in "iu":
            normalized = source.astype("<i8", order="C")
        else:
            raise ValueError("normalization identity contains an unsupported dtype")
        descriptor = canonical(
            {
                "name": name,
                "dtype": normalized.dtype.str,
                "shape": list(source.shape),
            }
        )
        for block in (descriptor, normalized.tobytes(order="C")):
            digest.update(len(block).to_bytes(8, "big"))
            digest.update(block)
    return digest.hexdigest()


def _feature_config(spec: Mapping[str, object]) -> dict[str, object]:
    names = spec["feature_names"]
    recipe = spec["dataset_recipe"]
    if type(names) is not list or type(recipe) is not dict:
        raise ValueError("typed artificial source recipe is malformed")
    kinds = recipe.get("feature_kinds")
    if type(kinds) is not list or len(kinds) != len(names):
        raise ValueError("typed artificial feature recipe is malformed")
    features = []
    for name, kind in zip(names, kinds, strict=True):
        if type(name) is not str or type(kind) is not str:
            raise ValueError("typed artificial feature recipe is malformed")
        features.append(
            {
                "name": name,
                "kind": kind.lower(),
                "lookback": 1,
                "normalization": "none",
                "normalization_window": 1,
                "min_periods": 1,
                "max_staleness_hours": 24.0,
            }
        )
    return {
        "base_timeframe": "1h",
        "bar_hours": 1.0,
        "calendar_kind": "continuous_24_7",
        "session_periods_per_year": None,
        "cross_asset_reference_symbol": None,
        "features": features,
        "global_feature_names": list(_GLOBAL_NAMES),
        "feature_numerics_schema": "portable_feature_numerics_v1",
        "schema_version": "market_build_v3",
    }


def _build_source(spec: Mapping[str, object]) -> VerifiedDatasetContent:
    dataset_recipe = spec.get("dataset_recipe")
    cost_recipe = spec.get("cost_recipe")
    account_recipe = spec.get("account_recipe")
    if (
        type(dataset_recipe) is not dict
        or type(cost_recipe) is not dict
        or type(account_recipe) is not dict
        or spec.get("row_count") != 9504
        or spec.get("actor_start") != 2903
        or spec.get("actor_stop_exclusive") != 9430
        or spec.get("actor_advances") != 6527
        or spec.get("reserve_advances") != 72
        or spec.get("scenario") != "BASEL0"
    ):
        raise ValueError("typed artificial component source is outside fixed scope")
    recipe = _object_mapping(dataset_recipe, "dataset_recipe")
    costs = _object_mapping(cost_recipe, "cost_recipe")
    rules = _object_mapping(recipe.get("instrument_rules"), "instrument_rules")
    ohlc = _object_mapping(recipe.get("ohlc"), "ohlc")
    if tuple(_string_list(recipe.get("symbols"), "dataset_recipe.symbols")) != _SYMBOLS:
        raise ValueError("typed artificial symbol roster differs from the fixed source")
    row_count = 9504
    start_ns = np.datetime64("2024-01-01T00:00:00", "ns").astype(np.int64)
    timestamps = (
        start_ns
        + np.arange(row_count, dtype=np.int64)
        * _int_value(recipe["interval_seconds"], "interval_seconds")
        * 1_000_000_000
    ).astype("datetime64[ns]")
    shape = (row_count, 2)
    active = np.ones(shape, dtype=np.bool_)
    open_price = np.full(
        shape, _float_value(ohlc["open"], "ohlc.open"), dtype=np.float64
    )
    high = np.full(
        shape,
        _float_value(ohlc["ordinary_high"], "ohlc.ordinary_high"),
        dtype=np.float64,
    )
    low = np.full(
        shape, _float_value(ohlc["ordinary_low"], "ohlc.ordinary_low"), dtype=np.float64
    )
    close = np.full(shape, _float_value(ohlc["close"], "ohlc.close"), dtype=np.float64)
    pulse = np.asarray(_PULSE_INDICES, dtype=np.int64)
    high[pulse, 0] = _float_value(
        ohlc["selected_pulse_high"], "ohlc.selected_pulse_high"
    )
    low[pulse, 0] = _float_value(ohlc["selected_pulse_low"], "ohlc.selected_pulse_low")
    volume = np.full(
        shape,
        _float_value(recipe["quote_volume_per_bar"], "quote_volume_per_bar"),
        dtype=np.float64,
    )
    feature_values = np.zeros((row_count, 2, 6), dtype=np.float64)
    feature_values[:, :, 0] = (high - low) / close
    feature_values[:, :, 4] = (2.0 * close - high - low) / (high - low)
    feature_values[:, :, 5] = (np.minimum(open_price, close) - low) / (high - low)
    features = feature_values.astype(np.float32)
    feature_available = np.ones(features.shape, dtype=np.bool_)
    feature_available[0, :, 1] = False
    feature_available[0, :, 3] = False
    feature_age_hours = np.zeros(features.shape, dtype=np.float32)
    feature_age_hours[0, :, 1] = 24.0
    feature_age_hours[0, :, 3] = 24.0
    feature_staleness = np.where(feature_available, 0.0, 1.0).astype(np.float32)
    global_features = np.zeros((row_count, len(_GLOBAL_NAMES)), dtype=np.float32)
    global_features[:, 0:2] = 1.0
    global_available = np.ones(global_features.shape, dtype=np.bool_)
    global_available[0, 2:4] = False
    global_staleness_hours = np.zeros(global_features.shape, dtype=np.float32)
    global_staleness_hours[0, 2:4] = 1.0
    global_missing_reason = np.zeros(global_features.shape, dtype=np.int16)
    global_missing_reason[0, 2:4] = 1
    available_at = np.broadcast_to(timestamps[:, None], shape).copy()
    event_count = np.zeros(shape, dtype=np.int32)
    event_count[
        :: _int_value(recipe["funding_event_stride"], "funding_event_stride"), :
    ] = 1
    funding = np.full(
        shape, _float_value(recipe["funding_rate"], "funding_rate"), dtype=np.float64
    )
    funding_price = np.full(
        shape,
        _float_value(recipe["funding_price_rate"], "funding_price_rate"),
        dtype=np.float64,
    )
    tick = np.empty(shape, dtype=np.float64)
    lot = np.empty(shape, dtype=np.float64)
    minimum = np.empty(shape, dtype=np.float64)
    for symbol_index, symbol in enumerate(_SYMBOLS):
        rule = _object_mapping(rules[symbol], f"instrument_rules.{symbol}")
        tick[:, symbol_index] = _float_value(rule["tick_size"], "tick_size")
        lot[:, symbol_index] = _float_value(rule["lot_size"], "lot_size")
        minimum[:, symbol_index] = _float_value(
            rule["minimum_notional"], "minimum_notional"
        )
    zero = np.zeros(shape, dtype=np.float64)
    arrays: dict[str, np.ndarray] = {
        "timestamps": timestamps,
        "available_at": available_at,
        "information_available": active.copy(),
        "features": features,
        "feature_available": feature_available,
        "feature_staleness": feature_staleness,
        "feature_staleness_hours": feature_age_hours,
        "feature_missing_reason": np.zeros(features.shape, dtype=np.int16),
        "global_features": global_features,
        "global_feature_available": global_available,
        "global_feature_staleness_hours": global_staleness_hours,
        "global_feature_missing_reason": global_missing_reason,
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "funding_rate": funding,
        "funding_price_rate": funding_price,
        "funding_event_count": event_count,
        "tradable": np.full(shape, recipe["tradable"] is True, dtype=np.bool_),
        "symbol_active": active,
        "fee_rate": np.full(
            shape, _float_value(costs["fee_rate"], "fee_rate"), dtype=np.float64
        ),
        "maker_fee_rate": zero.copy(),
        "taker_fee_rate": zero.copy(),
        "spread_rate": np.full(
            shape, _float_value(costs["spread_rate"], "spread_rate"), dtype=np.float64
        ),
        "max_participation_rate": np.full(
            shape,
            _float_value(costs["max_participation_rate"], "max_participation_rate"),
            dtype=np.float64,
        ),
        "minimum_notional": minimum,
        "lot_size": lot,
        "tick_size": tick,
        "borrow_available": np.full(
            shape, costs["borrow_available"] is True, dtype=np.bool_
        ),
        "borrow_rate": np.full(
            shape, _float_value(costs["borrow_rate"], "borrow_rate"), dtype=np.float64
        ),
        "funding_due": event_count > 0,
        "buy_allowed": active.copy(),
        "sell_allowed": active.copy(),
        "mark_price": close.copy(),
        "index_price": close.copy(),
        "dividend": zero.copy(),
        "split_factor": np.ones(shape, dtype=np.float64),
        "delisting_recovery": np.ones(shape, dtype=np.float64),
        "cash_rate": np.zeros(row_count, dtype=np.float64),
        "contract_multipliers": np.ones(2, dtype=np.float64),
        # The v4 artifact stores this compatibility alias outside ARRAY_FIELDS;
        # its loader requires exact equality with symbol_active.
        "asset_active": active.copy(),
    }
    if set(arrays) != {*ARRAY_FIELDS, "asset_active"}:
        raise ValueError("Q source array roster differs from the identity protocol")
    arrays = {name: _freeze_array(value) for name, value in arrays.items()}
    feature_names = tuple(_string_list(spec.get("feature_names"), "feature_names"))
    config_payload = _feature_config(spec)
    feature_config_digest = hashlib.sha256(canonical(config_payload)).hexdigest()
    normalization_digest = _named_arrays_digest(
        {
            "schema": "normalization_state_v1",
            "feature_config_digest": feature_config_digest,
        },
        arrays,
    )
    instruments: list[dict[str, object]] = []
    for symbol in _SYMBOLS:
        rule = _object_mapping(rules[symbol], f"instrument_rules.{symbol}")
        instruments.append(
            {
                "symbol": symbol,
                "listed_at": "2024-01-01T00:00:00+00:00",
                "delisted_at": None,
                "volume_unit": "quote_notional",
                "contract_multiplier": 1.0,
                "tick_size": _float_value(rule["tick_size"], "tick_size"),
                "lot_size": _float_value(rule["lot_size"], "lot_size"),
                "minimum_notional": _float_value(
                    rule["minimum_notional"], "minimum_notional"
                ),
                "execution_rules": [],
            }
        )
    economics_payload = {
        "schema_version": "execution_economics_profile_v1",
        "name": "Artificial Task18-shaped component economics",
        "fee_rate": _float_value(costs["fee_rate"], "fee_rate"),
        "maker_fee_rate": 0.0,
        "taker_fee_rate": 0.0,
        "spread_rate": _float_value(costs["spread_rate"], "spread_rate"),
        "max_participation_rate": _float_value(
            costs["max_participation_rate"], "max_participation_rate"
        ),
        "borrow_available": costs["borrow_available"] is True,
        "borrow_rate": _float_value(costs["borrow_rate"], "borrow_rate"),
    }
    identity: dict[str, object] = {
        "schema": "market_dataset_identity_v7",
        "config": config_payload,
        "feature_config_digest": feature_config_digest,
        "normalization_digest": normalization_digest,
        "symbols": list(_SYMBOLS),
        "instruments": instruments,
        "feature_names": list(feature_names),
        "global_feature_names": list(_GLOBAL_NAMES),
        "execution_economics": economics_payload,
    }
    contract = {
        "calendar_kind": "continuous_24_7",
        "feature_config_digest": feature_config_digest,
        "feature_names": list(feature_names),
        "global_feature_names": list(_GLOBAL_NAMES),
        "nominal_bar_hours": 1.0,
        "normalization_digest": normalization_digest,
        "periods_per_year": 8760,
        "symbols": list(_SYMBOLS),
        "volume_units": ["quote_notional", "quote_notional"],
    }
    identity["dataset_contract"] = contract
    dataset_id = content_id(identity, arrays)
    identity_payload_json = canonical(identity).decode("utf-8")
    dataset: dict[str, object] = {
        **contract,
        "dataset_id": dataset_id,
        "identity_payload_json": identity_payload_json,
    }
    frozen_dataset = _freeze_metadata(dataset)
    if not isinstance(frozen_dataset, Mapping):
        raise ValueError("synthetic Dataset contract could not be frozen")
    return VerifiedDatasetContent(
        dataset_id,
        MappingProxyType(arrays),
        cast(Mapping[str, object], frozen_dataset),
    )


def _native_h_protocol(
    spec: Mapping[str, object], content: VerifiedDatasetContent
) -> dict[str, object]:
    dataset_recipe = _object_mapping(spec.get("dataset_recipe"), "dataset_recipe")
    costs = _object_mapping(spec.get("cost_recipe"), "cost_recipe")
    risk = _object_mapping(spec.get("risk_recipe"), "risk_recipe")
    allocator = _object_mapping(spec.get("allocator_recipe"), "allocator_recipe")
    forecast = _object_mapping(spec.get("forecast_recipe"), "forecast_recipe")
    execution = _object_mapping(spec.get("execution_recipe"), "execution_recipe")
    account = _object_mapping(spec.get("account_recipe"), "account_recipe")
    timestamps = content.arrays["timestamps"]
    last_exclusive = timestamps[-1] + np.timedelta64(
        _int_value(dataset_recipe["interval_seconds"], "interval_seconds"), "s"
    )
    return {
        "schema": "issue810_artificial_h_component_protocol_v1",
        "scope": "ARTIFICIAL_COMPARISON_ONLY",
        "protocol_id": spec["protocol_id"],
        "dataset": {
            "dataset_id": content.dataset_id,
            "dataset_label": spec["dataset_label"],
            "identity_binding": spec["dataset_identity_binding"],
            "dimensions": [9504, 2, 6],
            "feature_names": _string_list(spec.get("feature_names"), "feature_names"),
            "fit_symbols": [str(spec["symbol"])],
            "prediction_symbols": _string_list(
                dataset_recipe.get("symbols"), "dataset_recipe.symbols"
            ),
            "start": str(timestamps[0]),
            "stop_exclusive": str(last_exclusive),
            "source": "in_memory_synthetic_bars_no_external_dataset_read",
        },
        "configuration": {
            "id": spec["configuration_id"],
            "horizon_hours": forecast["horizon_hours"],
            "ridge_alpha": forecast["ridge_alpha"],
            "forecast_origin": "predeclared_fixture_no_fit",
        },
        "account": f"{spec['configuration_id']}_{spec['symbol']}",
        "capital": {
            "currency": "USDT",
            "mode": "independent_symbol",
            "per_account": account["initial_cash"],
            "initial_quantities": account["initial_quantities"],
        },
        "scenario": spec["scenario"],
        "roles": [
            value.lower() for value in _string_list(spec.get("role_pair"), "role_pair")
        ],
        "actor": {
            "start": _int_value(spec["actor_start"], "actor_start"),
            "stop": _int_value(spec["actor_stop_exclusive"], "actor_stop_exclusive"),
            "rows": _int_value(spec["actor_advances"], "actor_advances"),
            "vintage_starts": _integer_list(
                forecast.get("vintage_starts"), "forecast_recipe.vintage_starts"
            ),
        },
        "terminal": {
            "stop_inclusive": _int_value(
                execution["reserve_stop_inclusive"], "reserve_stop_inclusive"
            ),
            "advances": _int_value(spec["reserve_advances"], "reserve_advances"),
        },
        "economics": {
            "dataset_fee": costs["fee_rate"],
            "dataset_spread": costs["spread_rate"],
            "impact_coefficient": costs["impact_coefficient"],
            "quote_notional_volume": dataset_recipe["quote_volume_per_bar"],
            "borrow_available": costs["borrow_available"],
            "borrow_rate": costs["borrow_rate"],
            "cash_rate": costs["cash_rate"],
            "funding_rate": costs["funding_rate"],
            "funding_event_interval_hours": _int_value(
                dataset_recipe["funding_event_stride"], "funding_event_stride"
            ),
        },
        "policy": {
            "allocator": {
                "lower_weight": allocator["lower_weight"],
                "upper_weight": allocator["upper_weight"],
                "max_turnover": allocator["max_turnover"],
                "risk_aversion": allocator["risk_aversion"],
            },
            "risk": {
                "max_gross": risk["max_gross"],
                "max_abs_weight": risk["max_abs_weight"],
                "max_turnover": None,
                "drawdown_start": risk["drawdown_start"],
                "drawdown_stop": risk["drawdown_stop"],
                "emergency_turnover_override": risk["emergency_turnover_override"],
                "fail_closed_tolerance": risk["fail_closed_tolerance"],
            },
        },
        "execution": {
            "kind": f"{execution['order_type']}/{execution['time_in_force']}",
            "latency_bars": execution["latency_bars"],
            "capacity": execution["capacity_source"],
            "max_participation_rate": costs["max_participation_rate"],
            "market_fee_overlay": costs["market_fee_overlay"],
            "market_spread_overlay": costs["market_spread_overlay"],
            "impact_coefficient": costs["impact_coefficient"],
            "random_seed": costs["random_seed"],
            "stochastic_slippage": costs["stochastic_slippage"],
            "decision_mark_source": execution["decision_mark_source"],
            "market_fill_price_source": execution["market_fill_price_source"],
        },
        "component_spec_sha256": hashlib.sha256(canonical(spec)).hexdigest(),
        "full_b4_authorized": False,
        "outer_attempt_authenticated": False,
        "research_authorized": False,
    }


def _stream_packet_digest(value: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _expected_forecast_payload(
    spec: Mapping[str, object], content: VerifiedDatasetContent
) -> dict[str, object]:
    recipe = _object_mapping(spec.get("forecast_recipe"), "forecast_recipe")
    starts = tuple(_integer_list(recipe.get("vintage_starts"), "vintage_starts"))
    stop = _int_value(spec["actor_stop_exclusive"], "actor_stop_exclusive")
    horizon = _int_value(recipe["horizon_hours"], "horizon_hours")
    feature_indices = tuple(
        _integer_list(recipe.get("feature_indices"), "feature_indices")
    )
    prediction_symbols = tuple(
        _string_list(recipe.get("prediction_symbols"), "prediction_symbols")
    )
    symbols = _string_tuple(content.dataset.get("symbols"), "Dataset symbols")
    timestamps = content.arrays["timestamps"].astype("datetime64[ns]").astype("int64")
    features = content.arrays["features"]
    close = content.arrays["close"]
    avail = content.arrays["available_at"].astype("datetime64[ns]").astype("int64")
    if (
        symbols != _SYMBOLS
        or prediction_symbols != symbols
        or feature_indices != tuple(range(6))
    ):
        raise ValueError("typed forecast fit/prediction roster differs from Q source")
    vintages: list[dict[str, object]] = []
    packets: list[dict[str, object]] = []
    for vintage_index, start in enumerate(starts):
        block_stop = (
            starts[vintage_index + 1] if vintage_index + 1 < len(starts) else stop
        )
        cutoff = int(timestamps[start]) - 1
        sample_count = start - 25
        training_rows = range(1, start - 24)
        if len(training_rows) != sample_count:
            raise ValueError("typed forecast training range is inconsistent")
        train_features = [
            [float(value) for value in features[index, 0].astype(np.float64)]
            for index in training_rows
        ]
        training_starts = [int(timestamps[index]) for index in training_rows]
        training_ends = [
            value + horizon * _NANOSECONDS_PER_HOUR for value in training_starts
        ]
        zeros = [0.0] * sample_count
        ones = [1.0] * sample_count
        training_scope = {
            "schema": "forecast_training_scope_v1",
            "feature_indices": list(feature_indices),
            "feature_names": list(_FEATURE_NAMES),
            "features": train_features,
            "labels": zeros,
            "sample_weights": ones,
            "horizon_hours": horizon,
            "fit_cutoff": cutoff,
            "trace": {
                "schema": "forecast_training_trace_v1",
                "row_symbols": [symbols[0]] * sample_count,
                "start_times": training_starts,
                "end_times": training_ends,
                "start_available_at": training_starts,
                "end_available_at": training_ends,
                "start_close": [float(close[index, 0]) for index in training_rows],
                "end_close": [
                    float(close[index + horizon, 0]) for index in training_rows
                ],
            },
        }
        training = {
            "schema": "simple_return_training_scope_v1",
            "log_training": training_scope,
            "labels": zeros,
            "return_unit": "expected_simple_return",
            "valuation_basis": "same_close_price_return",
            "variance_kind": "fit_prefix_marginal_variance",
            "fit_prefix_marginal_variance": 0.0,
        }
        mean = [float(features[start, 0, 0]), 0.0, 0.0, 0.0, 0.0, 0.0]
        pulse_value = float(features[_PULSE_INDICES[0], 0, 0])
        intercept = _float_value(recipe["intercept"], "intercept")
        pulse_return = _float_value(
            recipe["pulse_expected_simple_return"], "pulse_expected_simple_return"
        )
        nonpulse_return = _float_value(
            recipe["nonpulse_expected_simple_return"],
            "nonpulse_expected_simple_return",
        )
        coefficient = (pulse_return - intercept) / (pulse_value - mean[0])
        if intercept != nonpulse_return:
            raise ValueError("fixed projection does not reproduce the nonpulse return")
        coefficients = [coefficient, 0.0, 0.0, 0.0, 0.0, 0.0]
        model = {
            "feature_indices": list(feature_indices),
            "feature_mean": mean,
            "feature_scale": [1.0] * 6,
            "coefficients": coefficients,
            "intercept": intercept,
            "horizon_hours": horizon,
            "alpha": _float_value(recipe["ridge_alpha"], "ridge_alpha"),
            "n_samples": sample_count,
            "fit_cutoff": cutoff,
            "return_unit": "expected_simple_return",
            "valuation_basis": "same_close_price_return",
            "variance_kind": "fit_prefix_marginal_variance",
            "fit_prefix_marginal_variance": 0.0,
        }
        block = {
            "fit_cutoff": cutoff,
            "model_fit_time": int(timestamps[start]),
            "prediction_start": int(timestamps[start]),
            "prediction_stop": int(timestamps[block_stop]),
            "inference_delay_seconds": 0,
        }
        vintage_content: dict[str, object] = {
            "schema": "simple_return_ridge_vintage_v1",
            "return_unit": "expected_simple_return",
            "forecast_kind": "regularized_linear_projection",
            "valuation_basis": "same_close_price_return",
            "calibration_state": "uncalibrated",
            "variance_kind": "fit_prefix_marginal_variance",
            "block": block,
            "training": training,
            "model": model,
            "prediction_symbols": list(prediction_symbols),
        }
        vintage_digest = _stream_packet_digest(vintage_content)
        vintages.append({**vintage_content, "digest": vintage_digest})
        for index in range(start, block_stop):
            for symbol_index, symbol in enumerate(symbols):
                row_features = [
                    float(value)
                    for value in features[index, symbol_index].astype(np.float64)
                ]
                standardized = np.asarray(row_features, dtype=np.float64) - np.asarray(
                    mean, dtype=np.float64
                )
                expected_return = float(
                    _float_value(recipe["intercept"], "intercept")
                    + standardized @ np.asarray(coefficients, dtype=np.float64)
                )
                source_ns = int(avail[index, symbol_index])
                snapshot = {
                    "symbol": symbol,
                    "as_of": int(timestamps[index]),
                    "source_available_at": source_ns,
                    "feature_values": row_features,
                    "decision_close": float(close[index, symbol_index]),
                }
                packet_content: dict[str, object] = {
                    "schema": "simple_return_packet_v1",
                    **snapshot,
                    "snapshot_digest": _stream_packet_digest(snapshot),
                    "forecast_available_at": max(
                        int(timestamps[index]), int(timestamps[start]), source_ns
                    ),
                    "horizon_end": int(timestamps[index])
                    + horizon * _NANOSECONDS_PER_HOUR,
                    "horizon_seconds": horizon * 3600,
                    "expected_simple_return": expected_return,
                    "fit_prefix_marginal_variance": 0.0,
                    "vintage_digest": vintage_digest,
                    "return_unit": "expected_simple_return",
                    "valuation_basis": "same_close_price_return",
                    "variance_kind": "fit_prefix_marginal_variance",
                    "forecast_kind": "regularized_linear_projection",
                }
                packets.append(
                    {
                        **packet_content,
                        "digest": _stream_packet_digest(packet_content),
                    }
                )
    causal_content: dict[str, object] = {"vintages": vintages, "packets": packets}
    return {
        "schema": "simple_return_ridge_stream_v1",
        "availability_semantics": "declared_simulation_v1",
        "dataset_id": content.dataset_id,
        **causal_content,
        "causal_scope_digest": _stream_packet_digest(causal_content),
    }


def build_artificial_component_inputs(
    spec: Mapping[str, object],
) -> ArtificialComponentInputs:
    """Derive exact Q source identity, H protocol pin, and forecast from spec."""
    if type(spec) is not dict or not is_artificial_h_component_spec(spec):
        raise ValueError("artificial component requires a parsed exact spec object")
    spec_sha = hashlib.sha256(canonical(spec)).hexdigest()
    content = _build_source(spec)
    h_protocol = _native_h_protocol(spec, content)
    source_protocol_sha = hashlib.sha256(canonical(h_protocol)).hexdigest()
    q_document: dict[str, object] = {
        "schema": "issue810_artificial_h_component_q_binding_v1",
        "scope": "ARTIFICIAL_COMPARISON_ONLY",
        "dataset": {
            "dataset_id": content.dataset_id,
            "symbols": list(_SYMBOLS),
            "fit_symbols": ["BTCUSDT"],
            "prediction_symbols": list(_SYMBOLS),
            "feature_names": list(_FEATURE_NAMES),
        },
        "configurations": [
            {
                "id": "H24_RIDGE_A1",
                "horizon_hours": 24,
                "ridge_alpha": 1.0,
            }
        ],
        "accounts": ["H24_RIDGE_A1_BTCUSDT"],
        "scenarios": ["BASEL0"],
        "roles": ["nonrl", "cash"],
        "actor": {
            "start": 2903,
            "stop": 9430,
            "vintage_starts": [2903, 5087, 7295],
        },
        "terminal": {"stop_inclusive": 9502},
        "economics": {"reference_notional": 100000.0},
        "funding": {
            "coverage_bounds": {
                "lookback_hours": 24,
                "minimum_counted_events": 2,
                "maximum_counted_close_gap_hours": 9.0,
                "maximum_counted_close_age_hours": 8.0,
            }
        },
        "capital": {"per_account": 100000.0},
        "policy": {
            "risk": {
                "max_gross": 1.0,
                "max_abs_weight": 1.0,
                "max_turnover": None,
                "drawdown_start": 0.1,
                "drawdown_stop": 0.2,
                "emergency_turnover_override": True,
                "fail_closed_tolerance": 1e-10,
            }
        },
        "component_spec_sha256": spec_sha,
        "source_protocol_sha256": source_protocol_sha,
        "outer_attempt_authenticated": False,
        "full_b4_authorized": False,
        "research_authorized": False,
    }
    expected_forecast = _expected_forecast_payload(spec, content)
    return ArtificialComponentInputs(
        spec_sha,
        source_protocol_sha,
        content,
        canonical_json_bytes(q_document),
        canonical_json_bytes(expected_forecast),
    )


def artificial_component_binding(
    inputs: ArtificialComponentInputs, role: Literal["NONRL", "CASH"]
) -> tuple[FrozenMarketBinding, FrozenMarketPolicy]:
    """Create one Q-owned arm binding from the fixed recipe declarations."""
    if type(inputs) is not ArtificialComponentInputs or role not in {"NONRL", "CASH"}:
        raise ValueError("artificial component binding requires exact pair inputs")
    native_role = role.lower()
    protocol_document = inputs.protocol_document
    binding = FrozenMarketBinding(
        protocol_canonical=inputs.protocol_document_canonical,
        account_id="H24_RIDGE_A1_BTCUSDT",
        configuration_id="H24_RIDGE_A1",
        scenario="BASEL0",
        native_role=native_role,
        selected_symbol_index=0,
        expected_horizon_seconds=24 * 3600,
        observation_feature_indices=tuple(range(len(_FEATURE_NAMES))),
        window=FrozenMarketWindow(2903, 9430, 9502),
        comparison_only=True,
    )
    policy = FrozenMarketPolicy.from_effective_profile(
        dataset_id=inputs.content.dataset_id,
        profile=binding.effective_profile,
        risk=cast(
            dict[str, object],
            cast(dict[str, object], protocol_document["policy"])["risk"],
        ),
    )
    binding.assert_content(inputs.content)
    return binding, policy


def verify_artificial_component_forecast(
    *,
    output_index: NativeOutputIndex,
    inputs: ArtificialComponentInputs,
) -> str:
    """Require exact fixed recipe bytes, not merely causal shape or digests."""
    if type(output_index) is not NativeOutputIndex:
        raise ValueError("artificial forecast requires a pinned output index")
    raw = output_index.read_pinned_bytes(
        "forecasts/simple-return-stream.json", maximum=_FORECAST_MAXIMUM_BYTES
    )
    expected = inputs.forecast_payload_canonical
    if raw != expected:
        try:
            observed = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(
                "artificial forecast is not valid fixed-recipe JSON"
            ) from error
        if observed != inputs.forecast_payload:
            raise ValueError(
                "artificial forecast model, trace or packet differs from Q's fixed recipe"
            )
        raise ValueError("artificial forecast is not canonical fixed-recipe JSON")
    return hashlib.sha256(raw).hexdigest()
