"""Build the fixed, in-memory fee-only synthetic Dataset for Issue #810."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

from trade_rl.data.contracts import MarketCalendarKind, VolumeUnit
from trade_rl.data.market import MarketDataset

_FIXTURE_SCHEMA = "issue810_joint_fee_only_synthetic_dataset_v1"
_ROW_COUNT = 37
_SYMBOLS = ("AAA", "BBB")
_FEATURE_NAMES = ("synthetic_zero",)
_GLOBAL_FEATURE_NAMES = ("synthetic_global_zero",)


def build_joint_fee_only_fixture() -> tuple[MarketDataset, Mapping[str, object]]:
    """Return the content-identified Dataset and its fixed read-only recipe.

    This function only constructs deterministic in-memory synthetic inputs. Dataset
    cost arrays are zero; any later execution-cost configuration remains a separate
    caller-owned harness choice.
    """

    symbol_count = len(_SYMBOLS)
    price_shape = (_ROW_COUNT, symbol_count)
    feature_shape = (_ROW_COUNT, symbol_count, len(_FEATURE_NAMES))
    global_shape = (_ROW_COUNT, len(_GLOBAL_FEATURE_NAMES))
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        _ROW_COUNT, dtype=np.int64
    ) * np.timedelta64(1, "h")
    available_at = np.broadcast_to(timestamps[:, None], price_shape).copy()
    prices = np.full(price_shape, 100.0, dtype=np.float64)
    zero_prices = np.zeros(price_shape, dtype=np.float64)
    one_prices = np.ones(price_shape, dtype=np.float64)
    true_prices = np.ones(price_shape, dtype=np.bool_)
    false_prices = np.zeros(price_shape, dtype=np.bool_)
    features = np.zeros(feature_shape, dtype=np.float32)
    global_features = np.zeros(global_shape, dtype=np.float32)
    true_features = np.ones(feature_shape, dtype=np.bool_)
    true_global_features = np.ones(global_shape, dtype=np.bool_)
    zero_features = np.zeros(feature_shape, dtype=np.float32)
    zero_global_features = np.zeros(global_shape, dtype=np.float32)

    recipe: dict[str, object] = {
        "schema": _FIXTURE_SCHEMA,
        "source": "tools/issue810_joint_fee_only_fixture.py",
        "construction": "deterministic_in_memory_synthetic_only",
        "timestamp_start_utc": "2026-01-01T00:00:00Z",
        "timestamp_stop_utc_inclusive": "2026-01-02T12:00:00Z",
        "row_count": _ROW_COUNT,
        "bar_interval_seconds": 3_600,
        "calendar_kind": MarketCalendarKind.CONTINUOUS.value,
        "nominal_bar_hours": 1.0,
        "periods_per_year": 8_760,
        "symbols": _SYMBOLS,
        "feature_names": _FEATURE_NAMES,
        "global_feature_names": _GLOBAL_FEATURE_NAMES,
        "feature_units": ("dimensionless",),
        "global_feature_units": ("dimensionless",),
        "feature_dtype": "float32",
        "global_feature_dtype": "float32",
        "feature_value": 0.0,
        "global_feature_value": 0.0,
        "price_fields": ("open", "high", "low", "close", "mark_price", "index_price"),
        "price_value": 100.0,
        "index_price_value": 100.0,
        "volume": 1_000.0,
        "volume_units": tuple(VolumeUnit.BASE_ASSET.value for _ in _SYMBOLS),
        "contract_multipliers": tuple(1.0 for _ in _SYMBOLS),
        "all_availability_true": True,
        "available_at_matches_timestamps": True,
        "all_feature_staleness_hours": 0.0,
        "all_global_feature_staleness_hours": 0.0,
        "all_feature_staleness": 0.0,
        "feature_missing_reason": 0,
        "global_feature_missing_reason": 0,
        "dataset_fee_rate": 0.0,
        "dataset_maker_fee_rate": 0.0,
        "dataset_taker_fee_rate": 0.0,
        "dataset_spread_rate": 0.0,
        "dataset_funding_rate": 0.0,
        "funding_due": False,
        "funding_event_count": 0,
        "dataset_borrow_rate": 0.0,
        "dataset_cash_rate": 0.0,
        "dataset_dividend": 0.0,
        "split_factor": 1.0,
        "delisting_recovery": 1.0,
        "borrow_available": True,
        "tradable": True,
        "symbol_active": True,
        "buy_allowed": True,
        "sell_allowed": True,
        "dataset_max_participation_rate": 1.0,
        "minimum_notional": 0.0,
        "lot_size": 0.0,
        "tick_size": 0.0,
        "execution_cost_configuration_included": False,
        "feature_config_digest": "0" * 64,
        "normalization_digest": "0" * 64,
    }
    dataset = MarketDataset(
        # MarketDataset requires a provisional digest; with_content_identity below
        # replaces it with the content-derived identity returned by this factory.
        dataset_id="0" * 64,
        symbols=_SYMBOLS,
        timestamps=timestamps,
        features=features,
        global_features=global_features,
        open=prices.copy(),
        high=prices.copy(),
        low=prices.copy(),
        close=prices.copy(),
        volume=np.full(price_shape, 1_000.0, dtype=np.float64),
        funding_rate=zero_prices.copy(),
        tradable=true_prices.copy(),
        feature_available=true_features.copy(),
        feature_names=_FEATURE_NAMES,
        global_feature_names=_GLOBAL_FEATURE_NAMES,
        periods_per_year=8_760,
        calendar_kind=MarketCalendarKind.CONTINUOUS,
        nominal_bar_hours=1.0,
        funding_event_count=np.zeros(price_shape, dtype=np.int32),
        feature_staleness_hours=zero_features.copy(),
        feature_missing_reason=np.zeros(feature_shape, dtype=np.int16),
        global_feature_available=true_global_features.copy(),
        global_feature_staleness_hours=zero_global_features.copy(),
        global_feature_missing_reason=np.zeros(global_shape, dtype=np.int16),
        feature_staleness=zero_features.copy(),
        fee_rate=zero_prices.copy(),
        maker_fee_rate=zero_prices.copy(),
        taker_fee_rate=zero_prices.copy(),
        spread_rate=zero_prices.copy(),
        max_participation_rate=np.ones(price_shape, dtype=np.float64),
        minimum_notional=zero_prices.copy(),
        lot_size=zero_prices.copy(),
        tick_size=zero_prices.copy(),
        borrow_available=true_prices.copy(),
        borrow_rate=zero_prices.copy(),
        funding_due=false_prices.copy(),
        asset_active=true_prices.copy(),
        buy_allowed=true_prices.copy(),
        sell_allowed=true_prices.copy(),
        mark_price=prices.copy(),
        index_price=prices.copy(),
        dividend=zero_prices.copy(),
        split_factor=one_prices.copy(),
        delisting_recovery=one_prices.copy(),
        cash_rate=np.zeros(_ROW_COUNT, dtype=np.float64),
        information_available=true_prices.copy(),
        available_at=available_at,
        volume_units=(VolumeUnit.BASE_ASSET,) * symbol_count,
        feature_config_digest="0" * 64,
        normalization_digest="0" * 64,
        contract_multipliers=np.ones(symbol_count, dtype=np.float64),
    )
    identified = dataset.with_content_identity(
        {"source_declared_synthetic_recipe": recipe}
    )
    return identified, MappingProxyType(recipe)
