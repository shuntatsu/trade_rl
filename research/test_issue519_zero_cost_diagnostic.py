from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec

import numpy as np

from trade_rl.data.market import MarketDataset

MODULE = "research.issue519_zero_cost_diagnostic"


def _module():
    spec = find_spec(MODULE)
    assert spec is not None, "zero-cost diagnostic helper is not implemented"
    return import_module(MODULE)


def _dataset() -> MarketDataset:
    n = 8
    timestamps = np.arange(n, dtype=np.int64).astype("datetime64[h]")
    close = np.asarray([100.0, 101.0, 99.0, 102.0, 98.0, 103.0, 97.0, 104.0])[:, None]
    open_price = close.copy()
    high = close * 1.01
    low = close * 0.99
    features = np.zeros((n, 1, 1), dtype=np.float32)
    feature_available = np.ones_like(features, dtype=np.bool_)
    shape = (n, 1)
    dataset = MarketDataset(
        dataset_id="0" * 64,
        symbols=("BTCUSDT",),
        timestamps=timestamps,
        features=features,
        global_features=np.zeros((n, 1), dtype=np.float32),
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=np.full(shape, 1_000_000.0),
        funding_rate=np.full(shape, 0.001),
        tradable=np.ones(shape, dtype=np.bool_),
        feature_available=feature_available,
        feature_names=("signal",),
        global_feature_names=("global",),
        periods_per_year=8760,
        fee_rate=np.full(shape, 0.0005),
        maker_fee_rate=np.full(shape, 0.0001),
        taker_fee_rate=np.full(shape, 0.0002),
        spread_rate=np.full(shape, 0.0002),
        max_participation_rate=np.full(shape, 0.05),
        minimum_notional=np.full(shape, 10.0),
        lot_size=np.full(shape, 0.001),
        tick_size=np.full(shape, 0.01),
        borrow_rate=np.full(shape, 0.0003),
        cash_rate=np.full(n, 0.0001),
    )
    return dataset.with_content_identity({"fixture": "zero-cost-diagnostic"})


def test_zero_cost_dataset_changes_only_economic_cost_arrays() -> None:
    module = _module()
    source = _dataset()
    diagnostic = module.make_zero_cost_diagnostic_dataset(source)

    assert diagnostic.dataset_id != source.dataset_id
    assert diagnostic.identity_verified
    assert source.identity_verified
    for field in (
        "fee_rate",
        "maker_fee_rate",
        "taker_fee_rate",
        "spread_rate",
        "funding_rate",
        "borrow_rate",
        "cash_rate",
    ):
        assert np.all(diagnostic.resolved_array(field) == 0.0)
        assert np.any(source.resolved_array(field) != 0.0)

    for field in (
        "timestamps",
        "features",
        "feature_available",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "tradable",
        "max_participation_rate",
        "minimum_notional",
        "lot_size",
        "tick_size",
        "borrow_available",
        "buy_allowed",
        "sell_allowed",
        "mark_price",
        "index_price",
        "contract_multipliers",
    ):
        assert np.array_equal(diagnostic.resolved_array(field), source.resolved_array(field))


def test_direction_attribution_reconstructs_total_log_return() -> None:
    module = _module()
    returns = np.asarray([0.10, -0.05, 0.02, -0.01], dtype=np.float64)
    intents = ["LONG", "SHORT", "LONG", "FLAT"]
    result = module.direction_log_return_attribution(returns, intents)
    expected = float(np.log1p(returns).sum())
    assert np.isclose(
        result["long_log_return_contribution"]
        + result["short_log_return_contribution"]
        + result["flat_log_return_contribution"],
        expected,
    )
    assert result["long_intervals"] == 2
    assert result["short_intervals"] == 1
    assert result["flat_intervals"] == 1


def test_direction_attribution_rejects_length_mismatch() -> None:
    module = _module()
    returns = np.asarray([0.01, 0.02], dtype=np.float64)
    try:
        module.direction_log_return_attribution(returns, ["LONG"])
    except ValueError as exc:
        assert "length" in str(exc)
    else:
        raise AssertionError("length mismatch must fail")
