from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from trade_rl.data.artifacts.publication import (
    load_market_dataset_artifact,
    publish_market_dataset_artifact,
)
from trade_rl.data.build import ExecutionEconomicsProfile, MarketDatasetBuilder
from trade_rl.data.contracts import (
    FeatureKind,
    FeatureSpec,
    InstrumentContract,
    MarketBuildConfig,
)
from trade_rl.data.source import InMemoryMarketDataSource, RawMarketSeries
from trade_rl.simulation import BookState, ExecutionCostConfig, MarketExecutor


def _profile() -> ExecutionEconomicsProfile:
    return ExecutionEconomicsProfile(
        name="canonical_m2_research_assumption_v1",
        fee_rate=0.0005,
        spread_rate=0.0002,
        max_participation_rate=0.05,
        borrow_available=True,
        borrow_rate=0.0,
    )


def _dataset():
    n_bars = 8
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        n_bars
    ) * np.timedelta64(1, "h")
    close = np.full(n_bars, 100.0, dtype=np.float64)
    raw = RawMarketSeries(
        timestamps=timestamps,
        open=close.copy(),
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=np.full(n_bars, 1_000_000.0, dtype=np.float64),
        funding_rate=np.zeros(n_bars, dtype=np.float64),
        tradable=np.ones(n_bars, dtype=np.bool_),
    )
    config = MarketBuildConfig(
        base_timeframe="1h",
        features=(
            FeatureSpec(name="ret_1", kind=FeatureKind.LOG_RETURN, lookback=1),
        ),
    )
    contract = InstrumentContract(
        symbol="BTCUSDT",
        listed_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    return MarketDatasetBuilder(config).build(
        InMemoryMarketDataSource({"BTCUSDT": raw}),
        (contract,),
        execution_economics=_profile(),
    )


def test_execution_economics_survive_published_artifact_roundtrip(
    tmp_path: Path,
) -> None:
    original = _dataset()
    published = publish_market_dataset_artifact(tmp_path / "dataset", original)
    restored = load_market_dataset_artifact(tmp_path / "dataset")

    assert restored.dataset_id == original.dataset_id
    assert published.artifact_digest
    for field_name in (
        "fee_rate",
        "maker_fee_rate",
        "taker_fee_rate",
        "spread_rate",
        "max_participation_rate",
        "borrow_available",
        "borrow_rate",
    ):
        np.testing.assert_array_equal(
            restored.resolved_array(field_name),
            original.resolved_array(field_name),
        )
    identity = json.loads(restored.identity_payload_json or "{}")
    assert identity["execution_economics"] == _profile().to_payload()


def test_zero_overlay_charges_dataset_cost_exactly_once() -> None:
    dataset = _dataset()
    initial_capital = 1_000.0
    executor = MarketExecutor(dataset, ExecutionCostConfig.zero())
    book = BookState.zero(
        dataset.n_symbols,
        initial_capital,
        dataset.resolved_array("mark_price")[0],
        contract_multipliers=dataset.contract_multipliers,
    )

    result = executor.execute_interval(
        book,
        np.array([1.0]),
        start_index=0,
        bars=1,
    )

    filled_notional = float(np.sum(result.filled_notional_by_symbol))
    expected_unit_cost = _profile().fee_rate + _profile().spread_rate
    expected_cost = filled_notional * expected_unit_cost

    assert result.fill_count == 1
    assert filled_notional > 0.0
    assert result.interval_cost == pytest.approx(expected_cost, rel=0.0, abs=1e-12)
    assert result.cost_by_symbol[0] == pytest.approx(expected_cost, rel=0.0, abs=1e-12)
    assert result.book.total_cost == pytest.approx(expected_cost, rel=0.0, abs=1e-12)
    assert expected_cost == pytest.approx(0.7, rel=0.0, abs=1e-12)
