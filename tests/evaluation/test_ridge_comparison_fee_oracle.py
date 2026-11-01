from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from trade_rl.data import (
    MarketCalendarKind,
    MarketDataset,
    publish_market_dataset_artifact,
)
from trade_rl.data.artifacts.publication import PublishedDatasetArtifact
from trade_rl.data.contracts import VolumeUnit
from trade_rl.evaluation.comparison.strategies import SharedCashStrategyComparisonEntry
from trade_rl.evaluation.runs import (
    RidgeArtifactComparisonConfig,
    compare_ridge_artifact_shared_cash,
)
from trade_rl.risk import PreTradeRiskConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.orders.model import OrderEvent
from trade_rl.strategies.forecasts import (
    RidgeForecastArtifactBinding,
    RidgeForecastModel,
    publish_ridge_forecast_artifact,
)


def _flat_two_symbol_dataset() -> MarketDataset:
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + (
        np.arange(6, dtype=np.int64) * np.timedelta64(1, "h")
    )
    prices = np.tile(np.asarray([100.0, 200.0], dtype=np.float64), (6, 1))
    zeros = np.zeros((6, 2), dtype=np.float64)
    volume = np.tile(np.asarray([2.0, 1.0], dtype=np.float64), (6, 1))
    # Keep the entry bars capacity-limited, while allowing the reserved final
    # settlement bar to close the full positions in one execution interval.
    volume[-1] = (4.0, 2.0)
    return MarketDataset(
        dataset_id="0" * 64,
        symbols=("AAA", "BBB"),
        timestamps=timestamps,
        features=np.zeros((6, 2, 1), dtype=np.float32),
        global_features=np.zeros((6, 1), dtype=np.float32),
        open=prices,
        high=prices,
        low=prices,
        close=prices,
        volume=volume,
        funding_rate=zeros,
        tradable=np.ones((6, 2), dtype=np.bool_),
        feature_available=np.ones((6, 2, 1), dtype=np.bool_),
        feature_names=("signal",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
        calendar_kind=MarketCalendarKind.CONTINUOUS,
        nominal_bar_hours=1.0,
        fee_rate=zeros,
        maker_fee_rate=zeros,
        taker_fee_rate=zeros,
        spread_rate=zeros,
        max_participation_rate=np.ones((6, 2), dtype=np.float64),
        borrow_available=np.ones((6, 2), dtype=np.bool_),
        borrow_rate=zeros,
        funding_due=np.zeros((6, 2), dtype=np.bool_),
        cash_rate=np.zeros(6, dtype=np.float64),
        volume_units=(VolumeUnit.BASE_ASSET, VolumeUnit.BASE_ASSET),
    ).with_content_identity({"fixture": "issue810-fourarm-fee-oracle"})


def _publish_literal_ridge(
    dataset: MarketDataset,
    dataset_artifact: PublishedDatasetArtifact,
    artifact_root: Path,
    *,
    forecast_intercept: float,
) -> tuple[str, RidgeForecastArtifactBinding]:
    model = RidgeForecastModel(
        feature_indices=(0,),
        feature_mean=np.zeros(1, dtype=np.float64),
        feature_scale=np.ones(1, dtype=np.float64),
        coefficients=np.zeros(1, dtype=np.float64),
        intercept=forecast_intercept,
        horizon_hours=1,
        alpha=1.0,
        n_samples=2,
        fit_cutoff=dataset.timestamps[0] - np.timedelta64(1, "h"),
    )
    cutoff_ns = int(model.fit_cutoff.astype("datetime64[ns]").astype(np.int64))
    binding = RidgeForecastArtifactBinding(
        dataset_id=dataset.dataset_id,
        dataset_schema_version=dataset_artifact.schema_version,
        dataset_artifact_sha256=dataset_artifact.artifact_digest,
        fit_symbol_indices=(0, 1),
        fit_symbol_names=dataset.symbols,
        feature_names=dataset.feature_names,
        feature_indices=model.feature_indices,
        fit_cutoff_ns=cutoff_ns,
        horizon_hours=model.horizon_hours,
        alpha_hex=float(model.alpha).hex(),
        n_samples=model.n_samples,
        vintage_id="literal-fourarm-fee-oracle",
        source_commit="a" * 40,
        implementation_sha256="b" * 64,
        runtime_sha256="c" * 64,
    )
    return publish_ridge_forecast_artifact(artifact_root, model, binding), binding


def _literal_fee_oracle() -> tuple[
    Fraction, Fraction, Fraction, tuple[tuple[int, Fraction], ...]
]:
    prices = (Fraction(100), Fraction(200))
    fills = (
        (0, Fraction(1)),
        (0, Fraction(1)),
        (1, Fraction(1, 2)),
        (1, Fraction(1, 2)),
        (0, Fraction(-2)),
        (1, Fraction(-1)),
    )
    raw_trade_cash = sum(
        (-quantity * prices[symbol] for symbol, quantity in fills),
        start=Fraction(0),
    )
    absolute_notional = sum(
        (abs(quantity * prices[symbol]) for symbol, quantity in fills),
        start=Fraction(0),
    )
    expected_fees = absolute_notional * Fraction(1, 100)
    return raw_trade_cash, absolute_notional, expected_fees, fills


def _filled_order_events(
    entry: SharedCashStrategyComparisonEntry,
) -> tuple[OrderEvent, ...]:
    ledger = entry.replay.ledger_evidence
    assert ledger is not None
    return tuple(
        event
        for interval in ledger.intervals
        for event in interval.order_events
        if event.filled_quantity != 0.0
    )


def _assert_no_fills(entry: SharedCashStrategyComparisonEntry) -> None:
    ledger = entry.replay.ledger_evidence
    assert ledger is not None
    assert _filled_order_events(entry) == ()
    assert entry.replay.book.fill_count == 0
    assert entry.replay.diagnostics.total_cost == 0.0
    assert entry.replay.diagnostics.funding_pnl == 0.0
    assert entry.replay.diagnostics.borrow_cost == 0.0
    assert entry.metrics.total_cost == 0.0
    assert ledger.final_total_cost == 0.0
    assert ledger.final_funding_pnl == 0.0
    assert ledger.final_borrow_cost == 0.0
    assert ledger.final_cash == pytest.approx(1_000.0, rel=0.0, abs=1e-10)
    assert ledger.final_portfolio_value == pytest.approx(1_000.0, rel=0.0, abs=1e-10)
    assert ledger.terminal_exact_quantities == ("0", "0")
    assert ledger.active_order_remainders == ()


def _assert_fee_paying_fills(entry: SharedCashStrategyComparisonEntry) -> None:
    raw_trade_cash, absolute_notional, expected_fees, literal_fills = (
        _literal_fee_oracle()
    )
    assert raw_trade_cash == 0
    assert absolute_notional == 800
    assert expected_fees == 8
    expected_terminal_cash = Fraction(1_000) + raw_trade_cash - expected_fees
    assert expected_terminal_cash == 992

    ledger = entry.replay.ledger_evidence
    assert ledger is not None
    fills = _filled_order_events(entry)
    prices = (Fraction(100), Fraction(200))
    for symbol_index, price in enumerate(prices):
        expected = tuple(
            quantity for symbol, quantity in literal_fills if symbol == symbol_index
        )
        actual = tuple(
            Fraction(str(event.filled_quantity))
            for event in fills
            if event.symbol_index == symbol_index
        )
        assert actual == expected

    actual_trade_cash = sum(
        -Fraction(str(event.filled_quantity)) * Fraction(str(event.execution_price))
        for event in fills
        if event.execution_price is not None
    )
    actual_notional = sum(
        abs(Fraction(str(event.filled_quantity)) * Fraction(str(event.execution_price)))
        for event in fills
        if event.execution_price is not None
    )
    assert actual_trade_cash == raw_trade_cash
    assert actual_notional == absolute_notional
    assert len(fills) == len(literal_fills) == 6
    assert all(
        Fraction(str(event.execution_price)) == prices[event.symbol_index]
        for event in fills
        if event.execution_price is not None
    )

    expected_cost = float(expected_fees)
    expected_cash = float(expected_terminal_cash)
    assert entry.replay.book.fill_count == 6
    assert entry.replay.book.total_cost == pytest.approx(
        expected_cost, rel=0.0, abs=1e-10
    )
    assert entry.replay.diagnostics.total_cost == pytest.approx(
        expected_cost, rel=0.0, abs=1e-10
    )
    assert entry.replay.diagnostics.funding_pnl == 0.0
    assert entry.replay.diagnostics.borrow_cost == 0.0
    assert entry.metrics.total_cost == pytest.approx(expected_cost, rel=0.0, abs=1e-10)
    assert ledger.final_total_cost == pytest.approx(expected_cost, rel=0.0, abs=1e-10)
    assert ledger.final_funding_pnl == 0.0
    assert ledger.final_borrow_cost == 0.0
    assert entry.replay.book.cash == pytest.approx(expected_cash, rel=0.0, abs=1e-10)
    assert entry.replay.book.portfolio_value == pytest.approx(
        expected_cash, rel=0.0, abs=1e-10
    )
    assert ledger.final_cash == pytest.approx(expected_cash, rel=0.0, abs=1e-10)
    assert ledger.final_portfolio_value == pytest.approx(
        expected_cash, rel=0.0, abs=1e-10
    )
    assert ledger.terminal_exact_quantities == ("0", "0")
    assert ledger.active_order_remainders == ()


@pytest.mark.parametrize(
    ("forecast_intercept", "positive_cost_trades"),
    [(0.005, False), (0.1, True)],
    ids=("weak-cost-gated", "strong-cost-admitted"),
)
def test_public_four_arm_comparison_matches_literal_fee_oracle(
    tmp_path: Path,
    forecast_intercept: float,
    positive_cost_trades: bool,
) -> None:
    dataset = _flat_two_symbol_dataset()
    dataset_root = tmp_path / "dataset"
    dataset_artifact = publish_market_dataset_artifact(dataset_root, dataset)
    ridge_root = tmp_path / "ridge"
    ridge_digest, binding = _publish_literal_ridge(
        dataset,
        dataset_artifact,
        ridge_root,
        forecast_intercept=forecast_intercept,
    )
    config = RidgeArtifactComparisonConfig(
        start_index=0,
        stop_index=5,
        gross_budget=0.2,
        initial_capital=1_000.0,
        entry_threshold=0.001,
        exit_threshold=0.0,
        one_way_switch_cost=0.01,
        execution_cost=ExecutionCostConfig(
            fee_rate=0.01,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0,
            spread_rate=0.0,
            impact_rate=0.0,
            max_participation_rate=0.5,
            slippage_std=0.0,
            tail_slippage_probability=0.0,
            order_latency_bars=0,
            processing_bar_volume_capacity=True,
            partial_fill_carry=True,
        ),
        risk=PreTradeRiskConfig(
            max_gross=1.0,
            max_abs_weight=1.0,
            max_turnover=None,
            drawdown_start=1.0,
            drawdown_stop=1.0,
            emergency_turnover_override=False,
            fail_closed_tolerance=1e-10,
        ),
        minimum_hold_bars=0,
    )

    result = compare_ridge_artifact_shared_cash(
        dataset_root=dataset_root,
        ridge_artifact_root=ridge_root,
        expected_ridge_digest=ridge_digest,
        expected_binding=binding,
        config=config,
    )

    expected_names = (
        "cash",
        "legacy_hysteresis",
        "position_aware_zero_cost",
        "position_aware_cost",
    )
    assert tuple(entry.name for entry in result.entries) == expected_names
    entries = {entry.name: entry for entry in result.entries}
    _assert_no_fills(entries["cash"])
    for name in ("legacy_hysteresis", "position_aware_zero_cost"):
        _assert_fee_paying_fills(entries[name])
    if positive_cost_trades:
        _assert_fee_paying_fills(entries["position_aware_cost"])
    else:
        _assert_no_fills(entries["position_aware_cost"])
