from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.simulation import MarketExecutor
from trade_rl.simulation.accounting import BookState, EconomicTerminationReason
from trade_rl.simulation.diagnostics.funding import (
    FUNDING_BOUNDARY_ARTIFACT_SCHEMA_V2,
    build_funding_evidence_artifact,
    load_funding_evidence_artifact_bytes,
)
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.orders.model import OrderBookState


def _funding_dataset(
    *,
    funding_price_rate: np.ndarray | None = None,
    contract_multiplier: float = 1.0,
) -> MarketDataset:
    n_bars = 4
    shape = (n_bars, 1)
    close = np.full(shape, 100.0, dtype=np.float64)
    mark_price = close.copy()
    mark_price[1, 0] = 120.0
    funding_rate = np.zeros(shape, dtype=np.float64)
    funding_rate[1, 0] = 0.001
    funding_due = np.zeros(shape, dtype=np.bool_)
    funding_due[1, 0] = True
    return MarketDataset(
        dataset_id="f" * 64,
        symbols=("BTCUSDT",),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=np.zeros((n_bars, 1, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=close.copy(),
        high=np.maximum(close, mark_price),
        low=np.minimum(close, mark_price),
        close=close,
        volume=np.full(shape, 1_000.0, dtype=np.float64),
        funding_rate=funding_rate,
        tradable=np.ones(shape, dtype=np.bool_),
        feature_available=np.ones((n_bars, 1, 1), dtype=np.bool_),
        feature_names=("probe",),
        global_feature_names=("probe",),
        periods_per_year=8_760,
        funding_due=funding_due,
        mark_price=mark_price,
        contract_multipliers=np.array([contract_multiplier], dtype=np.float64),
        funding_price_rate=funding_price_rate,
    )


def test_usds_m_funding_uses_mark_price_position_notional() -> None:
    dataset = _funding_dataset()
    executor = MarketExecutor(
        dataset,
        replace(
            ExecutionCostConfig.zero(),
            processing_bar_volume_capacity=True,
            partial_fill_carry=True,
        ),
    )
    book = BookState.from_weights(
        weights=np.array([1.0], dtype=np.float64),
        capital=1_000.0,
        prices=dataset.close[0],
        contract_multipliers=dataset.resolved_array("contract_multipliers"),
    )

    result = executor.execute_orders(
        book,
        OrderBookState.empty(),
        (),
        start_index=0,
        bars=1,
    )

    # Binance USDⓈ-M funding amount is position notional at mark price × rate.
    # Quantity is 10 BTC; mark notional at the funding boundary is 10 × 120.
    assert result.interval_funding == pytest.approx(-1.2)
    assert result.book.funding_pnl == pytest.approx(-1.2)


def test_v7_funding_semantics_change_execution_policy_digest_only_for_v7() -> None:
    cost = ExecutionCostConfig.zero()
    legacy = MarketExecutor(_funding_dataset(), cost)
    event_price_rate = np.zeros((4, 1), dtype=np.float64)
    v7_dataset = _funding_dataset(
        funding_price_rate=event_price_rate
    ).with_content_identity()
    v7 = MarketExecutor(v7_dataset, cost)

    assert legacy.execution_policy_digest == cost.execution_policy_digest
    assert v7.execution_policy_digest != legacy.execution_policy_digest
    assert (
        v7.execution_policy_digest
        == MarketExecutor(v7_dataset, cost).execution_policy_digest
    )


@pytest.mark.parametrize(
    ("weight", "funding_due", "expected_funding"),
    [(1.0, True, -2.5), (-1.0, True, 2.5), (1.0, False, 0.0)],
)
def test_v7_funding_uses_identity_bound_event_price_rate(
    weight: float,
    funding_due: bool,
    expected_funding: float,
) -> None:
    funding_price_rate = np.zeros((4, 1), dtype=np.float64)
    funding_price_rate[1, 0] = 0.25
    dataset = _funding_dataset(
        funding_price_rate=funding_price_rate,
        contract_multiplier=2.0,
    )
    if not funding_due:
        due = dataset.resolved_array("funding_due").copy()
        due[1, 0] = False
        dataset = replace(dataset, funding_due=due)
    dataset = dataset.with_content_identity()

    executor = MarketExecutor(
        dataset,
        replace(
            ExecutionCostConfig.zero(),
            processing_bar_volume_capacity=True,
            partial_fill_carry=True,
        ),
    )
    book = BookState.from_weights(
        weights=np.array([weight], dtype=np.float64),
        capital=1_000.0,
        prices=dataset.close[0],
        contract_multipliers=dataset.resolved_array("contract_multipliers"),
    )

    result = executor.execute_orders(
        book,
        OrderBookState.empty(),
        (),
        start_index=0,
        bars=1,
    )

    # Independent oracle: signed contracts × contract multiplier × the stored
    # event-price rate, charged only on the declared funding boundary.
    expected = -float(
        book.quantities[0]
        * dataset.contract_multipliers[0]
        * funding_price_rate[1, 0]
        * funding_due
    )
    assert expected == pytest.approx(expected_funding)
    assert result.interval_funding == pytest.approx(expected)
    assert result.book.funding_pnl == pytest.approx(expected)
    if funding_due:
        evidence = result.funding_evidence[0]
        assert evidence.funding_price_rates == pytest.approx((0.25,))
        # The v2 receipt carries an actual independent pre-funding book mark.
        observed_pre = float(
            book.cash
            + np.sum(
                book.quantities
                * dataset.resolved_array("mark_price")[1]
                * dataset.resolved_array("contract_multipliers")
            )
        )
        assert evidence.equity_before_funding == pytest.approx(observed_pre)
        assert evidence.equity_after_funding == pytest.approx(observed_pre + expected)
        evidence_artifact = build_funding_evidence_artifact(
            dataset_id=dataset.dataset_id,
            execution_policy_digest=executor.execution_policy_digest,
            symbol_count=dataset.n_symbols,
            boundaries=result.funding_evidence,
        )
        assert evidence_artifact.schema_version == FUNDING_BOUNDARY_ARTIFACT_SCHEMA_V2
        round_trip = load_funding_evidence_artifact_bytes(evidence_artifact.raw_bytes)
        assert round_trip == evidence_artifact
        assert round_trip.boundaries[0].funding_amount == pytest.approx(expected)
    else:
        assert result.funding_evidence == ()
    assert funding_price_rate[1, 0] != pytest.approx(
        dataset.funding_rate[1, 0] * dataset.resolved_array("mark_price")[1, 0]
    )


def test_v7_stored_price_funding_can_trigger_retained_debt_insolvency() -> None:
    funding_price_rate = np.array([[0.0], [150.0], [75.0], [50.0]])
    base = _funding_dataset(
        funding_price_rate=funding_price_rate,
        contract_multiplier=2.0,
    )
    funding_due = np.ones((4, 1), dtype=np.bool_)
    funding_due[0, 0] = False
    dataset = replace(base, funding_due=funding_due).with_content_identity(
        {"source": "synthetic-v7-retained-debt-test"}
    )
    executor = MarketExecutor(
        dataset,
        replace(
            ExecutionCostConfig.zero(),
            processing_bar_volume_capacity=True,
            partial_fill_carry=True,
        ),
        insolvency_valuation="retain_debt",
    )
    book = BookState.from_weights(
        weights=np.array([1.0], dtype=np.float64),
        capital=1_000.0,
        prices=dataset.close[0],
        contract_multipliers=dataset.resolved_array("contract_multipliers"),
    )

    result = executor.execute_orders(
        book,
        OrderBookState.empty(),
        (),
        start_index=0,
        bars=3,
    )

    expected_funding = -float(
        book.quantities[0] * dataset.contract_multipliers[0] * funding_price_rate[1, 0]
    )
    assert funding_price_rate[1, 0] != pytest.approx(
        dataset.funding_rate[1, 0] * dataset.resolved_array("mark_price")[1, 0]
    )
    assert expected_funding == pytest.approx(-1_500.0)
    assert result.interval_funding == pytest.approx(expected_funding)
    assert result.book.funding_pnl == pytest.approx(expected_funding)
    assert result.book.insolvent is True
    assert result.book.termination_reason is EconomicTerminationReason.MARGIN_CALL
    assert result.bars_advanced == 1
    assert result.next_index == 1
    assert result.book.cash == pytest.approx(-300.0)
    assert result.book.portfolio_value == pytest.approx(-300.0)
    assert result.book.quantities.tolist() == [0.0]
    assert result.order_book.active_orders == ()
    assert len(result.funding_evidence) == 1
    assert result.funding_evidence[0].funding_price_rates == pytest.approx((150.0,))

    terminal_funding_pnl = result.book.funding_pnl
    with pytest.raises(ValueError, match="already terminated book"):
        executor.execute_orders(
            result.book,
            result.order_book,
            (),
            start_index=result.next_index,
            bars=1,
        )
    assert result.book.cash == pytest.approx(-300.0)
    assert result.book.portfolio_value == pytest.approx(-300.0)
    assert result.book.funding_pnl == pytest.approx(terminal_funding_pnl)
    assert result.book.quantities.tolist() == [0.0]
    assert result.order_book.active_orders == ()
