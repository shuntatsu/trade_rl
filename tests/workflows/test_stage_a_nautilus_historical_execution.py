from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

import numpy as np
import pytest

pytest.importorskip("nautilus_trader")

from tests.data.test_market_dataset_v2 import kwargs
from tests.workflows.test_stage_a_nautilus_historical_replay import _replay_for_market
from trade_rl.data.market import MarketDataset
from trade_rl.simulation.funding_evidence import FundingBoundaryEvidence
from trade_rl.workflows.stage_a_nautilus_historical_replay import (
    execute_stage_a_nautilus_historical_replay,
)


def _aligned_single_symbol_market() -> MarketDataset:
    values = kwargs(n_bars=100, n_symbols=1)
    close = np.asarray(values["close"], dtype=np.float64)
    values["mark_price"] = close + 0.2
    values["index_price"] = close
    return MarketDataset(**values)


def _timestamp_ns(market: MarketDataset, index: int) -> int:
    return int(market.timestamps[index].astype("datetime64[ns]").astype(np.int64))


@pytest.mark.nautilus
def test_stage_a_historical_execution_uses_actual_boundary_position_for_funding() -> (
    None
):
    market = _aligned_single_symbol_market()
    replay = _replay_for_market(market)
    start = replay.cell_identity.evaluation_range.start
    shared = replay.transition_end_indices[0]
    first_open = float(market.open[start + 1, 0])
    target_one_btc = first_open / 1_000.0
    mark = float(market.mark_price[shared, 0])
    rate = 0.001
    funding_amount = -mark * rate
    equity_before_funding = 1_000.0
    funding = FundingBoundaryEvidence(
        processing_index=shared,
        timestamp_ns=_timestamp_ns(market, shared),
        funding_due=(True,),
        signed_quantities=(1.0,),
        mark_prices=(mark,),
        contract_multipliers=(1.0,),
        funding_rates=(rate,),
        funding_amount=funding_amount,
        equity_before_funding=equity_before_funding,
        equity_after_funding=equity_before_funding + funding_amount,
    )
    replay = replace(
        replay,
        actions=((target_one_btc,), (0.0,)),
        equity_curve=(1_000.0, 1_000.0 + funding_amount, 1_000.0 + funding_amount),
        digest="",
    )

    result = execute_stage_a_nautilus_historical_replay(
        replay,
        market,
        funding_evidence=(funding,),
        no_trade_band=0.0,
    )

    assert [fill.position_lots for fill in result.execution.fills] == [1_000, 0]
    assert result.execution.terminal_position_lots == 0
    assert result.execution.terminal_open_orders == 0
    assert [
        (snapshot.timestamp_ns, snapshot.signed_quantity)
        for snapshot in result.execution.position_snapshots
    ] == [(funding.timestamp_ns, Decimal("1"))]
    assert len(result.funding_records) == 1
    record = result.funding_records[0]
    assert record.event_type == "funding"
    assert record.timestamp_ns == funding.timestamp_ns
    assert record.position_lots == 1_000
    assert record.funding_minor == int(Decimal(str(funding_amount)) * Decimal("1e8"))
    assert record.equity_minor == int(
        Decimal(str(funding.equity_after_funding)) * Decimal("1e8")
    )
