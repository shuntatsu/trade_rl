"""Read returned ledger/fill facts once; never book expected costs again."""

from __future__ import annotations

from math import fsum, isclose, sqrt
from typing import Any

from trade_rl.data.market import MarketDataset
from trade_rl.simulation.execution import ExecutionCostConfig

from .tier_d_protocol import CAPITAL


def ledger_decomposition(
    raw: dict[str, Any], dataset: MarketDataset, cost: ExecutionCostConfig
) -> dict[str, Any]:
    native = [row["native"] for row in raw["native_receipt"]["rows"]]
    closing = [row["native"] for row in raw["closing_rows"]]
    opening, price, fees, spreads, impacts = CAPITAL, [], [], [], []
    for facts in (*native, *closing):
        execution = facts["execution"]
        price.append(opening * execution["interval_gross_return"])
        opening = facts["book"]["equity"]
        for event in facts["order_events"]:
            if event["event_type"] not in ("filled", "partial_fill"):
                continue
            row, symbol = event["processing_index"], event["symbol_index"]
            notional = event["filled_notional"]
            fees.append(
                notional
                * cost.multiplier
                * (
                    cost.fee_rate
                    + cost.taker_fee_rate
                    + dataset.resolved_array("fee_rate")[row, symbol]
                    + dataset.resolved_array("taker_fee_rate")[row, symbol]
                )
            )
            spreads.append(
                notional
                * cost.multiplier
                * (
                    cost.spread_rate
                    + dataset.resolved_array("spread_rate")[row, symbol]
                )
            )
            impacts.append(
                notional
                * cost.multiplier
                * cost.impact_rate
                * sqrt(event["participation_rate"])
            )
    executions = [facts["execution"] for facts in (*native, *closing)]
    total_cost = fsum(item["interval_cost"] for item in executions)
    funding = fsum(item["interval_funding"] for item in executions)
    borrow = fsum(item["interval_borrow_cost"] for item in executions)
    interest = fsum(item["interval_cash_interest"] for item in executions)
    dividend = fsum(item["interval_dividend"] for item in executions)
    price_pnl = fsum(price)
    net = price_pnl - total_cost + funding - borrow + interest + dividend
    terminal = raw["final_book"]["equity"] - CAPITAL
    if not isclose(
        total_cost, fsum((*fees, *spreads, *impacts)), rel_tol=1e-10, abs_tol=1e-8
    ) or not isclose(net, terminal, rel_tol=1e-10, abs_tol=1e-8):
        raise ValueError("returned native ledger/fill decomposition does not reconcile")
    return {
        "currency": "USDT",
        "price_pnl": price_pnl,
        "fee": fsum(fees),
        "spread_cost": fsum(spreads),
        "impact_cost": fsum(impacts),
        "execution_cost": total_cost,
        "funding_pnl": funding,
        "borrow_cost": borrow,
        "cash_interest": interest,
        "dividends": dividend,
        "closing_execution_cost_subset": fsum(
            item["execution"]["interval_cost"] for item in closing
        ),
        "reconstructed_net_pnl": net,
        "native_terminal_net_pnl": terminal,
        "reconciliation_residual": net - terminal,
        "source": "public returned native intervals/fills; fee/spread/impact use declared source row rates",
        "limits": "not an independent historical source audit; closing cost is a subset, not another deduction",
    }
