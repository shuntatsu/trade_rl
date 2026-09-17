"""Paired carry research replay on the canonical execution ledger."""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import UTC, datetime
from fractions import Fraction
from typing import Any

import numpy as np

from trade_rl.artifacts.canonical import JsonValue, to_json_value
from trade_rl.data.market import MarketDataset
from trade_rl.simulation import BookState, ExecutionCostConfig, MarketExecutor
from trade_rl.simulation.orders.model import (
    OrderBookState,
    OrderIntent,
    OrderType,
    TimeInForce,
)
from trade_rl.simulation.quantities import (
    exact_quantity,
    parse_quantity,
    project_quantity,
)
from trade_rl.simulation.stateful.execution import (
    StatefulExecutionResult,
    execute_stateful_orders,
)
from trade_rl.strategies.carry import CarryConfig, FundingCarryBot


def futures_collateral(
    cash: float, quantities: np.ndarray, prices: np.ndarray
) -> float:
    """All idle USDT is assigned to futures; spot assets are not collateral."""
    return cash + float(np.dot(quantities[1::2], prices[1::2]))


def _execute_quantities(
    executor: MarketExecutor,
    book: BookState,
    orders: OrderBookState,
    target: np.ndarray,
    index: int,
) -> StatefulExecutionResult:
    """Submit absolute carry quantities without a weight/price round trip."""
    intents, cancellations = [], []
    for symbol, desired in enumerate(target):
        residual = exact_quantity(float(desired)) - exact_quantity(
            float(book.quantities[symbol])
        )
        active = orders.active_for_symbol(symbol)
        pending = sum(
            (
                exact_quantity(order.intent.requested_quantity)
                - parse_quantity(str(order.exact_cumulative_filled_quantity))
                for order in active
            ),
            Fraction(0),
        )
        if residual == pending:
            continue
        for previous in active:
            cancelled = previous.cancel(processing_index=index, reason="superseded")
            orders = orders.replace(cancelled)
            cancellations.append((previous, cancelled))
        if not residual:
            continue
        intents.append(
            OrderIntent.create(
                dataset_id=executor.dataset.dataset_id,
                target_identity=f"carry:{index}",
                execution_policy_digest=executor.execution_policy_digest,
                symbol_index=symbol,
                requested_quantity=project_quantity(residual),
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.GTC,
                limit_price=None,
                stop_price=None,
                submit_index=index,
                eligible_index=index + 1,
                expiry_index=None,
                submission_reference_price=float(executor.dataset.close[index, symbol]),
                decision_equity=book.portfolio_value,
                replaced_order_id=active[0].order_id if active else None,
            )
        )
    return execute_stateful_orders(
        executor,
        book,
        orders,
        intents,
        start_index=index,
        bars=1,
        reconciliation_cancellations=cancellations,
    )


def _validate_replay(
    dataset: MarketDataset, start: int, stop: int, capital: float, delay: int
) -> None:
    if (
        not dataset.identity_verified
        or dataset.n_symbols not in {2, 4}
        or any(
            dataset.symbols[i].removesuffix(":spot")
            != dataset.symbols[i + 1].removesuffix(":perp")
            or not dataset.symbols[i].endswith(":spot")
            or not dataset.symbols[i + 1].endswith(":perp")
            for i in range(0, dataset.n_symbols, 2)
        )
    ):
        raise ValueError(
            "carry replay requires identity-bound alternating spot/perpetual pairs"
        )
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(stop, bool)
        or not isinstance(stop, int)
        or not 0 <= start < stop < dataset.n_bars
        or stop - start < 3
    ):
        raise ValueError("carry window must allow actual entry and exit")
    if isinstance(capital, bool) or not math.isfinite(capital) or capital <= 0:
        raise ValueError("carry capital must be positive")
    if isinstance(delay, bool) or not isinstance(delay, int) or delay not in {0, 1}:
        raise ValueError("carry initial entry delay must be zero or one")
    for field in ("borrow_rate", "cash_rate", "dividend"):
        if np.any(dataset.resolved_array(field)):
            raise ValueError("carry wallet excludes borrow, interest and dividends")
    for field in ("split_factor", "contract_multipliers"):
        if np.any(dataset.resolved_array(field) != 1):
            raise ValueError("carry requires linear unit contracts without splits")
    if np.any(dataset.funding_rate[:, ::2]):
        raise ValueError("spot funding must be zero")


def replay_carry(
    dataset: MarketDataset,
    *,
    start_index: int,
    stop_index: int,
    initial_capital: float = 10_000.0,
    entry_delay_bars: int = 0,
) -> dict[str, Any]:
    """Replay one fixed carry treatment; no financial ledger outside BookState."""
    _validate_replay(
        dataset, start_index, stop_index, initial_capital, entry_delay_bars
    )
    config = CarryConfig()
    bot = FundingCarryBot(config)
    execution = replace(
        ExecutionCostConfig.zero(), processing_bar_volume_capacity=False
    )
    executor = MarketExecutor(dataset, execution)
    book = BookState.zero(
        dataset.n_symbols, initial_capital, dataset.close[start_index]
    )
    orders = OrderBookState.empty()
    returns: list[float] = []
    equity = [float(initial_capital)]
    quantities: list[list[float]] = []
    collateral: list[dict[str, float | int]] = []
    events: list[JsonValue] = []
    funding_events: list[JsonValue] = []
    gross: list[float] = []
    invalid: list[str] = []
    total_cost, funding_pnl, turnover = 0.0, 0.0, 0.0
    unresolved = None
    for index in range(start_index, stop_index):
        observed_collateral = futures_collateral(
            book.cash, book.quantities, dataset.close[index]
        )
        required = float(
            np.abs(book.quantities[1::2] * dataset.close[index, 1::2]).sum()
        )
        if observed_collateral < required:
            bot.stop("insufficient_futures_collateral")
        if index == stop_index - 1 or index < start_index + entry_delay_bars:
            target = np.zeros(dataset.n_symbols)
        else:
            timestamp = datetime.fromtimestamp(
                int(dataset.timestamps[index].astype("datetime64[s]").astype(np.int64)),
                UTC,
            )
            target = bot.decide(
                timestamp=timestamp,
                prices=dataset.close[index],
                quantities=book.quantities,
                equity=book.portfolio_value,
                drawdown=book.max_drawdown,
            )
        before = book.quantities.copy()
        cash_before = book.cash
        result = _execute_quantities(executor, book, orders, target, index)
        book, orders = result.book, result.order_book
        events.extend(to_json_value(event) for event in result.order_events)
        funding_events.extend(to_json_value(event) for event in result.funding_evidence)
        returns.append(result.interval_net_return)
        equity.append(book.portfolio_value)
        total_cost += result.interval_cost
        funding_pnl += result.interval_funding
        turnover += result.filled_turnover
        if result.termination_reason is not None:
            # Core termination can forcibly clear its reporting state without
            # fills. Preserve the unresolved inventory and invalidate this run.
            unresolved = before.copy()
            for event in result.order_events:
                unresolved[event.symbol_index] += event.filled_quantity
            quantities.append(unresolved.tolist())
            invalid.append("canonical_termination:" + result.termination_reason)
            bot.stop("canonical_termination")
            break
        processing = index + 1
        quantities.append(book.quantities.tolist())
        open_collateral = futures_collateral(
            cash_before, before, dataset.open[processing]
        )
        open_required = 0.5 * float(
            np.abs(before[1::2] * dataset.open[processing, 1::2]).sum()
        )
        if open_collateral < open_required:
            if "open_margin_breach" not in invalid:
                invalid.append("open_margin_breach")
            bot.stop("open_margin_breach")
        pre_funding_cash = book.cash - result.interval_funding
        adverse_collateral = futures_collateral(
            pre_funding_cash, book.quantities, dataset.high[processing]
        )
        adverse_notional = float(
            np.abs(book.quantities[1::2] * dataset.high[processing, 1::2]).sum()
        )
        close_collateral = futures_collateral(
            book.cash, book.quantities, dataset.close[processing]
        )
        collateral.append(
            {
                "processing_index": processing,
                "open_before_fills": open_collateral,
                "open_maintenance_requirement": open_required,
                "intrabar_before_funding": adverse_collateral,
                "maintenance_requirement": 0.5 * adverse_notional,
                "close_after_funding": close_collateral,
            }
        )
        gross.append(float(np.abs(book.position_values).sum() / book.portfolio_value))
        if adverse_collateral < 0.5 * adverse_notional:
            if "intrabar_margin_breach" not in invalid:
                invalid.append("intrabar_margin_breach")
            bot.stop("intrabar_margin_breach")
        if np.any(np.abs(book.quantities[::2] + book.quantities[1::2]) > 1e-9):
            bot.stop("unmatched_hedge")
        if book.max_drawdown >= config.maximum_drawdown:
            bot.stop("maximum_drawdown")
    values = np.asarray(returns)
    years = (
        dataset.timestamps[start_index : start_index + len(values)]
        .astype("datetime64[Y]")
        .astype(str)
    )
    annual = {
        year: float(np.prod(1 + values[years == year]) - 1)
        for year in sorted(set(years))
    }
    total = float(np.prod(1 + values) - 1)
    terminal = book.quantities if unresolved is None else unresolved
    flat = not invalid and bool(np.all(terminal == 0)) and not orders.active_orders
    complete = len(values) == stop_index - start_index
    return {
        "schema": "funding_carry_replay_v1",
        "dataset_id": dataset.dataset_id,
        "start_index": start_index,
        "stop_index": stop_index,
        "initial_capital": initial_capital,
        "entry_delay_bars": entry_delay_bars,
        "execution_policy_digest": executor.execution_policy_digest,
        "returns": returns,
        "equity": equity,
        "quantities": quantities,
        "order_events": events,
        "funding_evidence": funding_events,
        "collateral": collateral,
        "realized_gross": gross,
        "total_return": total,
        "year_returns": annual,
        "total_cost": total_cost,
        "funding_pnl": funding_pnl,
        "turnover_total": turnover,
        "ledger_max_drawdown": book.max_drawdown,
        "stop_reason": bot.stop_reason,
        "invalid_reasons": invalid,
        "execution_valid": not invalid,
        "complete": complete,
        "terminal_quantities": terminal.tolist(),
        "terminal_flat": flat,
        "qualified": bool(
            complete
            and flat
            and not bot.stop_reason
            and total > 0
            and book.max_drawdown < config.maximum_drawdown
            and all(value > 0 for value in annual.values())
        ),
        "production_eligible": False,
    }
