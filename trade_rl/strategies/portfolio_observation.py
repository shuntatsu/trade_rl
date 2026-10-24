"""Build detached portfolio-policy observations from current source-owned state."""

from __future__ import annotations

from trade_rl.data.market import MarketDataset
from trade_rl.risk.pretrade import PreTradeRiskConfig
from trade_rl.simulation.accounting import BookState, EconomicTerminationReason
from trade_rl.simulation.orders.model import PendingOrder
from trade_rl.strategies.interface import StrategyObservation
from trade_rl.strategies.portfolio import (
    PortfolioAccountSnapshot,
    PortfolioDecisionSnapshot,
    PortfolioPendingOrderSnapshot,
    PortfolioRiskLimits,
)
from trade_rl.strategies.position_intent import PositionIntent


def build_portfolio_decision_snapshot(
    dataset: MarketDataset,
    *,
    index: int,
    agent_stop_index: int,
    book: BookState,
    active_orders: tuple[PendingOrder, ...],
    risk_config: PreTradeRiskConfig,
) -> PortfolioDecisionSnapshot:
    """Copy one causal row and current account facts into immutable primitives.

    This strategy-side builder accepts current source state and an immutable
    active-order tuple, not the evaluation replay or execution engine. It does
    not infer free cash or reservation amounts from the reported account cash.
    """

    observations = tuple(
        StrategyObservation(
            index=index,
            timestamp=dataset.timestamps[index],
            symbol=dataset.symbols[symbol_index],
            features=dataset.features[index, symbol_index],
            feature_available=dataset.feature_available[index, symbol_index],
            feature_staleness=dataset.resolved_array("feature_staleness")[
                index, symbol_index
            ],
            global_features=dataset.global_features[index],
            global_feature_available=dataset.resolved_array("global_feature_available")[
                index
            ],
            current_intent=PositionIntent.FLAT,
            current_weight=float(book.weights[symbol_index]),
            current_position_quantity=float(book.quantities[symbol_index]),
            intent_gross_budget=None,
        )
        for symbol_index in range(dataset.n_symbols)
    )
    feature_staleness_rows: list[tuple[float, ...]] = []
    for observation in observations:
        if observation.feature_staleness is None:
            raise RuntimeError("replay observation omitted feature staleness")
        feature_staleness_rows.append(
            tuple(float(value) for value in observation.feature_staleness)
        )
    contract_multipliers = book.contract_multipliers
    assert contract_multipliers is not None
    reason = book.termination_reason
    termination_reason = (
        None
        if reason is None
        else (
            reason.value
            if isinstance(reason, EconomicTerminationReason)
            else str(reason)
        )
    )
    account = PortfolioAccountSnapshot(
        cash=float(book.cash),
        portfolio_value=float(book.portfolio_value),
        peak_value=float(book.peak_value),
        max_drawdown=float(book.max_drawdown),
        exact_quantities=tuple(str(value) for value in book.exact_quantities),
        quantities=tuple(float(value) for value in book.quantities),
        weights=tuple(float(value) for value in book.weights),
        mark_prices=tuple(float(value) for value in book.mark_prices),
        contract_multipliers=tuple(float(value) for value in contract_multipliers),
        turnover_total=float(book.turnover_total),
        total_cost=float(book.total_cost),
        funding_pnl=float(book.funding_pnl),
        borrow_cost=float(book.borrow_cost),
        margin_used=float(book.margin_used),
        maintenance_margin=float(book.maintenance_margin),
        maintenance_requirement=float(book.maintenance_requirement),
        margin_deficit=float(book.margin_deficit),
        insolvent=bool(book.insolvent),
        termination_reason=termination_reason,
    )
    order_snapshots = tuple(
        PortfolioPendingOrderSnapshot(
            symbol_index=int(order.intent.symbol_index),
            order_type=order.intent.order_type.value,
            time_in_force=order.intent.time_in_force.value,
            requested_quantity=float(order.intent.requested_quantity),
            remaining_quantity=float(order.remaining_quantity),
            status=order.status.value,
            submit_index=int(order.intent.submit_index),
            eligible_index=int(order.intent.eligible_index),
            last_processed_index=(
                None
                if order.last_processed_index is None
                else int(order.last_processed_index)
            ),
            trigger_index=(
                None if order.trigger_index is None else int(order.trigger_index)
            ),
            expiry_index=(
                None
                if order.intent.expiry_index is None
                else int(order.intent.expiry_index)
            ),
            reduce_only=bool(order.intent.reduce_only),
        )
        for order in active_orders
    )
    timestamp_ns = int(
        observations[0].timestamp.astype("datetime64[ns]").astype("int64")
    )
    return PortfolioDecisionSnapshot(
        index=index,
        timestamp_ns=timestamp_ns,
        agent_stop_index=agent_stop_index,
        remaining_decisions=agent_stop_index - index,
        symbols=tuple(dataset.symbols),
        feature_names=tuple(dataset.feature_names),
        global_feature_names=tuple(dataset.global_feature_names),
        features=tuple(
            tuple(float(value) for value in observation.features)
            for observation in observations
        ),
        feature_available=tuple(
            tuple(bool(value) for value in observation.feature_available)
            for observation in observations
        ),
        feature_staleness=tuple(feature_staleness_rows),
        global_features=tuple(
            float(value) for value in observations[0].global_features
        ),
        global_feature_available=tuple(
            bool(value) for value in observations[0].global_feature_available
        ),
        account=account,
        active_orders=order_snapshots,
        risk=PortfolioRiskLimits(
            max_gross=float(risk_config.max_gross),
            max_abs_weight=float(risk_config.max_abs_weight),
            max_turnover=(
                None
                if risk_config.max_turnover is None
                else float(risk_config.max_turnover)
            ),
            drawdown_start=float(risk_config.drawdown_start),
            drawdown_stop=float(risk_config.drawdown_stop),
            emergency_turnover_override=bool(risk_config.emergency_turnover_override),
            fail_closed_tolerance=float(risk_config.fail_closed_tolerance),
        ),
    )


__all__ = ["build_portfolio_decision_snapshot"]
