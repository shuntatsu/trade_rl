"""Corporate-action and end-of-bar lifecycle for stateful execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from trade_rl.simulation.stateful.runtime import StatefulExecutionRuntime

if TYPE_CHECKING:
    from trade_rl.simulation.execution import MarketExecutor


_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class StatefulBarContext:
    previous_index: int
    processing_index: int
    period_start_value: float
    open_prices: np.ndarray
    tick_size: np.ndarray
    lot_size: np.ndarray
    minimum_notional: np.ndarray
    processing_year_fraction: float


class StatefulBarLifecycle:
    """Apply the exact pre-fill and post-fill accounting phases for one bar."""

    def __init__(self, executor: MarketExecutor) -> None:
        self.executor = executor

    def begin_bar(
        self,
        runtime: StatefulExecutionRuntime,
        *,
        previous_index: int,
        processing_index: int,
    ) -> StatefulBarContext:
        executor = runtime.executor
        dataset = executor.dataset
        period_start_value = max(runtime.book.portfolio_value, _TOLERANCE)

        elapsed_hours = dataset.elapsed_hours(previous_index, processing_index)
        elapsed_year_fraction = dataset.elapsed_year_fraction(
            previous_index,
            processing_index,
        )
        if elapsed_hours <= dataset.bar_hours + _TOLERANCE:
            processing_year_fraction = elapsed_year_fraction
            gap_year_fraction = 0.0
        else:
            processing_year_fraction = (
                elapsed_year_fraction * dataset.bar_hours / elapsed_hours
            )
            gap_year_fraction = elapsed_year_fraction - processing_year_fraction

        if gap_year_fraction > 0.0:
            runtime.total_cash_interest += runtime.book.apply_cash_interest(
                float(dataset.resolved_array("cash_rate")[processing_index]),
                year_fraction=gap_year_fraction,
            )
            runtime.total_borrow += executor._charge_borrow(
                runtime.book,
                index=processing_index,
                year_fraction=gap_year_fraction,
            )
            executor._update_margin(runtime.book)
            if runtime.book.insolvent:
                runtime.cancel_active_orders(
                    processing_index=processing_index,
                    reason="economic_termination",
                )
                executor._flatten_after_termination(
                    runtime.book,
                    runtime.book.mark_prices,
                )

        split = dataset.resolved_array("split_factor")[processing_index]
        split_mask = np.abs(split - 1.0) > _TOLERANCE
        if np.any(split_mask):
            runtime.cancel_active_orders(
                processing_index=processing_index,
                reason="split_adjustment_required",
                symbol_mask=split_mask,
            )
        runtime.book.apply_split(split)

        inactive = ~dataset.resolved_array("asset_active")[processing_index]
        if np.any(inactive):
            runtime.cancel_active_orders(
                processing_index=processing_index,
                reason="inactive_asset",
                symbol_mask=inactive,
            )
            if np.any(inactive & (np.abs(runtime.book.quantities) > _TOLERANCE)):
                runtime.book.settle_positions(
                    mask=inactive,
                    prices=dataset.open[processing_index],
                    recovery=dataset.resolved_array("delisting_recovery")[
                        processing_index
                    ],
                )

        open_prices = dataset.open[processing_index]
        runtime.book.revalue(open_prices)
        runtime.book.refresh_drawdown()

        tick, lot, minimum = executor.effective_rule_arrays(index=processing_index)
        return StatefulBarContext(
            previous_index=previous_index,
            processing_index=processing_index,
            period_start_value=period_start_value,
            open_prices=open_prices,
            tick_size=tick,
            lot_size=lot,
            minimum_notional=minimum,
            processing_year_fraction=processing_year_fraction,
        )

    def finish_bar(
        self,
        runtime: StatefulExecutionRuntime,
        context: StatefulBarContext,
    ) -> None:
        executor = runtime.executor
        dataset = executor.dataset
        processing_index = context.processing_index

        if runtime.book.insolvent:
            runtime.cancel_active_orders(
                processing_index=processing_index,
                reason="economic_termination",
            )
            executor._flatten_after_termination(runtime.book, context.open_prices)

        runtime.total_dividend += runtime.book.apply_dividend(
            dataset.resolved_array("dividend")[processing_index]
        )
        runtime.total_cash_interest += runtime.book.apply_cash_interest(
            float(dataset.resolved_array("cash_rate")[processing_index]),
            year_fraction=context.processing_year_fraction,
        )
        funding_amount, borrow_amount = executor._charge_carry(
            runtime.book,
            index=processing_index,
            year_fraction=context.processing_year_fraction,
        )
        runtime.total_funding += funding_amount
        runtime.total_borrow += borrow_amount
        runtime.book.mark_to_market(
            mark_prices=dataset.resolved_array("mark_price")[processing_index],
            funding_amount=funding_amount,
            period_start_value=context.period_start_value,
        )
        runtime.record_funding_boundary(
            processing_index=processing_index,
            funding_amount=funding_amount,
        )
        executor._update_margin(runtime.book)
        if runtime.book.insolvent:
            runtime.cancel_active_orders(
                processing_index=processing_index,
                reason="economic_termination",
            )
            executor._flatten_after_termination(
                runtime.book,
                dataset.resolved_array("mark_price")[processing_index],
            )
