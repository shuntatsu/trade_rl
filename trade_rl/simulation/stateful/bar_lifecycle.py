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
    gap_cash_carry_delta: float


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

        gap_cash_carry_delta = 0.0
        if gap_year_fraction > 0.0:
            cash_before_gap_carry = runtime.book.cash
            annual_rate = float(dataset.resolved_array("cash_rate")[processing_index])
            cash_interest_before = runtime.capture_accounting_state()
            gap_interest = runtime.book.apply_cash_interest(
                annual_rate,
                year_fraction=gap_year_fraction,
            )
            runtime.record_accounting_transition(
                transition_type="cash_interest",
                processing_index=processing_index,
                state_before=cash_interest_before,
                evidence={
                    "annual_rate": annual_rate,
                    "basis_adjustment": 0.0,
                    "carry_phase": "gap",
                    "year_fraction": float(gap_year_fraction),
                },
            )
            runtime.total_cash_interest += gap_interest

            borrow_before = runtime.capture_accounting_state()
            borrow_rates = dataset.resolved_array("borrow_rate")[processing_index]
            gap_borrow = executor._charge_borrow(
                runtime.book,
                index=processing_index,
                year_fraction=gap_year_fraction,
            )
            runtime.record_accounting_transition(
                transition_type="borrow_charge",
                processing_index=processing_index,
                state_before=borrow_before,
                evidence={
                    "borrow_amount": float(gap_borrow),
                    "borrow_rate": tuple(float(value) for value in borrow_rates),
                    "borrow_rate_multiplier": float(
                        executor.cost.borrow_rate_multiplier
                    ),
                    "carry_phase": "gap",
                    "year_fraction": float(gap_year_fraction),
                },
            )
            runtime.total_borrow += gap_borrow
            gap_cash_carry_delta = runtime.book.cash - cash_before_gap_carry
            runtime.book.refresh_drawdown()

        split = dataset.resolved_array("split_factor")[processing_index]
        if np.any(split != 1.0):
            split_mask = np.abs(split - 1.0) > _TOLERANCE
            if np.any(split_mask):
                runtime.cancel_active_orders(
                    processing_index=processing_index,
                    reason="split_adjustment_required",
                    symbol_mask=split_mask,
                )
            split_before = runtime.capture_accounting_state()
            runtime.book.apply_split(split)
            runtime.record_accounting_transition(
                transition_type="split",
                processing_index=processing_index,
                state_before=split_before,
                evidence={"split_factors": tuple(float(value) for value in split)},
            )

        inactive = ~dataset.resolved_array("asset_active")[processing_index]
        if np.any(inactive):
            runtime.cancel_active_orders(
                processing_index=processing_index,
                reason="inactive_asset",
                symbol_mask=inactive,
            )
            if np.any(inactive & (np.abs(runtime.book.quantities) > _TOLERANCE)):
                settlement_prices = dataset.open[processing_index]
                recovery = dataset.resolved_array("delisting_recovery")[
                    processing_index
                ]
                settlement_before = runtime.capture_accounting_state()
                runtime.book.settle_positions(
                    mask=inactive,
                    prices=settlement_prices,
                    recovery=recovery,
                )
                runtime.record_accounting_transition(
                    transition_type="delisting_settlement",
                    processing_index=processing_index,
                    state_before=settlement_before,
                    evidence={
                        "inactive_mask": tuple(bool(value) for value in inactive),
                        "open_prices": tuple(
                            float(value) for value in settlement_prices
                        ),
                        "delisting_recovery": tuple(float(value) for value in recovery),
                    },
                )

        open_prices = dataset.open[processing_index]
        open_revalue_before = runtime.capture_accounting_state()
        runtime.book.revalue(open_prices)
        runtime.record_accounting_transition(
            transition_type="mark_revaluation",
            processing_index=processing_index,
            state_before=open_revalue_before,
            evidence={
                "mark_phase": "open",
                "mark_prices": tuple(float(value) for value in open_prices),
            },
        )
        runtime.book.refresh_drawdown()
        runtime.record_ohlc_drawdown_stress(
            processing_index=processing_index,
            phase="pre_fill",
        )
        if gap_year_fraction > 0.0:
            executor._update_margin(
                runtime.book,
                processing_index=processing_index,
            )
            if runtime.book.insolvent:
                runtime.cancel_active_orders(
                    processing_index=processing_index,
                    reason="economic_termination",
                )
                executor._flatten_after_termination(
                    runtime.book,
                    open_prices,
                    processing_index=processing_index,
                )

        tick, lot, minimum = executor._effective_rule_array_views(
            index=processing_index
        )
        return StatefulBarContext(
            previous_index=previous_index,
            processing_index=processing_index,
            period_start_value=period_start_value,
            open_prices=open_prices,
            tick_size=tick,
            lot_size=lot,
            minimum_notional=minimum,
            processing_year_fraction=processing_year_fraction,
            gap_cash_carry_delta=gap_cash_carry_delta,
        )

    def _apply_processing_cash_interest(
        self,
        runtime: StatefulExecutionRuntime,
        context: StatefulBarContext,
    ) -> float:
        dataset = runtime.executor.dataset
        annual_rate = float(
            dataset.resolved_array("cash_rate")[context.processing_index]
        )
        if abs(context.gap_cash_carry_delta) <= _TOLERANCE:
            return runtime.book.apply_cash_interest(
                annual_rate,
                year_fraction=context.processing_year_fraction,
            )

        interest_basis = runtime.book.clone()
        interest_basis.cash -= context.gap_cash_carry_delta
        amount = interest_basis.apply_cash_interest(
            annual_rate,
            year_fraction=context.processing_year_fraction,
        )
        runtime.book.cash += amount
        runtime.book.revalue(runtime.book.mark_prices)
        return amount

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
            executor._flatten_after_termination(
                runtime.book,
                context.open_prices,
                processing_index=processing_index,
            )

        dividends = dataset.resolved_array("dividend")[processing_index]
        dividend_before = runtime.capture_accounting_state()
        dividend_amount = runtime.book.apply_dividend(dividends)
        runtime.record_accounting_transition(
            transition_type="dividend",
            processing_index=processing_index,
            state_before=dividend_before,
            evidence={
                "dividend_per_unit": tuple(float(value) for value in dividends),
                "dividend_amount": float(dividend_amount),
            },
        )
        runtime.total_dividend += dividend_amount

        cash_rate = float(dataset.resolved_array("cash_rate")[processing_index])
        interest_before = runtime.capture_accounting_state()
        processing_interest = self._apply_processing_cash_interest(
            runtime,
            context,
        )
        runtime.record_accounting_transition(
            transition_type="cash_interest",
            processing_index=processing_index,
            state_before=interest_before,
            evidence={
                "annual_rate": cash_rate,
                "basis_adjustment": float(context.gap_cash_carry_delta),
                "carry_phase": "processing",
                "year_fraction": float(context.processing_year_fraction),
            },
        )
        runtime.total_cash_interest += processing_interest

        borrow_before = runtime.capture_accounting_state()
        borrow_rates = dataset.resolved_array("borrow_rate")[processing_index]
        funding_amount, borrow_amount = executor._charge_carry(
            runtime.book,
            index=processing_index,
            year_fraction=context.processing_year_fraction,
        )
        runtime.record_accounting_transition(
            transition_type="borrow_charge",
            processing_index=processing_index,
            state_before=borrow_before,
            evidence={
                "borrow_amount": float(borrow_amount),
                "borrow_rate": tuple(float(value) for value in borrow_rates),
                "borrow_rate_multiplier": float(executor.cost.borrow_rate_multiplier),
                "carry_phase": "processing",
                "year_fraction": float(context.processing_year_fraction),
            },
        )
        runtime.total_funding += funding_amount
        runtime.total_borrow += borrow_amount
        mark_prices = dataset.resolved_array("mark_price")[processing_index]
        funding_before = runtime.capture_accounting_state()
        runtime.book.mark_to_market(
            mark_prices=mark_prices,
            funding_amount=funding_amount,
            period_start_value=context.period_start_value,
        )
        runtime.record_accounting_transition(
            transition_type="funding_mark",
            processing_index=processing_index,
            state_before=funding_before,
            evidence={
                "funding_amount": float(funding_amount),
                "mark_prices": tuple(float(value) for value in mark_prices),
            },
        )
        runtime.record_funding_boundary(
            processing_index=processing_index,
            funding_amount=funding_amount,
        )
        runtime.record_ohlc_drawdown_stress(
            processing_index=processing_index,
            phase="post_fill",
        )
        executor._update_margin(
            runtime.book,
            processing_index=processing_index,
        )
        if runtime.book.insolvent:
            runtime.cancel_active_orders(
                processing_index=processing_index,
                reason="economic_termination",
            )
            executor._flatten_after_termination(
                runtime.book,
                dataset.resolved_array("mark_price")[processing_index],
                processing_index=processing_index,
            )
