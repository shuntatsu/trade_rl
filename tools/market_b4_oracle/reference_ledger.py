"""Independent accounting check for strictly paired native event rows.

This consumer-side ledger derives its transitions from the pinned Dataset and
execution policy, after the native fill stream has matched the Q-owned replay.
It is comparison-only: margin, order admission, receipt authentication, and
Full-B4 authority remain with their existing owners.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, cast

from tools.market_account_oracle import (
    FillEconomics,
    MarketFacts,
    MarketRow,
    ReferenceAccount,
    assert_observed_account,
    market_order_facts,
)
from tools.market_b4_oracle.expected_intent import exact_quantity
from tools.market_b4_oracle.frozen_market_replay import (
    AccountSnapshot,
    FrozenMarketBinding,
    FrozenMarketPolicy,
    ObservedFill,
    ReplayAdvance,
)
from tools.market_replay_oracle import VerifiedDatasetContent

_FILL_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class ReferenceLedgerCoverage:
    """Counts proven only after complete reference-ledger row coverage."""

    actor_rows: int
    reserve_rows: int
    fill_count: int


class _NativeReferenceLedger:
    """Recompute account arithmetic from pinned raw rows and native fills.

    The caller must first validate the paired native event stream and compare
    each row's observed fills with the independent Q replay. This class reads
    no Q-owned fill cost and makes no margin, order, or outer-attempt claim.
    """

    __slots__ = (
        "content",
        "policy",
        "binding",
        "volume_unit",
        "account",
        "next_decision_index",
        "actor_rows",
        "reserve_rows",
        "fill_count",
    )

    def __init__(
        self,
        *,
        content: VerifiedDatasetContent,
        policy: FrozenMarketPolicy,
        binding: FrozenMarketBinding,
    ) -> None:
        if (
            type(content) is not VerifiedDatasetContent
            or type(policy) is not FrozenMarketPolicy
            or type(binding) is not FrozenMarketBinding
        ):
            raise ValueError("reference requires exact frozen comparison inputs")
        binding.assert_content(content)
        if policy.dataset_id != content.dataset_id:
            raise ValueError("reference policy differs from the pinned Dataset")
        arrays = content.arrays
        selected = binding.selected_symbol_index
        volume_units = content.dataset.get("volume_units")
        if (
            type(volume_units) is not tuple
            or len(volume_units) != 2
            or volume_units[selected] != "quote_notional"
        ):
            raise ValueError(
                "fixed reference profile requires quote-notional raw volume"
            )
        marks = self._pair(arrays["mark_price"][binding.window.actor_start])
        multipliers = self._pair(arrays["contract_multipliers"])
        self.content = content
        self.policy = policy
        self.binding = binding
        self.volume_unit = cast(str, volume_units[selected])
        self.account = ReferenceAccount(
            cash=binding.initial_capital,
            quantities=("0", "0"),
            marks=marks,
            multipliers=multipliers,
        )
        self.next_decision_index = binding.window.actor_start
        self.actor_rows = 0
        self.reserve_rows = 0
        self.fill_count = 0
        if selected not in (0, 1):
            raise ValueError("native reference symbol is outside the two-slot account")

    @staticmethod
    def _pair(values: object) -> tuple[float, float]:
        try:
            items = tuple(cast(Iterable[object], values))
        except TypeError as error:
            raise ValueError(
                "raw account input must contain exactly two symbols"
            ) from error
        if len(items) != 2:
            raise ValueError("raw account input must contain exactly two symbols")
        return float(cast(Any, items[0])), float(cast(Any, items[1]))

    @staticmethod
    def _row_number(arrays: Any, name: str, row: int, symbol: int) -> float:
        return float(arrays[name][row, symbol])

    @staticmethod
    def _observed_values(snapshot: AccountSnapshot) -> dict[str, object]:
        return {
            "cash": snapshot.cash,
            "wealth": snapshot.wealth,
            "exact_quantities": tuple(
                str(value) for value in snapshot.exact_quantities
            ),
            "total_cost": snapshot.total_cost,
            "funding_pnl": snapshot.funding_pnl,
            "borrow_cost": snapshot.borrow_cost,
            "cash_interest": snapshot.cash_interest,
            "dividend": snapshot.dividend,
            "settlement_proceeds": snapshot.settlement_proceeds,
            "max_native_drawdown": snapshot.max_drawdown,
        }

    def _market_facts(
        self, processing: int, symbol: int, *, direction: int
    ) -> tuple[MarketFacts, float, float]:
        arrays = self.content.arrays
        decision = processing - 1
        current = MarketRow(
            index=processing,
            open_price=self._row_number(arrays, "open", processing, symbol),
            close_price=self._row_number(arrays, "close", processing, symbol),
            volume=self._row_number(arrays, "volume", processing, symbol),
        )
        previous = MarketRow(
            index=decision,
            open_price=self._row_number(arrays, "open", decision, symbol),
            close_price=self._row_number(arrays, "close", decision, symbol),
            volume=self._row_number(arrays, "volume", decision, symbol),
        )
        tick = (
            max(
                self._row_number(arrays, "tick_size", processing, symbol),
                self.policy.tick_size_floor,
            )
            * self.policy.tick_size_factor
        )
        lot = (
            max(
                self._row_number(arrays, "lot_size", processing, symbol),
                self.policy.lot_size_floor,
            )
            * self.policy.lot_size_factor
        )
        minimum = (
            max(
                self._row_number(arrays, "minimum_notional", processing, symbol),
                self.policy.minimum_notional_floor,
            )
            * self.policy.minimum_notional_factor
        )
        facts = market_order_facts(
            current,
            previous=previous,
            volume_unit=self.volume_unit,
            contract_multiplier=float(arrays["contract_multipliers"][symbol]),
            tick_size=tick,
            direction=direction,
            adverse_tick_rounding=self.policy.adverse_tick_rounding,
            processing_bar_volume_capacity=False,
        )
        return facts, lot, minimum

    def apply_advance(
        self,
        *,
        advance: ReplayAdvance,
        observed_fills: tuple[ObservedFill, ...],
        observed_account: AccountSnapshot,
    ) -> None:
        if type(advance) is not ReplayAdvance:
            raise ValueError("reference accounting requires an exact replay row")
        if (
            type(observed_fills) is not tuple
            or any(type(fill) is not ObservedFill for fill in observed_fills)
            or type(observed_account) is not AccountSnapshot
        ):
            raise ValueError("reference accounting requires exact observed row inputs")
        processing = advance.processing_index
        if (
            advance.decision_index != self.next_decision_index
            or processing != self.next_decision_index + 1
            or processing > self.binding.window.terminal_stop
        ):
            raise ValueError("reference rows are not a contiguous pinned sequence")
        expected_phase = (
            "actor" if processing <= self.binding.window.actor_stop else "reserve"
        )
        if advance.phase != expected_phase:
            raise ValueError("reference row phase differs from the pinned boundary")
        if (
            observed_account.insolvent
            or observed_account.termination_reason is not None
        ):
            raise ValueError("ReferenceAccount does not model terminated margin state")
        if observed_account.settlement_proceeds != 0.0:
            raise ValueError("ReferenceAccount does not model settlement transitions")

        arrays = self.content.arrays
        decision = advance.decision_index
        elapsed_ns = int(
            arrays["timestamps"][processing].astype("int64")
            - arrays["timestamps"][decision].astype("int64")
        )
        if elapsed_ns <= 0:
            raise ValueError("reference row clock must advance")
        year_fraction = elapsed_ns / (365.0 * 24 * 3600 * 1_000_000_000)
        symbol = self.binding.selected_symbol_index
        open_marks = self._pair(arrays["open"][processing])
        mark_prices = self._pair(arrays["mark_price"][processing])
        dividends = self._pair(arrays["dividend"][processing])
        funding_rates = self._pair(arrays["funding_price_rate"][processing])
        borrow_rates = self._pair(arrays["borrow_rate"][processing])
        funding_due = tuple(bool(value) for value in arrays["funding_due"][processing])
        if any(float(value) != 1.0 for value in arrays["split_factor"][processing]):
            raise ValueError("ReferenceAccount does not model split transitions")
        cash_rate = float(arrays["cash_rate"][processing])
        if cash_rate != 0.0:
            raise ValueError("fixed market profile requires zero raw cash interest")

        candidate = deepcopy(self.account)
        candidate.revalue(open_marks)
        candidate.observe()
        seen_order_ids: set[str] = set()
        for fill in observed_fills:
            if (
                fill.processing_index != processing
                or fill.order_id in seen_order_ids
                or not fill.exact_quantity
            ):
                raise ValueError("native fill row has duplicate or mismatched evidence")
            seen_order_ids.add(fill.order_id)
            direction = 1 if fill.exact_quantity > 0 else -1
            facts, lot_size, minimum_notional = self._market_facts(
                processing, symbol, direction=direction
            )
            if float(facts.execution_price) != fill.execution_price:
                raise ValueError("native fill price differs from raw Dataset execution")
            if fill.filled_lot_count is None:
                if lot_size != 0.0 or fill.lot_size != 0.0:
                    raise ValueError(
                        "unquantized fill differs from raw Dataset lot size"
                    )
                exact_lot_count = None
            else:
                if lot_size <= 0.0 or fill.lot_size != lot_size:
                    raise ValueError("native fill lot size differs from raw Dataset")
                exact_lot_count = fill.filled_lot_count
                if fill.exact_quantity != exact_lot_count * exact_quantity(lot_size):
                    raise ValueError(
                        "native fill quantity differs from exact lot count"
                    )
            economics = FillEconomics(
                fee_rate=self.policy.fee_rate,
                row_fee_rate=self._row_number(arrays, "fee_rate", processing, symbol),
                taker_fee_rate=self.policy.taker_fee_rate,
                row_taker_fee_rate=self._row_number(
                    arrays, "taker_fee_rate", processing, symbol
                ),
                spread_rate=self.policy.spread_rate,
                row_spread_rate=self._row_number(
                    arrays, "spread_rate", processing, symbol
                ),
                impact_rate=self.policy.impact_rate,
                cost_multiplier=self.policy.multiplier,
                maker=False,
            )
            calculation = candidate.apply_fill(
                symbol_index=symbol,
                reported_quantity=float(fill.exact_quantity),
                execution_price=facts.execution_price,
                economics=economics,
                market_notional=facts.market_notional,
                quantity_capacity=facts.quantity_capacity,
                lot_count=exact_lot_count,
                lot_size=lot_size,
            )
            participation_limit = min(
                self.policy.max_participation_rate,
                self._row_number(arrays, "max_participation_rate", processing, symbol),
            )
            if calculation.participation > participation_limit:
                raise ValueError("native fill exceeds pinned participation policy")
            reported_notional = (
                abs(float(fill.exact_quantity))
                * float(facts.execution_price)
                * float(arrays["contract_multipliers"][symbol])
            )
            if (
                not math.isfinite(reported_notional)
                or reported_notional + _FILL_TOLERANCE < minimum_notional
            ):
                raise ValueError("native fill is below pinned minimum notional")
        if observed_fills:
            candidate.observe()

        candidate.apply_dividend(dividends)
        candidate.apply_cash_interest(
            annual_rate=cash_rate,
            year_fraction=year_fraction,
        )
        candidate.apply_borrow(
            annual_rates=borrow_rates,
            year_fraction=year_fraction,
            rate_multiplier=self.policy.borrow_rate_multiplier,
        )
        candidate.revalue(mark_prices)
        candidate.apply_funding(price_rates=funding_rates, due=funding_due)
        candidate.observe()
        assert_observed_account(
            candidate.snapshot(), self._observed_values(observed_account)
        )

        self.account = candidate
        self.next_decision_index = processing
        if expected_phase == "actor":
            self.actor_rows += 1
        else:
            self.reserve_rows += 1
        self.fill_count += len(observed_fills)

    def assert_complete(self) -> ReferenceLedgerCoverage:
        if (
            self.next_decision_index != self.binding.window.terminal_stop
            or self.actor_rows != self.binding.window.actor_advances
            or self.reserve_rows != self.binding.window.reserve_advances
        ):
            raise ValueError("independent reference ledger lacks complete row coverage")
        if any(self.account.quantities):
            raise ValueError("independent reference ledger did not finish flat")
        return ReferenceLedgerCoverage(
            actor_rows=self.actor_rows,
            reserve_rows=self.reserve_rows,
            fill_count=self.fill_count,
        )
