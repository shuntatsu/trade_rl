"""Source-backed completed-arm verification for the adapted Task18 replay.

The maintained independent cash/quantity/mark oracle is supplemented by an
actual-fill fee/spread/impact and carry reconstruction. Expected cost beliefs
are never booked as realized costs. This software checker is not an
authenticated external verification report or an economic authorization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from tools.tier_d_nonrl.task18_costs import native_execution_cost
from tools.tier_d_nonrl.task18_frozen_inputs import FrozenTask18Inputs
from tools.tier_d_nonrl.task18_recipe import build_task18_protocol
from trade_rl.data.contracts import VolumeUnit
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.decision_value_ledger_oracle import (
    verify_paired_economic_ledger,
)
from trade_rl.evaluation.replay import SharedCashPortfolioReplayResult
from trade_rl.simulation.execution import MarketExecutor

_PROTOCOL = build_task18_protocol()
_PLAN = _PROTOCOL.index_plan
_ECONOMICS = _PROTOCOL.economics
_ARMS = {arm.arm_id: arm for arm in _PROTOCOL.arms}


@dataclass(frozen=True, slots=True)
class NativeArmVerification:
    passed: bool
    reason: str
    advances: int
    actor_decisions: int
    terminal_advances: int
    terminal_flat: bool
    pending_orders: int
    cost_oracle_passed: bool


def _near(left: float, right: float, *, atol: float = 1e-8) -> bool:
    return math.isclose(left, right, rel_tol=1e-11, abs_tol=atol)


def _costs_match(
    dataset: MarketDataset, arm_id: str, result: SharedCashPortfolioReplayResult
) -> bool:
    wrapped = result.ledger_evidence
    if wrapped is None or dataset.funding_price_rate is None:
        return False
    if any(unit != VolumeUnit.QUOTE_NOTIONAL for unit in dataset.volume_units):
        return False
    arm = _ARMS[arm_id]
    symbol = dataset.symbols.index(arm.symbol)
    cost = native_execution_cost(arm.scenario)
    fee = dataset.resolved_array("fee_rate")
    taker = dataset.resolved_array("taker_fee_rate")
    spread = dataset.resolved_array("spread_rate")
    limit = dataset.resolved_array("max_participation_rate")
    due = dataset.resolved_array("funding_due")
    marks = dataset.resolved_array("mark_price")
    multipliers = dataset.resolved_array("contract_multipliers")
    clocks = dataset.timestamps.astype("datetime64[ns]").astype("int64")
    active: dict[str, int] = {}
    seen: set[str] = set()
    for interval in wrapped.intervals.intervals:
        i = interval.next_index
        if (
            len(interval.exact_quantities_after) != len(dataset.symbols)
            or len(interval.exact_quantities_before) != len(dataset.symbols)
            or any(
                Fraction(q) != 0
                for j, q in enumerate(interval.exact_quantities_before)
                if j != symbol
            )
            or any(
                Fraction(q) != 0
                for j, q in enumerate(interval.exact_quantities_after)
                if j != symbol
            )
            or any(event.symbol_index != symbol for event in interval.order_events)
        ):
            return False
        if arm.role == "CASH" and (
            interval.order_events
            or any(Fraction(q) != 0 for q in interval.exact_quantities_before)
            or any(Fraction(q) != 0 for q in interval.exact_quantities_after)
        ):
            return False
        for event in interval.order_events:
            if event.event_type == "submitted":
                if (
                    event.order_id in seen
                    or event.processing_index != interval.start_index
                ):
                    return False
                seen.add(event.order_id)
                active[event.order_id] = event.processing_index
            elif event.order_id not in active:
                return False
            if event.filled_quantity != 0 and event.processing_index < (
                active[event.order_id] + max(1, cost.order_latency_bars)
            ):
                return False
            if event.new_status.terminal:
                active.pop(event.order_id)
        actual_fills = [e for e in interval.order_events if e.filled_quantity != 0]
        notional_sum = 0.0
        charges = []
        prior_volume = float(dataset.volume[i - 1, symbol])
        for fill in actual_fills:
            if (
                prior_volume <= 0
                or fill.trigger_segment != "open"
                or fill.available_volume_fraction != 1.0
            ):
                return False
            notional = fill.filled_notional
            participation = notional / prior_volume
            if not _near(fill.participation_rate, participation, atol=1e-14):
                return False
            notional_sum += notional
            charges.append(
                notional
                * cost.multiplier
                * (
                    float(fee[i, symbol])
                    + float(taker[i, symbol])
                    + float(spread[i, symbol])
                    + cost.impact_rate * math.sqrt(participation)
                )
            )
        if (
            notional_sum
            > prior_volume * min(cost.max_participation_rate, float(limit[i, symbol]))
            + 1e-8
        ):
            return False
        if not _near(interval.interval_cost, math.fsum(charges)):
            return False
        # Task18 permits only no-split, active accounts and zero cash/borrow
        # rates. Funding is charged on post-fill positions at the source row.
        if (
            np.any(dataset.resolved_array("split_factor")[i] != 1.0)
            or np.any(dataset.resolved_array("dividend")[i] != 0.0)
            or not np.all(dataset.resolved_array("asset_active")[i])
            or np.any(dataset.resolved_array("borrow_rate")[i] != 0.0)
            or dataset.resolved_array("cash_rate")[i] != 0.0
            or interval.interval_borrow_cost != 0.0
            or interval.interval_cash_interest != 0.0
            or interval.interval_dividend != 0.0
        ):
            return False
        quantities = tuple(float(Fraction(q)) for q in interval.exact_quantities_after)
        price_rates = tuple(float(x) for x in dataset.funding_price_rate[i])
        if not np.array_equal(
            dataset.funding_price_rate[i], dataset.funding_rate[i] * marks[i]
        ):
            return False
        expected_funding = -math.fsum(
            quantities[j] * float(multipliers[j]) * price_rates[j]
            for j in range(len(quantities))
            if due[i, j]
        )
        if not _near(interval.interval_funding, expected_funding):
            return False
        if len(interval.funding_events) != int(np.any(due[i])):
            return False
        for funding_event in interval.funding_events:
            if (
                funding_event.processing_index != i
                or funding_event.timestamp_ns != int(clocks[i])
                or funding_event.funding_due != tuple(bool(x) for x in due[i])
                or funding_event.mark_prices != tuple(float(x) for x in marks[i])
                or funding_event.contract_multipliers
                != tuple(float(x) for x in multipliers)
                or funding_event.funding_rates
                != tuple(float(x) for x in dataset.funding_rate[i])
                or funding_event.funding_price_rates != price_rates
                or len(funding_event.signed_quantities) != len(quantities)
                or not all(
                    _near(a, b, atol=1e-10)
                    for a, b in zip(
                        funding_event.signed_quantities, quantities, strict=True
                    )
                )
                or not _near(funding_event.funding_amount, expected_funding)
                or not _near(
                    funding_event.equity_after_funding, interval.portfolio_value_after
                )
            ):
                return False
    return not active


def _decisions_match(
    dataset: MarketDataset,
    arm_id: str,
    result: SharedCashPortfolioReplayResult,
    frozen: FrozenTask18Inputs,
) -> bool:
    wrapped = result.ledger_evidence
    if wrapped is None or frozen.binding.dataset_id != dataset.dataset_id:
        return False
    arm = _ARMS[arm_id]
    symbol = dataset.symbols.index(arm.symbol)
    marks = dataset.resolved_array("mark_price")
    multiplier = float(dataset.resolved_array("contract_multipliers")[symbol])
    risk = _PROTOCOL.risk
    for decision, interval in zip(
        wrapped.decisions,
        wrapped.intervals.intervals[: len(wrapped.decisions)],
        strict=True,
    ):
        i = decision.index
        q = Fraction(interval.exact_quantities_before[symbol])
        quantity = float(q)
        # Reproduce the conservative binary64 projection contract from the
        # exact rational evidence, not a live BookState or producer policy.
        while abs(Fraction(str(quantity))) > abs(q):
            quantity = math.nextafter(quantity, 0.0)
        current = (
            quantity * float(marks[i, symbol]) * multiplier
        ) / interval.portfolio_value_before
        if arm.role == "CASH":
            proposed = 0.0
        else:
            row = frozen.lookup_nonrl(
                arm_id,
                i,
                decision_time_ns=int(
                    dataset.timestamps[i].astype("datetime64[ns]").astype("int64")
                ),
            )
            if (
                row.feature_values
                != tuple(float(x) for x in dataset.features[i, symbol])
                or row.decision_close != float(marks[i, symbol])
                or not all(dataset.feature_available[i, symbol])
                or row.source_available_at_ns
                != int(
                    dataset.resolved_array("available_at")[i, symbol]
                    .astype("datetime64[ns]")
                    .astype("int64")
                )
            ):
                return False
            x = Fraction(current)
            mu = Fraction(row.expected_simple_return)
            funding = Fraction(row.funding_return)
            rate = Fraction(row.buy_cost)
            # Compare the archived affine-segment endpoint and kink utilities
            # using a separate direct formula; no solver or strategy invocation.
            candidates = tuple(
                sorted(
                    {-1.0, 0.0, 1.0, *((current,) if -1.0 <= current <= 1.0 else ())}
                )
            )

            def utility(w: float) -> tuple[Fraction, bool, Fraction, float, float]:
                y = Fraction(w)
                return (
                    mu * y
                    - funding * y
                    - rate * max(y - x, Fraction(0))
                    - rate * max(x - y, Fraction(0))
                    - rate * abs(y),
                    w == current,
                    -abs(y - x),
                    -abs(w),
                    -w,
                )

            proposed = max(candidates, key=utility)
        target = proposed
        dd = interval.max_drawdown_before
        if not math.isfinite(dd) or not 0.0 <= dd <= 1.0:
            return False
        scale = (
            1.0
            if dd <= risk.drawdown_start
            else (
                0.0
                if dd >= risk.drawdown_stop
                else 1.0
                - (dd - risk.drawdown_start)
                / (risk.drawdown_stop - risk.drawdown_start)
            )
        )
        target *= scale
        expected_proposal = tuple(
            proposed if j == symbol else 0.0 for j in range(len(dataset.symbols))
        )
        expected_target = tuple(
            target if j == symbol else 0.0 for j in range(len(dataset.symbols))
        )
        expected_holds = tuple(range(len(dataset.symbols))) if target == current else ()
        if (
            decision.proposal_weights != expected_proposal
            or decision.target_weights != expected_target
            or decision.requested_hold_symbols != ()
            or decision.effective_hold_symbols != expected_holds
        ):
            return False
    return True


def verify_native_arm(
    dataset: MarketDataset,
    arm_id: str,
    result: SharedCashPortfolioReplayResult,
    *,
    frozen: FrozenTask18Inputs,
) -> NativeArmVerification:
    """Verify only a completed, fully settled native arm; reject partial claims."""
    if arm_id not in _ARMS:
        raise ValueError("unknown Task18 verification arm")
    wrapped = result.ledger_evidence
    if wrapped is None:
        return NativeArmVerification(False, "missing_ledger", 0, 0, 0, False, 0, False)
    ledger = wrapped.intervals
    advances = len(ledger.intervals)
    actor = len(wrapped.decisions)
    terminal = max(0, advances - actor)
    pending = len(ledger.active_order_remainders)
    try:
        flat = all(Fraction(q) == 0 for q in ledger.terminal_exact_quantities) and all(
            q == 0 for q in result.book.exact_quantities
        )
        expected_policy = MarketExecutor(
            dataset,
            native_execution_cost(_ARMS[arm_id].scenario),
            insolvency_valuation="retain_debt",
        ).execution_policy_digest
        complete = (
            dataset.symbols == _PROTOCOL.dataset.symbols
            and dataset.identity_verified
            and ledger.dataset_id == dataset.dataset_id
            and ledger.execution_policy_digest == expected_policy
            and advances == _PLAN.native_advances_per_arm
            and actor == _PLAN.actor_stop_exclusive - _PLAN.actor_start_inclusive
            and terminal == _PLAN.terminal_advance_count
            and ledger.start_index == _PLAN.actor_start_inclusive
            and ledger.stop_index == _PLAN.terminal_stop_inclusive
            and flat
            and pending == 0
            and wrapped.decisions == result.decisions
            and tuple(d.index for d in wrapped.decisions)
            == tuple(range(_PLAN.actor_start_inclusive, _PLAN.actor_stop_exclusive))
            and len(result.returns.values) == advances
            and tuple(v.interval_net_return for v in ledger.intervals)
            == result.returns.values
        )
        if not complete:
            return NativeArmVerification(
                False,
                "incomplete_native_coverage",
                advances,
                actor,
                terminal,
                flat,
                pending,
                False,
            )
        # This oracle expects the raw base digest if execution_cost is passed.
        # We checked the actual retain-debt + v7 funding wrapper above.
        if not verify_paired_economic_ledger(
            result,
            dataset,
            expected_intervals=advances,
            initial_capital=_ECONOMICS.initial_capital_per_symbol_usdt,
            expected_start_index=_PLAN.actor_start_inclusive,
        ):
            return NativeArmVerification(
                False,
                "source_ledger_mismatch",
                advances,
                actor,
                terminal,
                flat,
                pending,
                False,
            )
        if not _decisions_match(dataset, arm_id, result, frozen):
            return NativeArmVerification(
                False,
                "frozen_decision_mismatch",
                advances,
                actor,
                terminal,
                flat,
                pending,
                False,
            )
        cost_ok = _costs_match(dataset, arm_id, result)
        return NativeArmVerification(
            cost_ok,
            "verified" if cost_ok else "source_cost_mismatch",
            advances,
            actor,
            terminal,
            flat,
            pending,
            cost_ok,
        )
    except (ValueError, TypeError, AttributeError, IndexError, ArithmeticError):
        return NativeArmVerification(
            False,
            "invalid_native_evidence",
            advances,
            actor,
            terminal,
            False,
            pending,
            False,
        )
