"""Task18 synthetic native-arm composition on the maintained replay owner.

No market loader or economic admission endpoint. An explicitly artificial
in-memory Dataset and a pre-stop pinned adapted frozen plan are required.
The owner must keep these inputs exclusive and unchanged. Completed-arm
observations are not authenticated external receipts or authorization.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from tools.tier_d_nonrl.task18_costs import native_execution_cost
from tools.tier_d_nonrl.task18_frozen_inputs import (
    FrozenInputBinding,
    FrozenInputsPin,
    FrozenTask18Inputs,
    encode_frozen_inputs,
    load_frozen_inputs,
)
from tools.tier_d_nonrl.task18_native_oracle import (
    NativeArmVerification,
    verify_native_arm,
)
from tools.tier_d_nonrl.task18_recipe import ArmSpec, build_task18_protocol
from trade_rl.data.contracts import MarketCalendarKind, VolumeUnit
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.replay import (
    SharedCashPortfolioReplayResult,
    run_shared_portfolio_replay,
)
from trade_rl.risk import PreTradeRisk, PreTradeRiskConfig
from trade_rl.strategies.portfolio import (
    PortfolioDecisionSnapshot,
    PortfolioTargetProposal,
)

_PROTOCOL = build_task18_protocol()
_PLAN = _PROTOCOL.index_plan
_ECONOMICS = _PROTOCOL.economics
_RISK = _PROTOCOL.risk
_ARMS = {arm.arm_id: arm for arm in _PROTOCOL.arms}


def choose_target_weight(
    *,
    current_weight: float,
    expected_simple_return: float,
    one_way_cost: float,
    funding_return: float,
) -> float:
    """Exact archived Task18 risk-neutral, symmetric-cost scalar objective.

    The old general allocator has zero quadratic risk penalty, zero cash and
    borrow returns, symmetric buy/sell/exit beliefs and [-1,1] bounds here.
    Its piecewise-linear maximum is at a bound, zero, or current weight.
    Compare the actual supplied IEEE values, not rounded endpoint scores.
    """
    for value in (current_weight, expected_simple_return, one_way_cost, funding_return):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not math.isfinite(value)
        ):
            raise ValueError("Task18 objective inputs must be finite real numbers")
    if one_way_cost < 0 or expected_simple_return < -1:
        raise ValueError("Task18 objective input outside its domain")
    current = float(current_weight)
    w0 = Fraction(current)
    signal = Fraction(float(expected_simple_return)) - Fraction(float(funding_return))
    cost = Fraction(float(one_way_cost))
    candidates = {-1.0, 0.0, 1.0}
    if -1.0 <= current <= 1.0:
        candidates.add(current)

    def key(weight: float) -> tuple[Fraction, bool, Fraction, float, float]:
        w = Fraction(weight)
        return (
            signal * w - cost * abs(w - w0) - cost * abs(w),
            weight == current,
            -abs(w - w0),
            -abs(weight),
            -weight,
        )

    return max(candidates, key=key)


def _risk() -> PreTradeRisk:
    return PreTradeRisk(
        PreTradeRiskConfig(
            max_gross=_RISK.max_gross,
            max_abs_weight=_RISK.max_abs_weight,
            max_turnover=_RISK.turnover_cap,
            drawdown_start=_RISK.drawdown_start,
            drawdown_stop=_RISK.drawdown_stop,
            emergency_turnover_override=_RISK.emergency_turnover_override,
            fail_closed_tolerance=_RISK.tolerance,
        )
    )


class _IndependentArmPolicy:
    """One selected symbol in a fresh independent book over the full Dataset."""

    def __init__(
        self, dataset: MarketDataset, frozen: FrozenTask18Inputs, arm: ArmSpec
    ):
        self._dataset = dataset
        self._frozen = frozen
        self._arm = arm
        self._symbol = dataset.symbols.index(arm.symbol)
        self._next_index = _PLAN.actor_start_inclusive

    def decide_portfolio(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> PortfolioTargetProposal:
        s = self._symbol
        index = snapshot.index
        account = snapshot.account
        if (
            index != self._next_index
            or snapshot.agent_stop_index != _PLAN.actor_stop_exclusive
            or snapshot.symbols != self._dataset.symbols
            or snapshot.feature_names != self._dataset.feature_names
            or snapshot.timestamp_ns
            != int(
                self._dataset.timestamps[index]
                .astype("datetime64[ns]")
                .astype(np.int64)
            )
            or account.insolvent
            or account.termination_reason is not None
            or account.portfolio_value <= 0.0
            or any(
                Fraction(q) != 0
                for j, q in enumerate(account.exact_quantities)
                if j != s
            )
            or any(
                o.symbol_index != s or o.order_type != "market"
                for o in snapshot.active_orders
            )
            or account.mark_prices
            != tuple(
                float(x) for x in self._dataset.resolved_array("mark_price")[index]
            )
        ):
            raise ValueError("native snapshot differs from independent Task18 arm")
        current = account.weights[s]
        if self._arm.role == "CASH":
            if (
                any(Fraction(q) != 0 for q in account.exact_quantities)
                or snapshot.active_orders
            ):
                raise ValueError("CASH control must stay flat")
            target = 0.0
        else:
            row = self._frozen.lookup_nonrl(
                self._arm.arm_id, index, decision_time_ns=snapshot.timestamp_ns
            )
            if (
                row.feature_values != snapshot.features[s]
                or not all(snapshot.feature_available[s])
                or row.decision_close != account.mark_prices[s]
                or row.decision_close != self._dataset.close[index, s]
                or row.source_available_at_ns
                != int(
                    self._dataset.resolved_array("available_at")[index, s]
                    .astype("datetime64[ns]")
                    .astype(np.int64)
                )
                or not self._dataset.resolved_array("information_available")[index, s]
            ):
                raise ValueError(
                    "native decision snapshot differs from the frozen inputs"
                )
            target = choose_target_weight(
                current_weight=current,
                expected_simple_return=row.expected_simple_return,
                one_way_cost=row.buy_cost,
                funding_return=row.funding_return,
            )
        # The native owner applies final risk once, then chooses exact HOLD
        # when the entire final vector is unchanged. No second ledger.
        weights = tuple(target if j == s else 0.0 for j in range(len(snapshot.symbols)))
        self._next_index += 1
        return PortfolioTargetProposal(weights)


@dataclass(frozen=True, slots=True)
class NativeArmObservation:
    arm_id: str
    result: SharedCashPortfolioReplayResult
    verification: NativeArmVerification

    @property
    def execution_authorized(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class SyntheticNativeAttempt:
    dataset: MarketDataset
    frozen: FrozenTask18Inputs
    plan_pin: FrozenInputsPin

    def __post_init__(self) -> None:
        if (
            type(self.frozen) is not FrozenTask18Inputs
            or type(self.plan_pin) is not FrozenInputsPin
        ):
            raise ValueError("native attempt requires typed frozen plan and pin")
        if self.frozen.binding.scope != "SYNTHETIC_ONLY":
            raise ValueError(
                "native attempt is synthetic-only; historical execution is not admitted"
            )
        raw = encode_frozen_inputs(self.frozen)
        if (
            len(raw) != self.plan_pin.size_bytes
            or hashlib.sha256(raw).hexdigest() != self.plan_pin.sha256
        ):
            raise ValueError("native attempt does not match the frozen plan pin")
        dataset = self.dataset
        if (
            not dataset.identity_verified
            or dataset.dataset_id != self.frozen.binding.dataset_id
            or dataset.symbols != _PROTOCOL.dataset.symbols
            or dataset.feature_names != _PROTOCOL.dataset.feature_names
            or dataset.n_bars != _PROTOCOL.dataset.row_count
            or dataset.calendar_kind != MarketCalendarKind.CONTINUOUS
            or not dataset.regular_cadence
            or dataset.bar_hours != 1.0
            or any(unit != VolumeUnit.QUOTE_NOTIONAL for unit in dataset.volume_units)
            or not np.array_equal(
                dataset.timestamps.astype("datetime64[ns]").astype(np.int64),
                int(
                    np.datetime64(_PROTOCOL.dataset.development_start, "ns").astype(
                        np.int64
                    )
                )
                + np.arange(_PROTOCOL.dataset.row_count, dtype=np.int64)
                * 3_600_000_000_000,
            )
            or dataset.funding_price_rate is None
        ):
            raise ValueError("native attempt Dataset differs from frozen plan")
        first, stop = _PLAN.actor_start_inclusive, _PLAN.actor_stop_exclusive
        native = slice(first, _PLAN.terminal_stop_inclusive + 1)
        for field, expected in (
            ("fee_rate", _ECONOMICS.base_fee_rate),
            ("maker_fee_rate", 0.0),
            ("taker_fee_rate", 0.0),
            ("spread_rate", _ECONOMICS.base_spread_rate),
            ("max_participation_rate", _ECONOMICS.maximum_participation_rate),
            ("borrow_available", True),
        ):
            if not np.all(dataset.resolved_array(field)[native] == expected):
                raise ValueError(
                    f"native attempt differs from frozen execution declaration: {field}"
                )
        snapshots = self.frozen.snapshots
        if (
            not np.array_equal(
                snapshots[..., :-1], dataset.features[first:stop].transpose(1, 0, 2)
            )
            or not np.array_equal(snapshots[..., -1], dataset.close[first:stop].T)
            or not np.array_equal(
                self.frozen.source_available_at_ns,
                dataset.resolved_array("available_at")[first:stop]
                .astype("datetime64[ns]")
                .astype(np.int64)
                .T,
            )
            or not dataset.feature_available[first:stop].all()
            or not dataset.resolved_array("information_available")[native].all()
            or not np.all(dataset.resolved_array("split_factor")[native] == 1.0)
            or not np.all(dataset.resolved_array("dividend")[native] == 0.0)
            or not np.all(dataset.resolved_array("asset_active")[native])
            or not np.all(dataset.resolved_array("borrow_rate")[native] == 0.0)
            or not np.all(dataset.resolved_array("cash_rate")[native] == 0.0)
            or not np.array_equal(
                dataset.funding_price_rate[native],
                dataset.funding_rate[native]
                * dataset.resolved_array("mark_price")[native],
            )
        ):
            raise ValueError("native attempt source or frozen snapshot differs")

    def iter_arms(self, arm_ids: tuple[str, ...]) -> Iterator[NativeArmObservation]:
        """Stream selected recipe-ordered arms; every arm starts one fresh book.

        A caller must supply an already verified pending roster for recovery.
        This iterator does not itself authenticate a stopped attempt or publish
        completion receipts. Any failure stops dispatch, never declares success.
        """
        if (
            type(arm_ids) is not tuple
            or not arm_ids
            or any(type(a) is not str or a not in _ARMS for a in arm_ids)
            or len(set(arm_ids)) != len(arm_ids)
            or tuple(a.arm_id for a in _PROTOCOL.arms if a.arm_id in arm_ids) != arm_ids
        ):
            raise ValueError(
                "native dispatch requires a nonempty ordered exact arm subset"
            )
        for arm_id in arm_ids:
            arm = _ARMS[arm_id]
            policy = _IndependentArmPolicy(self.dataset, self.frozen, arm)
            result = run_shared_portfolio_replay(
                self.dataset,
                policy,
                start_index=_PLAN.actor_start_inclusive,
                stop_index=_PLAN.terminal_stop_inclusive,
                initial_capital=_ECONOMICS.initial_capital_per_symbol_usdt,
                execution_cost=native_execution_cost(arm.scenario),
                risk=_risk(),
                settle_terminal_position=True,
                terminal_reserve_bars=_PLAN.terminal_advance_count,
                capture_ledger_evidence=True,
                insolvency_valuation="retain_debt",
                hold_unchanged_risk_target=True,
            )
            verification = verify_native_arm(
                self.dataset, arm_id, result, frozen=self.frozen
            )
            if not verification.passed:
                raise ValueError(
                    f"native arm {arm_id} failed verification: {verification.reason}"
                )
            yield NativeArmObservation(arm_id, result, verification)


def open_synthetic_native_attempt(
    dataset: MarketDataset,
    plan_path: Path,
    *,
    expected_binding: FrozenInputBinding,
    expected_pin: FrozenInputsPin,
) -> SyntheticNativeAttempt:
    """Open an exact external plan pin before running any artificial arm.

    The SYNTHETIC_ONLY scope is a caller declaration, not provenance proof.
    No production runner, market file loading or approval bypass is provided.
    """
    if (
        type(expected_binding) is not FrozenInputBinding
        or expected_binding.scope != "SYNTHETIC_ONLY"
    ):
        raise ValueError("native attempt only admits synthetic scope")
    frozen = load_frozen_inputs(
        plan_path, expected_binding=expected_binding, expected_pin=expected_pin
    )
    return SyntheticNativeAttempt(dataset, frozen, expected_pin)
