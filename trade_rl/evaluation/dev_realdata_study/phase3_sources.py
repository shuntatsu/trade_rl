"""Phase 3 sources, the vol-scaled passive policy and the arm builder (Issue #810).

Everything here is causal in the same sense as ``trial_sources``: signals read the
daily 00:00 UTC marks (and the vintages fitted before each date) at or before the
decision time. The frozen definitions are in
:mod:`~trade_rl.evaluation.dev_realdata_study.phase3_registry`.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.dev_realdata_study import runner
from trade_rl.evaluation.dev_realdata_study.phase3_registry import (
    KIND_ALLOCATOR,
    KIND_PASSIVE_VT,
    P3_TRIALS,
    SOURCE_T2,
    SOURCE_T4,
    SOURCE_T4_WEEKLY,
    SOURCE_T6,
    SOURCE_ZBLEND_T2_T4,
    Phase3Spec,
)
from trade_rl.evaluation.dev_realdata_study.protocol import (
    ARM_BUY_AND_HOLD,
    ARM_CASH,
    DevStudyProtocol,
)
from trade_rl.evaluation.dev_realdata_study.trial_sources import (
    BLEND_MIN_POOLED_OBSERVATIONS,
    DailyPanel,
    RidgeDailyComponent,
    TrialContext,
    _SeriesSource,
    build_trial_source,
    trial_spec,
    xsmom_row,
)
from trade_rl.evaluation.dev_realdata_study.trials import CONSTANTS, Phase2Constants
from trade_rl.strategies.allocation.cost_aware import (
    AllocationDecisionRecord,
    annualized_portfolio_vol,
    ewma_covariance,
    vol_target_scale,
)
from trade_rl.strategies.portfolio import (
    PortfolioDecisionSnapshot,
    PortfolioTargetProposal,
)

_NS_PER_HOUR = 3_600 * 1_000_000_000

_T2_ARM = "t2_xsmom"
_T4_ARM = "t4_ridge24_all_features"
_T6_ARM = "t6_blend"


# --- scaled source (P3-5) ---------------------------------------------------


class ScaledSource:
    """Multiply another source's expected returns by a constant (T4 weekly: x7).

    ``calls`` / ``null_packets`` are read from the inner source; ``history`` holds
    the scaled vectors that the allocator actually received.
    """

    def __init__(self, inner: Any, factor: float) -> None:
        if not math.isfinite(factor) or factor <= 0.0:
            raise ValueError("factor must be finite and positive")
        self._inner = inner
        self._factor = factor
        self.history: list[tuple[float, ...]] = []

    @property
    def calls(self) -> int:
        return int(self._inner.calls)

    @property
    def null_packets(self) -> int:
        return int(self._inner.null_packets)

    def expected_simple_returns(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> tuple[float, ...]:
        raw = self._inner.expected_simple_returns(snapshot)
        scaled = tuple(float(value) * self._factor for value in raw)
        self.history.append(scaled)
        return scaled


# --- z-blend of T2 and T4 (P3-6, P3-8) ----------------------------------------


def xsmom_series(
    marks: np.ndarray, *, one_way_cost: float, constants: Phase2Constants
) -> np.ndarray:
    """T2 expected returns for every day (row ``d`` uses marks up to day ``d``)."""

    out = np.full(marks.shape, np.nan, dtype=np.float64)
    for day in range(marks.shape[0]):
        out[day] = xsmom_row(marks, day, one_way_cost=one_way_cost, constants=constants)
    return out


class PairBlendSource(_SeriesSource):
    """z-blend of T2 and T4: the T6 construction with two components.

    The Ridge component (T4, per 24h) and the T2 series are each divided by their
    causal trailing population std pooled over symbols and dates ``(t-364d .. t)``;
    the mean of the two is multiplied by the trailing std of the Ridge component and
    by ``interval_scale``. Until ``blend_min_dates`` Ridge dates exist, or if either
    std is unavailable, it serves NaN (zero mu).
    """

    def __init__(
        self,
        panel: DailyPanel,
        ridge: RidgeDailyComponent,
        xsmom: np.ndarray,
        *,
        constants: Phase2Constants,
        interval_scale: float = 1.0,
    ) -> None:
        super().__init__(panel)
        if xsmom.shape != panel.marks.shape:
            raise ValueError("xsmom series must match the daily panel")
        self._ridge = ridge
        self._xsmom = xsmom
        self._constants = constants
        self._interval_scale = interval_scale

    @staticmethod
    def _pooled_std(values: np.ndarray, *, minimum: int) -> float | None:
        finite = values[np.isfinite(values)]
        if finite.size < minimum:
            return None
        std = float(np.std(finite))
        return std if math.isfinite(std) and std > 0.0 else None

    def values(self, day: int) -> np.ndarray:
        constants = self._constants
        n_symbols = len(self._panel.symbols)
        nothing = np.full(n_symbols, np.nan)
        low = max(0, day - (constants.blend_std_window_days - 1))
        rows = [self._ridge.value_at(q) for q in range(low, day + 1)]
        ridge_rows = [row for row in rows if row is not None]
        if len(ridge_rows) < constants.blend_min_dates:
            return nothing
        ridge_sigma = self._pooled_std(
            np.vstack(ridge_rows), minimum=constants.blend_min_dates
        )
        xsmom_sigma = self._pooled_std(
            self._xsmom[low : day + 1], minimum=BLEND_MIN_POOLED_OBSERVATIONS
        )
        if ridge_sigma is None or xsmom_sigma is None:
            return nothing
        total = np.zeros(n_symbols)
        for component, sigma in (
            (self._ridge.value_at(day), ridge_sigma),
            (self._xsmom[day], xsmom_sigma),
        ):
            if component is None:
                continue
            normalised = component / sigma
            total = total + np.where(np.isfinite(normalised), normalised, 0.0)
        return np.asarray(total / 2.0 * ridge_sigma * self._interval_scale)


# --- vol-scaled passive comparator (P3-7) -----------------------------------


class VolTargetedBuyAndHoldPolicy:
    """Long-only equal-weight book whose gross is scaled to a volatility cap.

    At every scheduled decision the relative composition is the account's current
    (drifted) weights, or equal weights while the book is empty. The composition at
    gross ``gross_budget`` has ex-ante volatility ``sigma_p``
    (``sqrt(w' Sigma w * periods_per_year)``, ``Sigma`` the EWMA/shrunk covariance of
    decision-time marks as in the cost-aware allocator); the target is that composition
    scaled by ``min(1, vol_target / sigma_p)``. A symbol whose target is within
    ``min_trade_fraction`` of its current weight is held at exact quantity. There is no
    calendar rebalance back to equal weights. Any invalid input, a short position or an
    account termination fails closed (HOLD every quantity). The policy owns no
    account; records mimic :class:`AllocationDecisionRecord`.
    Instances carry replay state and must not be reused across replays.
    """

    def __init__(
        self,
        *,
        symbols: tuple[str, ...],
        gross_budget: float,
        vol_target_annual: float,
        decision_interval_hours: int = 24,
        decision_anchor_hour_utc: int = 0,
        ewma_halflife_days: float = 60.0,
        shrinkage: float = 0.2,
        min_trade_fraction: float = 0.005,
        min_history_periods: int = 30,
        variance_floor: float = 1e-8,
        warmup_timestamps_ns: tuple[int, ...] = (),
        warmup_mark_prices: tuple[tuple[float, ...], ...] = (),
    ) -> None:
        if (
            type(symbols) is not tuple
            or not symbols
            or len(set(symbols)) != len(symbols)
        ):
            raise ValueError("symbols must be a non-empty tuple of unique names")
        if not 0.0 < gross_budget <= 1.0:
            raise ValueError("gross_budget must be within (0, 1]")
        if vol_target_annual <= 0.0 or not math.isfinite(vol_target_annual):
            raise ValueError("vol_target_annual must be finite and positive")
        if type(decision_interval_hours) is not int or decision_interval_hours < 1:
            raise ValueError("decision_interval_hours must be a positive integer")
        if not 0 <= decision_anchor_hour_utc < decision_interval_hours:
            raise ValueError("decision_anchor_hour_utc outside the interval")
        if len(warmup_timestamps_ns) != len(warmup_mark_prices):
            raise ValueError("warmup timestamps and prices must have equal length")
        self._symbols = symbols
        self._gross_budget = float(gross_budget)
        self._vol_target = float(vol_target_annual)
        self._interval_ns = decision_interval_hours * _NS_PER_HOUR
        self._anchor_ns = decision_anchor_hour_utc * _NS_PER_HOUR
        self._periods_per_year = 8_760.0 / decision_interval_hours
        self._halflife_periods = ewma_halflife_days * 24.0 / decision_interval_hours
        self._shrinkage = shrinkage
        self._min_trade_fraction = min_trade_fraction
        self._min_history_periods = min_history_periods
        self._variance_floor = variance_floor
        self._timestamps: list[int] = list(warmup_timestamps_ns)
        self._prices: list[np.ndarray] = [
            np.asarray(row, dtype=np.float64) for row in warmup_mark_prices
        ]
        for stamp in self._timestamps:
            if stamp % self._interval_ns != self._anchor_ns:
                raise ValueError("warmup timestamps must lie on the decision schedule")
        self._last_timestamp_ns: int | None = None
        self._records: list[AllocationDecisionRecord] = []

    @property
    def symbols(self) -> tuple[str, ...]:
        return self._symbols

    @property
    def records(self) -> tuple[AllocationDecisionRecord, ...]:
        return tuple(self._records)

    def _record(
        self,
        snapshot: PortfolioDecisionSnapshot,
        outcome: str,
        reason: str | None,
        traded: tuple[int, ...],
        *,
        ex_ante_vol: float | None = None,
        vol_scale: float = 1.0,
        final_vol: float | None = None,
    ) -> None:
        self._records.append(
            AllocationDecisionRecord(
                index=snapshot.index,
                timestamp_ns=snapshot.timestamp_ns,
                outcome=outcome,
                reason=reason,
                iterations=0,
                objective=None,
                traded_symbols=traded,
                ex_ante_vol=ex_ante_vol,
                vol_scale=vol_scale,
                final_vol=final_vol,
            )
        )

    def decide_portfolio(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> PortfolioTargetProposal:
        if type(snapshot) is not PortfolioDecisionSnapshot:
            raise TypeError("snapshot must be a PortfolioDecisionSnapshot")
        if snapshot.symbols != self._symbols:
            raise ValueError("snapshot symbols do not match the policy roster")
        if (
            self._last_timestamp_ns is not None
            and snapshot.timestamp_ns <= self._last_timestamp_ns
        ):
            raise ValueError("snapshot timestamps must increase between decisions")
        self._last_timestamp_ns = snapshot.timestamp_ns
        n_symbols = len(self._symbols)
        current = snapshot.account.weights
        hold_all = PortfolioTargetProposal(
            target_weights=current,
            hold_current_quantity_symbols=tuple(range(n_symbols)),
        )
        if snapshot.timestamp_ns % self._interval_ns != self._anchor_ns:
            return hold_all

        def fail(reason: str) -> PortfolioTargetProposal:
            self._record(snapshot, "fail_closed", reason, ())
            return hold_all

        marks = np.asarray(snapshot.account.mark_prices, dtype=np.float64)
        if not (np.all(np.isfinite(marks)) and np.all(marks > 0.0)):
            return fail("invalid_marks")
        self._timestamps.append(snapshot.timestamp_ns)
        self._prices.append(marks)
        account = snapshot.account
        if account.insolvent or account.termination_reason is not None:
            return fail("account_terminated")
        if len(self._prices) - 1 < self._min_history_periods:
            return fail("insufficient_history")
        prices = np.vstack(self._prices)
        covariance = ewma_covariance(
            prices[1:] / prices[:-1] - 1.0,
            halflife_periods=self._halflife_periods,
            shrinkage=self._shrinkage,
            variance_floor=self._variance_floor,
        )
        w_current = np.asarray(current, dtype=np.float64)
        if not np.all(np.isfinite(w_current)) or np.any(w_current < -1e-12):
            return fail("unexpected_short_or_invalid_weights")
        gross_current = float(np.abs(w_current).sum())
        if gross_current <= 1e-12:
            composition = np.full(n_symbols, 1.0 / n_symbols)
        else:
            composition = w_current / gross_current
        unit = composition * self._gross_budget
        try:
            ex_ante_vol = annualized_portfolio_vol(
                unit, covariance, periods_per_year=self._periods_per_year
            )
            scale = vol_target_scale(ex_ante_vol, self._vol_target)
        except ValueError:
            return fail("invalid_volatility")
        target = unit * scale
        held = [
            i
            for i in range(n_symbols)
            if abs(target[i] - w_current[i]) <= self._min_trade_fraction
        ]
        for i in held:
            target[i] = w_current[i]
        try:
            final_vol = annualized_portfolio_vol(
                target, covariance, periods_per_year=self._periods_per_year
            )
        except ValueError:
            return fail("invalid_volatility")
        traded = tuple(i for i in range(n_symbols) if i not in held)
        if not traded:
            self._record(
                snapshot,
                "hold_no_trade",
                None,
                (),
                ex_ante_vol=ex_ante_vol,
                vol_scale=scale,
                final_vol=final_vol,
            )
            return hold_all
        self._record(
            snapshot,
            "rebalance",
            None,
            traded,
            ex_ante_vol=ex_ante_vol,
            vol_scale=scale,
            final_vol=final_vol,
        )
        return PortfolioTargetProposal(
            target_weights=tuple(
                current[i] if i in held else float(target[i]) for i in range(n_symbols)
            ),
            hold_current_quantity_symbols=tuple(held),
        )


# --- context and builder --------------------------------------------------------


class Phase3Context(TrialContext):
    """:class:`TrialContext` plus the cached T2 series for the z-blend."""

    def __init__(self, dataset: MarketDataset, protocol: DevStudyProtocol) -> None:
        super().__init__(dataset, protocol)
        self._xsmom: np.ndarray | None = None

    def xsmom_series(self) -> np.ndarray:
        if self._xsmom is None:
            self._xsmom = xsmom_series(
                self.panel.marks,
                one_way_cost=self.one_way_cost,
                constants=self.constants,
            )
        return self._xsmom


def make_phase3_context(
    dataset: MarketDataset, protocol: DevStudyProtocol
) -> Phase3Context:
    """Module-level so the causality probe can build sources for a perturbed copy."""

    return Phase3Context(dataset, protocol)


def phase3_spec(arm: str) -> Phase3Spec:
    for spec in P3_TRIALS:
        if spec.arm == arm:
            return spec
    raise ValueError(f"unknown Phase 3 arm: {arm}")


def build_phase3_source(context: Phase3Context, spec: Phase3Spec) -> Any:
    """The expected-return source of an allocator arm."""

    if spec.source == SOURCE_T4:
        return build_trial_source(context, trial_spec(_T4_ARM))
    if spec.source == SOURCE_T4_WEEKLY:
        # T4's Ridge source stays on the 24h decision interval (horizon == interval);
        # the weekly arm multiplies its per-24h expected return by 7.
        return ScaledSource(
            build_trial_source(context, trial_spec(_T4_ARM)),
            spec.decision_interval_hours / 24.0,
        )
    if spec.source == SOURCE_T2:
        return build_trial_source(context, trial_spec(_T2_ARM))
    if spec.source == SOURCE_T6:
        return build_trial_source(context, trial_spec(_T6_ARM))
    if spec.source == SOURCE_ZBLEND_T2_T4:
        ridge = RidgeDailyComponent(
            context.forecast_source("ridge24_all"), context.panel, divisor=1.0
        )
        return PairBlendSource(
            context.panel,
            ridge,
            context.xsmom_series(),
            constants=CONSTANTS,
            interval_scale=spec.decision_interval_hours / 24.0,
        )
    raise ValueError(f"unregistered Phase 3 source: {spec.source}")


def build_phase3_arm(
    arm: str,
    dataset: MarketDataset,
    protocol: DevStudyProtocol,
    context: Phase3Context,
) -> runner.BuiltArm:
    """Fresh policy for one replay of a baseline or a Phase 3 arm."""

    if arm in (ARM_CASH, ARM_BUY_AND_HOLD):
        return runner.build_arm(arm, dataset, protocol, context.pr_c_sources)
    spec = phase3_spec(arm)
    if spec.kind == KIND_PASSIVE_VT:
        if spec.vol_target is None:
            raise ValueError("the passive comparator needs a volatility target")
        stamps, marks = runner.warmup_marks(
            dataset,
            protocol,
            decision_interval_hours=spec.decision_interval_hours,
            decision_anchor_hour_utc=spec.decision_anchor_hour_utc,
        )
        policy = VolTargetedBuyAndHoldPolicy(
            symbols=dataset.symbols,
            gross_budget=protocol.buy_and_hold_gross_budget,
            vol_target_annual=spec.vol_target,
            decision_interval_hours=spec.decision_interval_hours,
            decision_anchor_hour_utc=spec.decision_anchor_hour_utc,
            ewma_halflife_days=protocol.ewma_halflife_days,
            shrinkage=protocol.shrinkage,
            min_trade_fraction=protocol.min_trade_fraction,
            min_history_periods=protocol.min_history_periods,
            warmup_timestamps_ns=stamps,
            warmup_mark_prices=marks,
        )
        return runner.BuiltArm(policy, None)
    if spec.kind != KIND_ALLOCATOR:
        raise ValueError(f"unknown Phase 3 arm kind: {spec.kind}")
    source = build_phase3_source(context, spec)
    allocator = runner.build_allocator(
        dataset,
        protocol,
        source,
        long_only=spec.long_only,
        decision_interval_hours=spec.decision_interval_hours,
        decision_anchor_hour_utc=spec.decision_anchor_hour_utc,
        cost_penalty_multiple=spec.kappa,
        vol_target_annual=spec.vol_target,
    )
    return runner.BuiltArm(allocator, source)


__all__ = [
    "Phase3Context",
    "PairBlendSource",
    "ScaledSource",
    "VolTargetedBuyAndHoldPolicy",
    "build_phase3_arm",
    "build_phase3_source",
    "make_phase3_context",
    "phase3_spec",
    "xsmom_series",
]
