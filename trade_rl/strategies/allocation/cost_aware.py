"""Cost-aware multi-symbol allocator implementing the existing ``PortfolioPolicy``.

The policy owns no account and no P&L ledger. It reads the replay's detached
decision snapshot (current weights, mark prices, timestamp), asks a caller
supplied forecast source for per-symbol expected simple returns over the next
decision interval, estimates a covariance from past decision-time marks only, and
proposes one target-weight vector. The shared portfolio session still owns risk
constraints, fills, costs and accounting.

On every bar that is not a scheduled decision time, and whenever any input is
invalid or the solver does not converge, the policy proposes the current weights
with every symbol marked HOLD-current-quantity, so no order is generated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from trade_rl.strategies.allocation.solver import (
    AllocationSolverError,
    solve_cost_aware_allocation,
)
from trade_rl.strategies.portfolio import (
    PortfolioDecisionSnapshot,
    PortfolioTargetProposal,
)

_NS_PER_HOUR = 3_600 * 1_000_000_000
_CONSTRAINT_TOLERANCE = 1e-9


@runtime_checkable
class ExpectedReturnSource(Protocol):
    """Caller-supplied causal forecast source.

    ``expected_simple_returns`` receives the current decision snapshot and returns
    one finite expected **simple** return per ``snapshot.symbols`` entry for the
    interval until the next scheduled decision. It must use only information
    available at ``snapshot.timestamp_ns``. The allocator never sees how it was
    produced. Raising propagates; returning a wrong-length or non-finite vector
    makes the allocator fail closed (HOLD) for that decision.
    """

    def expected_simple_returns(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> tuple[float, ...]: ...


@dataclass(frozen=True, slots=True)
class AllocationDecisionRecord:
    """Audit record of one scheduled decision (not written for off-schedule bars)."""

    index: int
    timestamp_ns: int
    outcome: str  # "rebalance" | "hold_no_trade" | "fail_closed"
    reason: str | None
    iterations: int
    objective: float | None
    traded_symbols: tuple[int, ...]
    # Volatility-target diagnostics; ``None`` / ``1.0`` when no target is set.
    ex_ante_vol: float | None = None
    vol_scale: float = 1.0
    final_vol: float | None = None


def annualized_portfolio_vol(
    weights: object, covariance: object, *, periods_per_year: float
) -> float:
    """Ex-ante ``sqrt(w' Sigma w * periods_per_year)`` for per-period ``Sigma``.

    Raises ``ValueError`` for non-finite inputs or a negative quadratic form (beyond
    rounding), so callers can fail closed instead of trading on a broken estimate.
    """

    w = np.asarray(weights, dtype=np.float64)
    sigma = np.asarray(covariance, dtype=np.float64)
    if w.ndim != 1 or sigma.shape != (w.size, w.size):
        raise ValueError("weights and covariance must have matching shapes")
    if not (np.all(np.isfinite(w)) and np.all(np.isfinite(sigma))):
        raise ValueError("weights and covariance must be finite")
    if not math.isfinite(periods_per_year) or periods_per_year <= 0.0:
        raise ValueError("periods_per_year must be finite and positive")
    variance = float(w @ sigma @ w)
    if not math.isfinite(variance) or variance < -1e-15:
        raise ValueError("portfolio variance must be finite and non-negative")
    return math.sqrt(max(variance, 0.0) * periods_per_year)


def vol_target_scale(ex_ante_vol: float, target: float) -> float:
    """``min(1, target / ex_ante_vol)``: only ever scales exposure down."""

    if not math.isfinite(ex_ante_vol) or ex_ante_vol < 0.0:
        raise ValueError("ex_ante_vol must be finite and non-negative")
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError("target volatility must be finite and positive")
    if ex_ante_vol <= target:
        return 1.0
    return target / ex_ante_vol


def ewma_covariance(
    returns: object,
    *,
    halflife_periods: float,
    shrinkage: float,
    variance_floor: float,
) -> np.ndarray:
    """Exponentially weighted second-moment covariance of past simple returns.

    ``returns`` has shape ``(T, n)`` ordered oldest to newest. The newest row has
    weight 1 and a row ``k`` periods older has weight ``0.5 ** (k / halflife)``;
    weights are normalized to sum to one. Returns are not mean-centered (a
    standard short-horizon risk estimator). The result is shrunk toward its own
    diagonal, ``(1 - shrinkage) S + shrinkage diag(S)``, and ``variance_floor`` is
    added to the diagonal so the matrix is positive definite.
    """

    array = np.asarray(returns, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError("returns must be a non-empty (T, n) matrix")
    if not np.all(np.isfinite(array)):
        raise ValueError("returns must be finite")
    if not math.isfinite(halflife_periods) or halflife_periods <= 0.0:
        raise ValueError("halflife_periods must be finite and positive")
    if not math.isfinite(shrinkage) or not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must be within [0, 1]")
    if not math.isfinite(variance_floor) or variance_floor <= 0.0:
        raise ValueError("variance_floor must be finite and positive")
    ages = np.arange(array.shape[0] - 1, -1, -1, dtype=np.float64)
    weights = np.power(0.5, ages / halflife_periods)
    weights = weights / weights.sum()
    second_moment = (array * weights[:, None]).T @ array
    second_moment = 0.5 * (second_moment + second_moment.T)
    diagonal = np.diag(np.diag(second_moment))
    shrunk = (1.0 - shrinkage) * second_moment + shrinkage * diagonal
    return np.asarray(shrunk + variance_floor * np.eye(array.shape[1]))


def _real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite")
    return resolved


def _validated_symbols(symbols: object) -> tuple[str, ...]:
    if type(symbols) is not tuple or not symbols:
        raise TypeError("symbols must be a non-empty tuple")
    if any(not isinstance(value, str) or not value.strip() for value in symbols):
        raise ValueError("symbols must contain non-empty names")
    if len(set(symbols)) != len(symbols):
        raise ValueError("symbols must be unique")
    return symbols


class CostAwareAllocationPolicy:
    """Net-of-cost mean-variance target weights on a fixed decision schedule.

    Per scheduled decision it maximizes
    ``mu.w - (gamma / 2) w'Sigma w - sum_i c_i |w_i - w_cur_i|`` subject to
    ``|w_i| <= w_max`` (``0 <= w_i`` if ``long_only``) and ``sum |w| <= gross_max``,
    where ``mu`` comes from ``forecast_source`` (simple return per decision
    interval), ``Sigma`` is an EWMA covariance of per-interval simple returns
    between decision-time mark prices at or before the current time, and ``c_i`` is
    the explicit one-way cost per unit weight (see ``costs.one_way_cost_rates``).
    The L1 cost produces a no-trade region; remaining trades smaller than
    ``min_trade_fraction`` of equity are held at exact current quantity.

    Schedule: a decision occurs when the snapshot time, in hours since the Unix
    epoch, is congruent to ``decision_anchor_hour_utc`` modulo
    ``decision_interval_hours`` (00:00 UTC daily for the defaults). Decision-time marks are sampled only at those instants.
    ``warmup_timestamps_ns`` / ``warmup_mark_prices`` may seed the covariance with
    marks at earlier scheduled instants (strictly before the first snapshot).

    Optional mechanisms (both off by default, which leaves every decision unchanged):
    ``cost_penalty_multiple`` (kappa) multiplies ``c_i`` in the optimizer's L1 penalty
    only; execution costs are charged by the shared session and are not affected.
    ``vol_target_annual`` caps the ex-ante annualized volatility
    ``sqrt(w' Sigma w * periods_per_year)`` of the solved target with the policy's own
    ``Sigma`` (``periods_per_year`` defaults to ``8760 / decision_interval_hours``):
    if it exceeds the cap the whole solved target is scaled by ``cap / vol`` BEFORE
    the min-trade rule, so positions held at exact quantity by that rule can leave
    the final book slightly above the cap (``final_vol`` in the decision record).
    A non-finite volatility fails closed (HOLD).

    Instances carry replay state and must not be reused across replays.
    """

    __slots__ = (
        "_symbols",
        "_source",
        "_gamma",
        "_halflife_periods",
        "_shrinkage",
        "_w_max",
        "_gross_max",
        "_long_only",
        "_min_trade_fraction",
        "_costs",
        "_cost_penalty_multiple",
        "_vol_target",
        "_periods_per_year",
        "_interval_ns",
        "_anchor_ns",
        "_min_history_periods",
        "_variance_floor",
        "_max_iter",
        "_tol",
        "_sample_timestamps",
        "_sample_prices",
        "_last_timestamp_ns",
        "_records",
    )

    def __init__(
        self,
        *,
        symbols: tuple[str, ...],
        forecast_source: ExpectedReturnSource,
        cost_rates: tuple[float, ...],
        gamma: float = 5.0,
        ewma_halflife_days: float = 60.0,
        shrinkage: float = 0.2,
        w_max: float = 0.5,
        gross_max: float = 1.0,
        long_only: bool = False,
        min_trade_fraction: float = 0.005,
        decision_interval_hours: int = 24,
        decision_anchor_hour_utc: int = 0,
        min_history_periods: int = 30,
        variance_floor: float = 1e-8,
        max_iter: int = 2_000,
        tol: float = 1e-10,
        warmup_timestamps_ns: tuple[int, ...] = (),
        warmup_mark_prices: tuple[tuple[float, ...], ...] = (),
        cost_penalty_multiple: float = 1.0,
        vol_target_annual: float | None = None,
        periods_per_year: float | None = None,
    ) -> None:
        self._symbols = _validated_symbols(symbols)
        n_symbols = len(self._symbols)
        if not callable(getattr(forecast_source, "expected_simple_returns", None)):
            raise TypeError("forecast_source must define expected_simple_returns")
        self._source = forecast_source
        if type(cost_rates) is not tuple or len(cost_rates) != n_symbols:
            raise ValueError("cost_rates must be a tuple matching the symbol roster")
        costs = [_real(value, name="cost_rates") for value in cost_rates]
        if any(value < 0.0 for value in costs):
            raise ValueError("cost_rates must be non-negative")
        self._costs = np.asarray(costs, dtype=np.float64)
        self._cost_penalty_multiple = _real(
            cost_penalty_multiple, name="cost_penalty_multiple"
        )
        if self._cost_penalty_multiple <= 0.0:
            raise ValueError("cost_penalty_multiple must be positive")
        self._vol_target: float | None = None
        if vol_target_annual is not None:
            self._vol_target = _real(vol_target_annual, name="vol_target_annual")
            if self._vol_target <= 0.0:
                raise ValueError("vol_target_annual must be positive")
        self._gamma = _real(gamma, name="gamma")
        if self._gamma <= 0.0:
            raise ValueError("gamma must be positive")
        halflife_days = _real(ewma_halflife_days, name="ewma_halflife_days")
        if halflife_days <= 0.0:
            raise ValueError("ewma_halflife_days must be positive")
        if type(decision_interval_hours) is not int or decision_interval_hours < 1:
            raise ValueError("decision_interval_hours must be a positive integer")
        if (
            type(decision_anchor_hour_utc) is not int
            or not 0 <= decision_anchor_hour_utc < decision_interval_hours
        ):
            raise ValueError(
                "decision_anchor_hour_utc must be within [0, decision_interval_hours)"
            )
        self._interval_ns = decision_interval_hours * _NS_PER_HOUR
        self._anchor_ns = decision_anchor_hour_utc * _NS_PER_HOUR
        resolved_periods = (
            8_760.0 / decision_interval_hours
            if periods_per_year is None
            else _real(periods_per_year, name="periods_per_year")
        )
        if resolved_periods <= 0.0:
            raise ValueError("periods_per_year must be positive")
        self._periods_per_year = resolved_periods
        self._halflife_periods = halflife_days * 24.0 / decision_interval_hours
        self._shrinkage = _real(shrinkage, name="shrinkage")
        if not 0.0 <= self._shrinkage <= 1.0:
            raise ValueError("shrinkage must be within [0, 1]")
        self._w_max = _real(w_max, name="w_max")
        self._gross_max = _real(gross_max, name="gross_max")
        if self._w_max <= 0.0 or self._gross_max <= 0.0:
            raise ValueError("w_max and gross_max must be positive")
        if type(long_only) is not bool:
            raise TypeError("long_only must be a boolean")
        self._long_only = long_only
        self._min_trade_fraction = _real(min_trade_fraction, name="min_trade_fraction")
        if not 0.0 <= self._min_trade_fraction < 1.0:
            raise ValueError("min_trade_fraction must be within [0, 1)")
        if type(min_history_periods) is not int or min_history_periods < 2:
            raise ValueError("min_history_periods must be an integer of at least two")
        self._min_history_periods = min_history_periods
        self._variance_floor = _real(variance_floor, name="variance_floor")
        if self._variance_floor <= 0.0:
            raise ValueError("variance_floor must be positive")
        if type(max_iter) is not int or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        self._max_iter = max_iter
        self._tol = _real(tol, name="tol")
        if self._tol <= 0.0:
            raise ValueError("tol must be positive")
        self._sample_timestamps: list[int] = []
        self._sample_prices: list[np.ndarray] = []
        self._load_warmup(warmup_timestamps_ns, warmup_mark_prices)
        self._last_timestamp_ns: int | None = None
        self._records: list[AllocationDecisionRecord] = []

    def _load_warmup(
        self,
        timestamps: tuple[int, ...],
        prices: tuple[tuple[float, ...], ...],
    ) -> None:
        if type(timestamps) is not tuple or type(prices) is not tuple:
            raise TypeError("warmup timestamps and prices must be tuples")
        if len(timestamps) != len(prices):
            raise ValueError("warmup timestamps and prices must have equal length")
        previous: int | None = None
        for timestamp, row in zip(timestamps, prices, strict=True):
            if type(timestamp) is not int:
                raise TypeError("warmup timestamps must be native integers")
            if timestamp % self._interval_ns != self._anchor_ns:
                raise ValueError("warmup timestamps must lie on the decision schedule")
            if previous is not None and timestamp <= previous:
                raise ValueError("warmup timestamps must strictly increase")
            vector = np.asarray(row, dtype=np.float64)
            if vector.shape != (len(self._symbols),) or not self._valid_prices(vector):
                raise ValueError("warmup marks must be positive finite per symbol")
            self._sample_timestamps.append(timestamp)
            self._sample_prices.append(vector)
            previous = timestamp

    @staticmethod
    def _valid_prices(vector: np.ndarray) -> bool:
        return bool(np.all(np.isfinite(vector)) and np.all(vector > 0.0))

    @property
    def symbols(self) -> tuple[str, ...]:
        return self._symbols

    @property
    def records(self) -> tuple[AllocationDecisionRecord, ...]:
        return tuple(self._records)

    def decide_portfolio(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> PortfolioTargetProposal:
        if type(snapshot) is not PortfolioDecisionSnapshot:
            raise TypeError("snapshot must be a PortfolioDecisionSnapshot")
        if snapshot.symbols != self._symbols:
            raise ValueError("snapshot symbols do not match the allocator roster")
        if (
            self._last_timestamp_ns is not None
            and snapshot.timestamp_ns <= self._last_timestamp_ns
        ):
            raise ValueError("snapshot timestamps must increase between decisions")
        if (
            self._last_timestamp_ns is None
            and self._sample_timestamps
            and snapshot.timestamp_ns <= self._sample_timestamps[-1]
        ):
            raise ValueError("warmup marks must precede the first snapshot")
        self._last_timestamp_ns = snapshot.timestamp_ns

        current = snapshot.account.weights
        hold_all = PortfolioTargetProposal(
            target_weights=current,
            hold_current_quantity_symbols=tuple(range(len(self._symbols))),
        )
        if snapshot.timestamp_ns % self._interval_ns != self._anchor_ns:
            return hold_all

        def fail(reason: str) -> PortfolioTargetProposal:
            self._record(snapshot, "fail_closed", reason, 0, None, ())
            return hold_all

        marks = np.asarray(snapshot.account.mark_prices, dtype=np.float64)
        if not self._valid_prices(marks):
            return fail("invalid_marks")
        self._sample_timestamps.append(snapshot.timestamp_ns)
        self._sample_prices.append(marks)

        account = snapshot.account
        if account.insolvent or account.termination_reason is not None:
            return fail("account_terminated")
        if len(self._sample_prices) - 1 < self._min_history_periods:
            return fail("insufficient_history")

        prices = np.vstack(self._sample_prices)
        covariance = ewma_covariance(
            prices[1:] / prices[:-1] - 1.0,
            halflife_periods=self._halflife_periods,
            shrinkage=self._shrinkage,
            variance_floor=self._variance_floor,
        )

        raw = self._source.expected_simple_returns(snapshot)
        try:
            mu = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError):
            return fail("invalid_forecast")
        if (
            mu.shape != (len(self._symbols),)
            or not np.all(np.isfinite(mu))
            or np.any(mu <= -1.0)
        ):
            return fail("invalid_forecast")

        w_current = np.asarray(current, dtype=np.float64)
        try:
            solution = solve_cost_aware_allocation(
                expected_returns=mu,
                covariance=covariance,
                current_weights=w_current,
                costs=self._costs * self._cost_penalty_multiple,
                gamma=self._gamma,
                w_max=self._w_max,
                gross_max=self._gross_max,
                long_only=self._long_only,
                max_iter=self._max_iter,
                tol=self._tol,
            )
        except AllocationSolverError as exc:
            return fail(f"solver_{exc.reason}")

        target = solution.weights.copy()
        ex_ante_vol: float | None = None
        vol_scale = 1.0
        if self._vol_target is not None:
            try:
                ex_ante_vol = annualized_portfolio_vol(
                    target, covariance, periods_per_year=self._periods_per_year
                )
                vol_scale = vol_target_scale(ex_ante_vol, self._vol_target)
            except ValueError:
                return fail("invalid_volatility")
            target = target * vol_scale
        held: list[int] = []
        for i in range(len(self._symbols)):
            if abs(target[i] - w_current[i]) <= self._min_trade_fraction:
                trial = target.copy()
                trial[i] = w_current[i]
                if self._feasible(trial):
                    target = trial
                    held.append(i)
        if not self._feasible(target):
            return fail("constraint_violation")
        final_vol: float | None = None
        if self._vol_target is not None:
            try:
                final_vol = annualized_portfolio_vol(
                    target, covariance, periods_per_year=self._periods_per_year
                )
            except ValueError:
                return fail("invalid_volatility")
        traded = tuple(i for i in range(len(self._symbols)) if i not in held)
        if not traded:
            self._record(
                snapshot,
                "hold_no_trade",
                None,
                solution.iterations,
                solution.objective,
                (),
                ex_ante_vol=ex_ante_vol,
                vol_scale=vol_scale,
                final_vol=final_vol,
            )
            return hold_all
        weights = tuple(
            current[i] if i in held else float(target[i])
            for i in range(len(self._symbols))
        )
        self._record(
            snapshot,
            "rebalance",
            None,
            solution.iterations,
            solution.objective,
            traded,
            ex_ante_vol=ex_ante_vol,
            vol_scale=vol_scale,
            final_vol=final_vol,
        )
        return PortfolioTargetProposal(
            target_weights=weights,
            hold_current_quantity_symbols=tuple(held),
        )

    def _feasible(self, weights: np.ndarray) -> bool:
        floor = 0.0 if self._long_only else -self._w_max
        return bool(
            np.all(weights >= floor - _CONSTRAINT_TOLERANCE)
            and np.all(weights <= self._w_max + _CONSTRAINT_TOLERANCE)
            and float(np.abs(weights).sum()) <= self._gross_max + _CONSTRAINT_TOLERANCE
        )

    def _record(
        self,
        snapshot: PortfolioDecisionSnapshot,
        outcome: str,
        reason: str | None,
        iterations: int,
        objective: float | None,
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
                iterations=iterations,
                objective=objective,
                traded_symbols=traded,
                ex_ante_vol=ex_ante_vol,
                vol_scale=vol_scale,
                final_vol=final_vol,
            )
        )


__all__ = [
    "AllocationDecisionRecord",
    "CostAwareAllocationPolicy",
    "ExpectedReturnSource",
    "annualized_portfolio_vol",
    "ewma_covariance",
    "vol_target_scale",
]
