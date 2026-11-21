"""Weekly cross-sectional book of ``xs_lowvol_funding_v1`` as a ``PortfolioPolicy``.

Decisions are made at the 4h bar closing Monday 00:00 UTC (every 42 rows from the first
decision row). For each member of the month's universe the policy reads, from the row
``decision row - signal lag`` only,

* ``sigma``: the sample standard deviation of the last 180 4h log returns, and
* ``f``: the mean of the last 9 settled funding rates (an event's rate is its bar's funding
  sum over the bar's event count, exact whenever a bar holds at most one event),

ranks them, goes long the ``k = floor(n / 3)`` best and short the ``k`` worst names with
inverse-volatility weights (``sum |w| * sigma = 1`` per leg), scales the book to a 20%
ex-ante volatility with the 50/50 diagonal-shrunk covariance of the last 360 log returns,
and then applies, in this order, the per-name cap, the gross cap and the cash account. A
held name is closed by a forced-close rule from 42 bars before its last available bar (a
modelled one-week delisting announcement: it uses knowledge of the archive end).

The pure helpers (``zr``, ``select_legs``, ``leg_weights``, ``vol_target_scale``,
``cap_weights``, ``cash_factors``) are public so the oracles can test them against
hand-computed numbers. Nothing here reads a network, a store or a result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.trend_oos.data import StageDataError
from trade_rl.evaluation.xs_lowvol_funding import protocol
from trade_rl.strategies.agy_anchor.policy import risk_scale_of
from trade_rl.strategies.portfolio import (
    PortfolioDecisionSnapshot,
    PortfolioPolicy,
    PortfolioTargetProposal,
)

_SCALE_EPS = 1e-12
_NO_TRADE_ATOL = 1e-12
_EPS = 1e-9
_TIE_ATOL = 1e-12


# ----------------------------------------------------------------------- pure helpers


def average_ranks(values: np.ndarray) -> np.ndarray:
    """1-based ascending ranks; tied values share the mean of their ranks."""
    x = np.asarray(values, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ordered = x[order]
    ranks = np.empty(x.size, dtype=np.float64)
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and ordered[j + 1] == ordered[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def zr(values: np.ndarray) -> np.ndarray:
    """Cross-sectional rank score: (rank - mean rank) / population std of the ranks."""
    ranks = average_ranks(values)
    spread = float(np.std(ranks))
    if spread <= 0.0:
        return np.zeros_like(ranks)
    return (ranks - float(np.mean(ranks))) / spread


def score_of(arm: str, sigma: np.ndarray, funding: np.ndarray) -> np.ndarray:
    """The arm's score: both factors for the candidate, one factor for an ablation."""
    if arm == protocol.CANDIDATE:
        return protocol.SCORE_WEIGHT_VOL * zr(
            -sigma
        ) + protocol.SCORE_WEIGHT_FUNDING * zr(-funding)
    if arm == protocol.ARM_LOWVOL:
        return zr(-sigma)
    if arm == protocol.ARM_FUNDING:
        return zr(-funding)
    raise ValueError(f"arm {arm!r} has no score")


def select_legs(score: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Positions of the ``k`` highest (long) and ``k`` lowest (short) scores.

    Ties are broken by position (the caller passes names in symbol order).
    """
    s = np.asarray(score, dtype=np.float64)
    if k < 1 or 2 * k > s.size:
        raise ValueError("k must satisfy 1 <= k and 2k <= n")
    position = np.arange(s.size)
    long_order = np.lexsort((position, -s))
    short_order = np.lexsort((position, s))
    return long_order[:k], short_order[:k]


def leg_weights(
    sigma: np.ndarray, long_index: np.ndarray, short_index: np.ndarray
) -> np.ndarray:
    """Raw signed weights ``+-1 / (k sigma_i)``: each leg has ``sum |w| * sigma = 1``."""
    sig = np.asarray(sigma, dtype=np.float64)
    k = int(long_index.size)
    weights = np.zeros(sig.size, dtype=np.float64)
    weights[long_index] = 1.0 / (k * sig[long_index])
    weights[short_index] = -1.0 / (k * sig[short_index])
    return weights


def shrunk_annual_covariance(returns: np.ndarray) -> np.ndarray:
    """Annualised covariance of 4h log returns, shrunk 50/50 toward its own diagonal."""
    r = np.asarray(returns, dtype=np.float64)
    cov = np.atleast_2d(np.cov(r, rowvar=False, ddof=1))
    shrunk = (1.0 - protocol.COV_SHRINKAGE) * cov + protocol.COV_SHRINKAGE * np.diag(
        np.diag(cov)
    )
    return np.asarray(shrunk * protocol.PERIODS_PER_YEAR, dtype=np.float64)


def vol_target_scale(weights: np.ndarray, returns: np.ndarray) -> float:
    """Factor that brings ``weights`` to the 20% ex-ante annual volatility; 0 if undefined."""
    w = np.asarray(weights, dtype=np.float64)
    variance = float(w @ shrunk_annual_covariance(returns) @ w)
    if not math.isfinite(variance) or variance <= 0.0:
        return 0.0
    return protocol.VOL_TARGET / math.sqrt(variance)


def cap_weights(weights: np.ndarray) -> np.ndarray:
    """Per-name cap, then the gross cap scaled proportionally (the cash account follows)."""
    w = np.clip(
        np.asarray(weights, dtype=np.float64),
        -protocol.MAX_ABS_WEIGHT,
        protocol.MAX_ABS_WEIGHT,
    )
    gross = float(np.abs(w).sum())
    if gross > protocol.GROSS_CAP:
        w = w * (protocol.GROSS_CAP / gross)
    return w


def cash_measure(weights: np.ndarray) -> float:
    """Margin use of an all-perp book: ``0.4 * gross``."""
    return protocol.INITIAL_MARGIN_RATE * float(
        np.abs(np.asarray(weights, dtype=np.float64)).sum()
    )


def cash_factors(effective: np.ndarray) -> tuple[float, float, float]:
    """``(factor, margin factor, net factor)`` that bring ``effective`` onto the account.

    The margin bound is ``0.4 * gross <= 1.0``, the net bound ``sum of signed weights <=
    0.97``; the whole target is scaled by the tighter factor (1.0 when neither binds).
    """
    w = np.asarray(effective, dtype=np.float64)
    measure = cash_measure(w)
    net = float(w.sum())
    margin_factor = (
        protocol.CASH_LIMIT / measure if measure > protocol.CASH_LIMIT + _EPS else 1.0
    )
    net_factor = (
        protocol.NET_EXPOSURE_LIMIT / net
        if net > protocol.NET_EXPOSURE_LIMIT + _EPS
        else 1.0
    )
    return min(margin_factor, net_factor), margin_factor, net_factor


def funding_mean(
    rate: np.ndarray,
    count: np.ndarray,
    *,
    upto: int,
    events: int = protocol.FUNDING_EVENTS,
) -> float:
    """Mean of the last ``events`` settled funding rates stamped at rows ``<= upto``.

    A bar that holds ``c`` events carries their sum; each of its events counts as the
    bar's mean ``sum / c``. NaN when fewer than ``events`` events exist.
    """
    counts = np.asarray(count)[: upto + 1]
    rows = np.flatnonzero(counts)
    taken = 0
    total = 0.0
    for row in rows[::-1]:
        bar_events = int(counts[row])
        use = min(bar_events, events - taken)
        total += use * float(rate[row]) / bar_events
        taken += use
        if taken >= events:
            return total / events
    return math.nan


# -------------------------------------------------------------------------- inputs


class UnsupportedWriteOffError(StageDataError):
    """A position of a delisted name is still held at its last available bar.

    The spec has no write-off branch: the shared replay cannot continue once the engine
    settles a position on an asset that has just become unavailable, so the run refuses.
    This is a disclosed limitation that would need an amendment.
    """


@dataclass(slots=True)
class XsInputs:
    """The arrays the policy reads (mutable so the causality probe can restore rows)."""

    legs: tuple[str, ...]
    close: np.ndarray
    active: np.ndarray
    tradable: np.ndarray
    funding_rate: np.ndarray
    funding_count: np.ndarray
    member: np.ndarray
    forced: np.ndarray
    last_bar: np.ndarray

    @staticmethod
    def last_bar_rows(active: np.ndarray) -> np.ndarray:
        """Rows ``L`` where a leg is available at ``L`` and not at ``L + 1`` (its last bar).

        A leg that is merely cut off by the end of the dataset is never flagged.
        """
        act = np.asarray(active, dtype=np.bool_)
        last = np.zeros_like(act)
        if act.shape[0] >= 2:
            last[:-1] = act[:-1] & ~act[1:]
        return last

    @staticmethod
    def forced_rows(active: np.ndarray) -> np.ndarray:
        """Decision rows at which a name is closed: from 42 bars before its last bar on.

        For a leg whose last available bar is ``L`` the rows ``L - 42 .. L`` are flagged. The
        lead uses knowledge of the archive end (the public delisting announcement).
        """
        last = XsInputs.last_bar_rows(active)
        forced = np.zeros_like(last)
        lead = protocol.FORCED_CLOSE_LEAD_BARS
        for row, leg in zip(*np.nonzero(last), strict=True):
            forced[max(0, int(row) - lead) : int(row) + 1, int(leg)] = True
        return forced

    @classmethod
    def from_dataset(cls, dataset: MarketDataset, member: np.ndarray) -> XsInputs:
        active = np.asarray(dataset.resolved_array("asset_active"), dtype=np.bool_)
        mask = np.asarray(member, dtype=np.bool_)
        if mask.shape != active.shape:
            raise ValueError("membership must match the dataset (rows, legs)")
        return cls(
            legs=tuple(dataset.symbols),
            close=np.array(dataset.close, dtype=np.float64, copy=True),
            active=active.copy(),
            tradable=np.array(dataset.tradable, dtype=np.bool_, copy=True),
            funding_rate=np.array(dataset.funding_rate, dtype=np.float64, copy=True),
            funding_count=np.array(
                dataset.funding_event_count, dtype=np.int64, copy=True
            ),
            member=mask.copy(),
            forced=cls.forced_rows(active),
            last_bar=cls.last_bar_rows(active),
        )

    @property
    def n_rows(self) -> int:
        return int(self.close.shape[0])

    def copy(self) -> XsInputs:
        return XsInputs(
            legs=self.legs,
            close=self.close.copy(),
            active=self.active.copy(),
            tradable=self.tradable.copy(),
            funding_rate=self.funding_rate.copy(),
            funding_count=self.funding_count.copy(),
            member=self.member.copy(),
            forced=self.forced.copy(),
            last_bar=self.last_bar.copy(),
        )


@dataclass(slots=True)
class PolicyStats:
    """Reported (never gating) statistics of the weekly decisions."""

    weekly_decisions: int = 0
    book_decisions: int = 0
    flat_decisions: int = 0
    bound_decisions: int = 0
    bound_by_margin: int = 0
    bound_by_net_exposure: int = 0
    bound_by_both: int = 0
    forced_zeroed: int = 0
    names_ranked: list[int] = field(default_factory=list)
    pre_cap_gross: list[float] = field(default_factory=list)
    pre_cap_net: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        n = self.weekly_decisions
        return {
            "weekly_decisions": n,
            "book_decisions": self.book_decisions,
            "flat_decisions": self.flat_decisions,
            "cap_bound_decisions": self.bound_decisions,
            "cap_bound_fraction": (self.bound_decisions / n) if n else None,
            "bound_by_margin": self.bound_by_margin,
            "bound_by_net_exposure": self.bound_by_net_exposure,
            "bound_by_both": self.bound_by_both,
            "forced_close_zeroed_names": self.forced_zeroed,
            "mean_names_ranked": (
                float(np.mean(self.names_ranked)) if self.names_ranked else None
            ),
            "mean_pre_cap_gross": (
                float(np.mean(self.pre_cap_gross)) if self.pre_cap_gross else None
            ),
            "mean_pre_cap_net": (
                float(np.mean(self.pre_cap_net)) if self.pre_cap_net else None
            ),
            "cash_limit": protocol.CASH_LIMIT,
            "net_exposure_limit": protocol.NET_EXPOSURE_LIMIT,
            "initial_margin_rate": protocol.INITIAL_MARGIN_RATE,
        }


def book_target(
    inputs: XsInputs, *, arm: str, decision_row: int, signal_row: int
) -> tuple[np.ndarray, int, int]:
    """``(capped target weights, names ranked, forced names zeroed)`` of one weekly decision.

    ``decision_row`` is the row that decides (membership, tradability and the forced-close
    flags are read there); every price and funding input is read from ``signal_row`` and
    earlier. All-zero means a flat book.
    """
    n_legs = len(inputs.legs)
    flat = np.zeros(n_legs, dtype=np.float64)
    can_trade = (
        inputs.member[decision_row]
        & inputs.active[decision_row]
        & inputs.tradable[decision_row]
    )
    forced = inputs.forced[decision_row]
    if arm == protocol.CONTROL_CASH:
        return flat, 0, 0
    if arm == protocol.ARM_EW_LONG:
        chosen = np.flatnonzero(can_trade)
        if chosen.size == 0:
            return flat, 0, 0
        weights = np.zeros(n_legs, dtype=np.float64)
        weights[chosen] = 1.0 / chosen.size
        zeroed = int(np.count_nonzero(weights[forced]))
        weights[forced] = 0.0
        return cap_weights(weights), int(chosen.size), zeroed
    lo = signal_row - protocol.COV_BARS
    if lo < 0:
        return flat, 0, 0
    window = inputs.active[lo : signal_row + 1].all(axis=0)
    ok = can_trade & window
    candidates = np.flatnonzero(ok)
    if candidates.size == 0:
        return flat, 0, 0
    closes = inputs.close[lo : signal_row + 1][:, candidates]
    log_ret = np.log(closes[1:] / closes[:-1])
    sigma = np.std(log_ret[-protocol.SIGMA_BARS :], axis=0, ddof=1)
    funding = np.asarray(
        [
            funding_mean(
                inputs.funding_rate[:, leg],
                inputs.funding_count[:, leg],
                upto=signal_row,
            )
            for leg in candidates
        ],
        dtype=np.float64,
    )
    usable = np.isfinite(sigma) & (sigma > 0.0) & np.isfinite(funding)
    if not usable.any():
        return flat, 0, 0
    candidates = candidates[usable]
    sigma, funding, log_ret = sigma[usable], funding[usable], log_ret[:, usable]
    n = int(candidates.size)
    k = n // protocol.LEG_DIVISOR
    if k < 1:
        return flat, n, 0
    long_index, short_index = select_legs(score_of(arm, sigma, funding), k)
    raw = leg_weights(sigma, long_index, short_index)
    held = np.concatenate([long_index, short_index])
    scale = vol_target_scale(raw[held], log_ret[:, held])
    weights = np.zeros(n_legs, dtype=np.float64)
    weights[candidates] = raw * scale
    zeroed = int(np.count_nonzero(weights[forced]))
    weights[forced] = 0.0
    return cap_weights(weights), n, zeroed


class XsPolicy(PortfolioPolicy):
    """One arm of the weekly book over the stage's perp legs (see the module docstring)."""

    def __init__(
        self,
        *,
        arm: str,
        inputs: XsInputs,
        start_index: int,
        signal_lag_bars: int = 0,
        rebalance_every_bars: int = protocol.REBALANCE_EVERY_BARS,
        refuse_unclosed: bool = False,
        last_weekly_row: int | None = None,
    ) -> None:
        if arm not in protocol.ARMS:
            raise ValueError(f"unknown arm: {arm!r}")
        if start_index < 0 or signal_lag_bars < 0 or rebalance_every_bars < 1:
            raise ValueError("invalid start index, signal lag or stride")
        self._arm = arm
        self._inputs = inputs
        self._refuse_unclosed = refuse_unclosed
        self._last_weekly = last_weekly_row
        self._start = start_index
        self._lag = signal_lag_bars
        self._every = rebalance_every_bars
        self._last_index: int | None = None
        self._intended: np.ndarray | None = None
        self._last_risk_scale = 1.0
        self.stats = PolicyStats()

    def decide_portfolio(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> PortfolioTargetProposal:
        proposal = self._decide(snapshot)
        self._last_risk_scale = risk_scale_of(snapshot)
        return proposal

    def _validate(self, snapshot: PortfolioDecisionSnapshot) -> None:
        if tuple(snapshot.symbols) != self._inputs.legs:
            raise ValueError("snapshot symbols do not match the stage legs")
        if not 0 <= snapshot.index < self._inputs.n_rows:
            raise ValueError(f"snapshot.index {snapshot.index} outside the stage rows")
        if snapshot.index < self._start:
            raise ValueError("snapshot.index precedes the segment start")
        if self._last_index is not None and snapshot.index <= self._last_index:
            raise ValueError("snapshot.index must advance monotonically")
        self._last_index = snapshot.index

    @staticmethod
    def _held_proposal(
        snapshot: PortfolioDecisionSnapshot,
        scale: float,
        held: np.ndarray,
        intended: np.ndarray,
    ) -> PortfolioTargetProposal:
        """Proposal that keeps ``held`` legs where they are and re-targets the rest.

        Same non-compounding ``risk_scale`` contract as the anchor policy: risk multiplies
        the whole proposal by ``scale``, so an exact-quantity hold is only proposed while
        the scale is 1, and the pre-image ``current / scale`` otherwise.
        """
        current = np.asarray(snapshot.account.weights, dtype=np.float64)
        if scale >= 1.0:
            weights = np.where(held, current, intended)
            holds = tuple(int(i) for i in np.flatnonzero(held))
            return PortfolioTargetProposal(
                target_weights=tuple(float(w) for w in weights),
                hold_current_quantity_symbols=holds,
            )
        if scale > 0.0 and bool(held.any()):
            pre_image = np.where(held, current / scale, intended)
            if float(np.max(np.abs(pre_image), initial=0.0)) <= float(
                snapshot.risk.max_abs_weight
            ) and float(np.sum(np.abs(pre_image))) <= float(snapshot.risk.max_gross):
                return PortfolioTargetProposal(
                    target_weights=tuple(float(w) for w in pre_image)
                )
        return PortfolioTargetProposal(target_weights=tuple(float(w) for w in intended))

    def _decide(self, snapshot: PortfolioDecisionSnapshot) -> PortfolioTargetProposal:
        self._validate(snapshot)
        n_legs = len(self._inputs.legs)
        if self._arm == protocol.CONTROL_CASH:
            return PortfolioTargetProposal(target_weights=(0.0,) * n_legs)
        t = snapshot.index
        if self._refuse_unclosed:
            leftover = self._inputs.last_bar[t] & (
                np.abs(np.asarray(snapshot.account.quantities)) > _NO_TRADE_ATOL
            )
            if leftover.any():
                names = [self._inputs.legs[i] for i in np.flatnonzero(leftover)]
                raise UnsupportedWriteOffError(
                    f"{names} still held at the last available bar; the protocol has "
                    "no write-off branch (disclosed limitation)"
                )
        scale = risk_scale_of(snapshot)
        current = np.asarray(snapshot.account.weights, dtype=np.float64)

        if self._last_weekly is not None and t > self._last_weekly:
            # terminal unwind: the flat target is re-issued on every bar after the last
            # weekly decision, so partial fills complete before the terminal settlement
            self._intended = np.zeros(n_legs, dtype=np.float64)
            return PortfolioTargetProposal(target_weights=(0.0,) * n_legs)

        if (t - self._start) % self._every != 0:
            if self._intended is None:
                raise ValueError("the first decision must be a rebalance bar")
            forced_now = self._inputs.forced[t]
            intended = self._intended.copy()
            intended[forced_now] = 0.0
            self._intended = intended
            decreased = scale < self._last_risk_scale - _SCALE_EPS
            if decreased or scale <= 0.0:
                held = np.zeros(n_legs, dtype=bool)
            else:
                held = ~forced_now
            return self._held_proposal(snapshot, scale, held, intended)

        stats = self.stats
        stats.weekly_decisions += 1
        if self._last_weekly is not None and t == self._last_weekly:
            target, ranked, zeroed = np.zeros(n_legs, dtype=np.float64), 0, 0
        else:
            target, ranked, zeroed = book_target(
                self._inputs,
                arm=self._arm,
                decision_row=t,
                signal_row=max(0, t - self._lag),
            )
        stats.forced_zeroed += zeroed
        stats.names_ranked.append(ranked)
        if np.any(target):
            stats.book_decisions += 1
        else:
            stats.flat_decisions += 1
        self._intended = target
        can_act = self._inputs.active[t] & self._inputs.tradable[t]
        held = ~can_act & (np.abs(current) > _NO_TRADE_ATOL)
        effective = np.where(held, current, scale * target)
        factor, margin_factor, net_factor = cash_factors(effective)
        stats.pre_cap_gross.append(float(np.abs(effective).sum()))
        stats.pre_cap_net.append(float(effective.sum()))
        trades = not (
            held.all() or np.allclose(effective, current, atol=_NO_TRADE_ATOL)
        )
        binds = trades and factor < 1.0 and scale > 0.0
        if binds:
            stats.bound_decisions += 1
            if abs(margin_factor - net_factor) <= _TIE_ATOL:
                stats.bound_by_both += 1
            elif margin_factor < net_factor:
                stats.bound_by_margin += 1
            else:
                stats.bound_by_net_exposure += 1
            capped = effective * factor / scale
            return PortfolioTargetProposal(
                target_weights=tuple(float(w) for w in capped),
                hold_current_quantity_symbols=(),
            )
        return self._held_proposal(snapshot, scale, held, target)


__all__ = [
    "PolicyStats",
    "UnsupportedWriteOffError",
    "XsInputs",
    "XsPolicy",
    "average_ranks",
    "book_target",
    "cap_weights",
    "cash_factors",
    "cash_measure",
    "funding_mean",
    "leg_weights",
    "score_of",
    "select_legs",
    "shrunk_annual_covariance",
    "vol_target_scale",
    "zr",
]
