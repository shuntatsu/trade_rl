"""Deflated-Sharpe leaderboard over one family of the trial ledger.

The result is a plain immutable statistic table. It attaches no claim,
verdict or threshold.

A high Deflated Sharpe Ratio computed on data that was already used for
development is **not** out-of-sample evidence and does not authorise a
promotion, a final-test opening, an economic study or any live action. The
ranking only shows how candidates compare once the number of attempts behind
them is accounted for; it cannot repair a flawed protocol, leakage or data
that was reused across the very trials being ranked.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import StrEnum
from numbers import Real
from statistics import fmean

from trade_rl.evaluation.comparison.multiple_testing import (
    expected_maximum_sharpe,
    probabilistic_sharpe_ratio,
)
from trade_rl.evaluation.comparison.trial_ledger import (
    TrialLedger,
    TrialLedgerError,
    TrialStatus,
)


class VarianceSource(StrEnum):
    """Where the cross-trial Sharpe variance ``V`` came from."""

    COMPLETED_TRIALS = "COMPLETED_TRIALS"
    SUPPLIED = "SUPPLIED"


@dataclass(frozen=True, slots=True)
class LeaderboardEntry:
    """One COMPLETED trial's statistics under the family multiplicity."""

    rank: int
    trial_id: str
    method: str
    config_digest: str
    evaluation_data_label: str | None
    sharpe: float
    n_obs: int
    skew: float
    kurtosis: float
    psr_zero: float
    dsr: float


@dataclass(frozen=True, slots=True)
class FamilyLeaderboard:
    """Ranking of a family's COMPLETED trials by Deflated Sharpe Ratio.

    ``n_trials`` counts every trial in the family, FAILED and ABANDONED
    included, because failed attempts still consumed selection freedom.
    ``ledger_digest`` binds the table to the exact ledger it was computed
    from. Entries are ordered by descending DSR, ties broken by ``trial_id``.

    The table makes no claim. A high DSR on data already used for development
    is not out-of-sample evidence and authorises no promotion, final test,
    economic study or live action.
    """

    family: str
    n_trials: int
    n_completed: int
    n_failed: int
    n_abandoned: int
    trial_sr_variance: float
    variance_source: VarianceSource
    expected_max_sharpe: float
    ledger_digest: str
    entries: tuple[LeaderboardEntry, ...]


def _sample_variance(values: list[float]) -> float:
    mean = fmean(values)
    return math.fsum((item - mean) ** 2 for item in values) / (len(values) - 1)


def build_family_leaderboard(
    ledger: TrialLedger,
    family: str,
    *,
    trial_sr_variance: float | None = None,
) -> FamilyLeaderboard:
    """Rank a family's COMPLETED trials by Deflated Sharpe Ratio.

    * ``N`` is the number of all trials of ``family`` in the ledger, FAILED
      and ABANDONED included. It is a raw count: correlated trials make it a
      conservative (too large) stand-in for the effective number of
      independent trials.
    * ``V`` is the sample variance (``n - 1``) of the per-period SRs of the
      COMPLETED trials. With fewer than two COMPLETED trials the caller must
      supply ``trial_sr_variance``; nothing is invented. Supplying it while
      two or more trials have completed is ambiguous and refused. FAILED and
      ABANDONED trials have no SR, so ``V`` understates the dispersion that
      the full trial set would have shown.

    Every COMPLETED trial is scored with the same ``N``, ``V`` and expected
    maximum SR but its own ``n_obs``, skew and kurtosis. The function refuses
    an unknown family, an inadmissible ``V``, and any trial whose PSR
    variance term is not positive.

    The table carries no claim. A high DSR on known-used data is
    not out-of-sample evidence and authorises nothing.
    """

    if not isinstance(ledger, TrialLedger):
        raise TrialLedgerError("ledger must be a TrialLedger")
    if not isinstance(family, str):
        raise TrialLedgerError("family must be a string")
    members = ledger.family_records(family)
    if not members:
        raise TrialLedgerError(f"family {family!r} has no recorded trials")
    completed = [item for item in members if item.status is TrialStatus.COMPLETED]
    n_trials = len(members)
    n_failed = sum(item.status is TrialStatus.FAILED for item in members)
    n_abandoned = sum(item.status is TrialStatus.ABANDONED for item in members)
    moments = [item.moments() for item in completed]

    if len(completed) >= 2:
        if trial_sr_variance is not None:
            raise TrialLedgerError(
                "trial_sr_variance may be supplied only when fewer than two "
                "trials have completed"
            )
        variance = _sample_variance([item.sharpe for item in moments])
        source = VarianceSource.COMPLETED_TRIALS
    else:
        if trial_sr_variance is None:
            raise TrialLedgerError(
                "fewer than two completed trials: supply trial_sr_variance explicitly"
            )
        supplied: object = trial_sr_variance
        if isinstance(supplied, bool) or not isinstance(supplied, Real):
            raise TrialLedgerError("trial_sr_variance must be a finite number")
        variance = float(supplied)
        source = VarianceSource.SUPPLIED
    if not math.isfinite(variance) or variance < 0.0:
        raise TrialLedgerError("trial_sr_variance must be finite and non-negative")

    benchmark = expected_maximum_sharpe(n_trials, variance)
    scored: list[tuple[float, str, int]] = []
    rows: list[LeaderboardEntry] = []
    for record, stats in zip(completed, moments, strict=True):
        try:
            psr_zero = probabilistic_sharpe_ratio(
                stats.sharpe, stats.n_obs, stats.skew, stats.kurtosis, 0.0
            )
            dsr = probabilistic_sharpe_ratio(
                stats.sharpe, stats.n_obs, stats.skew, stats.kurtosis, benchmark
            )
        except ValueError as exc:
            raise TrialLedgerError(
                f"trial {record.trial_id!r} cannot be scored: {exc}"
            ) from exc
        scored.append((-dsr, record.trial_id, len(rows)))
        rows.append(
            LeaderboardEntry(
                rank=0,
                trial_id=record.trial_id,
                method=record.method,
                config_digest=record.config_digest,
                evaluation_data_label=record.evaluation_data_label,
                sharpe=stats.sharpe,
                n_obs=stats.n_obs,
                skew=stats.skew,
                kurtosis=stats.kurtosis,
                psr_zero=psr_zero,
                dsr=dsr,
            )
        )
    scored.sort()
    ranked = tuple(
        replace(rows[index], rank=position)
        for position, (_, _, index) in enumerate(scored, start=1)
    )
    return FamilyLeaderboard(
        family=family,
        n_trials=n_trials,
        n_completed=len(completed),
        n_failed=n_failed,
        n_abandoned=n_abandoned,
        trial_sr_variance=variance,
        variance_source=source,
        expected_max_sharpe=benchmark,
        ledger_digest=ledger.digest(),
        entries=ranked,
    )


__all__ = [
    "FamilyLeaderboard",
    "LeaderboardEntry",
    "VarianceSource",
    "build_family_leaderboard",
]
