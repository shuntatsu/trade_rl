"""Result-blind and result-driven rules for versioned Study protocols."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import median

from trade_rl.evaluation.experiments.analysis import (
    PPO_HOLDING_DURATION_COMPARISON_SCHEMA,
    PPO_SHARED_CASH_HOLDING_DURATION_COMPARISON_SCHEMA,
)
from trade_rl.evaluation.experiments.contracts import (
    PPO_HOLDING_DURATION_HORIZONS,
    PPO_HOLDING_DURATION_MAX_DRAWDOWN,
    ControlledFactor,
    ExperimentComparison,
    ExperimentDecisionKind,
    ExperimentDefinition,
    StudyPlan,
)
from trade_rl.evaluation.experiments.errors import ArtifactIntegrityError


@dataclass(frozen=True, slots=True)
class PPOHoldingDurationMetrics:
    """The persisted PPO-only selection summary for one holding horizon."""

    score: float
    median_excess_return: float
    worst_max_drawdown: float
    symbol_count: int
    terminal_settlement_complete: bool = True

    @property
    def eligible(self) -> bool:
        return (
            self.median_excess_return > 0.0
            and self.worst_max_drawdown <= PPO_HOLDING_DURATION_MAX_DRAWDOWN
            and self.terminal_settlement_complete
        )


@dataclass(frozen=True, slots=True)
class PPOSharedCashHoldingDurationMetrics:
    """Selection summary for the combined, shared-cash PPO portfolio."""

    score: float
    median_excess_return: float
    worst_max_drawdown: float
    seed_count: int
    terminal_settlement_complete: bool = True

    @property
    def eligible(self) -> bool:
        return (
            self.score > 0.0
            and self.median_excess_return > 0.0
            and self.worst_max_drawdown <= PPO_HOLDING_DURATION_MAX_DRAWDOWN
            and self.terminal_settlement_complete
        )


def ppo_holding_definition_matches(
    plan: StudyPlan,
    definition: ExperimentDefinition,
) -> bool:
    """Whether a definition occupies its preregistered horizon slot."""

    if not plan.is_ppo_holding_duration_study:
        return True
    if not 1 <= definition.sequence <= len(PPO_HOLDING_DURATION_HORIZONS):
        return False
    return (
        definition.factor is ControlledFactor.PPO_MINIMUM_HOLD
        and definition.candidate_config.ppo_minimum_hold_bars
        == PPO_HOLDING_DURATION_HORIZONS[definition.sequence - 1]
    )


def _selection_number(payload: Mapping[str, object], name: str) -> float:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactIntegrityError(
            f"PPO holding-duration selection metric {name} is malformed"
        )
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ArtifactIntegrityError(
            f"PPO holding-duration selection metric {name} is non-finite"
        )
    return resolved


def _selection_count(payload: Mapping[str, object], name: str) -> int:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ArtifactIntegrityError(
            f"PPO holding-duration selection count {name} is malformed"
        )
    return value


def ppo_holding_metrics(
    comparison: ExperimentComparison,
    *,
    expected_symbols: tuple[str, ...],
    expected_seeds: tuple[int, ...],
) -> PPOHoldingDurationMetrics:
    """Read the PPO summary used by the preregistered horizon decision rule."""

    factor_effect = comparison.to_payload()["factor_effect"]
    if not isinstance(factor_effect, Mapping):
        raise ArtifactIntegrityError("factor-effect comparison must be an object")
    if factor_effect.get("schema_version") != PPO_HOLDING_DURATION_COMPARISON_SCHEMA:
        raise ArtifactIntegrityError(
            "PPO holding-duration protocol requires factor-effect schema v3"
        )
    if factor_effect.get("seeds") != list(expected_seeds):
        raise ArtifactIntegrityError("PPO selection seed roster is incomplete")
    cross_seed = factor_effect.get("cross_seed")
    if not isinstance(cross_seed, Mapping):
        raise ArtifactIntegrityError("factor-effect cross_seed summary is missing")
    ppo = cross_seed.get("ppo")
    if not isinstance(ppo, Mapping):
        raise ArtifactIntegrityError("PPO cross-seed summary is missing")
    symbol_count = ppo.get("symbol_count")
    if (
        isinstance(symbol_count, bool)
        or not isinstance(symbol_count, int)
        or symbol_count != len(expected_symbols)
    ):
        raise ArtifactIntegrityError("PPO selection symbol roster is incomplete")
    seed_count = ppo.get("seed_count")
    if (
        isinstance(seed_count, bool)
        or not isinstance(seed_count, int)
        or seed_count != len(expected_seeds)
    ):
        raise ArtifactIntegrityError("PPO selection seed roster is incomplete")
    by_seed = ppo.get("by_seed")
    expected_seed_keys = {str(seed) for seed in expected_seeds}
    if not isinstance(by_seed, Mapping) or set(by_seed) != expected_seed_keys:
        raise ArtifactIntegrityError("PPO selection seed-symbol cells are incomplete")

    per_seed_candidate_returns: list[float] = []
    per_seed_excess_returns: list[float] = []
    per_seed_candidate_drawdowns: list[float] = []
    per_seed_baseline_drawdowns: list[float] = []
    per_seed_worst_drawdowns: list[float] = []
    baseline_settled_count = 0
    candidate_settled_count = 0
    for seed in expected_seeds:
        seed_metrics = by_seed.get(str(seed))
        if not isinstance(seed_metrics, Mapping):
            raise ArtifactIntegrityError("PPO per-seed selection metrics are malformed")
        seed_symbol_count = seed_metrics.get("symbol_count")
        if (
            isinstance(seed_symbol_count, bool)
            or not isinstance(seed_symbol_count, int)
            or seed_symbol_count != len(expected_symbols)
        ):
            raise ArtifactIntegrityError("PPO seed-symbol cells are incomplete")
        per_seed_candidate_returns.append(
            _selection_number(seed_metrics, "mean_candidate_total_return")
        )
        per_seed_excess_returns.append(
            _selection_number(seed_metrics, "mean_excess_total_return")
        )
        seed_candidate_drawdown = _selection_number(
            seed_metrics,
            "worst_candidate_max_drawdown",
        )
        seed_baseline_drawdown = _selection_number(
            seed_metrics,
            "worst_baseline_max_drawdown",
        )
        seed_account_drawdown = _selection_number(
            seed_metrics,
            "worst_account_max_drawdown",
        )
        if seed_account_drawdown != max(
            seed_candidate_drawdown,
            seed_baseline_drawdown,
        ):
            raise ArtifactIntegrityError(
                "PPO per-seed drawdown summary is inconsistent"
            )
        per_seed_candidate_drawdowns.append(seed_candidate_drawdown)
        per_seed_baseline_drawdowns.append(seed_baseline_drawdown)
        per_seed_worst_drawdowns.append(seed_account_drawdown)
        seed_baseline_settled = _selection_count(
            seed_metrics,
            "baseline_terminal_settlement_complete_account_count",
        )
        seed_candidate_settled = _selection_count(
            seed_metrics,
            "candidate_terminal_settlement_complete_account_count",
        )
        if seed_baseline_settled > len(
            expected_symbols
        ) or seed_candidate_settled > len(expected_symbols):
            raise ArtifactIntegrityError(
                "PPO terminal-settlement account count exceeds the symbol roster"
            )
        baseline_settled_count += seed_baseline_settled
        candidate_settled_count += seed_candidate_settled

    score = _selection_number(ppo, "median_candidate_total_return")
    median_excess_return = _selection_number(ppo, "median_excess_total_return")
    reported_candidate_drawdown = _selection_number(
        ppo,
        "worst_candidate_max_drawdown",
    )
    reported_baseline_drawdown = _selection_number(
        ppo,
        "worst_baseline_max_drawdown",
    )
    reported_worst_drawdown = _selection_number(ppo, "worst_account_max_drawdown")
    reported_baseline_settled = _selection_count(
        ppo,
        "baseline_terminal_settlement_complete_account_count",
    )
    reported_candidate_settled = _selection_count(
        ppo,
        "candidate_terminal_settlement_complete_account_count",
    )
    if (
        score != float(median(per_seed_candidate_returns))
        or median_excess_return != float(median(per_seed_excess_returns))
        or reported_candidate_drawdown != max(per_seed_candidate_drawdowns)
        or reported_baseline_drawdown != max(per_seed_baseline_drawdowns)
        or reported_worst_drawdown
        != max(reported_candidate_drawdown, reported_baseline_drawdown)
        or reported_worst_drawdown != max(per_seed_worst_drawdowns)
        or reported_baseline_settled != baseline_settled_count
        or reported_candidate_settled != candidate_settled_count
    ):
        raise ArtifactIntegrityError("PPO cross-seed selection summary is inconsistent")
    return PPOHoldingDurationMetrics(
        score=score,
        median_excess_return=median_excess_return,
        worst_max_drawdown=reported_worst_drawdown,
        symbol_count=symbol_count,
        terminal_settlement_complete=(
            baseline_settled_count == len(expected_seeds) * len(expected_symbols)
            and candidate_settled_count == len(expected_seeds) * len(expected_symbols)
        ),
    )


def ppo_shared_cash_holding_metrics(
    comparison: ExperimentComparison,
    *,
    expected_seeds: tuple[int, ...],
) -> PPOSharedCashHoldingDurationMetrics:
    """Validate and summarize a combined shared-cash PPO comparison."""

    if not expected_seeds or len(set(expected_seeds)) != len(expected_seeds):
        raise ArtifactIntegrityError("shared-cash PPO seed roster is malformed")
    factor_effect = comparison.to_payload()["factor_effect"]
    if not isinstance(factor_effect, Mapping):
        raise ArtifactIntegrityError("factor-effect comparison must be an object")
    if (
        factor_effect.get("schema_version")
        != PPO_SHARED_CASH_HOLDING_DURATION_COMPARISON_SCHEMA
    ):
        raise ArtifactIntegrityError(
            "shared-cash PPO protocol requires factor-effect schema v4"
        )
    if factor_effect.get("seeds") != list(expected_seeds):
        raise ArtifactIntegrityError("shared-cash PPO seed roster is incomplete")
    cross_seed = factor_effect.get("cross_seed")
    if not isinstance(cross_seed, Mapping):
        raise ArtifactIntegrityError("factor-effect cross_seed summary is missing")
    portfolio = cross_seed.get("shared_cash_ppo")
    if not isinstance(portfolio, Mapping):
        raise ArtifactIntegrityError("shared-cash PPO portfolio summary is missing")
    if _selection_count(portfolio, "seed_count") != len(expected_seeds):
        raise ArtifactIntegrityError("shared-cash PPO seed roster is incomplete")
    by_seed = portfolio.get("by_seed")
    expected_seed_keys = {str(seed) for seed in expected_seeds}
    if not isinstance(by_seed, Mapping) or set(by_seed) != expected_seed_keys:
        raise ArtifactIntegrityError("shared-cash PPO per-seed results are incomplete")

    candidate_returns: list[float] = []
    excess_returns: list[float] = []
    candidate_drawdowns: list[float] = []
    baseline_drawdowns: list[float] = []
    candidate_terminal_count = 0
    baseline_terminal_count = 0
    period_counts: set[int] = set()
    for seed in expected_seeds:
        seed_metrics = by_seed.get(str(seed))
        if not isinstance(seed_metrics, Mapping):
            raise ArtifactIntegrityError(
                "shared-cash PPO per-seed metrics are malformed"
            )
        baseline_return = _selection_number(seed_metrics, "baseline_total_return")
        candidate_return = _selection_number(seed_metrics, "candidate_total_return")
        excess_return = _selection_number(seed_metrics, "excess_total_return")
        if not math.isclose(
            excess_return,
            candidate_return - baseline_return,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ArtifactIntegrityError(
                "shared-cash PPO per-seed excess return is inconsistent"
            )
        baseline_drawdown = _selection_number(
            seed_metrics,
            "baseline_max_drawdown",
        )
        candidate_drawdown = _selection_number(
            seed_metrics,
            "candidate_max_drawdown",
        )
        worst_drawdown = _selection_number(
            seed_metrics,
            "worst_account_max_drawdown",
        )
        if min(baseline_drawdown, candidate_drawdown) < 0.0 or worst_drawdown != max(
            baseline_drawdown,
            candidate_drawdown,
        ):
            raise ArtifactIntegrityError(
                "shared-cash PPO per-seed drawdown summary is inconsistent"
            )
        baseline_settled = seed_metrics.get("baseline_terminal_settlement_complete")
        candidate_settled = seed_metrics.get("candidate_terminal_settlement_complete")
        if not isinstance(baseline_settled, bool) or not isinstance(
            candidate_settled, bool
        ):
            raise ArtifactIntegrityError(
                "shared-cash PPO terminal-settlement state is malformed"
            )
        period_count = _selection_count(seed_metrics, "return_period_count")
        if period_count == 0:
            raise ArtifactIntegrityError(
                "shared-cash PPO comparison has no return periods"
            )
        period_counts.add(period_count)
        candidate_returns.append(candidate_return)
        excess_returns.append(excess_return)
        candidate_drawdowns.append(candidate_drawdown)
        baseline_drawdowns.append(baseline_drawdown)
        candidate_terminal_count += int(candidate_settled)
        baseline_terminal_count += int(baseline_settled)

    if len(period_counts) != 1:
        raise ArtifactIntegrityError(
            "shared-cash PPO seed comparisons use different return periods"
        )
    score = _selection_number(portfolio, "median_candidate_total_return")
    median_excess_return = _selection_number(
        portfolio,
        "median_excess_total_return",
    )
    reported_candidate_drawdown = _selection_number(
        portfolio,
        "worst_candidate_max_drawdown",
    )
    reported_baseline_drawdown = _selection_number(
        portfolio,
        "worst_baseline_max_drawdown",
    )
    reported_worst_drawdown = _selection_number(
        portfolio,
        "worst_account_max_drawdown",
    )
    reported_baseline_settled = _selection_count(
        portfolio,
        "baseline_terminal_settlement_complete_seed_count",
    )
    reported_candidate_settled = _selection_count(
        portfolio,
        "candidate_terminal_settlement_complete_seed_count",
    )
    if (
        score != float(median(candidate_returns))
        or median_excess_return != float(median(excess_returns))
        or reported_candidate_drawdown != max(candidate_drawdowns)
        or reported_baseline_drawdown != max(baseline_drawdowns)
        or reported_worst_drawdown
        != max(reported_candidate_drawdown, reported_baseline_drawdown)
        or reported_baseline_settled != baseline_terminal_count
        or reported_candidate_settled != candidate_terminal_count
    ):
        raise ArtifactIntegrityError(
            "shared-cash PPO cross-seed summary is inconsistent"
        )
    return PPOSharedCashHoldingDurationMetrics(
        score=score,
        median_excess_return=median_excess_return,
        worst_max_drawdown=reported_worst_drawdown,
        seed_count=len(expected_seeds),
        terminal_settlement_complete=(
            baseline_terminal_count == len(expected_seeds)
            and candidate_terminal_count == len(expected_seeds)
        ),
    )


def ppo_study_metrics(
    plan: StudyPlan,
    comparison: ExperimentComparison,
) -> PPOHoldingDurationMetrics | PPOSharedCashHoldingDurationMetrics:
    """Apply the selector bound to the Study's immutable protocol version."""

    if plan.is_ppo_shared_cash_holding_duration_study:
        return ppo_shared_cash_holding_metrics(
            comparison,
            expected_seeds=plan.ppo_seeds,
        )
    return ppo_holding_metrics(
        comparison,
        expected_symbols=plan.symbols,
        expected_seeds=plan.ppo_seeds,
    )


def ppo_holding_expected_decision(
    metrics: PPOHoldingDurationMetrics | PPOSharedCashHoldingDurationMetrics,
) -> ExperimentDecisionKind:
    """Return the only valid decision under the frozen eligibility rule."""

    return (
        ExperimentDecisionKind.ACCEPT_CANDIDATE
        if metrics.eligible
        else ExperimentDecisionKind.KEEP_BASELINE
    )


def ppo_holding_winner_digest(
    eligible: Sequence[tuple[int, float, str]],
) -> str | None:
    """Choose highest PPO score, breaking exact ties toward the shorter hold."""

    if not eligible:
        return None
    return min(eligible, key=lambda item: (-item[1], item[0]))[2]


__all__ = [
    "PPOHoldingDurationMetrics",
    "PPOSharedCashHoldingDurationMetrics",
    "ppo_holding_definition_matches",
    "ppo_holding_expected_decision",
    "ppo_holding_metrics",
    "ppo_shared_cash_holding_metrics",
    "ppo_study_metrics",
    "ppo_holding_winner_digest",
]
