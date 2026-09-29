"""Result-blind and result-driven rules for versioned Study protocols."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import median

from trade_rl.evaluation.experiments.analysis import (
    PPO_HOLDING_DURATION_COMPARISON_SCHEMA,
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


def ppo_holding_expected_decision(
    metrics: PPOHoldingDurationMetrics,
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
    "ppo_holding_definition_matches",
    "ppo_holding_expected_decision",
    "ppo_holding_metrics",
    "ppo_holding_winner_digest",
]
