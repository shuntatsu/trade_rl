"""Fail-closed economic ranking for causal alpha V3 research candidates."""

from __future__ import annotations

from statistics import fmean
from typing import Mapping

from trade_rl.artifacts.hashing import content_digest
from trade_rl.workflows.universal_causal_alpha_v3_config import (
    CausalAlphaV3Candidate,
    CausalAlphaV3SelectionGate,
)
from trade_rl.workflows.universal_causal_alpha_v3_contracts import (
    CausalAlphaV3CandidateEvidence,
    CausalAlphaV3ReplayMetric,
    CausalAlphaV3SelectionEvidence,
)
from trade_rl.workflows.universal_causal_alpha_selection import (
    causal_alpha_unexplained_execution_rejection_count,
)


class CausalAlphaV3SelectionRejected(RuntimeError):
    """Complete V3 candidate evidence when no candidate clears selection."""

    def __init__(self, candidates: tuple[CausalAlphaV3CandidateEvidence, ...]) -> None:
        self.candidates = tuple(candidates)
        self.digest = content_digest(
            {
                "candidate_evidence_digests": tuple(item.digest for item in self.candidates),
                "schema_version": "causal_alpha_v3_selection_rejection_v1",
            }
        )
        super().__init__("no admissible causal alpha V3 candidate")

    def to_payload(self) -> dict[str, object]:
        return {
            "artifact_digest": self.digest,
            "candidate_evidence_digests": tuple(item.digest for item in self.candidates),
            "schema_version": "causal_alpha_v3_selection_rejection_v1",
        }


def _scope(metric: CausalAlphaV3ReplayMetric) -> tuple[str, int]:
    return (metric.symbol, metric.episode_index)


def _candidate_evidence(
    candidate: CausalAlphaV3Candidate,
    metrics: tuple[CausalAlphaV3ReplayMetric, ...],
    thresholds: CausalAlphaV3SelectionGate,
    *,
    complete_scope: frozenset[tuple[str, int]],
) -> CausalAlphaV3CandidateEvidence:
    if not metrics:
        raise ValueError("V3 candidate has no selection metrics")
    if any(item.candidate_digest != candidate.digest for item in metrics):
        raise ValueError("V3 candidate selection metrics drifted from candidate identity")
    observed_scope = frozenset(_scope(item) for item in metrics)
    if len(observed_scope) != len(metrics):
        raise ValueError("V3 candidate selection metric scope is duplicated")

    gross = tuple(item.gross_return for item in metrics)
    net = tuple(item.net_return for item in metrics)
    turnover = tuple(item.turnover_per_day for item in metrics)
    mean_gross = float(fmean(gross))
    mean_net = float(fmean(net))
    lower_tail = min(net)
    mean_turnover = float(fmean(turnover))
    total_cost = float(sum(item.total_execution_cost for item in metrics))
    total_trades = sum(item.trade_count for item in metrics)
    positive_fraction = sum(value > 0.0 for value in gross) / float(len(gross))
    unexplained = sum(
        causal_alpha_unexplained_execution_rejection_count(
            item.execution_rejection_reason_counts
        )
        for item in metrics
    )
    hard_risk = any(item.hard_risk_violation for item in metrics)

    reasons: list[str] = []
    if observed_scope != complete_scope:
        reasons.append("incomplete_selection_scope")
    if hard_risk:
        reasons.append("hard_risk_violation")
    if unexplained > thresholds.maximum_unexplained_execution_rejections:
        reasons.append("unexplained_execution_rejection")
    if total_trades == 0:
        reasons.append("no_meaningful_trades")
    if mean_gross < thresholds.minimum_mean_gross_return:
        reasons.append("mean_gross_return_below_minimum")
    if mean_net < thresholds.minimum_mean_net_return:
        reasons.append("mean_net_return_below_minimum")
    if lower_tail < thresholds.minimum_symbol_episode_net_return:
        reasons.append("lower_tail_net_return_below_floor")
    if mean_turnover > thresholds.maximum_mean_turnover_per_day:
        reasons.append("turnover_per_day_above_maximum")
    if positive_fraction < thresholds.minimum_positive_gross_episode_fraction:
        reasons.append("positive_gross_episode_fraction_below_minimum")

    return CausalAlphaV3CandidateEvidence(
        candidate=candidate,
        episode_metrics=metrics,
        lower_tail_net_return=lower_tail,
        mean_gross_return=mean_gross,
        mean_net_return=mean_net,
        turnover_per_day=mean_turnover,
        total_execution_cost=total_cost,
        positive_gross_episode_fraction=positive_fraction,
        total_trade_count=total_trades,
        unexplained_execution_rejection_count=unexplained,
        hard_risk_violation=hard_risk,
        admissible=not reasons,
        rejection_reasons=tuple(reasons),
    )


def rank_causal_alpha_v3_candidates(
    *,
    candidates: tuple[CausalAlphaV3Candidate, ...],
    metrics: Mapping[str, tuple[CausalAlphaV3ReplayMetric, ...]],
    thresholds: CausalAlphaV3SelectionGate,
    freeze_digest: str,
) -> CausalAlphaV3SelectionEvidence:
    values = tuple(candidates)
    if not values or len({item.digest for item in values}) != len(values):
        raise ValueError("V3 candidate grid must be non-empty and unique")
    if not isinstance(thresholds, CausalAlphaV3SelectionGate):
        raise TypeError("V3 selection thresholds are invalid")
    digests = {item.digest for item in values}
    if set(metrics) != digests:
        raise ValueError("V3 selection metrics must cover the complete frozen grid")
    scope_sets = {
        digest: frozenset(_scope(item) for item in metrics[digest]) for digest in digests
    }
    complete_scope = frozenset().union(*scope_sets.values())
    if not complete_scope:
        raise ValueError("V3 selection scope is empty")

    evidence = tuple(
        _candidate_evidence(
            candidate,
            tuple(metrics[candidate.digest]),
            thresholds,
            complete_scope=complete_scope,
        )
        for candidate in values
    )
    admissible = tuple(item for item in evidence if item.admissible)
    if not admissible:
        raise CausalAlphaV3SelectionRejected(evidence)
    selected = sorted(
        admissible,
        key=lambda item: (
            -item.lower_tail_net_return,
            -item.mean_net_return,
            -item.mean_gross_return,
            item.turnover_per_day,
            item.total_execution_cost,
            item.candidate.digest,
        ),
    )[0]
    return CausalAlphaV3SelectionEvidence(
        candidates=evidence,
        selected_candidate_digest=selected.candidate.digest,
        freeze_digest=freeze_digest,
    )


__all__ = [
    "CausalAlphaV3SelectionRejected",
    "rank_causal_alpha_v3_candidates",
]
