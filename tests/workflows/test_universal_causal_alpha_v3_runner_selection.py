from __future__ import annotations

import pytest

from trade_rl.learning.causal_alpha_v3 import (
    CausalAlphaV3FitConfig,
    CausalAlphaV3TargetConfig,
)
from trade_rl.workflows.universal_causal_alpha_v3_config import (
    CausalAlphaV3Candidate,
    CausalAlphaV3SelectionGate,
)
from trade_rl.workflows.universal_causal_alpha_v3_contracts import CausalAlphaV3ReplayMetric
from trade_rl.workflows.universal_causal_alpha_v3_selection import (
    CausalAlphaV3SelectionRejected,
    rank_causal_alpha_v3_candidates,
)


def _candidate(name: str, uncertainty: float) -> CausalAlphaV3Candidate:
    return CausalAlphaV3Candidate(
        name=name,
        fit=CausalAlphaV3FitConfig(ridge_strength=0.1),
        target=CausalAlphaV3TargetConfig(
            target_magnitudes=(0.0, 0.05, 0.1),
            uncertainty_multiplier=uncertainty,
            execution_cost_multiplier=1.5,
            edge_margin=0.001,
            alpha_rebalance_decisions=4,
            strong_reversal_threshold=0.02,
            max_target_delta=0.1,
        ),
    )


def _metric(candidate: CausalAlphaV3Candidate, episode: int, *, gross: float, net: float, turnover: float = 0.2) -> CausalAlphaV3ReplayMetric:
    token = f"{episode + 1:x}"
    return CausalAlphaV3ReplayMetric(
        run_manifest_digest="1" * 64,
        freeze_digest="2" * 64,
        candidate_digest=candidate.digest,
        symbol="BTCUSDT",
        episode_index=episode,
        contract_digest=(token * 64)[:64],
        fit_digest="4" * 64,
        forecast_digest="5" * 64,
        target_path_digest="6" * 64,
        gross_return=gross,
        net_return=net,
        turnover_per_day=turnover,
        total_execution_cost=1.0,
        trade_count=2,
        submitted_change_count=2,
        sign_flip_count=0,
        liquidity_deleveraging_count=0,
        execution_rejection_reason_counts=(),
        risk_projection_reason_counts=(),
        target_reason_counts=(("hold", 2), ("rebalance", 2)),
        hard_risk_violation=False,
    )


def _gate() -> CausalAlphaV3SelectionGate:
    return CausalAlphaV3SelectionGate(
        minimum_mean_gross_return=0.0,
        minimum_mean_net_return=0.0,
        minimum_symbol_episode_net_return=-0.05,
        maximum_mean_turnover_per_day=1.0,
        maximum_unexplained_execution_rejections=0,
        minimum_positive_gross_episode_fraction=0.5,
    )


def test_selection_ranks_admissible_candidates_by_tail_then_net() -> None:
    first = _candidate("first", 1.0)
    second = _candidate("second", 1.5)
    metrics = {
        first.digest: (
            _metric(first, 0, gross=0.02, net=0.010),
            _metric(first, 1, gross=0.01, net=0.005),
        ),
        second.digest: (
            _metric(second, 0, gross=0.03, net=0.012),
            _metric(second, 1, gross=0.01, net=0.007),
        ),
    }

    evidence = rank_causal_alpha_v3_candidates(
        candidates=(first, second),
        metrics=metrics,
        thresholds=_gate(),
        freeze_digest="2" * 64,
    )

    assert evidence.selected_candidate_digest == second.digest
    assert evidence.promotion_eligible is False
    assert all(item.admissible for item in evidence.candidates)


def test_selection_rejects_irrecoverable_tail_and_hard_risk() -> None:
    candidate = _candidate("bad", 1.0)
    bad_tail = _metric(candidate, 0, gross=0.02, net=-0.06)
    hard_risk = CausalAlphaV3ReplayMetric(
        **{
            **bad_tail.to_payload(include_digest=False),
            "net_return": 0.01,
            "hard_risk_violation": True,
        }
    )

    with pytest.raises(CausalAlphaV3SelectionRejected) as error:
        rank_causal_alpha_v3_candidates(
            candidates=(candidate,),
            metrics={candidate.digest: (bad_tail, hard_risk)},
            thresholds=_gate(),
            freeze_digest="2" * 64,
        )

    reasons = set(error.value.candidates[0].rejection_reasons)
    assert "lower_tail_net_return_below_floor" in reasons
    assert "hard_risk_violation" in reasons


def test_metric_identifies_irrecoverable_rejection_conditions() -> None:
    candidate = _candidate("bad", 1.0)
    metric = _metric(candidate, 0, gross=0.01, net=-0.051)

    assert metric.irrecoverably_rejected(_gate()) is True
