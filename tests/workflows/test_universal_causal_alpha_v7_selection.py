from __future__ import annotations

import hashlib
import math
from typing import Any

import pytest

from trade_rl.learning.causal_alpha_v6 import CausalAlphaV6Candidate
from trade_rl.learning.causal_alpha_v7 import CausalAlphaV7Candidate
from trade_rl.workflows.universal_causal_alpha_v6_replay import (
    CausalAlphaV6ReplayMetric,
)
from trade_rl.workflows.universal_causal_alpha_v7_attribution import (
    CausalAlphaV7AttributionCell,
    CausalAlphaV7AttributionEvidence,
)
from trade_rl.workflows.universal_causal_alpha_v7_replay import (
    CausalAlphaV7ReplayMetric,
)
from trade_rl.workflows.universal_causal_alpha_v7_selection import (
    evaluate_causal_alpha_v7_selection,
)

_SYMBOLS = tuple(f"S{index}" for index in range(9))


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _metric(
    candidate: CausalAlphaV7Candidate,
    symbol: str,
    episode: int,
    *,
    gross: float = 0.02,
    net: float = 0.01,
    turnover: float = 0.1,
    cost: float = 0.001,
    meaningful: bool = True,
    hard_risk: bool = False,
    rejections: tuple[tuple[str, int], ...] = (),
    source_forecast: str | None = None,
) -> CausalAlphaV7ReplayMetric:
    scope = f"{symbol}:{episode}"
    base = CausalAlphaV6ReplayMetric(
        run_manifest_digest=_digest("run"),
        v4_context_manifest_digest=_digest("context"),
        config_digest=_digest("v6-config"),
        candidate=CausalAlphaV6Candidate.FAST_ONLY,
        symbol=symbol,
        episode_index=episode,
        contract_digest=_digest(f"contract:{scope}"),
        fit_digest=_digest(f"fit:{episode}"),
        forecast_digest=_digest(f"effective:{candidate.value}:{scope}"),
        target_path_digest=_digest(f"v6-target:{candidate.value}:{scope}"),
        decision_count=1,
        gross_return=gross,
        gross_wealth=math.exp(gross),
        net_return=net,
        net_wealth=math.exp(net),
        reward_total=100.0 * net,
        reward_scale=100.0,
        turnover_per_day=turnover,
        total_execution_cost=cost,
        target_change_count=int(meaningful),
        submitted_change_count=int(meaningful),
        downstream_no_trade_suppression_count=0,
        executed_change_count=int(meaningful),
        closed_trade_count=int(meaningful),
        sign_flip_count=0,
        maximum_drawdown=0.01,
        actionable_coverage=1.0,
        flat_time_fraction=0.0,
        time_weighted_absolute_exposure=0.1,
        completed_holding_durations_hours=(),
        open_holding_duration_hours=1.0,
        execution_rejection_reason_counts=rejections,
        risk_projection_reason_counts=(),
        target_reason_counts=(("hold_position", 1),),
        hard_risk_violation=hard_risk,
        has_meaningful_execution=meaningful,
    )
    v7_target = _digest(f"v7-target:{candidate.value}:{scope}")
    attribution = CausalAlphaV7AttributionEvidence(
        candidate=candidate,
        target_path_digest=v7_target,
        boundaries_digest=_digest("boundaries"),
        step_economics_digest=_digest(f"steps:{candidate.value}:{scope}"),
        decision_count=1,
        gross_log_return=gross,
        net_log_return=net,
        total_execution_cost=cost,
        total_exposure_hours=1.0,
        cells=(
            CausalAlphaV7AttributionCell(
                dimension="exposure",
                key="long",
                support=1,
                gross_log_return=gross,
                net_log_return=net,
                execution_cost=cost,
                exposure_hours=1.0,
            ),
        ),
    )
    return CausalAlphaV7ReplayMetric(
        candidate=candidate,
        v6_metric=base,
        attribution=attribution,
        v7_target_path_digest=v7_target,
        source_forecast_digest=(
            _digest(f"source:{scope}") if source_forecast is None else source_forecast
        ),
        calibration_fit_digest=_digest(f"calibration:{episode}"),
        v7_config_digest=_digest("v7-config"),
    )


def _records(
    candidate: CausalAlphaV7Candidate,
    **kwargs: Any,
) -> tuple[CausalAlphaV7ReplayMetric, ...]:
    return tuple(
        _metric(candidate, symbol, episode, **kwargs)
        for episode in range(2)
        for symbol in _SYMBOLS
    )


def test_v7_selection_chooses_highest_eligible_after_cost_wealth() -> None:
    evidence = evaluate_causal_alpha_v7_selection(
        (
            *_records(CausalAlphaV7Candidate.V6_CONTROL, net=0.010),
            *_records(CausalAlphaV7Candidate.SYMMETRIC_CONTRARIAN, net=0.012),
            *_records(CausalAlphaV7Candidate.CAUSAL_CALIBRATED, net=0.015),
        ),
        expected_symbols=_SYMBOLS,
    )

    assert evidence.passed
    assert evidence.selected_candidate is CausalAlphaV7Candidate.CAUSAL_CALIBRATED
    assert evidence.paired_scope_count == 18
    assert tuple(item.candidate for item in evidence.candidates) == tuple(
        CausalAlphaV7Candidate
    )
    assert evidence.candidates[2].symbol_balanced_net_wealth == pytest.approx(
        math.exp(0.03)
    )


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    (
        ({"gross": 0.0}, "symbol_balanced_gross_wealth"),
        ({"net": -0.01}, "symbol_balanced_net_wealth"),
        ({"turnover": 1.01}, "turnover_p95"),
        ({"meaningful": False}, "no_meaningful_execution"),
        ({"hard_risk": True}, "hard_risk_violation"),
        (
            {"rejections": (("venue_rejected", 1),)},
            "unexplained_execution_rejection",
        ),
    ),
)
def test_v7_selection_keeps_every_universal_gate(
    kwargs: Any,
    reason: str,
) -> None:
    metrics = tuple(
        metric
        for candidate in CausalAlphaV7Candidate
        for metric in _records(candidate, **kwargs)
    )

    evidence = evaluate_causal_alpha_v7_selection(metrics, expected_symbols=_SYMBOLS)

    assert not evidence.passed
    assert all(
        reason in candidate.rejection_reasons for candidate in evidence.candidates
    )


def test_v7_selection_rejects_unpaired_source_forecast_identity() -> None:
    metrics = [
        metric for candidate in CausalAlphaV7Candidate for metric in _records(candidate)
    ]
    metrics[-1] = _metric(
        CausalAlphaV7Candidate.CAUSAL_CALIBRATED,
        "S8",
        1,
        source_forecast=_digest("drift"),
    )

    evidence = evaluate_causal_alpha_v7_selection(
        tuple(metrics),
        expected_symbols=_SYMBOLS,
    )

    assert not evidence.passed
    assert evidence.rejection_reasons == ("scope_pairing",)
