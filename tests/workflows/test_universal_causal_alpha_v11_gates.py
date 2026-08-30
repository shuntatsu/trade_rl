from __future__ import annotations

import math

from trade_rl.learning.causal_alpha_v6 import CausalAlphaV6Candidate
from trade_rl.learning.causal_alpha_v8 import CausalAlphaV8Candidate
from trade_rl.learning.causal_alpha_v11 import (
    CausalAlphaV11Candidate,
    CausalAlphaV11Config,
    CausalAlphaV11StudyArm,
    evaluate_v11_sizing_feasibility,
)
from trade_rl.workflows.universal_causal_alpha_v6_replay import (
    CausalAlphaV6ReplayMetric,
)
from trade_rl.workflows.universal_causal_alpha_v7_attribution import (
    CausalAlphaV7AttributionCell,
)
from trade_rl.workflows.universal_causal_alpha_v8_attribution import (
    CausalAlphaV8AttributionEvidence,
)
from trade_rl.workflows.universal_causal_alpha_v8_replay import (
    CausalAlphaV8ReplayMetric,
)
from trade_rl.workflows.universal_causal_alpha_v11_gates import (
    evaluate_causal_alpha_v11_selection,
)


def _digest(char: str) -> str:
    return char * 64


def _metric(
    *, candidate: CausalAlphaV8Candidate, symbol: str, episode: int
) -> CausalAlphaV8ReplayMetric:
    base = CausalAlphaV6ReplayMetric(
        run_manifest_digest=_digest("a"),
        v4_context_manifest_digest=_digest("b"),
        config_digest=_digest("c"),
        candidate=CausalAlphaV6Candidate.FAST_ONLY,
        symbol=symbol,
        episode_index=episode,
        contract_digest=_digest(str(episode + 1)),
        fit_digest=_digest("e"),
        forecast_digest=_digest("f"),
        target_path_digest=_digest("1"),
        decision_count=1,
        gross_return=0.02,
        gross_wealth=math.exp(0.02),
        net_return=0.01,
        net_wealth=math.exp(0.01),
        reward_total=1.0,
        reward_scale=100.0,
        turnover_per_day=0.1,
        total_execution_cost=0.001,
        target_change_count=1,
        submitted_change_count=1,
        downstream_no_trade_suppression_count=0,
        executed_change_count=1,
        closed_trade_count=1,
        sign_flip_count=0,
        maximum_drawdown=0.0,
        actionable_coverage=1.0,
        flat_time_fraction=0.0,
        time_weighted_absolute_exposure=0.1,
        completed_holding_durations_hours=(),
        open_holding_duration_hours=1.0,
        execution_rejection_reason_counts=(),
        risk_projection_reason_counts=(),
        target_reason_counts=(("entry", 1),),
        hard_risk_violation=False,
        has_meaningful_execution=True,
    )
    target_digest = _digest(str(tuple(CausalAlphaV8Candidate).index(candidate) + 4))
    attribution = CausalAlphaV8AttributionEvidence(
        candidate=candidate,
        target_path_digest=target_digest,
        boundaries_digest=_digest("3"),
        step_economics_digest=_digest("4"),
        decision_count=1,
        gross_log_return=0.02,
        net_log_return=0.01,
        total_execution_cost=0.001,
        total_exposure_hours=1.0,
        cells=(
            CausalAlphaV7AttributionCell(
                dimension="exposure",
                key="long",
                support=1,
                gross_log_return=0.02,
                net_log_return=0.01,
                execution_cost=0.001,
                exposure_hours=1.0,
            ),
        ),
    )
    return CausalAlphaV8ReplayMetric(
        candidate=candidate,
        v6_metric=base,
        attribution=attribution,
        v8_target_path_digest=target_digest,
        source_forecast_digest=_digest("5"),
        calibration_fit_digest=_digest("6"),
        v8_config_digest=_digest("7"),
    )


def test_v11_keeps_one_treatment_in_an_independent_three_way_gate() -> None:
    symbols = tuple(f"S{index}" for index in range(9))
    groups = tuple(
        tuple(
            _metric(candidate=candidate, symbol=symbol, episode=index)
            for index, symbol in enumerate(symbols)
        )
        for candidate in CausalAlphaV8Candidate
    )

    evidence = evaluate_causal_alpha_v11_selection(
        study_arm=CausalAlphaV11StudyArm.NEUTRAL_EXPIRY_2,
        cash_metrics=groups[0],
        control_metrics=groups[1],
        treatment_metrics=groups[2],
        expected_symbols=symbols,
        v11_config_digest=CausalAlphaV11Config().digest,
        diagnostic_digests=(_digest("8"),),
        sizing_feasibility=None,
    )

    assert tuple(
        item["candidate"] for item in evidence.to_payload()["candidates"]
    ) == tuple(candidate.value for candidate in CausalAlphaV11Candidate)
    assert evidence.study_arm is CausalAlphaV11StudyArm.NEUTRAL_EXPIRY_2
    assert evidence.passed


def test_v11_s1_non_executable_preflight_stops_before_selection() -> None:
    feasibility = evaluate_v11_sizing_feasibility(
        targets=[0.0, 0.099], entry_threshold=0.1, no_trade_band=0.05
    )

    evidence = evaluate_causal_alpha_v11_selection(
        study_arm=CausalAlphaV11StudyArm.CALIBRATED_EDGE_SIZING,
        cash_metrics=(),
        control_metrics=(),
        treatment_metrics=(),
        expected_symbols=tuple(f"S{index}" for index in range(9)),
        v11_config_digest=CausalAlphaV11Config().digest,
        diagnostic_digests=(_digest("8"),),
        sizing_feasibility=feasibility,
    )

    assert not evidence.passed
    assert evidence.rejection_reasons == ("sizing_non_executable",)
    assert evidence.source_v8 is None
