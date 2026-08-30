from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from trade_rl.learning.causal_alpha_v6 import (
    CausalAlphaV6Candidate,
)
from trade_rl.learning.causal_alpha_v8 import CausalAlphaV8Candidate
from trade_rl.learning.rollout_evaluation import ActionPathLifecycleTrace
from trade_rl.workflows.universal_causal_alpha_v6_replay import (
    CausalAlphaV6ReplayMetric,
)
from trade_rl.workflows.universal_causal_alpha_v7_attribution import (
    CausalAlphaV7AttributionCell,
)
from trade_rl.workflows.universal_causal_alpha_v8_attribution import (
    CausalAlphaV8AttributionEvidence,
)
from trade_rl.workflows.universal_causal_alpha_v8_gates import (
    evaluate_causal_alpha_v8_selection,
)
from trade_rl.workflows.universal_causal_alpha_v8_replay import (
    CausalAlphaV8ReplayMetric,
)


def _digest(char: str) -> str:
    return char * 64


def _metric() -> CausalAlphaV8ReplayMetric:
    base = CausalAlphaV6ReplayMetric(
        run_manifest_digest=_digest("a"),
        v4_context_manifest_digest=_digest("b"),
        config_digest=_digest("c"),
        candidate=CausalAlphaV6Candidate.FAST_ONLY,
        symbol="BTCUSDT",
        episode_index=3,
        contract_digest=_digest("d"),
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
    target_digest = _digest("2")
    attribution = CausalAlphaV8AttributionEvidence(
        candidate=CausalAlphaV8Candidate.ROBUST_CONTRARIAN,
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
        candidate=CausalAlphaV8Candidate.ROBUST_CONTRARIAN,
        v6_metric=base,
        attribution=attribution,
        v8_target_path_digest=target_digest,
        source_forecast_digest=_digest("5"),
        calibration_fit_digest=_digest("6"),
        v8_config_digest=_digest("7"),
    )


def _lifecycle_trace() -> ActionPathLifecycleTrace:
    return ActionPathLifecycleTrace(
        submitted_targets=np.asarray([[0.1]], dtype=np.float64),
        execution_intent_targets=np.asarray([[0.1]], dtype=np.float64),
        final_risk_targets=np.asarray([[0.1]], dtype=np.float64),
        applied_risk_scales=np.asarray([1.0], dtype=np.float64),
        hard_risk_evidence_available=np.asarray([True], dtype=np.bool_),
        hard_risk_violations=np.asarray([False], dtype=np.bool_),
        risk_reasons=((),),
        transition_classes=("entry",),
        flatten_initiators=("not_applicable",),
    )


def test_v8_replay_round_trip_revalidates_every_nested_digest() -> None:
    metric = _metric()

    restored = CausalAlphaV8ReplayMetric.from_payload(metric.to_payload())

    assert restored == metric
    assert restored.digest == metric.digest
    assert restored.lifecycle_trace is None
    assert restored.as_v7_metric().v7_config_digest == metric.v8_config_digest


def test_v8_replay_round_trip_preserves_optional_lifecycle_trace() -> None:
    trace = _lifecycle_trace()
    metric = replace(_metric(), lifecycle_trace=trace, digest="")

    restored = CausalAlphaV8ReplayMetric.from_payload(metric.to_payload())

    assert restored.lifecycle_trace is not None
    assert restored.lifecycle_trace.digest == trace.digest
    np.testing.assert_array_equal(
        restored.lifecycle_trace.execution_intent_targets,
        trace.execution_intent_targets,
    )


def test_v8_selection_preserves_candidate_names_and_unchanged_gates() -> None:
    template = _metric()
    symbols = tuple(f"S{index}" for index in range(9))
    metrics: list[CausalAlphaV8ReplayMetric] = []
    for candidate in CausalAlphaV8Candidate:
        for index, symbol in enumerate(symbols):
            scope = str(index + 10)
            base = replace(
                template.v6_metric,
                symbol=symbol,
                contract_digest=_digest(scope[0]),
                fit_digest=_digest("8"),
                forecast_digest=_digest("9"),
                target_path_digest=_digest("a"),
                digest="",
            )
            target_digest = _digest(
                str(tuple(CausalAlphaV8Candidate).index(candidate) + 3)
            )
            attribution = replace(
                template.attribution,
                candidate=candidate,
                target_path_digest=target_digest,
                digest="",
            )
            metrics.append(
                CausalAlphaV8ReplayMetric(
                    candidate=candidate,
                    v6_metric=base,
                    attribution=attribution,
                    v8_target_path_digest=target_digest,
                    source_forecast_digest=_digest("5"),
                    calibration_fit_digest=_digest("6"),
                    v8_config_digest=_digest("7"),
                )
            )

    evidence = evaluate_causal_alpha_v8_selection(
        tuple(metrics), expected_symbols=symbols
    )

    assert evidence.passed
    assert evidence.selected_candidate is CausalAlphaV8Candidate.V7_CONTROL
    assert tuple(
        item["candidate"] for item in evidence.to_payload()["candidates"]
    ) == tuple(  # type: ignore[index]
        candidate.value for candidate in CausalAlphaV8Candidate
    )
