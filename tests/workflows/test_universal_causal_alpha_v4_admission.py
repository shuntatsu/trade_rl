from __future__ import annotations

from dataclasses import replace

from trade_rl.workflows.universal_causal_alpha_v4_admission import (
    evaluate_causal_alpha_v4_admission,
)
from trade_rl.workflows.universal_causal_alpha_v4_replay import CausalAlphaV4ReplayMetric
from trade_rl.workflows.universal_causal_alpha_v4_selection import (
    evaluate_causal_alpha_v4_selection,
)
from trade_rl.workflows.universal_causal_alpha_v4_signal import (
    CausalAlphaV4LaneSignalEvidence,
    CausalAlphaV4SignalBootstrapEvidence,
    CausalAlphaV4SignalEvidence,
    CausalAlphaV4SignalLane,
    CausalAlphaV4SignalScopeMetric,
)


def _digest(char: str) -> str:
    return char * 64


def _metric(
    *,
    symbol: str,
    episode: int,
    gross: float = 0.02,
    net: float = 0.015,
    meaningful: bool = True,
    executed: int = 1,
    closed: int = 0,
    hard_risk: bool = False,
    rejection_counts: tuple[tuple[str, int], ...] = (),
) -> CausalAlphaV4ReplayMetric:
    return CausalAlphaV4ReplayMetric(
        run_manifest_digest=_digest("a"),
        v4_context_manifest_digest=_digest("b"),
        config_digest=_digest("c"),
        symbol=symbol,
        episode_index=episode,
        contract_digest=_digest(str((episode % 8) + 1)),
        fit_digest=_digest("d"),
        forecast_digest=_digest("e"),
        target_path_digest=_digest("f"),
        gross_return=gross,
        net_return=net,
        turnover_per_day=0.03,
        total_execution_cost=0.001,
        submitted_change_count=1 if meaningful else 0,
        downstream_no_trade_suppression_count=0,
        executed_change_count=executed,
        closed_trade_count=closed,
        sign_flip_count=0,
        maximum_drawdown=0.02,
        execution_rejection_reason_counts=rejection_counts,
        risk_projection_reason_counts=(),
        target_reason_counts=(("hold", 1),),
        hard_risk_violation=hard_risk,
        has_meaningful_execution=meaningful,
    )


def _selection_metrics() -> tuple[CausalAlphaV4ReplayMetric, ...]:
    return (
        _metric(symbol="BTCUSDT", episode=0),
        _metric(symbol="ETHUSDT", episode=0),
        _metric(symbol="BTCUSDT", episode=1),
        _metric(symbol="ETHUSDT", episode=1),
    )


def _bootstrap() -> CausalAlphaV4SignalBootstrapEvidence:
    return CausalAlphaV4SignalBootstrapEvidence(
        mean=0.1,
        p_value=0.0,
        lower_ci=0.05,
        upper_ci=0.15,
        block_size=2,
    )


def _scope(lane: CausalAlphaV4SignalLane) -> CausalAlphaV4SignalScopeMetric:
    return CausalAlphaV4SignalScopeMetric(
        run_manifest_digest=_digest("a"),
        fit_config_digest=_digest("2"),
        lane=lane,
        symbol="BTCUSDT",
        episode_index=0,
        contract_start=0,
        contract_stop=10,
        contract_digest=_digest("3"),
        fit_digest=_digest("d"),
        forecast_digest=_digest("e"),
        liveness_digest=_digest("4"),
        sample_count=2,
        direction_sample_count=2,
        rank_correlation=1.0,
        direction_accuracy=1.0,
        top_bottom_realized_spread=0.01,
        cohort_indices=(1, 2),
    )


def _lane(lane: CausalAlphaV4SignalLane) -> CausalAlphaV4LaneSignalEvidence:
    return CausalAlphaV4LaneSignalEvidence(
        lane=lane,
        metrics=(_scope(lane),),
        run_manifest_digest=_digest("a"),
        raw_scope_count=1,
        expected_raw_scope_count=1,
        independent_episode_count=1,
        rank_ic=_bootstrap(),
        top_bottom_spread=_bootstrap(),
        direction_accuracy_excess=_bootstrap(),
        gate_digest=_digest("9"),
        passed=True,
        rejection_reasons=(),
    )


def _signal() -> CausalAlphaV4SignalEvidence:
    return CausalAlphaV4SignalEvidence(
        fast_4h=_lane(CausalAlphaV4SignalLane.FAST_4H),
        slow_fused=_lane(CausalAlphaV4SignalLane.SLOW_FUSED),
        gate_digest=_digest("9"),
        passed=True,
        rejection_reasons=(),
    )


def _holdout_records() -> tuple[CausalAlphaV4ReplayMetric, ...]:
    return (
        _metric(symbol="BTCUSDT", episode=8, gross=0.02, net=0.015, closed=0),
        _metric(symbol="ETHUSDT", episode=8, gross=0.01, net=0.008, closed=0),
        _metric(symbol="BCHUSDT", episode=8, gross=-0.005, net=-0.004, closed=0),
    )


def test_admission_requires_passed_signal_selection_and_exact_holdout_cutoff() -> None:
    selection = evaluate_causal_alpha_v4_selection(_selection_metrics())
    evidence = evaluate_causal_alpha_v4_admission(
        _holdout_records(),
        signal_evidence=_signal(),
        selection_evidence=selection,
        fit_knowledge_cutoff=500,
        holdout_start=500,
    )

    assert evidence.passed is True
    assert evidence.fit_knowledge_cutoff == 500
    assert evidence.holdout_start == 500
    assert evidence.total_executed_change_count == 3
    assert evidence.total_closed_trade_count == 0
    assert evidence.meaningful_execution_scope_count == 3
    assert evidence.promotion_eligible is False


def test_admission_rejects_cutoff_drift_before_interpreting_holdout() -> None:
    selection = evaluate_causal_alpha_v4_selection(_selection_metrics())
    try:
        evaluate_causal_alpha_v4_admission(
            _holdout_records(),
            signal_evidence=_signal(),
            selection_evidence=selection,
            fit_knowledge_cutoff=499,
            holdout_start=500,
        )
    except ValueError as error:
        assert "cutoff" in str(error)
    else:
        raise AssertionError("V4 admission accepted a fit that crossed holdout start")


def test_admission_does_not_open_after_failed_signal_or_selection() -> None:
    signal = _signal()
    failed_signal = replace(
        signal,
        passed=False,
        rejection_reasons=("slow_fused:direction_accuracy_excess_lower_ci",),
        digest="",
    )
    selection = evaluate_causal_alpha_v4_selection(_selection_metrics())
    failed_selection_metrics = tuple(
        replace(
            metric,
            submitted_change_count=0,
            executed_change_count=0,
            has_meaningful_execution=False,
            digest="",
        )
        for metric in _selection_metrics()
    )
    failed_selection = evaluate_causal_alpha_v4_selection(failed_selection_metrics)

    for signal_evidence, selection_evidence in (
        (failed_signal, selection),
        (signal, failed_selection),
    ):
        try:
            evaluate_causal_alpha_v4_admission(
                _holdout_records(),
                signal_evidence=signal_evidence,
                selection_evidence=selection_evidence,
                fit_knowledge_cutoff=500,
                holdout_start=500,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("V4 admission bypassed a required upstream gate")


def test_admission_requires_non_negative_aggregate_economics_and_tail_floor() -> None:
    selection = evaluate_causal_alpha_v4_selection(_selection_metrics())
    values = list(_holdout_records())
    values[0] = replace(values[0], gross_return=-0.20, net_return=-0.20, digest="")
    evidence = evaluate_causal_alpha_v4_admission(
        tuple(values),
        signal_evidence=_signal(),
        selection_evidence=selection,
        fit_knowledge_cutoff=500,
        holdout_start=500,
    )

    assert "negative_aggregate_gross_return" in evidence.rejection_reasons
    assert "negative_aggregate_net_return" in evidence.rejection_reasons
    assert "symbol_net_return_below_floor" in evidence.rejection_reasons
    assert evidence.passed is False


def test_admission_rejects_majority_negative_gross_holdouts() -> None:
    selection = evaluate_causal_alpha_v4_selection(_selection_metrics())
    values = list(_holdout_records())
    values[0] = replace(values[0], gross_return=-0.001, net_return=0.02, digest="")
    values[1] = replace(values[1], gross_return=-0.001, net_return=0.02, digest="")
    evidence = evaluate_causal_alpha_v4_admission(
        tuple(values),
        signal_evidence=_signal(),
        selection_evidence=selection,
        fit_knowledge_cutoff=500,
        holdout_start=500,
    )

    assert evidence.negative_gross_symbol_count == 3
    assert "majority_negative_gross_holdouts" in evidence.rejection_reasons


def test_admission_rejects_risk_unexplained_rejection_and_no_execution() -> None:
    selection = evaluate_causal_alpha_v4_selection(_selection_metrics())
    values = list(_holdout_records())
    values[0] = replace(values[0], hard_risk_violation=True, digest="")
    values[1] = replace(
        values[1],
        execution_rejection_reason_counts=(("venue_rejected", 1),),
        digest="",
    )
    values = [
        replace(
            metric,
            submitted_change_count=0,
            executed_change_count=0,
            has_meaningful_execution=False,
            digest="",
        )
        for metric in values
    ]
    evidence = evaluate_causal_alpha_v4_admission(
        tuple(values),
        signal_evidence=_signal(),
        selection_evidence=selection,
        fit_knowledge_cutoff=500,
        holdout_start=500,
    )

    assert evidence.hard_risk_violation_count == 1
    assert evidence.unexplained_execution_rejection_count == 1
    assert evidence.meaningful_execution_scope_count == 0
    assert "hard_risk_violation" in evidence.rejection_reasons
    assert "unexplained_execution_rejection" in evidence.rejection_reasons
    assert "no_meaningful_execution" in evidence.rejection_reasons


def test_admission_requires_unique_symbol_holdout_records_and_identity_closure() -> None:
    selection = evaluate_causal_alpha_v4_selection(_selection_metrics())
    records = _holdout_records()
    duplicate = records + (records[0],)
    drifted = records[:-1] + (
        replace(records[-1], fit_digest=_digest("5"), digest=""),
    )

    for invalid in (duplicate, drifted):
        try:
            evaluate_causal_alpha_v4_admission(
                invalid,
                signal_evidence=_signal(),
                selection_evidence=selection,
                fit_knowledge_cutoff=500,
                holdout_start=500,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("V4 admission accepted invalid holdout identity")
