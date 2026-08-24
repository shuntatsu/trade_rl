from __future__ import annotations

import hashlib
from dataclasses import replace

from trade_rl.learning.causal_alpha_v6 import CausalAlphaV6Candidate
from trade_rl.workflows.universal_causal_alpha_v4_signal import (
    CausalAlphaV4LaneSignalEvidence,
    CausalAlphaV4SignalBootstrapEvidence,
    CausalAlphaV4SignalLane,
    CausalAlphaV4SignalScopeMetric,
)
from trade_rl.workflows.universal_causal_alpha_v6_signal import (
    CausalAlphaV6SignalScopeMetric,
    evaluate_causal_alpha_v6_signal_gate,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _v4_fast(*, passed: bool = True) -> CausalAlphaV4LaneSignalEvidence:
    metric = CausalAlphaV4SignalScopeMetric(
        run_manifest_digest=_digest("run"),
        fit_config_digest=_digest("v4-config"),
        lane=CausalAlphaV4SignalLane.FAST_4H,
        symbol="S0",
        episode_index=0,
        contract_start=0,
        contract_stop=10,
        contract_digest=_digest("contract"),
        fit_digest=_digest("fit"),
        forecast_digest=_digest("forecast"),
        liveness_digest=_digest("liveness"),
        sample_count=2,
        direction_sample_count=2,
        rank_correlation=0.5,
        direction_accuracy=0.75,
        top_bottom_realized_spread=0.01,
        cohort_indices=(0, 1),
    )
    bootstrap = CausalAlphaV4SignalBootstrapEvidence(
        mean=0.1,
        p_value=0.01,
        lower_ci=0.01,
        upper_ci=0.2,
        block_size=2,
    )
    return CausalAlphaV4LaneSignalEvidence(
        lane=CausalAlphaV4SignalLane.FAST_4H,
        metrics=(metric,),
        run_manifest_digest=metric.run_manifest_digest,
        raw_scope_count=1,
        expected_raw_scope_count=1,
        independent_episode_count=1,
        rank_ic=bootstrap,
        top_bottom_spread=bootstrap,
        direction_accuracy_excess=bootstrap,
        gate_digest=_digest("v4-gate"),
        passed=passed,
        rejection_reasons=() if passed else ("rank_ic_lower_ci",),
    )


def _metrics() -> tuple[CausalAlphaV6SignalScopeMetric, ...]:
    values: list[CausalAlphaV6SignalScopeMetric] = []
    for candidate in CausalAlphaV6Candidate:
        for episode in range(8):
            for symbol_index in range(9):
                symbol = f"S{symbol_index}"
                scope = f"{symbol}:{episode}"
                values.append(
                    CausalAlphaV6SignalScopeMetric(
                        candidate=candidate,
                        run_manifest_digest=_digest("run"),
                        config_digest=_digest("v6-config"),
                        symbol=symbol,
                        episode_index=episode,
                        contract_start=episode * 100,
                        contract_stop=episode * 100 + 99,
                        contract_digest=_digest(f"contract:{scope}"),
                        fit_digest=_digest(f"fit:{episode}"),
                        forecast_digest=_digest(f"forecast:{scope}"),
                        target_digest=_digest(f"target:{candidate.value}:{scope}"),
                        initial_weight=0.0,
                        decision_count=4,
                        actionable_count=4,
                        non_flat_target_count=2,
                        target_change_count=1,
                        sign_flip_count=0,
                        reason_counts=(("entry", 1), ("hold_position", 3)),
                        slow_direction_sample_count=4,
                        slow_direction_accuracy=0.75,
                    )
                )
    return tuple(values)


def _evaluate(metrics: tuple[CausalAlphaV6SignalScopeMetric, ...], *, fast_pass: bool = True):
    return evaluate_causal_alpha_v6_signal_gate(
        metrics,
        expected_symbols=tuple(f"S{index}" for index in range(9)),
        v4_fast_lane=_v4_fast(passed=fast_pass),
    )


def test_v6_signal_requires_exact_paired_72_scope_candidate_paths() -> None:
    evidence = _evaluate(_metrics())
    assert evidence.passed
    assert evidence.raw_scope_count_per_candidate == 72
    assert evidence.independent_episode_count == 8
    assert evidence.symbol_count == 9
    assert evidence.v4_fast_lane_passed
    assert evidence.fast_only.non_flat_target_count > 0
    assert evidence.fast_slow_retention.non_flat_target_count > 0


def test_v6_signal_rejects_missing_scope_and_pairing() -> None:
    evidence = _evaluate(_metrics()[:-1])
    assert not evidence.passed
    assert "raw_scope_count" in evidence.rejection_reasons
    assert "scope_pairing" in evidence.rejection_reasons


def test_v6_signal_rejects_duplicate_identity() -> None:
    metrics = _metrics()
    evidence = _evaluate((*metrics, metrics[0]))
    assert "duplicate_scope_identity" in evidence.rejection_reasons


def test_v6_signal_rejects_pair_identity_drift() -> None:
    metrics = list(_metrics())
    retention_index = 72
    metrics[retention_index] = replace(
        metrics[retention_index], initial_weight=0.05, digest=""
    )
    evidence = _evaluate(tuple(metrics))
    assert "scope_pairing" in evidence.rejection_reasons


def test_v6_signal_rejects_all_flat_fast_baseline() -> None:
    metrics = tuple(
        replace(metric, non_flat_target_count=0, digest="")
        if metric.candidate is CausalAlphaV6Candidate.FAST_ONLY
        else metric
        for metric in _metrics()
    )
    evidence = _evaluate(metrics)
    assert "fast_only_non_flat_target" in evidence.rejection_reasons


def test_v6_signal_is_bound_to_unchanged_v4_fast_lane() -> None:
    evidence = _evaluate(_metrics(), fast_pass=False)
    assert not evidence.passed
    assert evidence.rejection_reasons == ("v4_fast_4h",)


def test_v6_signal_rejects_config_drift_without_tuning() -> None:
    metrics = list(_metrics())
    metrics[0] = replace(metrics[0], config_digest=_digest("drift"), digest="")
    evidence = _evaluate(tuple(metrics))
    assert "config_identity" in evidence.rejection_reasons
