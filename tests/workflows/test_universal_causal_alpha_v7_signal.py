from __future__ import annotations

import hashlib
from dataclasses import replace

from trade_rl.learning.causal_alpha_v7 import CausalAlphaV7Candidate
from trade_rl.workflows.universal_causal_alpha_v7_signal import (
    CausalAlphaV7SignalScopeMetric,
    evaluate_causal_alpha_v7_signal_gate,
)

_SYMBOLS = tuple(f"S{index}" for index in range(9))


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _metric(
    candidate: CausalAlphaV7Candidate,
    symbol: str,
    episode: int,
    *,
    positive_support: int = 100,
    negative_support: int = 100,
    non_flat: int = 10,
) -> CausalAlphaV7SignalScopeMetric:
    scope = f"{symbol}:{episode}"
    return CausalAlphaV7SignalScopeMetric(
        candidate=candidate,
        run_manifest_digest=_digest("run"),
        v7_config_digest=_digest("config"),
        symbol=symbol,
        episode_index=episode,
        contract_start=1_000 + episode * 100,
        contract_stop=1_100 + episode * 100,
        contract_digest=_digest(f"contract:{scope}"),
        source_forecast_digest=_digest(f"source:{scope}"),
        calibration_fit_digest=_digest(f"calibration:{episode}"),
        calibration_return_model_digest=_digest(f"return:{episode}"),
        calibration_direction_model_digest=_digest(f"direction:{episode}"),
        target_path_digest=_digest(f"target:{candidate.value}:{scope}"),
        decision_count=100,
        actionable_count=100,
        non_flat_target_count=non_flat,
        target_change_count=2,
        sign_flip_count=0,
        positive_direction_support=positive_support,
        negative_direction_support=negative_support,
    )


def _metrics(**kwargs: int) -> tuple[CausalAlphaV7SignalScopeMetric, ...]:
    return tuple(
        _metric(candidate, symbol, episode, **kwargs)
        for candidate in CausalAlphaV7Candidate
        for episode in range(8)
        for symbol in _SYMBOLS
    )


def test_v7_signal_requires_three_paired_live_candidates() -> None:
    evidence = evaluate_causal_alpha_v7_signal_gate(
        _metrics(),
        expected_symbols=_SYMBOLS,
        v4_fast_lane_digest=_digest("v4-fast"),
        v4_fast_lane_passed=True,
    )

    assert evidence.passed
    assert evidence.raw_scope_count_per_candidate == 72
    assert evidence.independent_episode_count == 8
    assert evidence.symbol_count == 9
    assert tuple(item.candidate for item in evidence.candidates) == tuple(
        CausalAlphaV7Candidate
    )
    assert all(item.passed for item in evidence.candidates)


def test_v7_signal_fails_closed_on_direction_support_or_liveness() -> None:
    no_negative = evaluate_causal_alpha_v7_signal_gate(
        _metrics(negative_support=0),
        expected_symbols=_SYMBOLS,
        v4_fast_lane_digest=_digest("v4-fast"),
        v4_fast_lane_passed=True,
    )
    assert not no_negative.passed
    assert all(
        "negative_direction_support" in item.rejection_reasons
        for item in no_negative.candidates
    )

    no_targets = evaluate_causal_alpha_v7_signal_gate(
        _metrics(non_flat=0),
        expected_symbols=_SYMBOLS,
        v4_fast_lane_digest=_digest("v4-fast"),
        v4_fast_lane_passed=True,
    )
    assert not no_targets.passed
    assert all("non_flat_target" in item.rejection_reasons for item in no_targets.candidates)


def test_v7_signal_rejects_pairing_drift_and_failed_v4_lane() -> None:
    metrics = list(_metrics())
    metrics[-1] = replace(metrics[-1], source_forecast_digest=_digest("drift"), digest="")
    pairing = evaluate_causal_alpha_v7_signal_gate(
        tuple(metrics),
        expected_symbols=_SYMBOLS,
        v4_fast_lane_digest=_digest("v4-fast"),
        v4_fast_lane_passed=True,
    )
    assert pairing.rejection_reasons == ("scope_pairing",)

    v4_failed = evaluate_causal_alpha_v7_signal_gate(
        _metrics(),
        expected_symbols=_SYMBOLS,
        v4_fast_lane_digest=_digest("v4-fast"),
        v4_fast_lane_passed=False,
    )
    assert v4_failed.rejection_reasons == ("v4_fast_lane",)


def test_v7_signal_fails_closed_when_candidate_is_missing() -> None:
    metrics = tuple(
        metric
        for metric in _metrics()
        if metric.candidate is not CausalAlphaV7Candidate.CAUSAL_CALIBRATED
    )

    evidence = evaluate_causal_alpha_v7_signal_gate(
        metrics,
        expected_symbols=_SYMBOLS,
        v4_fast_lane_digest=_digest("v4-fast"),
        v4_fast_lane_passed=True,
    )

    missing = evidence.candidates[-1]
    assert not evidence.passed
    assert missing.rejection_reasons == (
        "raw_scope_count",
        "independent_episode_count",
        "symbol_coverage",
        "non_flat_target",
        "positive_direction_support",
        "negative_direction_support",
    )


def test_v7_signal_requires_exact_expected_symbol_set() -> None:
    metrics = tuple(
        replace(metric, symbol="OTHER", digest="")
        if metric.symbol == _SYMBOLS[-1]
        else metric
        for metric in _metrics()
    )

    evidence = evaluate_causal_alpha_v7_signal_gate(
        metrics,
        expected_symbols=_SYMBOLS,
        v4_fast_lane_digest=_digest("v4-fast"),
        v4_fast_lane_passed=True,
    )

    assert not evidence.passed
    assert all(
        "symbol_coverage" in candidate.rejection_reasons
        for candidate in evidence.candidates
    )
