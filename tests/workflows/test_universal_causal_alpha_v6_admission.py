from __future__ import annotations

import hashlib
import math
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from trade_rl.learning.causal_alpha_v6 import CausalAlphaV6Candidate
from trade_rl.workflows.universal_causal_alpha_v6_admission import (
    evaluate_causal_alpha_v6_admission,
)
from trade_rl.workflows.universal_causal_alpha_v6_replay import (
    CausalAlphaV6ReplayMetric,
)
from trade_rl.workflows.universal_causal_alpha_v6_selection import (
    CausalAlphaV6SelectionEvidence,
)
from trade_rl.workflows.universal_causal_alpha_v6_signal import (
    CausalAlphaV6SignalEvidence,
)

_SYMBOLS = tuple(f"S{index}" for index in range(9))


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _record(
    symbol: str,
    candidate: CausalAlphaV6Candidate,
    *,
    gross: float = 0.02,
    net: float = 0.01,
    hard_risk: bool = False,
    rejections: tuple[tuple[str, int], ...] = (),
) -> CausalAlphaV6ReplayMetric:
    return CausalAlphaV6ReplayMetric(
        run_manifest_digest=_digest("run"),
        v4_context_manifest_digest=_digest("context"),
        config_digest=_digest("config"),
        candidate=candidate,
        symbol=symbol,
        episode_index=0,
        contract_digest=_digest(f"contract:{symbol}"),
        fit_digest=_digest("holdout-fit"),
        forecast_digest=_digest(f"forecast:{symbol}"),
        target_path_digest=_digest(f"target:{candidate.value}:{symbol}"),
        decision_count=1,
        gross_return=gross,
        gross_wealth=math.exp(gross),
        net_return=net,
        net_wealth=math.exp(net),
        reward_total=net,
        reward_scale=1.0,
        turnover_per_day=0.1,
        total_execution_cost=0.001,
        target_change_count=1,
        submitted_change_count=1,
        downstream_no_trade_suppression_count=0,
        executed_change_count=1,
        closed_trade_count=1,
        sign_flip_count=0,
        maximum_drawdown=0.01,
        actionable_coverage=1.0,
        flat_time_fraction=0.0,
        time_weighted_absolute_exposure=0.1,
        completed_holding_durations_hours=(6.0,),
        open_holding_duration_hours=0.0,
        execution_rejection_reason_counts=rejections,
        risk_projection_reason_counts=(),
        target_reason_counts=(("entry", 1),),
        hard_risk_violation=hard_risk,
        has_meaningful_execution=True,
    )


def _records(
    candidate: CausalAlphaV6Candidate,
    **kwargs: object,
) -> tuple[CausalAlphaV6ReplayMetric, ...]:
    return tuple(_record(symbol, candidate, **kwargs) for symbol in _SYMBOLS)


def _upstream(
    *,
    selected: CausalAlphaV6Candidate = CausalAlphaV6Candidate.FAST_ONLY,
    signal_passed: bool = True,
    selection_passed: bool = True,
):
    signal = Mock(spec=CausalAlphaV6SignalEvidence)
    signal.passed = signal_passed
    signal.digest = _digest("signal")
    signal.fast_only = SimpleNamespace(
        metrics=(
            SimpleNamespace(
                run_manifest_digest=_digest("run"),
                config_digest=_digest("config"),
            ),
        )
    )
    selection = Mock(spec=CausalAlphaV6SelectionEvidence)
    selection.passed = selection_passed
    selection.digest = _digest("selection")
    selection.selected_candidate = selected
    selection.selected_config_digest = _digest("config")
    selected_summary = SimpleNamespace(
        run_manifest_digest=_digest("run"),
        v4_context_manifest_digest=_digest("context"),
        config_digest=_digest("config"),
    )
    selection.fast_only = selected_summary
    selection.fast_slow_retention = selected_summary
    return signal, selection


def _evaluate(
    selected_records: tuple[CausalAlphaV6ReplayMetric, ...],
    baseline_records: tuple[CausalAlphaV6ReplayMetric, ...],
    *,
    selected: CausalAlphaV6Candidate,
):
    signal, selection = _upstream(selected=selected)
    return evaluate_causal_alpha_v6_admission(
        selected_records,
        baseline_records,
        signal_evidence=signal,
        selection_evidence=selection,
        fit_knowledge_cutoff=100,
        holdout_start=100,
    )


def test_v6_admission_accepts_paired_fast_only_without_artificial_uplift() -> None:
    baseline = _records(CausalAlphaV6Candidate.FAST_ONLY)
    evidence = _evaluate(
        baseline,
        baseline,
        selected=CausalAlphaV6Candidate.FAST_ONLY,
    )
    assert evidence.passed
    assert evidence.aggregate_net_return == pytest.approx(0.09)
    assert evidence.aggregate_net_wealth == pytest.approx(math.exp(0.09))
    assert evidence.positive_net_symbol_count == 9
    assert evidence.worst_symbol_net_return == 0.01
    assert evidence.paired_holdout_count == 9
    assert not evidence.promotion_eligible


def test_v6_admission_requires_retention_not_underperform_fast_holdout() -> None:
    baseline = _records(CausalAlphaV6Candidate.FAST_ONLY, net=0.01)
    better = _records(CausalAlphaV6Candidate.FAST_SLOW_RETENTION, net=0.011)
    assert _evaluate(
        better,
        baseline,
        selected=CausalAlphaV6Candidate.FAST_SLOW_RETENTION,
    ).passed
    worse = _records(CausalAlphaV6Candidate.FAST_SLOW_RETENTION, net=0.009)
    evidence = _evaluate(
        worse,
        baseline,
        selected=CausalAlphaV6Candidate.FAST_SLOW_RETENTION,
    )
    assert not evidence.passed
    assert "retention_underperformed_fast_only" in evidence.rejection_reasons


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (
            lambda values: tuple(
                replace(item, gross_return=0.0, gross_wealth=1.0, digest="")
                for item in values
            ),
            "aggregate_gross_return",
        ),
        (
            lambda values: tuple(
                replace(
                    item,
                    net_return=0.0,
                    net_wealth=1.0,
                    reward_total=0.0,
                    digest="",
                )
                for item in values
            ),
            "aggregate_net_return",
        ),
        (
            lambda values: tuple(
                replace(
                    item,
                    net_return=(-0.001 if index < 4 else 0.01),
                    net_wealth=math.exp(-0.001 if index < 4 else 0.01),
                    reward_total=(-0.001 if index < 4 else 0.01),
                    digest="",
                )
                for index, item in enumerate(values)
            ),
            "positive_net_symbol_count",
        ),
        (
            lambda values: (
                replace(
                    values[0],
                    net_return=-0.021,
                    net_wealth=math.exp(-0.021),
                    reward_total=-0.021,
                    digest="",
                ),
                *values[1:],
            ),
            "worst_symbol_net_return",
        ),
        (
            lambda values: (
                replace(values[0], hard_risk_violation=True, digest=""),
                *values[1:],
            ),
            "hard_risk_violation",
        ),
        (
            lambda values: (
                replace(
                    values[0],
                    execution_rejection_reason_counts=(("venue_rejected", 1),),
                    digest="",
                ),
                *values[1:],
            ),
            "unexplained_execution_rejection",
        ),
    ],
)
def test_v6_admission_applies_fixed_holdout_gates(mutate, reason: str) -> None:
    baseline = _records(CausalAlphaV6Candidate.FAST_ONLY)
    selected = mutate(baseline)
    evidence = _evaluate(
        selected,
        baseline,
        selected=CausalAlphaV6Candidate.FAST_ONLY,
    )
    assert not evidence.passed
    assert reason in evidence.rejection_reasons


def test_v6_admission_blocks_upstream_cutoff_candidate_and_pairing_bypass() -> None:
    baseline = _records(CausalAlphaV6Candidate.FAST_ONLY)
    signal, selection = _upstream(signal_passed=False)
    with pytest.raises(ValueError, match="Signal"):
        evaluate_causal_alpha_v6_admission(
            baseline,
            baseline,
            signal_evidence=signal,
            selection_evidence=selection,
            fit_knowledge_cutoff=100,
            holdout_start=100,
        )
    signal, selection = _upstream(selection_passed=False)
    with pytest.raises(ValueError, match="Selection"):
        evaluate_causal_alpha_v6_admission(
            baseline,
            baseline,
            signal_evidence=signal,
            selection_evidence=selection,
            fit_knowledge_cutoff=100,
            holdout_start=100,
        )
    signal, selection = _upstream()
    with pytest.raises(ValueError, match="cutoff"):
        evaluate_causal_alpha_v6_admission(
            baseline,
            baseline,
            signal_evidence=signal,
            selection_evidence=selection,
            fit_knowledge_cutoff=99,
            holdout_start=100,
        )
    drifted = (
        replace(baseline[0], forecast_digest=_digest("drift"), digest=""),
        *baseline[1:],
    )
    with pytest.raises(ValueError, match="paired"):
        evaluate_causal_alpha_v6_admission(
            baseline,
            drifted,
            signal_evidence=signal,
            selection_evidence=selection,
            fit_knowledge_cutoff=100,
            holdout_start=100,
        )
