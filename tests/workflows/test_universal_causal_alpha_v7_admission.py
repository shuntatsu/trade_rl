from __future__ import annotations

from dataclasses import replace

from tests.workflows.test_universal_causal_alpha_v7_selection import (
    _SYMBOLS,
    _digest,
    _records,
)
from tests.workflows.test_universal_causal_alpha_v7_signal import _metrics
from trade_rl.learning.causal_alpha_v7 import CausalAlphaV7Candidate
from trade_rl.workflows.universal_causal_alpha_v7_admission import (
    evaluate_causal_alpha_v7_admission,
)
from trade_rl.workflows.universal_causal_alpha_v7_selection import (
    CausalAlphaV7SelectionEvidence,
    evaluate_causal_alpha_v7_selection,
)
from trade_rl.workflows.universal_causal_alpha_v7_signal import (
    CausalAlphaV7SignalEvidence,
    evaluate_causal_alpha_v7_signal_gate,
)


def _upstream() -> tuple[CausalAlphaV7SignalEvidence, CausalAlphaV7SelectionEvidence]:
    signal_metrics = tuple(
        replace(metric, v7_config_digest=_digest("v7-config"), digest="")
        for metric in _metrics()
    )
    signal = evaluate_causal_alpha_v7_signal_gate(
        signal_metrics,
        expected_symbols=_SYMBOLS,
        v4_fast_lane_digest=_digest("v4-fast"),
        v4_fast_lane_passed=True,
    )
    selection = evaluate_causal_alpha_v7_selection(
        (
            *_records(CausalAlphaV7Candidate.V6_CONTROL, net=0.010),
            *_records(CausalAlphaV7Candidate.SYMMETRIC_CONTRARIAN, net=0.012),
            *_records(CausalAlphaV7Candidate.CAUSAL_CALIBRATED, net=0.015),
        ),
        expected_symbols=_SYMBOLS,
    )
    return signal, selection


def test_v7_admission_accepts_profitable_selected_holdout_above_control() -> None:
    signal, selection = _upstream()
    selected = _records(CausalAlphaV7Candidate.CAUSAL_CALIBRATED, net=0.012)[:9]
    control = _records(CausalAlphaV7Candidate.V6_CONTROL, net=0.010)[:9]

    evidence = evaluate_causal_alpha_v7_admission(
        selected,
        control,
        signal_evidence=signal,
        selection_evidence=selection,
        fit_knowledge_cutoff=10_000,
        holdout_start=10_000,
    )

    assert evidence.passed
    assert evidence.selected_candidate is CausalAlphaV7Candidate.CAUSAL_CALIBRATED
    assert evidence.positive_net_symbol_count == 9


def test_v7_admission_rejects_selected_holdout_below_control() -> None:
    signal, selection = _upstream()
    selected = _records(CausalAlphaV7Candidate.CAUSAL_CALIBRATED, net=0.009)[:9]
    control = _records(CausalAlphaV7Candidate.V6_CONTROL, net=0.010)[:9]

    evidence = evaluate_causal_alpha_v7_admission(
        selected,
        control,
        signal_evidence=signal,
        selection_evidence=selection,
        fit_knowledge_cutoff=10_000,
        holdout_start=10_000,
    )

    assert not evidence.passed
    assert "selected_underperformed_control" in evidence.rejection_reasons
