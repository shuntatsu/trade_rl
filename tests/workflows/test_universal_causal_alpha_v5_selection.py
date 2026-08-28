from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from trade_rl.workflows.universal_causal_alpha_v5_replay import (
    CausalAlphaV5ReplayMetric,
)
from trade_rl.workflows.universal_causal_alpha_v5_selection import (
    evaluate_causal_alpha_v5_selection,
)

_SYMBOLS = tuple(f"S{index}" for index in range(9))


def _digest(char: str) -> str:
    return char * 64


def _metric(
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
) -> CausalAlphaV5ReplayMetric:
    return CausalAlphaV5ReplayMetric(
        run_manifest_digest=_digest("1"),
        v4_context_manifest_digest=_digest("2"),
        config_digest=_digest("3"),
        symbol=symbol,
        episode_index=episode,
        contract_digest=_digest("4"),
        fit_digest=_digest("5"),
        forecast_digest=_digest("6"),
        calibration_fit_digest=_digest("7"),
        target_path_digest=_digest("8"),
        gross_return=gross,
        net_return=net,
        turnover_per_day=turnover,
        total_execution_cost=cost,
        submitted_change_count=int(meaningful),
        downstream_no_trade_suppression_count=0,
        executed_change_count=int(meaningful),
        closed_trade_count=0,
        sign_flip_count=0,
        maximum_drawdown=0.01,
        active_coverage=0.5,
        flat_time_fraction=0.5,
        time_weighted_absolute_exposure=0.1,
        completed_holding_durations_hours=(),
        has_unclosed_position=False,
        execution_rejection_reason_counts=rejections,
        risk_projection_reason_counts=(),
        target_reason_counts=(("hold_flat", 1),),
        hard_risk_violation=hard_risk,
        has_meaningful_execution=meaningful,
    )


def _passing() -> tuple[CausalAlphaV5ReplayMetric, ...]:
    return tuple(
        _metric(
            symbol, episode, net=0.01 + episode * 0.001, turnover=0.01 * (episode + 1)
        )
        for episode in range(2)
        for symbol in _SYMBOLS
    )


def test_v5_selection_computes_hand_checkable_wealth_and_distributions() -> None:
    evidence = evaluate_causal_alpha_v5_selection(_passing(), expected_symbols=_SYMBOLS)
    expected_symbol_wealth = math.exp(0.021)
    assert evidence.passed
    assert np.isclose(evidence.symbol_balanced_net_wealth, expected_symbol_wealth)
    assert np.isclose(evidence.median_symbol_net_wealth, expected_symbol_wealth)
    assert evidence.positive_net_scope_fraction == 1.0
    assert evidence.worst_symbol_episode_net_return == 0.01
    assert evidence.scope_net_return_cvar_10 == 0.01
    assert np.isclose(evidence.turnover_p50, 0.015)
    assert np.isclose(evidence.turnover_p95, 0.02)
    assert np.isclose(evidence.total_execution_cost, 0.018)
    assert np.isclose(evidence.net_to_gross_retention, math.exp(0.021 - 0.04))
    assert tuple(summary.symbol for summary in evidence.symbol_summaries) == _SYMBOLS


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (
            lambda values: tuple(
                replace(item, net_return=-0.02, digest="") for item in values
            ),
            "symbol_balanced_net_wealth",
        ),
        (
            lambda values: tuple(
                replace(item, net_return=-0.01 if index < 10 else 0.01, digest="")
                for index, item in enumerate(values)
            ),
            "positive_net_scope_fraction",
        ),
        (
            lambda values: tuple(item for item in values if item.symbol != "S8"),
            "symbol_coverage",
        ),
        (
            lambda values: tuple(
                replace(
                    item,
                    submitted_change_count=0,
                    executed_change_count=0,
                    has_meaningful_execution=False,
                    digest="",
                )
                for item in values
            ),
            "no_meaningful_execution",
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
def test_v5_selection_gates_balanced_wealth_coverage_execution_and_risk(
    mutate, reason: str
) -> None:
    evidence = evaluate_causal_alpha_v5_selection(
        mutate(_passing()), expected_symbols=_SYMBOLS
    )
    assert not evidence.passed
    assert reason in evidence.rejection_reasons


def test_v5_selection_rejects_duplicate_scope_identity() -> None:
    values = _passing()
    with pytest.raises(ValueError, match="duplicated"):
        evaluate_causal_alpha_v5_selection(
            (*values, values[0]), expected_symbols=_SYMBOLS
        )


def test_v5_selection_requires_median_wealth_and_finite_exponentiation() -> None:
    values = tuple(
        replace(
            item,
            net_return=-0.01 if item.symbol in _SYMBOLS[:5] else 0.02,
            digest="",
        )
        for item in _passing()
    )
    evidence = evaluate_causal_alpha_v5_selection(values, expected_symbols=_SYMBOLS)
    assert "median_symbol_net_wealth" in evidence.rejection_reasons
    overflow = (
        replace(_passing()[0], net_return=1000.0, digest=""),
        *_passing()[1:],
    )
    with pytest.raises(ValueError, match="overflow"):
        evaluate_causal_alpha_v5_selection(overflow, expected_symbols=_SYMBOLS)
