from __future__ import annotations

from dataclasses import replace

from trade_rl.workflows.universal_causal_alpha_v4_replay import CausalAlphaV4ReplayMetric
from trade_rl.workflows.universal_causal_alpha_v4_selection import (
    evaluate_causal_alpha_v4_selection,
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
        turnover_per_day=0.05,
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


def _passing_metrics() -> tuple[CausalAlphaV4ReplayMetric, ...]:
    return (
        _metric(symbol="BTCUSDT", episode=0),
        _metric(symbol="ETHUSDT", episode=0),
        _metric(symbol="BTCUSDT", episode=1, gross=0.01, net=0.008),
        _metric(symbol="ETHUSDT", episode=1, gross=0.005, net=0.004),
    )


def test_selection_passes_with_meaningful_execution_and_zero_closed_trades() -> None:
    evidence = evaluate_causal_alpha_v4_selection(_passing_metrics())

    assert evidence.passed is True
    assert evidence.rejection_reasons == ()
    assert evidence.meaningful_execution_scope_count == 4
    assert evidence.total_executed_change_count == 4
    assert evidence.total_closed_trade_count == 0
    assert evidence.mean_gross_return >= 0.0
    assert evidence.mean_net_return >= 0.0
    assert evidence.worst_symbol_episode_net_return >= -0.05
    assert evidence.positive_gross_episode_fraction == 1.0


def test_selection_requires_non_negative_mean_gross_and_net() -> None:
    gross_bad = list(_passing_metrics())
    gross_bad[0] = replace(gross_bad[0], gross_return=-0.20, digest="")
    net_bad = list(_passing_metrics())
    net_bad[0] = replace(net_bad[0], net_return=-0.20, digest="")

    gross = evaluate_causal_alpha_v4_selection(tuple(gross_bad))
    net = evaluate_causal_alpha_v4_selection(tuple(net_bad))

    assert "mean_gross_return_below_minimum" in gross.rejection_reasons
    assert "mean_net_return_below_minimum" in net.rejection_reasons
    assert gross.passed is False
    assert net.passed is False


def test_selection_requires_worst_net_and_positive_gross_fraction() -> None:
    values = list(_passing_metrics())
    values[0] = replace(values[0], gross_return=-0.01, net_return=-0.06, digest="")
    values[1] = replace(values[1], gross_return=-0.01, digest="")
    evidence = evaluate_causal_alpha_v4_selection(tuple(values))

    assert "worst_symbol_episode_net_return_below_floor" in evidence.rejection_reasons
    assert "positive_gross_episode_fraction_below_minimum" in evidence.rejection_reasons
    assert evidence.passed is False


def test_selection_rejects_hard_risk_and_unexplained_execution_rejection() -> None:
    values = list(_passing_metrics())
    values[0] = replace(values[0], hard_risk_violation=True, digest="")
    values[1] = replace(
        values[1],
        execution_rejection_reason_counts=(("venue_rejected", 1),),
        digest="",
    )
    evidence = evaluate_causal_alpha_v4_selection(tuple(values))

    assert evidence.hard_risk_violation_count == 1
    assert evidence.unexplained_execution_rejection_count == 1
    assert "hard_risk_violation" in evidence.rejection_reasons
    assert "unexplained_execution_rejection" in evidence.rejection_reasons


def test_selection_allows_explained_no_fill_reasons() -> None:
    values = list(_passing_metrics())
    values[0] = replace(
        values[0],
        execution_rejection_reason_counts=(("below_minimum_notional", 1),),
        digest="",
    )
    evidence = evaluate_causal_alpha_v4_selection(tuple(values))

    assert evidence.unexplained_execution_rejection_count == 0
    assert evidence.passed is True


def test_selection_requires_at_least_one_meaningful_execution_scope() -> None:
    values = tuple(
        replace(
            metric,
            submitted_change_count=0,
            executed_change_count=0,
            has_meaningful_execution=False,
            digest="",
        )
        for metric in _passing_metrics()
    )
    evidence = evaluate_causal_alpha_v4_selection(values)

    assert evidence.meaningful_execution_scope_count == 0
    assert evidence.total_closed_trade_count == 0
    assert "no_meaningful_execution" in evidence.rejection_reasons
    assert evidence.passed is False


def test_selection_fails_closed_on_duplicate_scope_or_identity_drift() -> None:
    values = _passing_metrics()
    duplicate = values + (values[0],)
    drifted = values[:-1] + (
        replace(values[-1], config_digest=_digest("9"), digest=""),
    )

    for invalid in (duplicate, drifted):
        try:
            evaluate_causal_alpha_v4_selection(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("V4 selection accepted invalid replay identity")
