from __future__ import annotations

import hashlib
import math
from dataclasses import replace

import pytest

from trade_rl.learning.causal_alpha_v6 import CausalAlphaV6Candidate
from trade_rl.workflows.universal_causal_alpha_v6_replay import (
    CausalAlphaV6ReplayMetric,
)
from trade_rl.workflows.universal_causal_alpha_v6_selection import (
    evaluate_causal_alpha_v6_selection,
)

_SYMBOLS = tuple(f"S{index}" for index in range(9))


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _metric(
    candidate: CausalAlphaV6Candidate,
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
    flips: int = 0,
) -> CausalAlphaV6ReplayMetric:
    scope = f"{symbol}:{episode}"
    return CausalAlphaV6ReplayMetric(
        run_manifest_digest=_digest("run"),
        v4_context_manifest_digest=_digest("context"),
        config_digest=_digest("config"),
        candidate=candidate,
        symbol=symbol,
        episode_index=episode,
        contract_digest=_digest(f"contract:{scope}"),
        fit_digest=_digest(f"fit:{episode}"),
        forecast_digest=_digest(f"forecast:{scope}"),
        target_path_digest=_digest(f"target:{candidate.value}:{scope}"),
        decision_count=1,
        gross_return=gross,
        gross_wealth=math.exp(gross),
        net_return=net,
        net_wealth=math.exp(net),
        reward_total=net,
        reward_scale=1.0,
        turnover_per_day=turnover,
        total_execution_cost=cost,
        target_change_count=int(meaningful),
        submitted_change_count=int(meaningful),
        downstream_no_trade_suppression_count=0,
        executed_change_count=int(meaningful),
        closed_trade_count=int(meaningful),
        sign_flip_count=flips,
        maximum_drawdown=0.01,
        actionable_coverage=1.0,
        flat_time_fraction=0.0,
        time_weighted_absolute_exposure=0.1,
        completed_holding_durations_hours=(),
        open_holding_duration_hours=6.0,
        execution_rejection_reason_counts=rejections,
        risk_projection_reason_counts=(),
        target_reason_counts=(("hold_position", 1),),
        hard_risk_violation=hard_risk,
        has_meaningful_execution=meaningful,
    )


def _records(
    candidate: CausalAlphaV6Candidate,
    **kwargs: object,
) -> tuple[CausalAlphaV6ReplayMetric, ...]:
    return tuple(
        _metric(candidate, symbol, episode, **kwargs)
        for episode in range(2)
        for symbol in _SYMBOLS
    )


def _evaluate(
    fast: tuple[CausalAlphaV6ReplayMetric, ...],
    retention: tuple[CausalAlphaV6ReplayMetric, ...],
):
    return evaluate_causal_alpha_v6_selection(
        (*fast, *retention), expected_symbols=_SYMBOLS
    )


def test_v6_selection_computes_paired_balanced_symbol_economics() -> None:
    evidence = _evaluate(
        _records(CausalAlphaV6Candidate.FAST_ONLY),
        _records(CausalAlphaV6Candidate.FAST_SLOW_RETENTION),
    )
    assert evidence.passed
    assert evidence.selected_candidate is CausalAlphaV6Candidate.FAST_ONLY
    assert evidence.paired_scope_count == 18
    assert evidence.fast_only.symbol_balanced_gross_wealth == pytest.approx(
        math.exp(0.04)
    )
    assert evidence.fast_only.symbol_balanced_net_wealth == pytest.approx(
        math.exp(0.02)
    )
    assert evidence.fast_only.minimum_symbol_net_wealth == pytest.approx(
        math.exp(0.02)
    )
    assert evidence.fast_only.positive_net_scope_fraction == 1.0


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (
            lambda values: tuple(
                replace(item, gross_return=0.0, gross_wealth=1.0, digest="")
                for item in values
            ),
            "symbol_balanced_gross_wealth",
        ),
        (
            lambda values: tuple(
                replace(
                    item,
                    net_return=-0.01,
                    net_wealth=math.exp(-0.01),
                    reward_total=-0.01,
                    digest="",
                )
                if item.symbol == "S0"
                else item
                for item in values
            ),
            "minimum_symbol_net_wealth",
        ),
        (
            lambda values: tuple(
                replace(item, turnover_per_day=1.01, digest="") for item in values
            ),
            "turnover_p95",
        ),
        (
            lambda values: tuple(
                replace(
                    item,
                    target_change_count=0,
                    submitted_change_count=0,
                    executed_change_count=0,
                    closed_trade_count=0,
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
def test_v6_selection_applies_every_common_eligibility_gate(mutate, reason: str) -> None:
    fast = mutate(_records(CausalAlphaV6Candidate.FAST_ONLY))
    retention = mutate(_records(CausalAlphaV6Candidate.FAST_SLOW_RETENTION))
    evidence = _evaluate(fast, retention)
    assert not evidence.passed
    assert reason in evidence.fast_only.rejection_reasons
    assert reason in evidence.fast_slow_retention.rejection_reasons


def test_v6_selection_requires_at_least_half_positive_scopes() -> None:
    def candidate_records(candidate: CausalAlphaV6Candidate):
        return tuple(
            _metric(
                candidate,
                symbol,
                episode,
                net=0.03 if episode == 0 else -0.005,
            )
            for episode in range(3)
            for symbol in _SYMBOLS
        )

    evidence = _evaluate(
        candidate_records(CausalAlphaV6Candidate.FAST_ONLY),
        candidate_records(CausalAlphaV6Candidate.FAST_SLOW_RETENTION),
    )
    assert "positive_net_scope_fraction" in evidence.fast_only.rejection_reasons


def test_v6_selection_selects_only_eligible_candidate() -> None:
    failed_retention = tuple(
        replace(
            item,
            net_return=-0.01,
            net_wealth=math.exp(-0.01),
            reward_total=-0.01,
            digest="",
        )
        for item in _records(CausalAlphaV6Candidate.FAST_SLOW_RETENTION)
    )
    evidence = _evaluate(
        _records(CausalAlphaV6Candidate.FAST_ONLY), failed_retention
    )
    assert evidence.selected_candidate is CausalAlphaV6Candidate.FAST_ONLY


def test_v6_selection_requires_strict_non_dominated_retention() -> None:
    fast = _records(CausalAlphaV6Candidate.FAST_ONLY)
    better = _records(CausalAlphaV6Candidate.FAST_SLOW_RETENTION, net=0.012)
    selected = _evaluate(fast, better)
    assert selected.selected_candidate is CausalAlphaV6Candidate.FAST_SLOW_RETENTION

    more_turnover = tuple(
        replace(item, turnover_per_day=0.2, digest="") for item in better
    )
    assert _evaluate(fast, more_turnover).selected_candidate is CausalAlphaV6Candidate.FAST_ONLY
    more_cost = tuple(
        replace(item, total_execution_cost=0.002, digest="") for item in better
    )
    assert _evaluate(fast, more_cost).selected_candidate is CausalAlphaV6Candidate.FAST_ONLY
    more_flips = tuple(replace(item, sign_flip_count=1, digest="") for item in better)
    assert _evaluate(fast, more_flips).selected_candidate is CausalAlphaV6Candidate.FAST_ONLY


def test_v6_selection_rejects_unpaired_scope_identity() -> None:
    retention = list(_records(CausalAlphaV6Candidate.FAST_SLOW_RETENTION))
    retention[0] = replace(retention[0], forecast_digest=_digest("drift"), digest="")
    evidence = _evaluate(
        _records(CausalAlphaV6Candidate.FAST_ONLY), tuple(retention)
    )
    assert not evidence.passed
    assert evidence.rejection_reasons == ("scope_pairing",)
