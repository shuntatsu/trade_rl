from __future__ import annotations

from trade_rl.learning.causal_alpha_v3 import (
    CausalAlphaV3FitConfig,
    CausalAlphaV3TargetConfig,
)
from trade_rl.workflows.universal_causal_alpha_v3_config import (
    CausalAlphaV3Candidate,
    CausalAlphaV3SelectionGate,
)
from trade_rl.workflows.universal_causal_alpha_v3_contracts import (
    CausalAlphaV3ReplayMetric,
)
from trade_rl.workflows.universal_causal_alpha_v3_diagnostics import (
    build_causal_alpha_v3_selection_progress,
)


def _candidate(name: str, ridge: float) -> CausalAlphaV3Candidate:
    return CausalAlphaV3Candidate(
        name=name,
        fit=CausalAlphaV3FitConfig(ridge_strength=ridge),
        target=CausalAlphaV3TargetConfig(
            target_magnitudes=(0.0, 0.05, 0.1),
            uncertainty_multiplier=1.0,
            execution_cost_multiplier=1.5,
            edge_margin=0.001,
            alpha_rebalance_decisions=2,
            strong_reversal_threshold=0.02,
            max_target_delta=0.1,
        ),
    )


def _metric(
    candidate: CausalAlphaV3Candidate,
    symbol: str,
    episode: int,
    *,
    gross: float,
    net: float,
    turnover: float,
) -> CausalAlphaV3ReplayMetric:
    return CausalAlphaV3ReplayMetric(
        run_manifest_digest="1" * 64,
        freeze_digest="2" * 64,
        candidate_digest=candidate.digest,
        symbol=symbol,
        episode_index=episode,
        contract_digest=(f"{episode + 3:x}"[-1] * 64),
        fit_digest="a" * 64,
        forecast_digest="b" * 64,
        target_path_digest="c" * 64,
        gross_return=gross,
        net_return=net,
        turnover_per_day=turnover,
        total_execution_cost=1.0,
        trade_count=2,
        submitted_change_count=1,
        sign_flip_count=0,
        liquidity_deleveraging_count=0,
        execution_rejection_reason_counts=(),
        risk_projection_reason_counts=(),
        target_reason_counts=(("hold", 1),),
        hard_risk_violation=False,
    )


def _gate() -> CausalAlphaV3SelectionGate:
    return CausalAlphaV3SelectionGate(
        minimum_mean_gross_return=0.0,
        minimum_mean_net_return=0.0,
        minimum_symbol_episode_net_return=-0.05,
        maximum_mean_turnover_per_day=1.0,
        maximum_unexplained_execution_rejections=0,
        minimum_positive_gross_episode_fraction=0.5,
    )


def test_selection_progress_is_deterministic_and_descriptive_only() -> None:
    first = _candidate("first", 0.1)
    second = _candidate("second", 1.0)
    metrics = (
        _metric(first, "BTCUSDT", 0, gross=0.02, net=0.01, turnover=0.2),
        _metric(first, "ETHUSDT", 0, gross=-0.08, net=-0.06, turnover=0.1),
        _metric(second, "BTCUSDT", 0, gross=0.01, net=0.005, turnover=0.3),
        _metric(second, "ETHUSDT", 0, gross=0.02, net=0.01, turnover=0.2),
    )
    diagnostics = {
        metrics[0].identity: "d1",
        metrics[2].identity: "d2",
        metrics[3].identity: "d3",
    }

    payload = build_causal_alpha_v3_selection_progress(
        candidates=(first, second),
        train_symbols=("BTCUSDT", "ETHUSDT"),
        expected_replay_count=8,
        replay_metrics={metric.identity: metric for metric in metrics},
        diagnostics_identities=frozenset(diagnostics),
        thresholds=_gate(),
        fit_count=2,
        fit_cache_hits=7,
    )

    assert payload["schema_version"] == "causal_alpha_v3_selection_progress_v1"
    assert payload["research_only"] is True
    assert payload["promotion_eligible"] is False
    assert payload["expected_replay_count"] == 8
    assert payload["completed_replay_count"] == 4
    assert payload["completion_fraction"] == 0.5
    assert payload["diagnostics_completed_count"] == 3
    assert payload["fit_count"] == 2
    assert payload["fit_cache_hits"] == 7

    candidates = payload["candidates"]
    assert [item["name"] for item in candidates] == ["first", "second"]
    assert candidates[0]["completed_scope_count"] == 2
    assert candidates[0]["irrecoverably_rejected"] is True
    assert candidates[0]["worst_net_return"] == -0.06
    assert candidates[1]["irrecoverably_rejected"] is False

    symbols = payload["symbols"]
    assert list(symbols) == ["BTCUSDT", "ETHUSDT"]
    assert symbols["BTCUSDT"]["completed_scope_count"] == 2
    assert symbols["BTCUSDT"]["mean_net_return"] == 0.0075
    assert "selected_candidate_digest" not in payload
