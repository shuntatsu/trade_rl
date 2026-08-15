from __future__ import annotations

from trade_rl.workflows.universal_causal_alpha_v3_config import CausalAlphaV3SignalGate
from trade_rl.workflows.universal_causal_alpha_v3_signal import (
    CausalAlphaV3SignalScopeMetric,
)
from trade_rl.workflows.universal_causal_alpha_v3_signal_v2 import (
    evaluate_causal_alpha_v3_signal_gate_clustered,
)


def _sha(token: str) -> str:
    assert token in "0123456789abcdef"
    return token * 64


def _metric(*, symbol: str, episode_index: int) -> CausalAlphaV3SignalScopeMetric:
    identity_token = f"{episode_index + 1:x}"
    return CausalAlphaV3SignalScopeMetric(
        fit_config_digest=_sha("f"),
        symbol=symbol,
        episode_index=episode_index,
        contract_digest=_sha(identity_token),
        fit_digest=_sha("a"),
        forecast_digest=_sha("b"),
        sample_count=2,
        rank_correlation=0.2,
        direction_accuracy=0.7,
        top_bottom_realized_spread=0.2,
        cohort_indices=(10 + episode_index * 20, 20 + episode_index * 20),
    )


def _gate(*, minimum_scope_count: int, minimum_scope_coverage: float) -> CausalAlphaV3SignalGate:
    return CausalAlphaV3SignalGate(
        minimum_scope_count=minimum_scope_count,
        minimum_scope_coverage=minimum_scope_coverage,
        minimum_rank_ic_lower_ci=0.1,
        minimum_top_bottom_spread_lower_ci=0.1,
        minimum_direction_accuracy_excess_lower_ci=0.1,
        bootstrap_resamples=100,
        bootstrap_seed=0,
        bootstrap_block_size=1,
    )


def test_clustered_signal_gate_counts_raw_scopes_but_bootstraps_episodes() -> None:
    symbols = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT")
    metrics = tuple(
        _metric(symbol=symbol, episode_index=episode_index)
        for episode_index in range(2)
        for symbol in symbols
    )

    evidence = evaluate_causal_alpha_v3_signal_gate_clustered(
        metrics,
        expected_scope_count=len(metrics),
        gate=_gate(minimum_scope_count=4, minimum_scope_coverage=1.0),
    )

    assert evidence.passed is True
    assert evidence.scope_coverage == 1.0
    assert evidence.rank_ic.mean == 0.2
    assert evidence.top_bottom_spread.mean == 0.2
    assert evidence.direction_accuracy_excess.mean == 0.2
    assert "scope_count" not in evidence.rejection_reasons


def test_clustered_signal_gate_still_rejects_too_few_raw_scopes() -> None:
    metrics = tuple(
        _metric(symbol="BTCUSDT", episode_index=episode_index)
        for episode_index in range(2)
    )

    evidence = evaluate_causal_alpha_v3_signal_gate_clustered(
        metrics,
        expected_scope_count=4,
        gate=_gate(minimum_scope_count=3, minimum_scope_coverage=0.5),
    )

    assert evidence.passed is False
    assert evidence.scope_coverage == 0.5
    assert evidence.rejection_reasons == ("scope_count",)
