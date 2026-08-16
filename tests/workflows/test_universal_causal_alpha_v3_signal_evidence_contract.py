from __future__ import annotations

import pytest

from trade_rl.workflows.universal_causal_alpha_v3_signal import (
    CausalAlphaV3BootstrapEvidence,
    CausalAlphaV3SignalGateEvidence,
    CausalAlphaV3SignalScopeMetric,
)


def _sha(token: str) -> str:
    return token * 64


def _metric(
    *,
    symbol: str,
    fit_config_digest: str = "1" * 64,
    fit_digest: str = "3" * 64,
    contract_digest: str,
) -> CausalAlphaV3SignalScopeMetric:
    return CausalAlphaV3SignalScopeMetric(
        run_manifest_digest=_sha("9"),
        fit_config_digest=fit_config_digest,
        symbol=symbol,
        episode_index=0,
        contract_start=100,
        contract_stop=197,
        contract_digest=contract_digest,
        fit_digest=fit_digest,
        forecast_digest=_sha("4"),
        sample_count=2,
        rank_correlation=0.2,
        direction_accuracy=0.6,
        top_bottom_realized_spread=0.01,
        cohort_indices=(100, 150),
    )


def _evidence(
    metrics: tuple[CausalAlphaV3SignalScopeMetric, ...],
) -> CausalAlphaV3SignalGateEvidence:
    bootstrap = CausalAlphaV3BootstrapEvidence(
        mean=0.1,
        p_value=0.1,
        lower_ci=0.01,
        upper_ci=0.2,
        block_size=1,
    )
    return CausalAlphaV3SignalGateEvidence(
        metrics=metrics,
        run_manifest_digest=_sha("9"),
        raw_scope_count=len(metrics),
        expected_raw_scope_count=len(metrics),
        raw_scope_coverage=1.0,
        independent_episode_count=1,
        expected_independent_episode_count=1,
        rank_ic=bootstrap,
        top_bottom_spread=bootstrap,
        direction_accuracy_excess=bootstrap,
        gate_digest=_sha("7"),
        passed=True,
        rejection_reasons=(),
    )


def test_signal_evidence_contract_rejects_mixed_fit_config_metrics() -> None:
    metrics = (
        _metric(symbol="BTCUSDT", contract_digest=_sha("a")),
        _metric(
            symbol="ETHUSDT",
            fit_config_digest=_sha("2"),
            contract_digest=_sha("b"),
        ),
    )

    with pytest.raises(ValueError, match="fit config"):
        _evidence(metrics)


def test_signal_evidence_contract_rejects_cluster_fit_digest_drift() -> None:
    metrics = (
        _metric(symbol="BTCUSDT", contract_digest=_sha("a")),
        _metric(
            symbol="ETHUSDT",
            fit_digest=_sha("5"),
            contract_digest=_sha("b"),
        ),
    )

    with pytest.raises(ValueError, match="fit digest"):
        _evidence(metrics)
