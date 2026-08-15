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


def _gate() -> CausalAlphaV3SelectionGate:
    return CausalAlphaV3SelectionGate(
        minimum_mean_gross_return=0.0,
        minimum_mean_net_return=0.0,
        minimum_symbol_episode_net_return=-0.05,
        maximum_mean_turnover_per_day=1.0,
        maximum_unexplained_execution_rejections=0,
        minimum_positive_gross_episode_fraction=0.5,
    )


def test_explained_minimum_notional_no_fill_is_not_irrecoverable() -> None:
    candidate = CausalAlphaV3Candidate(
        name="baseline",
        fit=CausalAlphaV3FitConfig(ridge_strength=0.1),
        target=CausalAlphaV3TargetConfig(
            target_magnitudes=(0.0, 0.05),
            uncertainty_multiplier=1.0,
            execution_cost_multiplier=1.5,
            edge_margin=0.001,
            alpha_rebalance_decisions=4,
            strong_reversal_threshold=0.02,
            max_target_delta=0.05,
        ),
    )
    metric = CausalAlphaV3ReplayMetric(
        run_manifest_digest="1" * 64,
        freeze_digest="2" * 64,
        candidate_digest=candidate.digest,
        symbol="BTCUSDT",
        episode_index=0,
        contract_digest="3" * 64,
        fit_digest="4" * 64,
        forecast_digest="5" * 64,
        target_path_digest="6" * 64,
        gross_return=0.01,
        net_return=0.005,
        turnover_per_day=0.1,
        total_execution_cost=1.0,
        trade_count=1,
        submitted_change_count=1,
        sign_flip_count=0,
        liquidity_deleveraging_count=0,
        execution_rejection_reason_counts=(("below_minimum_notional", 3),),
        risk_projection_reason_counts=(),
        target_reason_counts=(("hold", 2),),
        hard_risk_violation=False,
    )

    assert metric.unexplained_execution_rejection_count == 0
    assert metric.irrecoverably_rejected(_gate()) is False
