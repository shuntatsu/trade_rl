from trade_rl.learning.episode_oracle_bc import (
    EpisodeBehaviorCloningHoldoutEvaluation,
    EpisodeBehaviorCloningRecord,
    aggregate_episode_behavior_cloning_holdouts,
)
from trade_rl.learning.evaluation import (
    ActionPathCollapseEvidence,
    PathPerformanceMetrics,
)


def _performance(net_return: float) -> PathPerformanceMetrics:
    return PathPerformanceMetrics(
        step_count=32,
        traded_step_count=4,
        trade_count=2,
        gross_return=net_return + 0.001,
        net_return=net_return,
        reward_total=net_return * 100.0,
        reward_mean=net_return * 100.0 / 32.0,
        trade_win_rate=0.5,
        positive_step_rate=0.5,
        turnover_total=0.5,
        turnover_mean=0.5 / 32.0,
        cost_total=10.0,
        cost_mean=10.0 / 32.0,
        maximum_drawdown=0.02,
    )


def _evidence() -> ActionPathCollapseEvidence:
    return ActionPathCollapseEvidence(
        decision_count=32,
        action_dimension_count=1,
        active_dimension_count=32,
        inactive_dimension_count=0,
        proposal_distance_count=8,
        submitted_change_count=8,
        downstream_no_trade_suppression_count=4,
        execution_rejection_count=0,
        executed_change_count=4,
        trade_count=2,
        constant_submitted_actions=False,
    )


def _holdout(
    episode_id: int, net_return: float
) -> EpisodeBehaviorCloningHoldoutEvaluation:
    performance = _performance(net_return)
    record = EpisodeBehaviorCloningRecord(
        episode_id=episode_id,
        start=100 + episode_id * 32,
        stop=133 + episode_id * 32,
        initial_state_mode="cash",
        oracle_performance=_performance(0.10),
        causal_policy_performance=performance,
        causal_policy_evidence=_evidence(),
        action_agreement_rate=0.5,
        action_mae=0.2,
        action_rmse=0.3,
        action_diagnostics={},
        heldout_oracle_regret=0.10 - net_return,
        normalized_oracle_regret=1.0,
    )
    return EpisodeBehaviorCloningHoldoutEvaluation(
        records=(record,),
        causal_policy_performance=performance,
        causal_policy_evidence=_evidence(),
        action_agreement_rate=0.5,
        action_mae=0.2,
        action_rmse=0.3,
        heldout_oracle_regret=0.10 - net_return,
        normalized_oracle_regret=1.0,
        causal_regret_upper_confidence_bound=1.0,
        causal_net_return_lower_confidence_bound=net_return,
        bootstrap_confidence_level=0.95,
        bootstrap_resamples=2_000,
    )


def test_aggregate_episode_holdouts_preserves_cross_symbol_support() -> None:
    aggregate = aggregate_episode_behavior_cloning_holdouts(
        (_holdout(0, 0.01), _holdout(1, -0.02)),
        seed_material="universal-holdout",
    )

    assert len(aggregate.records) == 2
    assert aggregate.causal_policy_performance.net_return == -0.02
    assert aggregate.causal_policy_evidence.decision_count == 64
    assert aggregate.causal_policy_evidence.executed_change_count == 8
    assert aggregate.causal_policy_evidence.trade_count == 4
    assert aggregate.causal_net_return_lower_confidence_bound <= 0.01
