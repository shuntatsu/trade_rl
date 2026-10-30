"""Tests for TD5 bounded real-development study runner and contracts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from trade_rl.data.market import MarketCalendarKind, MarketDataset
from trade_rl.evaluation.decision_value_dev_study import (
    DecisionValueBlockSpec,
    DecisionValueDevStudyProtocol,
    compute_real_scope_authorization_digest,
    default_2024q1_decision_value_dev_protocol,
    run_decision_value_dev_study,
)
from trade_rl.evaluation.decision_value_fit import (
    fit_synthetic_linear_decision_value,
    fit_synthetic_shallow_decision_value,
)
from trade_rl.evaluation.decision_value_plan import DecisionValueStudyPlan
from trade_rl.evaluation.paired_portfolio_action import run_paired_portfolio_action
from trade_rl.evaluation.paired_portfolio_label import (
    build_synthetic_paired_label,
    build_synthetic_paired_label_batch,
)
from trade_rl.risk.pretrade import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.portfolio import (
    PortfolioDecisionSnapshot,
    PortfolioTargetProposal,
)
from trade_rl.strategies.portfolio_policy_codec import (
    ACTION_HOLD,
    ACTION_LONG,
    ACTION_SHORT,
)

_HOLD = (ACTION_HOLD, ACTION_HOLD)
_LONG = (ACTION_LONG, ACTION_HOLD)
_SHORT = (ACTION_SHORT, ACTION_HOLD)
_CANDIDATES = (_HOLD, _LONG, _SHORT)
_COST = ExecutionCostConfig(
    fee_rate=0.0005, spread_rate=0.0002, impact_rate=0.0001, order_latency_bars=0
)
_RISK = PreTradeRisk(
    PreTradeRiskConfig(
        max_gross=0.5,
        max_abs_weight=0.25,
        max_turnover=1.0,
        drawdown_start=0.15,
        drawdown_stop=0.20,
    )
)


class _HoldPolicy:
    def decide_portfolio(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> PortfolioTargetProposal:
        return PortfolioTargetProposal(
            snapshot.account.weights,
            hold_current_quantity_symbols=(0, 1),
        )


def _compact_market(n_bars: int = 100) -> MarketDataset:
    timestamps = np.array(
        [
            np.datetime64("2024-01-01T00:00:00", "ns") + np.timedelta64(i, "h")
            for i in range(n_bars)
        ],
        dtype="datetime64[ns]",
    )
    rng = np.random.default_rng(42)
    # 2 symbols
    prices = np.cumprod(
        1.0 + rng.normal(0.0005, 0.01, size=(n_bars, 2)), axis=0
    ) * np.array([100.0, 50.0])
    high = prices * 1.005
    low = prices * 0.995
    volume = np.full((n_bars, 2), 1_000_000.0)
    features = np.zeros((n_bars, 2, 9), dtype=np.float64)
    # populate some signals
    features[:, :, 0] = rng.normal(0.0, 0.01, size=(n_bars, 2))
    features[:, :, 1] = rng.normal(0.0, 0.02, size=(n_bars, 2))
    funding = np.zeros((n_bars, 2), dtype=np.float64)
    available_at = np.repeat(timestamps[:, None], 2, axis=1)

    return MarketDataset(
        dataset_id="a" * 64,
        feature_config_digest="b" * 64,
        normalization_digest="c" * 64,
        symbols=("BTCUSDT", "ETHUSDT"),
        timestamps=timestamps,
        open=prices,
        high=high,
        low=low,
        close=prices,
        volume=volume,
        funding_rate=funding,
        features=features,
        available_at=available_at,
        tradable=np.ones_like(prices, dtype=np.bool_),
        feature_available=np.ones((n_bars, 2, 9), dtype=np.bool_),
        feature_names=tuple(f"f_{i}" for i in range(9)),
        global_feature_names=("g_0", "g_1"),
        global_features=np.zeros((n_bars, 2), dtype=np.float64),
        global_feature_available=np.ones((n_bars, 2), dtype=np.bool_),
        periods_per_year=8_760,
        calendar_kind=MarketCalendarKind.CONTINUOUS,
        nominal_bar_hours=1.0,
    )


def test_real_development_label_admission_and_fit() -> None:
    dataset = _compact_market(50)
    pair = run_paired_portfolio_action(
        dataset,
        reference_policy_factory=_HoldPolicy,
        start_index=5,
        stop_index=15,
        intervention_index=5,
        baseline_action=_HOLD,
        candidate_action=_LONG,
        target_abs_weights=(0.25, 0.25),
        initial_capital=100_000.0,
        execution_cost=_COST,
        risk=_RISK,
    )
    label = build_synthetic_paired_label(
        dataset,
        pair,
        baseline_action=_HOLD,
        candidate_action=_LONG,
        target_abs_weights=(0.25, 0.25),
        reference_policy_digest="d" * 64,
        expected_execution_policy_digest=_COST.execution_policy_digest,
        observation_record_available_at=dataset.timestamps[5],
        settlement_record_available_at=dataset.timestamps[15],
        fit_cutoff=dataset.timestamps[20],
        source_scope="real_development",
    )
    assert label.source_scope == "real_development"

    pair2 = run_paired_portfolio_action(
        dataset,
        reference_policy_factory=_HoldPolicy,
        start_index=6,
        stop_index=16,
        intervention_index=6,
        baseline_action=_HOLD,
        candidate_action=_SHORT,
        target_abs_weights=(0.25, 0.25),
        initial_capital=100_000.0,
        execution_cost=_COST,
        risk=_RISK,
    )
    label2 = build_synthetic_paired_label(
        dataset,
        pair2,
        baseline_action=_HOLD,
        candidate_action=_SHORT,
        target_abs_weights=(0.25, 0.25),
        reference_policy_digest="d" * 64,
        expected_execution_policy_digest=_COST.execution_policy_digest,
        observation_record_available_at=dataset.timestamps[6],
        settlement_record_available_at=dataset.timestamps[16],
        fit_cutoff=dataset.timestamps[20],
        source_scope="real_development",
    )
    # Also add candidate LONG at 6 and candidate SHORT at 5 to have enough candidates for Ridge
    pair3 = run_paired_portfolio_action(
        dataset,
        reference_policy_factory=_HoldPolicy,
        start_index=5,
        stop_index=15,
        intervention_index=5,
        baseline_action=_HOLD,
        candidate_action=_SHORT,
        target_abs_weights=(0.25, 0.25),
        initial_capital=100_000.0,
        execution_cost=_COST,
        risk=_RISK,
    )
    label3 = build_synthetic_paired_label(
        dataset,
        pair3,
        baseline_action=_HOLD,
        candidate_action=_SHORT,
        target_abs_weights=(0.25, 0.25),
        reference_policy_digest="d" * 64,
        expected_execution_policy_digest=_COST.execution_policy_digest,
        observation_record_available_at=dataset.timestamps[5],
        settlement_record_available_at=dataset.timestamps[15],
        fit_cutoff=dataset.timestamps[20],
        source_scope="real_development",
    )
    pair4 = run_paired_portfolio_action(
        dataset,
        reference_policy_factory=_HoldPolicy,
        start_index=6,
        stop_index=16,
        intervention_index=6,
        baseline_action=_HOLD,
        candidate_action=_LONG,
        target_abs_weights=(0.25, 0.25),
        initial_capital=100_000.0,
        execution_cost=_COST,
        risk=_RISK,
    )
    label4 = build_synthetic_paired_label(
        dataset,
        pair4,
        baseline_action=_HOLD,
        candidate_action=_LONG,
        target_abs_weights=(0.25, 0.25),
        reference_policy_digest="d" * 64,
        expected_execution_policy_digest=_COST.execution_policy_digest,
        observation_record_available_at=dataset.timestamps[6],
        settlement_record_available_at=dataset.timestamps[16],
        fit_cutoff=dataset.timestamps[20],
        source_scope="real_development",
    )
    pair5 = run_paired_portfolio_action(
        dataset,
        reference_policy_factory=_HoldPolicy,
        start_index=7,
        stop_index=17,
        intervention_index=7,
        baseline_action=_HOLD,
        candidate_action=_LONG,
        target_abs_weights=(0.25, 0.25),
        initial_capital=100_000.0,
        execution_cost=_COST,
        risk=_RISK,
    )
    label5 = build_synthetic_paired_label(
        dataset,
        pair5,
        baseline_action=_HOLD,
        candidate_action=_LONG,
        target_abs_weights=(0.25, 0.25),
        reference_policy_digest="d" * 64,
        expected_execution_policy_digest=_COST.execution_policy_digest,
        observation_record_available_at=dataset.timestamps[7],
        settlement_record_available_at=dataset.timestamps[17],
        fit_cutoff=dataset.timestamps[20],
        source_scope="real_development",
    )
    pair6 = run_paired_portfolio_action(
        dataset,
        reference_policy_factory=_HoldPolicy,
        start_index=7,
        stop_index=17,
        intervention_index=7,
        baseline_action=_HOLD,
        candidate_action=_SHORT,
        target_abs_weights=(0.25, 0.25),
        initial_capital=100_000.0,
        execution_cost=_COST,
        risk=_RISK,
    )
    label6 = build_synthetic_paired_label(
        dataset,
        pair6,
        baseline_action=_HOLD,
        candidate_action=_SHORT,
        target_abs_weights=(0.25, 0.25),
        reference_policy_digest="d" * 64,
        expected_execution_policy_digest=_COST.execution_policy_digest,
        observation_record_available_at=dataset.timestamps[7],
        settlement_record_available_at=dataset.timestamps[17],
        fit_cutoff=dataset.timestamps[20],
        source_scope="real_development",
    )

    batch = build_synthetic_paired_label_batch(
        (label, label2, label3, label4, label5, label6),
        fit_cutoff=dataset.timestamps[20],
        evaluation_start=dataset.timestamps[25],
        expected_reference_policy_digest="d" * 64,
    )
    b1 = fit_synthetic_linear_decision_value(batch, candidate_actions=_CANDIDATES)
    b2 = fit_synthetic_shallow_decision_value(batch, candidate_actions=_CANDIDATES)
    assert b1.source_scope == "real_development"
    assert b2.source_scope == "real_development"


def test_cross_scope_rejection() -> None:
    dataset = _compact_market(30)
    pair = run_paired_portfolio_action(
        dataset,
        reference_policy_factory=_HoldPolicy,
        start_index=5,
        stop_index=15,
        intervention_index=5,
        baseline_action=_HOLD,
        candidate_action=_LONG,
        target_abs_weights=(0.25, 0.25),
        initial_capital=100_000.0,
        execution_cost=_COST,
        risk=_RISK,
    )
    # Reject invalid scope
    with pytest.raises(ValueError, match="synthetic_only or real_development"):
        build_synthetic_paired_label(
            dataset,
            pair,
            baseline_action=_HOLD,
            candidate_action=_LONG,
            target_abs_weights=(0.25, 0.25),
            reference_policy_digest="d" * 64,
            expected_execution_policy_digest=_COST.execution_policy_digest,
            observation_record_available_at=dataset.timestamps[5],
            settlement_record_available_at=dataset.timestamps[15],
            fit_cutoff=dataset.timestamps[20],
            source_scope="real_market",
        )

    # Reject mixed batch
    l_synth = build_synthetic_paired_label(
        dataset,
        pair,
        baseline_action=_HOLD,
        candidate_action=_LONG,
        target_abs_weights=(0.25, 0.25),
        reference_policy_digest="d" * 64,
        expected_execution_policy_digest=_COST.execution_policy_digest,
        observation_record_available_at=dataset.timestamps[5],
        settlement_record_available_at=dataset.timestamps[15],
        fit_cutoff=dataset.timestamps[20],
        source_scope="synthetic_only",
    )
    l_real = build_synthetic_paired_label(
        dataset,
        pair,
        baseline_action=_HOLD,
        candidate_action=_LONG,
        target_abs_weights=(0.25, 0.25),
        reference_policy_digest="d" * 64,
        expected_execution_policy_digest=_COST.execution_policy_digest,
        observation_record_available_at=dataset.timestamps[5],
        settlement_record_available_at=dataset.timestamps[15],
        fit_cutoff=dataset.timestamps[20],
        source_scope="real_development",
    )
    with pytest.raises(ValueError, match="synthetic"):
        build_synthetic_paired_label_batch(
            (l_synth, replace(l_real, candidate_action=_SHORT)),
            fit_cutoff=dataset.timestamps[20],
            evaluation_start=dataset.timestamps[25],
            expected_reference_policy_digest="d" * 64,
        )


def test_default_2024q1_protocol_construction_and_fail_closed() -> None:
    proto = default_2024q1_decision_value_dev_protocol()
    assert proto.study_plan.data_scope == "real_development"
    assert len(proto.blocks) == 3
    assert (
        proto.dataset_id
        == "7cff150e0f4d18dcc457009232f53cc7ca3f350db4893c433db6494887979616"
    )
    assert proto.protocol_digest is not None

    # Tampered authorization digest is rejected
    tampered_plan = replace(proto.study_plan, real_scope_authorization_digest="0" * 64)
    with pytest.raises(ValueError, match="authorization"):
        replace(proto, study_plan=tampered_plan)

    # Missing blocks or >3 blocks rejected
    with pytest.raises(ValueError, match="between 1 and 3 blocks"):
        replace(proto, blocks=())
    with pytest.raises(ValueError, match="between 1 and 3 blocks"):
        replace(proto, blocks=(proto.blocks[0],) * 4)

    # Overlapping or non-chronological blocks rejected
    bad_blocks = (
        DecisionValueBlockSpec(
            block_id="b1",
            train_start_index=24,
            train_stop_index=954,
            train_step=6,
            fit_cutoff_index=984,
            eval_start_index=985,
            eval_stop_index=1400,
        ),
        DecisionValueBlockSpec(
            block_id="b2",
            train_start_index=24,
            train_stop_index=1338,
            train_step=6,
            fit_cutoff_index=1368,
            eval_start_index=1369,  # overlap with 1400
            eval_stop_index=1728,
        ),
    )
    with pytest.raises(ValueError):
        replace(proto, blocks=bad_blocks)


def test_run_decision_value_dev_study_end_to_end_on_compact_dataset(
    tmp_path: Path,
) -> None:
    dataset = _compact_market(90)
    dataset_id = dataset.dataset_id
    artifact_digest = "e" * 64
    symbols = dataset.symbols
    initial_capital = 100_000.0
    target_abs_weights = (0.25, 0.25)
    horizon_bars = 4
    terminal_reserve_bars = 1
    candidate_actions = (_HOLD, _LONG, _SHORT)
    fee_rate = 0.0005
    spread_rate = 0.0002
    impact_rate = 0.0001
    max_participation_rate = 0.05
    risk_config = PreTradeRiskConfig(
        max_gross=0.50,
        max_abs_weight=0.25,
        max_turnover=1.0,
        drawdown_start=0.15,
        drawdown_stop=0.20,
    )
    base_cost = ExecutionCostConfig(
        fee_rate=fee_rate,
        spread_rate=spread_rate,
        impact_rate=impact_rate,
        max_participation_rate=max_participation_rate,
        order_latency_bars=0,
    )
    blocks = (
        DecisionValueBlockSpec(
            block_id="test_block_1",
            train_start_index=0,
            train_stop_index=24,
            train_step=4,
            fit_cutoff_index=30,
            eval_start_index=35,
            eval_stop_index=55,
        ),
        DecisionValueBlockSpec(
            block_id="test_block_2",
            train_start_index=0,
            train_stop_index=40,
            train_step=4,
            fit_cutoff_index=55,
            eval_start_index=60,
            eval_stop_index=85,
        ),
    )
    ridge_features = (0, 1)
    ridge_horizon = 4
    alpha = 1.0
    entry_thresh = 0.001
    exit_thresh = 0.0005
    prior_disclosure = "compact synthetic test disclosure"

    auth_digest = compute_real_scope_authorization_digest(
        dataset_id=dataset_id,
        expected_artifact_digest=artifact_digest,
        symbols=symbols,
        initial_capital=initial_capital,
        target_abs_weights=target_abs_weights,
        horizon_bars=horizon_bars,
        terminal_reserve_bars=terminal_reserve_bars,
        candidate_actions=candidate_actions,
        fee_rate=fee_rate,
        spread_rate=spread_rate,
        impact_rate=impact_rate,
        max_participation_rate=max_participation_rate,
        risk_config=risk_config,
        blocks=blocks,
        ridge_feature_indices=ridge_features,
        ridge_horizon_hours=ridge_horizon,
        ridge_alpha=alpha,
        ridge_entry_threshold=entry_thresh,
        ridge_exit_threshold=exit_thresh,
        prior_usage_disclosure=prior_disclosure,
        label_prefix_bars=4,
        reference_min_fit_bars=6,
    )

    plan = DecisionValueStudyPlan(
        data_scope="real_development",
        account_mode="shared_portfolio",
        symbols=symbols,
        initial_capital=initial_capital,
        horizon_bars=horizon_bars,
        terminal_reserve_bars=terminal_reserve_bars,
        candidate_actions=candidate_actions,
        fee_rate=fee_rate,
        spread_rate=spread_rate,
        max_policy_improvements=1,
        learner_roster=("linear", "conservative_linear", "shallow"),
        real_scope_authorization_digest=auth_digest,
    )

    protocol = DecisionValueDevStudyProtocol(
        study_plan=plan,
        dataset_id=dataset_id,
        expected_artifact_digest=artifact_digest,
        blocks=blocks,
        target_abs_weights=target_abs_weights,
        risk_config=risk_config,
        base_execution_cost=base_cost,
        ridge_feature_indices=ridge_features,
        ridge_horizon_hours=ridge_horizon,
        ridge_alpha=alpha,
        ridge_entry_threshold=entry_thresh,
        ridge_exit_threshold=exit_thresh,
        prior_usage_disclosure=prior_disclosure,
        label_prefix_bars=4,
        reference_min_fit_bars=6,
    )

    out_dir = tmp_path / "study_output"
    result = run_decision_value_dev_study(
        dataset,
        protocol,
        actual_artifact_digest=artifact_digest,
        output_dir=out_dir,
    )

    assert result.schema_version == "decision_value_dev_study_result_v2"
    assert result.verdict in (
        "DEVELOPMENT_IMPROVED",
        "INCONCLUSIVE",
        "NO_IMPROVEMENT",
        "INVALID / BLOCKED",
    )
    assert len(result.blocks) == 2
    for b in result.blocks:
        assert b.causality_probe_passed
        assert b.ledger_evidence_reconciled
        assert b.training_rows > 0
        assert b.paired_replay_runs > 0
        assert len(b.b1c_model_coefficients_sha256) == 64
        assert len(b.arms) == 24
        # TD8 evidence: chosen margins, B1c dof, label population and shift.
        assert set(b.chosen_margin_multipliers) == {
            "linear_value",
            "conservative_linear_value",
            "shallow_value",
        }
        assert all(
            m in protocol.margin_multiplier_grid
            for m in b.chosen_margin_multipliers.values()
        )
        assert len(b.b1c_residual_dof) == len(candidate_actions) - 1
        assert all(dof >= 1.0 for dof in b.b1c_residual_dof)
        assert b.distinct_decision_times > 0
        assert b.ineligible_pairs >= 0
        assert b.nonflat_branch_states >= 0
        assert len(b.covariate_shift) > 0
        assert {r["scope"] for r in b.covariate_shift} == {"symbol", "global"}
        assert all(a.turnover_total >= 0.0 for a in b.arms)

    assert "conservative_linear_value" in result.aggregate_metrics

    assert (out_dir / "protocol.json").is_file()
    assert (out_dir / "result.json").is_file()
    assert (out_dir / "summary.md").is_file()

    # Write-once: running again with same output_dir must fail
    with pytest.raises(FileExistsError):
        run_decision_value_dev_study(
            dataset,
            protocol,
            actual_artifact_digest=artifact_digest,
            output_dir=out_dir,
        )


def test_study_preflight_rejections() -> None:
    dataset = _compact_market(50)
    proto = default_2024q1_decision_value_dev_protocol()

    # Dataset ID mismatch
    with pytest.raises(ValueError, match="dataset_id"):
        run_decision_value_dev_study(dataset, proto)

    # Artifact digest mismatch
    with pytest.raises(ValueError, match="actual_artifact_digest"):
        run_decision_value_dev_study(
            replace(dataset, dataset_id=proto.dataset_id),
            proto,
            actual_artifact_digest="f" * 64,
        )


def test_paired_portfolio_action_latency_2bar_preservation() -> None:
    from trade_rl.evaluation.decision_value_dev_study import _HoldQuantityPolicy

    dataset = _compact_market(60)
    cost_lat2 = replace(_COST, order_latency_bars=2)
    pair = run_paired_portfolio_action(
        dataset,
        reference_policy_factory=lambda: _HoldQuantityPolicy(
            target_abs_weights=(0.25, 0.25)
        ),
        start_index=10,
        stop_index=25,
        intervention_index=10,
        baseline_action=_HOLD,
        candidate_action=_LONG,
        target_abs_weights=(0.25, 0.25),
        initial_capital=100_000.0,
        execution_cost=cost_lat2,
        risk=_RISK,
    )
    assert pair.eligible_for_label
    assert pair.delta_per_initial_capital is not None
    assert pair.delta_per_initial_capital != 0.0


def test_cost_aware_ridge_policy_latency_2bar_fills() -> None:
    from trade_rl.evaluation.decision_value_dev_study import (
        _CostAwareRidgePortfolioPolicy,
    )
    from trade_rl.evaluation.replay import run_shared_portfolio_replay
    from trade_rl.strategies.forecasts.ridge import fit_ridge_forecast

    dataset = _compact_market(60)
    cost_lat2 = replace(_COST, order_latency_bars=2)
    model = fit_ridge_forecast(
        dataset,
        feature_indices=(0, 1),
        fit_cutoff=dataset.timestamps[30],
        horizon_hours=4,
        alpha=1.0,
    )
    policy = _CostAwareRidgePortfolioPolicy(
        dataset=dataset,
        model=model,
        target_abs_weights=(0.25, 0.25),
        entry_threshold=0.0001,
        exit_threshold=0.00005,
        one_way_switch_cost=0.0008,
    )
    replay_res = run_shared_portfolio_replay(
        dataset,
        policy,
        start_index=30,
        stop_index=55,
        initial_capital=100_000.0,
        execution_cost=cost_lat2,
        risk=_RISK,
        settle_terminal_position=True,
        capture_ledger_evidence=True,
    )
    assert replay_res.book.turnover_total > 0.0
    assert replay_res.ledger_evidence is not None
