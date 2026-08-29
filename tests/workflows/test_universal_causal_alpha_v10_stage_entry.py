from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import trade_rl.workflows.universal_causal_alpha_v10_stage_entry as stage_entry
from trade_rl.learning.causal_alpha_v10 import CausalAlphaV10Candidate
from trade_rl.learning.rollout_evaluation import ActionPathExecutionTrace
from trade_rl.risk.pretrade import PreTradeRiskConfig
from trade_rl.workflows.universal_causal_alpha_v10_stage_entry import (
    _execution_rebalance_contract,
    _path,
    _require_execution_rebalance_contract,
    _training_rows,
)


def test_v10_training_rows_bind_fast_and_slow_labels_without_symbol_features() -> None:
    decisions = np.arange(10, 90, 10, dtype=np.int64)
    local = SimpleNamespace(
        feature_names=("return_4h", "volatility"),
        values=np.column_stack((decisions, decisions + 1.0)),
        available=np.ones((len(decisions), 2), dtype=np.bool_),
    )
    global_context = SimpleNamespace(
        feature_names=("market_return",),
        values=decisions[:, None] + 2.0,
        available=np.ones((len(decisions), 1), dtype=np.bool_),
    )
    sample = SimpleNamespace(
        decision_indices=decisions,
        label_end_indices_4h=decisions + 2,
        label_end_indices_72h=decisions + 20,
        labels_4h=decisions / 1_000.0,
        labels_72h=decisions / 100.0,
        local_context=local,
        global_context=global_context,
    )
    prepared = SimpleNamespace(
        train_symbols=("AAAUSDT",),
        samples={"AAAUSDT": sample},
    )

    rows = _training_rows(prepared)["AAAUSDT"]

    assert rows.fast_label_end_indices.tolist() == (decisions + 2).tolist()
    assert rows.slow_label_end_indices.tolist() == (decisions + 20).tolist()
    assert rows.fast_labels.tolist() == (decisions / 1_000.0).tolist()
    assert rows.slow_labels.tolist() == (decisions / 100.0).tolist()
    assert rows.feature_names == (
        "local:return_4h",
        "local:volatility",
        "global:market_return",
    )
    assert all("symbol" not in name for name in rows.feature_names)


def _risk_config(
    *, exit_threshold: float = 0.03, drawdown_start: float = 0.12
) -> PreTradeRiskConfig:
    return PreTradeRiskConfig(
        max_gross=1.0,
        max_abs_weight=0.40,
        max_turnover=0.75,
        entry_threshold=0.10,
        exit_threshold=exit_threshold,
        no_trade_band=0.05,
        drawdown_start=drawdown_start,
        drawdown_stop=0.25,
        emergency_turnover_override=True,
        fail_closed_tolerance=1e-9,
    )


def test_v10_resolves_full_execution_contract_from_environment_and_closes_it() -> None:
    closed: list[bool] = []

    class Environment:
        pre_trade_risk = SimpleNamespace(config=_risk_config())

        def close(self) -> None:
            closed.append(True)

    prepared = SimpleNamespace(
        prepared_v3=SimpleNamespace(
            environment_factories={"BTCUSDT": Environment},
        )
    )

    contract = _execution_rebalance_contract(prepared, "BTCUSDT")

    assert contract.entry_threshold == 0.10
    assert contract.exit_threshold == 0.03
    assert contract.no_trade_band == 0.05
    assert contract.max_turnover == 0.75
    assert contract.drawdown_start == 0.12
    assert contract.drawdown_stop == 0.25
    assert contract.fail_closed_tolerance == 1e-9
    assert closed == [True]


def test_v10_execution_contract_digest_binds_non_entry_pretrade_fields() -> None:
    class EnvironmentA:
        pre_trade_risk = SimpleNamespace(config=_risk_config(drawdown_start=0.12))

        def close(self) -> None:
            pass

    class EnvironmentB:
        pre_trade_risk = SimpleNamespace(config=_risk_config(drawdown_start=0.13))

        def close(self) -> None:
            pass

    prepared = SimpleNamespace(
        prepared_v3=SimpleNamespace(
            environment_factories={
                "AAAUSDT": EnvironmentA,
                "BBBUSDT": EnvironmentB,
            },
        )
    )

    first = _execution_rebalance_contract(prepared, "AAAUSDT")
    second = _execution_rebalance_contract(prepared, "BBBUSDT")

    assert first.entry_threshold == second.entry_threshold == 0.10
    assert first.no_trade_band == second.no_trade_band == 0.05
    assert first.digest != second.digest


def test_v10_replay_rejects_execution_contract_drift_in_exit_threshold() -> None:
    environment = SimpleNamespace(
        pre_trade_risk=SimpleNamespace(config=_risk_config(exit_threshold=0.02))
    )
    expected_environment = SimpleNamespace(
        pre_trade_risk=SimpleNamespace(config=_risk_config(exit_threshold=0.03))
    )
    expected = stage_entry._environment_rebalance_contract(expected_environment)

    with pytest.raises(ValueError, match="execution rebalance contract drifted"):
        _require_execution_rebalance_contract(environment, expected)


def test_v10_closed_loop_replay_uses_new_leaf_schema() -> None:
    assert stage_entry._REPLAY_LEAF_SCHEMA == "causal_alpha_v10_replay_leaf_v3"


def test_v10_execution_diagnostics_persist_reconciled_boundary_trace() -> None:
    trace = ActionPathExecutionTrace(
        pre_action_weights=np.asarray([[0.0], [0.09], [0.08]]),
        risk_constrained_weights=np.asarray([[0.10], [0.09], [0.0]]),
        post_step_weights=np.asarray([[0.09], [0.08], [0.0]]),
        applied_risk_scales=np.asarray([1.0, 0.8, 0.5]),
        strategy_intent_changes=np.asarray([True, False, True]),
        realized_state_follows=np.asarray([False, True, False]),
        rebalance_reassertions=np.asarray([False, False, False]),
        hard_risk_violations=np.asarray([False, True, False]),
    )
    evaluation = SimpleNamespace(
        execution_trace=trace,
        collapse_evidence=SimpleNamespace(hard_risk_violation=True),
    )

    trace_payload = stage_entry._execution_trace_payload(evaluation)
    diagnostics = stage_entry._execution_diagnostics(evaluation, trace_payload)

    assert trace_payload["schema_version"] == "causal_alpha_v10_execution_trace_v1"
    assert trace_payload["pre_action_weights"] == [[0.0], [0.09], [0.08]]
    assert trace_payload["applied_risk_scales"] == [1.0, 0.8, 0.5]
    assert trace_payload["hard_risk_violations"] == [False, True, False]
    assert diagnostics["schema_version"] == "causal_alpha_v10_execution_diagnostics_v1"
    assert diagnostics["trace_digest"] == trace_payload["artifact_digest"]
    assert diagnostics["strategy_intent_change_count"] == 2
    assert diagnostics["realized_state_follow_count"] == 1
    assert diagnostics["rebalance_reassertion_count"] == 0
    assert diagnostics["hard_risk_violation"] is True
    assert diagnostics["minimum_applied_risk_scale"] == pytest.approx(0.5)
    assert diagnostics["pre_action_mean_abs_weight"] == pytest.approx(0.17 / 3.0)
    assert diagnostics["risk_constrained_mean_abs_weight"] == pytest.approx(0.19 / 3.0)
    assert diagnostics["post_step_mean_abs_weight"] == pytest.approx(0.17 / 3.0)


def test_v10_execution_trace_rejects_string_boolean_tampering() -> None:
    trace = ActionPathExecutionTrace(
        pre_action_weights=np.asarray([[0.0], [0.09], [0.08]]),
        risk_constrained_weights=np.asarray([[0.10], [0.09], [0.0]]),
        post_step_weights=np.asarray([[0.09], [0.08], [0.0]]),
        applied_risk_scales=np.asarray([1.0, 0.8, 0.5]),
        strategy_intent_changes=np.asarray([True, False, True]),
        realized_state_follows=np.asarray([False, True, False]),
        rebalance_reassertions=np.asarray([False, False, False]),
        hard_risk_violations=np.asarray([False, True, False]),
    )
    evaluation = SimpleNamespace(
        execution_trace=trace,
        collapse_evidence=SimpleNamespace(hard_risk_violation=True),
    )
    payload = stage_entry._execution_trace_payload(evaluation)
    payload["hard_risk_violations"] = [False, "false", False]

    with pytest.raises(ValueError, match="boolean"):
        stage_entry._validate_execution_trace_payload(payload)


def test_v10_hierarchical_replay_has_dedicated_closed_loop_path() -> None:
    assert callable(getattr(stage_entry, "_build_hierarchical_replay", None))


def test_v10_leaf_paths_are_candidate_symbol_episode_scoped() -> None:
    assert _path(
        CausalAlphaV10Candidate.HIERARCHICAL_WAVE, "BTCUSDT", 8
    ).as_posix() == ("selection/replays/08/BTCUSDT/hierarchical_wave.json")
