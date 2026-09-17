from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from trade_rl.evaluation.experiments.bootstrap.ridge_shared_cash_prereg import (
    RidgeSharedCashProtocol,
    canonical_ridge_shared_cash_protocol,
    load_ridge_shared_cash_protocol,
)

_SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
_FEATURE_INDICES = (0, 1, 2, 4, 5, 6, 7, 10, 60, 63, 114, 118)


def test_protocol_binds_upstream_and_shared_cash_authority() -> None:
    protocol = canonical_ridge_shared_cash_protocol()

    assert protocol.schema_version == "ridge_shared_cash_prereg_v1"
    assert protocol.issue_number == 627
    assert protocol.trigger_issue == 626
    assert protocol.trigger_status == "PROMOTE_RESEARCH_REFERENCE"
    assert protocol.trigger_run_id == 35201639813
    assert protocol.trigger_result_artifact_id == 10487739534
    assert protocol.trigger_fresh_artifact_id == 10488427586
    assert protocol.trigger_result_digest == (
        "78a39790c4d11dc903ac48f6044b0ebab46d2d6d26cbd8cf2c380be983b90d4b"
    )

    assert protocol.ridge_protocol_head == ("75999e53c70224c31a62b106e4a8d2caa4b920ac")
    assert protocol.ridge_protocol_digest == (
        "167af235eeba44b9ade780c95ef7bab11c7b211d81018b82ece5d7044f7c4eb3"
    )
    assert protocol.ridge_implementation_head == (
        "222a082ee28f4f0fd35081912a33649cce27c585"
    )
    assert protocol.ridge_implementation_seal_run_id == 35200471721

    assert protocol.shared_cash_issue == 615
    assert protocol.shared_cash_pr == 620
    assert protocol.shared_cash_implementation_head == (
        "4b9fc4bc6172b6c2a4e02e1e7cefad79575d6f70"
    )
    assert protocol.shared_cash_main_head == (
        "d18434799651cfc6c07e0840c40600dcf1dfa763"
    )
    assert protocol.shared_cash_api == (
        "trade_rl.evaluation.replay.run_shared_cash_replay"
    )
    assert protocol.require_result_blind_composition is True


def test_protocol_freezes_model_account_and_decision_rule() -> None:
    protocol = canonical_ridge_shared_cash_protocol()

    assert protocol.symbols == _SYMBOLS
    assert protocol.feature_indices == _FEATURE_INDICES
    assert protocol.fit_symbol_names == _SYMBOLS
    assert protocol.fit_cutoff == datetime(2023, 1, 1, tzinfo=UTC)
    assert protocol.evaluation_start == datetime(2023, 1, 1, tzinfo=UTC)
    assert protocol.evaluation_stop_exclusive == datetime(2025, 1, 1, tzinfo=UTC)
    assert protocol.ridge_horizon_hours == 24
    assert protocol.ridge_alpha == 1.0
    assert protocol.forecast_entry_threshold == 0.0025
    assert protocol.forecast_exit_threshold == 0.0005
    assert protocol.one_way_explicit_cost == pytest.approx(0.0007)

    assert protocol.fit_ridge_exactly_once is True
    assert protocol.baseline_candidate_share_same_model_object is True
    assert protocol.distinct_strategy_instance_per_symbol is True
    assert protocol.initial_capital == 100_000.0
    assert protocol.per_intent_gross_budget == 0.5
    assert protocol.portfolio_max_gross == 1.0
    assert protocol.portfolio_max_abs_weight == 1.0
    assert protocol.portfolio_max_turnover is None
    assert protocol.portfolio_drawdown_start == 1.0
    assert protocol.portfolio_drawdown_stop == 1.0
    assert protocol.one_shared_book is True
    assert protocol.one_risk_projection_per_bar is True
    assert protocol.one_execution_per_bar is True

    assert protocol.qualify_status == "QUALIFY_UNUSED_VALIDATION"
    assert protocol.stop_status == "STOP_BEFORE_UNUSED_VALIDATION"
    assert protocol.invalid_status == "INVALID_SHARED_CASH_EVIDENCE"
    assert protocol.qualify_requires_positive_full_return is True
    assert protocol.qualify_requires_full_return_above_baseline is True
    assert protocol.qualify_requires_positive_each_calendar_year is True
    assert protocol.qualify_requires_each_calendar_year_above_baseline is True
    assert protocol.qualify_requires_cost_reduction is True
    assert protocol.qualify_requires_turnover_reduction is True
    assert protocol.qualify_requires_drawdown_nonworse is True
    assert protocol.qualify_requires_equal_period_count is True
    assert protocol.qualify_requires_no_new_termination is True
    assert protocol.calendar_years == (2023, 2024)
    assert protocol.calendar_year_account_reset is False


def test_protocol_freezes_stop_and_production_boundaries() -> None:
    protocol = canonical_ridge_shared_cash_protocol()

    assert protocol.no_new_strategy_degree_of_freedom is True
    assert protocol.post_result_retuning_allowed is False
    assert protocol.symbol_subset_allowed is False
    assert protocol.portfolio_optimization_allowed is False
    assert protocol.unused_data_accessed is False
    assert protocol.final_test_accessed is False
    assert protocol.final_test_authorized is False
    assert protocol.operational_eligibility_established is False
    assert protocol.production_eligible is False
    assert protocol.live_trading_authorized is False
    assert protocol.merge_authorized is False
    assert protocol.shared_cash_economic_result_inspected is False
    assert protocol.shared_cash_execution_authorized is False


def test_protocol_rejects_semantic_drift() -> None:
    protocol = canonical_ridge_shared_cash_protocol()
    mutations: tuple[dict[str, object], ...] = (
        {"trigger_status": "KEEP_BASELINE"},
        {"ridge_protocol_digest": "0" * 64},
        {"ridge_implementation_head": "0" * 40},
        {"shared_cash_main_head": "0" * 40},
        {"symbols": tuple(reversed(_SYMBOLS))},
        {"feature_indices": tuple(reversed(_FEATURE_INDICES))},
        {"ridge_horizon_hours": 12},
        {"ridge_alpha": 2.0},
        {"forecast_entry_threshold": 0.003},
        {"one_way_explicit_cost": 0.001},
        {"fit_ridge_exactly_once": False},
        {"baseline_candidate_share_same_model_object": False},
        {"distinct_strategy_instance_per_symbol": False},
        {"per_intent_gross_budget": 0.25},
        {"portfolio_max_gross": 0.5},
        {"portfolio_max_turnover": 1.0},
        {"calendar_years": (2024,)},
        {"calendar_year_account_reset": True},
        {"qualify_requires_positive_full_return": False},
        {"qualify_requires_positive_each_calendar_year": False},
        {"qualify_requires_drawdown_nonworse": False},
        {"post_result_retuning_allowed": True},
        {"symbol_subset_allowed": True},
        {"portfolio_optimization_allowed": True},
        {"unused_data_accessed": True},
        {"final_test_accessed": True},
        {"production_eligible": True},
        {"shared_cash_economic_result_inspected": True},
        {"shared_cash_execution_authorized": True},
    )
    for mutation in mutations:
        with pytest.raises(ValueError, match="preregistered"):
            replace(protocol, **mutation)


def test_payload_is_strict_and_contains_no_shared_cash_result(tmp_path) -> None:
    protocol = canonical_ridge_shared_cash_protocol()
    payload = protocol.to_payload()

    forbidden = {
        "baseline_total_return",
        "candidate_total_return",
        "baseline_2023_return",
        "candidate_2023_return",
        "baseline_2024_return",
        "candidate_2024_return",
        "baseline_max_drawdown",
        "candidate_max_drawdown",
        "shared_cash_decision",
    }
    assert forbidden.isdisjoint(payload)

    path = tmp_path / "ridge-shared-cash-prereg.json"
    path.write_text(
        json.dumps(payload, allow_nan=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    loaded = load_ridge_shared_cash_protocol(path)
    assert isinstance(loaded, RidgeSharedCashProtocol)
    assert loaded == protocol
    assert loaded.digest == protocol.digest

    tampered = dict(payload)
    tampered["candidate_total_return"] = 1.0
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="keys differ"):
        load_ridge_shared_cash_protocol(path)


def test_loader_rejects_nonfinite_and_naive_time(tmp_path) -> None:
    protocol = canonical_ridge_shared_cash_protocol()
    path = tmp_path / "ridge-shared-cash-prereg.json"

    broken = protocol.to_payload()
    broken["one_way_explicit_cost"] = float("nan")
    path.write_text(json.dumps(broken, allow_nan=True), encoding="utf-8")
    with pytest.raises(ValueError, match="finite"):
        load_ridge_shared_cash_protocol(path)

    with pytest.raises(ValueError, match="timezone-aware"):
        replace(protocol, fit_cutoff=datetime(2023, 1, 1))
