from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from trade_rl.evaluation.experiments.bootstrap.agg_flow_capacity import (
    canonical_m2_aggtrades_flow_capacity_protocol,
)
from trade_rl.evaluation.experiments.bootstrap.agg_flow_predictive_screen import (
    AggTradesPredictiveFlowScreenProtocol,
    canonical_m2_aggtrades_predictive_flow_screen_protocol,
    load_aggtrades_predictive_flow_screen_protocol,
)


def test_canonical_predictive_screen_reuses_exact_pre_alpha_roster() -> None:
    protocol = canonical_m2_aggtrades_predictive_flow_screen_protocol()
    roster_authority = canonical_m2_aggtrades_flow_capacity_protocol()

    assert protocol.canonical_dataset_id == (
        "d7a04ede97a1bb37b811c3e071f325fa007525a6040927e6793d8cc7c10f538f"
    )
    assert protocol.canonical_dataset_artifact_digest == (
        "77362e148c713840dda64e0ef70e663cce6611407eac31fefbb9fccca73ae8f8"
    )
    assert protocol.provider_head_sha == "dd51da97845f4d8fe69c34b1c4e4859358911daf"
    assert protocol.provider_parser_blob_sha == (
        "15cfe8f63716451fdb8c08ae84fba66ca754c55e"
    )
    assert protocol.source_roster_protocol_digest == (
        "5fb013fb0a3d717846a701b23d2f4bfca8e742ebaef0e6f564a331053f2ed071"
    )
    assert protocol.source_roster_seal_artifact_id == 10319386570
    assert protocol.source_roster_seal_artifact_digest == (
        "dbc6ef286abb9e4c8530089328fb64b2cfaa5e98715a82740439470a45942e57"
    )
    assert protocol.discovery_start == datetime(2021, 1, 1, tzinfo=UTC)
    assert protocol.discovery_stop_exclusive == datetime(2023, 1, 1, tzinfo=UTC)
    assert protocol.evaluation_start == datetime(2023, 1, 1, tzinfo=UTC)
    assert protocol.sample_month_days == (1,)
    assert protocol.planned_days == roster_authority.planned_days
    assert protocol.planned_urls == roster_authority.planned_urls
    assert len(protocol.planned_days) == 24
    assert len(protocol.planned_urls) == 120
    assert len(set(protocol.planned_urls)) == 120
    assert all("2023-" not in url for url in protocol.planned_urls)


def test_predictor_and_label_alignment_are_causal_and_frozen() -> None:
    protocol = canonical_m2_aggtrades_predictive_flow_screen_protocol()

    assert protocol.buyer_taker_when_buyer_is_maker is False
    assert protocol.seller_taker_when_buyer_is_maker is True
    assert protocol.trade_notional_formula == "price_times_quantity"
    assert protocol.predictor_formula == "buy_minus_sell_over_buy_plus_sell"
    assert protocol.predictor_hour_alignment == "completed_utc_hour"
    assert protocol.predictor_available_at == "end_of_completed_utc_hour"
    assert protocol.execution_alignment == "next_bar_open_t_plus_1"
    assert protocol.label_formula == "log_open_t_plus_2_over_open_t_plus_1"
    assert protocol.label_horizon_bars == 1
    assert protocol.same_hour_return_allowed is False
    assert protocol.post_2022_observations_allowed is False


def test_statistic_decision_and_coverage_contract_are_frozen() -> None:
    protocol = canonical_m2_aggtrades_predictive_flow_screen_protocol()

    assert protocol.fit_rule == "ordinary_least_squares_with_intercept"
    assert protocol.reduction_rule == "chronological_math_fsum"
    assert protocol.report_pearson_correlation is True
    assert protocol.formal_decision_metric == "beta_sign"
    assert protocol.strict_positive_beta is True
    assert protocol.calendar_year_splits == (2021, 2022)
    assert protocol.min_valid_days_per_symbol == 20
    assert protocol.min_valid_days_per_year == 10
    assert protocol.min_valid_observations_per_symbol == 480
    assert protocol.min_valid_observations_per_year == 220
    assert protocol.min_positive_full_sample_symbols == 4
    assert protocol.min_positive_year_symbols == 3
    assert protocol.pass_status == "PASS_FLOW_SCREEN"
    assert protocol.no_signal_status == "NO_STABLE_FLOW_SIGNAL"
    assert protocol.invalid_status == "INVALID_FLOW_DIAGNOSTIC_COVERAGE"
    assert protocol.disjoint_validation_required_after_pass is True
    assert protocol.strategy_pnl_allowed is False


def test_protocol_rejects_post_result_tuning_and_alignment_mutation() -> None:
    protocol = canonical_m2_aggtrades_predictive_flow_screen_protocol()

    mutations: tuple[dict[str, object], ...] = (
        {"buyer_taker_when_buyer_is_maker": True},
        {"predictor_formula": "sell_minus_buy_over_total"},
        {"predictor_hour_alignment": "rolling_15m"},
        {"execution_alignment": "same_bar_close"},
        {"label_formula": "same_hour_log_return"},
        {"label_horizon_bars": 4},
        {"fit_rule": "ordinary_least_squares_without_intercept"},
        {"min_positive_full_sample_symbols": 3},
        {"min_positive_year_symbols": 2},
        {"sample_month_days": (1, 15)},
        {"post_2022_observations_allowed": True},
        {"strategy_pnl_allowed": True},
    )
    for mutation in mutations:
        with pytest.raises(ValueError, match="preregistered predictive-flow contract"):
            replace(protocol, **mutation)


def test_payload_is_content_addressed_strict_and_round_trips(tmp_path: object) -> None:
    protocol = canonical_m2_aggtrades_predictive_flow_screen_protocol()
    payload = protocol.to_payload()

    assert payload["protocol_digest"] == protocol.digest
    assert len(protocol.digest) == 64
    assert payload["planned_urls"] == list(protocol.planned_urls)
    assert {"pnl", "strategy", "candidate_result"}.isdisjoint(payload)

    path = tmp_path / "predictive-flow-protocol.json"  # type: ignore[operator]
    path.write_text(  # type: ignore[union-attr]
        json.dumps(payload, sort_keys=True),
        encoding="utf-8",
    )
    loaded = load_aggtrades_predictive_flow_screen_protocol(path)  # type: ignore[arg-type]
    assert loaded == protocol
    assert loaded.digest == protocol.digest

    tampered = dict(payload)
    tampered["label_horizon_bars"] = 4
    path.write_text(  # type: ignore[union-attr]
        json.dumps(tampered, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sealed canonical payload"):
        load_aggtrades_predictive_flow_screen_protocol(path)  # type: ignore[arg-type]

    extra = dict(payload)
    extra["pnl"] = 123.0
    path.write_text(  # type: ignore[union-attr]
        json.dumps(extra, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sealed canonical payload"):
        load_aggtrades_predictive_flow_screen_protocol(path)  # type: ignore[arg-type]


def test_protocol_type_is_immutable() -> None:
    protocol = canonical_m2_aggtrades_predictive_flow_screen_protocol()
    assert isinstance(protocol, AggTradesPredictiveFlowScreenProtocol)
    with pytest.raises((AttributeError, TypeError)):
        protocol.label_horizon_bars = 2  # type: ignore[misc]
