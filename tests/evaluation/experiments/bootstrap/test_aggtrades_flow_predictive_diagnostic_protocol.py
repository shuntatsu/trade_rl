from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from trade_rl.evaluation.experiments.bootstrap.agg_flow_predictive_diagnostic import (
    canonical_aggtrades_flow_predictive_diagnostic_protocol,
    flow_screen_status,
    load_aggtrades_flow_predictive_diagnostic_protocol,
)


def test_canonical_protocol_freezes_discovery_roster_and_causal_alignment() -> None:
    protocol = canonical_aggtrades_flow_predictive_diagnostic_protocol()

    assert protocol.symbols == (
        "BTCUSDT",
        "ETHUSDT",
        "BNBUSDT",
        "XRPUSDT",
        "ADAUSDT",
    )
    assert len(protocol.planned_days) == 24
    assert len(protocol.planned_urls) == 120
    assert len(set(protocol.planned_urls)) == 120
    assert all("2023-" not in url for url in protocol.planned_urls)
    assert protocol.sample_month_days == (1,)
    assert protocol.discovery_start == datetime(2021, 1, 1, tzinfo=UTC)
    assert protocol.discovery_stop_exclusive == datetime(2023, 1, 1, tzinfo=UTC)
    assert protocol.predictor_availability == "completed_utc_hour_end"
    assert protocol.predictor_formula == "buy_minus_sell_over_total_taker_notional"
    assert protocol.label_formula == "log_open_t_plus_2_over_open_t_plus_1"
    assert protocol.regression_intercept is True
    assert protocol.slope_reduction == "chronological_math_fsum_centered_ols"


def test_protocol_binds_existing_provider_roster_and_ohlcv_authorities() -> None:
    protocol = canonical_aggtrades_flow_predictive_diagnostic_protocol()

    assert protocol.provider_head_sha == "dd51da97845f4d8fe69c34b1c4e4859358911daf"
    assert (
        protocol.provider_parser_blob_sha == "15cfe8f63716451fdb8c08ae84fba66ca754c55e"
    )
    assert protocol.archive_roster_protocol_digest == (
        "5fb013fb0a3d717846a701b23d2f4bfca8e742ebaef0e6f564a331053f2ed071"
    )
    assert protocol.archive_evidence_run_id == 34766830666
    assert protocol.archive_evidence_artifact_id == 10320428830
    assert protocol.archive_evidence_artifact_digest == (
        "810e792636a696a0bc11e22ab53a21f378aaa0bfaeab8c3411123e9c6d9a5c35"
    )
    assert protocol.archive_evidence_fresh_verifier_run_id == 34767391268
    assert protocol.canonical_dataset_id == (
        "d7a04ede97a1bb37b811c3e071f325fa007525a6040927e6793d8cc7c10f538f"
    )
    assert protocol.canonical_dataset_artifact_digest == (
        "77362e148c713840dda64e0ef70e663cce6611407eac31fefbb9fccca73ae8f8"
    )


def test_flow_screen_decision_is_fixed_by_beta_sign_counts_only() -> None:
    assert (
        flow_screen_status(
            full_sample_betas=(0.1, 0.2, 0.3, 0.4, -0.1),
            year_2021_betas=(0.1, 0.2, 0.3, -0.1, -0.2),
            year_2022_betas=(0.1, 0.2, 0.3, -0.1, -0.2),
        )
        == "PASS_FLOW_SCREEN"
    )
    assert (
        flow_screen_status(
            full_sample_betas=(0.1, 0.2, 0.3, -0.1, -0.2),
            year_2021_betas=(0.1, 0.2, 0.3, -0.1, -0.2),
            year_2022_betas=(0.1, 0.2, 0.3, -0.1, -0.2),
        )
        == "NO_STABLE_FLOW_SIGNAL"
    )


def test_protocol_rejects_result_dependent_mutations() -> None:
    protocol = canonical_aggtrades_flow_predictive_diagnostic_protocol()

    mutations = (
        {"sample_month_days": (2,)},
        {"label_horizon_hours": 2},
        {"min_valid_observations": 479},
        {"full_sample_positive_symbols_required": 3},
        {"year_positive_symbols_required": 2},
        {"predictor_formula": "sell_minus_buy_over_total_taker_notional"},
        {"regression_intercept": False},
    )
    for mutation in mutations:
        with pytest.raises(ValueError, match="preregistered"):
            replace(protocol, **mutation)


def test_strict_json_roundtrip_and_no_strategy_result_fields(tmp_path) -> None:
    protocol = canonical_aggtrades_flow_predictive_diagnostic_protocol()
    payload = protocol.to_payload()

    forbidden = {
        "strategy_return",
        "candidate_return",
        "pnl",
        "sharpe",
        "winner",
        "evaluation_result",
    }
    assert forbidden.isdisjoint(payload)

    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    loaded = load_aggtrades_flow_predictive_diagnostic_protocol(path)
    assert loaded == protocol
    assert loaded.digest == protocol.digest

    bad = dict(payload)
    bad["pnl"] = 1.0
    path.write_text(json.dumps(bad, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="keys differ"):
        load_aggtrades_flow_predictive_diagnostic_protocol(path)
