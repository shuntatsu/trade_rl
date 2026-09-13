from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from trade_rl.evaluation.experiments.bootstrap.agg_flow_capacity import (
    AggTradesFlowCapacityProtocol,
    canonical_m2_aggtrades_flow_capacity_protocol,
    load_aggtrades_flow_capacity_protocol,
)


def test_canonical_protocol_is_frozen_before_numeric_flow() -> None:
    protocol = canonical_m2_aggtrades_flow_capacity_protocol()

    assert protocol.canonical_dataset_id == (
        "d7a04ede97a1bb37b811c3e071f325fa007525a6040927e6793d8cc7c10f538f"
    )
    assert protocol.canonical_dataset_artifact_digest == (
        "77362e148c713840dda64e0ef70e663cce6611407eac31fefbb9fccca73ae8f8"
    )
    assert protocol.canonical_study_digest == (
        "3d8404061a4082a8e9b3c786d9f5fc9a4347631dff39c201e3cba70470dfeb79"
    )
    assert protocol.provider_head_sha == "dd51da97845f4d8fe69c34b1c4e4859358911daf"
    assert protocol.provider_parser_blob_sha == (
        "15cfe8f63716451fdb8c08ae84fba66ca754c55e"
    )
    assert protocol.source_validation_run_id == 34758933034
    assert protocol.source_validation_artifact_id == 10318765553
    assert protocol.source_validation_artifact_digest == (
        "3abd6592169ace3f308059fd524199d6fcf76c38862d0caec5e58192f6c7f7a3"
    )
    assert protocol.source_validation_manifest_sha256 == (
        "9ebffe78c29dbe336e5ed1323b686f41fae56ef83c50104fae3b8e965eb01f0a"
    )
    assert protocol.causal_capacity_head_sha == (
        "db8193c81dd0ca334a15ffc1895d75884238f8a9"
    )
    assert protocol.processing_bar_volume_capacity is False
    assert protocol.market == "usds-m"
    assert protocol.symbols == (
        "BTCUSDT",
        "ETHUSDT",
        "BNBUSDT",
        "XRPUSDT",
        "ADAUSDT",
    )
    assert protocol.calibration_start == datetime(2021, 1, 1, tzinfo=UTC)
    assert protocol.calibration_stop_exclusive == datetime(2023, 1, 1, tzinfo=UTC)
    assert protocol.evaluation_start == datetime(2023, 1, 1, tzinfo=UTC)
    assert protocol.sample_month_days == (1,)
    assert protocol.burst_window_milliseconds == 5_000
    assert protocol.lower_tail_quantile == 0.10
    assert protocol.utilization_fraction == 0.25
    assert protocol.hard_ceiling == 0.05
    assert protocol.min_valid_days_per_symbol == 20
    assert protocol.min_valid_days_per_year == 10
    assert protocol.min_valid_hours_per_symbol == 480
    assert protocol.checksum_required is True
    assert protocol.checksum_suffix == ".CHECKSUM"
    assert protocol.buyer_taker_when_buyer_is_maker is False
    assert protocol.seller_taker_when_buyer_is_maker is True
    assert protocol.archive_url_template == (
        "https://data.binance.vision/data/futures/um/daily/aggTrades/"
        "{symbol}/{symbol}-aggTrades-{date}.zip"
    )
    assert protocol.trade_notional_formula == "price_times_quantity"
    assert protocol.hour_alignment == "utc_hour"
    assert protocol.burst_bin_alignment == "utc_epoch_floor_5000ms"
    assert protocol.fraction_formula == "min_peak_buy_sell_5s_over_hour_total"
    assert protocol.order_statistic_rule == "ceil_qn_minus_one"
    assert len(protocol.digest) == 64

    planned_days = protocol.planned_days
    assert len(planned_days) == 24
    assert planned_days[:3] == (
        datetime(2021, 1, 1, tzinfo=UTC),
        datetime(2021, 2, 1, tzinfo=UTC),
        datetime(2021, 3, 1, tzinfo=UTC),
    )
    assert planned_days[-1] == datetime(2022, 12, 1, tzinfo=UTC)

    planned_urls = protocol.planned_urls
    assert len(planned_urls) == 120
    assert len(set(planned_urls)) == 120
    assert planned_urls[0].endswith("/BTCUSDT/BTCUSDT-aggTrades-2021-01-01.zip")
    assert planned_urls[-1].endswith("/ADAUSDT/ADAUSDT-aggTrades-2022-12-01.zip")
    assert all("2023-" not in url for url in planned_urls)


def test_protocol_rejects_semantic_drift_and_post_evaluation_data() -> None:
    protocol = canonical_m2_aggtrades_flow_capacity_protocol()

    with pytest.raises(ValueError, match="evaluation_start"):
        replace(
            protocol,
            calibration_stop_exclusive=datetime(2023, 1, 2, tzinfo=UTC),
        )

    mutations = (
        {"provider_head_sha": "0" * 40},
        {"provider_parser_blob_sha": "0" * 40},
        {"source_validation_run_id": 1},
        {"source_validation_artifact_id": 1},
        {"source_validation_artifact_digest": "0" * 64},
        {"source_validation_manifest_sha256": "0" * 64},
        {"causal_capacity_head_sha": "0" * 40},
        {"processing_bar_volume_capacity": True},
        {"market": "spot"},
        {"symbols": ("BTCUSDT",)},
        {"sample_month_days": (1, 15)},
        {"burst_window_milliseconds": 1_000},
        {"lower_tail_quantile": 0.05},
        {"utilization_fraction": 0.50},
        {"hard_ceiling": 0.10},
        {"min_valid_days_per_symbol": 19},
        {"min_valid_days_per_year": 9},
        {"min_valid_hours_per_symbol": 479},
        {"checksum_required": False},
        {"checksum_suffix": ".sha256"},
        {"buyer_taker_when_buyer_is_maker": True},
        {"seller_taker_when_buyer_is_maker": False},
        {"archive_url_template": "https://example.invalid/{symbol}/{date}.zip"},
        {"trade_notional_formula": "quantity_only"},
        {"hour_alignment": "rolling_hour"},
        {"burst_bin_alignment": "data_dependent"},
        {"fraction_formula": "peak_buy_only"},
        {"order_statistic_rule": "linear_interpolation"},
    )
    for mutation in mutations:
        with pytest.raises(ValueError, match="preregistered"):
            replace(protocol, **mutation)


def test_order_statistic_rank_is_exact_and_interpolation_free() -> None:
    protocol = canonical_m2_aggtrades_flow_capacity_protocol()

    assert protocol.lower_tail_rank(1) == 0
    assert protocol.lower_tail_rank(10) == 0
    assert protocol.lower_tail_rank(11) == 1
    assert protocol.lower_tail_rank(480) == 47
    with pytest.raises(ValueError, match="positive"):
        protocol.lower_tail_rank(0)


def test_payload_is_strict_and_contains_no_pnl_inputs(tmp_path) -> None:
    protocol = canonical_m2_aggtrades_flow_capacity_protocol()
    payload = protocol.to_payload()

    forbidden = {
        "returns",
        "pnl",
        "strategy",
        "decision",
        "experiment_result",
        "candidate_result",
        "q10",
        "symbol_cap",
    }
    assert forbidden.isdisjoint(payload)

    path = tmp_path / "protocol.json"
    path.write_text(
        json.dumps(payload, allow_nan=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    loaded = load_aggtrades_flow_capacity_protocol(path)
    assert isinstance(loaded, AggTradesFlowCapacityProtocol)
    assert loaded == protocol
    assert loaded.digest == protocol.digest

    tampered = dict(payload)
    tampered["pnl"] = 1.0
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="keys differ"):
        load_aggtrades_flow_capacity_protocol(path)


def test_loader_rejects_nonfinite_and_naive_boundaries(tmp_path) -> None:
    protocol = canonical_m2_aggtrades_flow_capacity_protocol()
    payload = protocol.to_payload()
    path = tmp_path / "protocol.json"

    nonfinite = dict(payload)
    nonfinite["utilization_fraction"] = float("nan")
    path.write_text(json.dumps(nonfinite, allow_nan=True), encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        load_aggtrades_flow_capacity_protocol(path)

    with pytest.raises(ValueError, match="timezone-aware"):
        replace(protocol, calibration_start=datetime(2021, 1, 1))
