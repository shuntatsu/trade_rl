from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from trade_rl.evaluation.experiments.bootstrap import (
    BookDepthCapacityCalibrationProtocol,
    canonical_m2_book_depth_capacity_protocol,
    load_book_depth_capacity_calibration_protocol,
)


def test_canonical_book_depth_capacity_protocol_is_frozen_before_pnl() -> None:
    protocol = canonical_m2_book_depth_capacity_protocol()

    assert protocol.canonical_dataset_id == (
        "d7a04ede97a1bb37b811c3e071f325fa007525a6040927e6793d8cc7c10f538f"
    )
    assert protocol.canonical_dataset_artifact_digest == (
        "77362e148c713840dda64e0ef70e663cce6611407eac31fefbb9fccca73ae8f8"
    )
    assert protocol.canonical_study_digest == (
        "3d8404061a4082a8e9b3c786d9f5fc9a4347631dff39c201e3cba70470dfeb79"
    )
    assert protocol.market == "usds-m"
    assert protocol.reference_volume_timeframe == "1h"
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
    assert protocol.sample_month_days == (1, 15)
    assert protocol.capacity_bands == (-1, 1)
    assert protocol.quantile == 0.05
    assert protocol.book_utilization_fraction == 0.10
    assert protocol.participation_ceiling == 0.05
    assert protocol.min_valid_days_per_symbol == 30
    assert protocol.volume_alignment == "previous_completed_1h"
    assert len(protocol.digest) == 64

    planned = protocol.planned_days
    assert len(planned) == 48
    assert planned[:4] == (
        datetime(2021, 1, 1, tzinfo=UTC),
        datetime(2021, 1, 15, tzinfo=UTC),
        datetime(2021, 2, 1, tzinfo=UTC),
        datetime(2021, 2, 15, tzinfo=UTC),
    )
    assert planned[-1] == datetime(2022, 12, 15, tzinfo=UTC)


def test_protocol_rejects_post_evaluation_calibration_and_semantic_drift() -> None:
    protocol = canonical_m2_book_depth_capacity_protocol()

    with pytest.raises(ValueError, match="evaluation_start"):
        replace(
            protocol,
            calibration_stop_exclusive=datetime(2023, 1, 2, tzinfo=UTC),
        )

    mutations = (
        {"market": "spot"},
        {"reference_volume_timeframe": "4h"},
        {"symbols": ("BTCUSDT",)},
        {"canonical_dataset_id": "0" * 64},
        {"canonical_dataset_artifact_digest": "0" * 64},
        {"canonical_study_digest": "0" * 64},
        {"sample_month_days": (1, 10)},
        {"capacity_bands": (-2, 2)},
        {"quantile": 0.10},
        {"book_utilization_fraction": 0.20},
        {"participation_ceiling": 0.10},
        {"min_valid_days_per_symbol": 20},
        {"volume_alignment": "same_hour"},
    )
    for mutation in mutations:
        with pytest.raises(ValueError, match="preregistered"):
            replace(protocol, **mutation)


def test_protocol_rejects_naive_time_boundary() -> None:
    protocol = canonical_m2_book_depth_capacity_protocol()

    with pytest.raises(ValueError, match="timezone-aware"):
        replace(protocol, calibration_start=datetime(2021, 1, 1))


def test_protocol_payload_is_strict_and_contains_no_pnl_inputs(tmp_path) -> None:
    protocol = canonical_m2_book_depth_capacity_protocol()
    payload = protocol.to_payload()

    forbidden = {
        "returns",
        "pnl",
        "strategy",
        "decision",
        "experiment_result",
    }
    assert forbidden.isdisjoint(payload)

    path = tmp_path / "protocol.json"
    path.write_text(
        json.dumps(payload, allow_nan=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    loaded = load_book_depth_capacity_calibration_protocol(path)
    assert isinstance(loaded, BookDepthCapacityCalibrationProtocol)
    assert loaded == protocol
    assert loaded.digest == protocol.digest

    tampered = dict(payload)
    tampered["pnl"] = 1.0
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="keys differ|unknown"):
        load_book_depth_capacity_calibration_protocol(path)


def test_protocol_loader_rejects_invalid_digest_and_nonfinite_number(tmp_path) -> None:
    payload = canonical_m2_book_depth_capacity_protocol().to_payload()
    path = tmp_path / "protocol.json"

    bad_digest = dict(payload)
    bad_digest["canonical_dataset_id"] = "not-a-digest"
    path.write_text(json.dumps(bad_digest), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        load_book_depth_capacity_calibration_protocol(path)

    nonfinite = dict(payload)
    nonfinite["quantile"] = float("nan")
    path.write_text(json.dumps(nonfinite, allow_nan=True), encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        load_book_depth_capacity_calibration_protocol(path)
