from __future__ import annotations

import json

import pytest

from tools.tmp_issue601_target_preflight import (
    archive_url,
    assert_result_blind,
    build_manifest,
    build_outputs,
    canonical_manifest_bytes,
)
from trade_rl.evaluation.experiments.bootstrap.issue601_target_source_validator import (
    TARGET_MONTHS,
    TARGET_SYMBOLS,
    validate_target_archive_bytes,
)


def _unavailable_validations():
    return [
        validate_target_archive_bytes(
            symbol=symbol,
            month=month,
            archive_bytes=None,
            checksum_bytes=None,
        )
        for symbol in TARGET_SYMBOLS
        for month in TARGET_MONTHS
    ]


def test_archive_url_is_exact_frozen_monthly_vision_url() -> None:
    assert archive_url("BTCUSDT", "2021-01") == (
        "https://data.binance.vision/data/futures/um/monthly/klines/"
        "BTCUSDT/1h/BTCUSDT-1h-2021-01.zip"
    )
    with pytest.raises(ValueError):
        archive_url("DOGEUSDT", "2021-01")
    with pytest.raises(ValueError):
        archive_url("BTCUSDT", "2023-01")


def test_manifest_is_deterministic_exact_roster_and_result_blind() -> None:
    validations = _unavailable_validations()
    manifest = build_manifest(validations)
    raw = canonical_manifest_bytes(manifest)
    assert raw == canonical_manifest_bytes(manifest)
    restored = json.loads(raw)
    assert restored["planned_archive_count"] == 120
    assert restored["symbols"] == list(TARGET_SYMBOLS)
    assert restored["months"] == list(TARGET_MONTHS)
    assert len(restored["records"]) == 120
    assert restored["records"][0]["symbol"] == "BTCUSDT"
    assert restored["records"][0]["month"] == "2021-01"
    assert restored["records"][-1]["symbol"] == "ADAUSDT"
    assert restored["records"][-1]["month"] == "2022-12"
    assert_result_blind(restored)


def test_build_outputs_on_unavailable_fixture_is_canonical_partial_and_blind() -> None:
    report_raw, manifest_raw = build_outputs(_unavailable_validations())
    report = json.loads(report_raw)
    manifest = json.loads(manifest_raw)
    assert report["status"] == "PARTIAL_USDM_1H_TARGET_SOURCE"
    assert report["planned_archive_count"] == 120
    assert report["economic_values_inspected"] is False
    assert report["target_price_values_inspected"] is False
    assert report["target_return_computed"] is False
    assert report["premium_values_inspected"] is False
    assert manifest["economic_values_inspected"] is False
    assert manifest["target_return_computed"] is False
    assert_result_blind(report)
    assert_result_blind(manifest)


def test_manifest_rejects_roster_and_digest_forgery() -> None:
    manifest = build_manifest(_unavailable_validations())
    missing = dict(manifest)
    missing["records"] = list(manifest["records"])[1:]
    with pytest.raises(ValueError):
        canonical_manifest_bytes(missing)

    forged = dict(manifest)
    forged["validator_head"] = "0" * 40
    with pytest.raises(ValueError):
        canonical_manifest_bytes(forged)

    digest_forged = dict(manifest)
    digest_forged["content_digest"] = "0" * 64
    with pytest.raises(ValueError):
        canonical_manifest_bytes(digest_forged)


def test_result_blind_guard_rejects_economic_keys_recursively() -> None:
    assert_result_blind({"entries": [{"row_count": 1, "schema_valid": True}]})
    with pytest.raises(ValueError):
        assert_result_blind({"entries": [{"close": 1.0}]})
    with pytest.raises(ValueError):
        assert_result_blind({"nested": {"beta": -1.0}})
