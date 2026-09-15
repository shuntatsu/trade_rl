from __future__ import annotations

from pathlib import Path

TEST_FILE = Path("tests/evaluation/experiments/bootstrap/test_premium_pressure_calibration.py")

text = TEST_FILE.read_text()

replacements = {
    "eligible_observations=16_620,": "eligible_observations=16_621,",
    "eligible_observations=16_619,": "eligible_observations=16_620,",
    "eligible_observations<16620": "eligible_observations<16621",
    "da97eb59b08ef042a96ebea4aaa645af02b450e9": "6ffc414baa10e91f34df258fe5cfabacce65fd77",
    "c78d556f94e247d5bdc1c0e9a891d160e4c48c4f0b62cd5be7600874b3017d22": "18bf9625502eccbe475d157df90635f1c4645d3c4cced48ec0fc565723206731",
    "34961624663": "34962227821",
    "402f45a99ffc570a3f9c13427cb95db1b8cf1c99a0a3c996d73e68b307f7b2a1": "a9bfe64d5232747ac2ee543e3b65a4be22d4d034969603040391b7b0eabec988",
}
for old, new in replacements.items():
    count = text.count(old)
    if count < 1:
        raise SystemExit(f"expected RED token not found: {old}")
    text = text.replace(old, new)

authority_anchor = """    assert payload[\"target_source_run_id\"] == 34962227821
    assert payload[\"target_source_report_content_digest\"] == (
        \"a9bfe64d5232747ac2ee543e3b65a4be22d4d034969603040391b7b0eabec988\"
    )
"""
authority_replacement = """    assert payload[\"schema_version\"] == \"premium_pressure_calibration_result_v2\"
    assert payload[\"protocol_full_verify_run_id\"] == 34959430185
    assert payload[\"protocol_seal_run_id\"] == 34959849347
    assert payload[\"protocol_seal_artifact_id\"] == 10392936042
    assert payload[\"protocol_seal_artifact_api_digest\"] == (
        \"4b611462f8d4f9bb4d23ab15bb4d8a20525d4093e590c1afb62d1b74ca875e15\"
    )
    assert payload[\"protocol_fresh_artifact_id\"] == 10391714897
    assert payload[\"protocol_fresh_artifact_api_digest\"] == (
        \"858e930a03e442f047556801c3d9a536fbc89435b5f261cba621988dd8062acd\"
    )
    assert payload[\"premium_source_issue\"] == 570
    assert payload[\"premium_source_probe_head\"] == (
        \"80b65a773c622af530f074888eb01fff2f7f98df\"
    )
    assert payload[\"target_source_issue\"] == 601
    assert payload[\"target_source_validator_head\"] == (
        \"c707be853480e57362f6a3d64f116e4056c3553f\"
    )
    assert payload[\"target_source_validator_verification_run_id\"] == 34961165095
    assert payload[\"target_source_preflight_head\"] == (
        \"e64256dd40e76d9606edfbdebe1c37b86668d467\"
    )
    assert payload[\"target_source_run_id\"] == 34962227821
    assert payload[\"target_source_status\"] == \"PASS_USDM_1H_TARGET_SOURCE\"
    assert payload[\"target_source_publisher_artifact_id\"] == 10393693240
    assert payload[\"target_source_publisher_artifact_api_digest\"] == (
        \"0c7f240390a35174329890fa83e9615e304de5f1a84114007ead1fddaaaf37ff\"
    )
    assert payload[\"target_source_fresh_artifact_id\"] == 10393379932
    assert payload[\"target_source_fresh_artifact_api_digest\"] == (
        \"0c0c05b8f5256c918b90ae709ff05ce0f05f8e88a929f1820387cf60a12dfeff\"
    )
    assert payload[\"target_source_report_sha256\"] == (
        \"6833311f14e45ce634dd4ba13fdf8defeaa0b1a2b90cf7ff0598ee2054bde151\"
    )
    assert payload[\"target_source_report_content_digest\"] == (
        \"a9bfe64d5232747ac2ee543e3b65a4be22d4d034969603040391b7b0eabec988\"
    )
    assert payload[\"target_source_manifest_sha256\"] == (
        \"e287f03bf18619834108b5b452c26cdffbc66a6be3d51d418c928daa2e77dd15\"
    )
    assert payload[\"target_source_manifest_content_digest\"] == (
        \"9ced3fbab6e51fbd654768ec6a4ea28f98cc4b9f36c1d9d2a9a8f6540864934e\"
    )
"""
if text.count(authority_anchor) != 1:
    raise SystemExit("authority assertion anchor changed")
text = text.replace(authority_anchor, authority_replacement)

if "test_coverage_boundary_16620_invalid_16621_coverage_valid" in text:
    raise SystemExit("current RED additions already present")

text += r'''


def test_missing_target_timestamp_invalidates_only_affected_disjoint_label() -> None:
    first = _decision()
    second = first + timedelta(hours=48)
    premium = (_premium_point(first), _premium_point(second))
    target = tuple(
        sorted(
            (*_target_window(first), *_target_window(second)),
            key=lambda item: item.dataset_time_ms,
        )
    )
    assert len(build_training_pairs(premium, target)) == 2

    removed_raw_open = _ms(first + timedelta(hours=7))
    missing = tuple(item for item in target if item.raw_open_time_ms != removed_raw_open)
    surviving = build_training_pairs(premium, missing)
    assert len(surviving) == 1
    assert surviving[0].decision_time_ms == _ms(second)


def test_coverage_boundary_16620_invalid_16621_coverage_valid() -> None:
    n = canonical_premium_pressure_protocol().minimum_eligible_observations_per_symbol
    assert n == 16_621
    valid_pairs = tuple(
        TrainingPair(
            decision_time_ms=index * _HOUR_MS,
            x=float(index % 7),
            y=-float(index % 7),
        )
        for index in range(n)
    )
    valid = calibrate_symbol("BTCUSDT", valid_pairs)
    assert valid.eligible_observations == 16_621
    assert valid.failures == ()

    invalid = calibrate_symbol("BTCUSDT", valid_pairs[:-1])
    assert invalid.eligible_observations == 16_620
    assert invalid.failures == ("BTCUSDT:eligible_observations<16621",)
    assert invalid.beta is None


def test_symbol_calibration_rejects_resigned_semantic_forgery_at_construction() -> None:
    with pytest.raises(ValueError):
        PremiumPressureSymbolCalibration(
            symbol="BTCUSDT",
            eligible_observations=16_621,
            x_bar=None,
            y_bar=None,
            numerator=None,
            denominator=None,
            alpha=None,
            beta=None,
            negative_slope=False,
            failures=("BTCUSDT:eligible_observations<16621",),
        )

    with pytest.raises(ValueError):
        PremiumPressureSymbolCalibration(
            symbol="BTCUSDT",
            eligible_observations=16_621,
            x_bar=1.0,
            y_bar=2.0,
            numerator=-2.0,
            denominator=2.0,
            alpha=3.0,
            beta=-1.0,
            negative_slope=False,
            failures=(),
        )


def test_loader_rejects_missing_field_digest_tamper_and_resigned_nested_forgery() -> None:
    payload = _valid_result().to_dict()

    missing = json.loads(json.dumps(payload))
    missing.pop("target_source_manifest_content_digest")
    missing.pop("content_digest")
    missing["content_digest"] = content_digest(missing)
    with pytest.raises(ValueError):
        load_premium_pressure_result_bytes(
            json.dumps(missing, sort_keys=True, separators=(",", ":")).encode()
        )

    bad_digest = json.loads(json.dumps(payload))
    bad_digest["content_digest"] = "0" * 64
    with pytest.raises(ValueError):
        load_premium_pressure_result_bytes(
            json.dumps(bad_digest, sort_keys=True, separators=(",", ":")).encode()
        )

    forged = json.loads(json.dumps(payload))
    forged["symbol_results"][0]["negative_slope"] = not forged["symbol_results"][0][
        "negative_slope"
    ]
    forged.pop("content_digest")
    forged["content_digest"] = content_digest(forged)
    with pytest.raises(ValueError):
        load_premium_pressure_result_bytes(
            json.dumps(forged, sort_keys=True, separators=(",", ":")).encode()
        )

    authority = json.loads(json.dumps(payload))
    authority["target_source_publisher_artifact_api_digest"] = "0" * 64
    authority.pop("content_digest")
    authority["content_digest"] = content_digest(authority)
    with pytest.raises(ValueError):
        load_premium_pressure_result_bytes(
            json.dumps(authority, sort_keys=True, separators=(",", ":")).encode()
        )
'''

TEST_FILE.write_text(text)
