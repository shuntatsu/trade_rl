from __future__ import annotations

import json

import pytest

from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.experiments.bootstrap import (
    spot_flow_continuation_diagnostic as diagnostic,
)
from trade_rl.evaluation.experiments.bootstrap.spot_flow_continuation_prereg import (
    canonical_spot_flow_continuation_protocol,
)

SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
TARGET_PREFLIGHT_RUN_ID = 34941447564
TARGET_PREFLIGHT_PUBLISHER_ARTIFACT_ID = 10385451448
TARGET_PREFLIGHT_PUBLISHER_DIGEST = (
    "07ee6cbaee5ae36d5c564d8559f174e95e2fe63a2e68c7938fe84190e3b03ebb"
)
TARGET_PREFLIGHT_FRESH_ARTIFACT_ID = 10385516382
TARGET_PREFLIGHT_FRESH_DIGEST = (
    "7d02df596647ad109f3f049ec884757f50689f82f65b27f9de795ecfa88f5e7d"
)
TARGET_PREFLIGHT_REPORT_SHA256 = (
    "c8c1c33704a72b9a8bb1ca96d143bf7883ff4ebf1aeeab3d30f7fe8568ae4b6d"
)
TARGET_PREFLIGHT_REPORT_CONTENT_DIGEST = (
    "5a7af640640b23894f857726944a306aef522799ca8c983757e864d76262d54d"
)
TARGET_MANIFEST_DIGEST = (
    "69a4105e8839d38972a034b49c9a3d019d3bc330981fce7cb860276e9ca0b25b"
)
TARGET_SOURCE_STATUS = "PASS_USDM_15M_TARGET_SOURCE"
TARGET_SOURCE_VALIDATOR_HEAD = "12151cb014f62f603fc115b3129d14264eb93683"
TARGET_SOURCE_VALIDATOR_FULL_RUN_ID = 34951770631
TARGET_SOURCE_REBIND_RUN_ID = 34953161630
TARGET_SOURCE_REBIND_ARTIFACT_ID = 10389544902
TARGET_SOURCE_REBIND_ARTIFACT_DIGEST = (
    "2a1f1d7d700fc662862ebab5b41ff692d154dbfef793fb505ce7e1ea5647e2c2"
)
TARGET_SOURCE_REBIND_FRESH_ARTIFACT_ID = 10389694422
TARGET_SOURCE_REBIND_FRESH_ARTIFACT_DIGEST = (
    "8b5cbd19abb9517bf4885f3b9432b29024069f0005b8d18e89b82b2471765e3d"
)


def _build() -> diagnostic.SpotFlowDiagnosticResult:
    protocol = canonical_spot_flow_continuation_protocol()
    observations = {symbol: ([1.0] * 360, [0.01] * 360) for symbol in SYMBOLS}
    return diagnostic.build_spot_flow_diagnostic_result(
        observations,
        protocol,
        implementation_head="1" * 40,
        source_manifest_digest="2" * 64,
        target_manifest_digest=TARGET_MANIFEST_DIGEST,
        target_preflight_run_id=TARGET_PREFLIGHT_RUN_ID,
        target_preflight_artifact_id=TARGET_PREFLIGHT_PUBLISHER_ARTIFACT_ID,
        target_preflight_artifact_api_digest=TARGET_PREFLIGHT_PUBLISHER_DIGEST,
        execution_run_id=789,
    )


def test_result_binds_complete_target_preflight_authority() -> None:
    result = _build()

    assert result.target_preflight_status == "PASS_TARGET_SOURCE_PREFLIGHT"
    assert result.target_preflight_run_id == TARGET_PREFLIGHT_RUN_ID
    assert (
        result.target_preflight_publisher_artifact_id
        == TARGET_PREFLIGHT_PUBLISHER_ARTIFACT_ID
    )
    assert (
        result.target_preflight_publisher_artifact_api_digest
        == TARGET_PREFLIGHT_PUBLISHER_DIGEST
    )
    assert (
        result.target_preflight_fresh_artifact_id == TARGET_PREFLIGHT_FRESH_ARTIFACT_ID
    )
    assert (
        result.target_preflight_fresh_artifact_api_digest
        == TARGET_PREFLIGHT_FRESH_DIGEST
    )
    assert result.target_preflight_report_sha256 == TARGET_PREFLIGHT_REPORT_SHA256
    assert (
        result.target_preflight_report_content_digest
        == TARGET_PREFLIGHT_REPORT_CONTENT_DIGEST
    )
    assert result.target_manifest_digest == TARGET_MANIFEST_DIGEST


def test_result_binds_hardened_target_source_rebind_authority() -> None:
    result = _build()

    assert result.schema_version == "spot_flow_continuation_diagnostic_v2"
    assert result.target_source_status == TARGET_SOURCE_STATUS
    assert result.target_source_validator_head == TARGET_SOURCE_VALIDATOR_HEAD
    assert (
        result.target_source_validator_full_verify_run_id
        == TARGET_SOURCE_VALIDATOR_FULL_RUN_ID
    )
    assert result.target_source_rebind_run_id == TARGET_SOURCE_REBIND_RUN_ID
    assert (
        result.target_source_rebind_artifact_id == TARGET_SOURCE_REBIND_ARTIFACT_ID
    )
    assert (
        result.target_source_rebind_artifact_api_digest
        == TARGET_SOURCE_REBIND_ARTIFACT_DIGEST
    )
    assert (
        result.target_source_rebind_fresh_artifact_id
        == TARGET_SOURCE_REBIND_FRESH_ARTIFACT_ID
    )
    assert (
        result.target_source_rebind_fresh_artifact_api_digest
        == TARGET_SOURCE_REBIND_FRESH_ARTIFACT_DIGEST
    )


def test_builder_rejects_noncanonical_target_preflight_authority() -> None:
    protocol = canonical_spot_flow_continuation_protocol()
    observations = {symbol: ([1.0] * 360, [0.01] * 360) for symbol in SYMBOLS}
    with pytest.raises(ValueError, match="target.*authority|canonical|manifest"):
        diagnostic.build_spot_flow_diagnostic_result(
            observations,
            protocol,
            implementation_head="1" * 40,
            source_manifest_digest="2" * 64,
            target_manifest_digest="9" * 64,
            target_preflight_run_id=TARGET_PREFLIGHT_RUN_ID,
            target_preflight_artifact_id=TARGET_PREFLIGHT_PUBLISHER_ARTIFACT_ID,
            target_preflight_artifact_api_digest=TARGET_PREFLIGHT_PUBLISHER_DIGEST,
            execution_run_id=789,
        )


def test_loader_rejects_resigned_target_authority_forgery() -> None:
    payload = json.loads(diagnostic.canonical_spot_flow_result_bytes(_build()))
    payload["target_preflight_fresh_artifact_id"] = 1
    unsigned = dict(payload)
    unsigned.pop("content_digest", None)
    payload["content_digest"] = content_digest(unsigned)

    with pytest.raises(ValueError, match="target.*authority|canonical"):
        diagnostic.load_spot_flow_result_bytes(canonical_json_bytes(payload))


def test_loader_rejects_resigned_target_source_rebind_forgery() -> None:
    payload = json.loads(diagnostic.canonical_spot_flow_result_bytes(_build()))
    payload["target_source_rebind_artifact_id"] = 1
    unsigned = dict(payload)
    unsigned.pop("content_digest", None)
    payload["content_digest"] = content_digest(unsigned)

    with pytest.raises(ValueError, match="target.*authority|canonical"):
        diagnostic.load_spot_flow_result_bytes(canonical_json_bytes(payload))
