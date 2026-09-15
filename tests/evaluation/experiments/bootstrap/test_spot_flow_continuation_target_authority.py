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
TARGET_PREFLIGHT_RUN_ID = 34953161630
TARGET_PREFLIGHT_PUBLISHER_ARTIFACT_ID = 10389544902
TARGET_PREFLIGHT_PUBLISHER_DIGEST = (
    "2a1f1d7d700fc662862ebab5b41ff692d154dbfef793fb505ce7e1ea5647e2c2"
)
TARGET_PREFLIGHT_FRESH_ARTIFACT_ID = 10389694422
TARGET_PREFLIGHT_FRESH_DIGEST = (
    "8b5cbd19abb9517bf4885f3b9432b29024069f0005b8d18e89b82b2471765e3d"
)
TARGET_PREFLIGHT_REPORT_SHA256 = (
    "721c785ddac113f7d6be2b5af3e1aa10df34ae9a16b9656e02a64c874f4526ea"
)
TARGET_PREFLIGHT_REPORT_CONTENT_DIGEST = (
    "bd32c2e8fb9b4b80680fdfb0b26ca4b7fd6f01104597cba8c81990faa4f312d1"
)
TARGET_MANIFEST_DIGEST = (
    "69a4105e8839d38972a034b49c9a3d019d3bc330981fce7cb860276e9ca0b25b"
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

    assert result.target_preflight_status == "PASS_USDM_15M_TARGET_SOURCE"
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
