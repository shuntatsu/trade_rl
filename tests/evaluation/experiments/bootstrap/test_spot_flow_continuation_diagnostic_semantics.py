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


def _artifact_payload() -> dict[str, object]:
    protocol = canonical_spot_flow_continuation_protocol()
    observations = {symbol: ([1.0] * 360, [0.01] * 360) for symbol in SYMBOLS}
    result = diagnostic.build_spot_flow_diagnostic_result(
        observations,
        protocol,
        implementation_head="1" * 40,
        source_manifest_digest="2" * 64,
        target_manifest_digest="69a4105e8839d38972a034b49c9a3d019d3bc330981fce7cb860276e9ca0b25b",
        target_preflight_run_id=34953161630,
        target_preflight_artifact_id=10389544902,
        target_preflight_artifact_api_digest="2a1f1d7d700fc662862ebab5b41ff692d154dbfef793fb505ce7e1ea5647e2c2",
        execution_run_id=789,
    )
    return json.loads(diagnostic.canonical_spot_flow_result_bytes(result))


def _resign(payload: dict[str, object]) -> bytes:
    unsigned = dict(payload)
    unsigned.pop("content_digest", None)
    unsigned["content_digest"] = content_digest(unsigned)
    return canonical_json_bytes(unsigned)


def test_loader_rejects_beta_not_equal_to_published_numerator_over_denominator() -> (
    None
):
    payload = _artifact_payload()
    symbol_results = payload["symbol_results"]
    assert isinstance(symbol_results, list)
    first = dict(symbol_results[0])
    first["beta"] = 0.5
    first["positive_slope"] = True
    symbol_results[0] = first

    with pytest.raises(ValueError, match="beta|numerator|denominator|semantic"):
        diagnostic.load_spot_flow_result_bytes(_resign(payload))


def test_loader_rejects_coverage_forgery_even_with_self_consistent_digest() -> None:
    payload = _artifact_payload()
    symbol_results = payload["symbol_results"]
    assert isinstance(symbol_results, list)
    first = dict(symbol_results[0])
    first["eligible_observations"] = 359
    symbol_results[0] = first

    with pytest.raises(ValueError, match="coverage|eligible|beta|failure|semantic"):
        diagnostic.load_spot_flow_result_bytes(_resign(payload))


def test_symbol_result_rejects_failure_list_that_does_not_match_numeric_semantics() -> (
    None
):
    with pytest.raises(ValueError, match="failure|coverage|semantic"):
        diagnostic.SpotFlowSymbolCalibration(
            symbol="BTCUSDT",
            eligible_observations=360,
            numerator=3.6,
            denominator=360.0,
            beta=0.01,
            positive_slope=True,
            failures=("BTCUSDT:eligible_observations<360",),
        )
