from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.experiments.bootstrap.perp_index_basis_calibration import (
    PerpIndexBasisCalibrationResult,
    PerpIndexBasisSymbolCalibration,
    load_perp_index_basis_calibration_result,
)
from trade_rl.evaluation.experiments.bootstrap.perp_index_basis_prereg import (
    canonical_perp_index_basis_protocol,
)


def _baseline_artifact() -> dict[str, object]:
    protocol = canonical_perp_index_basis_protocol()
    betas = (-0.60, -0.50, -0.40, -0.30, 0.20)
    symbol_results = tuple(
        PerpIndexBasisSymbolCalibration(
            symbol=symbol,
            eligible_observations=8_760,
            numerator=2.0 * beta,
            denominator=2.0,
            beta=beta,
            negative_slope=beta < 0.0,
        )
        for symbol, beta in zip(protocol.symbols, betas, strict=True)
    )
    result = PerpIndexBasisCalibrationResult(
        protocol_digest=protocol.digest,
        dataset_id="1" * 64,
        symbols=protocol.symbols,
        symbol_results=symbol_results,
        negative_slope_count=4,
        status=protocol.valid_status,
        failures=(),
        calibration_head="4" * 40,
        source_manifest_digest="5" * 64,
    )
    return result.to_artifact_payload()


def _write_rehashed(path: Path, artifact: dict[str, object]) -> None:
    payload = deepcopy(artifact)
    payload.pop("content_digest")
    payload["content_digest"] = content_digest(payload)
    path.write_bytes(canonical_json_bytes(payload))


def test_loader_rejects_rehashed_low_coverage_artifact(tmp_path: Path) -> None:
    artifact = _baseline_artifact()
    results = artifact["symbol_results"]
    assert isinstance(results, list)
    first = results[0]
    assert isinstance(first, dict)
    first["eligible_observations"] = 1
    path = tmp_path / "low-coverage.json"
    _write_rehashed(path, artifact)

    with pytest.raises(ValueError, match="eligible|coverage|failure|semantic"):
        load_perp_index_basis_calibration_result(path)


def test_loader_rejects_rehashed_missing_beta_artifact(tmp_path: Path) -> None:
    artifact = _baseline_artifact()
    results = artifact["symbol_results"]
    assert isinstance(results, list)
    last = results[-1]
    assert isinstance(last, dict)
    last["beta"] = None
    path = tmp_path / "missing-beta.json"
    _write_rehashed(path, artifact)

    with pytest.raises(ValueError, match="beta|numeric|failure|semantic"):
        load_perp_index_basis_calibration_result(path)


def test_loader_rejects_rehashed_beta_sign_forgery(tmp_path: Path) -> None:
    artifact = _baseline_artifact()
    results = artifact["symbol_results"]
    assert isinstance(results, list)
    last = results[-1]
    assert isinstance(last, dict)
    last["beta"] = -0.20
    last["negative_slope"] = True
    artifact["negative_slope_count"] = 5
    path = tmp_path / "forged-beta.json"
    _write_rehashed(path, artifact)

    with pytest.raises(ValueError, match="beta|numerator|denominator|semantic"):
        load_perp_index_basis_calibration_result(path)


def test_loader_rejects_rehashed_forged_failures_and_status(tmp_path: Path) -> None:
    protocol = canonical_perp_index_basis_protocol()
    artifact = _baseline_artifact()
    artifact["failures"] = ["BTCUSDT:eligible_observations<8760"]
    artifact["status"] = protocol.invalid_coverage_status
    path = tmp_path / "forged-failures.json"
    _write_rehashed(path, artifact)

    with pytest.raises(ValueError, match="failure|coverage|semantic"):
        load_perp_index_basis_calibration_result(path)
