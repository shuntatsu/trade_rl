"""Synthetic, result-blind counterexamples for pinned R2 Dataset byte admission."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from tools.issue810_r2_dataset_source_preflight import (
    ORIGINAL_PINS,
    R2_DATASET_ID,
    R2_FEATURES,
    R2_ROWS,
    SourcePreflightError,
    _Pins,
    audit_dataset,
    publish_report,
)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, _Pins, dict[str, object]]:
    root = tmp_path / "original-input"
    root.mkdir()
    raw = b"artificial fixture: not market data, not a NumPy ZIP"
    (root / "arrays.npz").write_bytes(raw)
    dimensions = [R2_ROWS, 2, 6]
    record: dict[str, object] = {
        "schema_version": "market_dataset_artifact_v4",
        "dataset_id": R2_DATASET_ID,
        "arrays_file": "arrays.npz",
        "arrays_digest": _digest(raw),
        "arrays": {
            "timestamps": {"shape": [R2_ROWS], "dtype": "<M8[ns]"},
            "available_at": {"shape": dimensions[:2], "dtype": "<M8[ns]"},
            "features": {"shape": dimensions, "dtype": "<f4"},
            "feature_available": {"shape": dimensions, "dtype": "|b1"},
            "funding_price_rate": {"shape": dimensions[:2], "dtype": "<f8"},
        },
        "dataset": {
            "dataset_id": R2_DATASET_ID,
            "symbols": ["BTCUSDT", "ETHUSDT"],
            "feature_names": list(R2_FEATURES),
        },
    }
    record["artifact_digest"] = _digest(_canonical(record))
    manifest_bytes = _canonical(record)
    (root / "manifest.json").write_bytes(manifest_bytes)
    return root, _Pins(_digest(manifest_bytes), _digest(raw)), record


_PAIR_ERROR = "exactly manifest.json and arrays.npz"


def test_synthetic_identity_pass_still_refuses_source_and_economic_authority(
    tmp_path: Path,
) -> None:
    root, pins, _ = _fixture(tmp_path)
    result = audit_dataset(root, expected=pins)
    assert result["byte_identity"] == "SYNTHETIC_FIXTURE_ONLY"
    assert result["dataset_id"] == R2_DATASET_ID
    assert result["source_publication_time"] == "NOT_ESTABLISHED"
    assert result["source_ingestion_time"] == "NOT_ESTABLISHED"
    assert result["historical_point_in_time"] == "NOT_ESTABLISHED"
    assert result["task18_input_receipts"] == "NOT_VERIFIED"
    assert result["task18_to_r2_row_scope"] == "NOT_ESTABLISHED"
    assert result["market_result"] == "NOT_ESTABLISHED"
    assert result["economic_authority"] == "NOT_GRANTED"
    assert result["unused_or_live_authority"] == "NOT_GRANTED"
    assert result["arrays_payload_parsed"] is False
    assert result["r2_dimensions_declared"] == [9504, 2, 6]
    assert result["task18_witness_rows_reported"] == 8761


def test_original_pins_cannot_be_satisfied_by_renaming_synthetic_inputs(
    tmp_path: Path,
) -> None:
    root, _, _ = _fixture(tmp_path)
    with pytest.raises(SourcePreflightError, match="SHA256"):
        audit_dataset(root)
    fake_sha = _digest((root / "manifest.json").read_bytes())
    assert ORIGINAL_PINS.manifest_sha256 != fake_sha


@pytest.mark.parametrize("filename", ["manifest.json", "arrays.npz"])
def test_identical_length_byte_tamper_rejected(tmp_path: Path, filename: str) -> None:
    root, pins, _ = _fixture(tmp_path)
    p = root / filename
    original = p.read_bytes()
    p.write_bytes(b"Z" + original[1:])
    assert p.stat().st_size == len(original)
    with pytest.raises(SourcePreflightError, match="SHA256"):
        audit_dataset(root, expected=pins)


@pytest.mark.parametrize("filename", ["manifest.json", "arrays.npz"])
def test_missing_input_fail_closed(tmp_path: Path, filename: str) -> None:
    root, pins, _ = _fixture(tmp_path)
    (root / filename).unlink()
    with pytest.raises(SourcePreflightError, match=_PAIR_ERROR):
        audit_dataset(root, expected=pins)


def test_unexpected_file_in_frozen_directory_rejected(tmp_path: Path) -> None:
    root, pins, _ = _fixture(tmp_path)
    (root / "receipt.json").write_text("no economic authority")
    with pytest.raises(SourcePreflightError, match=_PAIR_ERROR):
        audit_dataset(root, expected=pins)


def test_symlinked_source_refused(tmp_path: Path) -> None:
    root, pins, _ = _fixture(tmp_path)
    original = root / "arrays.npz"
    target = tmp_path / "target.npz"
    original.rename(target)
    original.symlink_to(target)
    with pytest.raises(SourcePreflightError, match="regular source file"):
        audit_dataset(root, expected=pins)


def test_symlinked_dataset_directory_refused(tmp_path: Path) -> None:
    root, pins, _ = _fixture(tmp_path)
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(SourcePreflightError, match="non-symlink directory"):
        audit_dataset(alias, expected=pins)


def test_manifest_dataset_id_drift_rejected_despite_updated_byte_pins(
    tmp_path: Path,
) -> None:
    root, pins, record = _fixture(tmp_path)
    record["dataset_id"] = "f" * 64
    content = {k: v for k, v in record.items() if k != "artifact_digest"}
    record["artifact_digest"] = _digest(_canonical(content))
    updated = _canonical(record)
    (root / "manifest.json").write_bytes(updated)
    changed_pins = replace(pins, manifest_sha256=_digest(updated))
    with pytest.raises(SourcePreflightError, match="Dataset identity"):
        audit_dataset(root, expected=changed_pins)


def test_manifest_declared_internal_arrays_hash_drift_rejected(tmp_path: Path) -> None:
    root, pins, record = _fixture(tmp_path)
    record["arrays_digest"] = "f" * 64
    content = {k: v for k, v in record.items() if k != "artifact_digest"}
    record["artifact_digest"] = _digest(_canonical(content))
    updated = _canonical(record)
    (root / "manifest.json").write_bytes(updated)
    with pytest.raises(SourcePreflightError, match="embedded arrays digest"):
        audit_dataset(root, expected=replace(pins, manifest_sha256=_digest(updated)))


def test_manifest_feature_shape_drift_rejected(tmp_path: Path) -> None:
    root, pins, record = _fixture(tmp_path)
    record["arrays"]["features"]["shape"] = [8761, 2, 6]
    content = {k: v for k, v in record.items() if k != "artifact_digest"}
    record["artifact_digest"] = _digest(_canonical(content))
    updated = _canonical(record)
    (root / "manifest.json").write_bytes(updated)
    with pytest.raises(SourcePreflightError, match="declared feature dimensions"):
        audit_dataset(root, expected=replace(pins, manifest_sha256=_digest(updated)))


def test_manifest_artifact_digest_tamper_fails_even_if_external_pin_matches(
    tmp_path: Path,
) -> None:
    root, pins, record = _fixture(tmp_path)
    record["artifact_digest"] = "0" * 64
    updated = _canonical(record)
    (root / "manifest.json").write_bytes(updated)
    with pytest.raises(SourcePreflightError, match="artifact_digest"):
        audit_dataset(root, expected=replace(pins, manifest_sha256=_digest(updated)))


def test_duplicate_manifest_json_key_rejected(tmp_path: Path) -> None:
    root, pins, _ = _fixture(tmp_path)
    data = (root / "manifest.json").read_bytes()
    updated = data.replace(b'"dataset_id":', b'"dataset_id":"z","dataset_id":', 1)
    (root / "manifest.json").write_bytes(updated)
    with pytest.raises(SourcePreflightError, match="duplicate JSON field"):
        audit_dataset(root, expected=replace(pins, manifest_sha256=_digest(updated)))


def test_missing_manifest_file_never_emits_report(tmp_path: Path) -> None:
    root, pins, _ = _fixture(tmp_path)
    (root / "manifest.json").unlink()
    output = tmp_path / "audit-report.json"
    with pytest.raises(SourcePreflightError):
        publish_report(root, output, expected=pins)
    assert not output.exists()


def test_write_once_report_does_not_modify_existing_provenance(tmp_path: Path) -> None:
    root, pins, _ = _fixture(tmp_path)
    output = tmp_path / "audit-report.json"
    output.write_text("immutable preexisting work", encoding="utf-8")
    with pytest.raises(SourcePreflightError, match="report path already exists"):
        publish_report(root, output, expected=pins)
    assert output.read_text(encoding="utf-8") == "immutable preexisting work"


def test_report_cannot_be_written_into_source_tree(tmp_path: Path) -> None:
    root, pins, _ = _fixture(tmp_path)
    with pytest.raises(
        SourcePreflightError, match="outside original Dataset directory"
    ):
        publish_report(root, root / "evidence.json", expected=pins)
    assert sorted(p.name for p in root.iterdir()) == ["arrays.npz", "manifest.json"]


def test_cli_makes_no_report_without_original_pinned_bytes(tmp_path: Path) -> None:
    from tools.issue810_r2_dataset_source_preflight import main

    root, _, _ = _fixture(tmp_path)
    output = tmp_path / "not-approved.json"
    with pytest.raises(SystemExit) as err:
        main(["--dataset-root", str(root), "--report", str(output)])
    assert err.value.code == 2
    assert not output.exists()


def test_valid_synthetic_report_is_write_once_and_non_authoritative(
    tmp_path: Path,
) -> None:
    root, pins, _ = _fixture(tmp_path)
    output = tmp_path / "synthetic.json"
    payload = publish_report(root, output, expected=pins)
    assert payload["byte_identity"] == "SYNTHETIC_FIXTURE_ONLY"
    assert json.loads(output.read_text()) == payload
    assert payload["historical_point_in_time"] == "NOT_ESTABLISHED"
    assert payload["economic_authority"] == "NOT_GRANTED"
    with pytest.raises(SourcePreflightError, match="report path already exists"):
        publish_report(root, output, expected=pins)


def test_manifest_size_cap_refuses_giant_untrusted_input(tmp_path: Path) -> None:
    root, pins, _ = _fixture(tmp_path)
    giant = b"a" * (4 * 1024 * 1024 + 1)
    (root / "manifest.json").write_bytes(giant)
    with pytest.raises(SourcePreflightError, match="size cap"):
        audit_dataset(root, expected=replace(pins, manifest_sha256=_digest(giant)))


def test_arrays_streaming_size_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools import issue810_r2_dataset_source_preflight as source_audit

    root, pins, _ = _fixture(tmp_path)
    monkeypatch.setattr(source_audit, "MAX_ARRAYS_BYTES", 10)
    with pytest.raises(SourcePreflightError, match="arrays exceed size cap"):
        audit_dataset(root, expected=pins)
