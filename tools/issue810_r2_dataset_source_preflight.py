"""Offline R2 Dataset byte preflight; never certify historical market availability.

Expected identities originate from the previously byte-verified Oct-15 R2
protocol. The original two Dataset input files are NOT in that release.
Historical source publication/ingestion clocks require separate evidence.
This module neither parses economic arrays nor opens a research execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

R2_DATASET_ID = "e916ff716fb5e4b90d5805eaa79200824b09f103984fa845bbeaedb57bef194f"
R2_DATASET_SCHEMA = "market_dataset_artifact_v4"
R2_ROWS = 9504
R2_FEATURES = (
    "1h__log_return_24bar",
    "4h__log_return_1bar",
    "1d__log_return_1bar",
    "1h__realized_volatility_24bar",
    "1h__volume_zscore_24bar",
    "1h__funding_bps",
)
R2_ARCHIVED_SOURCE_HEAD = "30c5ab54fdb9db7f03b947c1687a09c8b929c461"
CHUNK_SIZE = 1024 * 1024
MAX_ARRAYS_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class _Pins:
    manifest_sha256: str
    arrays_sha256: str


ORIGINAL_PINS = _Pins(
    "6ac5099b98a32bd33e4f98c5ac5fb2dbffd19b0b9e3c7c3a9b5a8f6860810ee6",
    "25c97a4ebd1e81ac64f4bc7e78dce4ab102c522e548e30a2154fcfa0143842a3",
)


class SourcePreflightError(ValueError):
    """An input is absent, not original, or fails a byte-bound declaration."""


def _ensure(condition: bool, message: str) -> None:
    if not condition:
        raise SourcePreflightError(message)


def _open_regular(path: Path) -> int:
    _ensure(stat.S_ISREG(path.lstat().st_mode), "expected regular source file")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise SourcePreflightError("expected regular source file")
    return fd


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    consumed = 0
    with os.fdopen(_open_regular(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            consumed += len(chunk)
            _ensure(consumed <= MAX_ARRAYS_BYTES, "Dataset arrays exceed size cap")
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_bytes(path: Path) -> bytes:
    with os.fdopen(_open_regular(path), "rb") as handle:
        content = handle.read(4 * 1024 * 1024 + 1)
    _ensure(len(content) <= 4 * 1024 * 1024, "Dataset manifest exceeds size cap")
    return content


def _strict_json(content: bytes) -> dict[str, Any]:
    def unique_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        obj: dict[str, Any] = {}
        for name, item in pairs:
            _ensure(name not in obj, f"duplicate JSON field: {name}")
            obj[name] = item
        return obj

    def invalid_constant(name: str) -> None:
        raise SourcePreflightError(f"nonfinite JSON value: {name}")

    try:
        parsed = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=unique_fields,
            parse_constant=invalid_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SourcePreflightError("invalid original Dataset JSON manifest") from exc
    _ensure(isinstance(parsed, dict), "Dataset manifest root must be a mapping")
    return parsed


def _verified_manifest(raw: bytes, *, arrays_sha: str) -> None:
    manifest = _strict_json(raw)
    _ensure(
        set(manifest)
        == {
            "arrays",
            "arrays_digest",
            "arrays_file",
            "artifact_digest",
            "dataset",
            "dataset_id",
            "schema_version",
        },
        "original Dataset manifest fields are unexpected",
    )
    _ensure(manifest["schema_version"] == R2_DATASET_SCHEMA, "wrong Dataset schema")
    _ensure(manifest["dataset_id"] == R2_DATASET_ID, "Dataset identity differs")
    _ensure(manifest["arrays_file"] == "arrays.npz", "unexpected arrays filename")
    _ensure(
        manifest["arrays_digest"] == arrays_sha,
        "embedded arrays digest differs from source bytes",
    )
    nested = manifest["dataset"]
    _ensure(isinstance(nested, dict), "Dataset scalar metadata is invalid")
    _ensure(
        nested.get("dataset_id") == R2_DATASET_ID, "nested Dataset identity differs"
    )
    _ensure(
        nested.get("symbols") == ["BTCUSDT", "ETHUSDT"], "Dataset symbol scope differs"
    )
    declared_features = nested.get("feature_names")
    _ensure(declared_features == list(R2_FEATURES), "Dataset feature scope differs")

    description = manifest["arrays"]
    _ensure(isinstance(description, dict), "Dataset array metadata is invalid")
    expected_shapes = {
        "features": [R2_ROWS, 2, len(R2_FEATURES)],
        "feature_available": [R2_ROWS, 2, len(R2_FEATURES)],
        "timestamps": [R2_ROWS],
        "available_at": [R2_ROWS, 2],
    }
    for name, shape in expected_shapes.items():
        value = description.get(name)
        _ensure(
            isinstance(value, dict) and value.get("shape") == shape,
            f"original declared feature dimensions differ: {name}",
        )
    _ensure(
        "funding_price_rate" in description,
        "original v4 Dataset funding-price channel is missing",
    )
    content = {key: val for key, val in manifest.items() if key != "artifact_digest"}
    serialized = json.dumps(
        content,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    _ensure(
        hashlib.sha256(serialized).hexdigest() == manifest["artifact_digest"],
        "canonical Dataset artifact_digest differs",
    )


def audit_dataset(root: Path, *, expected: _Pins = ORIGINAL_PINS) -> dict[str, object]:
    """Only check R2 input *bytes*; do not parse arrays or trust source clocks."""
    _ensure(
        root.is_dir() and not root.is_symlink(),
        "input root must be a non-symlink directory",
    )
    members = {path.name for path in root.iterdir()}
    _ensure(
        members == {"manifest.json", "arrays.npz"},
        "Dataset directory must contain exactly manifest.json and arrays.npz",
    )
    manifest_path, arrays_path = root / "manifest.json", root / "arrays.npz"
    manifest_bytes = _manifest_bytes(manifest_path)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    _ensure(manifest_sha == expected.manifest_sha256, "manifest SHA256 differs")
    arrays_sha = _sha_file(arrays_path)
    _ensure(arrays_sha == expected.arrays_sha256, "arrays SHA256 differs")
    _verified_manifest(manifest_bytes, arrays_sha=arrays_sha)

    return {
        "schema": "issue810_r2_original_dataset_source_byte_preflight_v1",
        "byte_identity": (
            "VERIFIED_ORIGINAL_R2_DECLARED_BYTES"
            if expected == ORIGINAL_PINS
            else "SYNTHETIC_FIXTURE_ONLY"
        ),
        "dataset_id": R2_DATASET_ID,
        "archived_r2_source_head": R2_ARCHIVED_SOURCE_HEAD,
        "manifest_sha256": manifest_sha,
        "arrays_sha256": arrays_sha,
        "r2_dimensions_declared": [R2_ROWS, 2, len(R2_FEATURES)],
        "r2_features_declared": list(R2_FEATURES),
        "task18_witness_rows_reported": 8761,
        "task18_to_r2_row_scope": "NOT_ESTABLISHED",
        "task18_input_receipts": "NOT_VERIFIED",
        "arrays_payload_parsed": False,
        "source_publication_time": "NOT_ESTABLISHED",
        "source_ingestion_time": "NOT_ESTABLISHED",
        "historical_point_in_time": "NOT_ESTABLISHED",
        "market_result": "NOT_ESTABLISHED",
        "original_48_role_execution": "NOT_ESTABLISHED",
        "economic_authority": "NOT_GRANTED",
        "unused_or_live_authority": "NOT_GRANTED",
        "notes": (
            "These hashes attest bytes only, not historical provider availability. "
            "The Task18 8,761-row witness and R2 9,504-row scope remain separate. "
            "Never infer an external receive time from Dataset event timestamps."
        ),
    }


def publish_report(
    root: Path,
    output: Path,
    *,
    expected: _Pins = ORIGINAL_PINS,
) -> dict[str, object]:
    _ensure(
        not output.resolve().is_relative_to(root.resolve()),
        "report must be outside original Dataset directory",
    )
    _ensure(
        not output.exists() and not output.is_symlink(),
        "report path already exists",
    )
    report = audit_dataset(root, expected=expected)
    try:
        with output.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    except FileExistsError as exc:
        raise SourcePreflightError("report path already exists") from exc
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    options = parser.parse_args(argv)
    try:
        report = publish_report(options.dataset_root, options.report)
    except (OSError, SourcePreflightError) as exc:
        parser.exit(2, f"R2 ORIGINAL DATASET NOT VERIFIED: {exc}\n")
    print(report["byte_identity"])
    print("SOURCE-001 NOT ESTABLISHED; G4 / unused / live NOT GRANTED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
