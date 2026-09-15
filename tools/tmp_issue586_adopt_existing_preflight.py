from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest

SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
DATES = (
    "2021-01-15",
    "2021-01-16",
    "2021-07-15",
    "2021-07-16",
    "2022-01-15",
    "2022-01-16",
    "2022-07-15",
    "2022-07-16",
)
PREREG_HEAD = "89ec1092e438d69657840567e684739ca0c4e3d5"
PREREG_PROTOCOL_DIGEST = (
    "bf0aa2db248e9dcab745e92e6bde54de6b48276a785eec74670e0dedba8cb5e3"
)
LEGACY_PREFLIGHT_RUN_ID = 34941447564
LEGACY_PREFLIGHT_HEAD = "a3fc104ba25ceae55a637b075e3b0cbd24ebf2d6"
LEGACY_IMPLEMENTATION_SHA = "1a7a1ae12ebeb7d104f3278fcc0a0dbd6ac64ac8"
LEGACY_HELPER_COMMIT = "b6ba8ab97b87b960c98be75768a64b10fac37819"
LEGACY_TARGETED_VERIFY_RUN_ID = 34940742164
PUBLISHER_ARTIFACT_ID = 10385451448
PUBLISHER_ARTIFACT_API_DIGEST = (
    "07ee6cbaee5ae36d5c564d8559f174e95e2fe63a2e68c7938fe84190e3b03ebb"
)
FRESH_ARTIFACT_ID = 10385516382
FRESH_ARTIFACT_API_DIGEST = (
    "7d02df596647ad109f3f049ec884757f50689f82f65b27f9de795ecfa88f5e7d"
)
MANIFEST_SHA256 = "69a4105e8839d38972a034b49c9a3d019d3bc330981fce7cb860276e9ca0b25b"
REPORT_SHA256 = "c8c1c33704a72b9a8bb1ca96d143bf7883ff4ebf1aeeab3d30f7fe8568ae4b6d"
REPORT_CONTENT_DIGEST = (
    "5a7af640640b23894f857726944a306aef522799ca8c983757e864d76262d54d"
)
VALIDATOR_HEAD = "12151cb014f62f603fc115b3129d14264eb93683"
VALIDATOR_FULL_VERIFY_RUN_ID = 34951770631

MANIFEST_KEYS = {
    "archive_records",
    "fallback_used",
    "issue_number",
    "prereg_head",
    "prereg_protocol_digest",
    "replacement_source_used",
    "schema_version",
    "source_dates",
    "symbols",
    "target_dates",
    "target_source_family",
    "target_transport_mode",
}
RECORD_KEYS = {
    "available_and_valid",
    "checksum_sha256",
    "checksum_url",
    "checksum_verified",
    "csv_member",
    "date",
    "expected_row_count",
    "missing_grid_digest",
    "missing_grid_rows",
    "row_count",
    "strict_15m_grid",
    "structurally_valid",
    "symbol",
    "url",
    "zip_filename",
    "zip_sha256",
    "zip_size",
}
REPORT_KEYS = {
    "content_digest",
    "economic_values_reported",
    "evaluation_pnl_inspected",
    "failures",
    "fallback_used",
    "issue_number",
    "minimum_target_windows_per_symbol",
    "planned_archives",
    "prereg_head",
    "prereg_protocol_digest",
    "replacement_source_used",
    "schema_version",
    "spot_flow_computed",
    "status",
    "target_manifest_digest",
    "target_return_computed",
    "target_windows_by_symbol",
    "training_relation_computed",
    "valid_archives",
}


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_bool(value: object, *, field: str, expected: bool | None = None) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field} must be boolean")
    if expected is not None and value is not expected:
        raise ValueError(f"{field} differs from frozen authority")
    return value


def _require_int(value: object, *, field: str, expected: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if expected is not None and value != expected:
        raise ValueError(f"{field} differs from frozen authority")
    return value


def _require_str(value: object, *, field: str, expected: str | None = None) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    if expected is not None and value != expected:
        raise ValueError(f"{field} differs from frozen authority")
    return value


def _require_hex(value: object, *, field: str, length: int = 64) -> str:
    resolved = _require_str(value, field=field)
    if len(resolved) != length or resolved.lower() != resolved:
        raise ValueError(f"{field} must be lowercase hexadecimal")
    if any(character not in "0123456789abcdef" for character in resolved):
        raise ValueError(f"{field} must be lowercase hexadecimal")
    return resolved


def _load_canonical_json(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    decoded = json.loads(raw)
    if not isinstance(decoded, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    if canonical_json_bytes(decoded) != raw:
        raise ValueError(f"{path.name} is not canonical JSON")
    return decoded, raw


def _git_blob(commit: str, path: str) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", f"{commit}:{path}"], text=True
    ).strip()


def _audit_manifest(manifest: dict[str, Any]) -> dict[str, int]:
    if set(manifest) != MANIFEST_KEYS:
        raise ValueError("legacy manifest field set differs")
    _require_str(
        manifest["schema_version"],
        field="manifest.schema_version",
        expected="issue584_target_source_manifest_v1",
    )
    _require_int(manifest["issue_number"], field="manifest.issue_number", expected=584)
    _require_str(manifest["prereg_head"], field="manifest.prereg_head", expected=PREREG_HEAD)
    _require_str(
        manifest["prereg_protocol_digest"],
        field="manifest.prereg_protocol_digest",
        expected=PREREG_PROTOCOL_DIGEST,
    )
    if manifest["symbols"] != list(SYMBOLS):
        raise ValueError("legacy manifest symbol roster differs")
    if manifest["source_dates"] != [DATES[0], DATES[2], DATES[4], DATES[6]]:
        raise ValueError("legacy manifest source date roster differs")
    if manifest["target_dates"] != list(DATES):
        raise ValueError("legacy manifest target date roster differs")
    _require_str(
        manifest["target_source_family"],
        field="manifest.target_source_family",
        expected="binance_vision_contract_klines",
    )
    _require_str(
        manifest["target_transport_mode"],
        field="manifest.target_transport_mode",
        expected="VISION",
    )
    _require_bool(manifest["fallback_used"], field="manifest.fallback_used", expected=False)
    _require_bool(
        manifest["replacement_source_used"],
        field="manifest.replacement_source_used",
        expected=False,
    )

    records = manifest["archive_records"]
    if not isinstance(records, list) or len(records) != 40:
        raise ValueError("legacy manifest must contain exactly 40 archive records")
    expected_pairs = [(symbol, date) for symbol in SYMBOLS for date in DATES]
    actual_pairs: list[tuple[str, str]] = []
    verified = 0
    structurally_valid = 0
    complete = 0
    old_empty_missing_digest = _sha256(canonical_json_bytes([]))
    for index, item in enumerate(records):
        if not isinstance(item, dict) or set(item) != RECORD_KEYS:
            raise ValueError(f"archive_records[{index}] field set differs")
        symbol = _require_str(item["symbol"], field=f"records[{index}].symbol")
        date = _require_str(item["date"], field=f"records[{index}].date")
        actual_pairs.append((symbol, date))
        if (symbol, date) != expected_pairs[index]:
            raise ValueError("legacy archive roster order differs")
        url = (
            "https://data.binance.vision/data/futures/um/daily/klines/"
            f"{symbol}/15m/{symbol}-15m-{date}.zip"
        )
        _require_str(item["url"], field=f"records[{index}].url", expected=url)
        _require_str(
            item["checksum_url"],
            field=f"records[{index}].checksum_url",
            expected=url + ".CHECKSUM",
        )
        _require_str(
            item["zip_filename"],
            field=f"records[{index}].zip_filename",
            expected=f"{symbol}-15m-{date}.zip",
        )
        _require_str(
            item["csv_member"],
            field=f"records[{index}].csv_member",
            expected=f"{symbol}-15m-{date}.csv",
        )
        size = _require_int(item["zip_size"], field=f"records[{index}].zip_size")
        if size <= 0:
            raise ValueError("legacy ZIP size must be positive")
        zip_digest = _require_hex(item["zip_sha256"], field=f"records[{index}].zip_sha256")
        checksum_digest = _require_hex(
            item["checksum_sha256"], field=f"records[{index}].checksum_sha256"
        )
        if zip_digest != checksum_digest:
            raise ValueError("legacy archive/checksum digest differs")
        _require_bool(
            item["checksum_verified"],
            field=f"records[{index}].checksum_verified",
            expected=True,
        )
        verified += 1
        _require_int(
            item["expected_row_count"],
            field=f"records[{index}].expected_row_count",
            expected=96,
        )
        _require_int(item["row_count"], field=f"records[{index}].row_count", expected=96)
        _require_int(
            item["missing_grid_rows"],
            field=f"records[{index}].missing_grid_rows",
            expected=0,
        )
        _require_str(
            item["missing_grid_digest"],
            field=f"records[{index}].missing_grid_digest",
            expected=old_empty_missing_digest,
        )
        _require_bool(
            item["strict_15m_grid"],
            field=f"records[{index}].strict_15m_grid",
            expected=True,
        )
        _require_bool(
            item["structurally_valid"],
            field=f"records[{index}].structurally_valid",
            expected=True,
        )
        _require_bool(
            item["available_and_valid"],
            field=f"records[{index}].available_and_valid",
            expected=True,
        )
        structurally_valid += 1
        complete += 1
    if actual_pairs != expected_pairs:
        raise ValueError("legacy archive roster differs")
    return {
        "available_archive_count": len(records),
        "available_checksum_count": verified,
        "checksum_verified_count": verified,
        "structurally_valid_archive_count": structurally_valid,
        "complete_96_row_archive_count": complete,
    }


def _audit_report(report: dict[str, Any]) -> None:
    if set(report) != REPORT_KEYS:
        raise ValueError("legacy report field set differs")
    _require_str(
        report["schema_version"],
        field="report.schema_version",
        expected="issue584_target_source_preflight_v1",
    )
    _require_int(report["issue_number"], field="report.issue_number", expected=584)
    _require_str(report["prereg_head"], field="report.prereg_head", expected=PREREG_HEAD)
    _require_str(
        report["prereg_protocol_digest"],
        field="report.prereg_protocol_digest",
        expected=PREREG_PROTOCOL_DIGEST,
    )
    _require_int(report["planned_archives"], field="report.planned_archives", expected=40)
    _require_int(report["valid_archives"], field="report.valid_archives", expected=40)
    _require_int(
        report["minimum_target_windows_per_symbol"],
        field="report.minimum_target_windows_per_symbol",
        expected=360,
    )
    windows = report["target_windows_by_symbol"]
    if windows != {symbol: 384 for symbol in sorted(SYMBOLS)}:
        raise ValueError("legacy target-window structural coverage differs")
    _require_str(
        report["status"],
        field="report.status",
        expected="PASS_TARGET_SOURCE_PREFLIGHT",
    )
    if report["failures"] != []:
        raise ValueError("legacy target preflight contains failures")
    _require_str(
        report["target_manifest_digest"],
        field="report.target_manifest_digest",
        expected=MANIFEST_SHA256,
    )
    for field in (
        "economic_values_reported",
        "target_return_computed",
        "spot_flow_computed",
        "training_relation_computed",
        "evaluation_pnl_inspected",
        "fallback_used",
        "replacement_source_used",
    ):
        _require_bool(report[field], field=f"report.{field}", expected=False)
    actual_content = _require_str(report["content_digest"], field="report.content_digest")
    without_digest = dict(report)
    without_digest.pop("content_digest")
    if content_digest(without_digest) != actual_content or actual_content != REPORT_CONTENT_DIGEST:
        raise ValueError("legacy report content digest differs")


def audit(publisher_dir: Path, fresh_dir: Path) -> dict[str, object]:
    publisher_manifest, publisher_manifest_raw = _load_canonical_json(
        publisher_dir / "manifest.json"
    )
    fresh_manifest, fresh_manifest_raw = _load_canonical_json(fresh_dir / "manifest.json")
    publisher_report, publisher_report_raw = _load_canonical_json(publisher_dir / "report.json")
    fresh_report, fresh_report_raw = _load_canonical_json(fresh_dir / "report.json")
    verification, _ = _load_canonical_json(fresh_dir / "verification.json")

    if publisher_manifest_raw != fresh_manifest_raw:
        raise ValueError("publisher/fresh legacy manifests are not byte-identical")
    if publisher_report_raw != fresh_report_raw:
        raise ValueError("publisher/fresh legacy reports are not byte-identical")
    if _sha256(publisher_manifest_raw) != MANIFEST_SHA256:
        raise ValueError("legacy manifest raw SHA-256 differs")
    if _sha256(publisher_report_raw) != REPORT_SHA256:
        raise ValueError("legacy report raw SHA-256 differs")
    if content_digest(publisher_manifest) != MANIFEST_SHA256:
        raise ValueError("legacy manifest canonical digest differs")

    counts = _audit_manifest(publisher_manifest)
    _audit_report(publisher_report)
    if verification != {
        "economic_values_reported": False,
        "evaluation_pnl_inspected": False,
        "manifest_byte_equal": True,
        "report_byte_equal": True,
        "schema_version": "issue584_target_source_fresh_v2",
        "spot_flow_computed": False,
        "status": "PASS_TARGET_SOURCE_PREFLIGHT",
        "target_manifest_digest": MANIFEST_SHA256,
        "target_return_computed": False,
        "training_relation_computed": False,
    }:
        raise ValueError("legacy fresh verification differs")

    helper_path = "tools/tmp_issue584_target_preflight.py"
    diagnostic_path = (
        "trade_rl/evaluation/experiments/bootstrap/spot_flow_continuation_diagnostic.py"
    )
    helper_blob = _git_blob(LEGACY_HELPER_COMMIT, helper_path)
    diagnostic_blob = _git_blob(LEGACY_IMPLEMENTATION_SHA, diagnostic_path)
    validator_blob = _git_blob(
        VALIDATOR_HEAD,
        "trade_rl/evaluation/experiments/bootstrap/issue586_target_source_validator.py",
    )

    report_without_digest: dict[str, object] = {
        "schema_version": "issue586_existing_preflight_adoption_v1",
        "issue_number": 586,
        "status": "PASS_USDM_15M_TARGET_SOURCE",
        "decision_equivalent_to_frozen_pass_gate": True,
        "planned_archive_count": 40,
        **counts,
        "legacy_preflight_run_id": LEGACY_PREFLIGHT_RUN_ID,
        "legacy_preflight_head": LEGACY_PREFLIGHT_HEAD,
        "legacy_targeted_verify_run_id": LEGACY_TARGETED_VERIFY_RUN_ID,
        "legacy_implementation_sha": LEGACY_IMPLEMENTATION_SHA,
        "legacy_implementation_blob_sha": diagnostic_blob,
        "legacy_helper_commit": LEGACY_HELPER_COMMIT,
        "legacy_helper_blob_sha": helper_blob,
        "legacy_publisher_artifact_id": PUBLISHER_ARTIFACT_ID,
        "legacy_publisher_artifact_api_digest": PUBLISHER_ARTIFACT_API_DIGEST,
        "legacy_fresh_artifact_id": FRESH_ARTIFACT_ID,
        "legacy_fresh_artifact_api_digest": FRESH_ARTIFACT_API_DIGEST,
        "legacy_manifest_sha256": MANIFEST_SHA256,
        "legacy_report_sha256": REPORT_SHA256,
        "legacy_report_content_digest": REPORT_CONTENT_DIGEST,
        "validator_head": VALIDATOR_HEAD,
        "validator_full_verify_run_id": VALIDATOR_FULL_VERIFY_RUN_ID,
        "validator_blob_sha": validator_blob,
        "prereg_head": PREREG_HEAD,
        "prereg_protocol_digest": PREREG_PROTOCOL_DIGEST,
        "target_source_family": "binance_vision_contract_klines",
        "target_transport_mode": "VISION",
        "header_contract_verified_by_legacy_parser": True,
        "header_presence_individually_recorded": False,
        "close_time_contract_verified_by_legacy_parser": True,
        "requested_date_contract_verified_by_legacy_parser": True,
        "strict_order_contract_verified_by_legacy_parser": True,
        "economic_numeric_domain_validated_structurally_only": True,
        "publisher_fresh_manifest_byte_equal": True,
        "publisher_fresh_report_byte_equal": True,
        "network_fetch_performed_by_this_adoption": False,
        "economic_values_reported": False,
        "target_return_computed": False,
        "spot_flow_computed": False,
        "training_relation_computed": False,
        "evaluation_pnl_inspected": False,
        "production_eligible": False,
        "live_trading_authorized": False,
    }
    return {
        **report_without_digest,
        "content_digest": content_digest(report_without_digest),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--publisher-dir", type=Path, required=True)
    parser.add_argument("--fresh-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.publisher_dir, args.fresh_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(report))
    print("ISSUE586_ADOPTION_STATUS=" + str(report["status"]))
    print("DECISION_EQUIVALENT_TO_FROZEN_PASS_GATE=true")
    print("NETWORK_FETCH_PERFORMED_BY_THIS_ADOPTION=false")
    print("TARGET_RETURN_COMPUTED=false")
    print("TRAINING_RELATION_COMPUTED=false")


if __name__ == "__main__":
    main()
