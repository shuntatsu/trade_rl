from __future__ import annotations

import argparse
import hashlib
import io
import json
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.experiments.bootstrap.premium_pressure_calibration import (
    PremiumObservation,
    TargetOpenObservation,
    TrainingPair,
    build_premium_pressure_result,
    build_training_pairs,
    calibrate_symbol,
    canonical_premium_pressure_result_bytes,
    load_premium_pressure_result_bytes,
    parse_premium_index_1h_csv,
    parse_usdm_1h_target_csv,
)
from trade_rl.evaluation.experiments.bootstrap.premium_pressure_prereg import (
    canonical_premium_pressure_protocol,
)

IMPLEMENTATION_HEAD = "9751445fb98a5c99156b23afeb2bfcc445be1acd"
IMPLEMENTATION_VERIFY_RUN_ID = 34968108264
PREMIUM_REPORT_SHA256 = "059987f3692c0f249f2db5ef2e14f8cbaaefa5ca24dccf4c45bae009ca19ebe4"
PREMIUM_REPORT_CONTENT_DIGEST = "fc756962aa997cfb8beafdafe06e70e02602e23e09a5dce5a36d229ddb558f88"
TARGET_MANIFEST_SHA256 = "e287f03bf18619834108b5b452c26cdffbc66a6be3d51d418c928daa2e77dd15"
TARGET_MANIFEST_CONTENT_DIGEST = "9ced3fbab6e51fbd654768ec6a4ea28f98cc4b9f36c1d9d2a9a8f6540864934e"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
MONTHS = tuple(
    f"{year:04d}-{month:02d}"
    for year in (2021, 2022)
    for month in range(1, 13)
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_canonical_json(
    path: Path, *, expected_sha256: str, expected_content_digest: str
) -> dict[str, Any]:
    raw = path.read_bytes()
    if _sha256(raw) != expected_sha256:
        raise ValueError(f"{path.name} SHA-256 differs from immutable authority")
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"{path.name} is not JSON") from error
    if not isinstance(decoded, dict):
        raise ValueError(f"{path.name} must be a JSON object")
    if canonical_json_bytes(decoded) != raw:
        raise ValueError(f"{path.name} is not canonical JSON")
    unsigned = dict(decoded)
    observed = unsigned.pop("content_digest", None)
    if observed != expected_content_digest or content_digest(unsigned) != observed:
        raise ValueError(f"{path.name} content digest differs from authority")
    return decoded


def _download(url: str) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": "trade-rl-research/1.0"}
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            if response.status != 200:
                raise ValueError(f"unexpected HTTP status {response.status}")
            return response.read()
    except (urllib.error.URLError, TimeoutError) as error:
        raise ValueError("source transport failed") from error


def _checksum_digest(raw: bytes, *, expected_filename: str) -> str:
    try:
        text = raw.decode("utf-8").strip()
    except UnicodeDecodeError as error:
        raise ValueError("checksum is not UTF-8") from error
    parts = text.split()
    if len(parts) < 2:
        raise ValueError("checksum text is malformed")
    digest = parts[0]
    filename = parts[-1].lstrip("*")
    if (
        len(digest) != 64
        or digest.lower() != digest
        or any(char not in "0123456789abcdef" for char in digest)
    ):
        raise ValueError("checksum digest is malformed")
    if filename != expected_filename:
        raise ValueError("checksum filename differs from planned archive")
    return digest


def _csv_member(raw_zip: bytes, *, expected_member: str) -> bytes:
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        if len(members) != 1 or members[0].filename != expected_member:
            raise ValueError("ZIP member roster differs from frozen authority")
        return archive.read(members[0])


def _validate_premium_report(report: dict[str, Any]) -> list[dict[str, Any]]:
    if report.get("status") != "PASS_SPARSE_SOURCE":
        raise ValueError("premium source status differs from frozen PASS")
    if report.get("symbols") != list(SYMBOLS) or report.get("months") != list(MONTHS):
        raise ValueError("premium source roster differs from frozen authority")
    if report.get("planned_archives") != 120:
        raise ValueError("premium planned archive count differs")
    if report.get("missing_rows_imputed") is not False:
        raise ValueError("premium source reports imputation")
    if report.get("post_2022_data_used") is not False:
        raise ValueError("premium source crossed pre-2023 boundary")
    if report.get("premium_value_distribution_inspected") is not False:
        raise ValueError("premium source structural report inspected values")
    entries = report.get("entries")
    if not isinstance(entries, list) or len(entries) != 120:
        raise ValueError("premium source entries differ from frozen roster")
    expected = [(symbol, month) for symbol in SYMBOLS for month in MONTHS]
    observed: list[tuple[str, str]] = []
    for item in entries:
        if not isinstance(item, dict):
            raise ValueError("premium source entry is malformed")
        symbol = item.get("symbol")
        month = item.get("month")
        if not isinstance(symbol, str) or not isinstance(month, str):
            raise ValueError("premium source symbol/month is malformed")
        observed.append((symbol, month))
        for key in (
            "available",
            "checksum_available",
            "checksum_verified",
            "structurally_valid",
            "all_open_times_on_1h_grid",
            "strictly_increasing_unique",
            "all_non_time_fields_finite",
        ):
            if item.get(key) is not True:
                raise ValueError(f"premium entry failed structural gate: {key}")
        if item.get("field_count") != 12:
            raise ValueError("premium source field count differs")
    if observed != expected:
        raise ValueError("premium source entry order/roster differs")
    return entries


def _validate_target_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if manifest.get("symbols") != list(SYMBOLS) or manifest.get("months") != list(MONTHS):
        raise ValueError("target source roster differs from frozen authority")
    if manifest.get("planned_archive_count") != 120:
        raise ValueError("target planned archive count differs")
    if manifest.get("target_source_family") != "binance_vision_contract_klines":
        raise ValueError("target source family differs")
    if manifest.get("target_interval") != "1h":
        raise ValueError("target interval differs")
    for key in (
        "economic_values_inspected",
        "target_price_values_inspected",
        "premium_values_inspected",
        "target_return_computed",
        "evaluation_pnl_inspected",
        "final_test_authorized",
        "production_eligible",
        "live_trading_authorized",
        "fallback_used",
        "replacement_source_used",
    ):
        if manifest.get(key) is not False:
            raise ValueError(f"target manifest forbidden flag differs: {key}")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != 120:
        raise ValueError("target source records differ from frozen roster")
    expected = [(symbol, month) for symbol in SYMBOLS for month in MONTHS]
    observed: list[tuple[str, str]] = []
    for item in records:
        if not isinstance(item, dict):
            raise ValueError("target source record is malformed")
        symbol = item.get("symbol")
        month = item.get("month")
        if not isinstance(symbol, str) or not isinstance(month, str):
            raise ValueError("target source symbol/month is malformed")
        observed.append((symbol, month))
        for key in ("archive_available", "checksum_available", "checksum_verified", "schema_valid"):
            if item.get(key) is not True:
                raise ValueError(f"target record failed structural gate: {key}")
    if observed != expected:
        raise ValueError("target source record order/roster differs")
    return records


def _load_premium_entry(item: dict[str, Any]) -> tuple[PremiumObservation, ...]:
    symbol = str(item["symbol"])
    month = str(item["month"])
    url = str(item["url"])
    checksum_url = str(item["checksum_url"])
    expected_member = str(item["csv_member"])
    expected_sha = str(item["raw_sha256"])
    expected_checksum = str(item["checksum_expected_sha256"])
    filename = url.rsplit("/", 1)[-1]
    raw_zip = _download(url)
    raw_checksum = _download(checksum_url)
    if _sha256(raw_zip) != expected_sha or expected_sha != expected_checksum:
        raise ValueError(f"premium archive identity drifted: {symbol}:{month}")
    if _checksum_digest(raw_checksum, expected_filename=filename) != expected_sha:
        raise ValueError(f"premium provider checksum drifted: {symbol}:{month}")
    csv_bytes = _csv_member(raw_zip, expected_member=expected_member)
    return parse_premium_index_1h_csv(csv_bytes, expected_month=month)


def _load_target_record(item: dict[str, Any]) -> tuple[TargetOpenObservation, ...]:
    symbol = str(item["symbol"])
    month = str(item["month"])
    url = str(item["url"])
    checksum_url = str(item["checksum_url"])
    expected_sha = str(item["raw_zip_sha256"])
    expected_checksum = str(item["checksum_digest"])
    filename = url.rsplit("/", 1)[-1]
    expected_member = filename.removesuffix(".zip") + ".csv"
    raw_zip = _download(url)
    raw_checksum = _download(checksum_url)
    if _sha256(raw_zip) != expected_sha or expected_sha != expected_checksum:
        raise ValueError(f"target archive identity drifted: {symbol}:{month}")
    if _checksum_digest(raw_checksum, expected_filename=filename) != expected_sha:
        raise ValueError(f"target provider checksum drifted: {symbol}:{month}")
    csv_bytes = _csv_member(raw_zip, expected_member=expected_member)
    return parse_usdm_1h_target_csv(csv_bytes, expected_month=month)


def build_real_result(
    *, premium_report_path: Path, target_manifest_path: Path, execution_run_id: int
) -> tuple[bytes, bytes]:
    premium_report = _require_canonical_json(
        premium_report_path,
        expected_sha256=PREMIUM_REPORT_SHA256,
        expected_content_digest=PREMIUM_REPORT_CONTENT_DIGEST,
    )
    target_manifest = _require_canonical_json(
        target_manifest_path,
        expected_sha256=TARGET_MANIFEST_SHA256,
        expected_content_digest=TARGET_MANIFEST_CONTENT_DIGEST,
    )
    premium_entries = _validate_premium_report(premium_report)
    target_records = _validate_target_manifest(target_manifest)

    premium_by_symbol: dict[str, list[PremiumObservation]] = {symbol: [] for symbol in SYMBOLS}
    target_by_symbol: dict[str, list[TargetOpenObservation]] = {symbol: [] for symbol in SYMBOLS}
    for item in premium_entries:
        premium_by_symbol[str(item["symbol"])].extend(_load_premium_entry(item))
    for item in target_records:
        target_by_symbol[str(item["symbol"])].extend(_load_target_record(item))

    calibrations = []
    pair_counts: dict[str, int] = {}
    for symbol in SYMBOLS:
        premium_values = tuple(premium_by_symbol[symbol])
        target_values = tuple(target_by_symbol[symbol])
        pairs = build_training_pairs(premium_values, target_values)
        pair_counts[symbol] = len(pairs)
        calibrations.append(calibrate_symbol(symbol, pairs))

    result = build_premium_pressure_result(
        tuple(calibrations),
        implementation_head=IMPLEMENTATION_HEAD,
        implementation_verification_run_id=IMPLEMENTATION_VERIFY_RUN_ID,
        execution_run_id=execution_run_id,
    )
    result_bytes = canonical_premium_pressure_result_bytes(result)
    if load_premium_pressure_result_bytes(result_bytes) != result:
        raise ValueError("result did not round-trip through strict loader")
    evidence_without_digest: dict[str, object] = {
        "schema_version": "issue603_real_evidence_v1",
        "issue_number": 603,
        "implementation_head": IMPLEMENTATION_HEAD,
        "implementation_verification_run_id": IMPLEMENTATION_VERIFY_RUN_ID,
        "execution_run_id": execution_run_id,
        "premium_report_sha256": PREMIUM_REPORT_SHA256,
        "premium_report_content_digest": PREMIUM_REPORT_CONTENT_DIGEST,
        "target_manifest_sha256": TARGET_MANIFEST_SHA256,
        "target_manifest_content_digest": TARGET_MANIFEST_CONTENT_DIGEST,
        "premium_archive_count": 120,
        "target_archive_count": 120,
        "pair_counts_by_symbol": pair_counts,
        "result_sha256": _sha256(result_bytes),
        "result_content_digest": result.content_digest,
        "economic_values_logged": False,
        "evaluation_pnl_inspected": False,
        "final_test_authorized": False,
        "production_eligible": False,
        "live_trading_authorized": False,
    }
    evidence = {
        **evidence_without_digest,
        "content_digest": content_digest(evidence_without_digest),
    }
    return result_bytes, canonical_json_bytes(evidence)


def _self_test() -> None:
    protocol = canonical_premium_pressure_protocol()
    n = protocol.minimum_eligible_observations_per_symbol
    calibrations = []
    for index, symbol in enumerate(SYMBOLS):
        sign = -1.0 if index < 4 else 1.0
        pairs = tuple(
            TrainingPair(decision_time_ms=i * 3_600_000, x=float(i % 7), y=sign * float(i % 7))
            for i in range(n)
        )
        calibrations.append(calibrate_symbol(symbol, pairs))
    result = build_premium_pressure_result(
        tuple(calibrations),
        implementation_head=IMPLEMENTATION_HEAD,
        implementation_verification_run_id=IMPLEMENTATION_VERIFY_RUN_ID,
        execution_run_id=1,
    )
    raw = canonical_premium_pressure_result_bytes(result)
    if load_premium_pressure_result_bytes(raw) != result:
        raise SystemExit("self-test result round trip failed")
    if result.status != protocol.valid_status or result.negative_slope_count != 4:
        raise SystemExit("self-test frozen gate failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--premium-report", type=Path)
    parser.add_argument("--target-manifest", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--execution-run-id", type=int)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        print("ISSUE603_HELPER_SELF_TEST=PASS")
        return
    if any(
        value is None
        for value in (
            args.premium_report,
            args.target_manifest,
            args.result,
            args.evidence,
            args.execution_run_id,
        )
    ):
        parser.error("real mode requires source artifacts, outputs, and execution run id")
    if args.execution_run_id <= 0:
        parser.error("execution run id must be positive")
    result_bytes, evidence_bytes = build_real_result(
        premium_report_path=args.premium_report,
        target_manifest_path=args.target_manifest,
        execution_run_id=args.execution_run_id,
    )
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_bytes(result_bytes)
    args.evidence.write_bytes(evidence_bytes)
    print("ISSUE603_RESULT_WRITTEN=true")
    print("ISSUE603_ECONOMIC_VALUES_LOGGED=false")
    print("ISSUE603_EVALUATION_PNL_INSPECTED=false")


if __name__ == "__main__":
    main()
