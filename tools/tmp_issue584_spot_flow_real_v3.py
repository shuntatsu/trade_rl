from __future__ import annotations

import argparse
import hashlib
import io
import json
import urllib.error
import urllib.request
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.experiments.bootstrap.spot_flow_continuation_diagnostic import (
    TargetBar,
    aggregate_spot_day_csv,
    build_four_hour_label,
    build_spot_flow_diagnostic_result,
    canonical_spot_flow_result_bytes,
    load_spot_flow_result_bytes,
    parse_usdm_15m_klines_csv,
)
from trade_rl.evaluation.experiments.bootstrap.spot_flow_continuation_prereg import (
    canonical_spot_flow_continuation_protocol,
)

IMPLEMENTATION_HEAD = "53b005f4dd16de4b4bfe9593714db4d77d8da07f"
SPOT_SOURCE_RESULT_SHA256 = (
    "dfa2a5350ffdd416c1bc06fb48539cf882a009910929b9873fd7942ef53e6030"
)
SPOT_SOURCE_RESULT_CONTENT_DIGEST = (
    "06207c17f997ebd6c99db553cee07881a10e8b54d16af062a47e0fcb0a2cb51a"
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
TARGET_PREFLIGHT_RUN_ID = 34941447564
TARGET_PREFLIGHT_PUBLISHER_ARTIFACT_ID = 10385451448
TARGET_PREFLIGHT_PUBLISHER_DIGEST = (
    "07ee6cbaee5ae36d5c564d8559f174e95e2fe63a2e68c7938fe84190e3b03ebb"
)
QUARTER_HOUR_MS = 15 * 60 * 1000


def _load_json(path: Path) -> tuple[bytes, dict[str, Any]]:
    raw = path.read_bytes()
    try:
        decoded: Any = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path.name} is not valid UTF-8 JSON") from error
    if not isinstance(decoded, dict) or any(not isinstance(k, str) for k in decoded):
        raise ValueError(f"{path.name} must be a JSON object")
    if canonical_json_bytes(decoded) != raw:
        raise ValueError(f"{path.name} is not canonical JSON")
    return raw, decoded


def _without_content_digest(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("content_digest", None)
    return result


def _next_date(value: str) -> str:
    resolved = datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC)
    return (resolved + timedelta(days=1)).strftime("%Y-%m-%d")


def _day_start_ms(value: str) -> int:
    return int(
        datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC).timestamp() * 1000
    )


def _parse_checksum(payload: bytes, *, expected_name: str) -> str:
    try:
        text = payload.decode("utf-8").strip()
    except UnicodeDecodeError as error:
        raise ValueError("checksum is not UTF-8") from error
    parts = text.split()
    if len(parts) != 2:
        raise ValueError("checksum text is malformed")
    digest, raw_name = parts
    name = raw_name.lstrip("*")
    if (
        len(digest) != 64
        or digest.lower() != digest
        or any(char not in "0123456789abcdef" for char in digest)
    ):
        raise ValueError("checksum digest is malformed")
    if name != expected_name:
        raise ValueError("checksum filename differs from frozen archive")
    return digest


def _fetch(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "trade-rl-issue584-diagnostic-v3/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            if response.status != 200:
                raise ValueError(f"unexpected HTTP status {response.status}")
            return response.read()
    except (urllib.error.URLError, TimeoutError) as error:
        raise ValueError(f"transport failed for {url}") from error


def _read_single_member(
    archive_bytes: bytes,
    *,
    expected_member: str,
) -> bytes:
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            members = tuple(item for item in archive.infolist() if not item.is_dir())
            if len(members) != 1 or members[0].filename != expected_member:
                raise ValueError("ZIP member roster differs from frozen authority")
            return archive.read(members[0])
    except zipfile.BadZipFile as error:
        raise ValueError("archive is not a valid ZIP") from error


def _validate_spot_source_result(path: Path) -> list[dict[str, Any]]:
    protocol = canonical_spot_flow_continuation_protocol()
    raw, payload = _load_json(path)
    if hashlib.sha256(raw).hexdigest() != SPOT_SOURCE_RESULT_SHA256:
        raise ValueError("Spot source result SHA-256 differs from frozen authority")
    if payload.get("content_digest") != SPOT_SOURCE_RESULT_CONTENT_DIGEST:
        raise ValueError("Spot source result content digest differs")
    if (
        content_digest(_without_content_digest(payload))
        != SPOT_SOURCE_RESULT_CONTENT_DIGEST
    ):
        raise ValueError("Spot source result content digest does not verify")
    if payload.get("status") != protocol.spot_source_status:
        raise ValueError("Spot source result status differs from preregistration")
    if payload.get("protocol_head") != protocol.spot_source_protocol_head:
        raise ValueError("Spot source protocol head differs from preregistration")
    if payload.get("protocol_digest") != protocol.spot_source_protocol_digest:
        raise ValueError("Spot source protocol digest differs from preregistration")
    if payload.get("planned_archive_count") != protocol.spot_archive_count:
        raise ValueError("Spot archive count differs from preregistration")
    reports = payload.get("archive_reports")
    if not isinstance(reports, list) or len(reports) != protocol.spot_archive_count:
        raise ValueError("Spot archive reports are incomplete")
    expected_pairs = [
        (symbol, date) for symbol in protocol.symbols for date in protocol.spot_dates
    ]
    observed_pairs: list[tuple[str, str]] = []
    result: list[dict[str, Any]] = []
    for raw_report in reports:
        if not isinstance(raw_report, dict):
            raise ValueError("Spot archive report is malformed")
        report = dict(raw_report)
        symbol = report.get("symbol")
        date = report.get("date")
        if not isinstance(symbol, str) or not isinstance(date, str):
            raise ValueError("Spot archive identity is malformed")
        observed_pairs.append((symbol, date))
        if report.get("archive_available") is not True:
            raise ValueError("Spot archive is unavailable")
        if report.get("checksum_available") is not True:
            raise ValueError("Spot checksum is unavailable")
        if report.get("checksum_verified") is not True:
            raise ValueError("Spot checksum is unverified")
        if report.get("schema_valid") is not True:
            raise ValueError("Spot archive is structurally invalid")
        raw_sha = report.get("raw_zip_sha256")
        checksum_sha = report.get("checksum_digest")
        if not isinstance(raw_sha, str) or raw_sha != checksum_sha:
            raise ValueError("Spot source archive/checksum digests disagree")
        result.append(report)
    if observed_pairs != expected_pairs:
        raise ValueError("Spot source roster/order differs from preregistration")
    return result


def _validate_target_preflight(
    report_path: Path,
    manifest_path: Path,
) -> tuple[list[dict[str, Any]], str]:
    protocol = canonical_spot_flow_continuation_protocol()
    raw_report, report = _load_json(report_path)
    _, manifest = _load_json(manifest_path)
    if hashlib.sha256(raw_report).hexdigest() != TARGET_PREFLIGHT_REPORT_SHA256:
        raise ValueError("target preflight report SHA-256 differs")
    if report.get("content_digest") != TARGET_PREFLIGHT_REPORT_CONTENT_DIGEST:
        raise ValueError("target preflight report content digest differs")
    if (
        content_digest(_without_content_digest(report))
        != TARGET_PREFLIGHT_REPORT_CONTENT_DIGEST
    ):
        raise ValueError("target preflight report content digest does not verify")
    if report.get("status") != "PASS_TARGET_SOURCE_PREFLIGHT":
        raise ValueError("target preflight did not PASS")
    if report.get("planned_archives") != 40 or report.get("valid_archives") != 40:
        raise ValueError("target preflight archive roster is incomplete")
    manifest_digest = content_digest(manifest)
    if manifest_digest != TARGET_MANIFEST_DIGEST:
        raise ValueError("target manifest digest differs from frozen authority")
    if report.get("target_manifest_digest") != manifest_digest:
        raise ValueError("target report does not bind target manifest")
    if manifest.get("symbols") != list(protocol.symbols):
        raise ValueError("target manifest symbol roster differs")
    expected_dates: list[str] = []
    for source_date in protocol.spot_dates:
        expected_dates.extend((source_date, _next_date(source_date)))
    if manifest.get("target_dates") != expected_dates:
        raise ValueError("target manifest date roster differs")
    if manifest.get("target_source_family") != protocol.target_source_family:
        raise ValueError("target source family differs")
    if manifest.get("target_transport_mode") != protocol.target_transport_mode:
        raise ValueError("target transport differs")
    records = manifest.get("archive_records")
    if not isinstance(records, list) or len(records) != 40:
        raise ValueError("target manifest archive records are incomplete")
    expected_pairs = [
        (symbol, date) for symbol in protocol.symbols for date in expected_dates
    ]
    observed_pairs: list[tuple[str, str]] = []
    result: list[dict[str, Any]] = []
    for raw_record in records:
        if not isinstance(raw_record, dict):
            raise ValueError("target archive record is malformed")
        record = dict(raw_record)
        symbol = record.get("symbol")
        date = record.get("date")
        if not isinstance(symbol, str) or not isinstance(date, str):
            raise ValueError("target archive identity is malformed")
        observed_pairs.append((symbol, date))
        if record.get("available_and_valid") is not True:
            raise ValueError("target archive is not available and valid")
        if record.get("checksum_verified") is not True:
            raise ValueError("target checksum is not verified")
        if record.get("structurally_valid") is not True:
            raise ValueError("target archive is structurally invalid")
        if record.get("row_count") != 96 or record.get("missing_grid_rows") != 0:
            raise ValueError("target archive is not a complete native day")
        if record.get("strict_15m_grid") is not True:
            raise ValueError("target archive native grid is invalid")
        if record.get("zip_sha256") != record.get("checksum_sha256"):
            raise ValueError("target archive/checksum digests disagree")
        result.append(record)
    if observed_pairs != expected_pairs:
        raise ValueError("target archive roster/order differs from preregistration")
    return result, manifest_digest


def _fetch_spot_signals(
    reports: list[dict[str, Any]],
) -> tuple[dict[tuple[str, str], tuple[float | None, ...]], list[dict[str, object]]]:
    signals: dict[tuple[str, str], tuple[float | None, ...]] = {}
    evidence: list[dict[str, object]] = []
    for report in reports:
        symbol = str(report["symbol"])
        date = str(report["date"])
        url = str(report["url"])
        checksum_url = str(report["checksum_url"])
        expected_sha = str(report["raw_zip_sha256"])
        expected_member = str(report["member_name"])
        archive_name = url.rsplit("/", 1)[-1]
        archive_bytes = _fetch(url)
        checksum_bytes = _fetch(checksum_url)
        raw_sha = hashlib.sha256(archive_bytes).hexdigest()
        checksum_sha = _parse_checksum(checksum_bytes, expected_name=archive_name)
        if raw_sha != expected_sha or checksum_sha != expected_sha:
            raise ValueError(f"Spot source bytes changed for {symbol} {date}")
        csv_bytes = _read_single_member(
            archive_bytes,
            expected_member=expected_member,
        )
        daily = aggregate_spot_day_csv(csv_bytes, expected_date=date)
        if len(daily) != 96:
            raise ValueError("Spot daily aggregation did not produce 96 intervals")
        signals[(symbol, date)] = daily
        evidence.append(
            {
                "symbol": symbol,
                "date": date,
                "raw_zip_sha256": raw_sha,
                "csv_member": expected_member,
            }
        )
    return signals, evidence


def _fetch_target_bars(
    records: list[dict[str, Any]],
) -> tuple[dict[tuple[str, str], tuple[TargetBar, ...]], list[dict[str, object]]]:
    bars: dict[tuple[str, str], tuple[TargetBar, ...]] = {}
    evidence: list[dict[str, object]] = []
    for record in records:
        symbol = str(record["symbol"])
        date = str(record["date"])
        url = str(record["url"])
        checksum_url = str(record["checksum_url"])
        expected_sha = str(record["zip_sha256"])
        expected_member = str(record["csv_member"])
        archive_name = str(record["zip_filename"])
        archive_bytes = _fetch(url)
        checksum_bytes = _fetch(checksum_url)
        raw_sha = hashlib.sha256(archive_bytes).hexdigest()
        checksum_sha = _parse_checksum(checksum_bytes, expected_name=archive_name)
        if raw_sha != expected_sha or checksum_sha != expected_sha:
            raise ValueError(f"target source bytes changed for {symbol} {date}")
        csv_bytes = _read_single_member(
            archive_bytes,
            expected_member=expected_member,
        )
        parsed = parse_usdm_15m_klines_csv(csv_bytes, expected_date=date)
        if len(parsed) != 96:
            raise ValueError("target daily parser did not produce 96 bars")
        bars[(symbol, date)] = parsed
        evidence.append(
            {
                "symbol": symbol,
                "date": date,
                "raw_zip_sha256": raw_sha,
                "csv_member": expected_member,
            }
        )
    return bars, evidence


def _build_observations(
    signals_by_day: dict[tuple[str, str], tuple[float | None, ...]],
    bars_by_day: dict[tuple[str, str], tuple[TargetBar, ...]],
) -> dict[str, tuple[list[float], list[float]]]:
    protocol = canonical_spot_flow_continuation_protocol()
    observations: dict[str, tuple[list[float], list[float]]] = {}
    for symbol in protocol.symbols:
        symbol_signals: list[float] = []
        symbol_labels: list[float] = []
        for source_date in protocol.spot_dates:
            daily_signals = signals_by_day[(symbol, source_date)]
            target_bars = (
                *bars_by_day[(symbol, source_date)],
                *bars_by_day[(symbol, _next_date(source_date))],
            )
            start = _day_start_ms(source_date)
            for index, signal in enumerate(daily_signals, start=1):
                decision = start + index * QUARTER_HOUR_MS
                label = build_four_hour_label(
                    target_bars,
                    decision_time_ms=decision,
                )
                if signal is None or label is None:
                    continue
                symbol_signals.append(signal)
                symbol_labels.append(label)
        observations[symbol] = (symbol_signals, symbol_labels)
    return observations


def build_real_result(
    *,
    spot_result_path: Path,
    target_report_path: Path,
    target_manifest_path: Path,
    execution_run_id: int,
    output_dir: Path,
) -> None:
    protocol = canonical_spot_flow_continuation_protocol()
    spot_reports = _validate_spot_source_result(spot_result_path)
    target_records, target_manifest_digest = _validate_target_preflight(
        target_report_path,
        target_manifest_path,
    )
    signals_by_day, spot_evidence = _fetch_spot_signals(spot_reports)
    bars_by_day, target_evidence = _fetch_target_bars(target_records)
    observations = _build_observations(signals_by_day, bars_by_day)
    result = build_spot_flow_diagnostic_result(
        observations,
        protocol,
        implementation_head=IMPLEMENTATION_HEAD,
        source_manifest_digest=SPOT_SOURCE_RESULT_CONTENT_DIGEST,
        target_manifest_digest=target_manifest_digest,
        target_preflight_run_id=TARGET_PREFLIGHT_RUN_ID,
        target_preflight_artifact_id=TARGET_PREFLIGHT_PUBLISHER_ARTIFACT_ID,
        target_preflight_artifact_api_digest=TARGET_PREFLIGHT_PUBLISHER_DIGEST,
        execution_run_id=execution_run_id,
    )
    result_bytes = canonical_spot_flow_result_bytes(result)
    load_spot_flow_result_bytes(result_bytes)
    evidence_without_digest: dict[str, object] = {
        "schema_version": "issue584_spot_flow_execution_evidence_v3",
        "implementation_head": IMPLEMENTATION_HEAD,
        "execution_run_id": execution_run_id,
        "spot_source_result_sha256": SPOT_SOURCE_RESULT_SHA256,
        "spot_source_result_content_digest": SPOT_SOURCE_RESULT_CONTENT_DIGEST,
        "target_preflight_report_sha256": TARGET_PREFLIGHT_REPORT_SHA256,
        "target_preflight_report_content_digest": (
            TARGET_PREFLIGHT_REPORT_CONTENT_DIGEST
        ),
        "target_manifest_digest": target_manifest_digest,
        "spot_archive_count": len(spot_evidence),
        "target_archive_count": len(target_evidence),
        "spot_archives": spot_evidence,
        "target_archives": target_evidence,
        "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
        "result_content_digest": result.digest,
        "network_source_family": "data.binance.vision",
        "source_replacement_used": False,
        "target_replacement_used": False,
        "evaluation_pnl_inspected": False,
        "final_test_authorized": False,
        "production_eligible": False,
        "live_trading_authorized": False,
    }
    evidence = {
        **evidence_without_digest,
        "content_digest": content_digest(evidence_without_digest),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_bytes(result_bytes)
    (output_dir / "evidence.json").write_bytes(canonical_json_bytes(evidence))


def self_check() -> None:
    protocol = canonical_spot_flow_continuation_protocol()
    observations = {
        symbol: (
            [1.0] * protocol.minimum_eligible_observations_per_symbol,
            ([0.01] * protocol.minimum_eligible_observations_per_symbol)
            if index < 4
            else ([-0.01] * protocol.minimum_eligible_observations_per_symbol),
        )
        for index, symbol in enumerate(protocol.symbols)
    }
    result = build_spot_flow_diagnostic_result(
        observations,
        protocol,
        implementation_head=IMPLEMENTATION_HEAD,
        source_manifest_digest=SPOT_SOURCE_RESULT_CONTENT_DIGEST,
        target_manifest_digest=TARGET_MANIFEST_DIGEST,
        target_preflight_run_id=TARGET_PREFLIGHT_RUN_ID,
        target_preflight_artifact_id=TARGET_PREFLIGHT_PUBLISHER_ARTIFACT_ID,
        target_preflight_artifact_api_digest=TARGET_PREFLIGHT_PUBLISHER_DIGEST,
        execution_run_id=1,
    )
    if result.status != protocol.valid_status or result.positive_slope_count != 4:
        raise AssertionError("synthetic 4/5 gate self-check failed")
    raw = canonical_spot_flow_result_bytes(result)
    if load_spot_flow_result_bytes(raw) != result:
        raise AssertionError("canonical result round-trip self-check failed")

    date = protocol.spot_dates[0]
    start = _day_start_ms(date)
    row = [
        "1",
        "2",
        "3",
        "1",
        "1",
        str(start + 1),
        "False",
        "True",
    ]
    signals = aggregate_spot_day_csv(
        (",".join(row) + "\n").encode(),
        expected_date=date,
    )
    if signals[0] != 1.0 or any(item is not None for item in signals[1:]):
        raise AssertionError("Spot daily aggregation self-check failed")

    decision = start + QUARTER_HOUR_MS
    bars = tuple(
        TargetBar(open_time_ms=decision + i * QUARTER_HOUR_MS, open_price=100.0 + i)
        for i in range(17)
    )
    label = build_four_hour_label(bars, decision_time_ms=decision)
    if label is None or label <= 0.0:
        raise AssertionError("target label clock self-check failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--spot-result", type=Path)
    parser.add_argument("--target-report", type=Path)
    parser.add_argument("--target-manifest", type=Path)
    parser.add_argument("--execution-run-id", type=int)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.self_check:
        self_check()
        print("SELF_CHECK=PASS")
        return
    required = (
        args.spot_result,
        args.target_report,
        args.target_manifest,
        args.execution_run_id,
        args.output_dir,
    )
    if any(item is None for item in required):
        parser.error("real execution requires all source/output arguments")
    assert args.spot_result is not None
    assert args.target_report is not None
    assert args.target_manifest is not None
    assert args.execution_run_id is not None
    assert args.output_dir is not None
    if args.execution_run_id <= 0:
        parser.error("execution run id must be positive")
    build_real_result(
        spot_result_path=args.spot_result,
        target_report_path=args.target_report,
        target_manifest_path=args.target_manifest,
        execution_run_id=args.execution_run_id,
        output_dir=args.output_dir,
    )
    print("RESULT_WRITTEN=true")
    print("ECONOMIC_RESULT_NOT_INTERPRETED_IN_PUBLISHER=true")


if __name__ == "__main__":
    main()
