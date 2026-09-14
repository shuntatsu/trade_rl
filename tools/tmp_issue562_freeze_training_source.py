from __future__ import annotations

import calendar
import csv
import hashlib
import io
import json
import math
import os
import subprocess
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.experiments.bootstrap.signed_taker_flow_prereg import (
    canonical_signed_taker_flow_protocol,
)
from trade_rl.integrations.binance.vision import vision_monthly_kline_url

SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
EXPECTED_HEADER = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
)
SOURCE_PUBLISHER_RUN_ID = 34840494198
SOURCE_ARTIFACT_ID = 10345533785
SOURCE_ARTIFACT_API_DIGEST = (
    "sha256:689188893dec90b41819972e1cb151920ab198bf6bf27b1abeae2ab18e2fac0f"
)
SOURCE_REPORT_CONTENT_DIGEST = (
    "e3b627cc8efa63623135512d344a883c1b1be8a0e1bc169c24605ec63abaa693"
)
SOURCE_FRESH_RUN_ID = 34840645375
SOURCE_FRESH_ARTIFACT_ID = 10346140972
SOURCE_FRESH_ARTIFACT_API_DIGEST = (
    "sha256:16f91dcaab5f1914d19ee93934643cf27ad3ab94e1787cb04123b8dc3ad68a60"
)
CALIBRATION_HEAD = "dd3ecb617ecd2476e20e9215eeb422d2cdc95750"
FULL_CI_RUN_ID = 34855652829
INTERVAL_MS = 3_600_000


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _curl(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            "curl",
            "--fail",
            "--silent",
            "--show-error",
            "--location",
            "--retry",
            "3",
            "--retry-delay",
            "1",
            "--output",
            str(target),
            url,
        ],
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"download failed: {url}")


def _months(first: datetime, last: datetime) -> tuple[str, ...]:
    cursor = first.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    stop = last.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    values: list[str] = []
    while cursor <= stop:
        values.append(cursor.strftime("%Y-%m"))
        if cursor.month == 12:
            cursor = cursor.replace(year=cursor.year + 1, month=1)
        else:
            cursor = cursor.replace(month=cursor.month + 1)
    return tuple(values)


def _month_dt(period: str) -> datetime:
    year, month = (int(part) for part in period.split("-"))
    return datetime(year, month, 1, tzinfo=UTC)


def _inspect_archive(
    *, symbol: str, period: str, zip_path: Path, checksum_path: Path
) -> dict[str, object]:
    payload = zip_path.read_bytes()
    sha256 = hashlib.sha256(payload).hexdigest()
    checksum_bytes = checksum_path.read_bytes()
    try:
        checksum_text = checksum_bytes.decode("utf-8").strip()
    except UnicodeDecodeError as error:
        raise RuntimeError(f"checksum is not UTF-8: {symbol}:{period}") from error
    fields = checksum_text.split()
    if not fields or fields[0].lower() != sha256:
        raise RuntimeError(f"checksum mismatch: {symbol}:{period}")

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        if len(members) != 1 or not members[0].filename.lower().endswith(".csv"):
            raise RuntimeError(f"archive member contract failed: {symbol}:{period}")
        raw_csv = archive.read(members[0]).decode("utf-8-sig")
    rows = [row for row in csv.reader(io.StringIO(raw_csv)) if row]
    if not rows:
        raise RuntimeError(f"empty archive: {symbol}:{period}")
    try:
        int(rows[0][0])
        header_present = False
    except (ValueError, IndexError):
        header_present = True
    header = tuple(rows[0]) if header_present else ()
    if header_present and header != EXPECTED_HEADER:
        raise RuntimeError(f"header mismatch: {symbol}:{period}")
    data = rows[1:] if header_present else rows

    year, month = (int(part) for part in period.split("-"))
    expected_rows = calendar.monthrange(year, month)[1] * 24
    first_expected = int(datetime(year, month, 1, tzinfo=UTC).timestamp() * 1000)
    if month == 12:
        next_month = datetime(year + 1, 1, 1, tzinfo=UTC)
    else:
        next_month = datetime(year, month + 1, 1, tzinfo=UTC)
    next_month_ms = int(next_month.timestamp() * 1000)
    last_expected = next_month_ms - INTERVAL_MS

    open_times: list[int] = []
    for row in data:
        if len(row) != 12:
            raise RuntimeError(f"field count mismatch: {symbol}:{period}")
        open_ms = int(row[0])
        open_times.append(open_ms)
        open_price = float(row[1])
        high = float(row[2])
        low = float(row[3])
        close = float(row[4])
        quote = float(row[7])
        taker = float(row[10])
        if not all(
            math.isfinite(value)
            for value in (open_price, high, low, close, quote, taker)
        ):
            raise RuntimeError(f"non-finite structural value: {symbol}:{period}")
        if min(open_price, high, low, close) <= 0.0:
            raise RuntimeError(f"non-positive OHLC: {symbol}:{period}")
        if low > min(open_price, close) or high < max(open_price, close) or low > high:
            raise RuntimeError(f"OHLC invariant failed: {symbol}:{period}")
        if quote < 0.0 or taker < 0.0 or taker > quote:
            raise RuntimeError(f"taker/quote invariant failed: {symbol}:{period}")

    if not open_times:
        raise RuntimeError(f"no data rows: {symbol}:{period}")
    if len(open_times) > expected_rows:
        raise RuntimeError(f"too many monthly rows: {symbol}:{period}")
    if any(b <= a for a, b in zip(open_times, open_times[1:], strict=False)):
        raise RuntimeError(f"open times not strictly increasing: {symbol}:{period}")
    if any(
        timestamp < first_expected
        or timestamp >= next_month_ms
        or (timestamp - first_expected) % INTERVAL_MS != 0
        for timestamp in open_times
    ):
        raise RuntimeError(f"off-grid monthly timestamp: {symbol}:{period}")
    if any(
        (b - a) % INTERVAL_MS != 0
        for a, b in zip(open_times, open_times[1:], strict=False)
    ):
        raise RuntimeError(f"non-hourly timestamp gap: {symbol}:{period}")
    missing_bars = expected_rows - len(open_times)

    return {
        "symbol": symbol,
        "month": period,
        "zip_relative_path": str(zip_path.relative_to(Path(os.environ["OUTPUT_ROOT"]))),
        "checksum_relative_path": str(
            checksum_path.relative_to(Path(os.environ["OUTPUT_ROOT"]))
        ),
        "zip_size_bytes": len(payload),
        "zip_sha256": sha256,
        "checksum_file_sha256": hashlib.sha256(checksum_bytes).hexdigest(),
        "checksum_claim_sha256": fields[0].lower(),
        "csv_member": members[0].filename,
        "header_present": header_present,
        "field_count": 12,
        "expected_row_count": expected_rows,
        "row_count": len(data),
        "missing_native_bar_count": missing_bars,
        "complete_month": missing_bars == 0,
        "first_expected_open_time_ms": first_expected,
        "last_expected_open_time_ms": last_expected,
        "first_observed_open_time_ms": open_times[0],
        "last_observed_open_time_ms": open_times[-1],
        "grid_spacing_ms": INTERVAL_MS,
        "structural_valid": True,
    }


def main() -> None:
    protocol = canonical_signed_taker_flow_protocol()
    if protocol.symbols != SYMBOLS:
        raise RuntimeError("protocol symbol roster drifted")
    if protocol.feature_lookback_bars != 24:
        raise RuntimeError("feature lookback drifted")
    if protocol.label_endpoint_offset_bars != 25:
        raise RuntimeError("label endpoint drifted")

    one_hour = timedelta(hours=1)
    required_raw_open_start = protocol.fit_start - protocol.feature_lookback_bars * one_hour
    required_raw_open_end_exclusive = protocol.fit_cutoff - one_hour
    if required_raw_open_start != datetime(2020, 12, 31, 1, tzinfo=UTC):
        raise RuntimeError("derived raw start changed")
    if required_raw_open_end_exclusive != datetime(2022, 12, 31, 23, tzinfo=UTC):
        raise RuntimeError("derived raw end changed")
    months = _months(required_raw_open_start, required_raw_open_end_exclusive - one_hour)
    if months[0] != "2020-12" or months[-1] != "2022-12" or len(months) != 25:
        raise RuntimeError("derived monthly archive roster changed")

    output_root = Path(os.environ["OUTPUT_ROOT"])
    raw_root = output_root / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    for symbol in SYMBOLS:
        for period in months:
            month = _month_dt(period)
            url = vision_monthly_kline_url("usds-m", symbol, "1h", month)
            name = url.rsplit("/", 1)[-1]
            zip_path = raw_root / symbol / name
            checksum_path = zip_path.with_name(name + ".CHECKSUM")
            _curl(url, zip_path)
            _curl(url + ".CHECKSUM", checksum_path)
            entry = _inspect_archive(
                symbol=symbol,
                period=period,
                zip_path=zip_path,
                checksum_path=checksum_path,
            )
            entry["zip_url"] = url
            entry["checksum_url"] = url + ".CHECKSUM"
            entries.append(entry)

    if len(entries) != 125 or any(not bool(item["structural_valid"]) for item in entries):
        raise RuntimeError("source bundle archive contract failed")
    gap_entries = [item for item in entries if int(item["missing_native_bar_count"]) > 0]
    total_missing = sum(int(item["missing_native_bar_count"]) for item in entries)

    body: dict[str, object] = {
        "schema_version": "signed_taker_flow_training_source_bundle_v1",
        "issue_number": 562,
        "calibration_head": CALIBRATION_HEAD,
        "exact_head_full_ci_run_id": FULL_CI_RUN_ID,
        "protocol_digest": protocol.digest,
        "source_issue": 556,
        "source_publisher_run_id": SOURCE_PUBLISHER_RUN_ID,
        "source_artifact_id": SOURCE_ARTIFACT_ID,
        "source_artifact_api_digest": SOURCE_ARTIFACT_API_DIGEST,
        "source_report_content_digest": SOURCE_REPORT_CONTENT_DIGEST,
        "source_fresh_run_id": SOURCE_FRESH_RUN_ID,
        "source_fresh_artifact_id": SOURCE_FRESH_ARTIFACT_ID,
        "source_fresh_artifact_api_digest": SOURCE_FRESH_ARTIFACT_API_DIGEST,
        "source_status": "PASS",
        "bundle_validation_status": (
            "PASS_WITH_RECORDED_NATIVE_GAPS" if gap_entries else "PASS_COMPLETE_CLOCKS"
        ),
        "market": "usds-m",
        "archive_family": "data.binance.vision futures/um/monthly/klines",
        "interval": "1h",
        "symbols": list(SYMBOLS),
        "months": list(months),
        "planned_archives": 125,
        "archives_with_native_gaps": len(gap_entries),
        "total_missing_native_bars": total_missing,
        "native_gap_policy": "preserve official gaps; no imputation/source substitution; builder regular clock marks missing rows unavailable",
        "required_raw_open_start_inclusive": _iso(required_raw_open_start),
        "required_raw_open_end_exclusive": _iso(required_raw_open_end_exclusive),
        "fit_start": _iso(protocol.fit_start),
        "fit_cutoff": _iso(protocol.fit_cutoff),
        "feature_lookback_bars": protocol.feature_lookback_bars,
        "label_execution_offset_bars": protocol.label_execution_offset_bars,
        "label_endpoint_offset_bars": protocol.label_endpoint_offset_bars,
        "archives": entries,
        "feature_values_reported": False,
        "target_relation_computed": False,
        "evaluation_pnl_inspected": False,
        "final_test_accessed": False,
        "production_eligible": False,
        "live_trading_authorized": False,
    }
    manifest = {**body, "content_digest": content_digest(body)}
    manifest_path = output_root / "source-manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    print(
        json.dumps(
            {
                "source_manifest_digest": manifest["content_digest"],
                "archive_count": len(entries),
                "archives_with_native_gaps": len(gap_entries),
                "total_missing_native_bars": total_missing,
                "first_month": months[0],
                "last_month": months[-1],
                "target_relation_computed": False,
                "evaluation_pnl_inspected": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
