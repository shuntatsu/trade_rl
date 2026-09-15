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

from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.experiments.bootstrap.spot_flow_continuation_diagnostic import (
    parse_usdm_15m_klines_csv,
)
from trade_rl.evaluation.experiments.bootstrap.spot_flow_continuation_prereg import (
    canonical_spot_flow_continuation_protocol,
)

SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
SOURCE_DATES = ("2021-01-15", "2021-07-15", "2022-01-15", "2022-07-15")
INTERVAL_MS = 15 * 60 * 1_000
EXPECTED_ROWS_PER_DAY = 96
MIN_TARGET_WINDOWS = 360


def _day_start_ms(date: str) -> int:
    return int(
        datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=UTC).timestamp() * 1000
    )


def _next_date(date: str) -> str:
    value = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=UTC) + timedelta(days=1)
    return value.strftime("%Y-%m-%d")


def target_dates() -> tuple[str, ...]:
    result: list[str] = []
    for source_date in SOURCE_DATES:
        result.extend((source_date, _next_date(source_date)))
    return tuple(result)


def _archive_url(symbol: str, date: str) -> str:
    return (
        "https://data.binance.vision/data/futures/um/daily/klines/"
        f"{symbol}/15m/{symbol}-15m-{date}.zip"
    )


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
        raise ValueError("archive transport failed") from error


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
        raise ValueError("checksum filename does not match planned archive")
    return digest


def _missing_digest(missing: list[int]) -> str:
    return hashlib.sha256(canonical_json_bytes(missing)).hexdigest()


def _validate_archive(symbol: str, date: str) -> tuple[dict[str, object], set[int]]:
    filename = f"{symbol}-15m-{date}.zip"
    expected_member = f"{symbol}-15m-{date}.csv"
    url = _archive_url(symbol, date)
    checksum_url = url + ".CHECKSUM"
    raw_zip = _download(url)
    raw_checksum = _download(checksum_url)
    raw_sha256 = hashlib.sha256(raw_zip).hexdigest()
    expected_sha256 = _checksum_digest(raw_checksum, expected_filename=filename)
    if raw_sha256 != expected_sha256:
        raise ValueError("archive SHA-256 does not match sibling checksum")

    with zipfile.ZipFile(io.BytesIO(raw_zip)) as archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        if len(members) != 1 or members[0].filename != expected_member:
            raise ValueError("ZIP member roster differs from planned CSV member")
        csv_bytes = archive.read(members[0])

    bars = parse_usdm_15m_klines_csv(csv_bytes, expected_date=date)
    observed = {bar.open_time_ms for bar in bars}
    if len(observed) != len(bars):
        raise ValueError("target archive contains duplicate open timestamps")
    start = _day_start_ms(date)
    expected = {start + index * INTERVAL_MS for index in range(EXPECTED_ROWS_PER_DAY)}
    if not observed.issubset(expected):
        raise ValueError(
            "target archive contains timestamps outside planned native grid"
        )
    missing = sorted(expected - observed)
    record: dict[str, object] = {
        "symbol": symbol,
        "date": date,
        "url": url,
        "checksum_url": checksum_url,
        "zip_filename": filename,
        "csv_member": expected_member,
        "zip_size": len(raw_zip),
        "zip_sha256": raw_sha256,
        "checksum_sha256": expected_sha256,
        "checksum_verified": True,
        "row_count": len(bars),
        "expected_row_count": EXPECTED_ROWS_PER_DAY,
        "missing_grid_rows": len(missing),
        "missing_grid_digest": _missing_digest(missing),
        "strict_15m_grid": True,
        "structurally_valid": True,
    }
    return record, observed


def _target_window_count(source_date: str, observed: set[int]) -> int:
    start = _day_start_ms(source_date)
    count = 0
    for decision_index in range(1, 97):
        decision = start + decision_index * INTERVAL_MS
        required = {decision + offset * INTERVAL_MS for offset in range(17)}
        if required.issubset(observed):
            count += 1
    return count


def build_report() -> tuple[dict[str, object], dict[str, object]]:
    protocol = canonical_spot_flow_continuation_protocol()
    if protocol.symbols != SYMBOLS or protocol.spot_dates != SOURCE_DATES:
        raise ValueError("runtime preregistration roster differs from target preflight")
    if protocol.target_timeframe != "15m":
        raise ValueError("runtime preregistration target timeframe differs")
    if protocol.target_source_family != "binance_vision_contract_klines":
        raise ValueError("runtime preregistration target source family differs")
    if protocol.target_transport_mode != "VISION":
        raise ValueError("runtime preregistration target transport differs")
    if (
        protocol.target_source_fallback_allowed
        or protocol.target_source_replacement_allowed
    ):
        raise ValueError("runtime preregistration permits forbidden target fallback")

    records: list[dict[str, object]] = []
    by_symbol_date: dict[tuple[str, str], set[int]] = {}
    errors: list[str] = []
    for symbol in SYMBOLS:
        for date in target_dates():
            try:
                record, observed = _validate_archive(symbol, date)
            except Exception as error:  # fail closed but preserve structural diagnosis
                records.append(
                    {
                        "symbol": symbol,
                        "date": date,
                        "url": _archive_url(symbol, date),
                        "checksum_url": _archive_url(symbol, date) + ".CHECKSUM",
                        "available_and_valid": False,
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    }
                )
                errors.append(f"{symbol}:{date}:{type(error).__name__}")
                continue
            record["available_and_valid"] = True
            records.append(record)
            by_symbol_date[(symbol, date)] = observed

    target_windows: dict[str, int] = {}
    for symbol in SYMBOLS:
        total = 0
        for source_date in SOURCE_DATES:
            combined = set(by_symbol_date.get((symbol, source_date), set()))
            combined.update(
                by_symbol_date.get((symbol, _next_date(source_date)), set())
            )
            total += _target_window_count(source_date, combined)
        target_windows[symbol] = total

    all_archives_valid = len(errors) == 0 and len(records) == 40
    coverage_valid = all(
        value >= MIN_TARGET_WINDOWS for value in target_windows.values()
    )
    if all_archives_valid and coverage_valid:
        status = "PASS_TARGET_SOURCE_PREFLIGHT"
    elif all_archives_valid:
        status = "INVALID_TARGET_SOURCE_COVERAGE"
    else:
        status = "INCOMPATIBLE_TARGET_SOURCE"

    manifest: dict[str, object] = {
        "schema_version": "issue584_target_source_manifest_v1",
        "issue_number": 584,
        "prereg_head": "89ec1092e438d69657840567e684739ca0c4e3d5",
        "prereg_protocol_digest": protocol.digest,
        "symbols": list(SYMBOLS),
        "source_dates": list(SOURCE_DATES),
        "target_dates": list(target_dates()),
        "target_source_family": "binance_vision_contract_klines",
        "target_transport_mode": "VISION",
        "archive_records": records,
        "fallback_used": False,
        "replacement_source_used": False,
    }
    manifest_digest = content_digest(manifest)
    report_without_digest: dict[str, object] = {
        "schema_version": "issue584_target_source_preflight_v1",
        "issue_number": 584,
        "prereg_head": "89ec1092e438d69657840567e684739ca0c4e3d5",
        "prereg_protocol_digest": protocol.digest,
        "planned_archives": 40,
        "valid_archives": sum(
            item.get("available_and_valid") is True for item in records
        ),
        "target_windows_by_symbol": target_windows,
        "minimum_target_windows_per_symbol": MIN_TARGET_WINDOWS,
        "status": status,
        "failures": errors,
        "target_manifest_digest": manifest_digest,
        "economic_values_reported": False,
        "target_return_computed": False,
        "spot_flow_computed": False,
        "training_relation_computed": False,
        "evaluation_pnl_inspected": False,
        "fallback_used": False,
        "replacement_source_used": False,
    }
    report = {
        **report_without_digest,
        "content_digest": content_digest(report_without_digest),
    }
    return report, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    report, manifest = build_report()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_bytes(canonical_json_bytes(report))
    args.manifest.write_bytes(canonical_json_bytes(manifest))
    print("STATUS=" + str(report["status"]))
    print("PLANNED_ARCHIVES=" + str(report["planned_archives"]))
    print("VALID_ARCHIVES=" + str(report["valid_archives"]))
    print(
        "TARGET_WINDOWS="
        + json.dumps(report["target_windows_by_symbol"], sort_keys=True)
    )
    print("TARGET_MANIFEST_DIGEST=" + str(report["target_manifest_digest"]))
    print("ECONOMIC_VALUES_REPORTED=false")
    print("TARGET_RETURN_COMPUTED=false")
    print("TRAINING_RELATION_COMPUTED=false")


if __name__ == "__main__":
    main()
