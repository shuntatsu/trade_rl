from __future__ import annotations

import argparse
import calendar
import csv
import hashlib
import io
import json
import math
import time
import urllib.error
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

ISSUE_NUMBER = 570
SCHEMA_VERSION = "issue570_premium_sparse_source_v2"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
YEARS = (2021, 2022)
MONTHS = tuple(f"{year}-{month:02d}" for year in YEARS for month in range(1, 13))
ROOT = "https://data.binance.vision/data/futures/um/monthly/premiumIndexKlines"
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
HOUR_MS = 3_600_000
ANNUAL_MIN_COVERAGE = 0.95
SOURCE_PARENT_ISSUE = 555
SOURCE_PARENT_REPORT_DIGEST = "628d75b64b027a0c6161c87f989d90bf81c989e272bb812def8c7b4c8e3ccfdb"
USER_AGENT = "trade-rl-issue570-premium-sparse-source/2"


def _fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError) as error:
            last_error = error
            if attempt < 4:
                time.sleep(2.0 + 2.0 * attempt)
    assert last_error is not None
    raise last_error


def _month_bounds(month: str) -> tuple[int, int, int]:
    year, mon = map(int, month.split("-"))
    first = datetime(year, mon, 1, tzinfo=UTC)
    if mon == 12:
        next_month = datetime(year + 1, 1, 1, tzinfo=UTC)
    else:
        next_month = datetime(year, mon + 1, 1, tzinfo=UTC)
    first_ms = int(first.timestamp() * 1000)
    next_ms = int(next_month.timestamp() * 1000)
    return first_ms, next_ms, calendar.monthrange(year, mon)[1] * 24


def _iso_ms(value: int) -> str:
    return datetime.fromtimestamp(value / 1000.0, tz=UTC).isoformat().replace("+00:00", "Z")


def _max_missing_run(missing: list[int]) -> int:
    if not missing:
        return 0
    maximum = 1
    run = 1
    for left, right in zip(missing, missing[1:], strict=False):
        if right - left == HOUR_MS:
            run += 1
            maximum = max(maximum, run)
        else:
            run = 1
    return maximum


def _parse_archive(*, symbol: str, month: str, payload: bytes) -> dict[str, object]:
    del symbol
    first_ms, next_ms, expected_rows = _month_bounds(month)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = [member for member in archive.infolist() if not member.is_dir()]
        if len(members) != 1 or not members[0].filename.lower().endswith(".csv"):
            raise ValueError("archive must contain exactly one CSV member")
        member_name = members[0].filename
        with archive.open(members[0], "r") as raw:
            reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
            rows = list(reader)

    if not rows:
        raise ValueError("CSV is empty")
    first_row = rows[0]
    try:
        int(first_row[0])
    except (ValueError, IndexError):
        header_present = True
        header = tuple(first_row)
        if header != EXPECTED_HEADER:
            raise ValueError(f"unexpected header: {header!r}")
        data = rows[1:]
    else:
        header_present = False
        header = ()
        data = rows

    if not data:
        raise ValueError("CSV has no data rows")

    open_times: list[int] = []
    for row_index, row in enumerate(data):
        if len(row) != 12:
            raise ValueError(f"row {row_index} has {len(row)} fields, expected 12")
        try:
            open_ms = int(row[0])
            close_ms = int(row[6])
        except ValueError as error:
            raise ValueError(f"row {row_index} timestamp is not an integer") from error
        if open_ms < 0 or close_ms < 0:
            raise ValueError(f"row {row_index} has negative timestamp")
        if open_ms % HOUR_MS != 0:
            raise ValueError(f"row {row_index} open timestamp is off the 1h grid")
        if not first_ms <= open_ms < next_ms:
            raise ValueError(f"row {row_index} open timestamp lies outside requested month")
        if close_ms < open_ms or close_ms >= open_ms + HOUR_MS:
            raise ValueError(f"row {row_index} close timestamp lies outside its 1h bar")
        for field_index, text in enumerate(row):
            if field_index in (0, 6):
                continue
            try:
                value = float(text)
            except ValueError as error:
                raise ValueError(
                    f"row {row_index} field {field_index} is not numeric"
                ) from error
            if not math.isfinite(value):
                raise ValueError(f"row {row_index} field {field_index} is non-finite")
        open_times.append(open_ms)

    if any(right <= left for left, right in zip(open_times, open_times[1:], strict=False)):
        raise ValueError("open timestamps are not strictly increasing and unique")
    spacings = [right - left for left, right in zip(open_times, open_times[1:], strict=False)]
    if any(spacing <= 0 or spacing % HOUR_MS != 0 for spacing in spacings):
        raise ValueError("open timestamp spacing is not a positive integer multiple of 1h")

    expected_grid = list(range(first_ms, next_ms, HOUR_MS))
    expected_set = set(expected_grid)
    observed_set = set(open_times)
    if len(observed_set) != len(open_times):
        raise ValueError("duplicate open timestamp")
    missing = [timestamp for timestamp in expected_grid if timestamp not in observed_set]
    unexpected = [timestamp for timestamp in open_times if timestamp not in expected_set]
    if unexpected:
        raise ValueError("observed timestamp lies outside nominal month grid")
    if len(open_times) + len(missing) != expected_rows:
        raise ValueError("observed + missing row accounting does not match month grid")

    return {
        "csv_member": member_name,
        "header_present": header_present,
        "header": list(header),
        "field_count": 12,
        "observed_rows": len(open_times),
        "expected_rows": expected_rows,
        "missing_grid_rows": len(missing),
        "max_missing_run_hours": _max_missing_run(missing),
        "first_observed_open_time": _iso_ms(open_times[0]),
        "last_observed_open_time": _iso_ms(open_times[-1]),
        "all_open_times_on_1h_grid": True,
        "strictly_increasing_unique": True,
        "all_non_time_fields_finite": True,
        "structurally_valid": True,
    }


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def build_report() -> dict[str, object]:
    entries: list[dict[str, object]] = []
    failures: list[str] = []
    annual_observed = {symbol: {year: 0 for year in YEARS} for symbol in SYMBOLS}
    annual_expected = {
        symbol: {
            year: sum(calendar.monthrange(year, month)[1] * 24 for month in range(1, 13))
            for year in YEARS
        }
        for symbol in SYMBOLS
    }

    for symbol in SYMBOLS:
        for month in MONTHS:
            year = int(month[:4])
            name = f"{symbol}-1h-{month}.zip"
            url = f"{ROOT}/{symbol}/1h/{name}"
            checksum_url = url + ".CHECKSUM"
            entry: dict[str, object] = {
                "symbol": symbol,
                "month": month,
                "url": url,
                "checksum_url": checksum_url,
            }
            try:
                payload = _fetch(url)
            except Exception as error:
                entry.update(
                    {
                        "available": False,
                        "checksum_available": False,
                        "checksum_verified": False,
                        "structurally_valid": False,
                        "reason": f"archive_download:{type(error).__name__}",
                    }
                )
                failures.append(f"{symbol}:{month}:archive_download")
                entries.append(entry)
                continue

            raw_sha = hashlib.sha256(payload).hexdigest()
            entry.update(
                {
                    "available": True,
                    "raw_size_bytes": len(payload),
                    "raw_sha256": raw_sha,
                }
            )
            try:
                checksum_payload = _fetch(checksum_url)
                checksum_text = checksum_payload.decode("utf-8").strip()
                checksum_fields = checksum_text.split()
                expected_sha = checksum_fields[0].lower() if checksum_fields else ""
                checksum_valid_shape = (
                    len(expected_sha) == 64
                    and all(character in "0123456789abcdef" for character in expected_sha)
                )
                checksum_verified = checksum_valid_shape and expected_sha == raw_sha
                entry.update(
                    {
                        "checksum_available": True,
                        "checksum_expected_sha256": expected_sha,
                        "checksum_verified": checksum_verified,
                    }
                )
                if not checksum_verified:
                    raise ValueError("checksum mismatch or malformed checksum")
            except Exception as error:
                entry.update(
                    {
                        "checksum_available": bool(entry.get("checksum_available", False)),
                        "checksum_verified": False,
                        "structurally_valid": False,
                        "reason": f"checksum:{type(error).__name__}",
                    }
                )
                failures.append(f"{symbol}:{month}:checksum")
                entries.append(entry)
                continue

            try:
                structural = _parse_archive(symbol=symbol, month=month, payload=payload)
            except Exception as error:
                entry.update(
                    {
                        "structurally_valid": False,
                        "reason": f"parse:{type(error).__name__}:{error}",
                    }
                )
                failures.append(f"{symbol}:{month}:parse:{type(error).__name__}")
            else:
                entry.update(structural)
                annual_observed[symbol][year] += int(structural["observed_rows"])
            entries.append(entry)

    annual_coverage: dict[str, dict[str, object]] = {}
    coverage_gate_pass = True
    for symbol in SYMBOLS:
        by_year: dict[str, object] = {}
        for year in YEARS:
            observed = annual_observed[symbol][year]
            expected = annual_expected[symbol][year]
            coverage = observed / expected
            passed = coverage >= ANNUAL_MIN_COVERAGE
            coverage_gate_pass = coverage_gate_pass and passed
            by_year[str(year)] = {
                "observed_rows": observed,
                "expected_rows": expected,
                "missing_grid_rows": expected - observed,
                "coverage_fraction": coverage,
                "passes_95pct_gate": passed,
            }
        annual_coverage[symbol] = by_year

    archive_count_ok = len(entries) == 120 and all(entry.get("available") is True for entry in entries)
    checksum_ok = len(entries) == 120 and all(entry.get("checksum_verified") is True for entry in entries)
    structural_ok = len(entries) == 120 and all(entry.get("structurally_valid") is True for entry in entries)
    header_modes = {
        "exact_header" if entry.get("header_present") is True else "headerless"
        for entry in entries
        if entry.get("structurally_valid") is True
    }
    # Both exact-header and headerless archives normalize to the same 12-field schema.
    header_ok = header_modes.issubset({"exact_header", "headerless"}) and bool(header_modes)

    if archive_count_ok and checksum_ok and structural_ok and header_ok and coverage_gate_pass:
        status = "PASS_SPARSE_SOURCE"
    elif header_ok and any(entry.get("structurally_valid") is True for entry in entries):
        status = "PARTIAL_SOURCE"
    else:
        status = "INCOMPATIBLE_SOURCE"

    body: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "issue_number": ISSUE_NUMBER,
        "source_parent_issue": SOURCE_PARENT_ISSUE,
        "source_parent_report_digest": SOURCE_PARENT_REPORT_DIGEST,
        "source_family": "Binance Vision USD-M monthly premiumIndexKlines",
        "interval": "1h",
        "symbols": list(SYMBOLS),
        "months": list(MONTHS),
        "planned_archives": 120,
        "annual_min_coverage_fraction": ANNUAL_MIN_COVERAGE,
        "allowed_header_modes": ["exact_header", "headerless"],
        "observed_header_modes": sorted(header_modes),
        "sparse_missing_rows_allowed": True,
        "missing_rows_imputed": False,
        "replacement_archives_allowed": False,
        "post_2022_data_used": False,
        "premium_value_distribution_inspected": False,
        "target_relation_computed": False,
        "strategy_pnl_computed": False,
        "archive_count_gate_passed": archive_count_ok,
        "checksum_gate_passed": checksum_ok,
        "structural_gate_passed": structural_ok,
        "header_compatibility_gate_passed": header_ok,
        "annual_coverage_gate_passed": coverage_gate_pass,
        "annual_coverage": annual_coverage,
        "failures": failures,
        "status": status,
        "entries": entries,
    }
    digest = hashlib.sha256(_canonical_bytes(body)).hexdigest()
    return {**body, "content_digest": digest}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = build_report()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_canonical_bytes(report))

    summary = {
        "status": report["status"],
        "content_digest": report["content_digest"],
        "planned_archives": report["planned_archives"],
        "archive_count_gate_passed": report["archive_count_gate_passed"],
        "checksum_gate_passed": report["checksum_gate_passed"],
        "structural_gate_passed": report["structural_gate_passed"],
        "header_compatibility_gate_passed": report["header_compatibility_gate_passed"],
        "annual_coverage_gate_passed": report["annual_coverage_gate_passed"],
        "annual_missing_grid_rows": {
            symbol: {
                year: report["annual_coverage"][symbol][year]["missing_grid_rows"]
                for year in ("2021", "2022")
            }
            for symbol in SYMBOLS
        },
        "premium_value_distribution_inspected": False,
        "target_relation_computed": False,
        "strategy_pnl_computed": False,
    }
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    if report["status"] == "INCOMPATIBLE_SOURCE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
