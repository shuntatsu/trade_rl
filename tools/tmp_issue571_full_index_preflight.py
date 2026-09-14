from __future__ import annotations

import calendar
import csv
import hashlib
import io
import json
import math
import sys
import time
import urllib.error
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

ISSUE_NUMBER = 571
SCHEMA_VERSION = "issue571_full_index_preflight_v1"
BASE_MAIN_SHA = "c5a1ce8395feaedd8833f7e596fddc9d8f3115fc"
PREREG_PR_NUMBER = 572
PREREG_PR_HEAD = "b4cb6ac9f713021ffd141765812f12af31c2856e"
PREREG_PROTOCOL_DIGEST = (
    "dc729ea3d6221cd7306dfdccab71f0f50a221efbb45ce312e123c07840465013"
)
PREREG_SEAL_RUN_ID = 34866000147
PREREG_SEAL_ARTIFACT_ID = 10357270841
PREREG_SEAL_API_DIGEST = (
    "sha256:0a8bd8c1fcc4838ec0773b81a723062ebe438ac182ad3ac8dcb7e8da1428cea3"
)
PREREG_FRESH_ARTIFACT_ID = 10357655222
PREREG_FRESH_API_DIGEST = (
    "sha256:fdf3d453cc484d35b2c2193d843e5fb051289792ae9e832fc052e17f9e6e1358"
)
SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
MONTHS = tuple(
    f"{year:04d}-{month:02d}"
    for year in (2021, 2022)
    for month in range(1, 13)
)
INTERVAL = "1h"
INTERVAL_MS = 3_600_000
SOURCE_ROOT = (
    "https://data.binance.vision/data/futures/um/monthly/indexPriceKlines"
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _iso_ms(value: int) -> str:
    return datetime.fromtimestamp(value / 1000.0, tz=UTC).isoformat().replace(
        "+00:00", "Z"
    )


def _fetch(url: str) -> tuple[bytes | None, str | None]:
    last_error: str | None = None
    for attempt in range(3):
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "trade-rl-issue571-full-index-preflight/1"},
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return response.read(), None
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None, "http_404"
            last_error = f"http_{error.code}"
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last_error = type(error).__name__
        if attempt < 2:
            time.sleep(2**attempt)
    return None, last_error or "download_failed"


def _month_grid(month: str) -> tuple[list[int], int]:
    year, month_number = (int(value) for value in month.split("-"))
    days = calendar.monthrange(year, month_number)[1]
    start = int(datetime(year, month_number, 1, tzinfo=UTC).timestamp() * 1000)
    rows = days * 24
    return [start + offset * INTERVAL_MS for offset in range(rows)], rows


def _parse_checksum(raw: bytes, *, expected_name: str) -> tuple[str, bool]:
    text = raw.decode("utf-8").strip()
    parts = text.split()
    if len(parts) < 2:
        raise ValueError("checksum_file_malformed")
    digest = parts[0]
    if len(digest) != 64 or digest.lower() != digest:
        raise ValueError("checksum_digest_malformed")
    if any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("checksum_digest_malformed")
    observed_name = parts[-1].lstrip("*")
    return digest, observed_name == expected_name


def _probe_archive(symbol: str, month: str) -> dict[str, object]:
    zip_name = f"{symbol}-{INTERVAL}-{month}.zip"
    csv_name = f"{symbol}-{INTERVAL}-{month}.csv"
    url = f"{SOURCE_ROOT}/{symbol}/{INTERVAL}/{zip_name}"
    entry: dict[str, object] = {
        "symbol": symbol,
        "month": month,
        "url": url,
        "available": False,
        "checksum_available": False,
        "checksum_verified": False,
        "structurally_valid": False,
    }

    payload, error = _fetch(url)
    if payload is None:
        entry["reason"] = error or "archive_unavailable"
        return entry
    entry["available"] = True
    entry["raw_size"] = len(payload)
    entry["raw_sha256"] = _sha256(payload)

    checksum_bytes, checksum_error = _fetch(url + ".CHECKSUM")
    if checksum_bytes is None:
        entry["reason"] = checksum_error or "checksum_unavailable"
        return entry
    entry["checksum_available"] = True
    try:
        expected_sha, checksum_name_matches = _parse_checksum(
            checksum_bytes, expected_name=zip_name
        )
    except (UnicodeDecodeError, ValueError) as exc:
        entry["reason"] = f"{type(exc).__name__}:{exc}"
        return entry
    entry["checksum_expected_sha256"] = expected_sha
    entry["checksum_name_matches"] = checksum_name_matches
    entry["checksum_verified"] = bool(
        checksum_name_matches and expected_sha == entry["raw_sha256"]
    )
    if not entry["checksum_verified"]:
        entry["reason"] = "checksum_mismatch_or_name_mismatch"
        return entry

    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = [member for member in archive.infolist() if not member.is_dir()]
            if len(members) != 1:
                raise ValueError("archive_must_contain_exactly_one_regular_member")
            member = members[0]
            if member.filename != csv_name:
                raise ValueError("unexpected_csv_member_name")
            with archive.open(member, "r") as raw_csv:
                reader = csv.reader(
                    io.TextIOWrapper(raw_csv, encoding="utf-8-sig", newline="")
                )
                first = next(reader, None)
                if first is None:
                    raise ValueError("empty_csv")
                try:
                    int(first[0])
                    header_present = False
                    rows = iter((first, *reader))
                except (ValueError, IndexError):
                    header_present = True
                    if len(first) != 12:
                        raise ValueError("header_field_count_not_12")
                    rows = reader

                header = tuple(first) if header_present else ()
                open_times: list[int] = []
                row_count = 0
                for row in rows:
                    row_count += 1
                    if len(row) != 12:
                        raise ValueError("data_field_count_not_12")
                    open_ms = int(row[0])
                    close_ms = int(row[6])
                    if close_ms != open_ms + INTERVAL_MS - 1:
                        raise ValueError("close_time_contract_mismatch")
                    for index, text in enumerate(row):
                        if index in (0, 6):
                            continue
                        value = float(text)
                        if not math.isfinite(value):
                            raise ValueError(f"nonfinite_field_{index}")
                    if open_times and open_ms <= open_times[-1]:
                        raise ValueError("open_times_not_strictly_increasing")
                    open_times.append(open_ms)

        if not open_times:
            raise ValueError("no_data_rows")
        expected_grid, expected_rows = _month_grid(month)
        expected_set = set(expected_grid)
        observed_set = set(open_times)
        unexpected = sorted(observed_set - expected_set)
        if unexpected:
            raise ValueError("open_time_outside_requested_1h_month_grid")
        if len(observed_set) != len(open_times):
            raise ValueError("duplicate_open_time")
        missing = sorted(expected_set - observed_set)
        spacings = sorted(
            set(later - earlier for earlier, later in zip(open_times, open_times[1:]))
        )
        if any(value <= 0 or value % INTERVAL_MS != 0 for value in spacings):
            raise ValueError("observed_spacing_off_native_1h_grid")

        leading_missing = 0
        for timestamp in expected_grid:
            if timestamp in observed_set:
                break
            leading_missing += 1
        trailing_missing = 0
        for timestamp in reversed(expected_grid):
            if timestamp in observed_set:
                break
            trailing_missing += 1
        missing_bytes = "\n".join(str(value) for value in missing).encode("ascii")

        entry.update(
            {
                "csv_member": csv_name,
                "header_present": header_present,
                "header": list(header),
                "field_count": 12,
                "row_count": row_count,
                "expected_grid_rows": expected_rows,
                "first_open_time": _iso_ms(open_times[0]),
                "last_open_time": _iso_ms(open_times[-1]),
                "spacing_ms": spacings,
                "missing_grid_rows": len(missing),
                "leading_missing_grid_rows": leading_missing,
                "trailing_missing_grid_rows": trailing_missing,
                "missing_grid_open_times_sha256": _sha256(missing_bytes),
                "on_requested_1h_grid": True,
                "structurally_valid": True,
            }
        )
        return entry
    except (
        csv.Error,
        UnicodeDecodeError,
        ValueError,
        zipfile.BadZipFile,
    ) as exc:
        entry["reason"] = f"{type(exc).__name__}:{exc}"
        return entry


def _annual_summary(entries: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for symbol in SYMBOLS:
        for year in (2021, 2022):
            selected = [
                entry
                for entry in entries
                if entry["symbol"] == symbol
                and str(entry["month"]).startswith(f"{year:04d}-")
            ]
            expected = sum(int(entry.get("expected_grid_rows", 0)) for entry in selected)
            observed = sum(int(entry.get("row_count", 0)) for entry in selected)
            missing = sum(int(entry.get("missing_grid_rows", 0)) for entry in selected)
            valid_months = sum(bool(entry["structurally_valid"]) for entry in selected)
            summary.append(
                {
                    "symbol": symbol,
                    "year": year,
                    "planned_months": 12,
                    "valid_months": valid_months,
                    "expected_grid_rows": expected,
                    "observed_rows": observed,
                    "missing_grid_rows": missing,
                }
            )
    return summary


def build_report(*, probe_commit: str) -> dict[str, object]:
    entries = [
        _probe_archive(symbol, month) for symbol in SYMBOLS for month in MONTHS
    ]
    available = sum(bool(entry["available"]) for entry in entries)
    checksum_verified = sum(bool(entry["checksum_verified"]) for entry in entries)
    valid_entries = [entry for entry in entries if entry["structurally_valid"]]
    headers = {
        tuple(entry.get("header", []))
        for entry in valid_entries
        if bool(entry.get("header_present"))
    }
    header_presence_modes = sorted(
        {bool(entry.get("header_present")) for entry in valid_entries}
    )
    field_counts = {int(entry["field_count"]) for entry in valid_entries}
    normalization_compatible = len(headers) <= 1 and field_counts in ({12}, set())
    total_missing = sum(int(entry.get("missing_grid_rows", 0)) for entry in valid_entries)
    failures = [
        f"{entry['symbol']}:{entry['month']}:{entry.get('reason', 'invalid')}"
        for entry in entries
        if not bool(entry["structurally_valid"])
    ]

    if (
        len(entries) == 120
        and available == 120
        and checksum_verified == 120
        and len(valid_entries) == 120
        and normalization_compatible
    ):
        status = "PASS_FULL_INDEX_PREFLIGHT"
    elif valid_entries and normalization_compatible:
        status = "PARTIAL_FULL_INDEX_PREFLIGHT"
    else:
        status = "INCOMPATIBLE_FULL_INDEX_PREFLIGHT"

    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "issue_number": ISSUE_NUMBER,
        "base_main_sha": BASE_MAIN_SHA,
        "probe_commit": probe_commit,
        "prereg_pr_number": PREREG_PR_NUMBER,
        "prereg_pr_head": PREREG_PR_HEAD,
        "prereg_protocol_digest": PREREG_PROTOCOL_DIGEST,
        "prereg_seal_run_id": PREREG_SEAL_RUN_ID,
        "prereg_seal_artifact_id": PREREG_SEAL_ARTIFACT_ID,
        "prereg_seal_api_digest": PREREG_SEAL_API_DIGEST,
        "prereg_fresh_artifact_id": PREREG_FRESH_ARTIFACT_ID,
        "prereg_fresh_api_digest": PREREG_FRESH_API_DIGEST,
        "source_family": "binance_vision_usdm_monthly_indexPriceKlines_1h",
        "source_root": SOURCE_ROOT,
        "rest_semantic_authority": "/fapi/v1/indexPriceKlines",
        "symbols": list(SYMBOLS),
        "months": list(MONTHS),
        "interval": INTERVAL,
        "planned_archives": 120,
        "available_archives": available,
        "checksum_verified_archives": checksum_verified,
        "valid_archives": len(valid_entries),
        "distinct_headers": [list(value) for value in sorted(headers)],
        "header_presence_modes": header_presence_modes,
        "distinct_field_counts": sorted(field_counts),
        "normalization_compatible": normalization_compatible,
        "total_missing_grid_rows": total_missing,
        "annual_summary": _annual_summary(entries),
        "structural_failures": failures,
        "status": status,
        "sparse_rows_remain_unavailable": True,
        "replacement_source_used": False,
        "economic_values_inspected": False,
        "basis_or_return_computed": False,
        "target_relation_computed": False,
        "evaluation_pnl_inspected": False,
        "production_authorized": False,
        "entries": entries,
    }
    payload["content_digest"] = _sha256(_canonical_bytes(payload))
    return payload


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: preflight.py <output-path> <probe-commit>")
    output = Path(sys.argv[1])
    probe_commit = sys.argv[2]
    if len(probe_commit) != 40 or any(
        char not in "0123456789abcdef" for char in probe_commit
    ):
        raise SystemExit("probe commit must be a lowercase 40-character git sha")
    report = build_report(probe_commit=probe_commit)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_canonical_bytes(report) + b"\n")
    summary = {
        "status": report["status"],
        "planned_archives": report["planned_archives"],
        "available_archives": report["available_archives"],
        "checksum_verified_archives": report["checksum_verified_archives"],
        "valid_archives": report["valid_archives"],
        "total_missing_grid_rows": report["total_missing_grid_rows"],
        "distinct_field_counts": report["distinct_field_counts"],
        "header_presence_modes": report["header_presence_modes"],
        "structural_failures": report["structural_failures"],
        "annual_summary": report["annual_summary"],
        "content_digest": report["content_digest"],
    }
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
