from __future__ import annotations

import argparse
import hashlib
import io
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from trade_rl.evaluation.experiments.bootstrap.issue602_target_source_validator import (
    TARGET_MONTHS,
    TARGET_SYMBOLS,
    build_target_source_report,
    canonical_target_source_report_bytes,
    load_target_source_report_bytes,
    validate_target_archive_bytes,
)

VALIDATOR_HEAD = "d914d49aca7a30bdbf8ac394dc492b3e19e9d61c"
VALIDATOR_FULL_VERIFY_RUN_ID = 34961124202
_USER_AGENT = "trade-rl-issue602-target-preflight/1.0"
_MAX_ATTEMPTS = 3


def _download_optional(url: str) -> bytes | None:
    """Return exact bytes, classify only HTTP 404 as missing, retry transport failures."""

    last_error: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                if response.status != 200:
                    raise ValueError(
                        f"unexpected HTTP status {response.status} for {url}"
                    )
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            last_error = error
        except (urllib.error.URLError, TimeoutError) as error:
            last_error = error
        if attempt + 1 < _MAX_ATTEMPTS:
            time.sleep(2**attempt)
    raise ValueError(
        f"target-source transport failed after retries: {url}"
    ) from last_error


def build_report() -> dict[str, object]:
    """Fetch the exact frozen 120-archive roster and build a structural-only report."""

    entries: list[dict[str, object]] = []
    for symbol in TARGET_SYMBOLS:
        for month in TARGET_MONTHS:
            url = (
                "https://data.binance.vision/data/futures/um/monthly/klines/"
                f"{symbol}/1h/{symbol}-1h-{month}.zip"
            )
            archive_bytes = _download_optional(url)
            checksum_bytes = _download_optional(url + ".CHECKSUM")
            entries.append(
                validate_target_archive_bytes(
                    symbol=symbol,
                    month=month,
                    archive_bytes=archive_bytes,
                    checksum_bytes=checksum_bytes,
                )
            )
    return build_target_source_report(
        entries,
        validator_head=VALIDATOR_HEAD,
        validator_verification_run_id=VALIDATOR_FULL_VERIFY_RUN_ID,
    )


def write_report(output: Path) -> None:
    report = build_report()
    raw = canonical_target_source_report_bytes(report)
    if load_target_source_report_bytes(raw) != report:
        raise AssertionError("strict target-source report reload differs")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(raw)
    print("STATUS=" + str(report["status"]))
    print("PLANNED_ARCHIVES=" + str(report["planned_archive_count"]))
    print("AVAILABLE_ARCHIVES=" + str(report["available_archive_count"]))
    print("CHECKSUM_VERIFIED=" + str(report["checksum_verified_count"]))
    print("STRUCTURALLY_VALID=" + str(report["structurally_valid_archive_count"]))
    print("COMPLETE_ARCHIVES=" + str(report["complete_archive_count"]))
    print("TOTAL_MISSING_GRID_ROWS=" + str(report["total_missing_grid_rows"]))
    print("REPORT_SHA256=" + hashlib.sha256(raw).hexdigest())
    print("REPORT_CONTENT_DIGEST=" + str(report["content_digest"]))
    print("PREMIUM_ECONOMIC_VALUES_INSPECTED=false")
    print("TARGET_RELATION_COMPUTED=false")
    print("EVALUATION_PNL_INSPECTED=false")


def self_check() -> None:
    symbol = TARGET_SYMBOLS[0]
    month = TARGET_MONTHS[0]
    start_ms = 1609459200000
    interval_ms = 60 * 60 * 1000
    rows = [
        [
            start_ms + index * interval_ms,
            "1",
            "1",
            "1",
            "1",
            "0",
            start_ms + (index + 1) * interval_ms - 1,
            "0",
            "0",
            "0",
            "0",
            "0",
        ]
        for index in range(31 * 24)
    ]
    body = (
        "\n".join(",".join(str(value) for value in row) for row in rows) + "\n"
    ).encode()
    stream = io.BytesIO()
    member = f"{symbol}-1h-{month}.csv"
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(member, body)
    payload = stream.getvalue()
    archive_name = f"{symbol}-1h-{month}.zip"
    checksum = f"{hashlib.sha256(payload).hexdigest()}  {archive_name}\n".encode()
    report = validate_target_archive_bytes(
        symbol=symbol,
        month=month,
        archive_bytes=payload,
        checksum_bytes=checksum,
    )
    if report["schema_valid"] is not True:
        raise AssertionError("helper synthetic archive did not validate")
    if report["checksum_verified"] is not True:
        raise AssertionError("helper synthetic checksum did not validate")
    if report["row_count"] != 744 or report["missing_grid_rows"] != 0:
        raise AssertionError("helper synthetic monthly grid differs")
    if report["dataset_timestamp_semantics"] != "completed_bar_close_boundary":
        raise AssertionError("helper completed-bar clock differs")
    print("SELF_CHECK=PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if args.output is None:
        parser.error("--output is required unless --self-check is used")
    write_report(args.output)


if __name__ == "__main__":
    main()
