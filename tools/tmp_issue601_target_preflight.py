from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path

from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.experiments.bootstrap.issue601_target_source_validator import (
    TARGET_MONTHS,
    TARGET_SYMBOLS,
    TargetArchiveValidation,
    build_target_source_report,
    canonical_target_source_report_bytes,
    validate_target_archive_bytes,
)

_VALIDATOR_HEAD = "c707be853480e57362f6a3d64f116e4056c3553f"
_VALIDATOR_FULL_RUN_ID = 34961165095
_URL_TEMPLATE = (
    "https://data.binance.vision/data/futures/um/monthly/klines/"
    "{symbol}/1h/{symbol}-1h-{month}.zip"
)
_DOWNLOAD_ATTEMPTS = 3
_TIMEOUT_SECONDS = 90
_USER_AGENT = "trade-rl-issue601-structural-preflight/1.0"

_MANIFEST_RECORD_FIELDS = (
    "symbol",
    "month",
    "url",
    "checksum_url",
    "archive_available",
    "raw_zip_size_bytes",
    "raw_zip_sha256",
    "checksum_available",
    "checksum_digest",
    "checksum_verified",
    "schema_valid",
)
_FORBIDDEN_ECONOMIC_KEYS = {
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "return",
    "premium",
    "beta",
    "alpha",
    "pnl",
    "sharpe",
    "ic",
}


def archive_url(symbol: str, month: str) -> str:
    if symbol not in TARGET_SYMBOLS or month not in TARGET_MONTHS:
        raise ValueError("symbol/month is outside the frozen Issue 601 roster")
    return _URL_TEMPLATE.format(symbol=symbol, month=month)


def fetch_optional(url: str) -> bytes | None:
    """Fetch an exact frozen URL; only HTTP 404 maps to source-unavailable."""

    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    last_error: BaseException | None = None
    for attempt in range(_DOWNLOAD_ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
                if response.status != 200:
                    raise RuntimeError(f"unexpected HTTP status {response.status}")
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            last_error = error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last_error = error
        if attempt + 1 < _DOWNLOAD_ATTEMPTS:
            time.sleep(attempt + 1)
    raise RuntimeError(
        f"transport failed for frozen URL after {_DOWNLOAD_ATTEMPTS} attempts"
    ) from last_error


def retrieve_validations() -> list[TargetArchiveValidation]:
    validations: list[TargetArchiveValidation] = []
    for symbol in TARGET_SYMBOLS:
        for month in TARGET_MONTHS:
            url = archive_url(symbol, month)
            archive_bytes = fetch_optional(url)
            checksum_bytes = fetch_optional(url + ".CHECKSUM")
            validations.append(
                validate_target_archive_bytes(
                    symbol=symbol,
                    month=month,
                    archive_bytes=archive_bytes,
                    checksum_bytes=checksum_bytes,
                )
            )
    return validations


def _manifest_record(validation: TargetArchiveValidation) -> dict[str, object]:
    report = validation.report
    record = {field: report[field] for field in _MANIFEST_RECORD_FIELDS}
    if set(record) != set(_MANIFEST_RECORD_FIELDS):
        raise ValueError("manifest record fields differ")
    return record


def build_manifest(validations: Sequence[TargetArchiveValidation]) -> dict[str, object]:
    expected_count = len(TARGET_SYMBOLS) * len(TARGET_MONTHS)
    if len(validations) != expected_count:
        raise ValueError("manifest validation roster count differs")

    records = [_manifest_record(validation) for validation in validations]
    expected_pairs = [
        (symbol, month) for symbol in TARGET_SYMBOLS for month in TARGET_MONTHS
    ]
    observed_pairs = [(record["symbol"], record["month"]) for record in records]
    if observed_pairs != expected_pairs:
        raise ValueError("manifest validation roster order differs")

    unsigned: dict[str, object] = {
        "schema_version": "issue601_usdm_1h_target_manifest_v1",
        "issue_number": 601,
        "upstream_issue_number": 600,
        "validator_head": _VALIDATOR_HEAD,
        "validator_full_verification_run_id": _VALIDATOR_FULL_RUN_ID,
        "symbols": list(TARGET_SYMBOLS),
        "months": list(TARGET_MONTHS),
        "target_source_market": "USD_M",
        "target_source_family": "klines",
        "target_interval": "1h",
        "planned_archive_count": expected_count,
        "records": records,
        "fallback_used": False,
        "replacement_source_used": False,
        "economic_values_inspected": False,
        "target_price_values_inspected": False,
        "target_return_computed": False,
        "premium_values_inspected": False,
        "training_relation_computed": False,
        "evaluation_pnl_inspected": False,
        "final_test_authorized": False,
        "production_eligible": False,
        "live_trading_authorized": False,
    }
    return {**unsigned, "content_digest": content_digest(unsigned)}


def canonical_manifest_bytes(manifest: Mapping[str, object]) -> bytes:
    if manifest.get("schema_version") != "issue601_usdm_1h_target_manifest_v1":
        raise ValueError("manifest schema differs")
    if manifest.get("validator_head") != _VALIDATOR_HEAD:
        raise ValueError("manifest validator head differs")
    if manifest.get("validator_full_verification_run_id") != _VALIDATOR_FULL_RUN_ID:
        raise ValueError("manifest validator verification differs")
    if manifest.get("planned_archive_count") != 120:
        raise ValueError("manifest planned archive count differs")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != 120:
        raise ValueError("manifest records differ")
    expected_pairs = [
        (symbol, month) for symbol in TARGET_SYMBOLS for month in TARGET_MONTHS
    ]
    observed_pairs: list[tuple[object, object]] = []
    for raw in records:
        if not isinstance(raw, dict) or set(raw) != set(_MANIFEST_RECORD_FIELDS):
            raise ValueError("manifest record schema differs")
        observed_pairs.append((raw.get("symbol"), raw.get("month")))
    if observed_pairs != expected_pairs:
        raise ValueError("manifest record order differs")
    for field in (
        "fallback_used",
        "replacement_source_used",
        "economic_values_inspected",
        "target_price_values_inspected",
        "target_return_computed",
        "premium_values_inspected",
        "training_relation_computed",
        "evaluation_pnl_inspected",
        "final_test_authorized",
        "production_eligible",
        "live_trading_authorized",
    ):
        if manifest.get(field) is not False:
            raise ValueError(f"forbidden manifest boundary crossed: {field}")
    digest = manifest.get("content_digest")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("manifest content digest malformed")
    unsigned = dict(manifest)
    unsigned.pop("content_digest")
    if content_digest(unsigned) != digest:
        raise ValueError("manifest content digest mismatch")
    return canonical_json_bytes(dict(manifest))


def assert_result_blind(payload: object) -> None:
    if isinstance(payload, dict):
        overlap = _FORBIDDEN_ECONOMIC_KEYS.intersection(payload)
        if overlap:
            raise ValueError(f"forbidden economic keys emitted: {sorted(overlap)}")
        for value in payload.values():
            assert_result_blind(value)
    elif isinstance(payload, list):
        for value in payload:
            assert_result_blind(value)


def build_outputs(
    validations: Sequence[TargetArchiveValidation],
) -> tuple[bytes, bytes]:
    report = build_target_source_report(
        validations,
        validator_head=_VALIDATOR_HEAD,
        validator_verification_run_id=_VALIDATOR_FULL_RUN_ID,
    )
    manifest = build_manifest(validations)
    assert_result_blind(report)
    assert_result_blind(manifest)
    return canonical_target_source_report_bytes(report), canonical_manifest_bytes(
        manifest
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    validations = retrieve_validations()
    report_bytes, manifest_bytes = build_outputs(validations)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_bytes(report_bytes)
    args.manifest.write_bytes(manifest_bytes)

    report = json.loads(report_bytes)
    print(f"PLANNED_ARCHIVES={report['planned_archive_count']}")
    print("ECONOMIC_VALUES_INSPECTED=false")
    print("TARGET_PRICE_VALUES_INSPECTED=false")
    print("TARGET_RETURN_COMPUTED=false")
    print("PREMIUM_VALUES_INSPECTED=false")


if __name__ == "__main__":
    main()
