from __future__ import annotations

import argparse
import calendar
import hashlib
import json
from pathlib import Path

SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
MONTHS = tuple(
    f"{year}-{month:02d}" for year in (2021, 2022) for month in range(1, 13)
)
EXPECTED_ENTRY_FIELDS = {
    "checksum_expected_sha256",
    "checksum_url",
    "checksum_verified",
    "csv_member",
    "expected_grid_rows",
    "field_count",
    "first_open_time",
    "header_present",
    "kind",
    "last_open_time",
    "missing_grid_open_times_sha256",
    "missing_grid_rows",
    "month",
    "on_requested_1h_grid",
    "raw_sha256",
    "raw_size_bytes",
    "row_count",
    "structurally_valid",
    "symbol",
    "url",
}

PREREG_HEAD = "6ffc414baa10e91f34df258fe5cfabacce65fd77"
PREREG_PROTOCOL_DIGEST = (
    "18bf9625502eccbe475d157df90635f1c4645d3c4cced48ec0fc565723206731"
)
PREREG_FULL_VERIFY_RUN_ID = 34959430185
PREREG_SEAL_RUN_ID = 34959849347
PREREG_SEAL_ARTIFACT_ID = 10392936042
PREREG_SEAL_ARTIFACT_DIGEST = (
    "sha256:4b611462f8d4f9bb4d23ab15bb4d8a20525d4093e590c1afb62d1b74ca875e15"
)
PREREG_FRESH_ARTIFACT_ID = 10391714897
PREREG_FRESH_ARTIFACT_DIGEST = (
    "sha256:858e930a03e442f047556801c3d9a536fbc89435b5f261cba621988dd8062acd"
)

SOURCE_RUN_ID = 34908073064
SOURCE_PUBLISHER_ARTIFACT_ID = 10373058552
SOURCE_PUBLISHER_ARTIFACT_DIGEST = (
    "sha256:d6f5e6eaa7aa46b21b0f41dd75a0dec9022cebc839df881c4d8c5307dc89ab8d"
)
SOURCE_FRESH_ARTIFACT_ID = 10373388372
SOURCE_FRESH_ARTIFACT_DIGEST = (
    "sha256:513e183bdc23ef823f29cfc5a93f2803c997d51b1eb243e1fbefb1381378b407"
)
SOURCE_MANIFEST_SHA256 = (
    "834d4ab7b02ac0c97a0dadcb75c4c485e5892d0e9d66c13cfb998dbfea045706"
)
SOURCE_MANIFEST_CONTENT_DIGEST = (
    "92948941feb1edd8f561b3f505c8e3b30e529caeb0b48cf62d6aaa621334ee51"
)


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def content_digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def require_bool(value: object, *, field: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field} must be boolean")
    return value


def require_int(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be integer >= {minimum}")
    return value


def require_hex(value: object, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
        raise ValueError(f"{field} must be lowercase sha256")
    if any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field} must be lowercase sha256")
    return value


def load_identical(publisher: Path, fresh: Path) -> dict[str, object]:
    publisher_bytes = publisher.read_bytes()
    fresh_bytes = fresh.read_bytes()
    if publisher_bytes != fresh_bytes:
        raise ValueError("Issue 577 publisher/fresh source manifests differ")
    if hashlib.sha256(publisher_bytes).hexdigest() != SOURCE_MANIFEST_SHA256:
        raise ValueError("Issue 577 source manifest SHA-256 differs from authority")
    try:
        decoded = json.loads(publisher_bytes)
    except json.JSONDecodeError as error:
        raise ValueError("Issue 577 source manifest is malformed") from error
    if not isinstance(decoded, dict):
        raise ValueError("Issue 577 source manifest must be an object")
    if publisher_bytes != canonical_json_bytes(decoded):
        raise ValueError("Issue 577 source manifest is not canonical JSON")
    unsigned = dict(decoded)
    digest = unsigned.pop("content_digest", None)
    if digest != SOURCE_MANIFEST_CONTENT_DIGEST or content_digest(unsigned) != digest:
        raise ValueError("Issue 577 source manifest content digest is invalid")
    return decoded


def expected_hours(month: str) -> int:
    year, month_number = (int(part) for part in month.split("-"))
    return calendar.monthrange(year, month_number)[1] * 24


def validate_manifest(source: dict[str, object]) -> list[dict[str, object]]:
    if source.get("schema_version") != "issue577_basis_training_source_manifest_v1":
        raise ValueError("unexpected Issue 577 source schema")
    if source.get("issue_number") != 577 or source.get("interval") != "1h":
        raise ValueError("unexpected Issue 577 source identity")
    if source.get("symbols") != list(SYMBOLS) or source.get("months") != list(MONTHS):
        raise ValueError("source roster differs from frozen target roster")
    if source.get("planned_archives") != 240:
        raise ValueError("combined source archive count differs")
    if require_bool(
        source.get("native_missing_rows_remain_unavailable"),
        field="native_missing_rows_remain_unavailable",
    ) is not True:
        raise ValueError("native missing rows must remain unavailable")
    for field in (
        "replacement_source_used",
        "rest_or_auto_fallback_used",
        "economic_values_logged",
        "evaluation_or_final_data_accessed",
        "evaluation_pnl_computed",
    ):
        if require_bool(source.get(field), field=field) is not False:
            raise ValueError(f"{field} must be false")
    expected_root = "https://data.binance.vision/data/futures/um/monthly/klines"
    if source.get("perpetual_archive_root") != expected_root:
        raise ValueError("unexpected perpetual archive root")

    raw_entries = source.get("entries")
    if not isinstance(raw_entries, list) or len(raw_entries) != 240:
        raise ValueError("source entries are malformed")
    entries = [
        item
        for item in raw_entries
        if isinstance(item, dict) and item.get("kind") == "perpetual_klines"
    ]
    if len(entries) != 120:
        raise ValueError("expected exactly 120 perpetual kline entries")
    expected_pairs = [(symbol, month) for symbol in SYMBOLS for month in MONTHS]
    observed_pairs = [(item.get("symbol"), item.get("month")) for item in entries]
    if observed_pairs != expected_pairs:
        raise ValueError("perpetual target archive ordering/roster differs")

    missing_by_symbol = {symbol: 0 for symbol in SYMBOLS}
    for entry in entries:
        if set(entry) != EXPECTED_ENTRY_FIELDS:
            raise ValueError("perpetual entry fields differ from structural allowlist")
        symbol = entry["symbol"]
        month = entry["month"]
        if not isinstance(symbol, str) or not isinstance(month, str):
            raise ValueError("target entry symbol/month are malformed")
        url = (
            "https://data.binance.vision/data/futures/um/monthly/klines/"
            f"{symbol}/1h/{symbol}-1h-{month}.zip"
        )
        if entry["url"] != url or entry["checksum_url"] != url + ".CHECKSUM":
            raise ValueError("target archive URL authority differs")
        if entry["csv_member"] != f"{symbol}-1h-{month}.csv":
            raise ValueError("target CSV member authority differs")
        if require_bool(entry["checksum_verified"], field="checksum_verified") is not True:
            raise ValueError("target checksum is not verified")
        if require_bool(entry["structurally_valid"], field="structurally_valid") is not True:
            raise ValueError("target archive is not structurally valid")
        if require_bool(entry["on_requested_1h_grid"], field="on_requested_1h_grid") is not True:
            raise ValueError("target archive contains off-grid rows")
        require_bool(entry["header_present"], field="header_present")
        if require_int(entry["field_count"], field="field_count") != 12:
            raise ValueError("target kline field count differs")
        expected = expected_hours(month)
        if require_int(
            entry["expected_grid_rows"], field="expected_grid_rows"
        ) != expected:
            raise ValueError("expected monthly native-grid row count differs")
        row_count = require_int(entry["row_count"], field="row_count")
        missing = require_int(entry["missing_grid_rows"], field="missing_grid_rows")
        if row_count + missing != expected:
            raise ValueError("row_count + missing_grid_rows does not close monthly grid")
        require_int(entry["raw_size_bytes"], field="raw_size_bytes", minimum=1)
        raw_sha = require_hex(entry["raw_sha256"], field="raw_sha256")
        if require_hex(
            entry["checksum_expected_sha256"], field="checksum_expected_sha256"
        ) != raw_sha:
            raise ValueError("raw SHA does not equal checksum authority")
        require_hex(
            entry["missing_grid_open_times_sha256"],
            field="missing_grid_open_times_sha256",
        )
        if not isinstance(entry["first_open_time"], str) or not isinstance(
            entry["last_open_time"], str
        ):
            raise ValueError("target open-time bounds are malformed")
        missing_by_symbol[symbol] += missing

    if source.get("missing_contract_grid_rows_by_symbol") != missing_by_symbol:
        raise ValueError("source missing-row summary differs from target subset")
    return entries


def build_authority(
    source: dict[str, object], entries: list[dict[str, object]]
) -> dict[str, object]:
    missing_by_symbol = {symbol: 0 for symbol in SYMBOLS}
    for entry in entries:
        missing_by_symbol[str(entry["symbol"])] += int(entry["missing_grid_rows"])
    payload: dict[str, object] = {
        "schema_version": "issue600_premium_target_source_authority_v1",
        "issue_number": 600,
        "status": "PASS_PREMIUM_TARGET_SOURCE_AUTHORITY",
        "prereg_head": PREREG_HEAD,
        "prereg_protocol_digest": PREREG_PROTOCOL_DIGEST,
        "prereg_full_verification_run_id": PREREG_FULL_VERIFY_RUN_ID,
        "prereg_seal_run_id": PREREG_SEAL_RUN_ID,
        "prereg_seal_artifact_id": PREREG_SEAL_ARTIFACT_ID,
        "prereg_seal_artifact_api_digest": PREREG_SEAL_ARTIFACT_DIGEST,
        "prereg_fresh_artifact_id": PREREG_FRESH_ARTIFACT_ID,
        "prereg_fresh_artifact_api_digest": PREREG_FRESH_ARTIFACT_DIGEST,
        "adopted_from_issue": 577,
        "source_run_id": SOURCE_RUN_ID,
        "source_publisher_artifact_id": SOURCE_PUBLISHER_ARTIFACT_ID,
        "source_publisher_artifact_api_digest": SOURCE_PUBLISHER_ARTIFACT_DIGEST,
        "source_fresh_artifact_id": SOURCE_FRESH_ARTIFACT_ID,
        "source_fresh_artifact_api_digest": SOURCE_FRESH_ARTIFACT_DIGEST,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "source_manifest_content_digest": SOURCE_MANIFEST_CONTENT_DIGEST,
        "source_manifest_publisher_fresh_byte_equal": True,
        "source_subset_kind": "perpetual_klines",
        "target_source_market": "USD_M",
        "target_source_family": "klines",
        "target_interval": "1h",
        "archive_root": source["perpetual_archive_root"],
        "symbols": list(SYMBOLS),
        "months": list(MONTHS),
        "planned_target_archives": 120,
        "available_target_archives": 120,
        "checksum_verified_target_archives": 120,
        "structurally_valid_target_archives": 120,
        "missing_native_grid_rows_by_symbol": missing_by_symbol,
        "missing_native_grid_rows_total": sum(missing_by_symbol.values()),
        "native_missing_rows_remain_unavailable": True,
        "replacement_source_used": False,
        "rest_or_auto_fallback_used": False,
        "target_entries": entries,
        "premium_economic_values_inspected": False,
        "target_prices_or_returns_computed": False,
        "training_relation_executed": False,
        "evaluation_pnl_inspected": False,
        "final_test_authorized": False,
        "production_eligible": False,
        "live_trading_authorized": False,
        "market_data_network_fetch_performed_by_this_adoption": False,
    }
    return {**payload, "content_digest": content_digest(payload)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--publisher-manifest", type=Path, required=True)
    parser.add_argument("--fresh-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = load_identical(args.publisher_manifest, args.fresh_manifest)
    entries = validate_manifest(source)
    authority = build_authority(source, entries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(authority))
    print("STATUS=" + str(authority["status"]))
    print("PLANNED_TARGET_ARCHIVES=" + str(authority["planned_target_archives"]))
    print(
        "MISSING_NATIVE_GRID_ROWS_TOTAL="
        + str(authority["missing_native_grid_rows_total"])
    )
    print("ECONOMIC_VALUES_INSPECTED=false")
    print("TARGET_RELATION_COMPUTED=false")


if __name__ == "__main__":
    main()
