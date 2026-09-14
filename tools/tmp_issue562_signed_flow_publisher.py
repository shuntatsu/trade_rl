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

import numpy as np

from trade_rl.data.build import MarketDatasetBuilder
from trade_rl.data.contracts import (
    FeatureKind,
    FeatureSpec,
    InstrumentContract,
    MarketBuildConfig,
    VolumeUnit,
)
from trade_rl.data.source import InMemoryMarketDataSource, RawMarketSeries
from trade_rl.evaluation.experiments.bootstrap.signed_taker_flow_calibration import (
    calibrate_signed_taker_flow,
)
from trade_rl.evaluation.experiments.bootstrap.signed_taker_flow_prereg import (
    canonical_signed_taker_flow_protocol,
)

CALIBRATION_HEAD = "817e3c96d702ffca3be74309bd68b31ef02643c5"
IMPLEMENTATION_HEAD = "adf835cf45936dbc40bf8d4f74879825ae211ef9"
PROTOCOL_DIGEST = "5fa9be305c948c4724e106ebd4f4090d2fc977da05d0884b7d4aa9a03df58407"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
ROOT = "https://data.binance.vision/data/futures/um/monthly/klines"
SOURCE_OPEN_START = datetime(2020, 12, 31, 1, tzinfo=UTC)
SOURCE_OPEN_END = datetime(2023, 1, 1, 0, tzinfo=UTC)
ANALYSIS_ACTIVE_FLOOR = datetime(2020, 12, 1, 0, tzinfo=UTC)
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
SOURCE_AUTHORITY = {
    "issue": 556,
    "publisher_run": 34840494198,
    "publisher_artifact": 10345533785,
    "publisher_api_digest": "689188893dec90b41819972e1cb151920ab198bf6bf27b1abeae2ab18e2fac0f",
    "publisher_content_digest": "e3b627cc8efa63623135512d344a883c1b1be8a0e1bc169c24605ec63abaa693",
    "fresh_run": 34840645375,
    "fresh_artifact": 10346140972,
    "fresh_api_digest": "16f91dcaab5f1914d19ee93934643cf27ad3ab94e1787cb04123b8dc3ad68a60",
    "status": "PASS",
}
USER_AGENT = "trade-rl-issue562-signed-flow-publisher/1"


def _months() -> tuple[str, ...]:
    return ("2020-12",) + tuple(
        f"{year}-{month:02d}"
        for year in (2021, 2022)
        for month in range(1, 13)
    )


def _epoch_ms(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def _iso_ms(value: int) -> str:
    return (
        datetime.fromtimestamp(value / 1000.0, tz=UTC)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _fetch_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error = "network_error"
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                payload = response.read()
            if not payload:
                raise RuntimeError(f"empty response: {url}")
            return payload
        except urllib.error.HTTPError as error:
            last_error = f"http_{error.code}"
        except (urllib.error.URLError, TimeoutError) as error:
            last_error = type(error).__name__
        if attempt < 4:
            time.sleep(2 + 2 * attempt)
    raise RuntimeError(f"download failed after retries ({last_error}): {url}")


def _checksum_digest(payload: bytes, *, url: str) -> tuple[str, str]:
    try:
        text = payload.decode("utf-8").strip()
    except UnicodeDecodeError as error:
        raise RuntimeError(f"checksum is not UTF-8: {url}") from error
    fields = text.split()
    if not fields:
        raise RuntimeError(f"checksum is empty: {url}")
    digest = fields[0].lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise RuntimeError(f"checksum digest is malformed: {url}")
    return digest, text


def _canonical_digest(payload: dict[str, object]) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _month_bounds(month: str) -> tuple[int, int, int]:
    year, month_number = map(int, month.split("-"))
    start = datetime(year, month_number, 1, tzinfo=UTC)
    if month_number == 12:
        stop = datetime(year + 1, 1, 1, tzinfo=UTC)
    else:
        stop = datetime(year, month_number + 1, 1, tzinfo=UTC)
    expected_rows = calendar.monthrange(year, month_number)[1] * 24
    return _epoch_ms(start), _epoch_ms(stop), expected_rows


def _parse_archive(
    *,
    symbol: str,
    month: str,
    payload: bytes,
    checksum_payload: bytes,
    url: str,
) -> tuple[dict[str, object], list[tuple[int, float, float, float, float, float, float]]]:
    raw_sha = hashlib.sha256(payload).hexdigest()
    checksum_sha, checksum_text = _checksum_digest(checksum_payload, url=url + ".CHECKSUM")
    if raw_sha != checksum_sha:
        raise RuntimeError(f"archive checksum mismatch: {url}")

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        if len(members) != 1 or not members[0].filename.lower().endswith(".csv"):
            raise RuntimeError(f"archive member contract failed: {url}")
        member = members[0].filename
        with archive.open(members[0], "r") as raw:
            rows = list(
                csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
            )
    if not rows:
        raise RuntimeError(f"empty CSV: {url}")
    first = rows[0]
    try:
        int(first[0])
        header_present = False
    except (ValueError, IndexError):
        header_present = True
    if header_present and tuple(first) != EXPECTED_HEADER:
        raise RuntimeError(f"header mismatch: {url}")
    data = rows[1:] if header_present else rows
    if not data:
        raise RuntimeError(f"no data rows: {url}")

    month_start_ms, next_month_ms, expected_rows = _month_bounds(month)
    parsed: list[tuple[int, float, float, float, float, float, float]] = []
    open_times: list[int] = []
    for row in data:
        if len(row) != 12:
            raise RuntimeError(f"field count mismatch: {url}: {len(row)}")
        open_ms = int(row[0])
        if not month_start_ms <= open_ms < next_month_ms:
            raise RuntimeError(f"row outside monthly grid: {url}")
        if (open_ms - month_start_ms) % 3_600_000 != 0:
            raise RuntimeError(f"row off 1h grid: {url}")
        open_price = float(row[1])
        high = float(row[2])
        low = float(row[3])
        close = float(row[4])
        quote_volume = float(row[7])
        taker_quote = float(row[10])
        numbers = (open_price, high, low, close, quote_volume, taker_quote)
        if not all(math.isfinite(value) for value in numbers):
            raise RuntimeError(f"non-finite kline field: {url}")
        if min(open_price, high, low, close) <= 0.0:
            raise RuntimeError(f"non-positive OHLC field: {url}")
        if quote_volume < 0.0 or taker_quote < 0.0 or taker_quote > quote_volume:
            raise RuntimeError(f"taker/quote structural bound failed: {url}")
        if low > min(open_price, close) or high < max(open_price, close) or low > high:
            raise RuntimeError(f"OHLC invariant failed: {url}")
        open_times.append(open_ms)
        if _epoch_ms(SOURCE_OPEN_START) <= open_ms < _epoch_ms(SOURCE_OPEN_END):
            parsed.append(
                (open_ms, open_price, high, low, close, quote_volume, taker_quote)
            )
    if any(b <= a for a, b in zip(open_times, open_times[1:])):
        raise RuntimeError(f"timestamps not strictly increasing: {url}")
    if len(set(open_times)) != len(open_times):
        raise RuntimeError(f"duplicate hourly timestamp: {url}")
    if len(open_times) > expected_rows:
        raise RuntimeError(f"more rows than monthly grid: {url}")
    missing_rows = expected_rows - len(open_times)

    entry: dict[str, object] = {
        "symbol": symbol,
        "month": month,
        "url": url,
        "checksum_url": url + ".CHECKSUM",
        "raw_sha256": raw_sha,
        "raw_size_bytes": len(payload),
        "checksum_sha256": checksum_sha,
        "checksum_text": checksum_text,
        "checksum_verified": True,
        "csv_member": member,
        "header_present": header_present,
        "expected_grid_rows": expected_rows,
        "row_count": len(open_times),
        "missing_grid_rows": missing_rows,
        "first_open_time": _iso_ms(open_times[0]),
        "last_open_time": _iso_ms(open_times[-1]),
        "field_count": 12,
        "timestamps_on_monthly_1h_grid": True,
        "structural_bounds_valid": True,
    }
    return entry, parsed


def _download_source() -> tuple[list[dict[str, object]], dict[str, list[tuple[int, float, float, float, float, float, float]]]]:
    entries: list[dict[str, object]] = []
    rows_by_symbol: dict[
        str, list[tuple[int, float, float, float, float, float, float]]
    ] = {symbol: [] for symbol in SYMBOLS}
    for symbol in SYMBOLS:
        for month in _months():
            filename = f"{symbol}-1h-{month}.zip"
            url = f"{ROOT}/{symbol}/1h/{filename}"
            payload = _fetch_bytes(url)
            checksum_payload = _fetch_bytes(url + ".CHECKSUM")
            entry, rows = _parse_archive(
                symbol=symbol,
                month=month,
                payload=payload,
                checksum_payload=checksum_payload,
                url=url,
            )
            entries.append(entry)
            rows_by_symbol[symbol].extend(rows)
    if len(entries) != 125:
        raise RuntimeError(f"planned archive roster mismatch: {len(entries)}")
    for symbol, rows in rows_by_symbol.items():
        timestamps = [item[0] for item in rows]
        if not rows or any(b <= a for a, b in zip(timestamps, timestamps[1:])):
            raise RuntimeError(f"filtered source rows are not strictly increasing: {symbol}")
        if len(set(timestamps)) != len(timestamps):
            raise RuntimeError(f"filtered source rows contain duplicates: {symbol}")
    return entries, rows_by_symbol


def _source_manifest(
    entries: list[dict[str, object]],
    *,
    preflight_run_id: int,
    preflight_artifact_id: int,
    preflight_api_digest: str,
    preflight_content_digest: str,
) -> dict[str, object]:
    missing_by_symbol = {
        symbol: sum(
            int(item["missing_grid_rows"])
            for item in entries
            if item["symbol"] == symbol
        )
        for symbol in SYMBOLS
    }
    body: dict[str, object] = {
        "schema_version": "issue562_training_source_manifest_v1",
        "issue_number": 562,
        "calibration_head": CALIBRATION_HEAD,
        "source_feature_head": IMPLEMENTATION_HEAD,
        "protocol_digest": PROTOCOL_DIGEST,
        "source_authority": SOURCE_AUTHORITY,
        "source_preflight_authority": {
            "run_id": preflight_run_id,
            "artifact_id": preflight_artifact_id,
            "api_digest": preflight_api_digest,
            "content_digest": preflight_content_digest,
        },
        "symbols": list(SYMBOLS),
        "months": list(_months()),
        "planned_archives": 125,
        "archive_root": ROOT,
        "interval": "1h",
        "analysis_contract_active_floor": ANALYSIS_ACTIVE_FLOOR.isoformat().replace(
            "+00:00", "Z"
        ),
        "source_open_start_inclusive": SOURCE_OPEN_START.isoformat().replace(
            "+00:00", "Z"
        ),
        "source_open_end_exclusive": SOURCE_OPEN_END.isoformat().replace(
            "+00:00", "Z"
        ),
        "fit_start": "2021-01-01T01:00:00Z",
        "fit_cutoff": "2023-01-01T00:00:00Z",
        "missing_grid_rows_by_symbol": missing_by_symbol,
        "missing_rows_are_preserved_as_row_present_false": True,
        "evaluation_or_final_data_accessed": False,
        "return_or_pnl_computed": False,
        "entries": entries,
    }
    return {**body, "content_digest": _canonical_digest(body)}


def _series_from_rows(
    rows: list[tuple[int, float, float, float, float, float, float]],
) -> RawMarketSeries:
    open_ms = np.asarray([item[0] for item in rows], dtype=np.int64)
    timestamps = (open_ms + 3_600_000).astype("datetime64[ms]").astype(
        "datetime64[ns]"
    )
    count = len(rows)
    return RawMarketSeries(
        timestamps=timestamps,
        available_at=timestamps,
        open=np.asarray([item[1] for item in rows], dtype=np.float64),
        high=np.asarray([item[2] for item in rows], dtype=np.float64),
        low=np.asarray([item[3] for item in rows], dtype=np.float64),
        close=np.asarray([item[4] for item in rows], dtype=np.float64),
        volume=np.asarray([item[5] for item in rows], dtype=np.float64),
        funding_rate=np.zeros(count, dtype=np.float64),
        tradable=np.ones(count, dtype=np.bool_),
        funding_available=np.zeros(count, dtype=np.bool_),
        funding_event_count=np.zeros(count, dtype=np.int32),
        taker_buy_quote_volume=np.asarray(
            [item[6] for item in rows], dtype=np.float64
        ),
    )


def _build_dataset(
    rows_by_symbol: dict[
        str, list[tuple[int, float, float, float, float, float, float]]
    ],
    source_manifest_digest: str,
):
    source = InMemoryMarketDataSource(
        {symbol: _series_from_rows(rows_by_symbol[symbol]) for symbol in SYMBOLS}
    )
    config = MarketBuildConfig(
        base_timeframe="1h",
        features=(
            FeatureSpec(
                name="1h__signed_taker_quote_flow_24bar",
                kind=FeatureKind.SIGNED_TAKER_QUOTE_FLOW,
                lookback=24,
            ),
        ),
    )
    contracts = tuple(
        InstrumentContract(
            symbol=symbol,
            listed_at=ANALYSIS_ACTIVE_FLOOR,
            volume_unit=VolumeUnit.QUOTE_NOTIONAL,
        )
        for symbol in SYMBOLS
    )
    dataset = MarketDatasetBuilder(config).build(
        source,
        contracts,
        identity_provenance={
            "issue_number": 562,
            "protocol_digest": PROTOCOL_DIGEST,
            "calibration_head": CALIBRATION_HEAD,
            "implementation_head": IMPLEMENTATION_HEAD,
            "source_manifest_digest": source_manifest_digest,
        },
    )
    expected_first = np.datetime64("2020-12-31T02:00:00", "ns")
    expected_last = np.datetime64("2023-01-01T00:00:00", "ns")
    if dataset.timestamps[0] != expected_first or dataset.timestamps[-1] != expected_last:
        raise RuntimeError("resolved dataset clock differs from frozen source window")
    if dataset.symbols != SYMBOLS:
        raise RuntimeError("resolved dataset symbol roster differs from frozen roster")
    return dataset


def _write_json(path: Path, payload: dict[str, object]) -> bytes:
    data = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    path.write_bytes(data)
    return data


def execute(
    output_dir: Path,
    *,
    preflight_run_id: int,
    preflight_artifact_id: int,
    preflight_api_digest: str,
    preflight_content_digest: str,
) -> None:
    protocol = canonical_signed_taker_flow_protocol()
    if protocol.digest != PROTOCOL_DIGEST:
        raise RuntimeError("canonical protocol digest changed before publisher execution")
    if tuple(protocol.symbols) != SYMBOLS:
        raise RuntimeError("canonical protocol symbol roster changed before execution")

    entries, rows_by_symbol = _download_source()
    manifest = _source_manifest(
        entries,
        preflight_run_id=preflight_run_id,
        preflight_artifact_id=preflight_artifact_id,
        preflight_api_digest=preflight_api_digest,
        preflight_content_digest=preflight_content_digest,
    )
    manifest_digest = str(manifest["content_digest"])
    dataset = _build_dataset(rows_by_symbol, manifest_digest)

    result = calibrate_signed_taker_flow(
        dataset,
        protocol,
        calibration_head=CALIBRATION_HEAD,
        source_manifest_digest=manifest_digest,
    )
    if result.protocol_digest != PROTOCOL_DIGEST:
        raise RuntimeError("result protocol binding changed")
    if result.implementation_head != IMPLEMENTATION_HEAD:
        raise RuntimeError("result implementation binding changed")
    if result.calibration_head != CALIBRATION_HEAD:
        raise RuntimeError("result calibration-head binding changed")
    if result.source_manifest_digest != manifest_digest:
        raise RuntimeError("result source-manifest binding changed")
    if not result.training_relation_executed:
        raise RuntimeError("result did not record training-only execution")
    if any(
        (
            result.evaluation_pnl_inspected,
            result.evaluation_execution_authorized,
            result.final_test_authorized,
            result.shared_cash_profitability_established,
            result.production_eligible,
            result.live_trading_authorized,
        )
    ):
        raise RuntimeError("publisher crossed a forbidden evaluation/production boundary")

    output_dir.mkdir(parents=True, exist_ok=False)
    source_bytes = _write_json(output_dir / "source-manifest.json", manifest)
    result_payload = result.to_artifact_payload()
    result_bytes = _write_json(output_dir / "result.json", result_payload)
    metadata_body: dict[str, object] = {
        "schema_version": "issue562_signed_flow_publisher_metadata_v1",
        "issue_number": 562,
        "calibration_head": CALIBRATION_HEAD,
        "implementation_head": IMPLEMENTATION_HEAD,
        "protocol_digest": PROTOCOL_DIGEST,
        "dataset_id": dataset.dataset_id,
        "source_manifest_digest": manifest_digest,
        "source_manifest_json_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "result_content_digest": result.digest,
        "result_json_sha256": hashlib.sha256(result_bytes).hexdigest(),
        "training_relation_executed": True,
        "interpretation_deferred_until_fresh_reconstruction": True,
        "evaluation_pnl_inspected": False,
        "final_test_authorized": False,
        "production_eligible": False,
        "live_trading_authorized": False,
    }
    metadata = {**metadata_body, "content_digest": _canonical_digest(metadata_body)}
    _write_json(output_dir / "publisher-metadata.json", metadata)

    print("PUBLISHER_RESULT_CREATED=true")
    print("PUBLISHER_INTERPRETATION_DEFERRED=true")
    print(f"DATASET_ID={dataset.dataset_id}")
    print(f"SOURCE_MANIFEST_DIGEST={manifest_digest}")
    print(f"RESULT_CONTENT_DIGEST={result.digest}")
    print("EVALUATION_PNL_INSPECTED=false")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-run-id", type=int, required=True)
    parser.add_argument("--preflight-artifact-id", type=int, required=True)
    parser.add_argument("--preflight-api-digest", required=True)
    parser.add_argument("--preflight-content-digest", required=True)
    args = parser.parse_args()
    execute(
        args.output_dir,
        preflight_run_id=args.preflight_run_id,
        preflight_artifact_id=args.preflight_artifact_id,
        preflight_api_digest=args.preflight_api_digest,
        preflight_content_digest=args.preflight_content_digest,
    )


if __name__ == "__main__":
    main()
