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

from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.data.build import MarketDatasetBuilder
from trade_rl.data.contracts import (
    FeatureKind,
    FeatureSpec,
    InstrumentContract,
    MarketBuildConfig,
    VolumeUnit,
)
from trade_rl.data.market import MarketDataset
from trade_rl.data.source import InMemoryMarketDataSource, RawMarketSeries
from trade_rl.evaluation.experiments.bootstrap.signed_taker_flow_calibration import (
    calibrate_signed_taker_flow,
    load_signed_taker_flow_calibration_result,
)
from trade_rl.evaluation.experiments.bootstrap.signed_taker_flow_prereg import (
    canonical_signed_taker_flow_protocol,
)

CALIBRATION_HEAD = "dd3ecb617ecd2476e20e9215eeb422d2cdc95750"
IMPLEMENTATION_HEAD = "adf835cf45936dbc40bf8d4f74879825ae211ef9"
PROTOCOL_DIGEST = "5fa9be305c948c4724e106ebd4f4090d2fc977da05d0884b7d4aa9a03df58407"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
ROOT = "https://data.binance.vision/data/futures/um/monthly/klines"
SOURCE_OPEN_START = datetime(2020, 12, 31, 1, tzinfo=UTC)
SOURCE_OPEN_END = datetime(2023, 1, 1, 0, tzinfo=UTC)
ANALYSIS_ACTIVE_FLOOR = datetime(2020, 12, 1, 0, tzinfo=UTC)
USER_AGENT = "trade-rl-issue562-signed-flow-independent-verifier/1"
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


def _months() -> tuple[str, ...]:
    values = ["2020-12"]
    for year in (2021, 2022):
        for month in range(1, 13):
            values.append(f"{year}-{month:02d}")
    return tuple(values)


def _canonical_digest(payload: dict[str, object]) -> str:
    data = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _missing_grid_rows(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError("missing_grid_rows must be a non-negative integer")
    return value


def _load_json(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return raw


def _verify_self_digest(payload: dict[str, object], *, field: str) -> None:
    observed = payload.get(field)
    if not isinstance(observed, str) or len(observed) != 64:
        raise RuntimeError(f"{field} is missing or malformed")
    body = dict(payload)
    del body[field]
    if _canonical_digest(body) != observed:
        raise RuntimeError(f"{field} mismatch")


def _fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error = "network_error"
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                data = response.read()
            if not data:
                raise RuntimeError(f"empty response: {url}")
            return data
        except urllib.error.HTTPError as error:
            last_error = f"http_{error.code}"
        except (urllib.error.URLError, TimeoutError) as error:
            last_error = type(error).__name__
        if attempt < 4:
            time.sleep(2 + 2 * attempt)
    raise RuntimeError(f"download failed after retries ({last_error}): {url}")


def _parse_checksum(data: bytes, *, url: str) -> tuple[str, str]:
    try:
        text = data.decode("utf-8").strip()
    except UnicodeDecodeError as error:
        raise RuntimeError(f"checksum is not UTF-8: {url}") from error
    pieces = text.split()
    if not pieces:
        raise RuntimeError(f"checksum is empty: {url}")
    digest = pieces[0].lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise RuntimeError(f"checksum digest is malformed: {url}")
    return digest, text


def _month_grid(month: str) -> tuple[int, int, int]:
    year, month_number = [int(item) for item in month.split("-")]
    start = datetime(year, month_number, 1, tzinfo=UTC)
    if month_number == 12:
        stop = datetime(year + 1, 1, 1, tzinfo=UTC)
    else:
        stop = datetime(year, month_number + 1, 1, tzinfo=UTC)
    return (
        int(start.timestamp() * 1000),
        int(stop.timestamp() * 1000),
        calendar.monthrange(year, month_number)[1] * 24,
    )


def _iso(ms: int) -> str:
    return (
        datetime.fromtimestamp(ms / 1000.0, tz=UTC).isoformat().replace("+00:00", "Z")
    )


def _fresh_archive(
    *, symbol: str, month: str, url: str
) -> tuple[
    dict[str, object], list[tuple[int, float, float, float, float, float, float]]
]:
    payload = _fetch(url)
    checksum_payload = _fetch(url + ".CHECKSUM")
    raw_sha = hashlib.sha256(payload).hexdigest()
    expected_sha, checksum_text = _parse_checksum(
        checksum_payload, url=url + ".CHECKSUM"
    )
    if raw_sha != expected_sha:
        raise RuntimeError(f"fresh checksum mismatch: {url}")

    archive = zipfile.ZipFile(io.BytesIO(payload))
    members = [member for member in archive.infolist() if not member.is_dir()]
    if len(members) != 1 or not members[0].filename.lower().endswith(".csv"):
        raise RuntimeError(f"fresh archive member contract failed: {url}")
    member_name = members[0].filename
    with archive.open(members[0], "r") as handle:
        text = io.TextIOWrapper(handle, encoding="utf-8-sig", newline="")
        rows = list(csv.reader(text))
    archive.close()
    if not rows:
        raise RuntimeError(f"fresh CSV is empty: {url}")

    header_present = False
    try:
        int(rows[0][0])
    except (ValueError, IndexError):
        header_present = True
    data_rows = rows[1:] if header_present else rows
    if not data_rows:
        raise RuntimeError(f"fresh CSV has no data: {url}")

    month_start, month_stop, expected_grid_rows = _month_grid(month)
    all_open_times: list[int] = []
    selected: list[tuple[int, float, float, float, float, float, float]] = []
    source_start_ms = int(SOURCE_OPEN_START.timestamp() * 1000)
    source_stop_ms = int(SOURCE_OPEN_END.timestamp() * 1000)
    for row in data_rows:
        if len(row) != 12:
            raise RuntimeError(f"fresh field count mismatch: {url}: {len(row)}")
        open_ms = int(row[0])
        if not month_start <= open_ms < month_stop:
            raise RuntimeError(f"fresh row outside month: {url}")
        if (open_ms - month_start) % 3_600_000:
            raise RuntimeError(f"fresh row off 1h grid: {url}")
        values = tuple(float(row[index]) for index in (1, 2, 3, 4, 7, 10))
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"fresh non-finite field: {url}")
        open_price, high, low, close, quote, taker = values
        if min(open_price, high, low, close) <= 0.0:
            raise RuntimeError(f"fresh non-positive OHLC: {url}")
        if quote < 0.0 or taker < 0.0 or taker > quote:
            raise RuntimeError(f"fresh quote/taker bound failed: {url}")
        if low > min(open_price, close) or high < max(open_price, close) or low > high:
            raise RuntimeError(f"fresh OHLC invariant failed: {url}")
        all_open_times.append(open_ms)
        if source_start_ms <= open_ms < source_stop_ms:
            selected.append((open_ms, open_price, high, low, close, quote, taker))
    if len(set(all_open_times)) != len(all_open_times):
        raise RuntimeError(f"fresh duplicate timestamps: {url}")
    if any(right <= left for left, right in zip(all_open_times, all_open_times[1:])):
        raise RuntimeError(f"fresh timestamps not strictly increasing: {url}")
    if len(all_open_times) > expected_grid_rows:
        raise RuntimeError(f"fresh archive exceeds monthly grid: {url}")

    entry: dict[str, object] = {
        "symbol": symbol,
        "month": month,
        "url": url,
        "checksum_url": url + ".CHECKSUM",
        "raw_sha256": raw_sha,
        "raw_size_bytes": len(payload),
        "checksum_sha256": expected_sha,
        "checksum_text": checksum_text,
        "checksum_verified": True,
        "csv_member": member_name,
        "header_present": header_present,
        "expected_grid_rows": expected_grid_rows,
        "row_count": len(all_open_times),
        "missing_grid_rows": expected_grid_rows - len(all_open_times),
        "first_open_time": _iso(all_open_times[0]),
        "last_open_time": _iso(all_open_times[-1]),
        "field_count": 12,
        "timestamps_on_monthly_1h_grid": True,
        "structural_bounds_valid": True,
    }
    return entry, selected


def _fresh_source(
    published_manifest: dict[str, object],
) -> tuple[
    list[dict[str, object]],
    dict[str, list[tuple[int, float, float, float, float, float, float]]],
]:
    raw_entries = published_manifest.get("entries")
    if not isinstance(raw_entries, list) or len(raw_entries) != 125:
        raise RuntimeError("published source manifest does not contain 125 entries")
    by_key: dict[tuple[str, str], dict[str, object]] = {}
    for item in raw_entries:
        if not isinstance(item, dict):
            raise RuntimeError("published source manifest entry is not an object")
        symbol = item.get("symbol")
        month = item.get("month")
        if not isinstance(symbol, str) or not isinstance(month, str):
            raise RuntimeError("published source manifest entry key is malformed")
        key = (symbol, month)
        if key in by_key:
            raise RuntimeError("published source manifest contains duplicate entry")
        by_key[key] = item

    fresh_entries: list[dict[str, object]] = []
    rows_by_symbol: dict[
        str, list[tuple[int, float, float, float, float, float, float]]
    ] = {symbol: [] for symbol in SYMBOLS}
    for symbol in SYMBOLS:
        for month in _months():
            key = (symbol, month)
            published_entry = by_key.get(key)
            if published_entry is None:
                raise RuntimeError(
                    f"published source manifest missing {symbol}:{month}"
                )
            expected_url = f"{ROOT}/{symbol}/1h/{symbol}-1h-{month}.zip"
            if published_entry.get("url") != expected_url:
                raise RuntimeError(
                    f"published URL differs from frozen plan: {symbol}:{month}"
                )
            fresh_entry, rows = _fresh_archive(
                symbol=symbol, month=month, url=expected_url
            )
            if fresh_entry != published_entry:
                raise RuntimeError(f"fresh archive evidence differs: {symbol}:{month}")
            fresh_entries.append(fresh_entry)
            rows_by_symbol[symbol].extend(rows)
    return fresh_entries, rows_by_symbol


def _manifest_from_fresh_entries(
    entries: list[dict[str, object]],
    *,
    preflight_run_id: int,
    preflight_artifact_id: int,
    preflight_api_digest: str,
    preflight_content_digest: str,
) -> dict[str, object]:
    missing = {
        symbol: sum(
            _missing_grid_rows(item["missing_grid_rows"])
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
        "source_open_end_exclusive": SOURCE_OPEN_END.isoformat().replace("+00:00", "Z"),
        "fit_start": "2021-01-01T01:00:00Z",
        "fit_cutoff": "2023-01-01T00:00:00Z",
        "missing_grid_rows_by_symbol": missing,
        "missing_rows_are_preserved_as_row_present_false": True,
        "evaluation_or_final_data_accessed": False,
        "return_or_pnl_computed": False,
        "entries": entries,
    }
    return {**body, "content_digest": _canonical_digest(body)}


def _raw_series(
    rows: list[tuple[int, float, float, float, float, float, float]],
) -> RawMarketSeries:
    open_times = np.asarray([item[0] for item in rows], dtype=np.int64)
    timestamps = (
        (open_times + 3_600_000).astype("datetime64[ms]").astype("datetime64[ns]")
    )
    size = len(rows)
    return RawMarketSeries(
        timestamps=timestamps,
        available_at=timestamps,
        open=np.asarray([item[1] for item in rows], dtype=np.float64),
        high=np.asarray([item[2] for item in rows], dtype=np.float64),
        low=np.asarray([item[3] for item in rows], dtype=np.float64),
        close=np.asarray([item[4] for item in rows], dtype=np.float64),
        volume=np.asarray([item[5] for item in rows], dtype=np.float64),
        funding_rate=np.zeros(size, dtype=np.float64),
        tradable=np.ones(size, dtype=np.bool_),
        funding_available=np.zeros(size, dtype=np.bool_),
        funding_event_count=np.zeros(size, dtype=np.int32),
        taker_buy_quote_volume=np.asarray([item[6] for item in rows], dtype=np.float64),
    )


def _rebuild_dataset(
    rows_by_symbol: dict[
        str, list[tuple[int, float, float, float, float, float, float]]
    ],
    manifest_digest: str,
) -> MarketDataset:
    source_values: dict[str, RawMarketSeries] = {}
    for symbol in SYMBOLS:
        rows = rows_by_symbol[symbol]
        if not rows:
            raise RuntimeError(f"fresh reconstruction has no selected rows: {symbol}")
        source_values[symbol] = _raw_series(rows)
    source = InMemoryMarketDataSource(source_values)
    build_config = MarketBuildConfig(
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
    return MarketDatasetBuilder(build_config).build(
        source,
        contracts,
        identity_provenance={
            "issue_number": 562,
            "protocol_digest": PROTOCOL_DIGEST,
            "calibration_head": CALIBRATION_HEAD,
            "implementation_head": IMPLEMENTATION_HEAD,
            "source_manifest_digest": manifest_digest,
        },
    )


def _expected_json_bytes(payload: dict[str, object]) -> bytes:
    return canonical_json_bytes(payload)


def verify(
    published_dir: Path,
    verified_dir: Path,
    *,
    preflight_run_id: int,
    preflight_artifact_id: int,
    preflight_api_digest: str,
    preflight_content_digest: str,
) -> None:
    result_path = published_dir / "result.json"
    manifest_path = published_dir / "source-manifest.json"
    metadata_path = published_dir / "publisher-metadata.json"
    if {item.name for item in published_dir.iterdir()} != {
        "publisher-metadata.json",
        "result.json",
        "source-manifest.json",
    }:
        raise RuntimeError("published artifact members differ from frozen contract")

    published_result_payload = _load_json(result_path)
    published_manifest = _load_json(manifest_path)
    published_metadata = _load_json(metadata_path)
    _verify_self_digest(published_manifest, field="content_digest")
    _verify_self_digest(published_metadata, field="content_digest")
    published_result = load_signed_taker_flow_calibration_result(result_path)

    if published_result.calibration_head != CALIBRATION_HEAD:
        raise RuntimeError("published result calibration head is wrong")
    if published_result.implementation_head != IMPLEMENTATION_HEAD:
        raise RuntimeError("published result implementation head is wrong")
    if published_result.protocol_digest != PROTOCOL_DIGEST:
        raise RuntimeError("published result protocol digest is wrong")
    if (
        published_metadata.get("interpretation_deferred_until_fresh_reconstruction")
        is not True
    ):
        raise RuntimeError("publisher metadata did not defer interpretation")
    if published_metadata.get("evaluation_pnl_inspected") is not False:
        raise RuntimeError("publisher metadata crossed evaluation boundary")

    fresh_entries, fresh_rows = _fresh_source(published_manifest)
    fresh_manifest = _manifest_from_fresh_entries(
        fresh_entries,
        preflight_run_id=preflight_run_id,
        preflight_artifact_id=preflight_artifact_id,
        preflight_api_digest=preflight_api_digest,
        preflight_content_digest=preflight_content_digest,
    )
    if fresh_manifest != published_manifest:
        raise RuntimeError("fresh source manifest differs from publisher manifest")
    manifest_digest = str(fresh_manifest["content_digest"])
    if manifest_digest != published_result.source_manifest_digest:
        raise RuntimeError("published result does not bind fresh source manifest")

    dataset = _rebuild_dataset(fresh_rows, manifest_digest)
    if dataset.dataset_id != published_result.dataset_id:
        raise RuntimeError("fresh Dataset ID differs from published result")

    protocol = canonical_signed_taker_flow_protocol()
    if protocol.digest != PROTOCOL_DIGEST:
        raise RuntimeError("canonical protocol digest changed in verifier")
    reconstructed = calibrate_signed_taker_flow(
        dataset,
        protocol,
        calibration_head=CALIBRATION_HEAD,
        source_manifest_digest=manifest_digest,
    )
    reconstructed_payload = reconstructed.to_artifact_payload()
    if reconstructed_payload != published_result_payload:
        raise RuntimeError("fresh reconstructed result payload differs from publisher")
    reconstructed_bytes = _expected_json_bytes(reconstructed_payload)
    published_bytes = result_path.read_bytes()
    if reconstructed_bytes != published_bytes:
        raise RuntimeError("fresh reconstructed result bytes differ from publisher")
    if hashlib.sha256(published_bytes).hexdigest() != published_metadata.get(
        "result_json_sha256"
    ):
        raise RuntimeError("publisher result JSON SHA-256 metadata mismatch")
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != published_metadata.get(
        "source_manifest_json_sha256"
    ):
        raise RuntimeError("publisher source manifest JSON SHA-256 metadata mismatch")
    if reconstructed.digest != published_metadata.get("result_content_digest"):
        raise RuntimeError("publisher result content digest metadata mismatch")

    verified_dir.mkdir(parents=True, exist_ok=False)
    (verified_dir / "verified-result.json").write_bytes(reconstructed_bytes)
    verification_body: dict[str, object] = {
        "schema_version": "issue562_signed_flow_fresh_verification_v1",
        "issue_number": 562,
        "calibration_head": CALIBRATION_HEAD,
        "implementation_head": IMPLEMENTATION_HEAD,
        "protocol_digest": PROTOCOL_DIGEST,
        "dataset_id": dataset.dataset_id,
        "source_manifest_digest": manifest_digest,
        "result_content_digest": reconstructed.digest,
        "result_json_sha256": hashlib.sha256(reconstructed_bytes).hexdigest(),
        "byte_identical": True,
        "fresh_source_manifest_identical": True,
        "evaluation_pnl_inspected": False,
        "final_test_authorized": False,
        "production_eligible": False,
        "live_trading_authorized": False,
    }
    verification = {
        **verification_body,
        "content_digest": _canonical_digest(verification_body),
    }
    (verified_dir / "verification.json").write_bytes(_expected_json_bytes(verification))

    print("INDEPENDENT_RECONSTRUCTION_VERIFIED=true")
    print("RESULT_INTERPRETATION_NOW_AUTHORIZED=true")
    interpretation = {
        "status": reconstructed.status,
        "positive_slope_count": reconstructed.positive_slope_count,
        "failures": list(reconstructed.failures),
        "symbols": [
            {
                "symbol": item.symbol,
                "eligible_observations": item.eligible_observations,
                "beta": item.beta,
                "positive_slope": item.positive_slope,
            }
            for item in reconstructed.symbol_results
        ],
    }
    print(
        "VERIFIED_TRAINING_INTERPRETATION=" + json.dumps(interpretation, sort_keys=True)
    )
    print("EVALUATION_PNL_INSPECTED=false")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--published-dir", type=Path, required=True)
    parser.add_argument("--verified-dir", type=Path, required=True)
    parser.add_argument("--preflight-run-id", type=int, required=True)
    parser.add_argument("--preflight-artifact-id", type=int, required=True)
    parser.add_argument("--preflight-api-digest", required=True)
    parser.add_argument("--preflight-content-digest", required=True)
    args = parser.parse_args()
    verify(
        args.published_dir,
        args.verified_dir,
        preflight_run_id=args.preflight_run_id,
        preflight_artifact_id=args.preflight_artifact_id,
        preflight_api_digest=args.preflight_api_digest,
        preflight_content_digest=args.preflight_content_digest,
    )


if __name__ == "__main__":
    main()
