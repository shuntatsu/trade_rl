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
from trade_rl.data.source import RawIndexPriceSeries, RawMarketSeries
from trade_rl.evaluation.experiments.bootstrap.perp_index_basis_calibration import (
    calibrate_perp_index_basis,
    load_perp_index_basis_calibration_result,
)
from trade_rl.evaluation.experiments.bootstrap.perp_index_basis_prereg import (
    canonical_perp_index_basis_protocol,
)
from trade_rl.integrations.binance import (
    BinanceMarket,
    BinancePublicTransport,
    BinanceTransportMode,
)

ISSUE_NUMBER = 575
CALIBRATION_HEAD = "6d7139eb65ea3db95c6dc93e2b3123a65fb092db"
CALIBRATION_FULL_VERIFY_RUN_ID = 34906099963
SOURCE_IMPLEMENTATION_HEAD = "4911579874111bb6620481be8a4dbd7a8e664918"
SOURCE_IMPLEMENTATION_CI_RUN_ID = 34871818525
PROTOCOL_DIGEST = "1dc531fb15bde8cb5d87531456bf9220843b1380f33831764ee56c7312c2a091"
PREFLIGHT_RUN_ID = 34868358617
PREFLIGHT_ARTIFACT_ID = 10358261087
PREFLIGHT_API_DIGEST = (
    "sha256:63eaa8135aa42330809a5dceb00905e141a18594d6a61edc699e7a6e02d2dae6"
)
PREFLIGHT_FRESH_ARTIFACT_ID = 10357523519
PREFLIGHT_FRESH_API_DIGEST = (
    "sha256:f1399532a3a90b5c75e70cd06da931942a7c8b0370f7eba06b62916d3d4f16bf"
)
PREFLIGHT_REPORT_SHA256 = (
    "9a8b4d88c48885eb2aba8becb1ae347c6a1c20774dfc9d47281fa042b065f7bf"
)
PREFLIGHT_CONTENT_DIGEST = (
    "ccb22002ac28a0f2a1275e4c31fb5cd0cde59b72d14017e15d1f88141dcf0e65"
)
SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
MONTHS = tuple(f"{year}-{month:02d}" for year in (2021, 2022) for month in range(1, 13))
PERP_ROOT = "https://data.binance.vision/data/futures/um/monthly/klines"
INDEX_ROOT = "https://data.binance.vision/data/futures/um/monthly/indexPriceKlines"
INTERVAL_MS = 3_600_000
ACTIVE_FLOOR = datetime(2021, 1, 1, 0, tzinfo=UTC)
SOURCE_OPEN_START = datetime(2021, 1, 1, 0, tzinfo=UTC)
SOURCE_OPEN_END = datetime(2023, 1, 1, 0, tzinfo=UTC)
USER_AGENT = "trade-rl-issue575-basis-independent-verifier/1"
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


def _canonical_digest(payload: dict[str, object]) -> str:
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _normalize_epoch_ms(value: object) -> int:
    numeric = int(str(value))
    while abs(numeric) >= 10_000_000_000_000:
        numeric //= 1_000
    return numeric


def _epoch_ms(value: datetime) -> int:
    return int(value.timestamp() * 1_000)


def _iso_ms(value: int) -> str:
    return (
        datetime.fromtimestamp(value / 1_000, tz=UTC).isoformat().replace("+00:00", "Z")
    )


def _month_grid(month: str) -> tuple[list[int], int]:
    year, month_number = map(int, month.split("-"))
    start = datetime(year, month_number, 1, tzinfo=UTC)
    if month_number == 12:
        stop = datetime(year + 1, 1, 1, tzinfo=UTC)
    else:
        stop = datetime(year, month_number + 1, 1, tzinfo=UTC)
    start_ms = _epoch_ms(start)
    stop_ms = _epoch_ms(stop)
    expected = calendar.monthrange(year, month_number)[1] * 24
    return list(range(start_ms, stop_ms, INTERVAL_MS)), expected


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


def _checksum(data: bytes, *, expected_name: str, url: str) -> tuple[str, str]:
    text = data.decode("utf-8").strip()
    parts = text.split()
    if len(parts) < 2:
        raise RuntimeError(f"fresh checksum is malformed: {url}")
    digest = parts[0].lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise RuntimeError(f"fresh checksum digest is malformed: {url}")
    if parts[-1].lstrip("*") != expected_name:
        raise RuntimeError(f"fresh checksum filename mismatch: {url}")
    return digest, text


def _load_json(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return raw


def _verify_self_digest(
    payload: dict[str, object], field: str = "content_digest"
) -> None:
    observed = payload.get(field)
    if not isinstance(observed, str) or len(observed) != 64:
        raise RuntimeError(f"{field} missing or malformed")
    body = dict(payload)
    del body[field]
    if _canonical_digest(body) != observed:
        raise RuntimeError(f"{field} mismatch")


def _load_preflight(
    path: Path,
) -> tuple[dict[str, object], dict[tuple[str, str], dict[str, object]]]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PREFLIGHT_REPORT_SHA256:
        raise RuntimeError("fresh verifier preflight SHA-256 mismatch")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise RuntimeError("fresh verifier preflight is not an object")
    _verify_self_digest(payload)
    if payload.get("content_digest") != PREFLIGHT_CONTENT_DIGEST:
        raise RuntimeError("fresh verifier preflight content authority mismatch")
    if payload.get("status") != "PASS_FULL_INDEX_PREFLIGHT":
        raise RuntimeError("fresh verifier preflight status is not PASS")
    if payload.get("prereg_protocol_digest") != PROTOCOL_DIGEST:
        raise RuntimeError("fresh verifier protocol authority mismatch")
    entries = payload.get("entries")
    if not isinstance(entries, list) or len(entries) != 120:
        raise RuntimeError("fresh verifier preflight entry count mismatch")
    result: dict[tuple[str, str], dict[str, object]] = {}
    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            raise RuntimeError("fresh verifier preflight entry malformed")
        symbol = raw_entry.get("symbol")
        month = raw_entry.get("month")
        if not isinstance(symbol, str) or not isinstance(month, str):
            raise RuntimeError("fresh verifier preflight key malformed")
        key = (symbol, month)
        expected_url = f"{INDEX_ROOT}/{symbol}/1h/{symbol}-1h-{month}.zip"
        if key in result or raw_entry.get("url") != expected_url:
            raise RuntimeError("fresh verifier preflight roster mismatch")
        raw_sha = raw_entry.get("raw_sha256")
        checksum_sha = raw_entry.get("checksum_expected_sha256")
        if (
            raw_sha != checksum_sha
            or not isinstance(raw_sha, str)
            or len(raw_sha) != 64
        ):
            raise RuntimeError("fresh verifier preflight SHA binding malformed")
        if (
            raw_entry.get("checksum_verified") is not True
            or raw_entry.get("structurally_valid") is not True
        ):
            raise RuntimeError("fresh verifier preflight structural binding invalid")
        result[key] = dict(raw_entry)
    if set(result) != {(symbol, month) for symbol in SYMBOLS for month in MONTHS}:
        raise RuntimeError("fresh verifier preflight keys differ from frozen plan")
    return payload, result


def _fresh_perp_archive(
    *, symbol: str, month: str, url: str
) -> tuple[dict[str, object], list[tuple[int, float, float, float, float, float]]]:
    payload = _fetch(url)
    checksum_payload = _fetch(url + ".CHECKSUM")
    filename = url.rsplit("/", 1)[-1]
    digest = hashlib.sha256(payload).hexdigest()
    expected_digest, checksum_text = _checksum(
        checksum_payload, expected_name=filename, url=url + ".CHECKSUM"
    )
    if digest != expected_digest:
        raise RuntimeError(f"fresh perp archive checksum mismatch: {url}")
    expected_member = filename.removesuffix(".zip") + ".csv"
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        if len(members) != 1 or members[0].filename != expected_member:
            raise RuntimeError(f"fresh perp archive member mismatch: {url}")
        rows = [
            row
            for row in csv.reader(
                io.StringIO(archive.read(members[0]).decode("utf-8-sig"))
            )
            if row
        ]
    if not rows:
        raise RuntimeError(f"fresh perp archive empty: {url}")
    header_present = False
    try:
        int(rows[0][0])
    except (ValueError, IndexError):
        header_present = True
    if header_present and tuple(item.strip() for item in rows[0]) != EXPECTED_HEADER:
        raise RuntimeError(f"fresh perp header mismatch: {url}")
    data = rows[1:] if header_present else rows
    expected_grid, expected_rows = _month_grid(month)
    expected_set = set(expected_grid)
    times: list[int] = []
    selected: list[tuple[int, float, float, float, float, float]] = []
    for row in data:
        if len(row) != 12:
            raise RuntimeError(f"fresh perp field count mismatch: {url}")
        open_ms = _normalize_epoch_ms(row[0])
        close_ms = _normalize_epoch_ms(row[6])
        if open_ms not in expected_set or close_ms != open_ms + INTERVAL_MS - 1:
            raise RuntimeError(f"fresh perp clock contract mismatch: {url}")
        values = tuple(float(row[index]) for index in (1, 2, 3, 4, 7))
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"fresh perp non-finite field: {url}")
        open_price, high, low, close, quote = values
        if min(open_price, high, low, close) <= 0.0 or quote < 0.0:
            raise RuntimeError(f"fresh perp numeric bound mismatch: {url}")
        if low > min(open_price, close) or high < max(open_price, close) or low > high:
            raise RuntimeError(f"fresh perp OHLC invariant mismatch: {url}")
        times.append(open_ms)
        selected.append((open_ms, open_price, high, low, close, quote))
    if len(set(times)) != len(times) or any(b <= a for a, b in zip(times, times[1:])):
        raise RuntimeError(f"fresh perp timestamp order mismatch: {url}")
    observed = set(times)
    missing = sorted(expected_set - observed)
    spacings = sorted(set(b - a for a, b in zip(times, times[1:])))
    if any(value <= 0 or value % INTERVAL_MS for value in spacings):
        raise RuntimeError(f"fresh perp spacing mismatch: {url}")
    leading = 0
    for timestamp in expected_grid:
        if timestamp in observed:
            break
        leading += 1
    trailing = 0
    for timestamp in reversed(expected_grid):
        if timestamp in observed:
            break
        trailing += 1
    missing_bytes = "\n".join(str(value) for value in missing).encode("ascii")
    entry: dict[str, object] = {
        "symbol": symbol,
        "month": month,
        "url": url,
        "checksum_url": url + ".CHECKSUM",
        "raw_sha256": digest,
        "raw_size_bytes": len(payload),
        "checksum_sha256": expected_digest,
        "checksum_text": checksum_text,
        "checksum_verified": True,
        "csv_member": expected_member,
        "header_present": header_present,
        "header": list(EXPECTED_HEADER) if header_present else [],
        "field_count": 12,
        "row_count": len(times),
        "expected_grid_rows": expected_rows,
        "first_open_time": _iso_ms(times[0]),
        "last_open_time": _iso_ms(times[-1]),
        "spacing_ms": spacings,
        "missing_grid_rows": len(missing),
        "leading_missing_grid_rows": leading,
        "trailing_missing_grid_rows": trailing,
        "missing_grid_open_times_sha256": hashlib.sha256(missing_bytes).hexdigest(),
        "on_requested_1h_grid": True,
        "structurally_valid": True,
    }
    return entry, selected


def _fresh_perp(
    published_entries: object,
) -> tuple[
    list[dict[str, object]],
    dict[str, list[tuple[int, float, float, float, float, float]]],
]:
    if not isinstance(published_entries, list) or len(published_entries) != 120:
        raise RuntimeError("published perp manifest entry count mismatch")
    by_key: dict[tuple[str, str], dict[str, object]] = {}
    for raw in published_entries:
        if not isinstance(raw, dict):
            raise RuntimeError("published perp entry malformed")
        symbol = raw.get("symbol")
        month = raw.get("month")
        if not isinstance(symbol, str) or not isinstance(month, str):
            raise RuntimeError("published perp entry key malformed")
        by_key[(symbol, month)] = raw
    fresh_entries: list[dict[str, object]] = []
    rows = {symbol: [] for symbol in SYMBOLS}
    for symbol in SYMBOLS:
        for month in MONTHS:
            expected_url = f"{PERP_ROOT}/{symbol}/1h/{symbol}-1h-{month}.zip"
            published = by_key.get((symbol, month))
            if published is None or published.get("url") != expected_url:
                raise RuntimeError(f"published perp roster mismatch: {symbol}:{month}")
            entry, parsed = _fresh_perp_archive(
                symbol=symbol, month=month, url=expected_url
            )
            if entry != published:
                raise RuntimeError(f"fresh perp evidence differs: {symbol}:{month}")
            fresh_entries.append(entry)
            rows[symbol].extend(parsed)
    return fresh_entries, rows


def _index_expected(
    entries: dict[tuple[str, str], dict[str, object]],
) -> dict[str, str]:
    return {
        str(entries[(symbol, month)]["url"]): str(
            entries[(symbol, month)]["raw_sha256"]
        )
        for symbol in SYMBOLS
        for month in MONTHS
    }


def _fresh_index(
    entries: dict[tuple[str, str], dict[str, object]],
) -> dict[str, list[tuple[int, float]]]:
    transport = BinancePublicTransport(
        timeout_seconds=180.0,
        max_attempts=5,
        retry_backoff_seconds=1.0,
    )
    expected = _index_expected(entries)
    start_ms = _epoch_ms(SOURCE_OPEN_START)
    end_ms = _epoch_ms(SOURCE_OPEN_END)
    result: dict[str, list[tuple[int, float]]] = {}
    for symbol in SYMBOLS:
        raw_rows, sources = transport.load_index_price_klines(
            market=BinanceMarket.USDS_M,
            symbol=symbol,
            interval="1h",
            start_ms=start_ms,
            end_ms=end_ms,
            mode=BinanceTransportMode.VISION,
            expected_archive_sha256=expected,
        )
        expected_sources = tuple(
            str(entries[(symbol, month)]["url"]) for month in MONTHS
        )
        if sources != expected_sources:
            raise RuntimeError(f"fresh index source roster mismatch: {symbol}")
        parsed: list[tuple[int, float]] = []
        for row in raw_rows:
            open_ms = _normalize_epoch_ms(row[0])
            close = float(row[4])
            if not math.isfinite(close) or close <= 0.0:
                raise RuntimeError("fresh index close invalid")
            parsed.append((open_ms, close))
        if not parsed or any(b[0] <= a[0] for a, b in zip(parsed, parsed[1:])):
            raise RuntimeError(f"fresh index timestamp order mismatch: {symbol}")
        result[symbol] = parsed
    return result


def _missing_by_symbol(entries: list[dict[str, object]]) -> dict[str, int]:
    return {
        symbol: sum(
            int(item["missing_grid_rows"])
            for item in entries
            if item["symbol"] == symbol
        )
        for symbol in SYMBOLS
    }


def _index_missing(entries: list[object]) -> dict[str, int]:
    result = {symbol: 0 for symbol in SYMBOLS}
    for raw in entries:
        if not isinstance(raw, dict):
            raise RuntimeError("index entry malformed")
        symbol = raw.get("symbol")
        missing = raw.get("missing_grid_rows")
        if (
            symbol not in result
            or isinstance(missing, bool)
            or not isinstance(missing, int)
        ):
            raise RuntimeError("index missingness malformed")
        result[str(symbol)] += missing
    return result


def _rebuild_manifest(
    preflight: dict[str, object], perp_entries: list[dict[str, object]]
) -> dict[str, object]:
    index_entries = preflight["entries"]
    assert isinstance(index_entries, list)
    body: dict[str, object] = {
        "schema_version": "issue575_basis_training_source_manifest_v1",
        "issue_number": ISSUE_NUMBER,
        "calibration_head": CALIBRATION_HEAD,
        "calibration_full_verify_run_id": CALIBRATION_FULL_VERIFY_RUN_ID,
        "source_implementation_head": SOURCE_IMPLEMENTATION_HEAD,
        "source_implementation_ci_run_id": SOURCE_IMPLEMENTATION_CI_RUN_ID,
        "protocol_digest": PROTOCOL_DIGEST,
        "index_preflight_authority": {
            "run_id": PREFLIGHT_RUN_ID,
            "artifact_id": PREFLIGHT_ARTIFACT_ID,
            "api_digest": PREFLIGHT_API_DIGEST,
            "fresh_artifact_id": PREFLIGHT_FRESH_ARTIFACT_ID,
            "fresh_api_digest": PREFLIGHT_FRESH_API_DIGEST,
            "report_sha256": PREFLIGHT_REPORT_SHA256,
            "content_digest": PREFLIGHT_CONTENT_DIGEST,
        },
        "symbols": list(SYMBOLS),
        "months": list(MONTHS),
        "interval": "1h",
        "perp_archive_root": PERP_ROOT,
        "index_archive_root": INDEX_ROOT,
        "planned_perp_archives": 120,
        "planned_index_archives": 120,
        "planned_total_archives": 240,
        "source_open_start_inclusive": SOURCE_OPEN_START.isoformat().replace(
            "+00:00", "Z"
        ),
        "source_open_end_exclusive": SOURCE_OPEN_END.isoformat().replace("+00:00", "Z"),
        "fit_start": "2021-01-01T01:00:00Z",
        "fit_cutoff": "2023-01-01T00:00:00Z",
        "perp_missing_grid_rows_by_symbol": _missing_by_symbol(perp_entries),
        "index_missing_grid_rows_by_symbol": _index_missing(index_entries),
        "perp_entries": perp_entries,
        "index_preflight_entries": index_entries,
        "manifest_frozen_before_training_relation": True,
        "basis_or_return_computed_when_manifest_frozen": False,
        "evaluation_pnl_inspected": False,
        "final_test_accessed": False,
        "replacement_source_used": False,
    }
    return {**body, "content_digest": _canonical_digest(body)}


def _perp_series(
    rows: list[tuple[int, float, float, float, float, float]],
) -> RawMarketSeries:
    opens = np.asarray([item[0] for item in rows], dtype=np.int64)
    timestamps = (opens + INTERVAL_MS).astype("datetime64[ms]").astype("datetime64[ns]")
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
        funding_available=np.zeros(count, dtype=np.bool_),
        funding_event_count=np.zeros(count, dtype=np.int32),
        tradable=np.ones(count, dtype=np.bool_),
    )


def _index_series(rows: list[tuple[int, float]]) -> RawIndexPriceSeries:
    opens = np.asarray([item[0] for item in rows], dtype=np.int64)
    timestamps = (opens + INTERVAL_MS).astype("datetime64[ms]").astype("datetime64[ns]")
    return RawIndexPriceSeries(
        timestamps=timestamps,
        available_at=timestamps,
        close=np.asarray([item[1] for item in rows], dtype=np.float64),
    )


class IndependentFrozenBasisSource:
    def __init__(
        self,
        perp: dict[str, list[tuple[int, float, float, float, float, float]]],
        index: dict[str, list[tuple[int, float]]],
        manifest_digest: str,
    ) -> None:
        self._perp = {symbol: _perp_series(perp[symbol]) for symbol in SYMBOLS}
        self._index = {symbol: _index_series(index[symbol]) for symbol in SYMBOLS}
        self._digest = manifest_digest

    def load(self, symbol: str) -> RawMarketSeries:
        return self._perp[symbol]

    def load_index_price(self, symbol: str, timeframe: str) -> RawIndexPriceSeries:
        if timeframe != "1h":
            raise ValueError("independent frozen source supports only 1h")
        return self._index[symbol]

    @property
    def index_price_provenance(self) -> dict[str, object]:
        return {
            "schema_version": "issue575_basis_index_source_provenance_v1",
            "source_family": "binance_vision_usdm_monthly_indexPriceKlines_1h",
            "manifest_digest": self._digest,
            "index_preflight_content_digest": PREFLIGHT_CONTENT_DIGEST,
            "source_implementation_head": SOURCE_IMPLEMENTATION_HEAD,
        }


def _rebuild_dataset(
    *,
    perp: dict[str, list[tuple[int, float, float, float, float, float]]],
    index: dict[str, list[tuple[int, float]]],
    manifest_digest: str,
) -> MarketDataset:
    protocol = canonical_perp_index_basis_protocol()
    source = IndependentFrozenBasisSource(perp, index, manifest_digest)
    config = MarketBuildConfig(
        base_timeframe="1h",
        features=(
            FeatureSpec(
                name=protocol.feature_name,
                kind=FeatureKind.PERP_INDEX_LOG_BASIS_BPS,
                lookback=1,
            ),
        ),
    )
    contracts = tuple(
        InstrumentContract(
            symbol=symbol,
            listed_at=ACTIVE_FLOOR,
            volume_unit=VolumeUnit.QUOTE_NOTIONAL,
        )
        for symbol in SYMBOLS
    )
    return MarketDatasetBuilder(config).build(
        source,
        contracts,
        identity_provenance={
            "issue_number": ISSUE_NUMBER,
            "protocol_digest": PROTOCOL_DIGEST,
            "calibration_head": CALIBRATION_HEAD,
            "source_implementation_head": SOURCE_IMPLEMENTATION_HEAD,
            "source_manifest_digest": manifest_digest,
            "index_preflight_content_digest": PREFLIGHT_CONTENT_DIGEST,
        },
    )


def verify(*, published_dir: Path, verified_dir: Path, preflight_json: Path) -> None:
    expected_files = {"publisher-metadata.json", "result.json", "source-manifest.json"}
    if {item.name for item in published_dir.iterdir()} != expected_files:
        raise RuntimeError("publisher artifact members differ from frozen contract")
    result_path = published_dir / "result.json"
    manifest_path = published_dir / "source-manifest.json"
    metadata_path = published_dir / "publisher-metadata.json"
    published_manifest = _load_json(manifest_path)
    published_metadata = _load_json(metadata_path)
    _verify_self_digest(published_manifest)
    _verify_self_digest(published_metadata)
    if manifest_path.read_bytes() != canonical_json_bytes(published_manifest):
        raise RuntimeError("published source manifest is not canonical JSON")
    if metadata_path.read_bytes() != canonical_json_bytes(published_metadata):
        raise RuntimeError("published metadata is not canonical JSON")
    published_result = load_perp_index_basis_calibration_result(result_path)
    published_result_payload = _load_json(result_path)
    if published_result.calibration_head != CALIBRATION_HEAD:
        raise RuntimeError("published result calibration head mismatch")
    if published_result.source_implementation_head != SOURCE_IMPLEMENTATION_HEAD:
        raise RuntimeError("published result source implementation mismatch")
    if published_result.protocol_digest != PROTOCOL_DIGEST:
        raise RuntimeError("published result protocol mismatch")
    if (
        published_metadata.get("interpretation_deferred_until_fresh_reconstruction")
        is not True
    ):
        raise RuntimeError("publisher did not defer interpretation")
    if published_metadata.get("evaluation_pnl_inspected") is not False:
        raise RuntimeError("publisher crossed evaluation boundary")

    preflight, preflight_entries = _load_preflight(preflight_json)
    if published_manifest.get("index_preflight_entries") != preflight.get("entries"):
        raise RuntimeError("published index evidence differs from frozen preflight")
    fresh_perp_entries, fresh_perp_rows = _fresh_perp(
        published_manifest.get("perp_entries")
    )
    fresh_index_rows = _fresh_index(preflight_entries)
    fresh_manifest = _rebuild_manifest(preflight, fresh_perp_entries)
    if fresh_manifest != published_manifest:
        raise RuntimeError("fresh combined source manifest differs from publisher")
    fresh_manifest_bytes = canonical_json_bytes(fresh_manifest)
    if fresh_manifest_bytes != manifest_path.read_bytes():
        raise RuntimeError("fresh combined source manifest bytes differ from publisher")
    manifest_digest = str(fresh_manifest["content_digest"])
    if manifest_digest != published_result.source_manifest_digest:
        raise RuntimeError("published result does not bind fresh source manifest")

    dataset = _rebuild_dataset(
        perp=fresh_perp_rows,
        index=fresh_index_rows,
        manifest_digest=manifest_digest,
    )
    if dataset.dataset_id != published_result.dataset_id:
        raise RuntimeError("fresh Dataset ID differs from published result")
    protocol = canonical_perp_index_basis_protocol()
    reconstructed = calibrate_perp_index_basis(
        dataset,
        protocol,
        calibration_head=CALIBRATION_HEAD,
        source_manifest_digest=manifest_digest,
    )
    reconstructed_payload = reconstructed.to_artifact_payload()
    reconstructed_bytes = canonical_json_bytes(reconstructed_payload)
    if reconstructed_payload != published_result_payload:
        raise RuntimeError("fresh result payload differs from publisher")
    if reconstructed_bytes != result_path.read_bytes():
        raise RuntimeError("fresh result bytes differ from publisher")
    if hashlib.sha256(reconstructed_bytes).hexdigest() != published_metadata.get(
        "result_json_sha256"
    ):
        raise RuntimeError("publisher result JSON SHA-256 mismatch")
    if hashlib.sha256(fresh_manifest_bytes).hexdigest() != published_metadata.get(
        "source_manifest_json_sha256"
    ):
        raise RuntimeError("publisher source manifest JSON SHA-256 mismatch")
    if reconstructed.digest != published_metadata.get("result_content_digest"):
        raise RuntimeError("publisher result content digest mismatch")

    verified_dir.mkdir(parents=True, exist_ok=False)
    (verified_dir / "verified-result.json").write_bytes(reconstructed_bytes)
    verification_body: dict[str, object] = {
        "schema_version": "issue575_basis_fresh_verification_v1",
        "issue_number": ISSUE_NUMBER,
        "calibration_head": CALIBRATION_HEAD,
        "calibration_full_verify_run_id": CALIBRATION_FULL_VERIFY_RUN_ID,
        "source_implementation_head": SOURCE_IMPLEMENTATION_HEAD,
        "source_implementation_ci_run_id": SOURCE_IMPLEMENTATION_CI_RUN_ID,
        "protocol_digest": PROTOCOL_DIGEST,
        "dataset_id": dataset.dataset_id,
        "source_manifest_digest": manifest_digest,
        "result_content_digest": reconstructed.digest,
        "result_json_sha256": hashlib.sha256(reconstructed_bytes).hexdigest(),
        "byte_identical": True,
        "fresh_source_manifest_identical": True,
        "all_240_archives_revalidated": True,
        "result_interpretation_authorized": True,
        "evaluation_pnl_inspected": False,
        "evaluation_execution_authorized": False,
        "final_test_authorized": False,
        "shared_cash_profitability_established": False,
        "production_eligible": False,
        "live_trading_authorized": False,
    }
    verification = {
        **verification_body,
        "content_digest": _canonical_digest(verification_body),
    }
    (verified_dir / "verification.json").write_bytes(canonical_json_bytes(verification))

    print("INDEPENDENT_RECONSTRUCTION_VERIFIED=true")
    print("RESULT_INTERPRETATION_NOW_AUTHORIZED=true")
    print("ALL_240_ARCHIVES_REVALIDATED=true")
    print("EVALUATION_PNL_INSPECTED=false")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--published-dir", type=Path, required=True)
    parser.add_argument("--verified-dir", type=Path, required=True)
    parser.add_argument("--preflight-json", type=Path, required=True)
    args = parser.parse_args()
    verify(
        published_dir=args.published_dir,
        verified_dir=args.verified_dir,
        preflight_json=args.preflight_json,
    )


if __name__ == "__main__":
    main()
