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
PREFLIGHT_SCHEMA = "issue571_full_index_preflight_v2"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
MONTHS = tuple(f"{year}-{month:02d}" for year in (2021, 2022) for month in range(1, 13))
PERP_ROOT = "https://data.binance.vision/data/futures/um/monthly/klines"
INDEX_ROOT = "https://data.binance.vision/data/futures/um/monthly/indexPriceKlines"
INTERVAL_MS = 3_600_000
ACTIVE_FLOOR = datetime(2021, 1, 1, 0, tzinfo=UTC)
SOURCE_OPEN_START = datetime(2021, 1, 1, 0, tzinfo=UTC)
SOURCE_OPEN_END = datetime(2023, 1, 1, 0, tzinfo=UTC)
USER_AGENT = "trade-rl-issue575-basis-publisher/1"
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


def _month_grid(month: str) -> tuple[list[int], int, int]:
    year, month_number = map(int, month.split("-"))
    start = datetime(year, month_number, 1, tzinfo=UTC)
    if month_number == 12:
        stop = datetime(year + 1, 1, 1, tzinfo=UTC)
    else:
        stop = datetime(year, month_number + 1, 1, tzinfo=UTC)
    start_ms = _epoch_ms(start)
    stop_ms = _epoch_ms(stop)
    expected = calendar.monthrange(year, month_number)[1] * 24
    return list(range(start_ms, stop_ms, INTERVAL_MS)), start_ms, expected


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


def _provider_checksum(
    payload: bytes, *, expected_name: str, url: str
) -> tuple[str, str]:
    try:
        text = payload.decode("utf-8").strip()
    except UnicodeDecodeError as error:
        raise RuntimeError(f"checksum is not UTF-8: {url}") from error
    fields = text.split()
    if len(fields) < 2:
        raise RuntimeError(f"checksum is malformed: {url}")
    digest = fields[0].lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise RuntimeError(f"checksum digest is malformed: {url}")
    observed_name = fields[-1].lstrip("*")
    if observed_name != expected_name:
        raise RuntimeError(f"checksum filename mismatch: {url}")
    return digest, text


def _load_preflight(
    path: Path,
) -> tuple[dict[str, object], dict[tuple[str, str], dict[str, object]]]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PREFLIGHT_REPORT_SHA256:
        raise RuntimeError("index preflight report SHA-256 authority mismatch")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise RuntimeError("index preflight report must be an object")
    observed_digest = payload.get("content_digest")
    body = dict(payload)
    body.pop("content_digest", None)
    if (
        observed_digest != PREFLIGHT_CONTENT_DIGEST
        or _canonical_digest(body) != observed_digest
    ):
        raise RuntimeError("index preflight content digest authority mismatch")
    required = {
        "schema_version": PREFLIGHT_SCHEMA,
        "issue_number": 571,
        "status": "PASS_FULL_INDEX_PREFLIGHT",
        "prereg_protocol_digest": PROTOCOL_DIGEST,
        "planned_archives": 120,
        "available_archives": 120,
        "checksum_verified_archives": 120,
        "valid_archives": 120,
        "source_family": "binance_vision_usdm_monthly_indexPriceKlines_1h",
        "source_root": INDEX_ROOT,
        "interval": "1h",
        "total_missing_grid_rows": 528,
        "basis_or_return_computed": False,
        "target_relation_computed": False,
        "economic_values_inspected": False,
        "evaluation_pnl_inspected": False,
        "replacement_source_used": False,
        "sparse_rows_remain_unavailable": True,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise RuntimeError(f"index preflight field mismatch: {key}")
    if (
        tuple(payload.get("symbols", ())) != SYMBOLS
        or tuple(payload.get("months", ())) != MONTHS
    ):
        raise RuntimeError("index preflight roster differs from frozen plan")
    entries = payload.get("entries")
    if not isinstance(entries, list) or len(entries) != 120:
        raise RuntimeError("index preflight must contain exactly 120 entries")
    expected_order = [(symbol, month) for symbol in SYMBOLS for month in MONTHS]
    result: dict[tuple[str, str], dict[str, object]] = {}
    observed_order: list[tuple[str, str]] = []
    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            raise RuntimeError("index preflight entry must be an object")
        symbol = raw_entry.get("symbol")
        month = raw_entry.get("month")
        if not isinstance(symbol, str) or not isinstance(month, str):
            raise RuntimeError("index preflight entry key is malformed")
        key = (symbol, month)
        if key in result:
            raise RuntimeError("index preflight contains duplicate entry")
        expected_url = f"{INDEX_ROOT}/{symbol}/1h/{symbol}-1h-{month}.zip"
        if raw_entry.get("url") != expected_url:
            raise RuntimeError(f"index preflight URL mismatch: {symbol}:{month}")
        if raw_entry.get("available") is not True:
            raise RuntimeError(
                f"index archive unavailable in preflight: {symbol}:{month}"
            )
        if raw_entry.get("checksum_verified") is not True:
            raise RuntimeError(
                f"index checksum not verified in preflight: {symbol}:{month}"
            )
        if raw_entry.get("structurally_valid") is not True:
            raise RuntimeError(f"index archive invalid in preflight: {symbol}:{month}")
        raw_sha = raw_entry.get("raw_sha256")
        checksum_sha = raw_entry.get("checksum_expected_sha256")
        if (
            not isinstance(raw_sha, str)
            or len(raw_sha) != 64
            or raw_sha != checksum_sha
        ):
            raise RuntimeError(f"index SHA authority malformed: {symbol}:{month}")
        result[key] = dict(raw_entry)
        observed_order.append(key)
    if observed_order != expected_order:
        raise RuntimeError("index preflight entry order differs from frozen plan")
    return payload, result


def _parse_perp_archive(
    *, symbol: str, month: str, payload: bytes, checksum_payload: bytes, url: str
) -> tuple[dict[str, object], list[tuple[int, float, float, float, float, float]]]:
    filename = url.rsplit("/", 1)[-1]
    raw_sha = hashlib.sha256(payload).hexdigest()
    checksum_sha, checksum_text = _provider_checksum(
        checksum_payload, expected_name=filename, url=url + ".CHECKSUM"
    )
    if raw_sha != checksum_sha:
        raise RuntimeError(f"perp archive checksum mismatch: {url}")
    expected_member = filename.removesuffix(".zip") + ".csv"
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        if len(members) != 1 or members[0].filename != expected_member:
            raise RuntimeError(f"perp archive member contract failed: {url}")
        text = archive.read(members[0]).decode("utf-8-sig")
    rows = [row for row in csv.reader(io.StringIO(text)) if row]
    if not rows:
        raise RuntimeError(f"empty perp CSV: {url}")
    header_present = False
    try:
        int(rows[0][0])
    except (ValueError, IndexError):
        header_present = True
    if header_present and tuple(item.strip() for item in rows[0]) != EXPECTED_HEADER:
        raise RuntimeError(f"perp header mismatch: {url}")
    data = rows[1:] if header_present else rows
    if not data:
        raise RuntimeError(f"perp CSV contains no rows: {url}")

    expected_grid, month_start, expected_rows = _month_grid(month)
    expected_set = set(expected_grid)
    open_times: list[int] = []
    parsed: list[tuple[int, float, float, float, float, float]] = []
    for row in data:
        if len(row) != 12:
            raise RuntimeError(f"perp field count mismatch: {url}")
        open_ms = _normalize_epoch_ms(row[0])
        close_ms = _normalize_epoch_ms(row[6])
        if open_ms not in expected_set:
            raise RuntimeError(f"perp row outside requested 1h month grid: {url}")
        if close_ms != open_ms + INTERVAL_MS - 1:
            raise RuntimeError(f"perp close-time contract mismatch: {url}")
        open_price = float(row[1])
        high = float(row[2])
        low = float(row[3])
        close = float(row[4])
        quote_volume = float(row[7])
        numbers = (open_price, high, low, close, quote_volume)
        if not all(math.isfinite(value) for value in numbers):
            raise RuntimeError(f"perp non-finite field: {url}")
        if min(open_price, high, low, close) <= 0.0 or quote_volume < 0.0:
            raise RuntimeError(f"perp numeric bounds failed: {url}")
        if low > min(open_price, close) or high < max(open_price, close) or low > high:
            raise RuntimeError(f"perp OHLC invariant failed: {url}")
        open_times.append(open_ms)
        if _epoch_ms(SOURCE_OPEN_START) <= open_ms < _epoch_ms(SOURCE_OPEN_END):
            parsed.append((open_ms, open_price, high, low, close, quote_volume))
    if any(right <= left for left, right in zip(open_times, open_times[1:])):
        raise RuntimeError(f"perp timestamps not strictly increasing: {url}")
    if len(set(open_times)) != len(open_times):
        raise RuntimeError(f"perp duplicate timestamps: {url}")
    observed = set(open_times)
    missing = sorted(expected_set - observed)
    spacings = sorted(
        set(later - earlier for earlier, later in zip(open_times, open_times[1:]))
    )
    if any(value <= 0 or value % INTERVAL_MS != 0 for value in spacings):
        raise RuntimeError(f"perp observed spacing off 1h grid: {url}")
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
        "raw_sha256": raw_sha,
        "raw_size_bytes": len(payload),
        "checksum_sha256": checksum_sha,
        "checksum_text": checksum_text,
        "checksum_verified": True,
        "csv_member": expected_member,
        "header_present": header_present,
        "header": list(EXPECTED_HEADER) if header_present else [],
        "field_count": 12,
        "row_count": len(open_times),
        "expected_grid_rows": expected_rows,
        "first_open_time": _iso_ms(open_times[0]),
        "last_open_time": _iso_ms(open_times[-1]),
        "spacing_ms": spacings,
        "missing_grid_rows": len(missing),
        "leading_missing_grid_rows": leading,
        "trailing_missing_grid_rows": trailing,
        "missing_grid_open_times_sha256": hashlib.sha256(missing_bytes).hexdigest(),
        "on_requested_1h_grid": True,
        "structurally_valid": True,
    }
    return entry, parsed


def _download_perp() -> tuple[
    list[dict[str, object]],
    dict[str, list[tuple[int, float, float, float, float, float]]],
]:
    entries: list[dict[str, object]] = []
    rows: dict[str, list[tuple[int, float, float, float, float, float]]] = {
        symbol: [] for symbol in SYMBOLS
    }
    for symbol in SYMBOLS:
        for month in MONTHS:
            url = f"{PERP_ROOT}/{symbol}/1h/{symbol}-1h-{month}.zip"
            entry, selected = _parse_perp_archive(
                symbol=symbol,
                month=month,
                payload=_fetch_bytes(url),
                checksum_payload=_fetch_bytes(url + ".CHECKSUM"),
                url=url,
            )
            entries.append(entry)
            rows[symbol].extend(selected)
    if len(entries) != 120:
        raise RuntimeError("perp archive roster must contain exactly 120 entries")
    return entries, rows


def _index_expected_sha(
    preflight_entries: dict[tuple[str, str], dict[str, object]],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for key in [(symbol, month) for symbol in SYMBOLS for month in MONTHS]:
        entry = preflight_entries[key]
        url = entry["url"]
        digest = entry["raw_sha256"]
        assert isinstance(url, str) and isinstance(digest, str)
        result[url] = digest
    return result


def _download_index(
    preflight_entries: dict[tuple[str, str], dict[str, object]],
) -> dict[str, list[tuple[int, float]]]:
    expected = _index_expected_sha(preflight_entries)
    transport = BinancePublicTransport(
        timeout_seconds=180.0,
        max_attempts=5,
        retry_backoff_seconds=1.0,
    )
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
            str(preflight_entries[(symbol, month)]["url"]) for month in MONTHS
        )
        if sources != expected_sources:
            raise RuntimeError(f"index runtime source roster mismatch: {symbol}")
        parsed: list[tuple[int, float]] = []
        for row in raw_rows:
            if len(row) != 12:
                raise RuntimeError("index runtime row field count changed")
            open_ms = _normalize_epoch_ms(row[0])
            close_ms = _normalize_epoch_ms(row[6])
            if close_ms != open_ms + INTERVAL_MS - 1:
                raise RuntimeError("index runtime close-time contract changed")
            close = float(row[4])
            if not math.isfinite(close) or close <= 0.0:
                raise RuntimeError("index runtime close is invalid")
            parsed.append((open_ms, close))
        if not parsed or any(b[0] <= a[0] for a, b in zip(parsed, parsed[1:])):
            raise RuntimeError(
                f"index runtime rows are not strictly increasing: {symbol}"
            )
        result[symbol] = parsed
    return result


def _missing_by_symbol(entries: list[dict[str, object]]) -> dict[str, int]:
    return {
        symbol: sum(
            int(entry["missing_grid_rows"])
            for entry in entries
            if entry["symbol"] == symbol
        )
        for symbol in SYMBOLS
    }


def _index_missing_by_symbol(entries: list[object]) -> dict[str, int]:
    result = {symbol: 0 for symbol in SYMBOLS}
    for raw in entries:
        if not isinstance(raw, dict):
            raise RuntimeError("index preflight entry is malformed")
        symbol = raw.get("symbol")
        missing = raw.get("missing_grid_rows")
        if (
            symbol not in result
            or isinstance(missing, bool)
            or not isinstance(missing, int)
        ):
            raise RuntimeError("index preflight missingness is malformed")
        result[str(symbol)] += missing
    return result


def _source_manifest(
    *, preflight: dict[str, object], perp_entries: list[dict[str, object]]
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
        "index_missing_grid_rows_by_symbol": _index_missing_by_symbol(index_entries),
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
    open_times = np.asarray([item[0] for item in rows], dtype=np.int64)
    timestamps = (
        (open_times + INTERVAL_MS).astype("datetime64[ms]").astype("datetime64[ns]")
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
        funding_available=np.zeros(count, dtype=np.bool_),
        funding_event_count=np.zeros(count, dtype=np.int32),
        tradable=np.ones(count, dtype=np.bool_),
    )


def _index_series(rows: list[tuple[int, float]]) -> RawIndexPriceSeries:
    open_times = np.asarray([item[0] for item in rows], dtype=np.int64)
    timestamps = (
        (open_times + INTERVAL_MS).astype("datetime64[ms]").astype("datetime64[ns]")
    )
    return RawIndexPriceSeries(
        timestamps=timestamps,
        available_at=timestamps,
        close=np.asarray([item[1] for item in rows], dtype=np.float64),
    )


class FrozenBasisSource:
    def __init__(
        self,
        perp_rows: dict[str, list[tuple[int, float, float, float, float, float]]],
        index_rows: dict[str, list[tuple[int, float]]],
        manifest_digest: str,
    ) -> None:
        self._perp = {symbol: _perp_series(perp_rows[symbol]) for symbol in SYMBOLS}
        self._index = {symbol: _index_series(index_rows[symbol]) for symbol in SYMBOLS}
        self._manifest_digest = manifest_digest

    def load(self, symbol: str) -> RawMarketSeries:
        return self._perp[symbol]

    def load_index_price(self, symbol: str, timeframe: str) -> RawIndexPriceSeries:
        if timeframe != "1h":
            raise ValueError(
                "frozen basis source supports only native 1h index history"
            )
        return self._index[symbol]

    @property
    def index_price_provenance(self) -> dict[str, object]:
        return {
            "schema_version": "issue575_basis_index_source_provenance_v1",
            "source_family": "binance_vision_usdm_monthly_indexPriceKlines_1h",
            "manifest_digest": self._manifest_digest,
            "index_preflight_content_digest": PREFLIGHT_CONTENT_DIGEST,
            "source_implementation_head": SOURCE_IMPLEMENTATION_HEAD,
        }


def _build_dataset(
    *,
    perp_rows: dict[str, list[tuple[int, float, float, float, float, float]]],
    index_rows: dict[str, list[tuple[int, float]]],
    manifest_digest: str,
) -> MarketDataset:
    source = FrozenBasisSource(perp_rows, index_rows, manifest_digest)
    protocol = canonical_perp_index_basis_protocol()
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
    dataset = MarketDatasetBuilder(config).build(
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
    if dataset.symbols != SYMBOLS:
        raise RuntimeError("resolved Dataset symbol roster differs from frozen roster")
    if dataset.timestamps[0] != np.datetime64("2021-01-01T01:00:00", "ns"):
        raise RuntimeError("resolved Dataset starts outside frozen completed-bar clock")
    if dataset.timestamps[-1] != np.datetime64("2023-01-01T00:00:00", "ns"):
        raise RuntimeError("resolved Dataset ends outside frozen completed-bar clock")
    return dataset


def execute(output_dir: Path, *, preflight_json: Path) -> None:
    protocol = canonical_perp_index_basis_protocol()
    if protocol.digest != PROTOCOL_DIGEST or protocol.symbols != SYMBOLS:
        raise RuntimeError("canonical basis protocol changed before execution")
    preflight, preflight_entries = _load_preflight(preflight_json)
    perp_entries, perp_rows = _download_perp()
    index_rows = _download_index(preflight_entries)
    manifest = _source_manifest(preflight=preflight, perp_entries=perp_entries)
    manifest_digest = str(manifest["content_digest"])
    dataset = _build_dataset(
        perp_rows=perp_rows,
        index_rows=index_rows,
        manifest_digest=manifest_digest,
    )
    result = calibrate_perp_index_basis(
        dataset,
        protocol,
        calibration_head=CALIBRATION_HEAD,
        source_manifest_digest=manifest_digest,
    )
    if result.calibration_head != CALIBRATION_HEAD:
        raise RuntimeError("result calibration authority changed")
    if result.source_implementation_head != SOURCE_IMPLEMENTATION_HEAD:
        raise RuntimeError("result source implementation authority changed")
    if result.source_manifest_digest != manifest_digest:
        raise RuntimeError("result source manifest binding changed")
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
        raise RuntimeError("publisher crossed forbidden research boundary")

    output_dir.mkdir(parents=True, exist_ok=False)
    manifest_bytes = canonical_json_bytes(manifest)
    result_payload = result.to_artifact_payload()
    result_bytes = canonical_json_bytes(result_payload)
    (output_dir / "source-manifest.json").write_bytes(manifest_bytes)
    (output_dir / "result.json").write_bytes(result_bytes)
    metadata_body: dict[str, object] = {
        "schema_version": "issue575_basis_publisher_metadata_v1",
        "issue_number": ISSUE_NUMBER,
        "calibration_head": CALIBRATION_HEAD,
        "calibration_full_verify_run_id": CALIBRATION_FULL_VERIFY_RUN_ID,
        "source_implementation_head": SOURCE_IMPLEMENTATION_HEAD,
        "source_implementation_ci_run_id": SOURCE_IMPLEMENTATION_CI_RUN_ID,
        "protocol_digest": PROTOCOL_DIGEST,
        "dataset_id": dataset.dataset_id,
        "source_manifest_digest": manifest_digest,
        "source_manifest_json_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "result_content_digest": result.digest,
        "result_json_sha256": hashlib.sha256(result_bytes).hexdigest(),
        "training_relation_executed": True,
        "interpretation_deferred_until_fresh_reconstruction": True,
        "evaluation_pnl_inspected": False,
        "evaluation_execution_authorized": False,
        "final_test_authorized": False,
        "shared_cash_profitability_established": False,
        "production_eligible": False,
        "live_trading_authorized": False,
    }
    metadata = {**metadata_body, "content_digest": _canonical_digest(metadata_body)}
    metadata_bytes = canonical_json_bytes(metadata)
    (output_dir / "publisher-metadata.json").write_bytes(metadata_bytes)

    print("PUBLISHER_RESULT_CREATED=true")
    print("PUBLISHER_INTERPRETATION_DEFERRED=true")
    print(f"DATASET_ID={dataset.dataset_id}")
    print(f"SOURCE_MANIFEST_DIGEST={manifest_digest}")
    print(f"RESULT_CONTENT_DIGEST={result.digest}")
    print("EVALUATION_PNL_INSPECTED=false")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-json", type=Path, required=True)
    args = parser.parse_args()
    execute(args.output_dir, preflight_json=args.preflight_json)


if __name__ == "__main__":
    main()
