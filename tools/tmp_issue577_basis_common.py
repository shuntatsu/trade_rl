from __future__ import annotations

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
from collections.abc import Callable, Mapping
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
    NormalizationMode,
    VolumeUnit,
)
from trade_rl.data.market import MarketDataset
from trade_rl.data.source import RawIndexPriceSeries, RawMarketSeries
from trade_rl.evaluation.experiments.bootstrap.perp_index_basis_calibration import (
    PerpIndexBasisCalibrationResult,
    calibrate_perp_index_basis,
)
from trade_rl.evaluation.experiments.bootstrap.perp_index_basis_prereg import (
    canonical_perp_index_basis_protocol,
)

ISSUE_NUMBER = 577
PROTOCOL_DIGEST = "1dc531fb15bde8cb5d87531456bf9220843b1380f33831764ee56c7312c2a091"
PREREG_HEAD = "795b2e9efc2e7d9d65a7131a160c7da16ba02043"
PREREG_SEAL_RUN_ID = 34867687158
PREREG_SEAL_ARTIFACT_ID = 10357004799
PREREG_SEAL_API_DIGEST = (
    "sha256:e4b156b66d273bebe012b664ed0f72ef551d46cb48ca0a8fddb04ba5ff61c96b"
)
PREREG_FRESH_ARTIFACT_ID = 10357836707
PREREG_FRESH_API_DIGEST = (
    "sha256:ff4ce128b2b2103c309d94f2731505bd2c5e0029df99c47d9783b6e566c548f4"
)
INDEX_PREFLIGHT_RUN_ID = 34868358617
INDEX_PREFLIGHT_ARTIFACT_ID = 10358261087
INDEX_PREFLIGHT_API_DIGEST = (
    "sha256:63eaa8135aa42330809a5dceb00905e141a18594d6a61edc699e7a6e02d2dae6"
)
INDEX_PREFLIGHT_REPORT_SHA256 = (
    "9a8b4d88c48885eb2aba8becb1ae347c6a1c20774dfc9d47281fa042b065f7bf"
)
INDEX_PREFLIGHT_CONTENT_DIGEST = (
    "ccb22002ac28a0f2a1275e4c31fb5cd0cde59b72d14017e15d1f88141dcf0e65"
)
SOURCE_IMPLEMENTATION_HEAD = "4911579874111bb6620481be8a4dbd7a8e664918"
SOURCE_FINAL_VERIFICATION_HEAD = "fffbd1121b4ba79085ea09c113b7ce75c33c6f83"
SOURCE_FINAL_VERIFICATION_RUN_ID = 34905791445
CALIBRATION_HEAD = "6d7139eb65ea3db95c6dc93e2b3123a65fb092db"
CALIBRATION_VERIFICATION_RUN_ID = 34906099963
SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
MONTHS = tuple(
    f"{year:04d}-{month:02d}" for year in (2021, 2022) for month in range(1, 13)
)
INTERVAL = "1h"
INTERVAL_MS = 3_600_000
CONTRACT_ROOT = "https://data.binance.vision/data/futures/um/monthly/klines"
INDEX_ROOT = "https://data.binance.vision/data/futures/um/monthly/indexPriceKlines"
USER_AGENT = "trade-rl-issue577-basis-training/1"
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

FetchBytes = Callable[[str], bytes]
ContractRow = tuple[int, float, float, float, float, float]
IndexRow = tuple[int, float]


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_hex(value: object, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise RuntimeError(f"{field} must be a 64-character lowercase SHA-256")
    if value.lower() != value or any(char not in "0123456789abcdef" for char in value):
        raise RuntimeError(f"{field} must be a 64-character lowercase SHA-256")
    return value


def _require_nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"{field} must be a non-negative integer")
    return value


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
            if error.code == 404:
                raise RuntimeError(
                    f"required archive/checksum missing (404): {url}"
                ) from error
            last_error = f"http_{error.code}"
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last_error = type(error).__name__
        if attempt < 4:
            time.sleep(2 + 2 * attempt)
    raise RuntimeError(f"download failed after retries ({last_error}): {url}")


def _month_bounds(month: str) -> tuple[int, int, int]:
    year, month_number = (int(item) for item in month.split("-"))
    if not 1 <= month_number <= 12:
        raise ValueError(f"invalid month: {month}")
    start = datetime(year, month_number, 1, tzinfo=UTC)
    if month_number == 12:
        stop = datetime(year + 1, 1, 1, tzinfo=UTC)
    else:
        stop = datetime(year, month_number + 1, 1, tzinfo=UTC)
    expected_rows = calendar.monthrange(year, month_number)[1] * 24
    return int(start.timestamp() * 1000), int(stop.timestamp() * 1000), expected_rows


def _iso_ms(value: int) -> str:
    return (
        datetime.fromtimestamp(value / 1000.0, tz=UTC)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _parse_checksum(raw: bytes, *, expected_name: str, url: str) -> str:
    try:
        text = raw.decode("utf-8").strip()
    except UnicodeDecodeError as error:
        raise RuntimeError(f"checksum is not UTF-8: {url}") from error
    parts = text.split()
    if len(parts) < 2:
        raise RuntimeError(f"checksum is malformed: {url}")
    digest = _require_hex(parts[0], field="provider checksum digest")
    observed_name = parts[-1].lstrip("*")
    if observed_name != expected_name:
        raise RuntimeError(
            f"checksum filename mismatch: expected {expected_name}, got {observed_name}"
        )
    return digest


def _missing_hash(missing: list[int]) -> str:
    return _sha256("\n".join(str(value) for value in missing).encode("ascii"))


def _parse_archive(
    *,
    kind: str,
    symbol: str,
    month: str,
    payload: bytes,
    checksum_payload: bytes,
    url: str,
) -> tuple[dict[str, object], list[ContractRow] | list[IndexRow]]:
    if kind not in {"perpetual_klines", "indexPriceKlines"}:
        raise ValueError(f"unsupported archive kind: {kind}")
    zip_name = f"{symbol}-{INTERVAL}-{month}.zip"
    csv_name = f"{symbol}-{INTERVAL}-{month}.csv"
    raw_sha = _sha256(payload)
    expected_sha = _parse_checksum(
        checksum_payload,
        expected_name=zip_name,
        url=url + ".CHECKSUM",
    )
    if raw_sha != expected_sha:
        raise RuntimeError(f"archive/provider checksum mismatch: {url}")

    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = [item for item in archive.infolist() if not item.is_dir()]
            if len(members) != 1:
                raise RuntimeError(f"archive must contain exactly one file: {url}")
            if members[0].filename != csv_name:
                raise RuntimeError(
                    f"archive member mismatch: {members[0].filename!r} != {csv_name!r}"
                )
            with archive.open(members[0], "r") as raw_csv:
                rows = list(
                    csv.reader(
                        io.TextIOWrapper(raw_csv, encoding="utf-8-sig", newline="")
                    )
                )
    except zipfile.BadZipFile as error:
        raise RuntimeError(f"invalid ZIP archive: {url}") from error
    if not rows:
        raise RuntimeError(f"empty CSV archive: {url}")

    first = rows[0]
    try:
        int(first[0])
        header_present = False
    except (ValueError, IndexError):
        header_present = True
    if header_present and tuple(first) != EXPECTED_HEADER:
        raise RuntimeError(f"unexpected CSV header: {url}")
    data_rows = rows[1:] if header_present else rows
    if not data_rows:
        raise RuntimeError(f"archive contains no data rows: {url}")

    month_start, next_month, expected_rows = _month_bounds(month)
    open_times: list[int] = []
    contract_rows: list[ContractRow] = []
    index_rows: list[IndexRow] = []
    for row in data_rows:
        if len(row) != 12:
            raise RuntimeError(f"data field count is not 12: {url}")
        try:
            open_ms = int(row[0])
            close_ms = int(row[6])
            open_price = float(row[1])
            high = float(row[2])
            low = float(row[3])
            close = float(row[4])
            quote_volume = float(row[7])
        except ValueError as error:
            raise RuntimeError(f"numeric field parse failed: {url}") from error
        if not month_start <= open_ms < next_month:
            raise RuntimeError(f"row outside requested calendar month: {url}")
        if (open_ms - month_start) % INTERVAL_MS != 0:
            raise RuntimeError(f"row is off the native 1h grid: {url}")
        if close_ms != open_ms + INTERVAL_MS - 1:
            raise RuntimeError(f"close-time contract mismatch: {url}")
        prices = (open_price, high, low, close)
        if not all(math.isfinite(value) and value > 0.0 for value in prices):
            raise RuntimeError(f"OHLC must be finite and strictly positive: {url}")
        if low > min(open_price, close) or high < max(open_price, close) or low > high:
            raise RuntimeError(f"OHLC invariant failed: {url}")
        if not math.isfinite(quote_volume) or quote_volume < 0.0:
            raise RuntimeError(f"quote volume must be finite and non-negative: {url}")
        if open_times and open_ms <= open_times[-1]:
            raise RuntimeError(f"timestamps are not strictly increasing: {url}")
        open_times.append(open_ms)
        if kind == "perpetual_klines":
            contract_rows.append((open_ms, open_price, high, low, close, quote_volume))
        else:
            index_rows.append((open_ms, close))

    expected_grid = [
        month_start + index * INTERVAL_MS for index in range(expected_rows)
    ]
    observed = set(open_times)
    if len(observed) != len(open_times):
        raise RuntimeError(f"duplicate timestamp in archive: {url}")
    unexpected = observed.difference(expected_grid)
    if unexpected:
        raise RuntimeError(f"timestamp outside native monthly grid: {url}")
    missing = [timestamp for timestamp in expected_grid if timestamp not in observed]
    entry: dict[str, object] = {
        "kind": kind,
        "symbol": symbol,
        "month": month,
        "url": url,
        "checksum_url": url + ".CHECKSUM",
        "raw_sha256": raw_sha,
        "raw_size_bytes": len(payload),
        "checksum_expected_sha256": expected_sha,
        "checksum_verified": True,
        "csv_member": csv_name,
        "header_present": header_present,
        "field_count": 12,
        "expected_grid_rows": expected_rows,
        "row_count": len(open_times),
        "missing_grid_rows": len(missing),
        "missing_grid_open_times_sha256": _missing_hash(missing),
        "first_open_time": _iso_ms(open_times[0]),
        "last_open_time": _iso_ms(open_times[-1]),
        "on_requested_1h_grid": True,
        "structurally_valid": True,
    }
    return entry, contract_rows if kind == "perpetual_klines" else index_rows


def load_index_preflight(
    path: Path,
    *,
    expected_file_sha256: str = INDEX_PREFLIGHT_REPORT_SHA256,
    expected_content_digest: str = INDEX_PREFLIGHT_CONTENT_DIGEST,
) -> dict[tuple[str, str], dict[str, object]]:
    raw = path.read_bytes()
    if _sha256(raw) != expected_file_sha256:
        raise RuntimeError("index-preflight report file SHA-256 differs")
    try:
        report = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError("index-preflight report is not valid JSON") from error
    if not isinstance(report, dict):
        raise RuntimeError("index-preflight report must be an object")
    digest = report.get("content_digest")
    if digest != expected_content_digest:
        raise RuntimeError("index-preflight content digest authority differs")
    body = dict(report)
    body.pop("content_digest", None)
    if _canonical_digest(body) != digest:
        raise RuntimeError("index-preflight report self-digest differs")
    exact = {
        "issue_number": 571,
        "prereg_pr_head": PREREG_HEAD,
        "prereg_protocol_digest": PROTOCOL_DIGEST,
        "planned_archives": 120,
        "available_archives": 120,
        "checksum_verified_archives": 120,
        "valid_archives": 120,
        "source_root": INDEX_ROOT,
        "interval": INTERVAL,
        "status": "PASS_FULL_INDEX_PREFLIGHT",
        "sparse_rows_remain_unavailable": True,
        "replacement_source_used": False,
        "economic_values_inspected": False,
        "basis_or_return_computed": False,
        "target_relation_computed": False,
        "evaluation_pnl_inspected": False,
        "production_authorized": False,
    }
    for field_name, expected in exact.items():
        if report.get(field_name) != expected:
            raise RuntimeError(f"index-preflight field differs: {field_name}")
    if report.get("symbols") != list(SYMBOLS):
        raise RuntimeError("index-preflight symbol roster differs")
    if report.get("months") != list(MONTHS):
        raise RuntimeError("index-preflight month roster differs")
    entries = report.get("entries")
    if not isinstance(entries, list) or len(entries) != 120:
        raise RuntimeError("index-preflight entry roster differs")
    resolved: dict[tuple[str, str], dict[str, object]] = {}
    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            raise RuntimeError("index-preflight entry must be an object")
        symbol = raw_entry.get("symbol")
        month = raw_entry.get("month")
        if not isinstance(symbol, str) or not isinstance(month, str):
            raise RuntimeError("index-preflight entry key is malformed")
        entry_key = (symbol, month)
        if entry_key in resolved or symbol not in SYMBOLS or month not in MONTHS:
            raise RuntimeError("index-preflight entry key is duplicate or unexpected")
        expected_url = (
            f"{INDEX_ROOT}/{symbol}/{INTERVAL}/{symbol}-{INTERVAL}-{month}.zip"
        )
        if raw_entry.get("url") != expected_url:
            raise RuntimeError("index-preflight archive URL differs")
        for flag in ("available", "checksum_verified", "structurally_valid"):
            if raw_entry.get(flag) is not True:
                raise RuntimeError(f"index-preflight entry failed: {flag}")
        _require_hex(raw_entry.get("raw_sha256"), field="index raw_sha256")
        _require_hex(
            raw_entry.get("missing_grid_open_times_sha256"),
            field="index missing-grid digest",
        )
        missing = raw_entry.get("missing_grid_rows")
        if isinstance(missing, bool) or not isinstance(missing, int) or missing < 0:
            raise RuntimeError("index-preflight missing_grid_rows is malformed")
        resolved[entry_key] = raw_entry
    if set(resolved) != {(symbol, month) for symbol in SYMBOLS for month in MONTHS}:
        raise RuntimeError("index-preflight exact key roster differs")
    return resolved


def _archive_url(*, root: str, symbol: str, month: str) -> str:
    return f"{root}/{symbol}/{INTERVAL}/{symbol}-{INTERVAL}-{month}.zip"


def download_all_sources(
    preflight: Mapping[tuple[str, str], Mapping[str, object]],
    *,
    fetch_bytes: FetchBytes = _fetch_bytes,
) -> tuple[
    list[dict[str, object]],
    dict[str, list[ContractRow]],
    dict[str, list[IndexRow]],
]:
    entries: list[dict[str, object]] = []
    contract_by_symbol: dict[str, list[ContractRow]] = {
        symbol: [] for symbol in SYMBOLS
    }
    index_by_symbol: dict[str, list[IndexRow]] = {symbol: [] for symbol in SYMBOLS}
    for symbol in SYMBOLS:
        for month in MONTHS:
            contract_url = _archive_url(root=CONTRACT_ROOT, symbol=symbol, month=month)
            contract_entry, contract_rows = _parse_archive(
                kind="perpetual_klines",
                symbol=symbol,
                month=month,
                payload=fetch_bytes(contract_url),
                checksum_payload=fetch_bytes(contract_url + ".CHECKSUM"),
                url=contract_url,
            )
            assert isinstance(contract_rows, list)
            contract_by_symbol[symbol].extend(contract_rows)  # type: ignore[arg-type]
            entries.append(contract_entry)

            index_url = _archive_url(root=INDEX_ROOT, symbol=symbol, month=month)
            index_entry, index_rows = _parse_archive(
                kind="indexPriceKlines",
                symbol=symbol,
                month=month,
                payload=fetch_bytes(index_url),
                checksum_payload=fetch_bytes(index_url + ".CHECKSUM"),
                url=index_url,
            )
            authority = preflight[(symbol, month)]
            if index_entry["raw_sha256"] != authority.get("raw_sha256"):
                raise RuntimeError(
                    f"index raw SHA differs from frozen preflight: {symbol} {month}"
                )
            if index_entry["missing_grid_rows"] != authority.get("missing_grid_rows"):
                raise RuntimeError(
                    f"index missing-grid count differs from frozen preflight: {symbol} {month}"
                )
            if index_entry["missing_grid_open_times_sha256"] != authority.get(
                "missing_grid_open_times_sha256"
            ):
                raise RuntimeError(
                    f"index missing-grid identity differs from frozen preflight: {symbol} {month}"
                )
            index_by_symbol[symbol].extend(index_rows)  # type: ignore[arg-type]
            entries.append(index_entry)
    if len(entries) != 240:
        raise RuntimeError("combined raw archive roster is not exactly 240")
    for symbol in SYMBOLS:
        for rows, label in (
            (contract_by_symbol[symbol], "contract"),
            (index_by_symbol[symbol], "index"),
        ):
            timestamps = [item[0] for item in rows]
            if not timestamps or any(
                b <= a for a, b in zip(timestamps, timestamps[1:])
            ):
                raise RuntimeError(
                    f"{label} rows are not strictly increasing: {symbol}"
                )
            if len(set(timestamps)) != len(timestamps):
                raise RuntimeError(f"{label} rows contain duplicates: {symbol}")
    return entries, contract_by_symbol, index_by_symbol


def build_source_manifest(entries: list[dict[str, object]]) -> dict[str, object]:
    if len(entries) != 240:
        raise RuntimeError("source manifest requires exactly 240 archive entries")
    missing_contract = {
        symbol: sum(
            _require_nonnegative_int(
                entry["missing_grid_rows"], field="contract missing_grid_rows"
            )
            for entry in entries
            if entry["symbol"] == symbol and entry["kind"] == "perpetual_klines"
        )
        for symbol in SYMBOLS
    }
    missing_index = {
        symbol: sum(
            _require_nonnegative_int(
                entry["missing_grid_rows"], field="index missing_grid_rows"
            )
            for entry in entries
            if entry["symbol"] == symbol and entry["kind"] == "indexPriceKlines"
        )
        for symbol in SYMBOLS
    }
    body: dict[str, object] = {
        "schema_version": "issue577_basis_training_source_manifest_v1",
        "issue_number": ISSUE_NUMBER,
        "protocol_digest": PROTOCOL_DIGEST,
        "prereg_head": PREREG_HEAD,
        "prereg_seal_run_id": PREREG_SEAL_RUN_ID,
        "prereg_seal_artifact_id": PREREG_SEAL_ARTIFACT_ID,
        "prereg_seal_api_digest": PREREG_SEAL_API_DIGEST,
        "prereg_fresh_artifact_id": PREREG_FRESH_ARTIFACT_ID,
        "prereg_fresh_api_digest": PREREG_FRESH_API_DIGEST,
        "index_preflight_run_id": INDEX_PREFLIGHT_RUN_ID,
        "index_preflight_artifact_id": INDEX_PREFLIGHT_ARTIFACT_ID,
        "index_preflight_api_digest": INDEX_PREFLIGHT_API_DIGEST,
        "index_preflight_report_sha256": INDEX_PREFLIGHT_REPORT_SHA256,
        "index_preflight_content_digest": INDEX_PREFLIGHT_CONTENT_DIGEST,
        "source_implementation_head": SOURCE_IMPLEMENTATION_HEAD,
        "source_final_verification_head": SOURCE_FINAL_VERIFICATION_HEAD,
        "source_final_verification_run_id": SOURCE_FINAL_VERIFICATION_RUN_ID,
        "calibration_head": CALIBRATION_HEAD,
        "calibration_verification_run_id": CALIBRATION_VERIFICATION_RUN_ID,
        "symbols": list(SYMBOLS),
        "months": list(MONTHS),
        "planned_archives": 240,
        "perpetual_archive_root": CONTRACT_ROOT,
        "index_archive_root": INDEX_ROOT,
        "interval": INTERVAL,
        "fit_start": "2021-01-01T01:00:00Z",
        "fit_cutoff": "2023-01-01T00:00:00Z",
        "missing_contract_grid_rows_by_symbol": missing_contract,
        "missing_index_grid_rows_by_symbol": missing_index,
        "native_missing_rows_remain_unavailable": True,
        "replacement_source_used": False,
        "rest_or_auto_fallback_used": False,
        "economic_values_logged": False,
        "evaluation_or_final_data_accessed": False,
        "evaluation_pnl_computed": False,
        "entries": entries,
    }
    return {**body, "content_digest": _canonical_digest(body)}


def _contract_series(rows: list[ContractRow]) -> RawMarketSeries:
    open_ms = np.asarray([item[0] for item in rows], dtype=np.int64)
    timestamps = (
        (open_ms + INTERVAL_MS).astype("datetime64[ms]").astype("datetime64[ns]")
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
    )


def _index_series(rows: list[IndexRow]) -> RawIndexPriceSeries:
    open_ms = np.asarray([item[0] for item in rows], dtype=np.int64)
    timestamps = (
        (open_ms + INTERVAL_MS).astype("datetime64[ms]").astype("datetime64[ns]")
    )
    return RawIndexPriceSeries(
        timestamps=timestamps,
        available_at=timestamps,
        close=np.asarray([item[1] for item in rows], dtype=np.float64),
    )


class FrozenBasisSource:
    def __init__(
        self,
        contract_by_symbol: Mapping[str, list[ContractRow]],
        index_by_symbol: Mapping[str, list[IndexRow]],
        *,
        source_manifest_digest: str,
    ) -> None:
        _require_hex(source_manifest_digest, field="source_manifest_digest")
        self._contract = {
            symbol: _contract_series(contract_by_symbol[symbol]) for symbol in SYMBOLS
        }
        self._index = {
            symbol: _index_series(index_by_symbol[symbol]) for symbol in SYMBOLS
        }
        self._source_manifest_digest = source_manifest_digest

    def load(self, symbol: str) -> RawMarketSeries:
        return self._contract[symbol]

    def load_index_price(self, symbol: str, timeframe: str) -> RawIndexPriceSeries:
        if timeframe != INTERVAL:
            raise ValueError("sealed basis index source supports exactly 1h")
        return self._index[symbol]

    @property
    def index_price_provenance(self) -> Mapping[str, object]:
        return {
            "schema_version": "issue577_index_price_provenance_v1",
            "source_family": "indexPriceKlines",
            "market": "USD_M",
            "interval": INTERVAL,
            "manifest_digest": self._source_manifest_digest,
            "preflight_content_digest": INDEX_PREFLIGHT_CONTENT_DIGEST,
        }


def build_dataset(
    contract_by_symbol: Mapping[str, list[ContractRow]],
    index_by_symbol: Mapping[str, list[IndexRow]],
    *,
    source_manifest_digest: str,
) -> MarketDataset:
    source = FrozenBasisSource(
        contract_by_symbol,
        index_by_symbol,
        source_manifest_digest=source_manifest_digest,
    )
    config = MarketBuildConfig(
        base_timeframe=INTERVAL,
        features=(
            FeatureSpec(
                name="1h__perp_index_log_basis_bps",
                kind=FeatureKind.PERP_INDEX_LOG_BASIS_BPS,
                lookback=1,
                normalization=NormalizationMode.NONE,
                normalization_window=1,
                min_periods=1,
            ),
        ),
    )
    contracts = tuple(
        InstrumentContract(
            symbol=symbol,
            listed_at=datetime(2021, 1, 1, tzinfo=UTC),
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
            "prereg_head": PREREG_HEAD,
            "source_implementation_head": SOURCE_IMPLEMENTATION_HEAD,
            "source_final_verification_head": SOURCE_FINAL_VERIFICATION_HEAD,
            "calibration_head": CALIBRATION_HEAD,
            "source_manifest_digest": source_manifest_digest,
            "index_preflight_content_digest": INDEX_PREFLIGHT_CONTENT_DIGEST,
        },
    )
    if dataset.symbols != SYMBOLS:
        raise RuntimeError("Dataset symbol roster differs from frozen roster")
    if dataset.feature_names != ("1h__perp_index_log_basis_bps",):
        raise RuntimeError("Dataset feature identity differs from frozen basis feature")
    if dataset.timestamps[0] != np.datetime64("2021-01-01T01:00:00", "ns"):
        raise RuntimeError("Dataset first completed timestamp differs")
    if dataset.timestamps[-1] != np.datetime64("2023-01-01T00:00:00", "ns"):
        raise RuntimeError("Dataset last completed timestamp differs")
    return dataset


def build_result(
    dataset: MarketDataset,
    *,
    source_manifest_digest: str,
) -> PerpIndexBasisCalibrationResult:
    protocol = canonical_perp_index_basis_protocol()
    if protocol.digest != PROTOCOL_DIGEST:
        raise RuntimeError("canonical protocol digest changed before real execution")
    result = calibrate_perp_index_basis(
        dataset,
        protocol,
        calibration_head=CALIBRATION_HEAD,
        source_manifest_digest=source_manifest_digest,
    )
    if result.protocol_digest != PROTOCOL_DIGEST:
        raise RuntimeError("result protocol authority differs")
    if result.source_implementation_head != SOURCE_IMPLEMENTATION_HEAD:
        raise RuntimeError("result source implementation authority differs")
    if result.calibration_head != CALIBRATION_HEAD:
        raise RuntimeError("result calibration authority differs")
    if result.source_manifest_digest != source_manifest_digest:
        raise RuntimeError("result source manifest authority differs")
    if result.training_relation_executed is not True:
        raise RuntimeError("result does not record training-only execution")
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
        raise RuntimeError("result crossed a forbidden evaluation/production boundary")
    return result


def reconstruct_from_raw(
    preflight_path: Path,
    *,
    fetch_bytes: FetchBytes = _fetch_bytes,
) -> tuple[dict[str, object], MarketDataset, PerpIndexBasisCalibrationResult]:
    preflight = load_index_preflight(preflight_path)
    entries, contract_by_symbol, index_by_symbol = download_all_sources(
        preflight,
        fetch_bytes=fetch_bytes,
    )
    manifest = build_source_manifest(entries)
    manifest_digest = str(manifest["content_digest"])
    dataset = build_dataset(
        contract_by_symbol,
        index_by_symbol,
        source_manifest_digest=manifest_digest,
    )
    result = build_result(dataset, source_manifest_digest=manifest_digest)
    return manifest, dataset, result


def result_bytes(result: PerpIndexBasisCalibrationResult) -> bytes:
    return canonical_json_bytes(result.to_artifact_payload())


def manifest_bytes(manifest: dict[str, object]) -> bytes:
    return canonical_json_bytes(manifest)


def authority_summary() -> dict[str, object]:
    return {
        "issue_number": ISSUE_NUMBER,
        "protocol_digest": PROTOCOL_DIGEST,
        "prereg_head": PREREG_HEAD,
        "index_preflight_run_id": INDEX_PREFLIGHT_RUN_ID,
        "index_preflight_artifact_id": INDEX_PREFLIGHT_ARTIFACT_ID,
        "index_preflight_api_digest": INDEX_PREFLIGHT_API_DIGEST,
        "index_preflight_report_sha256": INDEX_PREFLIGHT_REPORT_SHA256,
        "index_preflight_content_digest": INDEX_PREFLIGHT_CONTENT_DIGEST,
        "source_implementation_head": SOURCE_IMPLEMENTATION_HEAD,
        "source_final_verification_head": SOURCE_FINAL_VERIFICATION_HEAD,
        "source_final_verification_run_id": SOURCE_FINAL_VERIFICATION_RUN_ID,
        "calibration_head": CALIBRATION_HEAD,
        "calibration_verification_run_id": CALIBRATION_VERIFICATION_RUN_ID,
    }


__all__ = [
    "CALIBRATION_HEAD",
    "INDEX_PREFLIGHT_API_DIGEST",
    "INDEX_PREFLIGHT_ARTIFACT_ID",
    "INDEX_PREFLIGHT_CONTENT_DIGEST",
    "INDEX_PREFLIGHT_REPORT_SHA256",
    "INDEX_PREFLIGHT_RUN_ID",
    "ISSUE_NUMBER",
    "MONTHS",
    "PROTOCOL_DIGEST",
    "SOURCE_FINAL_VERIFICATION_HEAD",
    "SOURCE_FINAL_VERIFICATION_RUN_ID",
    "SOURCE_IMPLEMENTATION_HEAD",
    "SYMBOLS",
    "authority_summary",
    "build_dataset",
    "build_result",
    "build_source_manifest",
    "download_all_sources",
    "load_index_preflight",
    "manifest_bytes",
    "reconstruct_from_raw",
    "result_bytes",
]
