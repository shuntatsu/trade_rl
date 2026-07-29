from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def write(relative: str, content: str) -> None:
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = content.strip() + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == normalized:
        return
    path.write_text(normalized, encoding="utf-8")


def replace_once(relative: str, old: str, new: str) -> None:
    path = ROOT / relative
    content = path.read_text(encoding="utf-8")
    if new in content:
        return
    count = content.count(old)
    if count != 1:
        raise RuntimeError(f"{relative}: expected one replacement anchor, found {count}")
    path.write_text(content.replace(old, new), encoding="utf-8")


def append_once(relative: str, marker: str, content: str) -> None:
    path = ROOT / relative
    current = path.read_text(encoding="utf-8")
    if marker in current:
        return
    path.write_text(current.rstrip() + "\n\n" + content.strip() + "\n", encoding="utf-8")


def update_json(relative: str, transform) -> None:
    path = ROOT / relative
    payload = json.loads(path.read_text(encoding="utf-8"))
    transform(payload)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def create_economic_semantics() -> None:
    write(
        "trade_rl/data/economic_semantics.py",
        r'''
"""Canonical explicit economic arrays shared by all market dataset adapters."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Mapping

import numpy as np

from trade_rl.data.contracts import InstrumentContract


def _readonly(value: object, *, shape: tuple[int, int], dtype: np.dtype, field: str) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim == 0:
        array = np.full(shape, array.item(), dtype=dtype)
    elif array.shape != shape:
        try:
            array = np.broadcast_to(array, shape)
        except ValueError as error:
            raise ValueError(f"{field} cannot be broadcast to economic-array shape") from error
    result = np.array(array, dtype=dtype, copy=True, order="C")
    if np.issubdtype(result.dtype, np.floating) and not np.isfinite(result).all():
        raise ValueError(f"{field} must contain only finite values")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class MarketEconomicSemantics:
    """All execution/accounting arrays that must never depend on dataset defaults."""

    symbol_active: np.ndarray
    asset_active: np.ndarray
    tradable: np.ndarray
    information_available: np.ndarray
    available_at: np.ndarray
    fee_rate: np.ndarray
    maker_fee_rate: np.ndarray
    taker_fee_rate: np.ndarray
    spread_rate: np.ndarray
    max_participation_rate: np.ndarray
    minimum_notional: np.ndarray
    lot_size: np.ndarray
    tick_size: np.ndarray
    borrow_available: np.ndarray
    borrow_rate: np.ndarray
    funding_due: np.ndarray
    buy_allowed: np.ndarray
    sell_allowed: np.ndarray
    mark_price: np.ndarray
    index_price: np.ndarray

    def __post_init__(self) -> None:
        shapes = {getattr(self, item.name).shape for item in fields(self)}
        if len(shapes) != 1:
            raise ValueError("economic arrays must share one bar-by-symbol shape")
        for item in fields(self):
            value = getattr(self, item.name)
            if value.flags.writeable:
                raise ValueError(f"economic array {item.name} must be immutable")

    def market_dataset_kwargs(self) -> Mapping[str, np.ndarray]:
        return {item.name: getattr(self, item.name) for item in fields(self)}


def build_market_economic_semantics(
    *,
    timestamps: np.ndarray,
    instruments: tuple[InstrumentContract, ...],
    row_present: object,
    raw_tradable: object,
    source_information_available: object,
    available_at: object,
    close: object,
    funding_event_count: object,
    fee_rate: object = 0.0,
    maker_fee_rate: object = 0.0,
    taker_fee_rate: object = 0.0,
    spread_rate: object = 0.0,
    max_participation_rate: object = 1.0,
    borrow_available: object = True,
    borrow_rate: object = 0.0,
    buy_allowed: object = True,
    sell_allowed: object = True,
    mark_price: object | None = None,
    index_price: object | None = None,
) -> MarketEconomicSemantics:
    """Resolve one point-in-time, explicit economic contract for a dataset."""

    resolved_timestamps = np.asarray(timestamps, dtype="datetime64[ns]")
    if resolved_timestamps.ndim != 1 or resolved_timestamps.size < 2:
        raise ValueError("economic timestamps must be a rank-one market clock")
    if not instruments:
        raise ValueError("economic semantics require instruments")
    shape = (len(resolved_timestamps), len(instruments))
    rows = _readonly(row_present, shape=shape, dtype=np.dtype(np.bool_), field="row_present")
    raw_trade = _readonly(raw_tradable, shape=shape, dtype=np.dtype(np.bool_), field="raw_tradable")
    source_info = _readonly(
        source_information_available,
        shape=shape,
        dtype=np.dtype(np.bool_),
        field="source_information_available",
    )
    resolved_available_at = _readonly(
        available_at,
        shape=shape,
        dtype=np.dtype("datetime64[ns]"),
        field="available_at",
    )
    close_array = _readonly(close, shape=shape, dtype=np.dtype(np.float64), field="close")
    funding_counts = _readonly(
        funding_event_count,
        shape=shape,
        dtype=np.dtype(np.int32),
        field="funding_event_count",
    )
    if np.any(funding_counts < 0):
        raise ValueError("funding_event_count must be non-negative")

    active = np.zeros(shape, dtype=np.bool_)
    tick = np.zeros(shape, dtype=np.float64)
    lot = np.zeros(shape, dtype=np.float64)
    minimum = np.zeros(shape, dtype=np.float64)
    for symbol_index, contract in enumerate(instruments):
        listed = np.datetime64(contract.listed_at.astimezone(__import__("datetime").UTC).replace(tzinfo=None), "ns")
        mask = resolved_timestamps >= listed
        if contract.delisted_at is not None:
            delisted = np.datetime64(contract.delisted_at.astimezone(__import__("datetime").UTC).replace(tzinfo=None), "ns")
            mask &= resolved_timestamps < delisted
        active[:, symbol_index] = mask
        resolved_tick, resolved_lot, resolved_minimum = contract.execution_rule_arrays(
            resolved_timestamps
        )
        tick[:, symbol_index] = resolved_tick
        lot[:, symbol_index] = resolved_lot
        minimum[:, symbol_index] = resolved_minimum

    causal_time = resolved_available_at <= np.broadcast_to(
        resolved_timestamps[:, None], shape
    )
    active_ro = _readonly(active, shape=shape, dtype=np.dtype(np.bool_), field="symbol_active")
    information = _readonly(
        source_info & rows & active & causal_time,
        shape=shape,
        dtype=np.dtype(np.bool_),
        field="information_available",
    )
    tradable = _readonly(
        raw_trade & rows & active,
        shape=shape,
        dtype=np.dtype(np.bool_),
        field="tradable",
    )
    resolved_borrow_available = _readonly(
        np.asarray(borrow_available, dtype=np.bool_) & active,
        shape=shape,
        dtype=np.dtype(np.bool_),
        field="borrow_available",
    )
    resolved_buy_allowed = _readonly(
        np.asarray(buy_allowed, dtype=np.bool_) & active,
        shape=shape,
        dtype=np.dtype(np.bool_),
        field="buy_allowed",
    )
    resolved_sell_allowed = _readonly(
        np.asarray(sell_allowed, dtype=np.bool_) & active,
        shape=shape,
        dtype=np.dtype(np.bool_),
        field="sell_allowed",
    )
    resolved_mark = close_array if mark_price is None else _readonly(
        mark_price, shape=shape, dtype=np.dtype(np.float64), field="mark_price"
    )
    resolved_index = close_array if index_price is None else _readonly(
        index_price, shape=shape, dtype=np.dtype(np.float64), field="index_price"
    )
    return MarketEconomicSemantics(
        symbol_active=active_ro,
        asset_active=active_ro,
        tradable=tradable,
        information_available=information,
        available_at=resolved_available_at,
        fee_rate=_readonly(fee_rate, shape=shape, dtype=np.dtype(np.float64), field="fee_rate"),
        maker_fee_rate=_readonly(maker_fee_rate, shape=shape, dtype=np.dtype(np.float64), field="maker_fee_rate"),
        taker_fee_rate=_readonly(taker_fee_rate, shape=shape, dtype=np.dtype(np.float64), field="taker_fee_rate"),
        spread_rate=_readonly(spread_rate, shape=shape, dtype=np.dtype(np.float64), field="spread_rate"),
        max_participation_rate=_readonly(
            max_participation_rate,
            shape=shape,
            dtype=np.dtype(np.float64),
            field="max_participation_rate",
        ),
        minimum_notional=_readonly(minimum, shape=shape, dtype=np.dtype(np.float64), field="minimum_notional"),
        lot_size=_readonly(lot, shape=shape, dtype=np.dtype(np.float64), field="lot_size"),
        tick_size=_readonly(tick, shape=shape, dtype=np.dtype(np.float64), field="tick_size"),
        borrow_available=resolved_borrow_available,
        borrow_rate=_readonly(borrow_rate, shape=shape, dtype=np.dtype(np.float64), field="borrow_rate"),
        funding_due=_readonly(funding_counts > 0, shape=shape, dtype=np.dtype(np.bool_), field="funding_due"),
        buy_allowed=resolved_buy_allowed,
        sell_allowed=resolved_sell_allowed,
        mark_price=resolved_mark,
        index_price=resolved_index,
    )


__all__ = ["MarketEconomicSemantics", "build_market_economic_semantics"]
''',
    )


def patch_dataset_builders() -> None:
    replace_once(
        "trade_rl/data/builder.py",
        "from trade_rl.data.cross_asset_features import (",
        "from trade_rl.data.economic_semantics import build_market_economic_semantics\nfrom trade_rl.data.cross_asset_features import (",
    )
    replace_once(
        "trade_rl/data/builder.py",
        '''        symbol_active = np.zeros_like(row_present)\n        tick_size = np.zeros_like(open_price)\n        lot_size = np.zeros_like(open_price)\n        minimum_notional = np.zeros_like(open_price)\n''',
        "",
    )
    replace_once(
        "trade_rl/data/builder.py",
        '''\n            listed = _utc_datetime64(contract.listed_at)\n            active = timestamps >= listed\n            if contract.delisted_at is not None:\n                active &= timestamps < _utc_datetime64(contract.delisted_at)\n            symbol_active[:, symbol_index] = active\n            resolved_tick, resolved_lot, resolved_minimum = (\n                contract.execution_rule_arrays(timestamps)\n            )\n            tick_size[:, symbol_index] = resolved_tick\n            lot_size[:, symbol_index] = resolved_lot\n            minimum_notional[:, symbol_index] = resolved_minimum\n\n        information_available &= symbol_active & row_present\n        causal_row_present = row_present & information_available\n        tradable = symbol_active & row_present & raw_tradable\n''',
        '''\n\n        economics = build_market_economic_semantics(\n            timestamps=timestamps,\n            instruments=instruments,\n            row_present=row_present,\n            raw_tradable=raw_tradable,\n            source_information_available=information_available,\n            available_at=available_at,\n            close=close,\n            funding_event_count=funding_event_count,\n        )\n        symbol_active = economics.symbol_active\n        information_available = economics.information_available\n        available_at = economics.available_at\n        causal_row_present = row_present & information_available\n        tradable = economics.tradable\n''',
    )
    replace_once(
        "trade_rl/data/builder.py",
        '''            tradable=tradable,\n            symbol_active=symbol_active,\n            information_available=information_available,\n            available_at=available_at,\n''',
        '''            **economics.market_dataset_kwargs(),\n''',
    )
    replace_once(
        "trade_rl/data/builder.py",
        '''            tick_size=tick_size,\n            lot_size=lot_size,\n            minimum_notional=minimum_notional,\n''',
        "",
    )

    replace_once(
        "trade_rl/integrations/postgres_market_dataset.py",
        "from trade_rl.data.contracts import FeatureSpec, VolumeUnit",
        "from trade_rl.data.contracts import (\n    FeatureSpec,\n    InstrumentContract,\n    InstrumentExecutionRule,\n    VolumeUnit,\n)\nfrom trade_rl.data.economic_semantics import build_market_economic_semantics",
    )
    replace_once(
        "trade_rl/integrations/postgres_market_dataset.py",
        '''    metadata_evidence_digest: str,\n    indicator_bundle: NativeIndicatorArtifactBundle | None = None,\n''',
        '''    metadata_evidence_digest: str,\n    execution_rule_histories: Mapping[str, Sequence[InstrumentExecutionRule]] | None = None,\n    indicator_bundle: NativeIndicatorArtifactBundle | None = None,\n''',
    )
    replace_once(
        "trade_rl/integrations/postgres_market_dataset.py",
        '''    n_bars = len(timestamps_ms)\n    price_shape = (n_bars, len(selected))\n    active = np.ones(price_shape, dtype=np.bool_)\n    tradable = np.ones(price_shape, dtype=np.bool_)\n    timestamps = timestamps_ms.astype("datetime64[ms]").astype("datetime64[ns]")\n    available_at = np.broadcast_to(timestamps[:, None], price_shape).copy()\n\n    log_returns = np.zeros(price_shape, dtype=np.float64)\n''',
        '''    n_bars = len(timestamps_ms)\n    price_shape = (n_bars, len(selected))\n    timestamps = timestamps_ms.astype("datetime64[ms]").astype("datetime64[ns]")\n    available_at = np.broadcast_to(timestamps[:, None], price_shape).copy()\n    if execution_rule_histories is not None:\n        missing_histories = set(selected) - set(execution_rule_histories)\n        unknown_histories = set(execution_rule_histories) - set(selected)\n        if missing_histories or unknown_histories:\n            raise ValueError("PostgreSQL execution-rule histories must match selected symbols")\n    instruments = tuple(\n        InstrumentContract(\n            symbol=symbol,\n            listed_at=start,\n            volume_unit=VolumeUnit.QUOTE_NOTIONAL,\n            tick_size=_metadata_number(metadata, symbol, "tick_size"),\n            lot_size=_metadata_number(metadata, symbol, "lot_size"),\n            minimum_notional=_metadata_number(metadata, symbol, "minimum_notional"),\n            execution_rules=(\n                ()\n                if execution_rule_histories is None\n                else tuple(execution_rule_histories[symbol])\n            ),\n        )\n        for symbol in selected\n    )\n    economics = build_market_economic_semantics(\n        timestamps=timestamps,\n        instruments=instruments,\n        row_present=np.ones(price_shape, dtype=np.bool_),\n        raw_tradable=np.ones(price_shape, dtype=np.bool_),\n        source_information_available=np.ones(price_shape, dtype=np.bool_),\n        available_at=available_at,\n        close=raw["close"],\n        funding_event_count=funding_counts,\n    )\n\n    log_returns = np.zeros(price_shape, dtype=np.float64)\n''',
    )
    replace_once(
        "trade_rl/integrations/postgres_market_dataset.py",
        '''    global_features = np.zeros((n_bars, 4), dtype=np.float32)\n    global_features[:, :2] = 1.0\n''',
        '''    global_features = np.zeros((n_bars, 4), dtype=np.float32)\n    global_features[:, 0] = economics.symbol_active.mean(axis=1)\n    global_features[:, 1] = (\n        economics.tradable & economics.information_available\n    ).mean(axis=1)\n''',
    )
    replace_once(
        "trade_rl/integrations/postgres_market_dataset.py",
        '''    tick_size = np.broadcast_to(\n        np.asarray(\n            [_metadata_number(metadata, symbol, "tick_size") for symbol in selected]\n        ),\n        price_shape,\n    ).copy()\n    lot_size = np.broadcast_to(\n        np.asarray(\n            [_metadata_number(metadata, symbol, "lot_size") for symbol in selected]\n        ),\n        price_shape,\n    ).copy()\n    minimum_notional = np.broadcast_to(\n        np.asarray(\n            [\n                _metadata_number(metadata, symbol, "minimum_notional")\n                for symbol in selected\n            ]\n        ),\n        price_shape,\n    ).copy()\n''',
        "",
    )
    replace_once(
        "trade_rl/integrations/postgres_market_dataset.py",
        '''        tradable=tradable,\n        feature_available=feature_available,\n''',
        '''        **economics.market_dataset_kwargs(),\n        feature_available=feature_available,\n''',
    )
    replace_once(
        "trade_rl/integrations/postgres_market_dataset.py",
        '''        minimum_notional=minimum_notional,\n        lot_size=lot_size,\n        tick_size=tick_size,\n        funding_due=funding_counts > 0,\n        asset_active=active,\n        symbol_active=active,\n        information_available=active,\n        available_at=available_at,\n''',
        "",
    )


def patch_pipeline_order() -> None:
    pipeline = "examples/binance-multitimeframe/full_research_pipeline.py"
    replace_once(
        pipeline,
        "def _build_dataset(\n",
        '''def validate_maintained_dataset_preset(
    dataset: MarketDataset,
    *,
    use_postgres: bool,
) -> None:
    if dataset.n_bars != _EXPECTED_15M_BARS:
        raise RuntimeError(
            f"expected {_EXPECTED_15M_BARS:,} 15-minute bars, observed {dataset.n_bars}"
        )
    expected_dataset_symbols = _SLOT_SYMBOLS if use_postgres else _SYMBOLS
    if dataset.symbols != expected_dataset_symbols:
        raise RuntimeError(f"unexpected symbol order: {dataset.symbols}")
    expected_features = tuple(
        spec.name
        for spec in binance_multitimeframe_feature_specs(
            base_timeframe="15m",
            feature_timeframes=_FEATURE_TIMEFRAMES,
        )
    )
    if len(expected_features) != 226:
        raise RuntimeError(
            f"extended feature contract must contain 226 features, got {len(expected_features)}"
        )
    expected_dataset_features = (
        (*expected_features, *(f"15m__symbol_id_{symbol}" for symbol in _SYMBOL_POOL))
        if use_postgres
        else expected_features
    )
    if dataset.feature_names != expected_dataset_features:
        raise RuntimeError(f"unexpected feature contract: {dataset.feature_names}")


def _build_dataset(
''',
    )
    replace_once(
        pipeline,
        '''                metadata=metadata,
                metadata_evidence_digest=metadata_evidence_digest,
                symbol_triplet_provenance=_ACTIVE_SYMBOL_TRIPLET,
''',
        '''                metadata=metadata,
                metadata_evidence_digest=metadata_evidence_digest,
                execution_rule_histories=execution_rule_histories,
                symbol_triplet_provenance=_ACTIVE_SYMBOL_TRIPLET,
''',
    )
    replace_once(
        pipeline,
        '''    published = publish_market_dataset_artifact(output, dataset)
    if dataset.n_bars != _EXPECTED_15M_BARS:
        raise RuntimeError(
            f"expected {_EXPECTED_15M_BARS:,} 15-minute bars, observed {dataset.n_bars}"
        )
    expected_dataset_symbols = _SLOT_SYMBOLS if use_postgres else _SYMBOLS
    if dataset.symbols != expected_dataset_symbols:
        raise RuntimeError(f"unexpected symbol order: {dataset.symbols}")
    expected_features = tuple(
        spec.name
        for spec in binance_multitimeframe_feature_specs(
            base_timeframe="15m",
            feature_timeframes=_FEATURE_TIMEFRAMES,
        )
    )
    if len(expected_features) != 226:
        raise RuntimeError(
            f"extended feature contract must contain 226 features, got {len(expected_features)}"
        )
    expected_dataset_features = (
        (*expected_features, *(f"15m__symbol_id_{symbol}" for symbol in _SYMBOL_POOL))
        if use_postgres
        else expected_features
    )
    if dataset.feature_names != expected_dataset_features:
        raise RuntimeError(f"unexpected feature contract: {dataset.feature_names}")
''',
        '''    validate_maintained_dataset_preset(dataset, use_postgres=use_postgres)
    published = publish_market_dataset_artifact(output, dataset)
''',
    )

def patch_binance_cache() -> None:
    path = "trade_rl/integrations/binance.py"
    replace_once(
        path,
        '''    def _request_bytes(self, url: str) -> bytes:\n        cache_path = self._vision_cache_path(url)\n        if cache_path is not None and cache_path.is_file():\n            payload = cache_path.read_bytes()\n            if not payload:\n                raise BinanceTransportError(\n                    f"cached Binance Vision archive is empty: {cache_path}"\n                )\n            return payload\n''',
        '''    def _validated_cached_vision_payload(self, url: str, cache_path: Path) -> bytes:\n        evidence_path = cache_path.with_suffix(".json")\n        if not evidence_path.is_file():\n            raise BinanceTransportError(\n                f"cached Binance Vision archive lacks content evidence: {cache_path}"\n            )\n        try:\n            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))\n        except (OSError, json.JSONDecodeError) as error:\n            raise BinanceTransportError("cached Binance Vision evidence is invalid") from error\n        expected = {\n            "acquired_at",\n            "downloader",\n            "etag",\n            "last_modified",\n            "schema_version",\n            "sha256",\n            "size_bytes",\n            "url",\n        }\n        if not isinstance(evidence, dict) or set(evidence) != expected:\n            raise BinanceTransportError("cached Binance Vision evidence fields are invalid")\n        payload = cache_path.read_bytes()\n        digest = hashlib.sha256(payload).hexdigest()\n        if (\n            evidence.get("schema_version") != "binance_vision_raw_cache_v1"\n            or evidence.get("url") != url\n            or evidence.get("size_bytes") != len(payload)\n            or evidence.get("sha256") != digest\n        ):\n            raise BinanceTransportError(\n                f"cached Binance Vision archive content digest or size mismatch: {cache_path}"\n            )\n        return payload\n\n    def _write_vision_cache(\n        self,\n        *,\n        url: str,\n        cache_path: Path,\n        payload: bytes,\n        etag: str | None,\n        last_modified: str | None,\n    ) -> None:\n        evidence = {\n            "acquired_at": datetime.now(UTC).isoformat(),\n            "downloader": _USER_AGENT,\n            "etag": etag,\n            "last_modified": last_modified,\n            "schema_version": "binance_vision_raw_cache_v1",\n            "sha256": hashlib.sha256(payload).hexdigest(),\n            "size_bytes": len(payload),\n            "url": url,\n        }\n        cache_path.parent.mkdir(parents=True, exist_ok=True)\n        evidence_path = cache_path.with_suffix(".json")\n        binary_temporary = cache_path.with_suffix(".bin.tmp")\n        evidence_temporary = evidence_path.with_suffix(".json.tmp")\n        binary_temporary.write_bytes(payload)\n        evidence_temporary.write_text(\n            json.dumps(evidence, allow_nan=False, sort_keys=True, separators=(",", ":"))\n            + "\\n",\n            encoding="utf-8",\n        )\n        binary_temporary.replace(cache_path)\n        evidence_temporary.replace(evidence_path)\n\n    def _request_bytes(self, url: str) -> bytes:\n        cache_path = self._vision_cache_path(url)\n        if cache_path is not None and cache_path.is_file():\n            return self._validated_cached_vision_payload(url, cache_path)\n''',
    )
    replace_once(
        path,
        '''                with urllib.request.urlopen(  # noqa: S310 - fixed HTTPS endpoints\n                    request,\n                    timeout=self.timeout_seconds,\n                ) as response:\n                    payload = response.read()\n                if not payload:\n''',
        '''                with urllib.request.urlopen(  # noqa: S310 - fixed HTTPS endpoints\n                    request,\n                    timeout=self.timeout_seconds,\n                ) as response:\n                    payload = response.read()\n                    headers = getattr(response, "headers", {})\n                    header_get = getattr(headers, "get", lambda _name: None)\n                    etag = header_get("ETag")\n                    last_modified = header_get("Last-Modified")\n                if not payload:\n''',
    )
    replace_once(
        path,
        '''                if cache_path is not None:\n                    cache_path.parent.mkdir(parents=True, exist_ok=True)\n                    temporary = cache_path.with_suffix(".tmp")\n                    temporary.write_bytes(payload)\n                    temporary.replace(cache_path)\n                return payload\n''',
        '''                if cache_path is not None:\n                    self._write_vision_cache(\n                        url=url,\n                        cache_path=cache_path,\n                        payload=payload,\n                        etag=None if etag is None else str(etag),\n                        last_modified=(\n                            None if last_modified is None else str(last_modified)\n                        ),\n                    )\n                return payload\n''',
    )


def patch_bc_gate() -> None:
    evaluation = "trade_rl/learning/evaluation.py"
    replace_once(
        evaluation,
        '''def _compounded_return(step_returns: np.ndarray) -> float:\n    return float(np.prod(1.0 + step_returns, dtype=np.float64) - 1.0)\n''',
        '''def _compounded_return(step_returns: np.ndarray) -> float:\n    return float(np.prod(1.0 + step_returns, dtype=np.float64) - 1.0)\n\n\ndef deterministic_bootstrap_upper_bound(\n    values: object,\n    *,\n    confidence_level: float,\n    resamples: int,\n    seed_material: str,\n) -> float:\n    """Return a reproducible one-sided bootstrap upper bound for the mean."""\n\n    sample = _finite_vector(values, field="bootstrap values")\n    if np.any(sample < 0.0):\n        raise ValueError("bootstrap regret values must be non-negative")\n    if (\n        not math.isfinite(confidence_level)\n        or not 0.5 < confidence_level < 1.0\n    ):\n        raise ValueError("bootstrap confidence_level must be within (0.5, 1)")\n    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 1_000:\n        raise ValueError("bootstrap resamples must be an integer of at least 1000")\n    if not isinstance(seed_material, str) or not seed_material:\n        raise ValueError("bootstrap seed_material must be non-empty")\n    seed = int(content_digest({"seed_material": seed_material})[:16], 16)\n    rng = np.random.default_rng(seed)\n    indices = rng.integers(0, len(sample), size=(resamples, len(sample)))\n    means = sample[indices].mean(axis=1, dtype=np.float64)\n    return float(np.quantile(means, confidence_level, method="higher"))\n''',
    )
    replace_once(
        evaluation,
        '''    maximum_causal_holdout_regret: float\n\n    def __post_init__(self) -> None:\n''',
        '''    maximum_causal_holdout_regret: float\n    minimum_causal_holdout_episodes: int = 1\n    maximum_causal_holdout_regret_upper_bound: float | None = None\n\n    def __post_init__(self) -> None:\n''',
    )
    replace_once(
        evaluation,
        '''        if (\n            not math.isfinite(self.maximum_causal_holdout_regret)\n            or self.maximum_causal_holdout_regret < 0.0\n        ):\n            raise ValueError("maximum_causal_holdout_regret must be non-negative")\n''',
        '''        if (\n            not math.isfinite(self.maximum_causal_holdout_regret)\n            or self.maximum_causal_holdout_regret < 0.0\n        ):\n            raise ValueError("maximum_causal_holdout_regret must be non-negative")\n        if (\n            isinstance(self.minimum_causal_holdout_episodes, bool)\n            or not isinstance(self.minimum_causal_holdout_episodes, int)\n            or self.minimum_causal_holdout_episodes <= 0\n        ):\n            raise ValueError("minimum_causal_holdout_episodes must be positive")\n        upper = self.maximum_causal_holdout_regret_upper_bound\n        if upper is not None and (not math.isfinite(upper) or upper < 0.0):\n            raise ValueError(\n                "maximum_causal_holdout_regret_upper_bound must be non-negative"\n            )\n''',
    )
    replace_once(
        evaluation,
        '''    causal_regret = (\n        None\n        if holdout is None\n        else max(0.0, -holdout.causal_policy_performance.net_return)\n    )\n''',
        '''    causal_regret = (\n        None\n        if holdout is None\n        else max(0.0, -holdout.causal_policy_performance.net_return)\n    )\n    causal_records = () if holdout is None else tuple(getattr(holdout, "records", ()))\n    causal_episode_support = len(causal_records) if causal_records else (0 if holdout is None else 1)\n    causal_regret_upper = (\n        None\n        if holdout is None\n        else getattr(holdout, "causal_regret_upper_confidence_bound", None)\n    )\n    if causal_regret_upper is None and holdout is not None and not causal_records:\n        causal_regret_upper = getattr(holdout, "heldout_oracle_regret", None)\n    upper_threshold = (\n        thresholds.maximum_causal_holdout_regret\n        if thresholds.maximum_causal_holdout_regret_upper_bound is None\n        else thresholds.maximum_causal_holdout_regret_upper_bound\n    )\n''',
    )
    replace_once(
        evaluation,
        '''        _gate_metric(\n            name="cash_baseline_after_cost_regret",\n''',
        '''        _gate_metric(\n            name="causal_regret_upper_confidence_bound",\n            observed=causal_regret_upper,\n            comparison="<=",\n            threshold=upper_threshold,\n            support=causal_episode_support,\n            minimum_support=thresholds.minimum_causal_holdout_episodes,\n            passed=(\n                causal_regret_upper is not None\n                and causal_regret_upper <= upper_threshold\n            ),\n            failure_reason=(\n                "causal holdout regret upper confidence bound exceeds the limit"\n            ),\n        ),\n        _gate_metric(\n            name="cash_baseline_after_cost_regret",\n''',
    )
    replace_once(
        evaluation,
        '''    "evaluate_behavior_cloning_gates",\n''',
        '''    "deterministic_bootstrap_upper_bound",\n    "evaluate_behavior_cloning_gates",\n''',
    )

    episode = "trade_rl/learning/episode_oracle_bc.py"
    replace_once(
        episode,
        '''    ActionPathCollapseEvidence,\n    PathPerformanceMetrics,\n)\n''',
        '''    ActionPathCollapseEvidence,\n    PathPerformanceMetrics,\n    deterministic_bootstrap_upper_bound,\n)\n''',
    )
    replace_once(
        episode,
        '''    normalized_oracle_regret: float\n    schema_version: str = EPISODE_ORACLE_BC_EVALUATION_SCHEMA\n''',
        '''    normalized_oracle_regret: float\n    causal_regret_upper_confidence_bound: float\n    bootstrap_confidence_level: float\n    bootstrap_resamples: int\n    schema_version: str = EPISODE_ORACLE_BC_EVALUATION_SCHEMA\n''',
    )
    replace_once(
        episode,
        '''            self.normalized_oracle_regret,\n        ):\n''',
        '''            self.normalized_oracle_regret,\n            self.causal_regret_upper_confidence_bound,\n            self.bootstrap_confidence_level,\n        ):\n''',
    )
    replace_once(
        episode,
        '''        if self.action_agreement_rate > 1.0:\n            raise ValueError("episode BC action agreement exceeds one")\n''',
        '''        if self.action_agreement_rate > 1.0:\n            raise ValueError("episode BC action agreement exceeds one")\n        if not 0.5 < self.bootstrap_confidence_level < 1.0:\n            raise ValueError("episode BC bootstrap confidence is invalid")\n        if self.bootstrap_resamples < 1_000:\n            raise ValueError("episode BC bootstrap resamples are insufficient")\n''',
    )
    replace_once(
        episode,
        '''            "normalized_oracle_regret": self.normalized_oracle_regret,\n            "records": tuple(record.to_dict() for record in self.records),\n''',
        '''            "normalized_oracle_regret": self.normalized_oracle_regret,\n            "causal_regret_upper_confidence_bound": (\n                self.causal_regret_upper_confidence_bound\n            ),\n            "bootstrap_confidence_level": self.bootstrap_confidence_level,\n            "bootstrap_resamples": self.bootstrap_resamples,\n            "records": tuple(record.to_dict() for record in self.records),\n''',
    )
    replace_once(
        episode,
        '''    output_root: Path,\n    action_tolerance: float = 0.05,\n''',
        '''    output_root: Path,\n    action_tolerance: float = 0.05,\n    bootstrap_confidence_level: float = 0.95,\n    bootstrap_resamples: int = 2_000,\n''',
    )
    replace_once(
        episode,
        '''    resolved_records = tuple(records)\n    worst = max(resolved_records, key=lambda record: record.normalized_oracle_regret)\n''',
        '''    resolved_records = tuple(records)\n    regret_upper = deterministic_bootstrap_upper_bound(\n        np.asarray(\n            [record.normalized_oracle_regret for record in resolved_records],\n            dtype=np.float64,\n        ),\n        confidence_level=bootstrap_confidence_level,\n        resamples=bootstrap_resamples,\n        seed_material=content_digest(\n            {\n                "batch_digest": batch.digest,\n                "validation_episode_ids": validation_ids,\n            }\n        ),\n    )\n    worst = max(resolved_records, key=lambda record: record.normalized_oracle_regret)\n''',
    )
    replace_once(
        episode,
        '''        normalized_oracle_regret=max(\n            record.normalized_oracle_regret for record in resolved_records\n        ),\n    )\n''',
        '''        normalized_oracle_regret=max(\n            record.normalized_oracle_regret for record in resolved_records\n        ),\n        causal_regret_upper_confidence_bound=regret_upper,\n        bootstrap_confidence_level=bootstrap_confidence_level,\n        bootstrap_resamples=bootstrap_resamples,\n    )\n''',
    )
    replace_once(
        episode,
        '''            "normalized_oracle_regret": holdout.normalized_oracle_regret,\n            "normalized_oracle_regret_maximum": 0.25,\n''',
        '''            "normalized_oracle_regret": holdout.normalized_oracle_regret,\n            "causal_regret_upper_confidence_bound": (\n                holdout.causal_regret_upper_confidence_bound\n            ),\n            "bootstrap_confidence_level": holdout.bootstrap_confidence_level,\n            "bootstrap_resamples": holdout.bootstrap_resamples,\n            "normalized_oracle_regret_maximum": 0.25,\n''',
    )

    training = "trade_rl/rl/training.py"
    replace_once(
        training,
        '''    behavior_cloning_min_causal_holdout_trades: int = 0\n    behavior_cloning_max_causal_holdout_regret: float = 0.0\n''',
        '''    behavior_cloning_min_causal_holdout_trades: int = 0\n    behavior_cloning_max_causal_holdout_regret: float = 0.0\n    behavior_cloning_causal_holdout_bootstrap_resamples: int = 2_000\n    behavior_cloning_causal_holdout_confidence_level: float = 0.95\n''',
    )
    replace_once(
        training,
        '''        if (\n            not math.isfinite(self.behavior_cloning_max_causal_holdout_regret)\n            or self.behavior_cloning_max_causal_holdout_regret < 0.0\n        ):\n            raise ValueError(\n                "behavior_cloning_max_causal_holdout_regret must be finite and non-negative"\n            )\n''',
        '''        if (\n            not math.isfinite(self.behavior_cloning_max_causal_holdout_regret)\n            or self.behavior_cloning_max_causal_holdout_regret < 0.0\n        ):\n            raise ValueError(\n                "behavior_cloning_max_causal_holdout_regret must be finite and non-negative"\n            )\n        if (\n            isinstance(self.behavior_cloning_causal_holdout_bootstrap_resamples, bool)\n            or not isinstance(\n                self.behavior_cloning_causal_holdout_bootstrap_resamples, int\n            )\n            or self.behavior_cloning_causal_holdout_bootstrap_resamples < 1_000\n        ):\n            raise ValueError(\n                "behavior_cloning_causal_holdout_bootstrap_resamples must be at least 1000"\n            )\n        if (\n            not math.isfinite(\n                self.behavior_cloning_causal_holdout_confidence_level\n            )\n            or not 0.5\n            < self.behavior_cloning_causal_holdout_confidence_level\n            < 1.0\n        ):\n            raise ValueError(\n                "behavior_cloning_causal_holdout_confidence_level must be within (0.5, 1)"\n            )\n''',
    )

    workflow = "trade_rl/workflows/training_run.py"
    replace_once(
        workflow,
        '''        "behavior_cloning_max_causal_holdout_regret",\n''',
        '''        "behavior_cloning_max_causal_holdout_regret",\n        "behavior_cloning_causal_holdout_bootstrap_resamples",\n        "behavior_cloning_causal_holdout_confidence_level",\n''',
    )

    sb3 = "trade_rl/integrations/sb3_training.py"
    replace_once(
        sb3,
        '''        maximum_causal_holdout_regret=float(\n            _required_hierarchical_config(\n                config, "behavior_cloning_max_causal_holdout_regret"\n            )\n        ),\n    )\n''',
        '''        maximum_causal_holdout_regret=float(\n            _required_hierarchical_config(\n                config, "behavior_cloning_max_causal_holdout_regret"\n            )\n        ),\n        minimum_causal_holdout_episodes=2,\n        maximum_causal_holdout_regret_upper_bound=float(\n            _required_hierarchical_config(\n                config, "behavior_cloning_max_causal_holdout_regret"\n            )\n        ),\n    )\n''',
    )
    replace_once(
        sb3,
        '''                            output_root=output_path.parent,\n                        )\n''',
        '''                            output_root=output_path.parent,\n                            bootstrap_confidence_level=(\n                                config.behavior_cloning_causal_holdout_confidence_level\n                            ),\n                            bootstrap_resamples=(\n                                config.behavior_cloning_causal_holdout_bootstrap_resamples\n                            ),\n                        )\n''',
    )

    for path in sorted((ROOT / "examples").rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if payload.get("schema_version") != "training_run_config_v3":
            continue
        training_payload = payload.get("training")
        if not isinstance(training_payload, dict):
            continue
        training_payload.setdefault(
            "behavior_cloning_causal_holdout_bootstrap_resamples", 2_000
        )
        training_payload.setdefault(
            "behavior_cloning_causal_holdout_confidence_level", 0.95
        )
        if path.name in {
            "training-target-weight-growth-ppo.json",
            "training-target-weight-constrained-growth.json",
            "training-target-weight-constrained-growth-discounted.json",
        }:
            training_payload["behavior_cloning_required_relative_improvement"] = max(
                float(training_payload.get("behavior_cloning_required_relative_improvement", 0.0)),
                0.005,
            )
            training_payload["behavior_cloning_min_causal_holdout_trades"] = max(
                int(training_payload.get("behavior_cloning_min_causal_holdout_trades", 0)),
                30,
            )
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def patch_change_intensity() -> None:
    write(
        "trade_rl/rl/action_telemetry.py",
        r'''
"""Metrics that distinguish deterministic composition, exploration and fills."""

from __future__ import annotations

import numpy as np


def _vector(value: object, *, field: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{field} must be a non-empty finite vector")
    return array


def hierarchical_action_stage_metrics(
    *,
    deterministic_composed: object,
    sampled_policy_action: object,
    submitted_target: object,
    effective_filled_weights: object,
) -> dict[str, float]:
    deterministic = _vector(deterministic_composed, field="deterministic_composed")
    sampled = _vector(sampled_policy_action, field="sampled_policy_action")
    submitted = _vector(submitted_target, field="submitted_target")
    effective = _vector(effective_filled_weights, field="effective_filled_weights")
    if len({len(deterministic), len(sampled), len(submitted), len(effective)}) != 1:
        raise ValueError("hierarchical action stages must have equal dimensions")
    return {
        "exploration_l1": float(np.sum(np.abs(sampled - deterministic))),
        "submission_l1": float(np.sum(np.abs(submitted - sampled))),
        "effective_action_l1": float(np.sum(np.abs(effective - sampled))),
    }


__all__ = ["hierarchical_action_stage_metrics"]
''',
    )
    replace_once(
        "trade_rl/rl/policies.py",
        '''class HierarchicalActorOutputs:\n    """Gate, proposal, and composed target-weight outputs for one policy batch."""\n''',
        '''class HierarchicalActorOutputs:\n    """Change-intensity, proposal, and composed target-weight outputs."""\n''',
    )
    replace_once(
        "trade_rl/rl/policies.py",
        '''    active_mask: torch.Tensor\n\n\nclass SharedPerAssetGateTargetHead''',
        '''    active_mask: torch.Tensor\n\n    @property\n    def change_intensity(self) -> torch.Tensor:\n        """Continuous interpolation intensity; not a Bernoulli trade decision."""\n\n        return self.gate_probabilities\n\n\nclass SharedPerAssetGateTargetHead''',
    )
    replace_once(
        "trade_rl/rl/environment_info.py",
        '''            "executed_target": np.asarray(\n                request.executed_target,\n                dtype=np.float64,\n            ).copy(),\n''',
        '''            "executed_target": np.asarray(\n                request.executed_target,\n                dtype=np.float64,\n            ).copy(),\n            "sampled_policy_action": np.asarray(\n                request.submitted_target,\n                dtype=np.float64,\n            ).copy(),\n            "effective_filled_weights": np.asarray(\n                request.hybrid.weights,\n                dtype=np.float64,\n            ).copy(),\n            "sampled_policy_to_filled_l1": action_path.policy_to_filled_l1,\n''',
    )
    tensorboard = "trade_rl/rl/tensorboard_logging.py"
    replace_once(
        tensorboard,
        '''    "trade_rl/action_abs_max",\n)\n''',
        '''    "trade_rl/action_abs_max",\n    "trade_rl/change_intensity_mean",\n    "trade_rl/exploration_l1_mean",\n    "trade_rl/effective_action_l1_mean",\n)\n''',
    )
    replace_once(
        tensorboard,
        '''            infos = self.locals.get("infos", ())\n''',
        '''            if self.n_calls % log_interval == 0:\n                observations = self.locals.get("obs_tensor")\n                output_factory = getattr(\n                    getattr(self.model, "policy", None),\n                    "hierarchical_actor_outputs",\n                    None,\n                )\n                if observations is not None and callable(output_factory):\n                    import torch\n\n                    with torch.no_grad():\n                        outputs = output_factory(observations)\n                    intensity = outputs.change_intensity.detach().cpu().numpy()\n                    deterministic = outputs.composed_actions.detach().cpu().numpy()\n                    sampled_matrix = np.asarray(\n                        self.locals.get("actions", ()), dtype=np.float64\n                    )\n                    if sampled_matrix.shape == deterministic.shape:\n                        self._extend("trade_rl/change_intensity_mean", intensity)\n                        self._extend(\n                            "trade_rl/exploration_l1_mean",\n                            np.sum(\n                                np.abs(sampled_matrix - deterministic), axis=1\n                            ),\n                        )\n            infos = self.locals.get("infos", ())\n''',
    )
    replace_once(
        tensorboard,
        '''                    self._extend(\n                        "trade_rl/interval_cost_mean",\n                        info.get("interval_cost", ()),\n                    )\n''',
        '''                    self._extend(\n                        "trade_rl/interval_cost_mean",\n                        info.get("interval_cost", ()),\n                    )\n                    self._extend(\n                        "trade_rl/effective_action_l1_mean",\n                        info.get("sampled_policy_to_filled_l1", ()),\n                    )\n''',
    )


def patch_docs_and_quickstart() -> None:
    update_json(
        "examples/quickstart/training.json",
        lambda payload: payload.__setitem__(
            "reward",
            {
                **payload.get("reward", {}),
                "absolute_growth_weight": 1.0,
                "incremental_drawdown_weight": 0.05,
                "baseline_underperformance_weight": 0.10,
                "terminal_equity_weight": 1.0,
                "margin_deficit_weight": 1.0,
            },
        ),
    )
    configuration = ROOT / "docs/CONFIGURATION.md"
    text = configuration.read_text(encoding="utf-8")
    text = text.replace("# Training Configuration v2", "# Training Configuration v3")
    text = text.replace(
        "維持対象のTop-level Schemaは`training_run_config_v2`です。",
        "維持対象のTop-level Schemaは`training_run_config_v3`です。",
    )
    text = text.replace('"schema_version": "training_run_config_v2"', '"schema_version": "training_run_config_v3"')
    text = text.replace("`structured_policy_export_v1`", "`structured_policy_export_v2`")
    configuration.write_text(text, encoding="utf-8")

    architecture = ROOT / "docs/ARCHITECTURE.md"
    text = architecture.read_text(encoding="utf-8")
    text = text.replace(
        "`training_run_config_v2`は`observation_encoder`を1つだけ持ちます。",
        "`training_run_config_v3`は`observation_encoder`と階層Actor契約を明示します。",
    )
    text = text.replace("`structured_policy_export_v1`", "`structured_policy_export_v2`")
    architecture.write_text(text, encoding="utf-8")
    append_once(
        "docs/ARCHITECTURE.md",
        "## Maintained contract clarifications",
        r'''
## Maintained contract clarifications

因果データ契約は`data/contracts.py`だけに閉じていません。FeatureとInstrumentの宣言は`contracts.py`、Barと`available_at`は`RawMarketSeries`、市場配列は`MarketDataset`、`values`・`available`・`staleness`は`SequenceObservation`がそれぞれ保持し、frozen型、read-only配列、実行時検証を重ねて将来参照を拒否します。

7種類のconstraint costはscalar rewardから独立したCost Critic／Lagrangian用チャネルです。すべてがhard constraintという意味ではありません。Weight、gross、margin、liquidation、exchange rule等のhard safetyは環境とpre-trade riskが強制し、turnoverやexecution cost等はsoft budgetとして扱います。

`hierarchical_gate_target_v1`という互換名はCheckpoint identityのため維持しますが、sigmoid出力の意味はBernoulli Gateではなく連続的な**change intensity**です。TensorBoardはdeterministic composed actionと探索後のsampled policy actionのL1差を、Environment infoはsampled policy actionから約定後effective filled weightsまでの差を記録します。

Constrained PPOのPR C正本は修正版PR #193です。PR #191は置換前のDraft履歴であり、維持対象実装の根拠には使用しません。

Workflow securityはrunnerの任意の表示名をallowlistするのではなく、GitHub-hosted形式かprivileged runnerかを分類し、privileged runnerについてtrigger、owner、main、Environment、権限、immutable checkoutを検証します。この方針をrunner classificationと呼びます。

構造化配信の正本は`structured_policy_export_v2`と`serving_bundle_v5`です。秘密鍵ファイルのloaderは`offline_keys`、鍵生成と署名は`offline_signing`、承認署名は`offline_approval`等の明示的offline moduleへ限定します。Import Linterはruntime/trainingからこれらへの静的依存を禁止しますが、OS sandboxそのものを主張するものではありません。
''',
    )
    append_once(
        "README.md",
        "## 維持対象の契約バージョン",
        r'''
## 維持対象の契約バージョン

維持対象の学習設定は`training_run_config_v3`、構造化Policy exportは`structured_policy_export_v2`、Serving bundleは`serving_bundle_v5`です。QuickstartはPipeline確認用ですが、Reward dataclassの既定値変更で意味が静かに変わらないよう、hybrid reward値をJSONへ明示しています。

データの因果性はFeature契約だけでなく、Raw Barのavailability、MarketDatasetのeconomic arrays、SequenceObservationのstalenessまで多層で検証します。constraint costは報酬と分離されていますが、hard safetyとLagrangian soft budgetは同義ではありません。
''',
    )
    append_once(
        "docs/BINANCE.md",
        "binance_vision_raw_cache_v1",
        r'''
### Raw archive content evidence

Vision archive cacheはpayloadだけを信用しません。各`.bin`に`binance_vision_raw_cache_v1` sidecarを併置し、URL、取得時刻、byte数、SHA-256、ETag、Last-Modified、downloader identityを固定します。再利用時はbyte列を再hashし、sidecar欠落、size不一致、digest不一致をfail closedします。
''',
    )


def write_contract_tests() -> None:
    write(
        "tests/data/test_economic_semantics.py",
        r'''
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from trade_rl.data.contracts import InstrumentContract, InstrumentExecutionRule
from trade_rl.data.economic_semantics import build_market_economic_semantics


def test_economic_semantics_are_explicit_point_in_time_and_immutable() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    timestamps = np.arange(
        np.datetime64("2026-01-01T00:15:00", "ns"),
        np.datetime64("2026-01-01T01:15:00", "ns"),
        np.timedelta64(15, "m"),
    )
    contract = InstrumentContract(
        symbol="BTCUSDT",
        listed_at=start,
        tick_size=0.1,
        lot_size=0.001,
        minimum_notional=5.0,
        execution_rules=(
            InstrumentExecutionRule(effective_at=start, tick_size=0.1, lot_size=0.001, minimum_notional=5.0),
            InstrumentExecutionRule(effective_at=start + timedelta(minutes=45), tick_size=0.2, lot_size=0.002, minimum_notional=10.0),
        ),
    )
    shape = (len(timestamps), 1)
    semantics = build_market_economic_semantics(
        timestamps=timestamps,
        instruments=(contract,),
        row_present=np.ones(shape, dtype=np.bool_),
        raw_tradable=np.ones(shape, dtype=np.bool_),
        source_information_available=np.ones(shape, dtype=np.bool_),
        available_at=np.broadcast_to(timestamps[:, None], shape),
        close=np.full(shape, 100.0),
        funding_event_count=np.zeros(shape, dtype=np.int32),
    )
    assert semantics.tick_size[:, 0].tolist() == [0.1, 0.1, 0.2, 0.2]
    assert semantics.minimum_notional[:, 0].tolist() == [5.0, 5.0, 10.0, 10.0]
    assert set(semantics.market_dataset_kwargs()) >= {"fee_rate", "spread_rate", "borrow_rate", "mark_price", "index_price"}
    assert all(not value.flags.writeable for value in semantics.market_dataset_kwargs().values())


def test_vision_and_postgres_use_the_same_constructor() -> None:
    root = Path(__file__).resolve().parents[2]
    assert "build_market_economic_semantics" in (root / "trade_rl/data/builder.py").read_text()
    assert "build_market_economic_semantics" in (root / "trade_rl/integrations/postgres_market_dataset.py").read_text()
''',
    )
    write(
        "tests/learning/test_oracle_bc_causal_gate_contract.py",
        r'''
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from trade_rl.learning.evaluation import deterministic_bootstrap_upper_bound


def test_bootstrap_upper_bound_is_deterministic_and_one_sided() -> None:
    values = np.array([0.01, 0.03, 0.02, 0.08, 0.04])
    first = deterministic_bootstrap_upper_bound(values, confidence_level=0.95, resamples=2_000, seed_material="a" * 64)
    second = deterministic_bootstrap_upper_bound(values, confidence_level=0.95, resamples=2_000, seed_material="a" * 64)
    assert first == second
    assert first >= float(np.mean(values))


def test_maintained_profiles_require_nontrivial_causal_evidence() -> None:
    root = Path(__file__).resolve().parents[2] / "examples/binance-multitimeframe"
    for name in (
        "training-target-weight-growth-ppo.json",
        "training-target-weight-constrained-growth.json",
        "training-target-weight-constrained-growth-discounted.json",
    ):
        training = json.loads((root / name).read_text())["training"]
        assert training["behavior_cloning_required_relative_improvement"] > 0.0
        assert training["behavior_cloning_min_causal_holdout_trades"] >= 30
        assert training["behavior_cloning_causal_holdout_bootstrap_resamples"] >= 2_000
        assert training["behavior_cloning_causal_holdout_confidence_level"] >= 0.95
''',
    )
    write(
        "tests/examples/test_dataset_publication_order_contract.py",
        r'''
from __future__ import annotations

import ast
from pathlib import Path


def test_dataset_validation_precedes_publication() -> None:
    source = (Path(__file__).resolve().parents[2] / "examples/binance-multitimeframe/full_research_pipeline.py").read_text()
    module = ast.parse(source)
    function = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "_build_dataset")
    ordered = []
    for statement in function.body:
        for node in ast.walk(statement):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                ordered.append(node.func.id)
    assert ordered.index("validate_maintained_dataset_preset") < ordered.index("publish_market_dataset_artifact")
''',
    )
    write(
        "tests/integrations/test_binance_cache_integrity_contract.py",
        r'''
from __future__ import annotations

import json
import urllib.request

import pytest

from trade_rl.integrations.binance import BinancePublicTransport, BinanceTransportError


class _Response:
    headers = {"ETag": '"fixture"', "Last-Modified": "Wed, 29 Jul 2026 00:00:00 GMT"}

    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


def test_vision_cache_sidecar_detects_tampering(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"official-archive"
    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout: _Response(payload))
    transport = BinancePublicTransport(cache_root=tmp_path, max_attempts=1)
    url = "https://data.binance.vision/data/futures/um/daily/klines/BTCUSDT/15m/file.zip"
    assert transport._request_bytes(url) == payload
    binary = next(tmp_path.rglob("*.bin"))
    evidence = json.loads(binary.with_suffix(".json").read_text())
    assert evidence["schema_version"] == "binance_vision_raw_cache_v1"
    assert evidence["size_bytes"] == len(payload)
    binary.write_bytes(b"tampered")
    with pytest.raises(BinanceTransportError, match="digest|size"):
        transport._request_bytes(url)
''',
    )
    write(
        "tests/rl/test_change_intensity_contract.py",
        r'''
from __future__ import annotations

import numpy as np
import torch

from trade_rl.rl.action_telemetry import hierarchical_action_stage_metrics
from trade_rl.rl.policies import HierarchicalActorOutputs


def test_hierarchical_gate_is_exposed_as_change_intensity() -> None:
    intensity = torch.tensor([[0.25, 0.75]])
    outputs = HierarchicalActorOutputs(torch.zeros_like(intensity), intensity, torch.ones_like(intensity), torch.full_like(intensity, 0.5), torch.zeros_like(intensity), torch.zeros_like(intensity), torch.ones_like(intensity, dtype=torch.bool))
    assert outputs.change_intensity is outputs.gate_probabilities


def test_action_stage_metrics_measure_exploration_and_effective_action() -> None:
    metrics = hierarchical_action_stage_metrics(
        deterministic_composed=np.array([0.1, 0.2]),
        sampled_policy_action=np.array([0.4, 0.0]),
        submitted_target=np.array([0.3, 0.0]),
        effective_filled_weights=np.array([0.25, 0.05]),
    )
    assert metrics == {"exploration_l1": 0.5, "submission_l1": 0.1, "effective_action_l1": 0.2}
''',
    )
    write(
        "tests/test_maintained_documentation_v3_contract.py",
        r'''
from __future__ import annotations

import json
from pathlib import Path


def test_docs_match_maintained_schemas_and_boundaries() -> None:
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text()
    architecture = (root / "docs/ARCHITECTURE.md").read_text()
    configuration = (root / "docs/CONFIGURATION.md").read_text()
    binance = (root / "docs/BINANCE.md").read_text()
    assert "training_run_config_v3" in readme
    assert "training_run_config_v3" in architecture
    assert "structured_policy_export_v2" in architecture
    assert "# Training Configuration v3" in configuration
    assert '"schema_version": "training_run_config_v3"' in configuration
    assert "structured_policy_export_v2" in configuration
    assert "change intensity" in architecture.lower()
    assert "constraint cost" in architecture.lower()
    assert "runner classification" in architecture.lower()
    assert "offline_signing" in architecture
    assert "PR #193" in architecture
    assert "binance_vision_raw_cache_v1" in binance


def test_quickstart_pins_hybrid_reward() -> None:
    root = Path(__file__).resolve().parents[1]
    reward = json.loads((root / "examples/quickstart/training.json").read_text())["reward"]
    assert reward["absolute_growth_weight"] == 1.0
    assert reward["incremental_drawdown_weight"] == 0.05
    assert reward["baseline_underperformance_weight"] == 0.10
    assert reward["terminal_equity_weight"] == 1.0
    assert reward["margin_deficit_weight"] == 1.0
''',
    )


def main() -> None:
    create_economic_semantics()
    patch_dataset_builders()
    patch_pipeline_order()
    patch_binance_cache()
    patch_bc_gate()
    patch_change_intensity()
    patch_docs_and_quickstart()
    write_contract_tests()


if __name__ == "__main__":
    main()
