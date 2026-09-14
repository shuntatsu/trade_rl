from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise RuntimeError(f"{path}: expected exactly one patch site")
    source.write_text(text.replace(old, new, 1), encoding="utf-8")


# source.py: add immutable sparse index-price contract + explicit runtime capability.
replace_once(
    "trade_rl/data/source.py",
    "\n\nclass MarketDataSource(Protocol):\n    def load(self, symbol: str) -> RawMarketSeries: ...\n",
    '''\n\n@dataclass(frozen=True, slots=True)\nclass RawIndexPriceSeries:\n    """Sparse native completed-bar index closes with explicit availability."""\n\n    timestamps: np.ndarray\n    close: np.ndarray\n    available_at: np.ndarray | None = None\n\n    def __post_init__(self) -> None:\n        timestamps = _readonly(self.timestamps)\n        if timestamps.ndim != 1 or not np.issubdtype(timestamps.dtype, np.datetime64):\n            raise ValueError(\n                "index-price timestamps must be a one-dimensional datetime64 array"\n            )\n        timestamp_ns = timestamps.astype("datetime64[ns]").astype(np.int64)\n        if timestamp_ns.size == 0:\n            raise ValueError("raw index-price series must not be empty")\n        if np.any(timestamp_ns == np.iinfo(np.int64).min):\n            raise ValueError("index-price timestamps must not contain NaT")\n        if np.any(np.diff(timestamp_ns) <= 0):\n            raise ValueError(\n                "index-price timestamps must be strictly increasing and unique"\n            )\n\n        close = _readonly(self.close, dtype=np.dtype(np.float64))\n        if close.shape != timestamps.shape:\n            raise ValueError("index-price close shape must match timestamps")\n        if not np.isfinite(close).all() or np.any(close <= 0.0):\n            raise ValueError("index-price close must be finite and strictly positive")\n\n        available_at_value = (\n            timestamps if self.available_at is None else self.available_at\n        )\n        available_at = _readonly(available_at_value)\n        if available_at.ndim != 1 or not np.issubdtype(\n            available_at.dtype, np.datetime64\n        ):\n            raise ValueError(\n                "index-price available_at must be a one-dimensional datetime64 array"\n            )\n        if available_at.shape != timestamps.shape:\n            raise ValueError("index-price available_at shape must match timestamps")\n        available_ns = available_at.astype("datetime64[ns]").astype(np.int64)\n        if np.any(available_ns == np.iinfo(np.int64).min):\n            raise ValueError("index-price available_at must not contain NaT")\n        if np.any(available_ns < timestamp_ns):\n            raise ValueError(\n                "index-price available_at cannot be earlier than the event timestamp"\n            )\n\n        object.__setattr__(self, "timestamps", timestamps.astype("datetime64[ns]"))\n        object.__setattr__(self, "close", close)\n        object.__setattr__(\n            self, "available_at", available_at.astype("datetime64[ns]")\n        )\n\n\nclass MarketDataSource(Protocol):\n    def load(self, symbol: str) -> RawMarketSeries: ...\n\n\n@runtime_checkable\nclass IndexPriceMarketDataSource(MarketDataSource, Protocol):\n    """Source with an explicit sparse native index-price history capability."""\n\n    def load_index_price(\n        self, symbol: str, timeframe: str\n    ) -> RawIndexPriceSeries: ...\n\n    @property\n    def index_price_provenance(self) -> Mapping[str, object]: ...\n''',
)

# contracts.py: add one exact feature kind and fail closed on semantic drift.
replace_once(
    "trade_rl/data/contracts.py",
    '    CROSS_ASSET_DISPERSION = "cross_asset_dispersion"\n',
    '    CROSS_ASSET_DISPERSION = "cross_asset_dispersion"\n    PERP_INDEX_LOG_BASIS_BPS = "perp_index_log_basis_bps"\n',
)
replace_once(
    "trade_rl/data/contracts.py",
    '''        names = tuple(spec.name for spec in self.features)\n        if len(set(names)) != len(names):\n            raise ValueError("feature names must be unique")\n        if any(spec.timeframe == self.base_timeframe for spec in self.features):\n''',
    '''        names = tuple(spec.name for spec in self.features)\n        if len(set(names)) != len(names):\n            raise ValueError("feature names must be unique")\n        basis_specs = tuple(\n            spec\n            for spec in self.features\n            if spec.kind is FeatureKind.PERP_INDEX_LOG_BASIS_BPS\n        )\n        if basis_specs:\n            if self.base_timeframe != "1h":\n                raise ValueError(\n                    "perp-index basis requires the 1h base decision clock"\n                )\n            for spec in basis_specs:\n                if spec.name != "1h__perp_index_log_basis_bps":\n                    raise ValueError("perp-index basis feature name is not canonical")\n                if spec.timeframe is not None:\n                    raise ValueError(\n                        "perp-index basis must use the native 1h base timeframe"\n                    )\n                if spec.lookback != 1:\n                    raise ValueError("perp-index basis lookback must equal 1")\n                if spec.normalization is not NormalizationMode.NONE:\n                    raise ValueError("perp-index basis normalization is forbidden")\n                if spec.normalization_window != 1:\n                    raise ValueError(\n                        "perp-index basis normalization_window must equal 1"\n                    )\n                if spec.min_periods != 1:\n                    raise ValueError("perp-index basis min_periods must equal 1")\n                if spec.alignment is not None:\n                    raise ValueError(\n                        "perp-index basis alignment must remain on decision time"\n                    )\n        if any(spec.timeframe == self.base_timeframe for spec in self.features):\n''',
)

# builder.py imports.
replace_once(
    "trade_rl/data/build/builder.py",
    '''from trade_rl.data.contracts import (\n    InstrumentContract,\n    MarketBuildConfig,\n    MarketCalendarKind,\n)\n''',
    '''from trade_rl.data.contracts import (\n    FeatureKind,\n    InstrumentContract,\n    MarketBuildConfig,\n    MarketCalendarKind,\n)\n''',
)
replace_once(
    "trade_rl/data/build/builder.py",
    '''from trade_rl.data.source import (\n    MarketDataSource,\n    MultiTimeframeMarketDataSource,\n    RawMarketSeries,\n)\n''',
    '''from trade_rl.data.source import (\n    IndexPriceMarketDataSource,\n    MarketDataSource,\n    MultiTimeframeMarketDataSource,\n    RawIndexPriceSeries,\n    RawMarketSeries,\n)\n''',
)

# builder.py helper for strict sparse exact-clock alignment.
replace_once(
    "trade_rl/data/build/builder.py",
    '''def _carry_feature(\n    event_values: np.ndarray,\n''',
    '''def _align_index_price_series(\n    raw: RawIndexPriceSeries,\n    timestamps: np.ndarray,\n) -> tuple[np.ndarray, np.ndarray, np.ndarray]:\n    """Align only exact sparse completed timestamps; never carry or synthesize."""\n\n    base_ns = timestamps.astype("datetime64[ns]").astype(np.int64)\n    event_ns = raw.timestamps.astype("datetime64[ns]").astype(np.int64)\n    positions = np.searchsorted(base_ns, event_ns)\n    if np.any(positions >= len(base_ns)) or np.any(base_ns[positions] != event_ns):\n        raise ValueError(\n            "index-price timestamps must match the base 1h clock exactly"\n        )\n    close = np.zeros(len(timestamps), dtype=np.float64)\n    present = np.zeros(len(timestamps), dtype=np.bool_)\n    information_available = np.zeros(len(timestamps), dtype=np.bool_)\n    close[positions] = raw.close\n    present[positions] = True\n    assert raw.available_at is not None\n    available_ns = raw.available_at.astype("datetime64[ns]").astype(np.int64)\n    information_available[positions] = available_ns <= event_ns\n    return close, present, information_available\n\n\ndef _carry_feature(\n    event_values: np.ndarray,\n''',
)

# builder.py bind explicit capability only if basis feature is actually requested.
replace_once(
    "trade_rl/data/build/builder.py",
    '''        symbols = tuple(contract.symbol for contract in instruments)\n        if len(set(symbols)) != len(symbols):\n            raise ValueError("instrument symbols must be unique")\n        raw_series = tuple(source.load(symbol) for symbol in symbols)\n''',
    '''        symbols = tuple(contract.symbol for contract in instruments)\n        if len(set(symbols)) != len(symbols):\n            raise ValueError("instrument symbols must be unique")\n        basis_requested = any(\n            spec.kind is FeatureKind.PERP_INDEX_LOG_BASIS_BPS\n            for spec in self.config.features\n        )\n        if basis_requested and not isinstance(source, IndexPriceMarketDataSource):\n            raise ValueError(\n                "perp-index basis requires an explicit index-price source capability"\n            )\n        raw_series = tuple(source.load(symbol) for symbol in symbols)\n''',
)

# builder.py add cache and special no-carry basis path before generic feature calculation.
replace_once(
    "trade_rl/data/build/builder.py",
    '''        native_cache: dict[tuple[str, str], RawMarketSeries] = {}\n        for symbol_index, contract in enumerate(instruments):\n            for feature_index, spec in enumerate(self.config.features):\n                if spec.kind in CROSS_ASSET_FEATURE_KINDS:\n                    continue\n                native_timeframe = spec.resolved_timeframe(self.config.base_timeframe)\n                if native_timeframe == self.config.base_timeframe:\n''',
    '''        native_cache: dict[tuple[str, str], RawMarketSeries] = {}\n        index_cache: dict[str, RawIndexPriceSeries] = {}\n        for symbol_index, contract in enumerate(instruments):\n            for feature_index, spec in enumerate(self.config.features):\n                if spec.kind in CROSS_ASSET_FEATURE_KINDS:\n                    continue\n                native_timeframe = spec.resolved_timeframe(self.config.base_timeframe)\n                if spec.kind is FeatureKind.PERP_INDEX_LOG_BASIS_BPS:\n                    assert isinstance(source, IndexPriceMarketDataSource)\n                    index_raw = index_cache.get(contract.symbol)\n                    if index_raw is None:\n                        index_raw = source.load_index_price(contract.symbol, "1h")\n                        index_cache[contract.symbol] = index_raw\n                    index_close, index_present, index_information = (\n                        _align_index_price_series(index_raw, timestamps)\n                    )\n                    valid = (\n                        causal_row_present[:, symbol_index]\n                        & symbol_active[:, symbol_index]\n                        & tradable[:, symbol_index]\n                        & index_present\n                        & index_information\n                        & np.isfinite(close[:, symbol_index])\n                        & (close[:, symbol_index] > 0.0)\n                    )\n                    ratios = np.ones(n_bars, dtype=np.float64)\n                    np.divide(\n                        close[:, symbol_index],\n                        index_close,\n                        out=ratios,\n                        where=valid,\n                    )\n                    values = np.zeros(n_bars, dtype=np.float64)\n                    values[valid] = 10_000.0 * portable_log(ratios[valid])\n                    available = valid\n                    age_hours = np.full(\n                        n_bars, spec.max_staleness_hours, dtype=np.float64\n                    )\n                    staleness = np.ones(n_bars, dtype=np.float64)\n                    age_hours[valid] = 0.0\n                    staleness[valid] = 0.0\n                elif native_timeframe == self.config.base_timeframe:\n''',
)

# builder.py bind source provenance only when basis requested, preserving legacy identity.
replace_once(
    "trade_rl/data/build/builder.py",
    '''        if identity_provenance is not None:\n            metadata["metadata_evidence"] = identity_provenance\n        if execution_economics is not None:\n''',
    '''        if identity_provenance is not None:\n            metadata["metadata_evidence"] = identity_provenance\n        if basis_requested:\n            assert isinstance(source, IndexPriceMarketDataSource)\n            metadata["index_price_source_evidence"] = dict(\n                source.index_price_provenance\n            )\n        if execution_economics is not None:\n''',
)
