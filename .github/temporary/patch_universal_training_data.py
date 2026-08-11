from pathlib import Path

path = Path("trade_rl/workflows/universal_training.py")
text = path.read_text()
if "def materialize_universal_train_datasets(" in text:
    raise SystemExit(0)
text = text.replace(
    "import numpy as np\n",
    "import numpy as np\n\n"
    "from trade_rl.data.contracts import FeatureSpec\n"
    "from trade_rl.integrations.postgres_indicator_artifacts import "
    "IndicatorArtifactConnection\n",
    1,
)
marker = "\n\n__all__ = [\n"
if marker not in text:
    raise SystemExit("universal_training __all__ marker not found")
block = r'''


def _universal_feature_schema_digest(feature_names: Sequence[str]) -> str:
    names = tuple(str(value) for value in feature_names)
    if not names or len(set(names)) != len(names) or any(not value for value in names):
        raise ValueError("Universal feature names must be non-empty and unique")
    return content_digest(
        {
            "feature_names": names,
            "profile": "binance_universal_target_local_v1",
            "schema_version": "universal_feature_schema_v1",
        }
    )


def materialize_universal_train_datasets(
    connection: IndicatorArtifactConnection,
    *,
    instrument_bundle: Any,
    metadata_resolution: Any,
    feature_specs: Sequence[FeatureSpec],
    indicator_loader: Any | None = None,
    dataset_builder: Any | None = None,
) -> dict[str, Any]:
    """Materialize only the immutable train-symbol partition from PostgreSQL."""

    from trade_rl.integrations.postgres_indicator_artifacts import (
        load_postgres_indicator_artifacts,
    )
    from trade_rl.integrations.postgres_market_dataset import (
        NATIVE_TIMEFRAMES,
        build_postgres_market_dataset,
    )

    partition = getattr(instrument_bundle, "partition", None)
    catalog = getattr(instrument_bundle, "catalog", None)
    train_symbols = tuple(getattr(partition, "train_symbols", ()))
    if not train_symbols or len(set(train_symbols)) != len(train_symbols):
        raise ValueError("Universal instrument bundle has invalid train_symbols")
    if any(not isinstance(symbol, str) or not symbol for symbol in train_symbols):
        raise ValueError("Universal train symbols must be non-empty strings")
    metadata = getattr(metadata_resolution, "metadata", None)
    if not isinstance(metadata, Mapping):
        raise TypeError("Universal metadata resolution must expose a mapping")
    missing_metadata = set(train_symbols) - set(metadata)
    if missing_metadata:
        raise ValueError(
            f"Universal metadata is missing train symbols: {sorted(missing_metadata)}"
        )
    evidence_digest = getattr(metadata_resolution, "evidence_digest", None)
    if not isinstance(evidence_digest, str):
        raise ValueError("Universal metadata evidence digest is unavailable")
    require_sha256(evidence_digest, field="Universal metadata evidence_digest")
    start_time = getattr(catalog, "research_start", None)
    end_time = getattr(catalog, "research_end", None)
    if start_time is None or end_time is None:
        raise ValueError("Universal catalog research interval is unavailable")

    resolved_indicator_loader = indicator_loader or load_postgres_indicator_artifacts
    resolved_dataset_builder = dataset_builder or build_postgres_market_dataset
    indicator_bundle = resolved_indicator_loader(
        connection,
        symbols=train_symbols,
        timeframes=NATIVE_TIMEFRAMES,
    )
    raw_bundle_symbols = tuple(getattr(indicator_bundle, "symbols", train_symbols))
    if raw_bundle_symbols != train_symbols:
        raise ValueError("Universal indicator bundle order does not match train_symbols")

    raw_histories = getattr(metadata_resolution, "execution_rule_histories", None)
    datasets: dict[str, Any] = {}
    reference_feature_names: tuple[str, ...] | None = None
    reference_timestamps: np.ndarray | None = None
    for symbol in train_symbols:
        histories = None
        if raw_histories is not None:
            if symbol not in raw_histories:
                raise ValueError(
                    f"Universal execution-rule history is missing train symbol {symbol}"
                )
            histories = {symbol: tuple(raw_histories[symbol])}
        dataset = resolved_dataset_builder(
            connection,
            symbols=(symbol,),
            symbol_vocabulary=train_symbols,
            start_time=start_time,
            end_time=end_time,
            metadata={symbol: metadata[symbol]},
            metadata_evidence_digest=evidence_digest,
            execution_rule_histories=histories,
            indicator_bundle=indicator_bundle.subset((symbol,)),
            feature_specs=tuple(feature_specs),
        )
        if tuple(getattr(dataset, "symbols", ())) != (symbol,):
            raise ValueError("Universal materialized dataset symbol mismatch")
        feature_names = tuple(getattr(dataset, "feature_names", ()))
        timestamps = np.asarray(getattr(dataset, "timestamps", None))
        if reference_feature_names is None:
            reference_feature_names = feature_names
            reference_timestamps = timestamps.copy(order="C")
        else:
            if feature_names != reference_feature_names:
                raise ValueError("Universal train dataset feature order mismatch")
            assert reference_timestamps is not None
            if not np.array_equal(timestamps, reference_timestamps):
                raise ValueError("Universal train dataset timestamp grid mismatch")
        datasets[symbol] = dataset
    return validate_universal_dataset_scope(datasets, train_symbols=train_symbols)


def fit_universal_shared_normalizer(
    datasets: Mapping[str, Any],
    *,
    train_symbols: Sequence[str],
    catalog_digest: str,
    split_manifest_digest: str,
    fold_train_range: tuple[int, int],
    max_samples_per_symbol: int = 100_000,
) -> SymbolBalancedStandardNormalizer:
    """Fit equal-symbol statistics on the explicit train-only fold range."""

    ordered = validate_universal_dataset_scope(
        datasets,
        train_symbols=train_symbols,
    )
    require_sha256(catalog_digest, field="Universal catalog_digest")
    require_sha256(split_manifest_digest, field="Universal split_manifest_digest")
    start, stop = fold_train_range
    if (
        isinstance(start, bool)
        or isinstance(stop, bool)
        or not isinstance(start, int)
        or not isinstance(stop, int)
        or start < 0
        or stop <= start
    ):
        raise ValueError("Universal fold_train_range is invalid")
    first = ordered[tuple(train_symbols)[0]]
    feature_names = tuple(getattr(first, "feature_names", ()))
    feature_schema_digest = _universal_feature_schema_digest(feature_names)
    symbol_features: dict[str, np.ndarray] = {}
    symbol_available: dict[str, np.ndarray] = {}
    for symbol in tuple(train_symbols):
        dataset = ordered[symbol]
        if tuple(getattr(dataset, "feature_names", ())) != feature_names:
            raise ValueError("Universal train dataset feature order mismatch")
        n_bars = int(getattr(dataset, "n_bars", 0))
        if stop > n_bars:
            raise ValueError("Universal fold_train_range exceeds a train dataset")
        values = np.asarray(getattr(dataset, "features", None), dtype=np.float64)
        available = np.asarray(getattr(dataset, "feature_available", None), dtype=np.bool_)
        if values.ndim != 3 or values.shape[1] != 1:
            raise ValueError("Universal train features must be single-symbol tensors")
        if available.shape != values.shape:
            raise ValueError("Universal train feature availability shape mismatch")
        symbol_features[symbol] = values[:, 0, :]
        symbol_available[symbol] = available[:, 0, :]
    return SymbolBalancedStandardNormalizer.fit(
        symbol_features,
        train_symbols=tuple(train_symbols),
        feature_schema_digest=feature_schema_digest,
        catalog_digest=catalog_digest,
        split_manifest_digest=split_manifest_digest,
        fold_train_range=fold_train_range,
        max_samples_per_symbol=max_samples_per_symbol,
        symbol_available=symbol_available,
    )
'''
text = text.replace(marker, block + marker, 1)
text = text.replace(
    '    "collect_universal_episode_teacher",\n',
    '    "collect_universal_episode_teacher",\n'
    '    "fit_universal_shared_normalizer",\n'
    '    "materialize_universal_train_datasets",\n',
    1,
)
path.write_text(text)
compile(text, str(path), "exec")
