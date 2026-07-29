from __future__ import annotations

from pathlib import Path

path = Path(__file__).with_name("apply_research_contract_hardening.py")
content = path.read_text(encoding="utf-8")
start = content.index("def patch_pipeline_order() -> None:\n")
end = content.index("\ndef patch_binance_cache() -> None:\n", start)
replacement = r"""def patch_pipeline_order() -> None:
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
"""
path.write_text(content[:start] + replacement + content[end:], encoding="utf-8")
