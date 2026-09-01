from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tests.rl.universal_trade_test_support import make_u1_feature_specs, make_u1_market
from trade_rl.data import write_market_dataset_files
from trade_rl.data.market import MarketDataset
from trade_rl.domain.universal_trade_rl_universe import UniversalTradeRLUniverseConfig
from trade_rl.rl.universal_normalization import UniversalTradePublishedSource
from trade_rl.rl.universal_trade_contract import UniversalTradePolicyContract
from trade_rl.workflows.universal_trade_rl_normalization import (
    fit_universal_trade_sequence_normalizer,
)
from trade_rl.workflows.universal_trade_rl_universe_access import (
    UniversalTradeRLAccessPhase,
    UniversalTradeRLUniverseAccess,
)
from trade_rl.workflows.universal_trade_rl_universe_config import (
    UniversalTradeRLSymbolSource,
)
from trade_rl.workflows.universal_trade_rl_universe_manifest import (
    UniversalTradeRLUniverseManifest,
    build_universal_trade_rl_universe_manifest,
)

_NS_PER_HOUR = 3_600_000_000_000


def _source(symbol: str, digest_char: str) -> UniversalTradeRLSymbolSource:
    return UniversalTradeRLSymbolSource(
        symbol=symbol,
        dataset_digest=digest_char * 64,
        first_timestamp_ns=1,
        last_timestamp_ns=2,
        row_count=2,
    )


def _manifest(
    train: tuple[UniversalTradeRLSymbolSource, ...],
    *,
    admission_digest_char: str = "a",
) -> UniversalTradeRLUniverseManifest:
    by_symbol = {item.symbol: item for item in train}
    by_symbol["LINKUSDT"] = _source("LINKUSDT", "c")
    by_symbol["AVAXUSDT"] = _source("AVAXUSDT", admission_digest_char)
    return build_universal_trade_rl_universe_manifest(
        config=UniversalTradeRLUniverseConfig(
            train_symbols=("BTCUSDT", "ETHUSDT"),
            development_symbols=("LINKUSDT",),
            admission_symbols=("AVAXUSDT",),
        ),
        sources=tuple(by_symbol[symbol] for symbol in sorted(by_symbol)),
    )


def _published_source(
    root: Path,
    dataset: MarketDataset,
) -> tuple[UniversalTradeRLSymbolSource, UniversalTradePublishedSource]:
    files = write_market_dataset_files(root, dataset)
    timestamp_ns = dataset.timestamps.astype("datetime64[ns]").astype(np.int64)
    return (
        UniversalTradeRLSymbolSource(
            symbol=dataset.symbols[0],
            dataset_digest=files.artifact_digest,
            first_timestamp_ns=int(timestamp_ns[0]),
            last_timestamp_ns=int(timestamp_ns[-1]),
            row_count=dataset.n_bars,
        ),
        UniversalTradePublishedSource(dataset.symbols[0], root),
    )


def _contract() -> UniversalTradePolicyContract:
    return UniversalTradePolicyContract(feature_specs=make_u1_feature_specs())


def _train_access(
    manifest: UniversalTradeRLUniverseManifest,
) -> UniversalTradeRLUniverseAccess:
    return UniversalTradeRLUniverseAccess.for_phase(
        manifest=manifest,
        phase=UniversalTradeRLAccessPhase.TRAIN,
    )


def _feature_moments(
    dataset: MarketDataset,
    *,
    feature_index: int,
    knowledge_cutoff_ns: int,
) -> tuple[float, float, int]:
    timestamps_ns = dataset.timestamps.astype("datetime64[ns]").astype(np.int64)
    values = np.asarray(dataset.features[:, 0, feature_index], dtype=np.float64)
    available = np.asarray(
        dataset.resolved_array("feature_available")[:, 0, feature_index],
        dtype=np.bool_,
    )
    staleness = np.asarray(
        dataset.resolved_array("feature_staleness_hours")[:, 0, feature_index],
        dtype=np.float64,
    )
    event_ns = timestamps_ns - np.rint(staleness * _NS_PER_HOUR).astype(np.int64)
    valid = available & np.isfinite(values) & np.isfinite(staleness)
    valid &= event_ns <= knowledge_cutoff_ns
    valid_values = values[valid]
    valid_events = event_ns[valid]
    _, unique_indices = np.unique(valid_events, return_index=True)
    samples = valid_values[unique_indices]
    assert samples.size > 0
    return float(samples.mean()), float(np.mean(samples * samples)), int(samples.size)


def test_published_artifact_digest_must_match_u0_manifest(tmp_path: Path) -> None:
    btc_a_source, _ = _published_source(
        tmp_path / "btc-a",
        make_u1_market(symbol="BTCUSDT", n_bars=5800, feature_level=1.0),
    )
    _, btc_b = _published_source(
        tmp_path / "btc-b",
        make_u1_market(symbol="BTCUSDT", n_bars=5800, feature_level=2.0),
    )
    eth_source, eth = _published_source(
        tmp_path / "eth",
        make_u1_market(symbol="ETHUSDT", n_bars=5800, feature_level=3.0),
    )
    manifest = _manifest((btc_a_source, eth_source))

    with pytest.raises(ValueError, match="digest|identity"):
        fit_universal_trade_sequence_normalizer(
            manifest=manifest,
            access=_train_access(manifest),
            sources=(btc_b, eth),
            contract=_contract(),
            knowledge_cutoff_ns=btc_a_source.last_timestamp_ns,
        )


def test_equal_symbol_statistics_ignore_unavailable_extreme(tmp_path: Path) -> None:
    btc = make_u1_market(symbol="BTCUSDT", n_bars=5800, feature_level=1.0)
    features = btc.features.copy()
    available = btc.resolved_array("feature_available").copy()
    features[100, 0, 0] = 1e9
    available[100, 0, 0] = False
    btc = replace(
        btc, features=features, feature_available=available
    ).with_content_identity({"fixture": "u1_unavailable_extreme_v1"})
    eth = make_u1_market(symbol="ETHUSDT", n_bars=6000, feature_level=10.0)
    btc_source, btc_published = _published_source(tmp_path / "btc", btc)
    eth_source, eth_published = _published_source(tmp_path / "eth", eth)
    manifest = _manifest((btc_source, eth_source))
    cutoff = min(btc_source.last_timestamp_ns, eth_source.last_timestamp_ns)

    normalizer = fit_universal_trade_sequence_normalizer(
        manifest=manifest,
        access=_train_access(manifest),
        sources=(btc_published, eth_published),
        contract=_contract(),
        knowledge_cutoff_ns=cutoff,
    )
    stats = normalizer.statistics_for("15m")
    mu_btc, q_btc, _ = _feature_moments(
        btc, feature_index=0, knowledge_cutoff_ns=cutoff
    )
    mu_eth, q_eth, _ = _feature_moments(
        eth, feature_index=0, knowledge_cutoff_ns=cutoff
    )
    expected_mean = 0.5 * (mu_btc + mu_eth)
    expected_q = 0.5 * (q_btc + q_eth)
    expected_scale = math.sqrt(max(expected_q - expected_mean * expected_mean, 0.0))
    assert stats.mean[0] == pytest.approx(expected_mean)
    assert stats.scale[0] == pytest.approx(expected_scale)
    row_weighted = (mu_btc * btc.n_bars + mu_eth * eth.n_bars) / (
        btc.n_bars + eth.n_bars
    )
    assert abs(expected_mean - row_weighted) > 1e-4


def test_carried_native_timeframe_events_are_deduplicated(tmp_path: Path) -> None:
    btc = make_u1_market(symbol="BTCUSDT", n_bars=5800, feature_level=1.0)
    eth = make_u1_market(symbol="ETHUSDT", n_bars=5800, feature_level=2.0)
    btc_source, btc_published = _published_source(tmp_path / "btc", btc)
    eth_source, eth_published = _published_source(tmp_path / "eth", eth)
    manifest = _manifest((btc_source, eth_source))

    normalizer = fit_universal_trade_sequence_normalizer(
        manifest=manifest,
        access=_train_access(manifest),
        sources=(btc_published, eth_published),
        contract=_contract(),
        knowledge_cutoff_ns=btc_source.last_timestamp_ns,
    )
    expected = (btc.n_bars - 1) // 16 + 1
    counts = dict(normalizer.statistics_for("4h").per_symbol_sample_counts)
    assert counts["BTCUSDT"][0] == expected
    assert expected < btc.n_bars


def test_knowledge_cutoff_excludes_future_source_events(tmp_path: Path) -> None:
    btc = make_u1_market(symbol="BTCUSDT", n_bars=5800, feature_level=1.0)
    features = btc.features.copy()
    features[5001:, 0, 0] = 1e9
    btc = replace(btc, features=features).with_content_identity(
        {"fixture": "u1_cutoff_future_mutation_v1"}
    )
    eth = make_u1_market(symbol="ETHUSDT", n_bars=5800, feature_level=2.0)
    btc_source, btc_published = _published_source(tmp_path / "btc", btc)
    eth_source, eth_published = _published_source(tmp_path / "eth", eth)
    manifest = _manifest((btc_source, eth_source))
    cutoff = int(btc.timestamps[5000].astype("datetime64[ns]").astype(np.int64))

    normalizer = fit_universal_trade_sequence_normalizer(
        manifest=manifest,
        access=_train_access(manifest),
        sources=(btc_published, eth_published),
        contract=_contract(),
        knowledge_cutoff_ns=cutoff,
    )
    stats = normalizer.statistics_for("15m")
    mu_btc, q_btc, _ = _feature_moments(
        btc, feature_index=0, knowledge_cutoff_ns=cutoff
    )
    mu_eth, q_eth, _ = _feature_moments(
        eth, feature_index=0, knowledge_cutoff_ns=cutoff
    )
    expected_mean = 0.5 * (mu_btc + mu_eth)
    expected_q = 0.5 * (q_btc + q_eth)
    assert stats.mean[0] == pytest.approx(expected_mean)
    assert stats.scale[0] == pytest.approx(
        math.sqrt(max(expected_q - expected_mean * expected_mean, 0.0))
    )


def test_generation_changes_identity_but_not_train_statistics(tmp_path: Path) -> None:
    btc = make_u1_market(symbol="BTCUSDT", n_bars=5800, feature_level=1.0)
    eth = make_u1_market(symbol="ETHUSDT", n_bars=5800, feature_level=2.0)
    btc_source, btc_published = _published_source(tmp_path / "btc", btc)
    eth_source, eth_published = _published_source(tmp_path / "eth", eth)
    train = (btc_source, eth_source)
    manifest_a = _manifest(train, admission_digest_char="a")
    manifest_b = _manifest(train, admission_digest_char="d")
    sources = (btc_published, eth_published)
    cutoff = btc_source.last_timestamp_ns

    normalizer_a = fit_universal_trade_sequence_normalizer(
        manifest=manifest_a,
        access=_train_access(manifest_a),
        sources=sources,
        contract=_contract(),
        knowledge_cutoff_ns=cutoff,
    )
    normalizer_b = fit_universal_trade_sequence_normalizer(
        manifest=manifest_b,
        access=_train_access(manifest_b),
        sources=sources,
        contract=_contract(),
        knowledge_cutoff_ns=cutoff,
    )
    assert normalizer_a.statistics_digest == normalizer_b.statistics_digest
    assert normalizer_a.digest != normalizer_b.digest
