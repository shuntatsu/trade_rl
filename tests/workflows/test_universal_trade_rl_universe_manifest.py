from __future__ import annotations

from copy import deepcopy

import pytest

from trade_rl.domain.universal_trade_rl_universe import (
    UniversalTradeRLSymbolExclusion,
    UniversalTradeRLSymbolRole,
    UniversalTradeRLUniverseConfig,
)
from trade_rl.workflows.universal_trade_rl_universe_config import (
    UniversalTradeRLSymbolSource,
    universal_trade_rl_source_catalog_digest,
)
from trade_rl.workflows.universal_trade_rl_universe_manifest import (
    UniversalTradeRLUniverseEntry,
    UniversalTradeRLUniverseManifest,
    build_universal_trade_rl_universe_manifest,
)


def _config() -> UniversalTradeRLUniverseConfig:
    return UniversalTradeRLUniverseConfig(
        train_symbols=("BTCUSDT", "ETHUSDT"),
        development_symbols=("LINKUSDT",),
        admission_symbols=("AVAXUSDT",),
        exclusions=(
            UniversalTradeRLSymbolExclusion(
                symbol="LUNA2USDT",
                reason="insufficient_contiguous_history",
            ),
        ),
    )


def _source(symbol: str, char: str) -> UniversalTradeRLSymbolSource:
    return UniversalTradeRLSymbolSource(
        symbol=symbol,
        dataset_digest=char * 64,
        first_timestamp_ns=1,
        last_timestamp_ns=100,
        row_count=100,
    )


def _sources() -> tuple[UniversalTradeRLSymbolSource, ...]:
    return (
        _source("AVAXUSDT", "a"),
        _source("BTCUSDT", "b"),
        _source("ETHUSDT", "c"),
        _source("LINKUSDT", "d"),
        _source("LUNA2USDT", "e"),
    )


def _manifest() -> UniversalTradeRLUniverseManifest:
    return build_universal_trade_rl_universe_manifest(
        config=_config(),
        sources=_sources(),
    )


def test_manifest_assigns_every_available_symbol_exactly_once() -> None:
    manifest = _manifest()

    assert tuple(entry.symbol for entry in manifest.entries) == (
        "AVAXUSDT",
        "BTCUSDT",
        "ETHUSDT",
        "LINKUSDT",
        "LUNA2USDT",
    )
    assert manifest.entry_for("BTCUSDT").role is UniversalTradeRLSymbolRole.TRAIN
    assert (
        manifest.entry_for("LINKUSDT").role
        is UniversalTradeRLSymbolRole.DEVELOPMENT
    )
    assert (
        manifest.entry_for("AVAXUSDT").role
        is UniversalTradeRLSymbolRole.ADMISSION
    )
    excluded = manifest.entry_for("LUNA2USDT")
    assert excluded.role is None
    assert excluded.exclusion_reason == "insufficient_contiguous_history"
    assert manifest.config_digest == _config().digest
    assert manifest.source_catalog_digest == universal_trade_rl_source_catalog_digest(
        _sources()
    )
    assert len(manifest.digest) == 64


def test_manifest_rejects_unassigned_available_symbol() -> None:
    sources = (*_sources(), _source("XRPUSDT", "f"))

    with pytest.raises(ValueError, match="unassigned"):
        build_universal_trade_rl_universe_manifest(
            config=_config(),
            sources=tuple(sorted(sources, key=lambda item: item.symbol)),
        )


def test_manifest_rejects_missing_configured_symbol() -> None:
    sources = tuple(item for item in _sources() if item.symbol != "AVAXUSDT")

    with pytest.raises(ValueError, match="missing configured symbol"):
        build_universal_trade_rl_universe_manifest(
            config=_config(),
            sources=sources,
        )


def test_manifest_rejects_missing_excluded_source() -> None:
    sources = tuple(item for item in _sources() if item.symbol != "LUNA2USDT")

    with pytest.raises(ValueError, match="missing configured symbol"):
        build_universal_trade_rl_universe_manifest(
            config=_config(),
            sources=sources,
        )


def test_entry_requires_exactly_one_role_or_exclusion() -> None:
    source = _source("BTCUSDT", "a")
    with pytest.raises(ValueError, match="exactly one"):
        UniversalTradeRLUniverseEntry(
            symbol=source.symbol,
            role=None,
            exclusion_reason=None,
            dataset_digest=source.dataset_digest,
            first_timestamp_ns=source.first_timestamp_ns,
            last_timestamp_ns=source.last_timestamp_ns,
            row_count=source.row_count,
        )
    with pytest.raises(ValueError, match="exactly one"):
        UniversalTradeRLUniverseEntry(
            symbol=source.symbol,
            role=UniversalTradeRLSymbolRole.TRAIN,
            exclusion_reason="also_excluded",
            dataset_digest=source.dataset_digest,
            first_timestamp_ns=source.first_timestamp_ns,
            last_timestamp_ns=source.last_timestamp_ns,
            row_count=source.row_count,
        )


def test_manifest_round_trips_from_payload() -> None:
    manifest = _manifest()

    restored = UniversalTradeRLUniverseManifest.from_payload(manifest.to_payload())

    assert restored == manifest
    assert restored.digest == manifest.digest


@pytest.mark.parametrize(
    ("entry_index", "field", "replacement"),
    (
        (0, "role", "train"),
        (0, "dataset_digest", "f" * 64),
        (0, "row_count", 101),
        (0, "first_timestamp_ns", 2),
        (0, "last_timestamp_ns", 101),
        (4, "exclusion_reason", "different_reason"),
    ),
)
def test_manifest_rejects_entry_tampering(
    entry_index: int, field: str, replacement: object
) -> None:
    payload = deepcopy(_manifest().to_payload())
    entries = [dict(item) for item in payload["entries"]]  # type: ignore[index]
    entries[entry_index][field] = replacement
    payload["entries"] = entries

    with pytest.raises(ValueError, match="digest|contract"):
        UniversalTradeRLUniverseManifest.from_payload(payload)


def test_manifest_rejects_top_level_digest_tampering() -> None:
    payload = dict(_manifest().to_payload())
    payload["config_digest"] = "f" * 64

    with pytest.raises(ValueError, match="digest"):
        UniversalTradeRLUniverseManifest.from_payload(payload)


def test_manifest_rejects_unsorted_entries_even_with_recomputed_payload_shape() -> None:
    manifest = _manifest()

    with pytest.raises(ValueError, match="sorted"):
        UniversalTradeRLUniverseManifest(
            config_digest=manifest.config_digest,
            source_catalog_digest=manifest.source_catalog_digest,
            entries=tuple(reversed(manifest.entries)),
        )
