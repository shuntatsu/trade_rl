from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.catalog.stored_instrument_catalog import (
    StoredIndicatorArtifactEvidence,
    StoredIndicatorSourceInventory,
    build_stored_instrument_catalog,
)
from trade_rl.workflows import universal_instrument_artifacts as artifacts_module
from trade_rl.workflows.universal_instrument_artifacts import (
    STORED_INSTRUMENTS_FILENAME,
    SYMBOL_DISJOINT_FILENAME,
    UNIVERSAL_INSTRUMENT_PARTITION_FILENAME,
    UniversalInstrumentArtifactBundle,
    UniversalInstrumentArtifactPaths,
    build_universal_instrument_artifact_bundle,
    load_universal_instrument_artifact_bundle,
    materialize_universal_instrument_artifacts,
    write_universal_instrument_artifact_bundle,
)
from trade_rl.workflows.universal_instrument_partition import (
    build_universal_instrument_partition,
)

_TIMEFRAMES = ("15m", "1h", "4h", "1d")
_START = datetime(2021, 1, 1, tzinfo=UTC)
_END = datetime(2026, 7, 1, tzinfo=UTC)


def _source(
    symbol_count: int = 15,
    *,
    zero_available: tuple[str, str] | None = None,
) -> StoredIndicatorSourceInventory:
    symbols = tuple(f"ASSET{index:02d}USDT" for index in range(symbol_count))
    artifacts = tuple(
        StoredIndicatorArtifactEvidence(
            symbol=symbol,
            timeframe=timeframe,
            row_count=100,
            feature_count=1,
            available_value_count=(
                0 if zero_available == (symbol, timeframe) else 100
            ),
            first_event_time_ms=1_609_459_200_000,
            last_event_time_ms=1_782_864_000_000,
            payload_schema=f"npz_native_indicator_v1:{content_digest(timeframe)}",
            payload_sha256=content_digest((symbol, timeframe)),
            payload_bytes=4096,
        )
        for symbol in symbols
        for timeframe in _TIMEFRAMES
    )
    return StoredIndicatorSourceInventory(
        cache_id="verified-native-indicators-v1",
        source_manifest_digest=content_digest("source-manifest"),
        market="usds-m",
        symbols=symbols,
        start_time=_START,
        end_time=_END,
        feature_config_digest=content_digest("features"),
        required_timeframes=_TIMEFRAMES,
        artifacts=artifacts,
    )


def _metadata(symbols: tuple[str, ...], *, version: str = "v1") -> dict[str, str]:
    return {
        symbol: content_digest({"metadata": symbol, "version": version})
        for symbol in symbols
    }


def _bundle(*, seed: int = 17, metadata_version: str = "v1") -> UniversalInstrumentArtifactBundle:
    source = _source()
    catalog = build_stored_instrument_catalog(
        source,
        research_start=_START,
        research_end=_END,
        metadata_digests=_metadata(source.symbols, version=metadata_version),
    )
    partition = build_universal_instrument_partition(catalog, seed=seed)
    return UniversalInstrumentArtifactBundle(
        catalog=catalog,
        symbol_disjoint_manifest=partition.symbol_disjoint_manifest,
        partition=partition,
    )


def test_paths_expose_exact_bundle_filenames(tmp_path: Path) -> None:
    root = tmp_path / "universal-instruments"

    paths = UniversalInstrumentArtifactPaths.for_root(root)

    assert paths.root == root
    assert paths.stored_instruments.name == STORED_INSTRUMENTS_FILENAME
    assert paths.symbol_disjoint.name == SYMBOL_DISJOINT_FILENAME
    assert paths.universal_partition.name == UNIVERSAL_INSTRUMENT_PARTITION_FILENAME


def test_bundle_requires_catalog_manifest_partition_cross_binding() -> None:
    bundle = _bundle(seed=17)

    assert bundle.partition.catalog_digest == bundle.catalog.digest
    assert bundle.partition.symbol_disjoint_manifest == (
        bundle.symbol_disjoint_manifest
    )
    assert set(bundle.symbol_disjoint_manifest.source_universe) == set(
        bundle.catalog.eligible_symbols
    )


def test_bundle_rejects_manifest_or_catalog_from_another_bundle() -> None:
    first = _bundle(seed=17, metadata_version="v1")
    different_manifest = _bundle(seed=23, metadata_version="v1")
    different_catalog = _bundle(seed=17, metadata_version="v2")

    with pytest.raises(ValueError, match="manifest|partition"):
        UniversalInstrumentArtifactBundle(
            catalog=first.catalog,
            symbol_disjoint_manifest=different_manifest.symbol_disjoint_manifest,
            partition=first.partition,
        )

    with pytest.raises(ValueError, match="catalog"):
        UniversalInstrumentArtifactBundle(
            catalog=different_catalog.catalog,
            symbol_disjoint_manifest=first.symbol_disjoint_manifest,
            partition=first.partition,
        )


def test_builds_one_deterministic_bundle_after_catalog_exclusion() -> None:
    source = _source(symbol_count=16)
    metadata = _metadata(source.symbols)
    metadata.pop(source.symbols[0])

    first = build_universal_instrument_artifact_bundle(
        source,
        research_start=_START,
        research_end=_END,
        metadata_digests=metadata,
        seed=17,
    )
    second = build_universal_instrument_artifact_bundle(
        source,
        research_start=_START,
        research_end=_END,
        metadata_digests=metadata,
        seed=17,
    )

    assert second == first
    assert first.catalog.eligible_symbols == source.symbols[1:]
    assert tuple(
        (item.symbol, item.reasons) for item in first.catalog.excluded_symbols
    ) == ((source.symbols[0], ("missing_execution_metadata",)),)
    assert first.symbol_disjoint_manifest.source_universe == tuple(
        sorted(first.catalog.eligible_symbols)
    )
    assert first.partition.catalog_digest == first.catalog.digest


def test_build_fails_closed_below_zero_shot_minimum() -> None:
    source = _source(symbol_count=15)
    metadata = _metadata(source.symbols)
    metadata.pop(source.symbols[0])

    with pytest.raises(ValueError, match="at least 15"):
        build_universal_instrument_artifact_bundle(
            source,
            research_start=_START,
            research_end=_END,
            metadata_digests=metadata,
            seed=17,
        )


def test_materializes_exact_file_closure_and_round_trips(tmp_path: Path) -> None:
    root = tmp_path / "universal-instruments"
    expected = _bundle(seed=17)

    paths = write_universal_instrument_artifact_bundle(root, expected)

    assert {path.name for path in root.iterdir()} == {
        STORED_INSTRUMENTS_FILENAME,
        SYMBOL_DISJOINT_FILENAME,
        UNIVERSAL_INSTRUMENT_PARTITION_FILENAME,
    }
    assert paths == UniversalInstrumentArtifactPaths.for_root(root)
    assert load_universal_instrument_artifact_bundle(root) == expected


def test_exact_existing_bundle_is_reused_but_mismatch_is_rejected(
    tmp_path: Path,
) -> None:
    root = tmp_path / "universal-instruments"
    first = _bundle(seed=17)
    different = _bundle(seed=23)
    write_universal_instrument_artifact_bundle(root, first)
    before = {path.name: path.read_bytes() for path in root.iterdir()}

    write_universal_instrument_artifact_bundle(root, first)

    assert {path.name: path.read_bytes() for path in root.iterdir()} == before
    with pytest.raises(FileExistsError, match="different|mismatch"):
        write_universal_instrument_artifact_bundle(root, different)
    assert {path.name: path.read_bytes() for path in root.iterdir()} == before


def test_partial_extra_and_non_directory_outputs_fail_closed(tmp_path: Path) -> None:
    bundle = _bundle()

    partial = tmp_path / "partial"
    partial.mkdir()
    marker = partial / STORED_INSTRUMENTS_FILENAME
    marker.write_text("partial", encoding="utf-8")
    with pytest.raises(FileExistsError, match="partial|closure|different"):
        write_universal_instrument_artifact_bundle(partial, bundle)
    assert marker.read_text(encoding="utf-8") == "partial"

    extra = tmp_path / "extra"
    write_universal_instrument_artifact_bundle(extra, bundle)
    (extra / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(ValueError, match="closure"):
        load_universal_instrument_artifact_bundle(extra)

    regular_file = tmp_path / "file"
    regular_file.write_text("not a directory", encoding="utf-8")
    with pytest.raises((FileExistsError, ValueError), match="directory"):
        write_universal_instrument_artifact_bundle(regular_file, bundle)


def test_symlink_bundle_root_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    root = tmp_path / "linked"
    try:
        root.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable on this platform")

    with pytest.raises(ValueError, match="symlink"):
        load_universal_instrument_artifact_bundle(root)
    with pytest.raises(FileExistsError, match="symlink"):
        write_universal_instrument_artifact_bundle(root, _bundle())


def test_tampered_dependency_is_rejected_on_load(tmp_path: Path) -> None:
    root = tmp_path / "universal-instruments"
    write_universal_instrument_artifact_bundle(root, _bundle())
    catalog_path = root / STORED_INSTRUMENTS_FILENAME
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    payload["eligible_symbols"] = payload["eligible_symbols"][:-1]
    catalog_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="digest|closure"):
        load_universal_instrument_artifact_bundle(root)


def test_staging_failure_leaves_no_published_or_temporary_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "universal-instruments"

    def fail_write(*args: object, **kwargs: object) -> Path:
        del args, kwargs
        raise OSError("injected write failure")

    monkeypatch.setattr(
        artifacts_module,
        "write_symbol_disjoint_manifest",
        fail_write,
    )

    with pytest.raises(OSError, match="injected"):
        write_universal_instrument_artifact_bundle(root, _bundle())

    assert not root.exists()
    assert not tuple(tmp_path.glob(".universal-instruments.staging-*"))


def test_materializer_validates_before_touching_filesystem(tmp_path: Path) -> None:
    root = tmp_path / "universal-instruments"
    source = _source(symbol_count=15)
    metadata = _metadata(source.symbols)
    metadata.pop(source.symbols[0])

    with pytest.raises(ValueError, match="at least 15"):
        materialize_universal_instrument_artifacts(
            root,
            source,
            research_start=_START,
            research_end=_END,
            metadata_digests=metadata,
            seed=17,
        )

    assert not root.exists()
    assert not tuple(tmp_path.glob(".universal-instruments.staging-*"))
