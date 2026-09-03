from __future__ import annotations

import importlib

import pytest

from trade_rl.domain.universal_trade_rl_universe import UniversalTradeRLUniverseConfig
from trade_rl.workflows.universal_trade_rl_universe_config import UniversalTradeRLSymbolSource
from trade_rl.workflows.universal_trade_rl_universe_manifest import (
    UniversalTradeRLUniverseManifest,
    build_universal_trade_rl_universe_manifest,
)

_STEP_NS = 15 * 60 * 1_000_000_000
_BARS_PER_DAY = 96
_EPISODE_BARS = 720 * 4
_START_NS = _STEP_NS * 2_000_000
_DEFAULT_BARS = 620 * _BARS_PER_DAY


def _module():
    try:
        return importlib.import_module(
            "trade_rl.workflows.universal_trade_rl_u2_time_partition"
        )
    except ModuleNotFoundError:
        pytest.fail("Universal Trade RL U2 time partition is not implemented")


def _manifest(
    *,
    total_bars: int = _DEFAULT_BARS,
    misaligned_symbol: str | None = None,
    row_count_delta_symbol: str | None = None,
) -> UniversalTradeRLUniverseManifest:
    config = UniversalTradeRLUniverseConfig(
        train_symbols=("BTCUSDT", "ETHUSDT"),
        development_symbols=("SOLUSDT",),
        admission_symbols=("XRPUSDT",),
    )
    definitions = (
        ("BTCUSDT", "a", 0, 0),
        ("ETHUSDT", "b", 10, 0),
        ("SOLUSDT", "c", 5, 20),
        ("XRPUSDT", "d", 0, 10),
    )
    sources: list[UniversalTradeRLSymbolSource] = []
    for symbol, digest_char, start_offset, end_trim in definitions:
        bar_count = total_bars - start_offset - end_trim
        first = _START_NS + start_offset * _STEP_NS
        last = first + (bar_count - 1) * _STEP_NS
        if symbol == misaligned_symbol:
            first += 1
            last += 1
        row_count = bar_count + (1 if symbol == row_count_delta_symbol else 0)
        sources.append(
            UniversalTradeRLSymbolSource(
                symbol=symbol,
                dataset_digest=digest_char * 64,
                first_timestamp_ns=first,
                last_timestamp_ns=last,
                row_count=row_count,
            )
        )
    return build_universal_trade_rl_universe_manifest(
        config=config,
        sources=tuple(sources),
    )


def test_u2_time_partition_uses_common_dense_15m_interval_and_exact_bar_fractions() -> None:
    module = _module()
    manifest = _manifest()

    partition = module.build_universal_trade_rl_u2_time_partition(manifest=manifest)

    common_bars = _DEFAULT_BARS - 30
    common_first = _START_NS + 10 * _STEP_NS
    common_last = common_first + (common_bars - 1) * _STEP_NS
    fit_stop = common_bars * 6 // 10
    d1_stop = common_bars * 7 // 10
    d2_stop = common_bars * 8 // 10

    assert partition.universe_manifest_digest == manifest.digest
    assert partition.common_first_timestamp_ns == common_first
    assert partition.common_last_timestamp_ns == common_last
    assert partition.common_bar_count == common_bars

    fit = partition.window("fit")
    seen = partition.window("seen_time_probe")
    d1 = partition.window("development_future_1")
    d2 = partition.window("development_future_2")
    admission = partition.window("admission_future")

    assert (fit.start_bar_index, fit.stop_bar_index_exclusive) == (0, fit_stop)
    assert (d1.start_bar_index, d1.stop_bar_index_exclusive) == (fit_stop, d1_stop)
    assert (d2.start_bar_index, d2.stop_bar_index_exclusive) == (d1_stop, d2_stop)
    assert (admission.start_bar_index, admission.stop_bar_index_exclusive) == (
        d2_stop,
        common_bars,
    )
    assert fit.bar_count + d1.bar_count + d2.bar_count + admission.bar_count == common_bars
    assert seen.bar_count == 60 * _BARS_PER_DAY
    assert seen.stop_bar_index_exclusive == fit.stop_bar_index_exclusive
    assert partition.fit_end_ns == fit.last_timestamp_ns


def test_u2_time_partition_predeclares_non_overlapping_complete_720h_evaluation_tiles() -> None:
    module = _module()
    partition = module.build_universal_trade_rl_u2_time_partition(manifest=_manifest())

    for window_name in (
        "seen_time_probe",
        "development_future_1",
        "development_future_2",
        "admission_future",
    ):
        window = partition.window(window_name)
        tiles = partition.tiles_for(window_name)
        assert len(tiles) == window.bar_count // _EPISODE_BARS
        assert len(tiles) >= 2
        for index, tile in enumerate(tiles):
            assert tile.tile_index == index
            assert tile.source_window == window_name
            assert tile.bar_count == _EPISODE_BARS
            assert tile.start_bar_index == window.start_bar_index + index * _EPISODE_BARS
            assert tile.stop_bar_index_exclusive == tile.start_bar_index + _EPISODE_BARS
        for left, right in zip(tiles, tiles[1:]):
            assert left.stop_bar_index_exclusive == right.start_bar_index


def test_u2_time_partition_round_trips_canonically() -> None:
    module = _module()
    partition = module.build_universal_trade_rl_u2_time_partition(manifest=_manifest())

    restored = module.UniversalTradeRLU2TimePartition.from_payload(partition.to_payload())

    assert restored == partition
    assert restored.digest == partition.digest
    assert restored.to_payload() == partition.to_payload()


def test_u2_time_partition_rejects_common_history_shorter_than_600_days() -> None:
    module = _module()
    manifest = _manifest(total_bars=600 * _BARS_PER_DAY - 1)

    with pytest.raises(ValueError, match="600|common"):
        module.build_universal_trade_rl_u2_time_partition(manifest=manifest)


def test_u2_time_partition_rejects_misaligned_or_non_dense_source_metadata() -> None:
    module = _module()

    with pytest.raises(ValueError, match="15m|align"):
        module.build_universal_trade_rl_u2_time_partition(
            manifest=_manifest(misaligned_symbol="ETHUSDT")
        )
    with pytest.raises(ValueError, match="dense|row_count|15m"):
        module.build_universal_trade_rl_u2_time_partition(
            manifest=_manifest(row_count_delta_symbol="SOLUSDT")
        )


def test_u2_time_partition_builder_has_no_numeric_dataset_input_surface() -> None:
    module = _module()
    manifest = _manifest()

    with pytest.raises(TypeError):
        module.build_universal_trade_rl_u2_time_partition(
            manifest=manifest,
            datasets={},
        )
