from __future__ import annotations

from typing import Any

import pytest

from trade_rl.workflows.universal_trade_rl_u2_preflight import (
    U2TrainingSource,
    U2TrainingSourceClosure,
    load_universal_trade_rl_u2_fit_sources,
)

_STEP_NS = 15 * 60 * 1_000_000_000
_FIT_FIRST_NS = 100 * _STEP_NS
_FIT_BAR_COUNT = 16
_FIT_LAST_NS = _FIT_FIRST_NS + (_FIT_BAR_COUNT - 1) * _STEP_NS
_FIT_STOP_NS = _FIT_LAST_NS + _STEP_NS
_SOURCE_FIRST_NS = _FIT_FIRST_NS - 4 * _STEP_NS
_SOURCE_ROW_COUNT = _FIT_BAR_COUNT + 12
_SOURCE_LAST_NS = _SOURCE_FIRST_NS + (_SOURCE_ROW_COUNT - 1) * _STEP_NS


class SpyLoader:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    def __call__(self, request: Any) -> Any:
        self.calls.append(request)
        return request


def _source(symbol: str, digest_char: str) -> U2TrainingSource:
    return U2TrainingSource(
        symbol=symbol,
        dataset_digest=digest_char * 64,
        source_first_timestamp_ns=_SOURCE_FIRST_NS,
        source_last_timestamp_ns=_SOURCE_LAST_NS,
        source_row_count=_SOURCE_ROW_COUNT,
        fit_first_timestamp_ns=_FIT_FIRST_NS,
        fit_last_timestamp_ns=_FIT_LAST_NS,
        fit_stop_timestamp_ns_exclusive=_FIT_STOP_NS,
        fit_bar_count=_FIT_BAR_COUNT,
    )


def _closure() -> U2TrainingSourceClosure:
    return U2TrainingSourceClosure(
        u2_contract_digest="1" * 64,
        universe_manifest_digest="2" * 64,
        u1_contract_digest="3" * 64,
        normalizer_digest="4" * 64,
        normalizer_provenance_digest="5" * 64,
        time_partition_digest="6" * 64,
        fit_first_timestamp_ns=_FIT_FIRST_NS,
        fit_last_timestamp_ns=_FIT_LAST_NS,
        fit_stop_timestamp_ns_exclusive=_FIT_STOP_NS,
        fit_bar_count=_FIT_BAR_COUNT,
        sources=(
            _source("BTCUSDT", "a"),
            _source("ETHUSDT", "b"),
        ),
    )


def test_mixed_valid_and_invalid_scope_rejects_before_any_numeric_read() -> None:
    loader = SpyLoader()

    with pytest.raises(ValueError, match="outside|Train|training|scope|symbol"):
        load_universal_trade_rl_u2_fit_sources(
            closure=_closure(),
            requested_symbols=("BTCUSDT", "XRPUSDT"),
            loader=loader,
        )

    assert loader.calls == []


def test_u2_training_source_rejects_fit_grid_not_aligned_to_source_grid() -> None:
    source_first = 100 * _STEP_NS
    source_rows = 200
    source_last = source_first + (source_rows - 1) * _STEP_NS
    fit_first = source_first + 5 * 60 * 1_000_000_000
    fit_rows = 32
    fit_last = fit_first + (fit_rows - 1) * _STEP_NS

    with pytest.raises(ValueError, match="FIT|source|align|grid"):
        U2TrainingSource(
            symbol="BTCUSDT",
            dataset_digest="a" * 64,
            source_first_timestamp_ns=source_first,
            source_last_timestamp_ns=source_last,
            source_row_count=source_rows,
            fit_first_timestamp_ns=fit_first,
            fit_last_timestamp_ns=fit_last,
            fit_stop_timestamp_ns_exclusive=fit_last + _STEP_NS,
            fit_bar_count=fit_rows,
        )
