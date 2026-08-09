from __future__ import annotations

import importlib
import importlib.util
from dataclasses import replace

import numpy as np
import pytest

from tests.data.test_market_dataset_v2 import kwargs
from tests.workflows.test_stage_a_historical_interval_evidence import (
    _funding_boundary,
    _two_transition_replay,
)
from trade_rl.data.market import MarketDataset

_MODULE = "trade_rl.workflows.stage_a_nautilus_historical_replay"


def _builder():
    spec = importlib.util.find_spec(_MODULE)
    assert spec is not None, "Stage A Nautilus historical replay workflow must exist"
    module = importlib.import_module(_MODULE)
    build = getattr(module, "build_stage_a_nautilus_historical_replay_intervals", None)
    assert callable(build), "Stage A Nautilus historical replay builder must exist"
    return build


def _single_symbol_market() -> MarketDataset:
    values = kwargs(n_bars=100, n_symbols=1)
    close = np.asarray(values["close"], dtype=np.float64)
    values["mark_price"] = close + 0.25
    values["index_price"] = close - 0.25
    return MarketDataset(**values)


def _replay_for_market(market: MarketDataset):
    replay = _two_transition_replay()
    cell_identity = replace(
        replay.cell_identity,
        dataset_id=market.dataset_id,
        digest="",
    )
    return replace(replay, cell_identity=cell_identity, digest="")


def _timestamp_ns(market: MarketDataset, index: int) -> int:
    return int(market.timestamps[index].astype("datetime64[ns]").astype(np.int64))


def test_historical_replay_bridge_binds_exact_source_bars_and_interval_evidence() -> (
    None
):
    build = _builder()
    market = _single_symbol_market()
    replay = _replay_for_market(market)
    start = replay.cell_identity.evaluation_range.start
    shared = replay.transition_end_indices[0]
    stop = replay.cell_identity.evaluation_range.stop
    funding = (
        _funding_boundary(start, timestamp_ns=_timestamp_ns(market, start)),
        _funding_boundary(shared, timestamp_ns=_timestamp_ns(market, shared)),
        _funding_boundary(stop, timestamp_ns=_timestamp_ns(market, stop)),
    )

    intervals = build(replay, market, funding_evidence=funding)

    assert tuple(item.evidence.action for item in intervals) == replay.actions
    assert tuple(
        tuple(
            boundary.processing_index for boundary in item.evidence.funding_boundaries
        )
        for item in intervals
    ) == ((start, shared), (stop,))
    assert tuple(
        tuple(bar.close_ns for bar in item.source_bars) for item in intervals
    ) == (
        tuple(_timestamp_ns(market, index) for index in range(start + 1, shared + 1)),
        tuple(_timestamp_ns(market, index) for index in range(shared + 1, stop + 1)),
    )


def test_historical_replay_bridge_rejects_dataset_identity_mismatch() -> None:
    build = _builder()
    market = _single_symbol_market()
    replay = _replay_for_market(market)
    mismatched_identity = replace(
        replay.cell_identity,
        dataset_id="f" * 64,
        digest="",
    )
    mismatched_replay = replace(
        replay,
        cell_identity=mismatched_identity,
        digest="",
    )

    with pytest.raises(ValueError, match="dataset identity mismatch"):
        build(mismatched_replay, market)
