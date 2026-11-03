"""Phase 2 daily signals must not use marks or funding known only after decision."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from tests.evaluation.test_dev_realdata_phase2_trials import small_protocol, synthetic_market
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.dev_realdata_study.trial_sources import (
    DailyPanel,
    SeriesLookupSource,
    TrialContext,
    build_trial_source,
    trial_spec,
)

DECISION = "2021-10-04T00:00:00"
PREVIOUS = "2021-10-03T00:00:00"


def _ns(value: str) -> int:
    return int(np.datetime64(value, "ns").astype(np.int64))


def _delayed(
    market: MarketDataset, *, source_time: str, arrival_time: str
) -> MarketDataset:
    row = int(np.searchsorted(market.timestamps, np.datetime64(source_time, "ns")))
    assert market.timestamps[row] == np.datetime64(source_time, "ns")
    available_at = market.available_at
    assert available_at is not None
    changed = available_at.copy()
    changed[row, 0] = np.datetime64(arrival_time, "ns")
    return replace(market, available_at=changed, information_available=None)


def _adapter(panel: DailyPanel) -> SeriesLookupSource:
    series = np.full(panel.marks.shape, 0.01, dtype=np.float64)
    return SeriesLookupSource(panel, series)


@pytest.mark.parametrize(
    ("source_time", "arrival_time"),
    [
        ("2021-10-04T00:00:00", "2021-10-04T01:00:00"),
        ("2021-10-03T16:00:00", "2021-10-04T01:00:00"),
    ],
    ids=("unarrived_daily_mark", "unarrived_funding_event"),
)
def test_phase2_rejects_unarrived_source_before_recording_signal(
    source_time: str, arrival_time: str
) -> None:
    market = synthetic_market()
    source_row = int(np.searchsorted(market.timestamps, np.datetime64(source_time)))
    if source_time.endswith("16:00:00"):
        assert bool(market.funding_due[source_row, 0])
    delayed = _delayed(market, source_time=source_time, arrival_time=arrival_time)
    panel = DailyPanel(delayed)
    adapter = _adapter(panel)
    snapshot = SimpleNamespace(symbols=market.symbols, timestamp_ns=_ns(DECISION))
    # This is a property of the recorded source arrival, not of the future
    # perturbation probe: the original panel accepted both late observations.
    with pytest.raises(ValueError, match="Phase 2 source not available"):
        adapter.expected_simple_returns(snapshot)  # type: ignore[arg-type]
    assert adapter.calls == 0
    assert adapter.null_packets == 0
    assert adapter.history == []
    assert panel.position(_ns(PREVIOUS)) >= 0


def test_delayed_funding_received_before_next_daily_decision_is_valid() -> None:
    market = synthetic_market()
    delayed = _delayed(
        market,
        source_time="2021-10-03T16:00:00",
        arrival_time="2021-10-03T18:00:00",
    )
    panel = DailyPanel(delayed)
    assert panel.position(_ns(DECISION)) >= 0
    adapter = _adapter(panel)
    result = adapter.expected_simple_returns(
        SimpleNamespace(symbols=market.symbols, timestamp_ns=_ns(DECISION))
    )  # type: ignore[arg-type]
    assert result == (0.01,) * market.n_symbols
    assert adapter.calls == 1


def test_late_future_observation_does_not_block_prior_decisions() -> None:
    market = synthetic_market()
    delayed = _delayed(
        market,
        source_time="2021-10-05T00:00:00",
        arrival_time="2021-10-05T01:00:00",
    )
    panel = DailyPanel(delayed)
    assert panel.position(_ns(DECISION)) >= 0
    with pytest.raises(ValueError, match="Phase 2 source not available"):
        panel.position(_ns("2021-10-05T00:00:00"))


def test_non_daily_non_funding_source_delay_is_not_used_by_daily_signals() -> None:
    market = synthetic_market()
    row = int(np.searchsorted(market.timestamps, np.datetime64("2021-10-03T17:00:00")))
    assert not bool(market.funding_due[row, 0])
    delayed = _delayed(
        market,
        source_time="2021-10-03T17:00:00",
        arrival_time="2021-10-04T01:00:00",
    )
    assert DailyPanel(delayed).position(_ns(DECISION)) >= 0

def test_registered_t3_source_rejects_unarrived_due_funding() -> None:
    market = synthetic_market()
    delayed = _delayed(
        market,
        source_time="2021-10-03T16:00:00",
        arrival_time="2021-10-04T01:00:00",
    )
    context = TrialContext(delayed, small_protocol())
    adapter = build_trial_source(context, trial_spec("t3_funding_carry"))
    snapshot = SimpleNamespace(symbols=market.symbols, timestamp_ns=_ns(DECISION))
    with pytest.raises(ValueError, match="Phase 2 source not available"):
        adapter.expected_simple_returns(snapshot)
    assert adapter.calls == 0
    assert adapter.null_packets == 0
    assert adapter.history == []
