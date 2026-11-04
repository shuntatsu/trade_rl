"""A finite funding input must not overflow daily carry accumulation."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tests.evaluation.test_dev_realdata_phase2_trials import synthetic_market
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.dev_realdata_study.trial_sources import DailyPanel


def _finite_funding(*, due_events: int, non_due: bool = False) -> MarketDataset:
    market = synthetic_market()
    funding = np.zeros_like(market.funding_rate)
    due = market.funding_due
    assert due is not None
    rows = np.flatnonzero(~due[:, 0] if non_due else due[:, 0])
    assert len(rows) >= due_events
    funding[rows[:due_events], 0] = 1e308
    changed = replace(market, funding_rate=funding)
    assert np.isfinite(changed.funding_rate).all()
    return changed


def test_daily_panel_rejects_overflowed_funding_prefix_from_finite_events() -> None:
    market = _finite_funding(due_events=2)
    with pytest.raises(ValueError, match="cumulative funding must be finite"):
        DailyPanel(market)


def test_daily_panel_preserves_one_large_but_finite_funding_event() -> None:
    panel = DailyPanel(_finite_funding(due_events=1))
    assert np.isfinite(panel._funding_prefix).all()
    assert panel._funding_prefix[-1, 0] == pytest.approx(1e308)


def test_daily_panel_ignores_huge_values_on_non_funding_due_rows() -> None:
    panel = DailyPanel(_finite_funding(due_events=2, non_due=True))
    assert np.isfinite(panel._funding_prefix).all()
    assert np.count_nonzero(panel._funding_prefix) == 0
