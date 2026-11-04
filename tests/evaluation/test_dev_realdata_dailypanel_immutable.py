"""The Phase 2 daily panel must not let callers modify past signal inputs."""

from __future__ import annotations

import numpy as np
import pytest

from tests.evaluation.test_dev_realdata_phase2_trials import synthetic_market
from trade_rl.evaluation.dev_realdata_study.trial_sources import (
    DailyPanel,
    tsmom_series,
)
from trade_rl.evaluation.dev_realdata_study.trials import CONSTANTS


@pytest.fixture(scope="module")
def panel() -> DailyPanel:
    return DailyPanel(synthetic_market())


@pytest.mark.parametrize(
    "field",
    [
        "rows",
        "timestamps_ns",
        "marks",
        "_stamps",
        "_funding_prefix",
        "_unavailable_signal_history",
    ],
)
def test_daily_panel_source_buffers_are_deeply_immutable(
    panel: DailyPanel, field: str
) -> None:
    values = getattr(panel, field)
    assert isinstance(values, np.ndarray)
    before = values.copy()

    assert values.flags.writeable is False
    with pytest.raises(ValueError):
        values.flat[0] = 0
    with pytest.raises(ValueError):
        values.setflags(write=True)
    np.testing.assert_array_equal(values, before)


def test_daily_panel_mark_and_funding_signals_remain_stable_after_write_attempts(
    panel: DailyPanel,
) -> None:
    expected_trend = tsmom_series(
        panel.marks,
        one_way_cost=0.0007,
        constants=CONSTANTS,
    )
    expected_carry = panel.funding_carry_series(7)

    with pytest.raises(ValueError):
        panel.marks[100, 0] = 1e12
    with pytest.raises(ValueError):
        panel._funding_prefix[-1, 0] = 1e12

    np.testing.assert_array_equal(
        tsmom_series(panel.marks, one_way_cost=0.0007, constants=CONSTANTS),
        expected_trend,
    )
    np.testing.assert_array_equal(panel.funding_carry_series(7), expected_carry)
