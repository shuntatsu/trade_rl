"""Reject invalid daily funding lookbacks before indexing prefix history."""

from __future__ import annotations

import numpy as np
import pytest

from tests.evaluation.test_dev_realdata_phase2_trials import synthetic_market
from trade_rl.evaluation.dev_realdata_study.trial_sources import DailyPanel


@pytest.fixture(scope="module")
def panel() -> DailyPanel:
    return DailyPanel(synthetic_market())


@pytest.mark.parametrize(
    "window_days",
    (
        pytest.param(0, id="zero"),
        pytest.param(-1, id="negative-one"),
        pytest.param(-2, id="negative-future-lookahead"),
        pytest.param(np.int64(-7), id="negative-numpy-integer"),
        pytest.param(True, id="boolean-true"),
        pytest.param(False, id="boolean-false"),
        pytest.param(np.bool_(True), id="numpy-boolean"),
        pytest.param(7.0, id="float-whole-number"),
        pytest.param(float("nan"), id="float-nan"),
        pytest.param(float("inf"), id="float-inf"),
        pytest.param("7", id="string"),
    ),
)
def test_funding_carry_rejects_nonpositive_or_noninteger_windows(
    panel: DailyPanel, window_days: object
) -> None:
    # In particular, a negative lookback would place the prefix low bound
    # after the decision row, sampling future funding with a finite result.
    with pytest.raises(
        ValueError, match="funding window_days must be a positive integer"
    ):
        panel.funding_carry_series(window_days)  # type: ignore[arg-type]


def test_positive_numpy_integer_preserves_funding_window_semantics(
    panel: DailyPanel,
) -> None:
    np.testing.assert_array_equal(
        panel.funding_carry_series(np.int64(7)),
        panel.funding_carry_series(7),
    )


def test_one_day_funding_window_uses_only_completed_history(panel: DailyPanel) -> None:
    carry = panel.funding_carry_series(1)
    assert np.isnan(carry[0]).all()
    assert np.isfinite(carry[1:]).all()


def test_large_positive_numpy_window_does_not_wrap_into_future(
    panel: DailyPanel,
) -> None:
    # np.int64(130_000) * nanoseconds_per_day wraps to a negative value.
    # A Python-integer lookback predates the available history instead.
    assert np.isnan(panel.funding_carry_series(np.int64(130_000))).all()
