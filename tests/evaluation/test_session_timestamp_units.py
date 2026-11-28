"""Session boundaries are UTC nanoseconds, not raw datetime64 ticks (#1357)."""

from dataclasses import replace

import numpy as np
import pytest

from tests.evaluation.test_native_fixed_linear_carry import _HoldAfterEntry, _cost
from tests.evaluation.test_session_elapsed_replay import _Policy
from tests.evaluation.test_session_elapsed_replay import _dataset as _spot_dataset
from tests.evaluation.test_session_elapsed_replay import _run as _run_spot
from tests.evaluation.test_session_linear_carry import _dataset as _signed_dataset
from tests.evaluation.test_session_linear_carry import _run as _run_signed
from trade_rl.evaluation.replay import run_shared_portfolio_replay
from trade_rl.evaluation.series import SessionElapsedReturnSeries
from trade_rl.evaluation.session_returns import (
    load_session_elapsed_return_payload,
    session_elapsed_return_payload,
)


def _with_unit(dataset, unit):
    result = replace(
        dataset,
        timestamps=dataset.timestamps.astype(f"datetime64[{unit}]"),
        identity_payload_json=None,
    ).with_content_identity()
    assert result.identity_verified
    assert result.timestamps.dtype == np.dtype(f"datetime64[{unit}]")
    return result


@pytest.mark.parametrize("unit", ["ns", "ms", "s", "h"])
def test_session_return_codec_uses_actual_utc_boundaries(unit):
    reference = _spot_dataset()
    dataset = _with_unit(reference, unit)
    boundaries = tuple(int(x) for x in reference.timestamps.view(np.int64))
    returns = SessionElapsedReturnSeries.from_boundaries(
        (0.0,) * 5, boundaries_ns=boundaries
    )
    payload = session_elapsed_return_payload(dataset, returns, start_index=0)
    assert load_session_elapsed_return_payload(dataset, payload) == returns
    if unit != "ns":
        raw_ticks = tuple(int(x) for x in dataset.timestamps.view(np.int64))
        forged = SessionElapsedReturnSeries.from_boundaries(
            (0.0,) * 5, boundaries_ns=raw_ticks
        )
        with pytest.raises(ValueError, match="Dataset interval"):
            session_elapsed_return_payload(dataset, forged, start_index=0)


@pytest.mark.parametrize("unit", ["ns", "ms", "s", "h"])
def test_spot_session_replay_is_independent_of_timestamp_storage_unit(unit):
    reference = _spot_dataset()
    expected = _run_spot(reference, _Policy())
    dataset = _with_unit(reference, unit)
    actual = _run_spot(
        dataset, _Policy(), expected_boundaries_ns=expected.returns.boundaries_ns
    )
    assert actual.returns == expected.returns
    assert actual.book.cash == pytest.approx(expected.book.cash)
    assert actual.book.exact_quantities == expected.book.exact_quantities


@pytest.mark.parametrize("unit", ["ns", "ms", "s", "h"])
def test_signed_session_direct_and_paired_use_actual_elapsed_time(unit):
    reference = _signed_dataset()
    expected = _run_signed(reference, _HoldAfterEntry(-1))
    dataset = _with_unit(reference, unit)
    direct = run_shared_portfolio_replay(
        dataset,
        _HoldAfterEntry(-1),
        start_index=0,
        stop_index=5,
        initial_capital=1000,
        execution_cost=_cost(),
        settle_terminal_position=True,
        capture_ledger_evidence=True,
        native_session_linear_carry=True,
    )
    paired = _run_signed(
        dataset,
        _HoldAfterEntry(-1),
        expected_boundaries_ns=expected.returns.boundaries_ns,
    )
    assert direct.returns == paired.returns == expected.returns
    assert direct.ledger_evidence == paired.ledger_evidence
    assert direct.book.cash == pytest.approx(expected.book.cash)
    assert direct.book.exact_quantities == expected.book.exact_quantities
