"""SD1 (#1146): a leakage-injection counterexample the harness must flag.

The synthetic features below deliberately use information from after the decision time.
The future-perturbation probe must catch every one of them, must not flag a causal
feature, and a plain "the label looks too good" check must not be the only defence.
Synthetic price random walks only; no market value is read.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from trade_rl.evaluation.signal_measurement import (
    SignalMeasurementError,
    affine_label_copy,
    forward_log_returns,
    future_perturbation_probe,
    rank_ic,
)

ROWS, SYMBOLS = 400, 3
CUTOFFS = (50, 150, 300)
HORIZON = 3


def _prices(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100.0 * np.exp(
        np.cumsum(0.01 * rng.standard_normal((ROWS, SYMBOLS)), axis=0)
    )


# ---------------------------------------------------------------- builders (rows x symbols)


def causal_momentum(prices: np.ndarray, lookback: int = 5) -> np.ndarray:
    out = np.full(prices.shape, np.nan)
    out[lookback:] = np.log(prices[lookback:] / prices[:-lookback])
    return out


def causal_expanding_standardise(prices: np.ndarray) -> np.ndarray:
    """Momentum standardised by the mean and deviation of strictly earlier rows."""

    feature = causal_momentum(prices)
    out = np.full(feature.shape, np.nan)
    for t in range(30, feature.shape[0]):
        past = feature[:t]
        past = past[np.isfinite(past).all(axis=1)]
        if past.shape[0] > 10:
            out[t] = (feature[t] - past.mean(axis=0)) / past.std(axis=0)
    return out


def leaky_forward_return(prices: np.ndarray) -> np.ndarray:
    """A direct copy of the matured label: ``ln(p[t+h] / p[t])``."""

    return forward_log_returns(prices, HORIZON)


def leaky_one_step_lookahead(prices: np.ndarray) -> np.ndarray:
    out = np.full(prices.shape, np.nan)
    out[:-1] = np.log(prices[1:] / prices[:-1])
    return out


def leaky_centered_average(prices: np.ndarray) -> np.ndarray:
    """Distance to a centred 5-bar average: uses two future bars. Not a label copy."""

    out = np.full(prices.shape, np.nan)
    for t in range(2, prices.shape[0] - 2):
        out[t] = prices[t] / prices[t - 2 : t + 3].mean(axis=0) - 1.0
    return out


def leaky_global_standardise(prices: np.ndarray) -> np.ndarray:
    """Causal momentum standardised with full-sample statistics (a future-stat leak)."""

    feature = causal_momentum(prices)
    return (feature - np.nanmean(feature, axis=0)) / np.nanstd(feature, axis=0)


def leaky_distant_future(prices: np.ndarray) -> np.ndarray:
    out = np.full(prices.shape, np.nan)
    out[:-50] = prices[50:] / prices[:-50] - 1.0
    return out


CAUSAL = {
    "momentum": causal_momentum,
    "expanding_standardise": causal_expanding_standardise,
}
LEAKY = {
    "forward_return_copy": leaky_forward_return,
    "one_step_lookahead": leaky_one_step_lookahead,
    "centered_average": leaky_centered_average,
    "global_standardise": leaky_global_standardise,
    "distant_future": leaky_distant_future,
}


# ----------------------------------------------------------------------------- the probe


@pytest.mark.parametrize("name", sorted(CAUSAL))
def test_a_causal_feature_is_not_flagged(name: str) -> None:
    result = future_perturbation_probe(
        CAUSAL[name], _prices(), cutoffs=CUTOFFS, seed=0, relative_scale=0.05
    )

    assert result.leaks is False
    assert result.leaking_cutoffs == ()
    assert result.cutoffs == CUTOFFS


@pytest.mark.parametrize("name", sorted(LEAKY))
def test_every_feature_that_uses_the_future_is_flagged_at_every_cutoff(
    name: str,
) -> None:
    result = future_perturbation_probe(
        LEAKY[name], _prices(), cutoffs=CUTOFFS, seed=0, relative_scale=0.05
    )

    assert result.leaks is True
    assert result.leaking_cutoffs == CUTOFFS
    assert all(changed >= 1 for _, changed in result.changed_rows)


def test_the_probe_reports_how_many_past_rows_a_leak_moved() -> None:
    result = future_perturbation_probe(
        leaky_one_step_lookahead, _prices(), cutoffs=(100,), seed=0, relative_scale=0.05
    )

    # Row 100 reads p[101]; rows before it read only unperturbed prices.
    assert result.changed_rows == ((100, 1),)
    wide = future_perturbation_probe(
        leaky_global_standardise, _prices(), cutoffs=(100,), seed=0, relative_scale=0.05
    )
    assert wide.changed_rows[0][1] > 50  # global statistics move every earlier row


def test_the_probe_is_deterministic_and_does_not_modify_its_input() -> None:
    prices = _prices()
    snapshot = prices.copy()

    first = future_perturbation_probe(
        leaky_centered_average, prices, cutoffs=CUTOFFS, seed=3, relative_scale=0.05
    )
    second = future_perturbation_probe(
        leaky_centered_average, prices, cutoffs=CUTOFFS, seed=3, relative_scale=0.05
    )

    assert first == second
    assert np.array_equal(prices, snapshot)


# --------------------------------------- why the probe is needed beyond "the IC looks great"


def test_a_label_copy_is_flagged_by_the_identity_screen_but_a_subtle_leak_is_not() -> (
    None
):
    prices = _prices()
    label = forward_log_returns(prices, HORIZON)

    assert affine_label_copy(leaky_forward_return(prices), label) is True
    assert affine_label_copy(np.exp(5.0 * leaky_forward_return(prices)), label) is True
    assert affine_label_copy(3.0 * label - 0.2, label) is True
    # The centred average leaks, yet it is not an identity copy of the label.
    assert affine_label_copy(leaky_centered_average(prices), label) is False
    assert affine_label_copy(causal_momentum(prices), label) is False
    assert affine_label_copy(np.zeros_like(label), label) is False


def test_the_leaky_feature_earns_a_spectacular_ic_the_causal_one_does_not() -> None:
    prices = _prices()
    label = forward_log_returns(prices, HORIZON)

    leaky_ic = rank_ic(leaky_forward_return(prices).ravel(), label.ravel())
    causal_ic = rank_ic(causal_momentum(prices).ravel(), label.ravel())

    assert leaky_ic is not None and leaky_ic > 0.999
    assert causal_ic is not None and abs(causal_ic) < 0.1


# ---------------------------------------------------------------------------- validation


def test_the_probe_validates_its_inputs() -> None:
    prices = _prices()
    builder: Callable[[np.ndarray], np.ndarray] = causal_momentum

    for cutoffs in ((), (-1,), (ROWS - 1,), (ROWS,), (5.0,), (True,)):
        with pytest.raises(SignalMeasurementError):
            future_perturbation_probe(
                builder,
                prices,
                cutoffs=cutoffs,
                seed=0,
                relative_scale=0.05,  # type: ignore[arg-type]
            )
    for scale in (0.0, -0.1, 1.0, float("nan")):
        with pytest.raises(SignalMeasurementError):
            future_perturbation_probe(
                builder, prices, cutoffs=(10,), seed=0, relative_scale=scale
            )
    with pytest.raises(SignalMeasurementError):
        future_perturbation_probe(
            builder, prices[:, 0], cutoffs=(10,), seed=0, relative_scale=0.05
        )
    with pytest.raises(SignalMeasurementError):
        future_perturbation_probe(
            lambda x: x[:-1], prices, cutoffs=(10,), seed=0, relative_scale=0.05
        )
    with pytest.raises(SignalMeasurementError):
        future_perturbation_probe(
            builder, np.zeros((50, 2)), cutoffs=(10,), seed=0, relative_scale=0.05
        )


def test_the_identity_screen_validates_and_needs_enough_rows() -> None:
    with pytest.raises(SignalMeasurementError):
        affine_label_copy(np.zeros(3), np.zeros(4))
    with pytest.raises(SignalMeasurementError):
        affine_label_copy(np.zeros(3), np.zeros(3), tolerance=0.0)
    assert (
        affine_label_copy(np.array([1.0, 2.0]), np.array([1.0, 2.0])) is False
    )  # too few rows


def test_zero_in_future_field_cannot_mask_a_lookahead() -> None:
    """The unrelated future column must not hide a zero-valued future dependency."""
    inputs = np.array([[1.0, 1.0], [3.0, 2.0], [0.0, 4.0], [5.0, 6.0]])

    def future_first_column(values: np.ndarray) -> np.ndarray:
        result = np.zeros((values.shape[0], 1), dtype=np.float64)
        result[:-1, 0] = values[1:, 0]
        return result

    result = future_perturbation_probe(
        future_first_column, inputs, cutoffs=(1,), seed=0, relative_scale=0.05
    )
    assert result.changed_rows == ((1, 1),)
    assert result.leaks is True


def test_probe_rejects_nonfinite_inputs_instead_of_silent_clearance() -> None:
    inputs = np.array([[1.0, 2.0], [np.nan, 3.0], [4.0, 5.0]])
    with pytest.raises(SignalMeasurementError, match="finite"):
        future_perturbation_probe(
            lambda values: values.copy(),
            inputs,
            cutoffs=(0,),
            seed=0,
            relative_scale=0.05,
        )


def test_all_zero_future_rows_still_expose_a_leaky_dependency() -> None:
    """A nonzero prefix does not make the all-zero future safe to ignore."""
    inputs = np.array([[1.0, 2.0], [3.0, 4.0], [0.0, 0.0], [0.0, 0.0]])

    def next_row(values: np.ndarray) -> np.ndarray:
        result = np.zeros((values.shape[0], 1), dtype=np.float64)
        result[:-1, 0] = values[1:, 0]
        return result

    result = future_perturbation_probe(
        next_row, inputs, cutoffs=(1,), seed=0, relative_scale=0.05
    )
    assert result.changed_rows == ((1, 1),)
    assert result.leaks is True


def test_precision_noop_future_cell_refuses_causal_clearance() -> None:
    """A subnormal that rounds to itself cannot be treated as perturbed."""
    tiny = np.nextafter(0.0, 1.0)
    inputs = np.array([[1.0, 1.0], [3.0, 2.0], [tiny, 4.0], [5.0, 6.0]])

    with pytest.raises(SignalMeasurementError, match="future perturbation"):
        future_perturbation_probe(
            lambda values: values.copy(),
            inputs,
            cutoffs=(1,),
            seed=0,
            relative_scale=0.05,
        )
