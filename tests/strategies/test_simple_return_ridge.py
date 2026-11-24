"""Raw simple-return Ridge fit and causal unit-separation oracles."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from tools.tier_d_nonrl.task18_recipe import build_task18_protocol
from trade_rl.data.market import MarketDataset
from trade_rl.strategies.forecasts.ridge import fit_ridge_forecast
from trade_rl.strategies.forecasts.supervised import build_causal_forecast_training_set


def _market(n: int = 180) -> MarketDataset:
    t = np.arange(n, dtype=np.float64)
    # Deliberately nonconstant returns and unequal eligible symbol counts.
    close = np.stack(
        (
            100 + t * 0.4 + 3 * np.sin(t / 8),
            220 - np.minimum(t, 180) * 0.3 + 4 * np.cos(t / 11),
        ),
        axis=1,
    )
    features = np.stack(
        (
            np.stack((np.sin(t / 4), t / 100, np.ones(n)), axis=1),
            np.stack((np.cos(t / 6), (t + 11) / 100, np.ones(n)), axis=1),
        ),
        axis=1,
    ).astype(np.float32)
    available = np.ones_like(features, dtype=np.bool_)
    available[:7, 1, :] = False
    available[19, 0, 0] = False
    timestamps = np.datetime64("2025-01-01T00:00:00", "ns") + np.arange(
        n
    ) * np.timedelta64(1, "h")
    return MarketDataset(
        dataset_id="c" * 64,
        symbols=("ARTIFICIAL-A", "ARTIFICIAL-B"),
        timestamps=timestamps,
        features=features,
        global_features=np.zeros((n, 1), dtype=np.float32),
        open=close.copy(),
        high=close.copy(),
        low=close.copy(),
        close=close.copy(),
        volume=np.full_like(close, 1_000_000.0),
        funding_rate=np.zeros_like(close),
        tradable=np.ones_like(close, dtype=np.bool_),
        feature_available=available,
        feature_names=("signal", "trend", "constant"),
        global_feature_names=("synthetic_global",),
        periods_per_year=8760,
    )


def _oracle(dataset, cutoff, horizon, alpha):
    # Independent direct selection on an hourly fixture, and augmented weighted
    # least squares rather than the implementation's normal-equation solver.
    xs, ys, ws = [], [], []
    times = dataset.timestamps
    per_symbol = []
    for symbol in range(2):
        rows = []
        for start in range(dataset.n_bars - horizon):
            end = start + horizon
            if times[end] >= cutoff:
                continue
            if not all(dataset.feature_available[start, symbol, :]):
                continue
            if not all(dataset.information_available[[start, end], symbol]):
                continue
            if any(dataset.available_at[[start, end], symbol] >= cutoff):
                continue
            rows.append(
                (
                    dataset.features[start, symbol, :],
                    float(dataset.close[end, symbol])
                    / float(dataset.close[start, symbol])
                    - 1.0,
                )
            )
        per_symbol.append(rows)
    total = sum(map(len, per_symbol))
    for rows in per_symbol:
        for x, y in rows:
            xs.append(x)
            ys.append(y)
            ws.append(total / len(per_symbol) / len(rows))
    x, y, w = np.asarray(xs, dtype=float), np.asarray(ys), np.asarray(ws)
    mean = np.average(x, weights=w, axis=0)
    scale = np.sqrt(np.average((x - mean) ** 2, weights=w, axis=0))
    scale[scale <= 1e-12] = 1.0
    z = (x - mean) / scale
    design = np.column_stack((np.ones(len(y)), z))
    n_features = dataset.n_features
    penalty = np.column_stack((np.zeros(n_features), np.eye(n_features))) * math.sqrt(
        alpha
    )
    coefficients, *_ = np.linalg.lstsq(
        np.vstack((design * np.sqrt(w)[:, None], penalty)),
        np.concatenate((y * np.sqrt(w), np.zeros(n_features))),
        rcond=None,
    )
    return (
        mean,
        scale,
        coefficients,
        np.average((y - np.average(y, weights=w)) ** 2, weights=w),
        len(y),
    )


@pytest.mark.parametrize(
    "configuration", build_task18_protocol().configurations, ids=lambda c: c.name
)
@pytest.mark.parametrize("activation", (90, 120, 150))
def test_direct_simple_ridge_matches_independent_oracle_for_all_twelve_vintages(
    configuration, activation
):
    from trade_rl.strategies.forecasts.simple_return import fit_simple_return_ridge

    data = _market()
    cutoff = data.timestamps[activation] - np.timedelta64(1, "ns")
    model = fit_simple_return_ridge(
        data,
        feature_indices=(0, 1, 2),
        fit_symbol_indices=(0, 1),
        fit_cutoff=cutoff,
        horizon_hours=configuration.horizon_hours,
        alpha=configuration.alpha,
    )
    mean, scale, coeffs, variance, count = _oracle(
        data, cutoff, configuration.horizon_hours, configuration.alpha
    )
    assert model.return_unit == "expected_simple_return"
    assert model.valuation_basis == "same_close_price_return"
    assert model.fit_prefix_marginal_variance == pytest.approx(variance, abs=1e-14)
    assert model.n_samples == count
    np.testing.assert_allclose(model.feature_mean, mean, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(model.feature_scale, scale, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(model.coefficients, coeffs[1:], rtol=1e-10, atol=1e-12)
    assert model.intercept == pytest.approx(coeffs[0], abs=1e-12)
    expected = coeffs[0] + ((data.features[activation, 0] - mean) / scale) @ coeffs[1:]
    kwargs = {
        "decision_time": data.timestamps[activation],
        "feature_available": data.feature_available[activation, 0],
        "feature_names": data.feature_names,
    }
    # A linear projection may extrapolate below the economic lower bound.
    # Do not rescue it by clipping or changing a fitted coefficient.
    if expected < -1.0:
        with pytest.raises(ValueError, match="at least -1"):
            model.predict_simple_return(data.features[activation, 0], **kwargs)
    else:
        assert model.predict_simple_return(
            data.features[activation, 0], **kwargs
        ) == pytest.approx(expected, abs=1e-12)
    assert model.fit_cutoff == cutoff
    assert len(model.training_scope_sha256) == 64


def test_strict_publication_and_information_masks_are_not_legacy_log_semantics():
    from trade_rl.strategies.forecasts.simple_return import fit_simple_return_ridge

    data = _market()
    cutoff = data.timestamps[110]
    available_at = data.available_at.copy()
    information = data.information_available.copy()
    # Both exact-cutoff publication and late-but-before-cutoff rows must be
    # excluded from the archived strict simple-return selection.
    for row, publication in ((30, cutoff), (35, data.timestamps[80])):
        available_at[row, 0] = publication
        information[row, 0] = False
    information[45, 1] = False
    changed = replace(
        data, available_at=available_at, information_available=information
    )
    kwargs = dict(
        feature_indices=(0, 1, 2),
        fit_symbol_indices=(0, 1),
        fit_cutoff=cutoff,
        horizon_hours=24,
    )
    direct = fit_simple_return_ridge(changed, **kwargs)
    log_rows = build_causal_forecast_training_set(changed, **kwargs)
    expected = _oracle(changed, cutoff, 24, 1.0)
    assert direct.n_samples == expected[-1] < log_rows.n_samples
    np.testing.assert_allclose(
        direct.coefficients, expected[2][1:], atol=1e-12, rtol=1e-10
    )


def test_future_only_mutation_preserves_raw_simple_fit_identity_and_parameters():
    from trade_rl.strategies.forecasts.simple_return import fit_simple_return_ridge

    data = _market()
    activation = 120
    close = data.close.copy()
    close[activation:] *= 9
    features = data.features.copy()
    features[activation:] *= -7
    changed = replace(
        data, close=close, open=close, high=close, low=close, features=features
    )
    kwargs = dict(
        feature_indices=(0, 1, 2),
        fit_symbol_indices=(0, 1),
        fit_cutoff=data.timestamps[activation] - np.timedelta64(1, "ns"),
        horizon_hours=72,
    )
    a = fit_simple_return_ridge(data, **kwargs)
    b = fit_simple_return_ridge(changed, **kwargs)
    assert a.training_scope_sha256 == b.training_scope_sha256
    assert a.intercept == b.intercept
    assert a.fit_prefix_marginal_variance == b.fit_prefix_marginal_variance
    np.testing.assert_array_equal(a.coefficients, b.coefficients)
    assert not a.coefficients.flags.writeable


def test_simple_fit_is_not_log_fit_or_exponentiated_log_prediction():
    from trade_rl.strategies.forecasts.ridge import RidgeForecastStrategy
    from trade_rl.strategies.forecasts.simple_return import fit_simple_return_ridge

    data = _market()
    kwargs = dict(
        feature_indices=(0, 1, 2),
        fit_symbol_indices=(0, 1),
        fit_cutoff=data.timestamps[120],
        horizon_hours=72,
    )
    simple = fit_simple_return_ridge(data, **kwargs)
    log = fit_ridge_forecast(data, **kwargs)
    x = data.features[120, 0]
    direct = simple.predict_simple_return(
        x,
        decision_time=data.timestamps[120],
        feature_available=data.feature_available[120, 0],
        feature_names=data.feature_names,
    )
    assert abs(direct - math.expm1(log.predict(x))) > 1e-5
    assert not hasattr(simple, "predict")
    with pytest.raises((TypeError, ValueError), match="RidgeForecastModel|log"):
        RidgeForecastStrategy(simple, entry_threshold=0.01, exit_threshold=0.0)


@pytest.mark.parametrize("alpha", (True, 0, -1, float("nan"), float("inf"), "1"))
def test_invalid_alpha_refused(alpha):
    from trade_rl.strategies.forecasts.simple_return import fit_simple_return_ridge

    with pytest.raises(ValueError, match="alpha"):
        fit_simple_return_ridge(
            _market(),
            feature_indices=(0,),
            fit_cutoff=_market().timestamps[100],
            horizon_hours=24,
            alpha=alpha,
        )


def test_prediction_rejects_pre_cutoff_and_unavailable_or_nonreal_features():
    from trade_rl.strategies.forecasts.simple_return import fit_simple_return_ridge

    data = _market()
    model = fit_simple_return_ridge(
        data,
        feature_indices=(0, 1, 2),
        fit_cutoff=data.timestamps[100],
        horizon_hours=24,
    )
    x = data.features[100, 0]
    ok = np.ones(3, dtype=bool)
    with pytest.raises(ValueError, match="cutoff"):
        model.predict_simple_return(
            x,
            decision_time=data.timestamps[99],
            feature_available=ok,
            feature_names=data.feature_names,
        )
    with pytest.raises(ValueError, match="available"):
        model.predict_simple_return(
            x,
            decision_time=data.timestamps[100],
            feature_available=np.array([True, False, True]),
            feature_names=data.feature_names,
        )
    for invalid in (
        np.ones(3, dtype=bool),
        np.ones(3, dtype=complex),
        np.ma.array(x, mask=[False, True, False]),
        np.array([1.0, float("nan"), 1.0]),
    ):
        with pytest.raises(ValueError):
            model.predict_simple_return(
                invalid,
                decision_time=data.timestamps[100],
                feature_available=ok,
                feature_names=data.feature_names,
            )
    with pytest.raises(ValueError):
        model.predict_simple_return(
            x,
            decision_time=data.timestamps[100],
            feature_available=np.ones(3, dtype=int),
            feature_names=data.feature_names,
        )


def test_full_size_task18_twelve_simple_fits_match_frozen_dimensions_and_independent_oracle():
    from trade_rl.strategies.forecasts.simple_return import fit_simple_return_ridge

    protocol = build_task18_protocol()
    n = protocol.dataset.row_count
    t = np.arange(n, dtype=np.float64)
    # Bounded artificial prices, six arbitrary features, exactly the archived
    # dimension/cutoff schedule, but no historical dataset or empirical result.
    close = np.column_stack(
        (100 + 8 * np.sin(t / 40) + t / 2000, 200 + 5 * np.cos(t / 55) - t / 4000)
    )
    features = np.stack(
        [
            np.column_stack(
                (
                    np.sin(t / 4 + s),
                    np.cos(t / 7 + s),
                    np.sin(t / 30 + s),
                    np.cos(t / 24 + s),
                    t / n,
                    np.ones(n),
                )
            )
            for s in (0, 1)
        ],
        axis=1,
    ).astype(np.float32)
    available = np.ones_like(features, dtype=np.bool_)
    available[:31, 1, :] = False
    available[101, 0, 2] = False
    dataset = replace(
        _market(n=protocol.dataset.row_count),
        close=close,
        open=close,
        high=close,
        low=close,
        mark_price=close,
        features=features,
        feature_available=available,
        feature_names=tuple(f"artificial_{i}" for i in range(6)),
        feature_staleness_hours=np.zeros_like(features),
        feature_missing_reason=None,
        feature_staleness=None,
    )
    assert dataset.n_bars == 9504
    assert dataset.n_symbols == 2 and dataset.n_features == 6
    scope_digests = {}
    count = 0
    for activation in protocol.index_plan.vintage_start_indices:
        cutoff = dataset.timestamps[activation] - np.timedelta64(1, "ns")
        for configuration in protocol.configurations:
            horizon, alpha = configuration.horizon_hours, configuration.alpha
            model = fit_simple_return_ridge(
                dataset,
                feature_indices=tuple(range(6)),
                fit_symbol_indices=(0, 1),
                fit_cutoff=cutoff,
                horizon_hours=horizon,
                alpha=alpha,
            )
            mean, scale, coeffs, variance, rows = _oracle(
                dataset, cutoff, horizon, alpha
            )
            assert model.n_samples == rows
            np.testing.assert_allclose(model.feature_mean, mean, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(
                model.feature_scale, scale, rtol=1e-12, atol=1e-12
            )
            np.testing.assert_allclose(
                model.coefficients, coeffs[1:], rtol=1e-9, atol=1e-12
            )
            assert model.intercept == pytest.approx(coeffs[0], abs=1e-12)
            assert model.fit_prefix_marginal_variance == pytest.approx(
                variance, abs=1e-14
            )
            key = (activation, horizon)
            assert (
                scope_digests.setdefault(key, model.training_scope_sha256)
                == model.training_scope_sha256
            )
            count += 1
    assert count == protocol.index_plan.ridge_vintage_count == 12
    assert len(set(scope_digests.values())) == 6


def test_prediction_requires_the_actual_ordered_feature_layout_and_no_unit_coercion():
    from trade_rl.strategies.forecasts.simple_return import fit_simple_return_ridge

    data = _market()
    model = fit_simple_return_ridge(
        data, feature_indices=(2, 0), fit_cutoff=data.timestamps[100], horizon_hours=24
    )
    kwargs = dict(
        decision_time=data.timestamps[100], feature_available=np.ones(3, dtype=bool)
    )
    value = model.predict_simple_return(
        data.features[100, 0], feature_names=data.feature_names, **kwargs
    )
    assert math.isfinite(value)
    with pytest.raises(ValueError, match="feature_names"):
        model.predict_simple_return(
            data.features[100, 0],
            feature_names=tuple(reversed(data.feature_names)),
            **kwargs,
        )
    with pytest.raises(ValueError, match="feature_names"):
        model.predict_simple_return(
            data.features[100, 0],
            feature_names=("signal", "signal", "constant"),
            **kwargs,
        )


def test_missing_explicit_fit_symbol_and_lossy_cutoff_are_refused():
    from trade_rl.strategies.forecasts.simple_return import fit_simple_return_ridge

    data = _market()
    available = data.feature_available.copy()
    available[:, 1, :] = False
    changed = replace(
        data,
        feature_available=available,
        feature_staleness=None,
        feature_staleness_hours=None,
        feature_missing_reason=None,
    )
    kwargs = dict(feature_indices=(0, 1), fit_symbol_indices=(0, 1), horizon_hours=24)
    with pytest.raises(ValueError, match="fit symbol has no eligible training rows"):
        fit_simple_return_ridge(changed, fit_cutoff=data.timestamps[100], **kwargs)
    with pytest.raises(ValueError, match="fit_cutoff"):
        fit_simple_return_ridge(
            data,
            fit_cutoff=np.datetime64("1970-01-01T00:00:00.000000000500", "ps"),
            **kwargs,
        )
