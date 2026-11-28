"""Deterministic causality fuzz over every non-cross-asset feature implementation.

For each decision row t < cut, outputs computed on the full series must equal
those computed on the prefix ``series[:cut]`` and on a series whose rows ``>= cut``
were mutated. The visible Ichimoku cloud is also compared with a hand-computed
oracle and the divergence detector must be prefix/mutation invariant and never
backdate an event before its pivot could be confirmed. Synthetic data only.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from trade_rl.data.contracts import (
    FeatureAlignment,
    FeatureKind,
    FeatureSpec,
    IchimokuPeriods,
)
from trade_rl.data.features import calculate_feature_events
from trade_rl.data.features.divergence import DivergenceConfig, detect_divergences
from trade_rl.data.features.indicators.ichimoku_cloud import visible_ichimoku_cloud
from trade_rl.data.source import RawMarketSeries

_CROSS_ASSET = frozenset(
    {
        FeatureKind.RELATIVE_RETURN_TO_BTC,
        FeatureKind.ROLLING_CORRELATION_TO_BTC,
        FeatureKind.ROLLING_BETA_TO_BTC,
        FeatureKind.CROSS_SECTIONAL_MOMENTUM_RANK,
        FeatureKind.CROSS_ASSET_DISPERSION,
    }
)
_ICHIMOKU_CLOUD = (
    FeatureKind.ICHIMOKU_CLOUD_POSITION,
    FeatureKind.ICHIMOKU_CLOUD_THICKNESS,
    FeatureKind.ICHIMOKU_CLOUD_EDGE_DISTANCE,
    FeatureKind.ICHIMOKU_CLOUD_TWIST,
)
_ICHIMOKU_LINES = (
    FeatureKind.ICHIMOKU_TENKAN_DISTANCE,
    FeatureKind.ICHIMOKU_KIJUN_DISTANCE,
)
_N = 220
_SEED = 20261128
_GAPS = (35, 120)
_LEGACY_LOOKBACKS = (5, 14, 26, 52)
_FEATURE_TRIALS = 2
_CUTS = 4
_VISIBLE = FeatureAlignment.VISIBLE_DISPLACED_CLOUD


def _series(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    r = rng.standard_t(4, n) * 0.01
    close = 100 * np.exp(np.cumsum(r))
    open_ = close * np.exp(rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.004, n)))
    funding_available = np.zeros(n, dtype=bool)
    funding_available[::8] = True
    funding = np.zeros(n)
    funding[funding_available] = rng.normal(0, 1e-4, int(funding_available.sum()))
    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": rng.lognormal(7, 0.5, n),
        "funding": funding,
        "funding_available": funding_available,
    }


def _mutate(s: dict[str, np.ndarray], cut: int) -> dict[str, np.ndarray]:
    # Independent of test order: the mutation depends only on the cut row.
    rng = np.random.default_rng((_SEED, cut))
    m = {key: value.copy() for key, value in s.items()}
    scale = rng.uniform(0.3, 3.0)
    for key in ("open", "high", "low", "close"):
        m[key][cut:] *= scale
    m["high"][cut:] *= 1.05
    m["low"][cut:] *= 0.95
    m["volume"][cut:] *= rng.uniform(0.1, 10)
    m["funding"][cut:] = rng.normal(0, 1e-3, len(m["funding"]) - cut)
    return m


def _present(n: int, *, gaps: bool = True) -> np.ndarray:
    mask = np.ones(n, dtype=bool)
    for gap in _GAPS if gaps else ():
        if gap < n:
            mask[gap] = False
    return mask


def _features(spec: FeatureSpec, s: dict[str, np.ndarray], stop: int | None = None):
    window = slice(None, stop)
    n = len(s["close"][window])
    return calculate_feature_events(
        spec,
        open_price=s["open"][window],
        high=s["high"][window],
        low=s["low"][window],
        close=s["close"][window],
        volume=s["volume"][window],
        funding_rate=s["funding"][window],
        funding_available=s["funding_available"][window],
        # Explicit Ichimoku profiles reset warm-up at unusable rows, so they get
        # gaps. Implicit kinds run gap-free: legacy ADX, for a usable segment of
        # exactly 2 * lookback - 1 rows, reports its first row unavailable in a
        # prefix yet available once later rows exist (availability only, never a
        # different value). That pre-existing behavior is outside this test.
        row_present=_present(n, gaps=spec.ichimoku_periods is not None),
        active=np.ones(n, dtype=bool),
    )


def _specs() -> dict[FeatureKind, list[FeatureSpec]]:
    """Every non-cross kind, implicit (legacy) and explicit where valid."""
    specs: dict[FeatureKind, list[FeatureSpec]] = {}
    for kind in FeatureKind:
        if kind in _CROSS_ASSET:
            continue
        for lookback in _LEGACY_LOOKBACKS:
            try:
                spec = FeatureSpec(kind.value, kind, lookback=lookback)
            except ValueError:
                continue  # e.g. implicit chikou declares exactly lookback=26
            specs.setdefault(kind, []).append(spec)
    for periods in (
        IchimokuPeriods(9, 26, 52, 0),
        IchimokuPeriods(5, 8, 13, 0),
    ):
        for kind in (*_ICHIMOKU_LINES, *_ICHIMOKU_CLOUD):
            specs[kind].append(
                FeatureSpec(f"u{kind.value}", kind, ichimoku_periods=periods)
            )
    for periods in (
        IchimokuPeriods(9, 26, 52, 26),
        IchimokuPeriods(9, 26, 52, 25),
        IchimokuPeriods(3, 5, 8, 5),
    ):
        for kind in (*_ICHIMOKU_CLOUD, FeatureKind.ICHIMOKU_CHIKOU_DISTANCE):
            specs[kind].append(
                FeatureSpec(
                    f"v{kind.value}",
                    kind,
                    ichimoku_periods=periods,
                    alignment=_VISIBLE,
                )
            )
    return specs


@pytest.fixture(scope="module")
def trials() -> list[tuple[dict[str, np.ndarray], list[int]]]:
    rng = np.random.default_rng(_SEED)
    result = []
    for _ in range(_FEATURE_TRIALS):
        s = _series(rng, _N)
        cuts = sorted(
            int(c) for c in rng.choice(np.arange(60, _N - 5), _CUTS, replace=False)
        )
        result.append((s, cuts))
    return result


def test_every_non_cross_kind_has_implicit_and_explicit_coverage():
    specs = _specs()
    assert set(specs) == set(FeatureKind) - _CROSS_ASSET
    new = {
        FeatureKind.ICHIMOKU_CLOUD_EDGE_DISTANCE: ("implicit", "unshifted", "visible"),
        FeatureKind.ICHIMOKU_CLOUD_TWIST: ("implicit", "unshifted", "visible"),
        FeatureKind.ICHIMOKU_CHIKOU_DISTANCE: ("implicit", "visible"),
    }
    for kind, modes in new.items():
        seen = set()
        for spec in specs[kind]:
            if spec.ichimoku_periods is None:
                seen.add("implicit")
            elif spec.alignment is _VISIBLE:
                seen.add("visible")
            else:
                seen.add("unshifted")
        assert seen == set(modes)


def test_feature_prefix_truncation_and_future_mutation_invariance(trials):
    specs = _specs()
    checks = 0
    for s, cuts in trials:
        for kind_specs in specs.values():
            for spec in kind_specs:
                full = _features(spec, s)
                for cut in cuts:
                    prefix = _features(spec, s, cut)
                    mutated = _features(spec, _mutate(s, cut))
                    for index in range(3):
                        checks += 2
                        np.testing.assert_array_equal(
                            full[index][:cut], prefix[index], err_msg=f"{spec.name}"
                        )
                        np.testing.assert_array_equal(
                            full[index][:cut],
                            mutated[index][:cut],
                            err_msg=f"{spec.name} cut={cut}",
                        )
    assert checks > 1000


def _raw(s: dict[str, np.ndarray], stop: int | None = None) -> RawMarketSeries:
    window = slice(None, stop)
    n = len(s["close"][window])
    start = datetime(2024, 1, 1, tzinfo=UTC)
    stamps = np.array(
        [
            np.datetime64((start + timedelta(hours=i)).replace(tzinfo=None), "ns")
            for i in range(n)
        ]
    )
    return RawMarketSeries(
        timestamps=stamps,
        open=s["open"][window],
        high=s["high"][window],
        low=s["low"][window],
        close=s["close"][window],
        volume=s["volume"][window],
        funding_rate=s["funding"][window],
        funding_available=s["funding_available"][window],
        tradable=np.ones(n, dtype=bool),
    )


@pytest.mark.parametrize("displacement", [26, 25, 5])
def test_visible_cloud_is_causal_and_matches_hand_oracle(trials, displacement):
    for s, cuts in trials:
        cloud = visible_ichimoku_cloud(_raw(s), displacement=displacement)
        for cut in cuts:
            prefix = visible_ichimoku_cloud(_raw(s, cut), displacement=displacement)
            mutated = visible_ichimoku_cloud(
                _raw(_mutate(s, cut)), displacement=displacement
            )
            for field in ("top", "bottom", "span_a", "span_b", "ready"):
                np.testing.assert_array_equal(
                    getattr(cloud, field)[:cut], getattr(prefix, field)
                )
                np.testing.assert_array_equal(
                    getattr(cloud, field)[:cut], getattr(mutated, field)[:cut]
                )
        high, low = s["high"], s["low"]
        first = 51 + displacement
        assert np.flatnonzero(cloud.ready)[0] == first
        for t in range(first, _N):
            source = t - displacement

            def mid(period: int) -> float:
                window = slice(source - period + 1, source + 1)
                return (high[window].max() + low[window].min()) / 2

            a, b = (mid(9) + mid(26)) / 2, mid(52)
            assert np.isclose(cloud.span_a[t], a, rtol=1e-12, atol=0)
            assert np.isclose(cloud.span_b[t], b, rtol=1e-12, atol=0)
            assert cloud.top[t] == max(cloud.span_a[t], cloud.span_b[t])


def test_new_cloud_features_match_independent_visible_oracle(trials):
    s, _ = trials[0]
    high, low, close = s["high"], s["low"], s["close"]
    for displacement in (26, 25):
        periods = IchimokuPeriods(9, 26, 52, displacement)
        outputs = {
            kind: _features(
                FeatureSpec("f", kind, ichimoku_periods=periods, alignment=_VISIBLE),
                s,
            )
            for kind in (
                FeatureKind.ICHIMOKU_CLOUD_EDGE_DISTANCE,
                FeatureKind.ICHIMOKU_CLOUD_TWIST,
                FeatureKind.ICHIMOKU_CHIKOU_DISTANCE,
            )
        }
        for t in range(_N):
            source = t - displacement
            window_usable = t >= 51 + displacement and all(
                _present(_N)[source - 51 : t + 1]
            )
            edge, twist, chikou = (
                outputs[FeatureKind.ICHIMOKU_CLOUD_EDGE_DISTANCE],
                outputs[FeatureKind.ICHIMOKU_CLOUD_TWIST],
                outputs[FeatureKind.ICHIMOKU_CHIKOU_DISTANCE],
            )
            assert edge[1][t] == window_usable == twist[1][t]
            lag_usable = t >= displacement and all(
                _present(_N)[t - displacement : t + 1]
            )
            assert chikou[1][t] == lag_usable
            if lag_usable:
                assert np.isclose(
                    chikou[0][t],
                    (close[t] - close[t - displacement]) / close[t],
                    rtol=1e-12,
                )
            if not window_usable:
                continue

            def mid(period: int) -> float:
                window = slice(source - period + 1, source + 1)
                return (high[window].max() + low[window].min()) / 2

            a, b = (mid(9) + mid(26)) / 2, mid(52)
            top, bottom = max(a, b), min(a, b)
            c = close[t]
            expected = (
                (c - top) / c if c > top else (c - bottom) / c if c < bottom else 0.0
            )
            assert np.isclose(edge[0][t], expected, rtol=1e-12, atol=0)
            assert np.isclose(twist[0][t], (a - b) / c, rtol=1e-12, atol=0)


def _indicators(s: dict[str, np.ndarray], stop: int | None = None):
    out = {}
    for name, kind, lookback in (
        ("rsi", FeatureKind.RSI, 14),
        ("macd", FeatureKind.MACD_HISTOGRAM, 26),
        ("cci", FeatureKind.CCI, 20),
    ):
        window = slice(None, stop)
        n = len(s["close"][window])
        values, valid, _ = calculate_feature_events(
            FeatureSpec(name, kind, lookback=lookback),
            open_price=s["open"][window],
            high=s["high"][window],
            low=s["low"][window],
            close=s["close"][window],
            volume=s["volume"][window],
            funding_rate=s["funding"][window],
            funding_available=s["funding_available"][window],
            row_present=np.ones(n, dtype=bool),
            active=np.ones(n, dtype=bool),
        )
        out[name] = np.where(valid, values, np.nan).astype(np.float64)
    return out


@pytest.mark.parametrize(
    "config",
    [
        DivergenceConfig(),
        DivergenceConfig(pivot_period=3, min_indicators=1),
        DivergenceConfig(pivot_period=8, pivot_source="high_low", min_indicators=1),
        DivergenceConfig(pivot_period=5, validate_lines=False, min_indicators=2),
    ],
)
def test_divergence_prefix_mutation_invariance_and_no_backdating(trials, config):
    events_seen = 0
    for s, cuts in trials:
        full = detect_divergences(
            high=s["high"],
            low=s["low"],
            close=s["close"],
            indicators=_indicators(s),
            config=config,
        )
        for cut in cuts:
            prefix = detect_divergences(
                high=s["high"][:cut],
                low=s["low"][:cut],
                close=s["close"][:cut],
                indicators=_indicators(s, cut),
                config=config,
            )
            m = _mutate(s, cut)
            mutated = detect_divergences(
                high=m["high"],
                low=m["low"],
                close=m["close"],
                indicators=_indicators(m),
                config=config,
            )
            for name in ("bull_regular", "bear_regular", "bull_hidden", "bear_hidden"):
                np.testing.assert_array_equal(
                    getattr(full, name)[:cut], getattr(prefix, name)
                )
                np.testing.assert_array_equal(
                    getattr(full, name)[:cut], getattr(mutated, name)[:cut]
                )
        for event in full.events:
            events_seen += 1
            assert event.observed_at >= event.pivot_at + config.pivot_period
            assert event.anchor_at < event.observed_at
    assert events_seen > 0
