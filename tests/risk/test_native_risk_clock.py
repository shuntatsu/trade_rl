from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import pytest

from trade_rl.data.native_clock import NativeBarClock, NativeClockKind
from trade_rl.risk.inputs import (
    RollingPortfolioRiskInputsConfig,
    RollingPortfolioRiskInputsProvider,
)
from trade_rl.risk.portfolio import PortfolioRiskConfig, PortfolioRiskModel


@dataclass
class PriceHistory:
    timestamps: np.ndarray
    close: np.ndarray
    available_at: np.ndarray

    @property
    def n_bars(self) -> int:
        return len(self.timestamps)

    @property
    def n_symbols(self) -> int:
        return self.close.shape[1]

    @property
    def periods_per_year(self) -> int:
        raise AssertionError("native risk must not read nominal frequency")


def _history(dates: np.ndarray) -> PriceHistory:
    increments = np.array([[0.03, -0.01], [-0.02, 0.04], [0.05, 0.02], [0.01, -0.03]])
    close = np.exp(np.vstack((np.zeros(2), np.cumsum(increments, axis=0))))
    return PriceHistory(dates, close, np.repeat(dates[:, None], 2, axis=1))


def _provider(
    clock: NativeBarClock, *, annualize: bool = True
) -> RollingPortfolioRiskInputsProvider:
    return RollingPortfolioRiskInputsProvider(
        RollingPortfolioRiskInputsConfig(
            lookback_bars=3, minimum_observations=3, annualize_covariance=annualize
        ),
        native_clock=clock,
    )


def test_calendar_sample_covariance_literal_oracle_and_future_invariance() -> None:
    dates = np.arange("2024-01", "2024-06", dtype="datetime64[M]")
    data = _history(dates)
    provider = _provider(NativeBarClock(NativeClockKind.UTC_MONTH))
    result = provider.inputs(data, index=3)
    values = np.diff(np.log(data.close[:4]), axis=0)
    durations = (31, 29, 31)
    total = sum(durations)
    residuals = values - np.asarray(durations)[:, None] * (values.sum(axis=0) / total)
    exposure = Fraction(total) - Fraction(sum(d * d for d in durations), total)
    expected = residuals.T @ residuals / float(exposure / 365)
    np.testing.assert_allclose(result.covariance, expected, rtol=1e-13, atol=0)
    np.testing.assert_allclose(result.beta, expected[:, 0] / expected[0, 0], rtol=1e-13)
    np.testing.assert_allclose(
        result.stress_losses, np.minimum(np.quantile(values, 0.05, axis=0), 0)
    )
    data.close[-1] *= 10
    data.timestamps[-1] = np.datetime64("2030-06")
    assert provider.inputs(data, index=3).digest == result.digest
    assert provider.identity_digest != provider.config.digest


@pytest.mark.parametrize("days", [3, 7])
def test_equal_intervals_reduce_to_existing_sample_formula(days: int) -> None:
    dates = np.datetime64("2024-01-01") + np.arange(5) * np.timedelta64(days, "D")
    clock = (
        NativeBarClock(NativeClockKind.UTC_WEEK)
        if days == 7
        else NativeBarClock(NativeClockKind.FIXED_ELAPSED, days * 86400, dates[0])
    )
    data = _history(dates)
    values = np.diff(np.log(data.close[:4]), axis=0)
    result = _provider(clock).inputs(data, index=3)
    np.testing.assert_allclose(
        result.covariance,
        np.cov(values, rowvar=False) * float(Fraction(365, days)),
        rtol=1e-13,
    )
    unannualized = _provider(clock, annualize=False).inputs(data, index=3)
    np.testing.assert_allclose(
        unannualized.covariance, np.cov(values, rowvar=False), rtol=1e-13
    )


def test_missing_interval_or_delayed_publication_is_rejected() -> None:
    dates = np.arange("2024-01", "2024-06", dtype="datetime64[M]")
    provider = _provider(NativeBarClock(NativeClockKind.UTC_MONTH))
    data = _history(dates.copy())
    data.timestamps[1] = np.datetime64("2024-03")
    with pytest.raises(ValueError, match="consecutive"):
        provider.inputs(data, index=3)
    data = _history(dates.copy())
    data.available_at[1, 0] = dates[4]
    with pytest.raises(ValueError, match="not available"):
        provider.inputs(data, index=3)
    data = _history(dates.copy())
    data.available_at[1, 0] = np.datetime64("NaT")
    with pytest.raises(ValueError):
        provider.inputs(data, index=3)


def test_clock_identity_is_versioned_and_legacy_identity_is_unchanged() -> None:
    weekly = _provider(NativeBarClock(NativeClockKind.UTC_WEEK))
    fixed = _provider(
        NativeBarClock(
            NativeClockKind.FIXED_ELAPSED, 7 * 86400, np.datetime64("2024-01-01")
        )
    )
    equivalent_fixed = _provider(
        NativeBarClock(
            NativeClockKind.FIXED_ELAPSED,
            7 * 86400,
            np.datetime64("2024-01-01T00:00:00", "ns"),
        )
    )
    assert fixed.identity_digest == equivalent_fixed.identity_digest
    assert weekly.identity_digest != fixed.identity_digest
    legacy = RollingPortfolioRiskInputsProvider(weekly.config)
    assert legacy.identity_digest == legacy.config.digest


def test_monthly_risk_inputs_feed_existing_model_without_nominal_frequency() -> None:
    data = _history(np.arange("2024-01", "2024-06", dtype="datetime64[M]"))
    clock = NativeBarClock(NativeClockKind.UTC_MONTH)
    result = _provider(clock).inputs(data, index=3)
    model = PortfolioRiskModel(PortfolioRiskConfig(volatility_target=0.01))
    constrained = model.constrain(
        np.array([0.6, 0.3]),
        portfolio_value=100_000.0,
        market_notional=np.full(2, 1_000_000_000.0),
        covariance=result.covariance,
        beta=result.beta,
        stress_losses=result.stress_losses,
    )
    assert constrained.was_constrained
    assert "volatility_target" in constrained.reasons
    assert (
        np.sqrt(constrained.weights @ result.covariance @ constrained.weights)
        <= 0.01 + 1e-12
    )
    raw = _provider(clock, annualize=False).inputs(data, index=3)
    np.testing.assert_allclose(
        raw.covariance,
        np.cov(np.diff(np.log(data.close[:4]), axis=0), rowvar=False),
        rtol=1e-13,
    )


@pytest.mark.parametrize("field", ["timestamps", "available_at", "close"])
@pytest.mark.parametrize("owner_view", [False, True])
def test_native_risk_rejects_retained_missingness(field: str, owner_view: bool) -> None:
    data = _history(np.arange("2024-01", "2024-06", dtype="datetime64[M]"))
    array = getattr(data, field)
    mask = np.zeros(array.shape, dtype=bool)
    mask.flat[1] = True
    masked = np.ma.array(array, mask=mask)
    value = masked.astype(masked.dtype).view(np.ndarray) if owner_view else masked
    if owner_view:
        assert isinstance(value.base, np.ma.MaskedArray)
    setattr(data, field, value)
    with pytest.raises(ValueError, match="masked"):
        _provider(NativeBarClock(NativeClockKind.UTC_MONTH)).inputs(data, index=3)


@pytest.mark.parametrize("dtype", [np.complex128, np.bool_, np.str_])
def test_native_risk_rejects_nonreal_price_dtype(dtype) -> None:
    data = _history(np.arange("2024-01", "2024-06", dtype="datetime64[M]"))
    data.close = data.close.astype(dtype)
    with pytest.raises(ValueError, match="real numbers"):
        _provider(NativeBarClock(NativeClockKind.UTC_MONTH)).inputs(data, index=3)


def test_native_risk_exact_publication_and_lookback_boundary() -> None:
    dates = np.arange("2024-01", "2024-06", dtype="datetime64[M]").astype(
        "datetime64[ns]"
    )
    data = _history(dates.copy())
    provider = _provider(NativeBarClock(NativeClockKind.UTC_MONTH))
    expected = provider.inputs(data, index=4)
    data.timestamps[0] = np.datetime64("NaT")
    data.available_at[0] = np.datetime64("NaT")
    data.close[0] = np.nan
    data.available_at[2] = dates[3]
    assert provider.inputs(data, index=4).digest == expected.digest
    data.available_at[4, 0] = dates[4] + np.timedelta64(1, "ns")
    with pytest.raises(ValueError, match="not available"):
        provider.inputs(data, index=4)
    data.available_at[4, 0] = dates[4]
    data.available_at[2, 0] = dates[2] - np.timedelta64(1, "ns")
    with pytest.raises(ValueError, match="not available"):
        provider.inputs(data, index=4)


def test_native_hourly_covariance_exactly_preserves_legacy_numerics() -> None:
    class LegacyHistory(PriceHistory):
        @property
        def periods_per_year(self) -> int:
            return 8760

    dates = np.datetime64("2024-01-01") + np.arange(5) * np.timedelta64(1, "h")
    source = _history(dates)
    data = LegacyHistory(source.timestamps, source.close, source.available_at)
    clock = NativeBarClock(NativeClockKind.FIXED_ELAPSED, 3600, dates[0])
    provider = _provider(clock)
    legacy = RollingPortfolioRiskInputsProvider(provider.config)
    actual = provider.inputs(data, index=3)
    expected = legacy.inputs(data, index=3)
    np.testing.assert_array_equal(actual.covariance, expected.covariance)
    np.testing.assert_array_equal(actual.beta, expected.beta)
    np.testing.assert_array_equal(actual.stress_losses, expected.stress_losses)
    assert actual.provider_digest != expected.provider_digest
