from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketDataset
from trade_rl.rl.actions import AlphaContract
from trade_rl.rl.environment_provider_contract import (
    EnvironmentProviderContractBuilder,
)
from trade_rl.rl.market_inputs import CausalMarketView, MarketInputResolver
from trade_rl.strategies.trend import TrendConfig, TrendStrategy


def market() -> MarketDataset:
    n_bars = 128
    close = np.column_stack(
        [
            np.linspace(100.0, 140.0, n_bars),
            np.linspace(80.0, 110.0, n_bars),
        ]
    )
    open_price = np.vstack([close[0], close[:-1]])
    return MarketDataset(
        dataset_id="a" * 64,
        symbols=("A", "B"),
        timestamps=np.datetime64("2026-01-01T00:00:00", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=np.zeros((n_bars, 2, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=open_price,
        high=np.maximum(open_price, close) + 1.0,
        low=np.minimum(open_price, close) - 1.0,
        close=close,
        volume=np.full((n_bars, 2), 1_000.0),
        funding_rate=np.zeros((n_bars, 2)),
        tradable=np.ones((n_bars, 2), dtype=np.bool_),
        feature_available=np.ones((n_bars, 2, 1), dtype=np.bool_),
        feature_names=("ret",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


class CausalAlphaProvider:
    identity_digest = "1" * 64
    minimum_index = 101

    def predict(self, market: CausalMarketView) -> np.ndarray:
        return np.zeros(len(market.symbols), dtype=np.float64)


class LegacyAlphaProvider:
    artifact_digest = "2" * 64
    minimum_index = 99

    def predict_at(self, dataset: MarketDataset, index: int) -> np.ndarray:
        return np.zeros(dataset.n_symbols, dtype=np.float64)


class FactorProvider:
    artifact_digest = "3" * 64
    n_factors = 2
    minimum_index = 105

    def basis_at(self, dataset: MarketDataset, index: int) -> np.ndarray:
        return np.eye(self.n_factors, dataset.n_symbols, dtype=np.float64)


def callable_alpha_provider(dataset: MarketDataset, index: int) -> np.ndarray:
    return np.zeros(dataset.n_symbols, dtype=np.float64)


def build_contract(
    *,
    dataset: MarketDataset | None = None,
    trend_strategy: TrendStrategy | None = None,
    market_input_resolver: MarketInputResolver | None = None,
    alpha_provider: object | None = None,
    alpha_enabled: bool = False,
    alpha_artifact_digest: str | None = None,
    alpha_contract: AlphaContract | None = None,
    factor_basis: np.ndarray | None = None,
    factor_basis_provider: object | None = None,
    factor_artifact_digest: str | None = None,
    factor_count: int | None = None,
):
    return EnvironmentProviderContractBuilder(
        dataset or market(),
        trend_strategy=trend_strategy,
        market_input_resolver=market_input_resolver,
        alpha_provider=alpha_provider,
        alpha_enabled=alpha_enabled,
        alpha_artifact_digest=alpha_artifact_digest,
        alpha_contract=alpha_contract,
        factor_basis=factor_basis,
        factor_basis_provider=factor_basis_provider,
        factor_artifact_digest=factor_artifact_digest,
        factor_count=factor_count,
    ).build()


def test_default_contract_preserves_default_provider_state() -> None:
    dataset = market()
    contract = build_contract(dataset=dataset)

    assert contract.market_input_resolver is None
    assert isinstance(contract.trend_strategy, TrendStrategy)
    assert contract.alpha_provider is None
    assert contract.alpha_enabled is False
    assert asdict(contract.alpha_contract) == asdict(AlphaContract())
    assert contract.alpha_artifact_digest is None
    assert contract.static_factor_basis is None
    assert contract.factor_basis_provider is None
    assert contract.factor_artifact_digest is None
    assert contract.factor_count == 0
    assert contract.minimum_start_index == contract.trend_strategy.minimum_history_for(
        dataset
    )


def test_explicit_resolver_owns_trend_and_alpha_mode() -> None:
    dataset = market()
    trend = TrendStrategy(
        TrendConfig(fast_lookback=2, base_lookback=4, slow_lookback=8)
    )
    provider = CausalAlphaProvider()
    resolver = MarketInputResolver(
        trend_strategy=trend,
        alpha_provider=provider,
        alpha_enabled=True,
    )

    contract = build_contract(
        dataset=dataset,
        market_input_resolver=resolver,
        alpha_artifact_digest="4" * 64,
    )

    assert contract.market_input_resolver is resolver
    assert contract.trend_strategy is trend
    assert contract.alpha_provider is None
    assert contract.alpha_enabled is True
    assert contract.alpha_artifact_digest == "4" * 64
    assert contract.minimum_start_index == 8


def test_causal_alpha_provider_is_wrapped_compatibly() -> None:
    provider = CausalAlphaProvider()

    contract = build_contract(alpha_provider=provider, alpha_enabled=True)

    assert contract.market_input_resolver is not None
    assert contract.market_input_resolver.alpha_provider is provider
    assert contract.market_input_resolver.alpha_enabled is True
    assert contract.alpha_provider is provider
    assert contract.alpha_artifact_digest == provider.identity_digest
    assert contract.minimum_start_index == provider.minimum_index


def test_explicit_alpha_digest_precedes_provider_digest() -> None:
    provider = LegacyAlphaProvider()

    contract = build_contract(
        alpha_provider=provider,
        alpha_enabled=True,
        alpha_artifact_digest="5" * 64,
    )

    assert contract.alpha_artifact_digest == "5" * 64


def test_static_factor_basis_is_float64_copied_and_digest_bound() -> None:
    basis = np.array([[1.0, -1.0], [0.5, 0.5]], dtype=np.float32)
    expected_payload = tuple(tuple(float(value) for value in row) for row in basis)

    contract = build_contract(factor_basis=basis)

    assert contract.factor_count == 2
    assert contract.static_factor_basis is not None
    assert contract.static_factor_basis.dtype == np.float64
    assert not np.shares_memory(contract.static_factor_basis, basis)
    np.testing.assert_allclose(contract.static_factor_basis, basis)
    assert contract.factor_artifact_digest == content_digest(
        {"schema_version": "static_factor_basis_v1", "value": expected_payload}
    )

    basis[0, 0] = 99.0
    assert contract.static_factor_basis[0, 0] == pytest.approx(1.0)


def test_factor_provider_infers_count_digest_and_minimum_index() -> None:
    provider = FactorProvider()

    contract = build_contract(factor_basis_provider=provider)

    assert contract.factor_basis_provider is provider
    assert contract.factor_count == provider.n_factors
    assert contract.factor_artifact_digest == provider.artifact_digest
    assert contract.minimum_start_index == provider.minimum_index


def test_provider_minimum_index_uses_maximum_in_alpha_then_factor_order() -> None:
    alpha = LegacyAlphaProvider()
    factor = FactorProvider()

    contract = build_contract(
        alpha_provider=alpha,
        factor_basis_provider=factor,
    )

    assert contract.minimum_start_index == factor.minimum_index


def test_resolver_trend_mismatch_preserves_error() -> None:
    explicit = TrendStrategy(
        TrendConfig(fast_lookback=2, base_lookback=4, slow_lookback=8)
    )
    resolver = MarketInputResolver(
        trend_strategy=TrendStrategy(
            TrendConfig(fast_lookback=3, base_lookback=6, slow_lookback=12)
        )
    )

    with pytest.raises(
        ValueError,
        match="market_input_resolver trend differs from trend_strategy",
    ):
        build_contract(trend_strategy=explicit, market_input_resolver=resolver)


def test_enabled_alpha_without_any_provider_preserves_error() -> None:
    with pytest.raises(ValueError, match="alpha_enabled requires an alpha_provider"):
        build_contract(alpha_enabled=True, alpha_artifact_digest="4" * 64)


def test_enabled_alpha_without_digest_preserves_error() -> None:
    with pytest.raises(
        ValueError,
        match="alpha_artifact_digest is required when the component is enabled",
    ):
        build_contract(alpha_provider=callable_alpha_provider, alpha_enabled=True)


def test_invalid_alpha_digest_is_rejected() -> None:
    with pytest.raises(ValueError, match="alpha_artifact_digest"):
        build_contract(
            alpha_provider=LegacyAlphaProvider(),
            alpha_enabled=True,
            alpha_artifact_digest="invalid",
        )


@pytest.mark.parametrize(
    ("basis", "message"),
    [
        (
            np.ones((2, 3)),
            "factor_basis must have shape \\(n_factors, n_symbols\\)",
        ),
        (np.array([[1.0, np.nan]]), "factor_basis must be finite"),
    ],
)
def test_invalid_static_factor_basis_preserves_errors(
    basis: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_contract(factor_basis=basis)


@pytest.mark.parametrize("factor_count", [True, -1, 1.5])
def test_invalid_factor_count_preserves_error(factor_count: object) -> None:
    with pytest.raises(
        ValueError,
        match="factor_count must be a non-negative integer",
    ):
        build_contract(factor_count=factor_count)  # type: ignore[arg-type]


def test_static_factor_count_mismatch_preserves_error() -> None:
    with pytest.raises(ValueError, match="factor_count does not match factor_basis"):
        build_contract(
            factor_basis=np.ones((2, 2)),
            factor_count=3,
        )


def test_enabled_factor_without_digest_preserves_error() -> None:
    with pytest.raises(
        ValueError,
        match="factor_artifact_digest is required when the component is enabled",
    ):
        build_contract(factor_count=1)


@pytest.mark.parametrize("minimum_index", [True, -1, 128, 1.5])
def test_invalid_alpha_provider_minimum_index_preserves_error(
    minimum_index: object,
) -> None:
    provider = LegacyAlphaProvider()
    provider.minimum_index = minimum_index  # type: ignore[assignment]

    with pytest.raises(ValueError, match="alpha_provider minimum_index is invalid"):
        build_contract(alpha_provider=provider)


def test_alpha_minimum_index_is_validated_before_factor_minimum_index() -> None:
    alpha = LegacyAlphaProvider()
    alpha.minimum_index = -1  # type: ignore[assignment]
    factor = FactorProvider()
    factor.minimum_index = -1  # type: ignore[assignment]

    with pytest.raises(ValueError, match="alpha_provider minimum_index is invalid"):
        build_contract(alpha_provider=alpha, factor_basis_provider=factor)
