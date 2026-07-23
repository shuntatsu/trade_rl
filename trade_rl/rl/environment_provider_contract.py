"""Deterministic provider-contract construction for market environments."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketDataset
from trade_rl.domain.common import require_sha256
from trade_rl.rl.actions import AlphaContract
from trade_rl.rl.market_inputs import CausalAlphaProvider, MarketInputResolver
from trade_rl.strategies.trend import TrendStrategy


class AlphaProvider(Protocol):
    """Legacy environment alpha provider with explicit artifact identity."""

    @property
    def artifact_digest(self) -> str: ...

    def predict_at(self, dataset: MarketDataset, index: int) -> np.ndarray: ...


class FactorBasisProvider(Protocol):
    """Causal factor-basis provider with explicit artifact identity."""

    @property
    def artifact_digest(self) -> str: ...

    @property
    def n_factors(self) -> int: ...

    def basis_at(self, dataset: MarketDataset, index: int) -> np.ndarray: ...


AlphaProviderInput = (
    AlphaProvider
    | CausalAlphaProvider
    | Callable[[MarketDataset, int], np.ndarray]
)
FactorBasisProviderInput = FactorBasisProvider | Callable[[MarketDataset, int], np.ndarray]


@dataclass(frozen=True, slots=True)
class EnvironmentProviderContract:
    """Resolved static trend, alpha, and factor provider fields."""

    market_input_resolver: MarketInputResolver | None
    trend_strategy: TrendStrategy
    alpha_provider: AlphaProviderInput | None
    alpha_enabled: bool
    alpha_contract: AlphaContract
    alpha_artifact_digest: str | None
    static_factor_basis: np.ndarray | None
    factor_basis_provider: FactorBasisProviderInput | None
    factor_artifact_digest: str | None
    factor_count: int
    minimum_start_index: int


class EnvironmentProviderContractBuilder:
    """Build provider identity and history requirements without runtime state."""

    def __init__(
        self,
        dataset: MarketDataset,
        *,
        trend_strategy: TrendStrategy | None,
        market_input_resolver: MarketInputResolver | None,
        alpha_provider: AlphaProviderInput | None,
        alpha_enabled: bool,
        alpha_artifact_digest: str | None,
        alpha_contract: AlphaContract | None,
        factor_basis: np.ndarray | None,
        factor_basis_provider: FactorBasisProviderInput | None,
        factor_artifact_digest: str | None,
        factor_count: int | None,
    ) -> None:
        self.dataset = dataset
        self.trend_strategy = trend_strategy
        self.market_input_resolver = market_input_resolver
        self.alpha_provider = alpha_provider
        self.alpha_enabled = alpha_enabled
        self.alpha_artifact_digest = alpha_artifact_digest
        self.alpha_contract = alpha_contract
        self.factor_basis = factor_basis
        self.factor_basis_provider = factor_basis_provider
        self.factor_artifact_digest = factor_artifact_digest
        self.factor_count = factor_count

    def build(self) -> EnvironmentProviderContract:
        resolved_trend = self.trend_strategy or (
            self.market_input_resolver.trend_strategy
            if self.market_input_resolver is not None
            else TrendStrategy()
        )
        market_input_resolver = self.market_input_resolver
        if (
            market_input_resolver is None
            and self.alpha_provider is not None
            and hasattr(self.alpha_provider, "predict")
            and hasattr(self.alpha_provider, "identity_digest")
        ):
            market_input_resolver = MarketInputResolver(
                trend_strategy=resolved_trend,
                alpha_provider=self.alpha_provider,  # type: ignore[arg-type]
                alpha_enabled=bool(self.alpha_enabled),
            )
        if market_input_resolver is not None and self.trend_strategy is not None:
            if market_input_resolver.trend_strategy != self.trend_strategy:
                raise ValueError(
                    "market_input_resolver trend differs from trend_strategy"
                )

        alpha_enabled = (
            market_input_resolver.alpha_enabled
            if market_input_resolver is not None
            else bool(self.alpha_enabled)
        )
        if (
            alpha_enabled
            and self.alpha_provider is None
            and market_input_resolver is None
        ):
            raise ValueError("alpha_enabled requires an alpha_provider")
        alpha_contract = self.alpha_contract or AlphaContract()
        alpha_artifact_digest = self._resolve_provider_digest(
            enabled=alpha_enabled,
            provider=self.alpha_provider,
            explicit=self.alpha_artifact_digest,
            field_name="alpha_artifact_digest",
        )

        static_factor_basis = self._validated_static_basis(self.factor_basis)
        resolved_factor_count = self._resolve_factor_count(
            factor_count=self.factor_count,
            provider=self.factor_basis_provider,
        )
        if static_factor_basis is not None:
            if resolved_factor_count not in (0, static_factor_basis.shape[0]):
                raise ValueError("factor_count does not match factor_basis")
            resolved_factor_count = static_factor_basis.shape[0]
        factor_artifact_digest = self._resolve_provider_digest(
            enabled=resolved_factor_count > 0,
            provider=self.factor_basis_provider,
            explicit=self.factor_artifact_digest,
            field_name="factor_artifact_digest",
            static_payload=(
                None
                if static_factor_basis is None
                else tuple(
                    tuple(float(value) for value in row)
                    for row in static_factor_basis
                )
            ),
        )

        provider_minimums = [resolved_trend.minimum_history_for(self.dataset)]
        for provider_name, provider in (
            ("alpha_provider", self.alpha_provider),
            ("factor_basis_provider", self.factor_basis_provider),
        ):
            if provider is None:
                continue
            minimum_index = getattr(provider, "minimum_index", 0)
            if (
                isinstance(minimum_index, bool)
                or not isinstance(minimum_index, int)
                or minimum_index < 0
                or minimum_index >= self.dataset.n_bars
            ):
                raise ValueError(f"{provider_name} minimum_index is invalid")
            provider_minimums.append(minimum_index)

        return EnvironmentProviderContract(
            market_input_resolver=market_input_resolver,
            trend_strategy=resolved_trend,
            alpha_provider=self.alpha_provider,
            alpha_enabled=alpha_enabled,
            alpha_contract=alpha_contract,
            alpha_artifact_digest=alpha_artifact_digest,
            static_factor_basis=static_factor_basis,
            factor_basis_provider=self.factor_basis_provider,
            factor_artifact_digest=factor_artifact_digest,
            factor_count=resolved_factor_count,
            minimum_start_index=max(provider_minimums),
        )

    @staticmethod
    def _resolve_provider_digest(
        *,
        enabled: bool,
        provider: object | None,
        explicit: str | None,
        field_name: str,
        static_payload: object | None = None,
    ) -> str | None:
        if not enabled:
            return None
        resolved = explicit
        if resolved is None and provider is not None:
            candidate = getattr(provider, "artifact_digest", None)
            if not isinstance(candidate, str):
                candidate = getattr(provider, "identity_digest", None)
            if isinstance(candidate, str):
                resolved = candidate
        if resolved is None and static_payload is not None:
            resolved = content_digest(
                {"schema_version": "static_factor_basis_v1", "value": static_payload}
            )
        if resolved is None:
            raise ValueError(f"{field_name} is required when the component is enabled")
        require_sha256(resolved, field=field_name)
        return resolved

    def _validated_static_basis(self, value: np.ndarray | None) -> np.ndarray | None:
        if value is None:
            return None
        basis = np.asarray(value, dtype=np.float64)
        if basis.ndim != 2 or basis.shape[1] != self.dataset.n_symbols:
            raise ValueError("factor_basis must have shape (n_factors, n_symbols)")
        if not np.isfinite(basis).all():
            raise ValueError("factor_basis must be finite")
        return basis.copy()

    @staticmethod
    def _resolve_factor_count(
        *,
        factor_count: int | None,
        provider: object | None,
    ) -> int:
        resolved = factor_count
        if resolved is None and provider is not None:
            candidate = getattr(provider, "n_factors", None)
            if isinstance(candidate, int) and not isinstance(candidate, bool):
                resolved = candidate
        if resolved is None:
            return 0
        if isinstance(resolved, bool) or not isinstance(resolved, int) or resolved < 0:
            raise ValueError("factor_count must be a non-negative integer")
        return resolved


__all__ = [
    "AlphaProvider",
    "EnvironmentProviderContract",
    "EnvironmentProviderContractBuilder",
    "FactorBasisProvider",
]
