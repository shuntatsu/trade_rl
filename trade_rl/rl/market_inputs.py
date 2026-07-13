"""Shared causal Trend and Alpha input resolution for training and Serving."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Protocol

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketDataset
from trade_rl.domain.common import require_sha256
from trade_rl.strategies.trend import TrendStrategy, TrendTargets


def _readonly_copy(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value).copy(order="C")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class CausalMarketView:
    """Copied policy-safe market prefix ending at one decision index."""

    dataset_id: str
    symbols: tuple[str, ...]
    timestamps: np.ndarray
    features: np.ndarray
    global_features: np.ndarray
    tradable: np.ndarray
    symbol_active: np.ndarray
    information_available: np.ndarray
    feature_available: np.ndarray
    feature_staleness: np.ndarray
    feature_names: tuple[str, ...]
    global_feature_names: tuple[str, ...]

    @property
    def current_index(self) -> int:
        return int(self.timestamps.shape[0] - 1)

    @classmethod
    def from_dataset(cls, dataset: MarketDataset, index: int) -> CausalMarketView:
        if not 0 <= index < dataset.n_bars:
            raise IndexError("causal market view index is outside the dataset")
        symbol_active = dataset.symbol_active
        information_available = dataset.information_available
        feature_staleness = dataset.feature_staleness
        assert symbol_active is not None
        assert information_available is not None
        assert feature_staleness is not None
        stop = index + 1
        observable_tradable = dataset.tradable[:stop] & information_available[:stop]
        return cls(
            dataset_id=dataset.dataset_id,
            symbols=dataset.symbols,
            timestamps=_readonly_copy(dataset.timestamps[:stop]),
            features=_readonly_copy(dataset.features[:stop]),
            global_features=_readonly_copy(dataset.global_features[:stop]),
            tradable=_readonly_copy(observable_tradable),
            symbol_active=_readonly_copy(symbol_active[:stop]),
            information_available=_readonly_copy(information_available[:stop]),
            feature_available=_readonly_copy(dataset.feature_available[:stop]),
            feature_staleness=_readonly_copy(feature_staleness[:stop]),
            feature_names=dataset.feature_names,
            global_feature_names=dataset.global_feature_names,
        )


class CausalAlphaProvider(Protocol):
    """Alpha provider that receives no market rows after the decision index."""

    @property
    def identity_digest(self) -> str: ...

    def predict(self, market: CausalMarketView) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class MarketInputResolver:
    """Resolve deterministic Trend and optional Alpha through one causal path."""

    trend_strategy: TrendStrategy = field(default_factory=TrendStrategy)
    alpha_provider: CausalAlphaProvider | None = None
    alpha_enabled: bool = False

    def __post_init__(self) -> None:
        if self.alpha_enabled and self.alpha_provider is None:
            raise ValueError("enabled alpha requires a causal alpha provider")
        if self.alpha_provider is not None:
            require_sha256(
                self.alpha_provider.identity_digest,
                field="alpha_provider.identity_digest",
            )

    @property
    def digest(self) -> str:
        return content_digest(
            {
                "schema": "market_input_resolver_v1",
                "trend": asdict(self.trend_strategy.config),
                "alpha_enabled": self.alpha_enabled,
                "alpha_provider_digest": (
                    self.alpha_provider.identity_digest
                    if self.alpha_provider is not None
                    else None
                ),
            }
        )

    def resolve(
        self,
        dataset: MarketDataset,
        index: int,
    ) -> tuple[TrendTargets, np.ndarray]:
        trends = self.trend_strategy.targets(dataset, index)
        if not self.alpha_enabled:
            return trends, np.zeros(dataset.n_symbols, dtype=np.float64)

        provider = self.alpha_provider
        assert provider is not None
        raw = provider.predict(CausalMarketView.from_dataset(dataset, index))
        alpha = np.asarray(raw, dtype=np.float64).reshape(-1)
        if alpha.shape != (dataset.n_symbols,) or not np.isfinite(alpha).all():
            raise ValueError("causal alpha provider returned an invalid vector")
        eligible = dataset.eligibility_mask(index, require_features=True)
        alpha = alpha.copy()
        alpha[~eligible] = 0.0
        gross = float(np.abs(alpha).sum())
        return trends, (alpha / gross if gross > 1.0 else alpha)
