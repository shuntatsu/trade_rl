"""Causal rolling inputs for advanced portfolio-risk constraints."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256


class PortfolioRiskDataset(Protocol):
    @property
    def n_bars(self) -> int: ...

    @property
    def n_symbols(self) -> int: ...

    @property
    def periods_per_year(self) -> int: ...

    @property
    def close(self) -> np.ndarray: ...


class PortfolioRiskInputsProvider(Protocol):
    @property
    def identity_digest(self) -> str: ...

    @property
    def minimum_index(self) -> int: ...

    def inputs(
        self, dataset: PortfolioRiskDataset, *, index: int
    ) -> PortfolioRiskInputs: ...


def _readonly_array(
    value: np.ndarray, *, shape: tuple[int, ...], field: str
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).copy(order="C")
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{field} must be finite with shape {shape}")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class PortfolioRiskInputs:
    covariance: np.ndarray
    beta: np.ndarray
    stress_losses: np.ndarray
    as_of_index: int
    provider_digest: str
    observation_count: int
    schema_version: str = "portfolio_risk_inputs_v1"
    digest: str = ""

    def __post_init__(self) -> None:
        beta = np.asarray(self.beta, dtype=np.float64).reshape(-1)
        n_symbols = int(beta.size)
        if n_symbols <= 0:
            raise ValueError("portfolio risk inputs require at least one symbol")
        covariance = _readonly_array(
            self.covariance,
            shape=(n_symbols, n_symbols),
            field="covariance",
        )
        beta = _readonly_array(beta, shape=(n_symbols,), field="beta")
        stress = _readonly_array(
            self.stress_losses,
            shape=(n_symbols,),
            field="stress_losses",
        )
        if np.any(stress > 0.0):
            raise ValueError("stress_losses must be non-positive return shocks")
        if isinstance(self.as_of_index, bool) or not isinstance(self.as_of_index, int):
            raise ValueError("as_of_index must be an integer")
        if self.as_of_index < 0:
            raise ValueError("as_of_index must be non-negative")
        if (
            isinstance(self.observation_count, bool)
            or not isinstance(self.observation_count, int)
            or self.observation_count < 2
        ):
            raise ValueError("observation_count must be at least two")
        require_sha256(self.provider_digest, field="provider_digest")
        if self.schema_version != "portfolio_risk_inputs_v1":
            raise ValueError("unsupported portfolio risk inputs schema")
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "stress_losses", stress)
        expected = content_digest(self.digest_payload())
        if self.digest and self.digest != expected:
            raise ValueError("portfolio risk inputs digest mismatch")
        object.__setattr__(self, "digest", expected)

    def digest_payload(self) -> dict[str, object]:
        return {
            "as_of_index": self.as_of_index,
            "beta": tuple(float(value) for value in self.beta),
            "covariance": tuple(
                tuple(float(value) for value in row) for row in self.covariance
            ),
            "observation_count": self.observation_count,
            "provider_digest": self.provider_digest,
            "schema_version": self.schema_version,
            "stress_losses": tuple(float(value) for value in self.stress_losses),
        }


@dataclass(frozen=True, slots=True)
class RollingPortfolioRiskInputsConfig:
    lookback_bars: int = 96
    minimum_observations: int = 20
    benchmark_index: int = 0
    stress_quantile: float = 0.05
    annualize_covariance: bool = True
    schema_version: str = "rolling_portfolio_risk_inputs_v1"

    def __post_init__(self) -> None:
        for name, value in (
            ("lookback_bars", self.lookback_bars),
            ("minimum_observations", self.minimum_observations),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f"{name} must be an integer of at least two")
        if self.minimum_observations > self.lookback_bars:
            raise ValueError("minimum_observations cannot exceed lookback_bars")
        if (
            isinstance(self.benchmark_index, bool)
            or not isinstance(self.benchmark_index, int)
            or self.benchmark_index < 0
        ):
            raise ValueError("benchmark_index must be a non-negative integer")
        if (
            not math.isfinite(self.stress_quantile)
            or not 0.0 < self.stress_quantile < 0.5
        ):
            raise ValueError("stress_quantile must be within (0, 0.5)")
        if not isinstance(self.annualize_covariance, bool):
            raise ValueError("annualize_covariance must be a boolean")
        if self.schema_version != "rolling_portfolio_risk_inputs_v1":
            raise ValueError("unsupported rolling portfolio risk inputs schema")

    @property
    def digest(self) -> str:
        return content_digest(asdict(self))


class RollingPortfolioRiskInputsProvider:
    """Compute risk inputs from completed close-to-close returns through one index."""

    def __init__(self, config: RollingPortfolioRiskInputsConfig | None = None) -> None:
        self.config = config or RollingPortfolioRiskInputsConfig()

    @property
    def identity_digest(self) -> str:
        return self.config.digest

    @property
    def minimum_index(self) -> int:
        return self.config.minimum_observations

    def inputs(
        self,
        dataset: PortfolioRiskDataset,
        *,
        index: int,
    ) -> PortfolioRiskInputs:
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("portfolio risk index must be an integer")
        if not 0 <= index < dataset.n_bars:
            raise ValueError("portfolio risk index is outside the dataset")
        if not 0 <= self.config.benchmark_index < dataset.n_symbols:
            raise ValueError("benchmark_index is outside dataset symbols")
        start = max(1, index - self.config.lookback_bars + 1)
        prices = np.asarray(dataset.close[start - 1 : index + 1], dtype=np.float64)
        if prices.shape[1:] != (dataset.n_symbols,) or np.any(prices <= 0.0):
            raise ValueError(
                "portfolio risk closes must be positive and symbol aligned"
            )
        log_returns = np.diff(np.log(prices), axis=0)
        if (
            log_returns.shape[0] < self.config.minimum_observations
            or not np.isfinite(log_returns).all()
        ):
            raise ValueError("portfolio risk inputs have insufficient finite history")
        centered = log_returns - np.mean(log_returns, axis=0, keepdims=True)
        covariance = centered.T @ centered / float(log_returns.shape[0] - 1)
        if self.config.annualize_covariance:
            covariance = covariance * float(dataset.periods_per_year)
        benchmark_variance = float(
            covariance[self.config.benchmark_index, self.config.benchmark_index]
        )
        if benchmark_variance <= 1e-18:
            raise ValueError("benchmark variance is insufficient for beta")
        beta = covariance[:, self.config.benchmark_index] / benchmark_variance
        stress = np.minimum(
            np.quantile(log_returns, self.config.stress_quantile, axis=0),
            0.0,
        )
        return PortfolioRiskInputs(
            covariance=covariance,
            beta=beta,
            stress_losses=stress,
            as_of_index=index,
            provider_digest=self.identity_digest,
            observation_count=int(log_returns.shape[0]),
        )


__all__ = [
    "PortfolioRiskInputs",
    "PortfolioRiskInputsProvider",
    "RollingPortfolioRiskInputsConfig",
    "RollingPortfolioRiskInputsProvider",
]
