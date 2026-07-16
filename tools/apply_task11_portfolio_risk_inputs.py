from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"missing Task 11 anchor in {path}: {old[:160]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def add_tests() -> None:
    target = ROOT / "tests/risk/test_portfolio_risk_inputs.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        r'''from __future__ import annotations

from dataclasses import replace

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.risk.portfolio import PortfolioRiskConfig, PortfolioRiskModel
from trade_rl.rl.actions import ActionSpec
from trade_rl.rl.environment import ResidualMarketEnv, ResidualMarketEnvConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.trend import TrendConfig, TrendStrategy


def _dataset(*, future_shift: float = 0.0) -> MarketDataset:
    n_bars = 180
    phase = np.arange(n_bars, dtype=np.float64)
    returns = np.column_stack(
        (
            0.0003 + 0.0015 * np.sin(phase / 7.0),
            0.0002 + 0.0010 * np.sin(phase / 7.0 + 0.4),
            -0.0001 + 0.0012 * np.cos(phase / 11.0),
        )
    )
    close = 100.0 * np.exp(np.cumsum(returns, axis=0))
    if future_shift:
        close[121:] *= 1.0 + future_shift
    open_price = np.vstack((close[0], close[:-1]))
    return MarketDataset(
        dataset_id="a" * 64,
        symbols=("BTCUSDT", "ETHUSDT", "BNBUSDT"),
        timestamps=np.datetime64("2026-01-01T00:00:00", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=np.zeros((n_bars, 3, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=open_price,
        high=np.maximum(open_price, close) * 1.001,
        low=np.minimum(open_price, close) * 0.999,
        close=close,
        volume=np.full((n_bars, 3), 1_000_000.0),
        funding_rate=np.zeros((n_bars, 3)),
        tradable=np.ones((n_bars, 3), dtype=np.bool_),
        feature_available=np.ones((n_bars, 3, 1), dtype=np.bool_),
        feature_names=("ret",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def test_rolling_portfolio_risk_inputs_are_causal_and_finite() -> None:
    from trade_rl.risk.inputs import (
        RollingPortfolioRiskInputsConfig,
        RollingPortfolioRiskInputsProvider,
    )

    provider = RollingPortfolioRiskInputsProvider(
        RollingPortfolioRiskInputsConfig(
            lookback_bars=60,
            minimum_observations=30,
            benchmark_index=0,
            stress_quantile=0.05,
        )
    )
    first = provider.inputs(_dataset(), index=120)
    shifted = provider.inputs(_dataset(future_shift=0.5), index=120)

    np.testing.assert_allclose(first.covariance, shifted.covariance)
    np.testing.assert_allclose(first.beta, shifted.beta)
    np.testing.assert_allclose(first.stress_losses, shifted.stress_losses)
    assert first.as_of_index == 120
    assert first.covariance.shape == (3, 3)
    assert first.beta.shape == (3,)
    assert first.stress_losses.shape == (3,)
    assert np.isfinite(first.covariance).all()
    assert np.isfinite(first.beta).all()
    assert np.all(first.stress_losses <= 0.0)
    assert len(first.digest) == 64
    assert len(provider.identity_digest) == 64


def test_rolling_portfolio_risk_inputs_reject_insufficient_history() -> None:
    import pytest

    from trade_rl.risk.inputs import (
        RollingPortfolioRiskInputsConfig,
        RollingPortfolioRiskInputsProvider,
    )

    provider = RollingPortfolioRiskInputsProvider(
        RollingPortfolioRiskInputsConfig(
            lookback_bars=40,
            minimum_observations=30,
        )
    )
    with pytest.raises(ValueError, match="insufficient"):
        provider.inputs(_dataset(), index=20)


def test_environment_wires_causal_inputs_into_advanced_portfolio_risk() -> None:
    from trade_rl.risk.inputs import RollingPortfolioRiskInputsProvider

    dataset = _dataset()
    risk = PortfolioRiskModel(
        PortfolioRiskConfig(
            volatility_target=0.01,
            max_abs_beta=0.15,
            max_stress_loss=0.0005,
        )
    )
    env = ResidualMarketEnv(
        dataset,
        trend_strategy=TrendStrategy(
            TrendConfig(fast_lookback=4, base_lookback=8, slow_lookback=16)
        ),
        action_spec=ActionSpec(
            mode="target_weight",
            alpha_enabled=False,
            risk_tilt_enabled=False,
            target_weight_count=3,
        ),
        portfolio_risk=risk,
        config=ResidualMarketEnvConfig(
            episode_bars=8,
            decision_every=1,
            initial_capital=100_000.0,
            execution_cost=ExecutionCostConfig.zero(),
        ),
    )
    assert isinstance(env.portfolio_risk_inputs_provider, RollingPortfolioRiskInputsProvider)
    env.reset(options={"start_idx": 120, "initial_state_mode": "cash"})
    constrained = env._constrain_target(np.array([0.6, 0.3, -0.1]), env.hybrid)

    assert constrained.was_constrained is True
    assert any(reason.startswith("portfolio:") for reason in constrained.reasons)
    payload = env._digest_payload()
    assert isinstance(payload["portfolio_risk_inputs_digest"], str)
    assert len(payload["portfolio_risk_inputs_digest"]) == 64
    future = replace(dataset, close=_dataset(future_shift=0.5).close)
    provider = env.portfolio_risk_inputs_provider
    assert provider is not None
    baseline = provider.inputs(dataset, index=120)
    changed = provider.inputs(future, index=120)
    np.testing.assert_allclose(baseline.covariance, changed.covariance)
''',
        encoding="utf-8",
    )


def add_implementation() -> None:
    module = ROOT / "trade_rl/risk/inputs.py"
    module.write_text(
        r'''"""Causal rolling inputs for advanced portfolio-risk constraints."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256


class PortfolioRiskDataset(Protocol):
    n_bars: int
    n_symbols: int
    periods_per_year: int
    close: np.ndarray


class PortfolioRiskInputsProvider(Protocol):
    @property
    def identity_digest(self) -> str: ...

    @property
    def minimum_index(self) -> int: ...

    def inputs(self, dataset: PortfolioRiskDataset, *, index: int) -> PortfolioRiskInputs: ...


def _readonly_array(value: np.ndarray, *, shape: tuple[int, ...], field: str) -> np.ndarray:
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
        if not math.isfinite(self.stress_quantile) or not 0.0 < self.stress_quantile < 0.5:
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
            raise ValueError("portfolio risk closes must be positive and symbol aligned")
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
''',
        encoding="utf-8",
    )

    replace_once(
        "trade_rl/risk/portfolio.py",
        '''class PortfolioRiskModel:
    """Apply deterministic concentration, liquidity and scenario constraints."""
''',
        '''class PortfolioRiskModel:
    """Apply deterministic concentration, liquidity and scenario constraints."""

    @property
    def requires_advanced_inputs(self) -> bool:
        return any(
            value is not None
            for value in (
                self.config.volatility_target,
                self.config.max_abs_beta,
                self.config.max_stress_loss,
            )
        )
''',
    )

    replace_once(
        "trade_rl/rl/environment.py",
        '''from trade_rl.risk.portfolio import PortfolioRiskModel
from trade_rl.risk.pretrade import PreTradeRisk, RiskConstrainedTarget
''',
        '''from trade_rl.risk.inputs import (
    PortfolioRiskInputsProvider,
    RollingPortfolioRiskInputsProvider,
)
from trade_rl.risk.portfolio import PortfolioRiskModel
from trade_rl.risk.pretrade import PreTradeRisk, RiskConstrainedTarget
''',
    )
    replace_once(
        "trade_rl/rl/environment.py",
        '''        portfolio_risk: PortfolioRiskModel | None = None,
        normalizer: ObservationNormalizer | None = None,
''',
        '''        portfolio_risk: PortfolioRiskModel | None = None,
        portfolio_risk_inputs_provider: PortfolioRiskInputsProvider | None = None,
        normalizer: ObservationNormalizer | None = None,
''',
    )
    replace_once(
        "trade_rl/rl/environment.py",
        '''        self.pre_trade_risk = pre_trade_risk or PreTradeRisk()
        self.portfolio_risk = portfolio_risk or PortfolioRiskModel()
        self.normalizer = normalizer
''',
        '''        self.pre_trade_risk = pre_trade_risk or PreTradeRisk()
        self.portfolio_risk = portfolio_risk or PortfolioRiskModel()
        resolved_risk_provider = portfolio_risk_inputs_provider
        if self.portfolio_risk.requires_advanced_inputs and resolved_risk_provider is None:
            resolved_risk_provider = RollingPortfolioRiskInputsProvider()
        self.portfolio_risk_inputs_provider = resolved_risk_provider
        if resolved_risk_provider is not None:
            require_sha256(
                resolved_risk_provider.identity_digest,
                field="portfolio_risk_inputs_provider.identity_digest",
            )
            minimum_index = resolved_risk_provider.minimum_index
            if (
                isinstance(minimum_index, bool)
                or not isinstance(minimum_index, int)
                or minimum_index < 0
                or minimum_index >= dataset.n_bars
            ):
                raise ValueError("portfolio risk inputs minimum_index is invalid")
            self._minimum_start_index = max(self._minimum_start_index, minimum_index)
        self.normalizer = normalizer
''',
    )
    replace_once(
        "trade_rl/rl/environment.py",
        '''            "portfolio_risk": asdict(self.portfolio_risk.config),
            "pre_trade_risk": asdict(self.pre_trade_risk.config),
''',
        '''            "portfolio_risk": asdict(self.portfolio_risk.config),
            "portfolio_risk_inputs_digest": (
                None
                if self.portfolio_risk_inputs_provider is None
                else self.portfolio_risk_inputs_provider.identity_digest
            ),
            "pre_trade_risk": asdict(self.pre_trade_risk.config),
''',
    )
    replace_once(
        "trade_rl/rl/environment.py",
        '''        portfolio = self.portfolio_risk.constrain(
            pretrade.weights,
            portfolio_value=max(book.portfolio_value, 1e-12),
            market_notional=self._market_notional(self.current_index),
        )
''',
        '''        risk_inputs = None
        if self.portfolio_risk.requires_advanced_inputs:
            provider = self.portfolio_risk_inputs_provider
            if provider is None:
                raise RuntimeError("advanced portfolio risk requires a causal input provider")
            risk_inputs = provider.inputs(self.dataset, index=self.current_index)
        portfolio = self.portfolio_risk.constrain(
            pretrade.weights,
            portfolio_value=max(book.portfolio_value, 1e-12),
            market_notional=self._market_notional(self.current_index),
            covariance=None if risk_inputs is None else risk_inputs.covariance,
            beta=None if risk_inputs is None else risk_inputs.beta,
            stress_losses=None if risk_inputs is None else risk_inputs.stress_losses,
        )
''',
    )

    replace_once(
        "trade_rl/risk/__init__.py",
        '''from trade_rl.risk.pretrade import (
''',
        '''from trade_rl.risk.inputs import (
    PortfolioRiskInputs,
    PortfolioRiskInputsProvider,
    RollingPortfolioRiskInputsConfig,
    RollingPortfolioRiskInputsProvider,
)
from trade_rl.risk.pretrade import (
''',
    )
    replace_once(
        "trade_rl/risk/__init__.py",
        '''__all__ = ["PreTradeRisk", "PreTradeRiskConfig", "RiskConstrainedTarget"]
''',
        '''__all__ = [
    "PortfolioRiskInputs",
    "PortfolioRiskInputsProvider",
    "PreTradeRisk",
    "PreTradeRiskConfig",
    "RiskConstrainedTarget",
    "RollingPortfolioRiskInputsConfig",
    "RollingPortfolioRiskInputsProvider",
]
''',
    )


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"tests", "implementation"}:
        raise SystemExit("usage: apply_task11_portfolio_risk_inputs.py tests|implementation")
    if sys.argv[1] == "tests":
        add_tests()
    else:
        add_implementation()
