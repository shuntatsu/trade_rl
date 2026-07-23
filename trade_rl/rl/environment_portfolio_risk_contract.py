"""Portfolio-risk input construction contract for the market environment."""

from __future__ import annotations

from dataclasses import dataclass

from trade_rl.data.market import MarketDataset
from trade_rl.domain.common import require_sha256
from trade_rl.risk.inputs import (
    PortfolioRiskInputsProvider,
    RollingPortfolioRiskInputsProvider,
)
from trade_rl.risk.portfolio import PortfolioRiskModel


@dataclass(frozen=True, slots=True)
class EnvironmentPortfolioRiskContract:
    """Validated portfolio-risk collaborators and causal history boundary."""

    portfolio_risk: PortfolioRiskModel
    inputs_provider: PortfolioRiskInputsProvider | None
    minimum_start_index: int


class EnvironmentPortfolioRiskContractBuilder:
    """Resolve portfolio-risk inputs without environment mutable state."""

    def __init__(
        self,
        dataset: MarketDataset,
        *,
        portfolio_risk: PortfolioRiskModel | None,
        inputs_provider: PortfolioRiskInputsProvider | None,
    ) -> None:
        self.dataset = dataset
        self.portfolio_risk = portfolio_risk
        self.inputs_provider = inputs_provider

    def build(
        self,
        *,
        minimum_start_index: int,
    ) -> EnvironmentPortfolioRiskContract:
        portfolio_risk = self.portfolio_risk or PortfolioRiskModel()
        provider = self.inputs_provider
        if portfolio_risk.requires_advanced_inputs and provider is None:
            provider = RollingPortfolioRiskInputsProvider()
        if provider is not None:
            require_sha256(
                provider.identity_digest,
                field="portfolio_risk_inputs_provider.identity_digest",
            )
            provider_minimum = provider.minimum_index
            if (
                isinstance(provider_minimum, bool)
                or not isinstance(provider_minimum, int)
                or provider_minimum < 0
                or provider_minimum >= self.dataset.n_bars
            ):
                raise ValueError("portfolio risk inputs minimum_index is invalid")
            minimum_start_index = max(
                minimum_start_index,
                provider_minimum,
            )
        return EnvironmentPortfolioRiskContract(
            portfolio_risk=portfolio_risk,
            inputs_provider=provider,
            minimum_start_index=minimum_start_index,
        )


__all__ = [
    "EnvironmentPortfolioRiskContract",
    "EnvironmentPortfolioRiskContractBuilder",
]
