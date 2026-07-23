"""Construction-time dependency resolution for ``ResidualMarketEnv``."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketDataset
from trade_rl.domain.common import require_sha256
from trade_rl.risk.inputs import (
    PortfolioRiskInputsProvider,
    RollingPortfolioRiskInputsProvider,
)
from trade_rl.risk.portfolio import PortfolioRiskModel
from trade_rl.risk.pretrade import PreTradeRisk
from trade_rl.rl.actions import (
    ACTION_SCHEMA,
    ActionMode,
    ActionSpec,
    ActionValidationMode,
    AlphaContract,
    BaselineResidualComposer,
)
from trade_rl.rl.environment_config import ResidualMarketEnvConfig
from trade_rl.rl.episode import minimum_reward_start_index
from trade_rl.rl.market_inputs import MarketInputResolver
from trade_rl.rl.rewards import RewardTracker
from trade_rl.strategies.trend import TrendStrategy

SignalProvider = object | Callable[[MarketDataset, int], np.ndarray]


@dataclass(frozen=True, slots=True)
class EnvironmentDependencyRequest:
    dataset: MarketDataset
    trend_strategy: TrendStrategy | None
    market_input_resolver: MarketInputResolver | None
    alpha_provider: SignalProvider | None
    alpha_enabled: bool
    alpha_artifact_digest: str | None
    alpha_contract: AlphaContract | None
    factor_basis: np.ndarray | None
    factor_basis_provider: SignalProvider | None
    factor_artifact_digest: str | None
    factor_count: int | None
    action_spec: ActionSpec | None
    composer: BaselineResidualComposer | None
    pre_trade_risk: PreTradeRisk | None
    portfolio_risk: PortfolioRiskModel | None
    portfolio_risk_inputs_provider: PortfolioRiskInputsProvider | None
    config: ResidualMarketEnvConfig | None


@dataclass(frozen=True, slots=True)
class EnvironmentDependencies:
    trend_strategy: TrendStrategy
    market_input_resolver: MarketInputResolver | None
    alpha_enabled: bool
    alpha_artifact_digest: str | None
    alpha_contract: AlphaContract
    static_factor_basis: np.ndarray | None
    factor_basis_provider: SignalProvider | None
    factor_artifact_digest: str | None
    action_spec: ActionSpec
    action_names: tuple[str, ...]
    action_spec_digest: str
    composer: BaselineResidualComposer
    pre_trade_risk: PreTradeRisk
    portfolio_risk: PortfolioRiskModel
    portfolio_risk_inputs_provider: PortfolioRiskInputsProvider | None
    config: ResidualMarketEnvConfig
    reward_tracker: RewardTracker
    nominal_episode_bars: int
    nominal_decision_bars: int
    resolved_decision_hours: float
    minimum_start_index: int


class EnvironmentDependencyResolver:
    """Resolve immutable constructor dependencies without mutating an environment."""

    @staticmethod
    def resolve(request: EnvironmentDependencyRequest) -> EnvironmentDependencies:
        dataset = request.dataset
        resolved_trend = request.trend_strategy or (
            request.market_input_resolver.trend_strategy
            if request.market_input_resolver is not None
            else TrendStrategy()
        )
        market_input_resolver = request.market_input_resolver
        if (
            market_input_resolver is None
            and request.alpha_provider is not None
            and hasattr(request.alpha_provider, "predict")
            and hasattr(request.alpha_provider, "identity_digest")
        ):
            market_input_resolver = MarketInputResolver(
                trend_strategy=resolved_trend,
                alpha_provider=request.alpha_provider,
                alpha_enabled=bool(request.alpha_enabled),
            )
        if market_input_resolver is not None and request.trend_strategy is not None:
            if market_input_resolver.trend_strategy != request.trend_strategy:
                raise ValueError(
                    "market_input_resolver trend differs from trend_strategy"
                )
        alpha_enabled = (
            market_input_resolver.alpha_enabled
            if market_input_resolver is not None
            else bool(request.alpha_enabled)
        )
        if (
            alpha_enabled
            and request.alpha_provider is None
            and market_input_resolver is None
        ):
            raise ValueError("alpha_enabled requires an alpha_provider")
        alpha_contract = request.alpha_contract or AlphaContract()
        alpha_artifact_digest = _resolve_provider_digest(
            enabled=alpha_enabled,
            provider=request.alpha_provider,
            explicit=request.alpha_artifact_digest,
            field_name="alpha_artifact_digest",
        )
        static_factor_basis = _validated_static_basis(dataset, request.factor_basis)
        resolved_factor_count = _resolve_factor_count(
            factor_count=request.factor_count,
            provider=request.factor_basis_provider,
        )
        if static_factor_basis is not None:
            if resolved_factor_count not in (0, static_factor_basis.shape[0]):
                raise ValueError("factor_count does not match factor_basis")
            resolved_factor_count = static_factor_basis.shape[0]
        factor_artifact_digest = _resolve_provider_digest(
            enabled=resolved_factor_count > 0,
            provider=request.factor_basis_provider,
            explicit=request.factor_artifact_digest,
            field_name="factor_artifact_digest",
            static_payload=(
                None
                if static_factor_basis is None
                else tuple(
                    tuple(float(value) for value in row) for row in static_factor_basis
                )
            ),
        )
        provider_minimums = [resolved_trend.minimum_history_for(dataset)]
        for provider_name, provider in (
            ("alpha_provider", request.alpha_provider),
            ("factor_basis_provider", request.factor_basis_provider),
        ):
            if provider is None:
                continue
            minimum_index = getattr(provider, "minimum_index", 0)
            if (
                isinstance(minimum_index, bool)
                or not isinstance(minimum_index, int)
                or minimum_index < 0
                or minimum_index >= dataset.n_bars
            ):
                raise ValueError(f"{provider_name} minimum_index is invalid")
            provider_minimums.append(minimum_index)
        minimum_start_index = max(provider_minimums)
        composer = request.composer or BaselineResidualComposer()
        pre_trade_risk = request.pre_trade_risk or PreTradeRisk()
        portfolio_risk = request.portfolio_risk or PortfolioRiskModel()
        resolved_risk_provider = request.portfolio_risk_inputs_provider
        if portfolio_risk.requires_advanced_inputs and resolved_risk_provider is None:
            resolved_risk_provider = RollingPortfolioRiskInputsProvider()
        if resolved_risk_provider is not None:
            require_sha256(
                resolved_risk_provider.identity_digest,
                field="portfolio_risk_inputs_provider.identity_digest",
            )
            risk_minimum_index = resolved_risk_provider.minimum_index
            if (
                isinstance(risk_minimum_index, bool)
                or not isinstance(risk_minimum_index, int)
                or risk_minimum_index < 0
                or risk_minimum_index >= dataset.n_bars
            ):
                raise ValueError("portfolio risk inputs minimum_index is invalid")
            minimum_start_index = max(minimum_start_index, risk_minimum_index)
        config = request.config or ResidualMarketEnvConfig()
        if pre_trade_risk.config.max_gross > config.execution_cost.max_leverage:
            raise ValueError("pre-trade max_gross cannot exceed execution max_leverage")
        if config.random_initial_gross > pre_trade_risk.config.max_gross:
            raise ValueError("random_initial_gross cannot exceed pre-trade max_gross")
        action_spec = request.action_spec
        if action_spec is None:
            action_spec = ActionSpec(
                alpha_enabled=alpha_enabled,
                n_factors=resolved_factor_count,
                validation_mode=config.action_validation_mode,
            )
        if action_spec.alpha_enabled != alpha_enabled:
            raise ValueError("action_spec alpha mode does not match environment")
        if action_spec.n_factors != resolved_factor_count:
            raise ValueError("action_spec factor count does not match environment")
        if (
            action_spec.mode is ActionMode.TARGET_WEIGHT
            and action_spec.target_weight_count != dataset.n_symbols
        ):
            raise ValueError("target weight count does not match dataset symbols")
        action_names = action_spec.names_for_symbols(dataset.symbols)
        nominal_episode_bars = config.resolve_nominal_episode_bars(dataset)
        nominal_decision_bars = config.resolve_nominal_decision_bars(dataset)
        if nominal_decision_bars > nominal_episode_bars:
            raise ValueError("decision interval cannot exceed episode duration")
        reward_config = config.resolved_reward_config()
        resolved_decision_hours = (
            nominal_decision_bars * dataset.bar_hours
            if config.decision_every is not None
            else config.decision_hours
        )
        if config.episode_hour_choices and any(
            choice + 1e-12 < resolved_decision_hours
            for choice in config.episode_hour_choices
        ):
            raise ValueError(
                "episode_hour_choices cannot be shorter than the resolved "
                "decision interval"
            )
        reward_tracker = RewardTracker(
            reward_config,
            decision_hours=resolved_decision_hours,
        )
        if (
            config.require_full_reward_preroll
            and reward_config.baseline_underperformance_weight > 0.0
        ):
            minimum_start_index = minimum_reward_start_index(
                dataset,
                signal_minimum=minimum_start_index,
                window_hours=reward_config.baseline_window_hours,
            )
        return EnvironmentDependencies(
            trend_strategy=resolved_trend,
            market_input_resolver=market_input_resolver,
            alpha_enabled=alpha_enabled,
            alpha_artifact_digest=alpha_artifact_digest,
            alpha_contract=alpha_contract,
            static_factor_basis=static_factor_basis,
            factor_basis_provider=request.factor_basis_provider,
            factor_artifact_digest=factor_artifact_digest,
            action_spec=action_spec,
            action_names=action_names,
            action_spec_digest=_action_spec_digest(action_spec, action_names),
            composer=composer,
            pre_trade_risk=pre_trade_risk,
            portfolio_risk=portfolio_risk,
            portfolio_risk_inputs_provider=resolved_risk_provider,
            config=config,
            reward_tracker=reward_tracker,
            nominal_episode_bars=nominal_episode_bars,
            nominal_decision_bars=nominal_decision_bars,
            resolved_decision_hours=resolved_decision_hours,
            minimum_start_index=minimum_start_index,
        )


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


def _validated_static_basis(
    dataset: MarketDataset,
    value: np.ndarray | None,
) -> np.ndarray | None:
    if value is None:
        return None
    basis = np.asarray(value, dtype=np.float64)
    if basis.ndim != 2 or basis.shape[1] != dataset.n_symbols:
        raise ValueError("factor_basis must have shape (n_factors, n_symbols)")
    if not np.isfinite(basis).all():
        raise ValueError("factor_basis must be finite")
    return basis.copy()


def _resolve_factor_count(*, factor_count: int | None, provider: object | None) -> int:
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


def _action_spec_digest(
    action_spec: ActionSpec,
    action_names: tuple[str, ...],
) -> str:
    return content_digest(
        {
            "schema_version": ACTION_SCHEMA,
            "alpha_enabled": action_spec.alpha_enabled,
            "mode": ActionMode(action_spec.mode).value,
            "risk_tilt_enabled": action_spec.risk_tilt_enabled,
            "n_factors": action_spec.n_factors,
            "names": action_names,
            "target_weight_count": action_spec.target_weight_count,
            "validation_mode": ActionValidationMode(action_spec.validation_mode).value,
        }
    )


__all__ = [
    "EnvironmentDependencies",
    "EnvironmentDependencyRequest",
    "EnvironmentDependencyResolver",
]
