"""Gymnasium environment for baseline-anchored residual portfolio control."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

import gymnasium as gym
import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketCalendarKind, MarketDataset
from trade_rl.risk.emergency import CausalEmergencyRiskMonitor
from trade_rl.risk.inputs import PortfolioRiskInputsProvider
from trade_rl.risk.portfolio import PortfolioRiskModel
from trade_rl.risk.pretrade import PreTradeRisk, RiskConstrainedTarget
from trade_rl.rl.actions import (
    ACTION_SCHEMA,
    ActionMode,
    ActionSpec,
    ActionValidationMode,
    AlphaContract,
    BaselineResidualComposer,
    ResidualAction,
    ResidualActionV2,
    TargetWeightAction,
)
from trade_rl.rl.diagnostics import ActionDiagnosticsAccumulator
from trade_rl.rl.environment_config import (
    RESET_STATE_MODES as _RESET_STATE_MODES,
)
from trade_rl.rl.environment_config import (
    ResidualMarketEnvConfig,
)
from trade_rl.rl.environment_decision import EnvironmentDecisionRequest
from trade_rl.rl.environment_execution import (
    EnvironmentExecutionCoordinator,
    TargetExecutionRequest,
)
from trade_rl.rl.environment_info import (
    EnvironmentStepInfoRequest,
    EnvironmentTerminalInfoRequest,
)
from trade_rl.rl.environment_observation import EnvironmentObservationRuntime
from trade_rl.rl.environment_observation_contract import (
    EnvironmentObservationContractBuilder,
)
from trade_rl.rl.environment_portfolio_risk_contract import (
    EnvironmentPortfolioRiskContractBuilder,
)
from trade_rl.rl.environment_provider_contract import (
    AlphaProvider,
    EnvironmentProviderContractBuilder,
    FactorBasisProvider,
)
from trade_rl.rl.environment_reward import EnvironmentRewardRequest
from trade_rl.rl.environment_risk import EnvironmentRiskRequest
from trade_rl.rl.environment_runtime_services import (
    EnvironmentRuntimeServicesBuilder,
)
from trade_rl.rl.episode import (
    complete_reward_history_steps,
    minimum_reward_start_index,
)
from trade_rl.rl.market_inputs import MarketInputResolver
from trade_rl.rl.normalization import ObservationNormalizer
from trade_rl.rl.observations import (
    ObservationExecutionState,
    PendingOrderObservationState,
    PolicyObservationSnapshot,
)
from trade_rl.rl.rewards import (
    REWARD_SCHEMA,
    RewardTracker,
)
from trade_rl.rl.sequence_normalization import SequenceFeatureNormalizer
from trade_rl.simulation import MarketExecutor
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import (
    ExecutionResult,
    ExecutionRuleStress,
)
from trade_rl.simulation.orders import OrderBookState
from trade_rl.simulation.stateful_execution import StatefulExecutionResult
from trade_rl.strategies.trend import TrendStrategy, TrendTargets


class ResidualMarketEnv(gym.Env[np.ndarray | dict[str, np.ndarray], np.ndarray]):
    """Dynamic residual-action environment with an independent shadow book."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        dataset: MarketDataset,
        *,
        trend_strategy: TrendStrategy | None = None,
        market_input_resolver: MarketInputResolver | None = None,
        alpha_provider: AlphaProvider
        | Callable[[MarketDataset, int], np.ndarray]
        | None = None,
        alpha_enabled: bool = False,
        alpha_artifact_digest: str | None = None,
        alpha_contract: AlphaContract | None = None,
        factor_basis: np.ndarray | None = None,
        factor_basis_provider: FactorBasisProvider
        | Callable[[MarketDataset, int], np.ndarray]
        | None = None,
        factor_artifact_digest: str | None = None,
        factor_count: int | None = None,
        action_spec: ActionSpec | None = None,
        composer: BaselineResidualComposer | None = None,
        pre_trade_risk: PreTradeRisk | None = None,
        portfolio_risk: PortfolioRiskModel | None = None,
        portfolio_risk_inputs_provider: PortfolioRiskInputsProvider | None = None,
        normalizer: ObservationNormalizer | None = None,
        sequence_normalizer: SequenceFeatureNormalizer | None = None,
        execution_rule_stress: ExecutionRuleStress | None = None,
        config: ResidualMarketEnvConfig | None = None,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        provider_contract = EnvironmentProviderContractBuilder(
            dataset,
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
        self.market_input_resolver = provider_contract.market_input_resolver
        self.trend_strategy = provider_contract.trend_strategy
        self.alpha_provider = provider_contract.alpha_provider
        self.alpha_enabled = provider_contract.alpha_enabled
        self.alpha_contract = provider_contract.alpha_contract
        self.alpha_artifact_digest = provider_contract.alpha_artifact_digest
        self._static_factor_basis = provider_contract.static_factor_basis
        self.factor_basis_provider = provider_contract.factor_basis_provider
        self.factor_artifact_digest = provider_contract.factor_artifact_digest
        resolved_factor_count = provider_contract.factor_count
        self._minimum_start_index = provider_contract.minimum_start_index
        self.composer = composer or BaselineResidualComposer()
        self.pre_trade_risk = pre_trade_risk or PreTradeRisk()
        portfolio_risk_contract = EnvironmentPortfolioRiskContractBuilder(
            dataset,
            portfolio_risk=portfolio_risk,
            inputs_provider=portfolio_risk_inputs_provider,
        ).build(minimum_start_index=self._minimum_start_index)
        self.portfolio_risk = portfolio_risk_contract.portfolio_risk
        self.portfolio_risk_inputs_provider = portfolio_risk_contract.inputs_provider
        self._minimum_start_index = portfolio_risk_contract.minimum_start_index
        self.normalizer = normalizer
        self.sequence_normalizer = sequence_normalizer
        self.execution_rule_stress = execution_rule_stress
        self.config = config or ResidualMarketEnvConfig()
        self.emergency_risk_monitor = CausalEmergencyRiskMonitor(
            self.config.emergency_risk
        )
        if (
            self.pre_trade_risk.config.max_gross
            > self.config.execution_cost.max_leverage
        ):
            raise ValueError("pre-trade max_gross cannot exceed execution max_leverage")
        if self.config.random_initial_gross > self.pre_trade_risk.config.max_gross:
            raise ValueError("random_initial_gross cannot exceed pre-trade max_gross")
        if action_spec is None:
            action_spec = ActionSpec(
                alpha_enabled=self.alpha_enabled,
                n_factors=resolved_factor_count,
                validation_mode=self.config.action_validation_mode,
            )
        if action_spec.alpha_enabled != self.alpha_enabled:
            raise ValueError("action_spec alpha mode does not match environment")
        if action_spec.n_factors != resolved_factor_count:
            raise ValueError("action_spec factor count does not match environment")
        if (
            action_spec.mode is ActionMode.TARGET_WEIGHT
            and action_spec.target_weight_count != dataset.n_symbols
        ):
            raise ValueError("target weight count does not match dataset symbols")
        self.action_spec = action_spec
        self._action_names = action_spec.names_for_symbols(dataset.symbols)
        self._nominal_episode_bars = self.config.resolve_nominal_episode_bars(dataset)
        self._nominal_decision_bars = self.config.resolve_nominal_decision_bars(dataset)
        if self._nominal_decision_bars > self._nominal_episode_bars:
            raise ValueError("decision interval cannot exceed episode duration")

        reward_config = self.config.resolved_reward_config()
        resolved_decision_hours = (
            self._nominal_decision_bars * dataset.bar_hours
            if self.config.decision_every is not None
            else self.config.decision_hours
        )
        if self.config.episode_hour_choices and any(
            choice + 1e-12 < resolved_decision_hours
            for choice in self.config.episode_hour_choices
        ):
            raise ValueError(
                "episode_hour_choices cannot be shorter than the resolved "
                "decision interval"
            )
        self._resolved_decision_hours = resolved_decision_hours
        self.reward_tracker = RewardTracker(
            reward_config,
            decision_hours=resolved_decision_hours,
        )
        if (
            self.config.require_full_reward_preroll
            and reward_config.baseline_underperformance_weight > 0.0
        ):
            self._minimum_start_index = minimum_reward_start_index(
                dataset,
                signal_minimum=self._minimum_start_index,
                window_hours=reward_config.baseline_window_hours,
            )
        self.hybrid_executor = MarketExecutor(
            dataset,
            self.config.execution_cost,
            rule_stress=self.execution_rule_stress,
        )
        self.shadow_executor = MarketExecutor(
            dataset,
            self.config.execution_cost,
            rule_stress=self.execution_rule_stress,
        )
        self.executor = self.hybrid_executor
        self._reward_history_cache: dict[int, tuple[float, ...]] = {}

        observation_contract = EnvironmentObservationContractBuilder(
            dataset,
            self.config,
            action_spec=self.action_spec,
            normalizer=self.normalizer,
            sequence_normalizer=self.sequence_normalizer,
            alpha_artifact_digest=self.alpha_artifact_digest,
            factor_artifact_digest=self.factor_artifact_digest,
            action_spec_digest=self.action_spec_digest,
        ).build(minimum_start_index=self._minimum_start_index)
        self.observation_builder = observation_contract.observation_builder
        self.layout = observation_contract.layout
        self.asset_active_column = observation_contract.asset_active_column
        self.sequence_observation_builder = (
            observation_contract.sequence_observation_builder
        )
        self.sequence_policy_plane = observation_contract.sequence_policy_plane
        self.sequence_layout_metadata = observation_contract.sequence_layout_metadata
        self._observation_schema = observation_contract.observation_schema
        self._observation_contract_digest = (
            observation_contract.observation_contract_digest
        )
        self.observation_space = observation_contract.observation_space
        self.action_space = observation_contract.action_space
        self._minimum_start_index = observation_contract.minimum_start_index
        runtime_services = EnvironmentRuntimeServicesBuilder(
            dataset,
            self.config,
            minimum_start_index=self._minimum_start_index,
            observation_contract=observation_contract,
            normalizer=self.normalizer,
            sequence_normalizer=self.sequence_normalizer,
            action_spec=self.action_spec,
            composer=self.composer,
            pre_trade_risk=self.pre_trade_risk,
            alpha_enabled=self.alpha_enabled,
            emergency_risk_monitor=self.emergency_risk_monitor,
            portfolio_risk=self.portfolio_risk,
            portfolio_risk_inputs_provider=self.portfolio_risk_inputs_provider,
            reward_tracker=self.reward_tracker,
            hybrid_executor=self.hybrid_executor,
            shadow_executor=self.shadow_executor,
        ).build()
        self._episode_sampler = runtime_services.episode_sampler
        self._execution_coordinator = runtime_services.execution_coordinator
        self._observation_assembler = runtime_services.observation_assembler
        self._decision_planner = runtime_services.decision_planner
        self._risk_projector = runtime_services.risk_projector
        self._reward_coordinator = runtime_services.reward_coordinator
        self._info_builder = runtime_services.info_builder
        self._termination_coordinator = runtime_services.termination_coordinator
        self._environment_digest = content_digest(self._digest_payload())

        self.start_index = self._minimum_start_index
        self.end_index = self.start_index + 1
        self.current_index = self.start_index
        initial_prices = dataset.close[self.start_index]
        self.hybrid = BookState.zero(
            dataset.n_symbols,
            self.config.initial_capital,
            initial_prices,
            contract_multipliers=dataset.resolved_array("contract_multipliers"),
        )
        self.shadow = self.hybrid.clone()
        self._decision_step_index = 0
        self._episode_seed = self.config.execution_cost.random_seed
        self._episode_hours = self.config.episode_hours
        self._initial_state_mode = "cash"
        self._previous_action = np.zeros(self.action_spec.size, dtype=np.float32)
        self._pending_hybrid_target: np.ndarray | None = None
        self._pending_shadow_target: np.ndarray | None = None
        self._hybrid_order_book = OrderBookState.empty()
        self._shadow_order_book = OrderBookState.empty()
        self._position_age = np.zeros(dataset.n_symbols, dtype=np.float64)
        self._execution_state = ObservationExecutionState.zero(dataset.n_symbols)
        self._action_diagnostics = ActionDiagnosticsAccumulator()
        self._has_reset = False

    def _digest_payload(self) -> dict[str, object]:
        return {
            "action_schema": ACTION_SCHEMA,
            "action_spec": {
                "alpha_enabled": self.action_spec.alpha_enabled,
                "mode": ActionMode(self.action_spec.mode).value,
                "risk_tilt_enabled": self.action_spec.risk_tilt_enabled,
                "n_factors": self.action_spec.n_factors,
                "names": self._action_names,
                "target_weight_count": self.action_spec.target_weight_count,
                "validation_mode": ActionValidationMode(
                    self.action_spec.validation_mode
                ).value,
            },
            "alpha_artifact_digest": self.alpha_artifact_digest,
            "alpha_contract": asdict(self.alpha_contract),
            "calendar_kind": MarketCalendarKind(self.dataset.calendar_kind).value,
            "dataset_id": self.dataset.dataset_id,
            "environment_config": {
                "accept_legacy_actions": self.config.accept_legacy_actions,
                "decision_every": self.config.decision_every,
                "decision_hours": self.config.decision_hours,
                "signal_delay_decisions": self.config.signal_delay_decisions,
                "resolved_decision_hours": self._resolved_decision_hours,
                "episode_bars": self.config.episode_bars,
                "episode_hour_choices": self.config.episode_hour_choices,
                "episode_hours": self.config.episode_hours,
                "execution_cost": asdict(self.config.execution_cost),
                "finite_horizon_observation": self.config.finite_horizon_observation,
                "fail_on_incomplete_emergency_liquidation": (
                    self.config.fail_on_incomplete_emergency_liquidation
                ),
                "structured_sequence_observation": self.config.structured_sequence_observation,
                "sequence_windows": self.config.resolved_sequence_windows,
                "require_full_reward_preroll": self.config.require_full_reward_preroll,
                "initial_capital": self.config.initial_capital,
                "initial_state_modes": self.config.initial_state_modes,
                "episode_sampling_mode": self.config.episode_sampling_mode,
                "regime_feature_index": self.config.regime_feature_index,
                "regime_bins": self.config.regime_bins,
                "stress_quantile": self.config.stress_quantile,
                "liquidate_on_end": self.config.liquidate_on_end,
                "minimum_equity_fraction": self.config.minimum_equity_fraction,
                "random_initial_gross": self.config.random_initial_gross,
                "stress_drawdown_fraction": self.config.stress_drawdown_fraction,
                "partial_fill_fraction": self.config.partial_fill_fraction,
            },
            "factor_artifact_digest": self.factor_artifact_digest,
            "market_input_resolver_digest": (
                None
                if self.market_input_resolver is None
                else self.market_input_resolver.digest
            ),
            "normalizer_digest": (
                None if self.normalizer is None else self.normalizer.digest
            ),
            "sequence_normalizer_digest": (
                None
                if self.sequence_normalizer is None
                else self.sequence_normalizer.digest
            ),
            "observation_schema": self._observation_schema,
            "observation_contract_digest": self._observation_contract_digest,
            "portfolio_risk": asdict(self.portfolio_risk.config),
            "portfolio_risk_inputs_digest": (
                None
                if self.portfolio_risk_inputs_provider is None
                else self.portfolio_risk_inputs_provider.identity_digest
            ),
            "pre_trade_risk": asdict(self.pre_trade_risk.config),
            "reward": asdict(self.reward_tracker.config),
            "reward_schema": REWARD_SCHEMA,
            "schema_version": "residual_market_environment_v3",
            "trend": asdict(self.trend_strategy.config),
        }

    @property
    def observation_schema(self) -> str:
        return self._observation_schema

    @property
    def observation_contract_digest(self) -> str:
        return self._observation_contract_digest

    @property
    def dataset_id(self) -> str:
        return self.dataset.dataset_id

    @property
    def initial_capital(self) -> float:
        return self.config.initial_capital

    @property
    def environment_digest(self) -> str:
        return self._environment_digest

    @property
    def execution_policy_digest(self) -> str:
        return self.hybrid_executor.execution_policy_digest

    @property
    def hybrid_order_book(self) -> OrderBookState:
        return self._hybrid_order_book

    @property
    def shadow_order_book(self) -> OrderBookState:
        return self._shadow_order_book

    @property
    def episode_bars(self) -> int:
        return self._nominal_episode_bars

    @property
    def minimum_start_index(self) -> int:
        return self._minimum_start_index

    @property
    def decision_bars(self) -> int:
        return self._nominal_decision_bars

    @property
    def decision_hours(self) -> float:
        return self._resolved_decision_hours

    @property
    def action_names(self) -> tuple[str, ...]:
        return self._action_names

    @property
    def action_spec_digest(self) -> str:
        return content_digest(
            {
                "schema_version": ACTION_SCHEMA,
                "alpha_enabled": self.action_spec.alpha_enabled,
                "mode": ActionMode(self.action_spec.mode).value,
                "risk_tilt_enabled": self.action_spec.risk_tilt_enabled,
                "n_factors": self.action_spec.n_factors,
                "names": self._action_names,
                "target_weight_count": self.action_spec.target_weight_count,
                "validation_mode": ActionValidationMode(
                    self.action_spec.validation_mode
                ).value,
            }
        )

    @staticmethod
    def _drawdown(book: BookState) -> float:
        value = max(book.portfolio_value, 0.0)
        return min(1.0, max(0.0, 1.0 - value / max(book.peak_value, value, 1e-12)))

    def _alpha_at(self, index: int) -> np.ndarray:
        if self.market_input_resolver is not None:
            return self.market_input_resolver.resolve(self.dataset, index)[1]
        if not self.alpha_enabled or self.alpha_provider is None:
            return np.zeros(self.dataset.n_symbols, dtype=np.float64)
        provider = self.alpha_provider
        if hasattr(provider, "predict_at"):
            value = provider.predict_at(self.dataset, index)
        else:
            value = provider(self.dataset, index)
        return self.alpha_contract.prepare(
            np.asarray(value, dtype=np.float64),
            n_symbols=self.dataset.n_symbols,
        )

    def _factor_basis_at(self, index: int) -> np.ndarray:
        if self.action_spec.n_factors == 0:
            return np.empty((0, self.dataset.n_symbols), dtype=np.float64)
        if self._static_factor_basis is not None:
            return self._static_factor_basis.copy()
        provider = self.factor_basis_provider
        if provider is None:
            raise RuntimeError("factor basis is configured without a provider")
        if hasattr(provider, "basis_at"):
            value = provider.basis_at(self.dataset, index)
        else:
            value = provider(self.dataset, index)
        basis = np.asarray(value, dtype=np.float64)
        if basis.shape != (self.action_spec.n_factors, self.dataset.n_symbols):
            raise ValueError("factor provider returned an invalid basis")
        if not np.isfinite(basis).all():
            raise ValueError("factor provider returned non-finite values")
        return basis

    def _market_inputs(self) -> tuple[TrendTargets, np.ndarray, np.ndarray]:
        if self.market_input_resolver is None:
            trends = self.trend_strategy.targets(self.dataset, self.current_index)
            alpha = self._alpha_at(self.current_index)
        else:
            trends, alpha = self.market_input_resolver.resolve(
                self.dataset, self.current_index
            )
        return trends, alpha, self._factor_basis_at(self.current_index)

    def _observation_runtime(self) -> EnvironmentObservationRuntime:
        return EnvironmentObservationRuntime(
            current_index=self.current_index,
            start_index=self.start_index,
            end_index=self.end_index,
            hybrid=self.hybrid,
            shadow=self.shadow,
            hybrid_order_book=self._hybrid_order_book,
            execution_state=self._execution_state,
            previous_action=self._previous_action,
            pending_hybrid_target=self._pending_hybrid_target,
        )

    def _pending_order_observation_state(self) -> PendingOrderObservationState:
        return self._observation_assembler.pending_order_state(
            self._observation_runtime()
        )

    def _flat_observation_pair(self) -> tuple[np.ndarray, np.ndarray]:
        trends, alpha, factor_basis = self._market_inputs()
        return self._observation_assembler.flat_pair(
            self._observation_runtime(),
            trends=trends,
            alpha=alpha,
            factor_basis=factor_basis,
            pre_trade_risk=self.pre_trade_risk,
        )

    def observation_snapshot(self) -> PolicyObservationSnapshot:
        """Export the exact current training observation for serving parity."""

        if not self._has_reset:
            raise RuntimeError("environment must be reset before exporting observation")
        trends, alpha, factor_basis = self._market_inputs()
        return self._observation_assembler.snapshot(
            self._observation_runtime(),
            trends=trends,
            alpha=alpha,
            factor_basis=factor_basis,
            pre_trade_risk=self.pre_trade_risk,
            execution_policy_digest=self.execution_policy_digest,
        )

    def _observation(self) -> np.ndarray | dict[str, np.ndarray]:
        trends, alpha, factor_basis = self._market_inputs()
        return self._observation_assembler.observation(
            self._observation_runtime(),
            trends=trends,
            alpha=alpha,
            factor_basis=factor_basis,
            pre_trade_risk=self.pre_trade_risk,
        )

    def _episode_end(self, start: int, *, hours: float, bars: int | None) -> int:
        return self._episode_sampler.episode_end(start, hours=hours, bars=bars)

    def _valid_starts(self, *, hours: float, bars: int | None) -> np.ndarray:
        return self._episode_sampler.valid_starts(hours=hours, bars=bars)

    def _sample_episode_contract(
        self,
        options: dict[str, object],
    ) -> tuple[int, int, float]:
        contract = self._episode_sampler.sample(options, self.np_random)
        return contract.start_index, contract.end_index, contract.hours

    def _initial_weights(self, *, mode: str, start: int) -> tuple[np.ndarray, float]:
        trends = self.trend_strategy.targets(self.dataset, start)
        if mode == "cash":
            weights = np.zeros(self.dataset.n_symbols, dtype=np.float64)
            peak = self.config.initial_capital
        elif mode == "baseline":
            weights = trends.base.copy()
            peak = self.config.initial_capital
        elif mode == "random":
            raw = self.np_random.normal(size=self.dataset.n_symbols)
            gross = float(np.abs(raw).sum())
            weights = (
                np.zeros_like(raw)
                if gross <= 1e-15
                else raw / gross * self.config.random_initial_gross
            )
            constrained = self.pre_trade_risk.constrain(
                weights,
                current=np.zeros_like(weights),
                drawdown=0.0,
            )
            weights = constrained.weights
            peak = self.config.initial_capital
        elif mode == "stress":
            weights = trends.base.copy()
            peak = self.config.initial_capital / (
                1.0 - self.config.stress_drawdown_fraction
            )
        elif mode == "partial_fill":
            weights = trends.base * self.config.partial_fill_fraction
            peak = self.config.initial_capital
        elif mode == "restore":
            raise ValueError("restore mode requires initial_book in reset options")
        else:  # pragma: no cover - validated in config/options
            raise RuntimeError("unhandled initial state mode")
        initial_drawdown = (
            self.config.stress_drawdown_fraction if mode == "stress" else 0.0
        )
        hard_constrained = self.pre_trade_risk.constrain(
            weights,
            current=weights,
            drawdown=initial_drawdown,
        )
        return hard_constrained.weights, peak

    def _make_initial_book(
        self, *, weights: np.ndarray, peak: float, start: int
    ) -> BookState:
        book = BookState.from_weights(
            weights=weights,
            capital=self.config.initial_capital,
            prices=self.dataset.close[start],
            peak_value=peak,
            max_gross=self.pre_trade_risk.config.max_gross,
            contract_multipliers=self.dataset.resolved_array("contract_multipliers"),
        )
        book.max_drawdown = self._drawdown(book)
        gross_notional = float(np.abs(book.position_values).sum())
        book.set_margin(
            margin_used=gross_notional / self.config.execution_cost.max_leverage,
            maintenance_margin=self.config.execution_cost.maintenance_margin_rate,
            maintenance_requirement=(
                self.config.execution_cost.maintenance_margin_rate * gross_notional
            ),
        )
        return book

    def _bars_between(self, start: int, stop: int) -> int:
        remaining = stop - start
        if remaining <= 0:
            raise ValueError("reward pre-roll interval must be non-empty")
        if self.config.decision_every is not None:
            return min(self.config.decision_every, remaining)
        if self.dataset.regular_cadence:
            return min(
                self.dataset.bars_for_hours(self.config.decision_hours), remaining
            )
        return self.dataset.bars_until(
            start,
            self.config.decision_hours,
            maximum_index=stop,
        )

    def _baseline_reward_history(
        self, *, reward_start: int, history_steps: int
    ) -> tuple[float, ...]:
        if history_steps == 0:
            return ()
        cached = self._reward_history_cache.get(reward_start)
        if cached is not None:
            return cached
        history_start = self.dataset.lookback_index(
            reward_start, self.reward_tracker.config.baseline_window_hours
        )
        if history_start < self.trend_strategy.minimum_history_for(self.dataset):
            return ()
        book = self._make_initial_book(
            weights=np.zeros(self.dataset.n_symbols, dtype=np.float64),
            peak=self.config.initial_capital,
            start=history_start,
        )
        executor = MarketExecutor(
            self.dataset,
            self.config.execution_cost,
            rule_stress=self.execution_rule_stress,
        )
        executor.reset_random_state(reward_start)
        cursor = history_start
        pending_target: np.ndarray | None = None
        returns: list[float] = []
        while cursor < reward_start:
            submitted_target = self.trend_strategy.targets(self.dataset, cursor).base
            if self.config.signal_delay_decisions == 0:
                target = submitted_target
            else:
                target = (
                    book.weights.copy() if pending_target is None else pending_target
                )
                pending_target = submitted_target.copy()
            constrained = self.pre_trade_risk.constrain(
                target,
                current=book.weights,
                drawdown=self._drawdown(book),
            )
            result = executor.execute_interval(
                book,
                constrained.weights,
                start_index=cursor,
                bars=self._bars_between(cursor, reward_start),
            )
            book = result.book
            cursor = result.next_index
            returns.append(float(result.interval_log_return))
            if book.insolvent:
                raise RuntimeError("baseline reward pre-roll terminated economically")
        if len(returns) < history_steps:
            return ()
        history = tuple(returns[-history_steps:])
        self._reward_history_cache[reward_start] = history
        return history

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray | dict[str, np.ndarray], dict[str, Any]]:
        resolved_seed = (
            self.config.execution_cost.random_seed
            if seed is None and not self._has_reset
            else seed
        )
        super().reset(seed=resolved_seed)
        self._has_reset = True
        resolved_options = options or {}
        start, end, resolved_hours = self._sample_episode_contract(resolved_options)
        raw_mode = resolved_options.get("initial_state_mode")
        if raw_mode is None:
            mode = str(self.np_random.choice(self.config.initial_state_modes))
        elif not isinstance(raw_mode, str) or raw_mode not in _RESET_STATE_MODES:
            raise ValueError("initial_state_mode option is not supported")
        else:
            mode = raw_mode
        self.start_index = start
        self.current_index = start
        self.end_index = end
        if mode == "restore":
            supplied = resolved_options.get("initial_book")
            if not isinstance(supplied, BookState):
                raise ValueError("restore mode requires a BookState initial_book")
            if supplied.quantities.shape != (self.dataset.n_symbols,):
                raise ValueError("initial_book does not match dataset symbols")
            if not np.array_equal(
                np.asarray(supplied.contract_multipliers),
                self.dataset.resolved_array("contract_multipliers"),
            ):
                raise ValueError(
                    "initial_book contract multipliers do not match dataset"
                )
            if not math.isclose(
                supplied.portfolio_value,
                self.config.initial_capital,
                rel_tol=0.0,
                abs_tol=max(1e-8, self.config.initial_capital * 1e-9),
            ):
                raise ValueError("initial_book value must match initial_capital")
            self.hybrid = supplied.clone()
            self.hybrid.revalue(self.dataset.close[start])
            if not math.isclose(
                self.hybrid.portfolio_value,
                self.config.initial_capital,
                rel_tol=0.0,
                abs_tol=max(1e-8, self.config.initial_capital * 1e-9),
            ):
                raise ValueError(
                    "initial_book value after start-price revaluation must match "
                    "initial_capital"
                )
            self.shadow = self.hybrid.clone()
            weights = self.hybrid.weights.copy()
        else:
            weights, peak = self._initial_weights(mode=mode, start=start)
            self.hybrid = self._make_initial_book(
                weights=weights,
                peak=peak,
                start=start,
            )
            self.shadow = self.hybrid.clone()
        self._episode_seed = int(
            self.np_random.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32)
        )
        self.hybrid_executor.reset_random_state(self._episode_seed)
        self.shadow_executor.reset_random_state(self._episode_seed)
        self._decision_step_index = 0
        self._episode_hours = resolved_hours
        self._initial_state_mode = mode
        self._previous_action = np.zeros(self.action_spec.size, dtype=np.float32)
        self._pending_hybrid_target = None
        self._pending_shadow_target = None
        self._hybrid_order_book = OrderBookState.empty()
        self._shadow_order_book = OrderBookState.empty()
        self._position_age = np.zeros(self.dataset.n_symbols, dtype=np.float64)
        if mode == "partial_fill":
            raw_requested = self.trend_strategy.targets(self.dataset, start).base
            requested = self.pre_trade_risk.constrain(
                raw_requested,
                current=raw_requested,
                drawdown=0.0,
            ).weights
            fill_ratio = np.full(
                self.dataset.n_symbols,
                self.config.partial_fill_fraction,
                dtype=np.float64,
            )
            self._execution_state = ObservationExecutionState(
                requested_weights=requested,
                fill_ratio=fill_ratio,
                unfilled_turnover=np.abs(requested - weights),
                participation=np.zeros(self.dataset.n_symbols),
                execution_cost=np.zeros(self.dataset.n_symbols),
                position_age=np.zeros(self.dataset.n_symbols),
            )
        else:
            self._execution_state = ObservationExecutionState.zero(
                self.dataset.n_symbols,
                requested_weights=weights,
            )
        self._action_diagnostics.reset()
        reward_history_steps = complete_reward_history_steps(
            self.dataset,
            reward_start=start,
            window_hours=self.reward_tracker.config.baseline_window_hours,
            window_steps=self.reward_tracker.baseline_window_steps,
        )
        baseline_history = self._baseline_reward_history(
            reward_start=start, history_steps=reward_history_steps
        )
        reward_history_steps = len(baseline_history)
        if (
            self.config.require_full_reward_preroll
            and self.reward_tracker.config.baseline_underperformance_weight > 0.0
            and reward_history_steps != self.reward_tracker.baseline_window_steps
        ):
            raise ValueError("episode start lacks the complete reward pre-roll window")
        self.reward_tracker.reset(
            hybrid_drawdown=self._drawdown(self.hybrid),
            shadow_drawdown=self._drawdown(self.shadow),
            hybrid_history=baseline_history,
            shadow_history=baseline_history,
        )
        return self._observation(), {
            "episode_seed": self._episode_seed,
            "episode_hours": self._episode_hours,
            "initial_state_mode": mode,
            "start_index": start,
            "end_index": end,
            "reward_history_steps": reward_history_steps,
        }

    def _market_notional(self, index: int) -> np.ndarray:
        return self._risk_projector.market_notional(index)

    def _constrain_target(
        self,
        proposal: np.ndarray,
        book: BookState,
    ) -> RiskConstrainedTarget:
        return self._risk_projector.project(
            EnvironmentRiskRequest(
                proposal=proposal,
                book=book,
                current_index=self.current_index,
            )
        )

    def _parse_action(
        self,
        value: np.ndarray,
    ) -> tuple[
        ResidualAction | ResidualActionV2 | TargetWeightAction,
        np.ndarray,
        int,
        float,
    ]:
        return self._decision_planner.parse_action(value)

    def _decision_bar_count(self) -> int:
        return self._decision_planner.decision_bar_count(
            current_index=self.current_index,
            end_index=self.end_index,
        )

    @staticmethod
    def _merge_liquidation_return(liquidation: ExecutionResult) -> BookState:
        return EnvironmentExecutionCoordinator.merge_liquidation_return(liquidation)

    @staticmethod
    def _liquidation_complete(liquidation: ExecutionResult) -> bool:
        return EnvironmentExecutionCoordinator.liquidation_complete(liquidation)

    def _execution_observation_state(
        self,
        *,
        requested_weights: np.ndarray,
        result: ExecutionResult | StatefulExecutionResult,
        previous_weights: np.ndarray,
    ) -> ObservationExecutionState:
        state, position_age = self._execution_coordinator.execution_observation_state(
            position_age=self._position_age,
            requested_weights=requested_weights,
            result=result,
            previous_weights=previous_weights,
        )
        self._position_age = position_age
        return state

    def _execute_stateful_target(
        self,
        *,
        executor: MarketExecutor,
        book: BookState,
        order_book: OrderBookState,
        target: np.ndarray,
        bars: int,
        book_kind: str,
    ) -> StatefulExecutionResult:
        return self._execution_coordinator.execute_target(
            executor=executor,
            book=book,
            order_book=order_book,
            request=TargetExecutionRequest(
                target=target,
                start_index=self.current_index,
                decision_step_index=self._decision_step_index,
                bars=bars,
                book_kind=book_kind,
            ),
        )

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray | dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self.current_index >= self.end_index:
            raise RuntimeError("step called after the episode ended")
        trends, alpha, factor_basis = self._market_inputs()
        decision = self._decision_planner.plan(
            EnvironmentDecisionRequest(
                action=action,
                trends=trends,
                alpha=alpha,
                factor_basis=factor_basis,
                hybrid_weights=self.hybrid.weights,
                shadow_weights=self.shadow.weights,
                pending_hybrid_target=self._pending_hybrid_target,
                pending_shadow_target=self._pending_shadow_target,
                current_index=self.current_index,
                end_index=self.end_index,
            )
        )
        self._pending_hybrid_target = decision.next_pending_hybrid_target
        self._pending_shadow_target = decision.next_pending_shadow_target
        hybrid_risk = self._risk_projector.project(
            EnvironmentRiskRequest(
                proposal=decision.executed_hybrid_target,
                book=self.hybrid,
                current_index=self.current_index,
            )
        )
        shadow_risk = self._risk_projector.project(
            EnvironmentRiskRequest(
                proposal=decision.executed_shadow_target,
                book=self.shadow,
                current_index=self.current_index,
            )
        )
        previous_hybrid_weights = self.hybrid.weights.copy()
        hybrid_execution = self._execute_stateful_target(
            executor=self.hybrid_executor,
            book=self.hybrid,
            order_book=self._hybrid_order_book,
            target=hybrid_risk.weights,
            bars=decision.bars,
            book_kind="hybrid",
        )
        shadow_execution = self._execute_stateful_target(
            executor=self.shadow_executor,
            book=self.shadow,
            order_book=self._shadow_order_book,
            target=shadow_risk.weights,
            bars=decision.bars,
            book_kind="shadow",
        )
        if hybrid_execution.bars_advanced != shadow_execution.bars_advanced:
            raise RuntimeError("hybrid and shadow books advanced different bar counts")

        self.hybrid = hybrid_execution.book
        self.shadow = shadow_execution.book
        self._hybrid_order_book = hybrid_execution.order_book
        self._shadow_order_book = shadow_execution.order_book
        self.current_index = hybrid_execution.next_index
        self._decision_step_index += 1
        time_limit_reached = self.current_index >= self.end_index
        pending_hybrid_target = self._pending_hybrid_target
        pending_target_discarded = bool(
            time_limit_reached
            and self.config.signal_delay_decisions == 1
            and pending_hybrid_target is not None
        )
        discarded_pending_target = (
            pending_hybrid_target.copy()
            if pending_target_discarded and pending_hybrid_target is not None
            else None
        )
        if time_limit_reached:
            self._pending_hybrid_target = None
            self._pending_shadow_target = None
        transition = self._termination_coordinator.resolve(
            hybrid=self.hybrid,
            shadow=self.shadow,
            current_index=self.current_index,
            time_limit_reached=time_limit_reached,
            hybrid_log_return=hybrid_execution.interval_log_return,
            shadow_log_return=shadow_execution.interval_log_return,
        )
        self.hybrid = transition.hybrid
        self.shadow = transition.shadow
        hybrid_log_return = transition.hybrid_log_return
        shadow_log_return = transition.shadow_log_return
        hybrid_liquidation = transition.hybrid_liquidation
        shadow_liquidation = transition.shadow_liquidation
        liquidation_complete = transition.liquidation_complete
        liquidation_terminal = transition.liquidation_terminal
        emergency_deleverage = transition.emergency_deleverage
        hybrid_terminated = transition.hybrid_terminated
        shadow_terminated = transition.shadow_terminated
        economic_transition = transition.economic_transition
        terminated = economic_transition.terminated
        truncated = economic_transition.truncated
        action_delta_l1 = float(
            np.abs(decision.maintained_action - self._previous_action).sum()
        )
        projection_distance = hybrid_risk.projection_l1
        reward_breakdown = self._reward_coordinator.step(
            EnvironmentRewardRequest(
                hybrid_log_return=hybrid_log_return,
                shadow_log_return=shadow_log_return,
                hybrid=self.hybrid,
                shadow=self.shadow,
                projection_distance=projection_distance,
                hybrid_terminated=hybrid_terminated,
                shadow_terminated=shadow_terminated,
            )
        )
        self._execution_state = self._execution_observation_state(
            requested_weights=hybrid_risk.weights,
            result=hybrid_execution,
            previous_weights=previous_hybrid_weights,
        )
        self._previous_action = decision.maintained_action.copy()
        self._action_diagnostics.update(
            action=decision.maintained_action,
            saturated_count=decision.saturated_count,
            action_delta_l1=action_delta_l1,
            projection_l1=projection_distance,
            constrained=hybrid_risk.was_constrained,
            turnover_overridden=hybrid_risk.turnover_overridden,
        )
        termination_reason = economic_transition.reason
        terminal_accounting_mode = (
            "liquidate_at_close"
            if liquidation_terminal
            else "mark_to_market"
            if time_limit_reached
            else "economic_termination"
        )
        terminal_liquidation_cost = (
            float(hybrid_liquidation.interval_cost)
            if liquidation_terminal and hybrid_liquidation is not None
            else 0.0
        )
        info = self._info_builder.step_info(
            EnvironmentStepInfoRequest(
                action_delta_l1=action_delta_l1,
                raw_max_abs=decision.raw_max_abs,
                saturated_count=decision.saturated_count,
                composition=decision.composition,
                decision_step_index=self._decision_step_index,
                hybrid_log_return=hybrid_log_return,
                shadow_log_return=shadow_log_return,
                emergency_deleverage=emergency_deleverage,
                execution_delay_warmup=decision.execution_delay_warmup,
                submitted_target=decision.submitted_hybrid_target,
                executed_target=decision.executed_hybrid_target,
                hybrid=self.hybrid,
                reward_breakdown=reward_breakdown,
                hybrid_execution=hybrid_execution,
                hybrid_risk=hybrid_risk,
                hybrid_terminated=hybrid_terminated,
                shadow_execution=shadow_execution,
                shadow_risk=shadow_risk,
                shadow_terminated=shadow_terminated,
                liquidation_complete=liquidation_complete,
                liquidation_terminal=liquidation_terminal,
                termination_reason=termination_reason,
                terminal_accounting_mode=terminal_accounting_mode,
                terminal_liquidation_cost=terminal_liquidation_cost,
                pending_target_discarded=pending_target_discarded,
                discarded_pending_target=discarded_pending_target,
                hybrid_liquidation=hybrid_liquidation,
                shadow_liquidation=shadow_liquidation,
            )
        )
        if terminated or truncated:
            info.update(self._terminal_info())
        return (
            self._observation(),
            reward_breakdown.scaled_total,
            terminated,
            truncated,
            info,
        )

    def _terminal_info(self) -> dict[str, object]:
        return self._info_builder.terminal_info(
            EnvironmentTerminalInfoRequest(
                episode_hours=self._episode_hours,
                episode_seed=self._episode_seed,
                action_diagnostics=self._action_diagnostics.snapshot(),
                hybrid=self.hybrid,
                shadow=self.shadow,
                initial_state_mode=self._initial_state_mode,
            )
        )
