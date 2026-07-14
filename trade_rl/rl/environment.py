"""Gymnasium environment for baseline-anchored residual portfolio control."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict
from typing import Protocol

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketCalendarKind, MarketDataset
from trade_rl.domain.common import require_sha256
from trade_rl.evaluation.metrics import PerformanceMetrics, evaluate_performance
from trade_rl.evaluation.series import ReturnKind, ReturnSeries
from trade_rl.risk.pretrade import PreTradeRisk, RiskConstrainedTarget
from trade_rl.rl.actions import (
    ACTION_SCHEMA,
    ActionSpec,
    ActionValidationMode,
    AlphaContract,
    BaselineResidualComposer,
    ResidualAction,
    ResidualActionV2,
)
from trade_rl.rl.diagnostics import ActionDiagnosticsAccumulator
from trade_rl.rl.environment_config import (
    RESET_STATE_MODES as _RESET_STATE_MODES,
)
from trade_rl.rl.environment_config import (
    ResidualMarketEnvConfig,
)
from trade_rl.rl.episode import (
    complete_reward_history_steps,
    minimum_reward_start_index,
)
from trade_rl.rl.market_inputs import MarketInputResolver
from trade_rl.rl.normalization import ObservationNormalizer
from trade_rl.rl.observations import (
    OBSERVATION_SCHEMA,
    ObservationBuilder,
    ObservationExecutionState,
    ObservationInput,
    observation_passthrough_indices,
)
from trade_rl.rl.rewards import (
    REWARD_SCHEMA,
    RewardTracker,
)
from trade_rl.rl.transition import classify_economic_transition
from trade_rl.simulation.accounting import BookState, EconomicTerminationReason
from trade_rl.simulation.execution import (
    ExecutionResult,
    MarketExecutor,
)
from trade_rl.strategies.trend import TrendStrategy, TrendTargets

_LIQUIDATION_TOLERANCE = 1e-12


class AlphaProvider(Protocol):
    @property
    def artifact_digest(self) -> str: ...

    def predict_at(self, dataset: MarketDataset, index: int) -> np.ndarray: ...


class FactorBasisProvider(Protocol):
    @property
    def artifact_digest(self) -> str: ...

    @property
    def n_factors(self) -> int: ...

    def basis_at(self, dataset: MarketDataset, index: int) -> np.ndarray: ...


class ResidualMarketEnv(gym.Env[np.ndarray, np.ndarray]):
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
        normalizer: ObservationNormalizer | None = None,
        config: ResidualMarketEnvConfig | None = None,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        resolved_trend = trend_strategy or (
            market_input_resolver.trend_strategy
            if market_input_resolver is not None
            else TrendStrategy()
        )
        if (
            market_input_resolver is None
            and alpha_provider is not None
            and hasattr(alpha_provider, "predict")
            and hasattr(alpha_provider, "identity_digest")
        ):
            market_input_resolver = MarketInputResolver(
                trend_strategy=resolved_trend,
                alpha_provider=alpha_provider,  # type: ignore[arg-type]
                alpha_enabled=bool(alpha_enabled),
            )
        if market_input_resolver is not None and trend_strategy is not None:
            if market_input_resolver.trend_strategy != trend_strategy:
                raise ValueError(
                    "market_input_resolver trend differs from trend_strategy"
                )
        self.market_input_resolver = market_input_resolver
        self.trend_strategy = resolved_trend
        self.alpha_provider = alpha_provider
        self.alpha_enabled = (
            market_input_resolver.alpha_enabled
            if market_input_resolver is not None
            else bool(alpha_enabled)
        )
        if (
            self.alpha_enabled
            and self.alpha_provider is None
            and market_input_resolver is None
        ):
            raise ValueError("alpha_enabled requires an alpha_provider")
        self.alpha_contract = alpha_contract or AlphaContract()
        self.alpha_artifact_digest = self._resolve_provider_digest(
            enabled=self.alpha_enabled,
            provider=alpha_provider,
            explicit=alpha_artifact_digest,
            field_name="alpha_artifact_digest",
        )
        self._static_factor_basis = self._validated_static_basis(factor_basis)
        self.factor_basis_provider = factor_basis_provider
        resolved_factor_count = self._resolve_factor_count(
            factor_count=factor_count,
            provider=factor_basis_provider,
        )
        if self._static_factor_basis is not None:
            if resolved_factor_count not in (0, self._static_factor_basis.shape[0]):
                raise ValueError("factor_count does not match factor_basis")
            resolved_factor_count = self._static_factor_basis.shape[0]
        self.factor_artifact_digest = self._resolve_provider_digest(
            enabled=resolved_factor_count > 0,
            provider=factor_basis_provider,
            explicit=factor_artifact_digest,
            field_name="factor_artifact_digest",
            static_payload=(
                None
                if self._static_factor_basis is None
                else tuple(
                    tuple(float(value) for value in row)
                    for row in self._static_factor_basis
                )
            ),
        )
        provider_minimums = [self.trend_strategy.minimum_history_for(dataset)]
        for provider_name, provider in (
            ("alpha_provider", alpha_provider),
            ("factor_basis_provider", factor_basis_provider),
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
        self._minimum_start_index = max(provider_minimums)
        self.composer = composer or BaselineResidualComposer()
        self.pre_trade_risk = pre_trade_risk or PreTradeRisk()
        self.normalizer = normalizer
        self.config = config or ResidualMarketEnvConfig()
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
        self.action_spec = action_spec
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
        self.hybrid_executor = MarketExecutor(dataset, self.config.execution_cost)
        self.shadow_executor = MarketExecutor(dataset, self.config.execution_cost)
        self.executor = self.hybrid_executor
        self._reward_history_cache: dict[int, tuple[float, ...]] = {}

        self.observation_builder = ObservationBuilder(
            action_size=self.action_spec.size,
            n_factors=self.action_spec.n_factors,
            finite_horizon=self.config.finite_horizon_observation,
        )
        layout = self.observation_builder.layout(dataset)
        if normalizer is not None:
            if normalizer.size != layout.size:
                raise ValueError("normalizer size does not match observation layout")
            bound_dataset_ids = {
                identity
                for identity in (normalizer.dataset_id, normalizer.source_dataset_id)
                if identity is not None
            }
            if bound_dataset_ids and dataset.dataset_id not in bound_dataset_ids:
                raise ValueError(
                    "normalizer dataset identity does not match environment"
                )
            if normalizer.observation_schema != OBSERVATION_SCHEMA:
                raise ValueError(
                    "normalizer observation schema does not match environment"
                )
            observation_schema_digest = self.observation_builder.schema_digest(dataset)
            if (
                normalizer.observation_schema_digest is not None
                and normalizer.observation_schema_digest != observation_schema_digest
            ):
                raise ValueError(
                    "normalizer observation schema digest does not match environment"
                )
            if (
                normalizer.action_spec_digest is not None
                and normalizer.action_spec_digest != self.action_spec_digest
            ):
                raise ValueError(
                    "normalizer action identity does not match environment"
                )
            for field_name, expected, observed in (
                (
                    "alpha artifact",
                    self.alpha_artifact_digest,
                    normalizer.alpha_artifact_digest,
                ),
                (
                    "factor artifact",
                    self.factor_artifact_digest,
                    normalizer.factor_artifact_digest,
                ),
            ):
                if observed is not None and observed != expected:
                    raise ValueError(
                        f"normalizer {field_name} identity does not match environment"
                    )
            required_passthrough = set(
                observation_passthrough_indices(
                    dataset,
                    action_size=self.action_spec.size,
                    n_factors=self.action_spec.n_factors,
                    finite_horizon=self.config.finite_horizon_observation,
                )
            )
            if not required_passthrough.issubset(normalizer.passthrough_indices):
                raise ValueError(
                    "normalizer must preserve observation mask and activity indices"
                )
        self.layout = layout
        self.asset_active_column = 4 * dataset.n_features
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(layout.size,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_spec.size,),
            dtype=np.float32,
        )
        self._environment_digest = content_digest(self._digest_payload())

        self.start_index = self._minimum_start_index
        self.end_index = self.start_index + 1
        self.current_index = self.start_index
        initial_prices = dataset.close[self.start_index]
        self.hybrid = BookState.zero(
            dataset.n_symbols,
            self.config.initial_capital,
            initial_prices,
        )
        self.shadow = self.hybrid.clone()
        self._decision_step_index = 0
        self._episode_seed = self.config.execution_cost.random_seed
        self._episode_hours = self.config.episode_hours
        self._initial_state_mode = "cash"
        self._previous_action = np.zeros(self.action_spec.size, dtype=np.float32)
        self._position_age = np.zeros(dataset.n_symbols, dtype=np.float64)
        self._execution_state = ObservationExecutionState.zero(dataset.n_symbols)
        self._action_diagnostics = ActionDiagnosticsAccumulator()
        self._has_reset = False
        self._valid_start_cache: dict[tuple[float, int | None], np.ndarray] = {}

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

    def _digest_payload(self) -> dict[str, object]:
        return {
            "action_schema": ACTION_SCHEMA,
            "action_spec": {
                "alpha_enabled": self.action_spec.alpha_enabled,
                "n_factors": self.action_spec.n_factors,
                "names": self.action_spec.names,
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
                "resolved_decision_hours": self._resolved_decision_hours,
                "episode_bars": self.config.episode_bars,
                "episode_hour_choices": self.config.episode_hour_choices,
                "episode_hours": self.config.episode_hours,
                "execution_cost": asdict(self.config.execution_cost),
                "finite_horizon_observation": self.config.finite_horizon_observation,
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
            "observation_schema": OBSERVATION_SCHEMA,
            "pre_trade_risk": asdict(self.pre_trade_risk.config),
            "reward": asdict(self.reward_tracker.config),
            "reward_schema": REWARD_SCHEMA,
            "schema_version": "residual_market_environment_v3",
            "trend": asdict(self.trend_strategy.config),
        }

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
    def episode_bars(self) -> int:
        return self._nominal_episode_bars

    @property
    def decision_bars(self) -> int:
        return self._nominal_decision_bars

    @property
    def decision_hours(self) -> float:
        return self._resolved_decision_hours

    @property
    def action_names(self) -> tuple[str, ...]:
        return self.action_spec.names

    @property
    def action_spec_digest(self) -> str:
        return content_digest(
            {
                "schema_version": ACTION_SCHEMA,
                "alpha_enabled": self.action_spec.alpha_enabled,
                "n_factors": self.action_spec.n_factors,
                "names": self.action_spec.names,
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

    def _observation(self) -> np.ndarray:
        trends, alpha, factor_basis = self._market_inputs()
        raw = self.observation_builder.build(
            ObservationInput(
                dataset=self.dataset,
                index=self.current_index,
                trends=trends,
                alpha=alpha,
                factor_basis=factor_basis,
                hybrid=self.hybrid,
                shadow=self.shadow,
                start_index=self.start_index,
                end_index=self.end_index,
                hybrid_risk_scale=self.pre_trade_risk.risk_scale(
                    self._drawdown(self.hybrid)
                ),
                shadow_risk_scale=self.pre_trade_risk.risk_scale(
                    self._drawdown(self.shadow)
                ),
                execution_state=self._execution_state,
                previous_action=self._previous_action,
                action_size=self.action_spec.size,
                finite_horizon=self.config.finite_horizon_observation,
            )
        )
        return raw if self.normalizer is None else self.normalizer.transform(raw)

    def _episode_end(self, start: int, *, hours: float, bars: int | None) -> int:
        if bars is not None:
            end = start + bars
            if end >= self.dataset.n_bars:
                raise ValueError("episode does not fit inside the dataset")
            return end
        end = self.dataset.forward_index(start, hours)
        if self.dataset.elapsed_hours(start, end) + 1e-9 < hours:
            raise ValueError("episode duration does not fit inside the dataset")
        return end

    def _valid_starts(self, *, hours: float, bars: int | None) -> np.ndarray:
        key = (float(hours), bars)
        cached = self._valid_start_cache.get(key)
        if cached is not None:
            return cached.copy()
        minimum = self._minimum_start_index
        valid: list[int] = []
        for start in range(minimum, self.dataset.n_bars - 1):
            try:
                self._episode_end(start, hours=hours, bars=bars)
            except ValueError:
                continue
            valid.append(start)
        if not valid:
            raise ValueError("dataset is too short for the configured episode")
        resolved = np.asarray(valid, dtype=np.int64)
        self._valid_start_cache[key] = resolved
        return resolved.copy()

    def _sample_episode_contract(
        self,
        options: dict[str, object],
    ) -> tuple[int, int, float]:
        raw_hours = options.get("episode_hours")
        raw_bars = options.get("episode_bars", self.config.episode_bars)
        if raw_hours is not None and raw_bars is not None:
            raise ValueError(
                "episode_hours and episode_bars reset options are mutually exclusive"
            )
        if raw_hours is not None:
            if isinstance(raw_hours, bool) or not isinstance(raw_hours, int | float):
                raise ValueError("episode_hours option must be numeric")
            hours = float(raw_hours)
        elif self.config.episode_hour_choices:
            viable_hours: list[float] = []
            for choice in self.config.episode_hour_choices:
                try:
                    self._valid_starts(hours=float(choice), bars=None)
                except ValueError:
                    continue
                viable_hours.append(float(choice))
            if not viable_hours:
                raise ValueError(
                    "none of the configured episode durations fit the dataset"
                )
            hours = float(self.np_random.choice(viable_hours))
        else:
            hours = self.config.episode_hours
        if not math.isfinite(hours) or hours <= 0.0:
            raise ValueError("episode_hours option must be finite and positive")
        bars: int | None
        if raw_bars is None:
            bars = None
        else:
            if (
                isinstance(raw_bars, bool)
                or not isinstance(raw_bars, int)
                or raw_bars <= 0
            ):
                raise ValueError("episode_bars option must be a positive integer")
            bars = raw_bars

        if "start_idx" in options:
            raw_start = options["start_idx"]
            if isinstance(raw_start, bool) or not isinstance(raw_start, int):
                raise ValueError("start_idx must be an integer")
            start = raw_start
            minimum = self._minimum_start_index
            if start < minimum:
                raise ValueError(
                    "start_idx does not have sufficient causal or signal history"
                )
            end = self._episode_end(start, hours=hours, bars=bars)
        else:
            valid_starts = self._valid_starts(hours=hours, bars=bars)
            if self.config.episode_sampling_mode in {
                "regime_balanced",
                "stress_tail",
            }:
                feature_index = self.config.regime_feature_index
                if feature_index >= len(self.dataset.global_feature_names):
                    raise ValueError("regime_feature_index is outside global features")
                available = self.dataset.resolved_array("global_feature_available")[
                    valid_starts,
                    feature_index,
                ]
                candidate_starts = valid_starts[available]
                if candidate_starts.size == 0:
                    candidate_starts = valid_starts
                regime_values = self.dataset.global_features[
                    candidate_starts,
                    feature_index,
                ]
                if self.config.episode_sampling_mode == "stress_tail":
                    threshold = float(
                        np.quantile(
                            np.abs(regime_values),
                            self.config.stress_quantile,
                        )
                    )
                    stressed = candidate_starts[np.abs(regime_values) >= threshold]
                    if stressed.size:
                        candidate_starts = stressed
                else:
                    quantiles = np.unique(
                        np.quantile(
                            regime_values,
                            np.linspace(0.0, 1.0, self.config.regime_bins + 1),
                        )
                    )
                    if quantiles.size > 2:
                        bins = np.digitize(
                            regime_values,
                            quantiles[1:-1],
                            right=True,
                        )
                        chosen_bin = int(self.np_random.choice(np.unique(bins)))
                        candidate_starts = candidate_starts[bins == chosen_bin]
                start = int(self.np_random.choice(candidate_starts))
            else:
                start = int(self.np_random.choice(valid_starts))
            end = self._episode_end(start, hours=hours, bars=bars)
        resolved_hours = self.dataset.elapsed_hours(start, end)
        return start, end, resolved_hours

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
        executor = MarketExecutor(self.dataset, self.config.execution_cost)
        executor.reset_random_state(reward_start)
        cursor = history_start
        returns: list[float] = []
        while cursor < reward_start:
            target = self.trend_strategy.targets(self.dataset, cursor).base
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
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
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

    def _constrain_target(
        self,
        proposal: np.ndarray,
        book: BookState,
    ) -> RiskConstrainedTarget:
        target = np.asarray(proposal, dtype=np.float64).reshape(-1).copy()
        if target.shape != (self.dataset.n_symbols,) or not np.isfinite(target).all():
            raise ValueError("proposal does not match dataset symbols")
        return self.pre_trade_risk.constrain(
            target,
            current=book.weights,
            drawdown=self._drawdown(book),
        )

    def _parse_action(
        self,
        value: np.ndarray,
    ) -> tuple[ResidualAction | ResidualActionV2, np.ndarray, int, float]:
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
        if vector.shape == (2,) and self.config.accept_legacy_actions:
            legacy = ResidualAction.from_array(vector)
            migrated = np.zeros(self.action_spec.size, dtype=np.float32)
            if legacy.trend_mix >= 0.0:
                migrated[0] = legacy.trend_mix
            else:
                migrated[1] = -legacy.trend_mix
            if self.alpha_enabled:
                migrated[3] = legacy.alpha_budget
            saturated = int(np.count_nonzero(np.abs(vector) > 1.0))
            return (
                legacy,
                migrated,
                saturated,
                float(np.max(np.abs(vector), initial=0.0)),
            )
        parsed = self.action_spec.parse(value)
        return (
            parsed,
            parsed.as_array(alpha_enabled=self.alpha_enabled),
            parsed.saturated_count,
            parsed.raw_max_abs,
        )

    def _decision_bar_count(self) -> int:
        remaining = self.end_index - self.current_index
        if remaining <= 0:
            raise RuntimeError("step called after the episode ended")
        if self.config.decision_every is not None:
            return min(self.config.decision_every, remaining)
        if self.dataset.regular_cadence:
            return min(
                self.dataset.bars_for_hours(self.config.decision_hours), remaining
            )
        return self.dataset.bars_until(
            self.current_index,
            self.config.decision_hours,
            maximum_index=self.end_index,
        )

    @staticmethod
    def _merge_liquidation_return(liquidation: ExecutionResult) -> BookState:
        result = liquidation.book
        if abs(liquidation.interval_net_return) <= 1e-15:
            return result
        if result.returns_history:
            previous = result.returns_history[-1]
            result.returns_history[-1] = (1.0 + previous) * (
                1.0 + liquidation.interval_net_return
            ) - 1.0
        else:
            result.returns_history.append(liquidation.interval_net_return)
        result.peak_value = max(result.peak_value, result.portfolio_value)
        result.max_drawdown = max(
            result.max_drawdown,
            1.0 - max(result.portfolio_value, 0.0) / max(result.peak_value, 1e-12),
        )
        return result

    @staticmethod
    def _liquidation_complete(liquidation: ExecutionResult) -> bool:
        return bool(
            liquidation.unfilled_turnover <= _LIQUIDATION_TOLERANCE
            and np.all(np.abs(liquidation.book.quantities) <= _LIQUIDATION_TOLERANCE)
        )

    def _execution_observation_state(
        self,
        *,
        requested_weights: np.ndarray,
        result: ExecutionResult,
        previous_weights: np.ndarray,
    ) -> ObservationExecutionState:
        requested = np.asarray(requested_weights, dtype=np.float64).reshape(-1)
        requested_notional = result.requested_notional_by_symbol
        filled_notional = result.filled_notional_by_symbol
        if requested_notional.shape != (self.dataset.n_symbols,):
            requested_notional = np.zeros(self.dataset.n_symbols, dtype=np.float64)
        if filled_notional.shape != (self.dataset.n_symbols,):
            filled_notional = np.zeros(self.dataset.n_symbols, dtype=np.float64)
        fill_ratio = np.ones(self.dataset.n_symbols, dtype=np.float64)
        positive = requested_notional > 1e-12
        fill_ratio[positive] = np.minimum(
            1.0,
            filled_notional[positive] / requested_notional[positive],
        )
        total_requested = max(float(requested_notional.sum()), 1e-12)
        unfilled = (
            np.maximum(requested_notional - filled_notional, 0.0) / total_requested
        )
        participation = result.participation_by_symbol
        if participation.shape != (self.dataset.n_symbols,):
            participation = np.zeros(self.dataset.n_symbols, dtype=np.float64)
        cost = result.cost_by_symbol
        if cost.shape != (self.dataset.n_symbols,):
            cost = np.zeros(self.dataset.n_symbols, dtype=np.float64)
        cost = cost / max(self.config.initial_capital, 1e-12)
        current_weights = result.book.weights
        changed = np.abs(current_weights - previous_weights) > 1e-10
        held = np.abs(current_weights) > 1e-10
        self._position_age = np.where(
            held,
            np.where(changed, 0.0, self._position_age + result.bars_advanced),
            0.0,
        )
        return ObservationExecutionState(
            requested_weights=requested,
            fill_ratio=fill_ratio,
            unfilled_turnover=unfilled,
            participation=participation,
            execution_cost=cost,
            position_age=self._position_age.copy(),
        )

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        if self.current_index >= self.end_index:
            raise RuntimeError("step called after the episode ended")
        trends, alpha, factor_basis = self._market_inputs()
        parsed_action, maintained_action, saturated_count, raw_max_abs = (
            self._parse_action(action)
        )
        composition = self.composer.compose(
            parsed_action,
            trends,
            alpha,
            alpha_enabled=self.alpha_enabled,
            factor_basis=factor_basis,
            max_gross=self.pre_trade_risk.config.max_gross,
        )
        hybrid_risk = self._constrain_target(composition.proposal, self.hybrid)
        shadow_risk = self._constrain_target(trends.base, self.shadow)
        bars = self._decision_bar_count()
        previous_hybrid_weights = self.hybrid.weights.copy()
        hybrid_execution = self.hybrid_executor.execute_interval(
            self.hybrid,
            hybrid_risk.weights,
            start_index=self.current_index,
            bars=bars,
        )
        shadow_execution = self.shadow_executor.execute_interval(
            self.shadow,
            shadow_risk.weights,
            start_index=self.current_index,
            bars=bars,
        )
        if hybrid_execution.bars_advanced != shadow_execution.bars_advanced:
            raise RuntimeError("hybrid and shadow books advanced different bar counts")

        self.hybrid = hybrid_execution.book
        self.shadow = shadow_execution.book
        self.current_index = hybrid_execution.next_index
        self._decision_step_index += 1
        time_limit_reached = self.current_index >= self.end_index
        hybrid_log_return = hybrid_execution.interval_log_return
        shadow_log_return = shadow_execution.interval_log_return
        hybrid_liquidation: ExecutionResult | None = None
        shadow_liquidation: ExecutionResult | None = None
        emergency_deleverage = False
        drawdown_after_execution = self._drawdown(self.hybrid)
        drawdown_stop = min(
            self.reward_tracker.config.drawdown_stop,
            self.pre_trade_risk.config.drawdown_stop,
        )
        if (
            not self.hybrid.insolvent
            and drawdown_after_execution + 1e-12 >= drawdown_stop
        ):
            emergency_deleverage = True
            hybrid_liquidation = self.hybrid_executor.liquidate_at_close(
                self.hybrid,
                index=self.current_index,
            )
            if not self._liquidation_complete(hybrid_liquidation):
                raise RuntimeError(
                    "hybrid liquidation could not fully exit at drawdown stop"
                )
            self.hybrid = self._merge_liquidation_return(hybrid_liquidation)
            hybrid_log_return += hybrid_liquidation.interval_log_return
            self.hybrid.terminate(EconomicTerminationReason.DRAWDOWN_STOP)
        liquidation_terminal = (
            time_limit_reached
            and self.config.liquidate_on_end
            and not emergency_deleverage
        )
        liquidation_complete = True
        if liquidation_terminal:
            hybrid_liquidation = self.hybrid_executor.liquidate_at_close(
                self.hybrid,
                index=self.current_index,
            )
            shadow_liquidation = self.shadow_executor.liquidate_at_close(
                self.shadow,
                index=self.current_index,
            )
            liquidation_complete = self._liquidation_complete(
                hybrid_liquidation
            ) and self._liquidation_complete(shadow_liquidation)
            self.hybrid = self._merge_liquidation_return(hybrid_liquidation)
            self.shadow = self._merge_liquidation_return(shadow_liquidation)
            hybrid_log_return += hybrid_liquidation.interval_log_return
            shadow_log_return += shadow_liquidation.interval_log_return

        threshold = self.config.initial_capital * self.config.minimum_equity_fraction
        if self.hybrid.portfolio_value <= threshold and not self.hybrid.insolvent:
            self.hybrid.terminate(EconomicTerminationReason.MINIMUM_EQUITY)
        if self.shadow.portfolio_value <= threshold and not self.shadow.insolvent:
            self.shadow.terminate(EconomicTerminationReason.MINIMUM_EQUITY)
        hybrid_terminated = self.hybrid.insolvent
        shadow_terminated = self.shadow.insolvent
        economic_transition = classify_economic_transition(
            hybrid=self.hybrid,
            shadow=self.shadow,
            time_limit_reached=time_limit_reached,
            liquidation_terminal=liquidation_terminal,
            liquidation_complete=liquidation_complete,
        )
        terminated = economic_transition.terminated
        truncated = economic_transition.truncated
        action_delta_l1 = float(np.abs(maintained_action - self._previous_action).sum())
        projection_distance = hybrid_risk.projection_l1
        reward_breakdown = self.reward_tracker.step(
            hybrid_log_return=hybrid_log_return,
            shadow_log_return=shadow_log_return,
            hybrid_drawdown=self._drawdown(self.hybrid),
            shadow_drawdown=self._drawdown(self.shadow),
            projection_distance=projection_distance,
            hybrid_margin_deficit_fraction=(
                self.hybrid.margin_deficit / self.config.initial_capital
            ),
            hybrid_equity_fraction=max(self.hybrid.portfolio_value, 0.0)
            / self.config.initial_capital,
            shadow_equity_fraction=max(self.shadow.portfolio_value, 0.0)
            / self.config.initial_capital,
            hybrid_terminated=hybrid_terminated,
            shadow_terminated=shadow_terminated,
        )
        self._execution_state = self._execution_observation_state(
            requested_weights=hybrid_risk.weights,
            result=hybrid_execution,
            previous_weights=previous_hybrid_weights,
        )
        self._previous_action = maintained_action.copy()
        self._action_diagnostics.update(
            action=maintained_action,
            saturated_count=saturated_count,
            action_delta_l1=action_delta_l1,
            projection_l1=projection_distance,
            constrained=hybrid_risk.was_constrained,
            turnover_overridden=hybrid_risk.turnover_overridden,
        )
        termination_reason = economic_transition.reason
        info: dict[str, object] = {
            "action_delta_l1": action_delta_l1,
            "action_raw_max_abs": raw_max_abs,
            "action_saturated_count": saturated_count,
            "bars_advanced": hybrid_execution.bars_advanced,
            "composition": composition,
            "decision_step_index": self._decision_step_index,
            "excess_log_return": hybrid_log_return - shadow_log_return,
            "emergency_deleverage": emergency_deleverage,
            "drawdown_after": self._drawdown(self.hybrid),
            "portfolio_value_after": self.hybrid.portfolio_value,
            "reward_growth_raw": reward_breakdown.absolute_log_growth,
            "reward_baseline_penalty_delta": (
                0.0
                if self.reward_tracker.config.baseline_underperformance_weight == 0.0
                else reward_breakdown.baseline_penalty
                / self.reward_tracker.config.baseline_underperformance_weight
            ),
            "reward_baseline_penalty_weighted": reward_breakdown.baseline_penalty,
            "reward_drawdown_penalty_delta": reward_breakdown.incremental_drawdown,
            "reward_drawdown_penalty_weighted": reward_breakdown.drawdown_penalty,
            "reward_total_raw": reward_breakdown.unscaled_total,
            "reward_total_scaled": reward_breakdown.scaled_total,
            "reward_context_before": self.reward_tracker.last_context_before,
            "reward_context_after": self.reward_tracker.last_context_after,
            "rolling_hybrid_log_growth": (
                self.reward_tracker.last_context_after.rolling_hybrid_log_growth
            ),
            "rolling_baseline_log_growth": (
                self.reward_tracker.last_context_after.rolling_shadow_log_growth
            ),
            "rolling_growth_gap": (
                self.reward_tracker.last_context_after.rolling_growth_gap
            ),
            "hybrid_execution": hybrid_execution,
            "hybrid_risk": hybrid_risk,
            "hybrid_terminated": hybrid_terminated,
            "interval_cost": hybrid_execution.interval_cost,
            "interval_funding": hybrid_execution.interval_funding,
            "interval_gross_return": hybrid_execution.interval_gross_return,
            "interval_net_return": hybrid_execution.interval_net_return,
            "liquidation_complete": liquidation_complete,
            "liquidation_terminal": liquidation_terminal,
            "projection_distance_l1": projection_distance,
            "reward_breakdown": reward_breakdown,
            "shadow_execution": shadow_execution,
            "shadow_interval_net_return": shadow_execution.interval_net_return,
            "shadow_risk": shadow_risk,
            "shadow_terminated": shadow_terminated,
            "termination_reason": termination_reason,
        }
        if hybrid_liquidation is not None:
            info["hybrid_liquidation"] = hybrid_liquidation
        if shadow_liquidation is not None:
            info["shadow_liquidation"] = shadow_liquidation
        if terminated or truncated:
            info.update(self._terminal_info())
        return (
            self._observation(),
            reward_breakdown.scaled_total,
            terminated,
            truncated,
            info,
        )

    def _book_metrics(self, book: BookState) -> PerformanceMetrics:
        return evaluate_performance(
            ReturnSeries(
                values=tuple(book.returns_history),
                kind=ReturnKind.BASE_BAR,
                periods_per_year=self.dataset.periods_per_year,
            ),
            turnover_total=book.turnover_total,
            total_cost=book.total_cost,
            funding_pnl=book.funding_pnl - book.borrow_cost,
            n_trades=book.fill_count,
        )

    def _terminal_info(self) -> dict[str, object]:
        hybrid_metrics = self._book_metrics(self.hybrid)
        shadow_metrics = self._book_metrics(self.shadow)
        return {
            "episode_hours": self._episode_hours,
            "episode_seed": self._episode_seed,
            "action_diagnostics": self._action_diagnostics.snapshot(),
            "hybrid_metrics": hybrid_metrics,
            "hybrid_rebalance_events": self.hybrid.rebalance_events,
            "initial_state_mode": self._initial_state_mode,
            "shadow_metrics": shadow_metrics,
            "shadow_rebalance_events": self.shadow.rebalance_events,
            "excess_total_return": (
                hybrid_metrics.total_return - shadow_metrics.total_return
            ),
        }
