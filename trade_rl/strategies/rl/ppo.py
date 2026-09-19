"""Teacher-free PPO adapter for the lean per-symbol strategy contract."""

from __future__ import annotations

import importlib
import math
from functools import partial
from numbers import Integral
from typing import Protocol, cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from trade_rl.data.market import MarketDataset
from trade_rl.risk import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation import BookState, ExecutionCostConfig, MarketExecutor
from trade_rl.strategies.dataset_scope import (
    validated_feature_indices,
    validated_symbol_indices,
)
from trade_rl.strategies.interface import StrategyObservation
from trade_rl.strategies.position_intent import (
    PositionIntent,
    target_weight_for_intent,
)
from trade_rl.strategies.rl.ppo_normalization import (
    PPOFeatureNormalizer,
    fit_ppo_feature_normalizer,
)

PPO_OBSERVATION_SCHEMA = "ppo_observation_v2"
PPO_GLOBAL_FEATURE_NAMES: tuple[str, ...] = ()
PPO_TRAINING_LAYOUT_SEQUENTIAL = "sequential"
PPO_TRAINING_LAYOUT_INTERLEAVED = "interleaved"
_PPO_BATCH_SIZE = 64


class _PredictPolicy(Protocol):
    def predict(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool = True,
    ) -> tuple[object, object]: ...


def ppo_observation_contract_payload() -> dict[str, object]:
    """Return the frozen semantic PPO observation contract for persisted evidence."""

    return {
        "schema_version": PPO_OBSERVATION_SCHEMA,
        "global_feature_names": list(PPO_GLOBAL_FEATURE_NAMES),
        "includes_local_feature_staleness": True,
        "layout": [
            "local_values",
            "local_available",
            "local_staleness",
            "current_intent",
            "current_weight",
        ],
    }


def _validated_indices(feature_indices: tuple[int, ...]) -> tuple[int, ...]:
    indices = tuple(feature_indices)
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("feature_indices must be non-empty and unique")
    if any(
        isinstance(index, bool) or not isinstance(index, int) or index < 0
        for index in indices
    ):
        raise ValueError("feature_indices must contain non-negative integers")
    return indices


def _validated_training_layout(
    training_layout: str,
    rollout_steps_per_env: int | None,
) -> str:
    if training_layout not in {
        PPO_TRAINING_LAYOUT_SEQUENTIAL,
        PPO_TRAINING_LAYOUT_INTERLEAVED,
    }:
        raise ValueError("training_layout must be 'sequential' or 'interleaved'")
    if (
        training_layout == PPO_TRAINING_LAYOUT_SEQUENTIAL
        and rollout_steps_per_env is not None
    ):
        raise ValueError("sequential training does not accept rollout_steps_per_env")
    return training_layout


def _validated_interleaved_rollout_steps(
    rollout_steps_per_env: int | None,
    *,
    n_envs: int,
) -> int:
    if (
        isinstance(rollout_steps_per_env, bool)
        or not isinstance(rollout_steps_per_env, int)
        or rollout_steps_per_env <= 0
    ):
        raise ValueError(
            "rollout_steps_per_env must be a positive integer for interleaved training"
        )
    if rollout_steps_per_env * n_envs % _PPO_BATCH_SIZE != 0:
        raise ValueError(
            "interleaved rollout batch must be divisible by PPO batch_size=64"
        )
    return rollout_steps_per_env


def _encode_observation(
    observation: StrategyObservation,
    feature_indices: tuple[int, ...],
    feature_normalizer: PPOFeatureNormalizer | None = None,
) -> np.ndarray:
    indices = _validated_indices(feature_indices)
    if max(indices) >= observation.features.size:
        raise ValueError("feature index is outside observation features")

    selected = np.asarray(observation.features[list(indices)], dtype=np.float64)
    available = np.asarray(
        observation.feature_available[list(indices)],
        dtype=np.bool_,
    )
    if observation.feature_staleness is None:
        raise ValueError("PPO Observation v2 requires feature staleness")
    staleness = np.asarray(
        observation.feature_staleness[list(indices)],
        dtype=np.float64,
    )
    finite = np.isfinite(selected)
    usable = available & finite
    values = np.where(usable, selected, 0.0)
    if feature_normalizer is not None:
        values = feature_normalizer.transform(selected, usable)

    state = np.asarray(
        [float(observation.current_intent), observation.current_weight],
        dtype=np.float64,
    )
    encoded = np.concatenate(
        (
            values,
            usable.astype(np.float64),
            staleness,
            state,
        ),
    ).astype(np.float32)
    encoded.setflags(write=False)
    return encoded


def _intent_from_action(action: object) -> PositionIntent:
    values = np.asarray(action).reshape(-1)
    if values.size != 1:
        raise ValueError("PPO action must contain exactly one value")
    raw = values[0]
    if isinstance(raw, (bool, np.bool_)) or not isinstance(raw, Integral):
        raise ValueError("PPO action must be an integer in {0, 1, 2}")
    value = int(raw)
    if value not in {0, 1, 2}:
        raise ValueError("PPO action must be an integer in {0, 1, 2}")
    return PositionIntent(value - 1)


def _desired_quantity_from_weight(
    book: BookState,
    target_weight: float,
    *,
    symbol_index: int,
) -> float:
    multipliers = np.asarray(book.contract_multipliers, dtype=np.float64)
    denominator = float(book.mark_prices[symbol_index] * multipliers[symbol_index])
    return float(target_weight * book.portfolio_value / denominator)


def _weight_for_desired_quantity(
    book: BookState,
    desired_quantity: float,
    *,
    symbol_index: int,
) -> float:
    if book.portfolio_value <= 0.0:
        return 0.0
    multipliers = np.asarray(book.contract_multipliers, dtype=np.float64)
    return float(
        desired_quantity
        * book.mark_prices[symbol_index]
        * multipliers[symbol_index]
        / book.portfolio_value
    )


def _default_risk(executor: MarketExecutor) -> PreTradeRisk:
    hard_limit = min(1.0, float(executor.cost.max_leverage))
    return PreTradeRisk(
        PreTradeRiskConfig(
            max_gross=hard_limit,
            max_abs_weight=hard_limit,
            max_turnover=None,
            drawdown_start=1.0,
            drawdown_stop=1.0,
        )
    )


class PPOIntentStrategy:
    """Map a deterministic three-action policy to SHORT/FLAT/LONG intent."""

    def __init__(
        self,
        policy: _PredictPolicy,
        *,
        feature_indices: tuple[int, ...],
        feature_normalizer: PPOFeatureNormalizer | None = None,
    ) -> None:
        self.policy = policy
        self.feature_indices = _validated_indices(feature_indices)
        if feature_normalizer is not None:
            feature_normalizer.validate_features(self.feature_indices)
        self.feature_normalizer = feature_normalizer

    def decide(self, observation: StrategyObservation) -> PositionIntent:
        encoded = _encode_observation(
            observation,
            self.feature_indices,
            self.feature_normalizer,
        )
        action, _ = self.policy.predict(encoded, deterministic=True)
        return _intent_from_action(action)


class PPOTradingEnv(gym.Env):
    """Per-symbol episodes sharing one policy and one observation schema."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        dataset: MarketDataset,
        *,
        feature_indices: tuple[int, ...],
        symbol_indices: tuple[int, ...] | None = None,
        start_index: int,
        stop_index: int,
        gross_budget: float,
        initial_capital: float = 100_000.0,
        execution_cost: ExecutionCostConfig | None = None,
        risk_config: PreTradeRiskConfig | None = None,
        feature_normalizer: PPOFeatureNormalizer | None = None,
    ) -> None:
        super().__init__()
        if risk_config is not None and not isinstance(risk_config, PreTradeRiskConfig):
            raise ValueError("risk_config must be a PreTradeRiskConfig or None")
        if dataset.n_symbols <= 0:
            raise ValueError("PPOTradingEnv requires at least one symbol")
        if (
            isinstance(start_index, bool)
            or not isinstance(start_index, int)
            or isinstance(stop_index, bool)
            or not isinstance(stop_index, int)
            or not 0 <= start_index < stop_index < dataset.n_bars
        ):
            raise ValueError(
                "environment range must satisfy 0 <= start < stop < n_bars"
            )
        if not math.isfinite(initial_capital) or initial_capital <= 0.0:
            raise ValueError("initial_capital must be finite and positive")
        target_weight_for_intent(PositionIntent.LONG, gross_budget=gross_budget)

        self.dataset = dataset
        self.feature_indices = validated_feature_indices(dataset, feature_indices)
        self.symbol_indices = validated_symbol_indices(dataset, symbol_indices)
        self.start_index = start_index
        self.stop_index = stop_index
        self.gross_budget = gross_budget
        self.initial_capital = initial_capital
        self.execution_cost = execution_cost or ExecutionCostConfig.zero()
        self.risk_config = risk_config
        if feature_normalizer is not None:
            feature_normalizer.validate_features(self.feature_indices)
            feature_normalizer.validate_training_scope(
                dataset, self.symbol_indices, start_index, stop_index
            )
        self.feature_normalizer = feature_normalizer
        if risk_config is not None and (
            risk_config.max_gross > self.execution_cost.max_leverage
            or risk_config.max_abs_weight > self.execution_cost.max_leverage
        ):
            raise ValueError(
                "risk max_gross/max_abs_weight must not exceed execution max_leverage"
            )

        observation_size = 3 * len(self.feature_indices) + 2
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_size,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)

        self.active_symbol_index = -1
        self._active_symbol_offset = -1
        self._execution_seed_stream: np.random.Generator | None = None
        self.executor = MarketExecutor(self.dataset, self.execution_cost)
        self.risk = (
            _default_risk(self.executor)
            if self.risk_config is None
            else PreTradeRisk(self.risk_config)
        )
        self.book = self._initial_book()
        self.current_intent = PositionIntent.FLAT
        self.desired_quantity = 0.0
        self.index = self.start_index
        self._terminated = False

    def _initial_book(self) -> BookState:
        initial_prices = self.dataset.resolved_array("mark_price")[self.start_index]
        return BookState.zero(
            self.dataset.n_symbols,
            self.initial_capital,
            initial_prices,
            contract_multipliers=self.dataset.contract_multipliers,
        )

    def _strategy_observation(self) -> StrategyObservation:
        if self.active_symbol_index < 0:
            raise RuntimeError("PPOTradingEnv must be reset before observation")
        symbol_index = self.active_symbol_index
        return StrategyObservation(
            index=self.index,
            timestamp=self.dataset.timestamps[self.index],
            symbol=self.dataset.symbols[symbol_index],
            features=self.dataset.features[self.index, symbol_index],
            feature_available=self.dataset.feature_available[self.index, symbol_index],
            feature_staleness=self.dataset.resolved_array("feature_staleness")[
                self.index, symbol_index
            ],
            global_features=self.dataset.global_features[self.index],
            global_feature_available=self.dataset.resolved_array(
                "global_feature_available"
            )[self.index],
            current_intent=self.current_intent,
            current_weight=float(self.book.weights[symbol_index]),
        )

    def _encoded_observation(self) -> np.ndarray:
        return _encode_observation(
            self._strategy_observation(),
            self.feature_indices,
            self.feature_normalizer,
        ).copy()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        del options
        super().reset(seed=seed)
        if seed is not None:
            self._active_symbol_offset = -1
        self._active_symbol_offset = (self._active_symbol_offset + 1) % len(
            self.symbol_indices
        )
        self.active_symbol_index = self.symbol_indices[self._active_symbol_offset]
        if seed is not None:
            self._execution_seed_stream = np.random.default_rng(seed)
            execution_seed = seed
        elif self._execution_seed_stream is None:
            execution_seed = self.execution_cost.random_seed
            self._execution_seed_stream = np.random.default_rng(execution_seed)
        else:
            execution_seed = int(
                self._execution_seed_stream.integers(0, np.iinfo(np.int64).max)
            )
        self.executor = MarketExecutor(self.dataset, self.execution_cost)
        self.executor.reset_random_state(execution_seed)
        self.risk = (
            _default_risk(self.executor)
            if self.risk_config is None
            else PreTradeRisk(self.risk_config)
        )
        self.book = self._initial_book()
        self.current_intent = PositionIntent.FLAT
        self.desired_quantity = 0.0
        self.index = self.start_index
        self._terminated = False
        return self._encoded_observation(), {
            "symbol_index": self.active_symbol_index,
            "symbol": self.dataset.symbols[self.active_symbol_index],
        }

    def step(
        self,
        action: object,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        if self._terminated:
            raise RuntimeError("cannot step a terminated PPOTradingEnv")
        if self.active_symbol_index < 0:
            raise RuntimeError("PPOTradingEnv must be reset before stepping")

        symbol_index = self.active_symbol_index
        intent = _intent_from_action(action)
        changed_intent = intent is not self.current_intent
        if changed_intent:
            proposal_weight = target_weight_for_intent(
                intent,
                gross_budget=self.gross_budget,
            )
            self.desired_quantity = _desired_quantity_from_weight(
                self.book,
                proposal_weight,
                symbol_index=symbol_index,
            )

        proposal_weight = _weight_for_desired_quantity(
            self.book,
            self.desired_quantity,
            symbol_index=symbol_index,
        )
        proposal_weights = np.zeros(self.dataset.n_symbols, dtype=np.float64)
        proposal_weights[symbol_index] = proposal_weight
        constrained = self.risk.constrain(
            proposal_weights,
            current=self.book.weights,
            drawdown=self.book.max_drawdown,
        )
        target_weight = float(constrained.weights[symbol_index])
        if (
            constrained.was_constrained
            and "drawdown_deleveraging" not in constrained.reasons
            and any(reason != "max_turnover" for reason in constrained.reasons)
        ):
            self.desired_quantity = _desired_quantity_from_weight(
                self.book,
                target_weight,
                symbol_index=symbol_index,
            )

        execution = self.executor.execute_interval(
            self.book,
            constrained.weights,
            start_index=self.index,
            bars=1,
        )
        if execution.next_index <= self.index:
            raise RuntimeError("execution did not advance PPO environment index")

        self.book = execution.book
        self.current_intent = intent
        self.index = execution.next_index
        reward = math.log1p(execution.interval_net_return)
        self._terminated = (
            self.index >= self.stop_index or self.book.termination_reason is not None
        )
        observation = self._encoded_observation()
        info: dict[str, object] = {
            "symbol_index": symbol_index,
            "symbol": self.dataset.symbols[symbol_index],
            "intent": intent,
            "target_weight": target_weight,
            "interval_net_return": execution.interval_net_return,
        }
        return observation, reward, self._terminated, False, info


def fit_ppo_strategy(
    dataset: MarketDataset,
    *,
    feature_indices: tuple[int, ...],
    fit_symbol_indices: tuple[int, ...] | None = None,
    start_index: int,
    stop_index: int,
    gross_budget: float,
    total_timesteps: int,
    seed: int = 0,
    initial_capital: float = 100_000.0,
    execution_cost: ExecutionCostConfig | None = None,
    training_layout: str = PPO_TRAINING_LAYOUT_SEQUENTIAL,
    rollout_steps_per_env: int | None = None,
    risk_config: PreTradeRiskConfig | None = None,
    normalize_features: bool = False,
) -> PPOIntentStrategy:
    """Fit one teacher-free policy with an explicit multi-symbol training layout."""

    if (
        isinstance(total_timesteps, bool)
        or not isinstance(total_timesteps, int)
        or total_timesteps <= 0
    ):
        raise ValueError("total_timesteps must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    layout = _validated_training_layout(training_layout, rollout_steps_per_env)

    indices = validated_feature_indices(dataset, feature_indices)
    if not isinstance(normalize_features, bool):
        raise ValueError("normalize_features must be boolean")
    normalizer = (
        fit_ppo_feature_normalizer(
            dataset,
            feature_indices=indices,
            fit_symbol_indices=fit_symbol_indices,
            start_index=start_index,
            stop_index=stop_index,
        )
        if normalize_features
        else None
    )
    ppo_options: dict[str, object] = {}
    if layout == PPO_TRAINING_LAYOUT_SEQUENTIAL:
        env: object = PPOTradingEnv(
            dataset,
            feature_indices=indices,
            symbol_indices=fit_symbol_indices,
            start_index=start_index,
            stop_index=stop_index,
            gross_budget=gross_budget,
            initial_capital=initial_capital,
            execution_cost=execution_cost,
            risk_config=risk_config,
            feature_normalizer=normalizer,
        )
    else:
        if execution_cost is not None and execution_cost.slippage_std > 0.0:
            raise ValueError(
                "interleaved training requires deterministic execution slippage"
            )
        symbol_indices = validated_symbol_indices(dataset, fit_symbol_indices)
        rollout_steps = _validated_interleaved_rollout_steps(
            rollout_steps_per_env,
            n_envs=len(symbol_indices),
        )
        try:
            vector_module = importlib.import_module("stable_baselines3.common.vec_env")
            dummy_vec_env = getattr(vector_module, "DummyVecEnv")
        except (ImportError, AttributeError) as error:
            raise RuntimeError(
                "stable-baselines3 is required; install the train-sb3 extra"
            ) from error
        env = dummy_vec_env(
            [
                partial(
                    PPOTradingEnv,
                    dataset,
                    feature_indices=indices,
                    symbol_indices=(symbol_index,),
                    start_index=start_index,
                    stop_index=stop_index,
                    gross_budget=gross_budget,
                    initial_capital=initial_capital,
                    execution_cost=execution_cost,
                    risk_config=risk_config,
                    feature_normalizer=normalizer,
                )
                for symbol_index in symbol_indices
            ]
        )
        ppo_options = {
            "n_steps": rollout_steps,
            "batch_size": _PPO_BATCH_SIZE,
        }

    try:
        module = importlib.import_module("stable_baselines3")
        ppo_class = getattr(module, "PPO")
        torch_module = importlib.import_module("torch")
        set_num_threads = getattr(torch_module, "set_num_threads")
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "stable-baselines3 and torch are required; install the train-sb3 extra"
        ) from error
    set_num_threads(1)

    model = ppo_class(
        "MlpPolicy",
        env,
        policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64, 64]}},
        seed=seed,
        ent_coef=0.0,
        device="cpu",
        verbose=0,
        **ppo_options,
    )
    model.learn(total_timesteps=total_timesteps)
    return PPOIntentStrategy(
        cast(_PredictPolicy, model),
        feature_indices=indices,
        feature_normalizer=normalizer,
    )


__all__ = [
    "PPO_GLOBAL_FEATURE_NAMES",
    "PPO_OBSERVATION_SCHEMA",
    "PPO_TRAINING_LAYOUT_INTERLEAVED",
    "PPO_TRAINING_LAYOUT_SEQUENTIAL",
    "PPOIntentStrategy",
    "PPOTradingEnv",
    "fit_ppo_strategy",
    "ppo_observation_contract_payload",
]
