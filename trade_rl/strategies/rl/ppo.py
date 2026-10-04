"""Teacher-free PPO adapter for the lean per-symbol strategy contract."""

from __future__ import annotations

import importlib
import math
from functools import partial
from typing import cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from trade_rl.data.market import MarketDataset
from trade_rl.risk import PreTradeRisk, PreTradeRiskConfig
from trade_rl.risk.pretrade import should_rebind_strategy_proposal
from trade_rl.simulation import BookState, ExecutionCostConfig, MarketExecutor
from trade_rl.strategies.dataset_scope import (
    validated_symbol_indices,
    validated_training_scope,
)
from trade_rl.strategies.interface import StrategyObservation
from trade_rl.strategies.position_duration import (
    constrain_intent_for_minimum_hold,
    next_position_age_bars,
)
from trade_rl.strategies.position_intent import (
    PositionIntent,
    target_weight_for_intent,
)
from trade_rl.strategies.rl.intent import (
    PPO_GLOBAL_FEATURE_NAMES,
    PPO_OBSERVATION_SCHEMA,
    PPO_OBSERVATION_SCHEMA_V3,
    PPO_OBSERVATION_SCHEMAS,
    _encode_observation_fields,
    _intent_from_action,
    _PredictPolicy,
    _ThreeActionIntentStrategy,
    ppo_observation_contract_payload,
)
from trade_rl.strategies.rl.ppo_normalization import (
    PPOFeatureNormalizer,
    fit_ppo_feature_normalizer,
)
from trade_rl.strategies.rl.ppo_training import (
    PPO_DEFAULT_GAE_LAMBDA,
    PPO_DEFAULT_GAMMA,
    PPO_DEFAULT_N_STEPS,
    PPO_MINIBATCH_SIZE,
    PPO_NORMALIZE_ADVANTAGE,
    PPO_TRAINING_LAYOUT_INTERLEAVED,
    PPO_TRAINING_LAYOUT_SEQUENTIAL,
    expected_ppo_realized_timesteps,
    validated_interleaved_rollout_steps,
    validated_ppo_gamma,
    validated_training_layout,
)

_PPO_LEARNING_RATE = 3e-4
_PPO_DEFAULT_N_STEPS = PPO_DEFAULT_N_STEPS
_PPO_BATCH_SIZE = PPO_MINIBATCH_SIZE
_PPO_N_EPOCHS = 10
_PPO_GAMMA = PPO_DEFAULT_GAMMA
_PPO_GAE_LAMBDA = PPO_DEFAULT_GAE_LAMBDA
_PPO_CLIP_RANGE = 0.2
_PPO_CLIP_RANGE_VF: float | None = None
_PPO_NORMALIZE_ADVANTAGE = PPO_NORMALIZE_ADVANTAGE
_PPO_ENT_COEF = 0.0
_PPO_VF_COEF = 0.5
_PPO_MAX_GRAD_NORM = 0.5
_PPO_USE_SDE = False
_PPO_SDE_SAMPLE_FREQ = -1
_PPO_TARGET_KL: float | None = None


def _validated_training_layout(
    training_layout: str,
    rollout_steps_per_env: int | None,
) -> str:
    return validated_training_layout(training_layout, rollout_steps_per_env)


def _validated_interleaved_rollout_steps(
    rollout_steps_per_env: int | None,
    *,
    n_envs: int,
) -> int:
    return validated_interleaved_rollout_steps(
        rollout_steps_per_env,
        n_envs=n_envs,
    )


def _validate_sequential_symbol_coverage(
    total_timesteps: int,
    *,
    episode_steps: int,
    n_symbols: int,
    rollout_steps: int = _PPO_DEFAULT_N_STEPS,
    algorithm: str = "PPO",
    require_single_symbol_coverage: bool = False,
) -> None:
    """Check nominal full-window budget capacity across sequential fit symbols."""

    if episode_steps <= 0 or (n_symbols <= 1 and not require_single_symbol_coverage):
        return
    effective_timesteps = (
        (total_timesteps + rollout_steps - 1) // rollout_steps
    ) * rollout_steps
    required_timesteps = episode_steps * n_symbols
    if effective_timesteps < required_timesteps:
        raise ValueError(
            f"sequential {algorithm} requires nominal budget capacity for at least "
            "one full episode per fit symbol; symbol coverage budget check failed; "
            f"rollout-rounded budget={effective_timesteps}, "
            f"required={required_timesteps}"
        )


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


def _agent_stop_index(
    *,
    start_index: int,
    stop_index: int,
    execution_cost: ExecutionCostConfig,
    settle_terminal_position: bool,
) -> int:
    if not settle_terminal_position:
        return stop_index
    agent_stop = stop_index - execution_cost.order_latency_bars - 1
    if agent_stop <= start_index:
        raise ValueError(
            "terminal settlement requires at least one agent interval before the close"
        )
    return agent_stop


class PPOIntentStrategy(_ThreeActionIntentStrategy):
    """PPO-named adapter over the shared three-action intent contract."""

    _family_name = "PPO"


class PPOTradingEnv(gym.Env):
    """Per-symbol episodes sharing one policy and one observation schema."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        dataset: MarketDataset,
        *,
        feature_indices: tuple[int, ...],
        symbol_indices: tuple[int, ...] | None = None,
        information_symbol_indices: tuple[int, ...] | None = None,
        start_index: int,
        stop_index: int,
        gross_budget: float,
        initial_capital: float = 100_000.0,
        execution_cost: ExecutionCostConfig | None = None,
        risk_config: PreTradeRiskConfig | None = None,
        feature_normalizer: PPOFeatureNormalizer | None = None,
        settle_terminal_position: bool = False,
        minimum_hold_bars: int = 0,
        observation_schema: str = PPO_OBSERVATION_SCHEMA,
    ) -> None:
        super().__init__()
        if risk_config is not None and not isinstance(risk_config, PreTradeRiskConfig):
            raise ValueError("risk_config must be a PreTradeRiskConfig or None")
        if not isinstance(settle_terminal_position, bool):
            raise ValueError("settle_terminal_position must be boolean")
        if (
            isinstance(minimum_hold_bars, bool)
            or not isinstance(minimum_hold_bars, int)
            or minimum_hold_bars < 0
        ):
            raise ValueError("minimum_hold_bars must be a non-negative integer")
        if observation_schema not in PPO_OBSERVATION_SCHEMAS:
            raise ValueError("unsupported PPO observation schema")
        if minimum_hold_bars > 0 and observation_schema != PPO_OBSERVATION_SCHEMA_V3:
            raise ValueError("PPO minimum hold requires the age-aware observation")
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
        self.symbol_indices = validated_symbol_indices(dataset, symbol_indices)
        information_scope = (
            self.symbol_indices
            if information_symbol_indices is None
            else information_symbol_indices
        )
        (
            self.feature_indices,
            self.information_symbol_indices,
        ) = validated_training_scope(
            dataset,
            feature_indices=feature_indices,
            fit_symbol_indices=information_scope,
        )
        self._feature_index_array = np.asarray(self.feature_indices, dtype=np.intp)
        self._feature_index_array.setflags(write=False)
        self._feature_staleness = dataset.resolved_array("feature_staleness")
        if not set(self.symbol_indices).issubset(self.information_symbol_indices):
            raise ValueError(
                "symbol_indices must be contained in information_symbol_indices"
            )
        self.start_index = start_index
        self.stop_index = stop_index
        self.gross_budget = gross_budget
        self.initial_capital = initial_capital
        self.execution_cost = execution_cost or ExecutionCostConfig.zero()
        self.risk_config = risk_config
        self.settle_terminal_position = settle_terminal_position
        self.minimum_hold_bars = minimum_hold_bars
        self.observation_schema = observation_schema
        self.agent_stop_index = _agent_stop_index(
            start_index=start_index,
            stop_index=stop_index,
            execution_cost=self.execution_cost,
            settle_terminal_position=settle_terminal_position,
        )
        if feature_normalizer is not None:
            feature_normalizer.validate_features(self.feature_indices)
            feature_normalizer.validate_training_scope(
                dataset,
                self.information_symbol_indices,
                start_index,
                self.agent_stop_index,
            )
        self.feature_normalizer = feature_normalizer
        if risk_config is not None and (
            risk_config.max_gross > self.execution_cost.max_leverage
            or risk_config.max_abs_weight > self.execution_cost.max_leverage
        ):
            raise ValueError(
                "risk max_gross/max_abs_weight must not exceed execution max_leverage"
            )

        observation_size = (
            3 * len(self.feature_indices)
            + 2
            + int(self.observation_schema == PPO_OBSERVATION_SCHEMA_V3)
        )
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
            PreTradeRisk.default_for_execution(
                max_leverage=self.executor.cost.max_leverage
            )
            if self.risk_config is None
            else PreTradeRisk(self.risk_config)
        )
        self._default_risk_weight_limit = min(
            self.risk.config.max_gross,
            self.risk.config.max_abs_weight,
        )
        self._proposal_weights = np.zeros(dataset.n_symbols, dtype=np.float64)
        self.book = self._initial_book()
        self.current_intent = PositionIntent.FLAT
        self.position_age_bars = 0
        self._minimum_hold_locked = False
        self.minimum_hold_suppressed_count = 0
        self.minimum_hold_decision_count = 0
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
            position_age_bars=self.position_age_bars,
        )

    def _encoded_observation(self) -> np.ndarray:
        symbol_index = self.active_symbol_index
        if symbol_index < 0:
            raise RuntimeError("PPOTradingEnv must be reset before observation")
        return _encode_observation_fields(
            self.dataset.features[self.index, symbol_index, self._feature_index_array],
            self.dataset.feature_available[
                self.index, symbol_index, self._feature_index_array
            ],
            self._feature_staleness[
                self.index, symbol_index, self._feature_index_array
            ],
            self.current_intent,
            float(self.book.weights[symbol_index]),
            self.feature_normalizer,
            position_age_bars=self.position_age_bars,
            observation_schema=self.observation_schema,
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
            PreTradeRisk.default_for_execution(
                max_leverage=self.executor.cost.max_leverage
            )
            if self.risk_config is None
            else PreTradeRisk(self.risk_config)
        )
        self.book = self._initial_book()
        self.current_intent = PositionIntent.FLAT
        self.position_age_bars = 0
        self._minimum_hold_locked = False
        self.desired_quantity = 0.0
        self.index = self.start_index
        self._terminated = False
        return self._encoded_observation(), {
            "symbol_index": self.active_symbol_index,
            "symbol": self.dataset.symbols[self.active_symbol_index],
        }

    def _settle_terminal_position(self) -> tuple[float, dict[str, object]]:
        if not self.settle_terminal_position:
            return 0.0, {}
        if self.active_symbol_index < 0:
            raise RuntimeError("PPOTradingEnv must be reset before settlement")
        if self.index < self.agent_stop_index:
            return 0.0, {}

        symbol_index = self.active_symbol_index
        settlement_start_weight = float(self.book.weights[symbol_index])
        settlement_log_return = 0.0
        settlement_intervals = 0
        settlement_cost = 0.0
        settlement_funding = 0.0
        settlement_borrow = 0.0
        settlement_dividend = 0.0
        settlement_cash_interest = 0.0
        settlement_requested_turnover = 0.0
        settlement_filled_turnover = 0.0
        settlement_risk_reasons: list[str] = []

        self.current_intent = PositionIntent.FLAT
        self.desired_quantity = 0.0
        while self.index < self.stop_index and self.book.termination_reason is None:
            quantity_before = float(self.book.quantities[symbol_index])
            proposal_weights = np.zeros(self.dataset.n_symbols, dtype=np.float64)
            constrained = self.risk.constrain(
                proposal_weights,
                current=self.book.weights,
                drawdown=self.book.max_drawdown,
            )
            settlement_risk_reasons.extend(constrained.reasons)
            execution = self.executor.execute_interval(
                self.book,
                constrained.weights,
                start_index=self.index,
                bars=1,
            )
            if execution.next_index <= self.index:
                raise RuntimeError(
                    "terminal settlement did not advance PPO environment"
                )
            self.book = execution.book
            self.index = execution.next_index
            self.position_age_bars = next_position_age_bars(
                self.position_age_bars,
                previous_quantity=quantity_before,
                filled_quantity=float(self.book.quantities[symbol_index]),
            )
            settlement_log_return += math.log1p(execution.interval_net_return)
            settlement_intervals += 1
            settlement_cost += execution.interval_cost
            settlement_funding += execution.interval_funding
            settlement_borrow += execution.interval_borrow_cost
            settlement_dividend += execution.interval_dividend
            settlement_cash_interest += execution.interval_cash_interest
            settlement_requested_turnover += execution.requested_turnover
            settlement_filled_turnover += execution.filled_turnover

        return settlement_log_return, {
            "terminal_settlement_intervals": settlement_intervals,
            "terminal_settlement_start_weight": settlement_start_weight,
            "terminal_settlement_log_return": settlement_log_return,
            "terminal_settlement_net_return": math.expm1(settlement_log_return),
            "terminal_settlement_cost_amount": settlement_cost,
            "terminal_settlement_funding_amount": settlement_funding,
            "terminal_settlement_borrow_cost_amount": settlement_borrow,
            "terminal_settlement_dividend_amount": settlement_dividend,
            "terminal_settlement_cash_interest_amount": settlement_cash_interest,
            "terminal_settlement_requested_turnover": settlement_requested_turnover,
            "terminal_settlement_filled_turnover": settlement_filled_turnover,
            "terminal_settlement_risk_reasons": tuple(settlement_risk_reasons),
            "terminal_settlement_final_weight": float(self.book.weights[symbol_index]),
            "terminal_settlement_termination_reason": self.book.termination_reason,
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
        requested_intent = _intent_from_action(action)
        was_minimum_hold_locked = self._minimum_hold_locked
        hold_decision = constrain_intent_for_minimum_hold(
            requested_intent,
            current_quantity=float(self.book.quantities[symbol_index]),
            position_age_bars=self.position_age_bars,
            minimum_hold_bars=self.minimum_hold_bars,
        )
        minimum_hold_unlocked = was_minimum_hold_locked and not hold_decision.suppressed
        self.minimum_hold_decision_count += 1
        if hold_decision.suppressed:
            self.minimum_hold_suppressed_count += 1
        intent = hold_decision.effective_intent
        changed_intent = intent is not self.current_intent
        if hold_decision.target_quantity_override is not None:
            self.desired_quantity = hold_decision.target_quantity_override
        elif changed_intent or minimum_hold_unlocked:
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
        proposal_weights = self._proposal_weights
        proposal_weights.fill(0.0)
        proposal_weights[symbol_index] = proposal_weight
        drawdown = self.book.max_drawdown
        if (
            self.risk_config is None
            and math.isfinite(proposal_weight)
            and abs(proposal_weight) <= self._default_risk_weight_limit
            and math.isfinite(drawdown)
            and 0.0 <= drawdown <= 1.0
        ):
            target_weights = proposal_weights
            was_constrained = False
            risk_reasons: tuple[str, ...] = ()
        else:
            constrained = self.risk.constrain(
                proposal_weights,
                current=self.book.weights,
                drawdown=drawdown,
            )
            target_weights = constrained.weights
            was_constrained = constrained.was_constrained
            risk_reasons = constrained.reasons
            target_weight = float(target_weights[symbol_index])
            if should_rebind_strategy_proposal(constrained):
                self.desired_quantity = _desired_quantity_from_weight(
                    self.book,
                    target_weight,
                    symbol_index=symbol_index,
                )
        target_weight = float(target_weights[symbol_index])

        # The environment owns this history; avoid copying its full prefix into
        # the transactional execution clone on every one-bar training step.
        return_history = self.book.returns_history
        self.book.returns_history = []
        try:
            execution = self.executor.execute_interval(
                self.book,
                target_weights,
                start_index=self.index,
                bars=1,
            )
            if execution.next_index <= self.index:
                raise RuntimeError("execution did not advance PPO environment index")
        except BaseException:
            self.book.returns_history = return_history
            raise

        interval_returns = execution.book.returns_history
        execution.book.returns_history = return_history
        return_history.extend(interval_returns)

        quantity_before = float(self.book.quantities[symbol_index])
        self.desired_quantity *= float(
            self.dataset.resolved_array("split_factor")[
                execution.next_index, symbol_index
            ]
        )
        self.book = execution.book
        self.current_intent = intent
        self.index = execution.next_index
        self.position_age_bars = next_position_age_bars(
            self.position_age_bars,
            previous_quantity=quantity_before,
            filled_quantity=float(self.book.quantities[symbol_index]),
        )
        self._minimum_hold_locked = (
            hold_decision.suppressed
            and float(self.book.quantities[symbol_index]) != 0.0
        )
        realized_weight = float(self.book.weights[symbol_index])
        reward = math.log1p(execution.interval_net_return)
        settlement_info: dict[str, object] = {}
        if (
            self.settle_terminal_position
            and self.book.termination_reason is None
            and self.index >= self.agent_stop_index
        ):
            settlement_log_return, settlement_info = self._settle_terminal_position()
            reward += settlement_log_return
        self._terminated = (
            self.index >= self.stop_index or self.book.termination_reason is not None
        )
        observation = self._encoded_observation()
        info: dict[str, object] = {
            "symbol_index": symbol_index,
            "symbol": self.dataset.symbols[symbol_index],
            "intent": requested_intent,
            "requested_intent": requested_intent,
            "effective_intent": intent,
            "minimum_hold_suppressed": hold_decision.suppressed,
            "minimum_hold_unlocked": minimum_hold_unlocked,
            "target_quantity_override": hold_decision.target_quantity_override,
            "position_age_bars": self.position_age_bars,
            "target_weight": target_weight,
            "realized_weight": realized_weight,
            "was_constrained": was_constrained,
            "risk_reasons": risk_reasons,
            "interval_net_return": execution.interval_net_return,
            "interval_cost_amount": execution.interval_cost,
            "interval_funding_amount": execution.interval_funding,
            "interval_borrow_cost_amount": execution.interval_borrow_cost,
            "interval_dividend_amount": execution.interval_dividend,
            "interval_cash_interest_amount": execution.interval_cash_interest,
            "requested_turnover": execution.requested_turnover,
            "filled_turnover": execution.filled_turnover,
            "fill_ratio": execution.fill_ratio,
            "termination_reason": execution.termination_reason,
        }
        info.update(settlement_info)
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
    gamma: float = PPO_DEFAULT_GAMMA,
    normalize_features: bool = False,
    settle_terminal_position: bool = False,
    minimum_hold_bars: int = 0,
    observation_schema: str = PPO_OBSERVATION_SCHEMA,
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
    ppo_gamma = validated_ppo_gamma(gamma)
    if not isinstance(settle_terminal_position, bool):
        raise ValueError("settle_terminal_position must be boolean")
    if (
        isinstance(minimum_hold_bars, bool)
        or not isinstance(minimum_hold_bars, int)
        or minimum_hold_bars < 0
    ):
        raise ValueError("minimum_hold_bars must be a non-negative integer")
    if observation_schema not in PPO_OBSERVATION_SCHEMAS:
        raise ValueError("unsupported PPO observation schema")
    if minimum_hold_bars > 0 and observation_schema != PPO_OBSERVATION_SCHEMA_V3:
        raise ValueError("PPO minimum hold requires the age-aware observation")
    layout = _validated_training_layout(training_layout, rollout_steps_per_env)
    resolved_execution_cost = execution_cost or ExecutionCostConfig.zero()
    policy_stop_index = _agent_stop_index(
        start_index=start_index,
        stop_index=stop_index,
        execution_cost=resolved_execution_cost,
        settle_terminal_position=settle_terminal_position,
    )

    indices, fit_symbols = validated_training_scope(
        dataset,
        feature_indices=feature_indices,
        fit_symbol_indices=fit_symbol_indices,
    )
    if layout == PPO_TRAINING_LAYOUT_SEQUENTIAL:
        _validate_sequential_symbol_coverage(
            total_timesteps,
            episode_steps=policy_stop_index - start_index,
            n_symbols=len(fit_symbols),
        )
    if not isinstance(normalize_features, bool):
        raise ValueError("normalize_features must be boolean")
    normalizer = (
        fit_ppo_feature_normalizer(
            dataset,
            feature_indices=indices,
            fit_symbol_indices=fit_symbols,
            start_index=start_index,
            stop_index=policy_stop_index,
        )
        if normalize_features
        else None
    )
    ppo_n_steps = _PPO_DEFAULT_N_STEPS
    if layout == PPO_TRAINING_LAYOUT_SEQUENTIAL:
        env: object = PPOTradingEnv(
            dataset,
            feature_indices=indices,
            symbol_indices=fit_symbols,
            information_symbol_indices=fit_symbols,
            start_index=start_index,
            stop_index=stop_index,
            gross_budget=gross_budget,
            initial_capital=initial_capital,
            execution_cost=execution_cost,
            risk_config=risk_config,
            feature_normalizer=normalizer,
            settle_terminal_position=settle_terminal_position,
            minimum_hold_bars=minimum_hold_bars,
            observation_schema=observation_schema,
        )
    else:
        if execution_cost is not None and execution_cost.slippage_std > 0.0:
            raise ValueError(
                "interleaved training requires deterministic execution slippage"
            )
        rollout_steps = _validated_interleaved_rollout_steps(
            rollout_steps_per_env,
            n_envs=len(fit_symbols),
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
                    information_symbol_indices=fit_symbols,
                    start_index=start_index,
                    stop_index=stop_index,
                    gross_budget=gross_budget,
                    initial_capital=initial_capital,
                    execution_cost=execution_cost,
                    risk_config=risk_config,
                    feature_normalizer=normalizer,
                    settle_terminal_position=settle_terminal_position,
                    minimum_hold_bars=minimum_hold_bars,
                    observation_schema=observation_schema,
                )
                for symbol_index in fit_symbols
            ]
        )
        ppo_n_steps = rollout_steps

    try:
        module = importlib.import_module("stable_baselines3")
        ppo_class = getattr(module, "PPO")
        torch_module = importlib.import_module("torch")
        set_num_threads = getattr(torch_module, "set_num_threads")
        activation_fn = getattr(getattr(torch_module, "nn"), "Tanh")
        optimizer_class = getattr(getattr(torch_module, "optim"), "Adam")
        torch_layers = importlib.import_module("stable_baselines3.common.torch_layers")
        features_extractor_class = getattr(torch_layers, "FlattenExtractor")
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "stable-baselines3 and torch are required; install the train-sb3 extra"
        ) from error
    set_num_threads(1)

    model = ppo_class(
        "MlpPolicy",
        env,
        learning_rate=_PPO_LEARNING_RATE,
        n_steps=ppo_n_steps,
        batch_size=_PPO_BATCH_SIZE,
        n_epochs=_PPO_N_EPOCHS,
        gamma=ppo_gamma,
        gae_lambda=_PPO_GAE_LAMBDA,
        clip_range=_PPO_CLIP_RANGE,
        clip_range_vf=_PPO_CLIP_RANGE_VF,
        normalize_advantage=_PPO_NORMALIZE_ADVANTAGE,
        ent_coef=_PPO_ENT_COEF,
        vf_coef=_PPO_VF_COEF,
        max_grad_norm=_PPO_MAX_GRAD_NORM,
        use_sde=_PPO_USE_SDE,
        sde_sample_freq=_PPO_SDE_SAMPLE_FREQ,
        target_kl=_PPO_TARGET_KL,
        policy_kwargs={
            "net_arch": {"pi": [64, 64], "vf": [64, 64]},
            "activation_fn": activation_fn,
            "ortho_init": True,
            "features_extractor_class": features_extractor_class,
            "share_features_extractor": True,
            "optimizer_class": optimizer_class,
            "optimizer_kwargs": {"eps": 1e-5},
        },
        seed=seed,
        device="cpu",
        verbose=0,
    )
    model.learn(total_timesteps=total_timesteps)
    if isinstance(env, PPOTradingEnv):
        training_minimum_hold_suppressed_count = env.minimum_hold_suppressed_count
    else:
        sub_environments = getattr(env, "envs", None)
        if not isinstance(sub_environments, list) or any(
            not isinstance(sub_environment, PPOTradingEnv)
            for sub_environment in sub_environments
        ):
            raise RuntimeError(
                "PPO vector environment cannot report minimum-hold suppressions"
            )
        training_minimum_hold_suppressed_count = sum(
            sub_environment.minimum_hold_suppressed_count
            for sub_environment in sub_environments
        )
    realized_timesteps = getattr(model, "num_timesteps", None)
    expected_timesteps = expected_ppo_realized_timesteps(
        total_timesteps,
        training_layout=layout,
        rollout_steps_per_env=rollout_steps_per_env,
        n_envs=len(fit_symbols),
    )
    if realized_timesteps != expected_timesteps:
        raise RuntimeError("Stable-Baselines3 PPO realized an unexpected step count")
    return PPOIntentStrategy(
        cast(_PredictPolicy, model),
        feature_indices=indices,
        feature_names=tuple(dataset.feature_names[index] for index in indices),
        feature_normalizer=normalizer,
        observation_schema=observation_schema,
        minimum_hold_bars=minimum_hold_bars,
        training_minimum_hold_suppressed_count=(training_minimum_hold_suppressed_count),
    )


__all__ = [
    "PPO_GLOBAL_FEATURE_NAMES",
    "PPO_OBSERVATION_SCHEMA",
    "PPO_OBSERVATION_SCHEMA_V3",
    "PPO_OBSERVATION_SCHEMAS",
    "PPO_TRAINING_LAYOUT_INTERLEAVED",
    "PPO_TRAINING_LAYOUT_SEQUENTIAL",
    "PPOIntentStrategy",
    "PPOTradingEnv",
    "expected_ppo_realized_timesteps",
    "fit_ppo_strategy",
    "ppo_observation_contract_payload",
]
