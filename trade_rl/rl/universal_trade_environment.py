"""U1 single-instrument environment surface without changing base economics."""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketDataset
from trade_rl.rl.actions import (
    ActionMode,
    ActionSpec,
    ActionValidationMode,
    BaselineResidualComposer,
    ResidualComposition,
    TargetWeightAction,
)
from trade_rl.rl.environment import ResidualMarketEnv
from trade_rl.rl.environment_config import EpisodeBoundaryMode
from trade_rl.rl.universal_normalization import UniversalTradeSequenceNormalizer
from trade_rl.rl.universal_trade_action import parse_normalized_target_exposure
from trade_rl.rl.universal_trade_contract import (
    UNIVERSAL_TRADE_ACTION_SCHEMA,
    UNIVERSAL_TRADE_OBSERVATION_SCHEMA,
    UNIVERSAL_TRADE_SEQUENCE_WINDOWS,
    UniversalTradePolicyContract,
)
from trade_rl.rl.universal_trade_observation import UniversalTradeObservationBuilder
from trade_rl.rl.universal_trade_reward import universal_net_log_growth_reward
from trade_rl.rl.universal_trade_runtime import UniversalTradeRuntimeSnapshot
from trade_rl.strategies.trend import TrendTargets

_U1_REWARD_RECONCILIATION_ATOL = 1e-10


class _UniversalTradeTargetComposer(BaselineResidualComposer):
    """Keep U1 target weights raw until the maintained Risk projection stage."""

    @staticmethod
    def _compose_target(
        action: TargetWeightAction,
        trends: TrendTargets,
        *,
        max_gross: float,
    ) -> ResidualComposition:
        if action.weights.shape != trends.base.shape:
            raise ValueError("target weight count does not match trend targets")
        if not np.isfinite(max_gross) or max_gross <= 0.0:
            raise ValueError("max_gross must be finite and positive")
        proposal = np.asarray(action.weights, dtype=np.float64).reshape(-1).copy()
        zeros = np.zeros_like(trends.base)
        raw_gross = float(np.abs(proposal).sum())
        return ResidualComposition(
            action=action,
            baseline=trends.base.copy(),
            trend_component=zeros.copy(),
            alpha_component=zeros.copy(),
            factor_component=zeros.copy(),
            residual_component=proposal - trends.base,
            proposal=proposal,
            raw_gross=raw_gross,
            target_gross=raw_gross,
        )


class UniversalTradeMarketEnv(ResidualMarketEnv):
    """Expose U1 semantics while retaining maintained Risk/Execution economics."""

    def __init__(self, dataset: MarketDataset, **kwargs: Any) -> None:
        if dataset.n_symbols != 1:
            raise ValueError(
                "Universal Trade RL U1 environment requires a single-symbol dataset"
            )
        if "composer" in kwargs:
            raise ValueError("Universal Trade RL fixes the U1 target composer")
        super().__init__(dataset, composer=_UniversalTradeTargetComposer(), **kwargs)

    def universal_trade_runtime_snapshot(self) -> UniversalTradeRuntimeSnapshot:
        """Return scalar U1 state from existing Risk/Execution/accounting state."""

        if not self._has_reset:
            raise RuntimeError("environment must be reset before exporting U1 runtime")

        index = self.current_index
        bar_hours = self.dataset.bar_hours
        pending_target = self._pending_hybrid_target
        pending_order = self._pending_order_observation_state()
        drawdown = self._drawdown(self.hybrid)
        mark_price = float(self.dataset.resolved_array("mark_price")[index, 0])
        index_price = float(self.dataset.resolved_array("index_price")[index, 0])

        return UniversalTradeRuntimeSnapshot(
            policy_requested_weight=float(self._previous_action[0]),
            pending_target_weight=(
                0.0 if pending_target is None else float(pending_target[0])
            ),
            pending_target_active=pending_target is not None,
            risk_projected_weight=float(self._execution_state.requested_weights[0]),
            current_weight=float(self.hybrid.weights[0]),
            previous_action=float(self._previous_action[0]),
            fill_ratio=float(self._execution_state.fill_ratio[0]),
            unfilled_turnover_ratio=float(self._execution_state.unfilled_turnover[0]),
            participation_ratio=float(self._execution_state.participation[0]),
            execution_cost_rate=float(self._execution_state.execution_cost[0]),
            position_age_hours=float(self._execution_state.position_age[0] * bar_hours),
            pending_notional_ratio=float(pending_order.remaining_notional_ratio[0]),
            pending_order_type_code=float(pending_order.order_type_code[0]),
            pending_order_status_code=float(pending_order.status_code[0]),
            pending_order_age_hours=float(pending_order.age_bars[0] * bar_hours),
            pending_order_eligible_delay_hours=float(
                pending_order.eligible_delay_bars[0] * bar_hours
            ),
            pending_order_triggered=bool(pending_order.triggered[0]),
            pending_order_expiry_distance_hours=float(
                pending_order.expiry_distance_bars[0] * bar_hours
            ),
            asset_active=bool(self.dataset.resolved_array("asset_active")[index, 0]),
            tradable=bool(self.dataset.observable_tradable(index)[0]),
            borrow_available=bool(
                self.dataset.resolved_array("borrow_available")[index, 0]
            ),
            borrow_rate=float(self.dataset.resolved_array("borrow_rate")[index, 0]),
            mark_index_basis=mark_price / index_price - 1.0,
            current_drawdown=drawdown,
            current_gross_exposure=float(self.hybrid.gross_exposure),
            current_net_exposure=float(self.hybrid.net_exposure),
            cash_weight=float(self.hybrid.cash_weight),
            risk_scale=float(self.pre_trade_risk.risk_scale(drawdown)),
            margin_utilization=float(self.hybrid.margin_utilization),
        )


class UniversalTradeEnvironment(gym.Env[dict[str, np.ndarray], np.ndarray]):
    """Thin U1 policy facade over the maintained single-instrument market env."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        base_env: UniversalTradeMarketEnv,
        *,
        contract: UniversalTradePolicyContract,
        normalizer: UniversalTradeSequenceNormalizer | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(base_env, UniversalTradeMarketEnv):
            raise TypeError("U1 wrapper requires a UniversalTradeMarketEnv")
        if not isinstance(contract, UniversalTradePolicyContract):
            raise TypeError("U1 wrapper requires a Universal Trade policy contract")

        self._base_env = base_env
        self.contract = contract
        self.sequence_normalizer = normalizer
        self._validate_base_contract()
        self._observation_builder = UniversalTradeObservationBuilder(
            contract=contract,
            normalizer=normalizer,
        )
        self.observation_space = self._observation_builder.observation_space
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )
        self.metadata = dict(getattr(base_env, "metadata", self.metadata))
        normalizer_digest = None if normalizer is None else normalizer.digest
        self._observation_contract_digest = content_digest(
            {
                "contract_digest": contract.digest,
                "normalizer_digest": normalizer_digest,
                "observation_builder_schema_digest": (
                    self._observation_builder.schema_digest
                ),
                "schema_version": "universal_trade_environment_observation_v1",
                "state_layout_digest": self._observation_builder.state_layout_digest,
            }
        )
        self._environment_digest = content_digest(
            {
                "contract_digest": contract.digest,
                "observation_contract_digest": self._observation_contract_digest,
                "schema_version": "universal_trade_environment_v1",
                "source_environment_digest": base_env.environment_digest,
            }
        )

    @staticmethod
    def _contract_error(field: str) -> ValueError:
        return ValueError(f"U1 environment contract drift: {field}")

    def _validate_base_contract(self) -> None:
        base = self._base_env
        config = base.config
        action_spec = base.action_spec

        if type(base.composer) is not _UniversalTradeTargetComposer:
            raise self._contract_error("target composer")
        if (
            action_spec.mode is not ActionMode.TARGET_WEIGHT
            or action_spec.target_weight_count != 1
            or action_spec.size != 1
            or action_spec.alpha_enabled
            or action_spec.risk_tilt_enabled
            or action_spec.n_factors != 0
        ):
            raise self._contract_error("action schema")
        if ActionValidationMode(action_spec.validation_mode) is not ActionValidationMode.STRICT:
            raise self._contract_error("action validation mode")
        if config.accept_legacy_actions:
            raise self._contract_error("legacy actions")
        if config.action_validation_mode is not ActionValidationMode.STRICT:
            raise self._contract_error("config action validation mode")
        if config.signal_delay_decisions != 1:
            raise self._contract_error("signal delay")
        if config.decision_hours != 0.25 or config.decision_every is not None:
            raise self._contract_error("decision cadence")
        if (
            config.episode_hours != 720.0
            or config.episode_hour_choices != ()
            or config.episode_bars is not None
        ):
            raise self._contract_error("episode horizon")
        if config.initial_state_modes != ("cash",):
            raise self._contract_error("initial state modes")
        if config.episode_boundary_mode is not EpisodeBoundaryMode.EXTERNAL_TRUNCATION:
            raise self._contract_error("episode boundary mode")
        if config.finite_horizon_observation:
            raise self._contract_error("finite horizon observation")
        if not config.structured_sequence_observation:
            raise self._contract_error("structured sequence observation")
        if config.resolved_sequence_windows != UNIVERSAL_TRADE_SEQUENCE_WINDOWS:
            raise self._contract_error("sequence windows")
        if config.liquidate_on_end:
            raise self._contract_error("terminal liquidation")

        expected_feature_names = tuple(spec.name for spec in self.contract.feature_specs)
        if base.dataset.feature_names != expected_feature_names:
            raise self._contract_error("feature order")

        configured_reward = config.resolved_reward_config()
        runtime_reward = base.reward_tracker.config
        for reward_config in (configured_reward, runtime_reward):
            if (
                not reward_config.is_pure_net_log_growth()
                or reward_config.scale != self.contract.reward_scale
            ):
                raise self._contract_error("reward")

    @property
    def dataset(self) -> MarketDataset:
        return self._base_env.dataset

    @property
    def base_env(self) -> UniversalTradeMarketEnv:
        return self._base_env

    @property
    def action_spec(self) -> ActionSpec:
        return self._base_env.action_spec

    @property
    def action_names(self) -> tuple[str, ...]:
        return self._base_env.action_names

    @property
    def action_spec_digest(self) -> str:
        return content_digest(
            {
                "contract_digest": self.contract.digest,
                "policy_weight_scale": self.contract.policy_weight_scale,
                "schema_version": UNIVERSAL_TRADE_ACTION_SCHEMA,
                "size": 1,
            }
        )

    @property
    def observation_schema(self) -> str:
        return UNIVERSAL_TRADE_OBSERVATION_SCHEMA

    @property
    def observation_contract_digest(self) -> str:
        return self._observation_contract_digest

    @property
    def environment_digest(self) -> str:
        return self._environment_digest

    @property
    def initial_capital(self) -> float:
        return self._base_env.initial_capital

    @property
    def decision_hours(self) -> float:
        return self._base_env.decision_hours

    @property
    def minimum_start_index(self) -> int:
        return self._base_env.minimum_start_index

    @property
    def current_index(self) -> int:
        return self._base_env.current_index

    @property
    def hybrid(self) -> Any:
        return self._base_env.hybrid

    @property
    def shadow(self) -> Any:
        return self._base_env.shadow

    @property
    def pre_trade_risk(self) -> Any:
        return self._base_env.pre_trade_risk

    @property
    def normalizer(self) -> None:
        return None

    @property
    def alpha_artifact_digest(self) -> str | None:
        return self._base_env.alpha_artifact_digest

    @property
    def factor_artifact_digest(self) -> str | None:
        return self._base_env.factor_artifact_digest

    def _policy_observation(self) -> dict[str, np.ndarray]:
        return self._observation_builder.build(
            dataset=self._base_env.dataset,
            index=self._base_env.current_index,
            runtime=self._base_env.universal_trade_runtime_snapshot(),
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        resolved_options = dict(options or {})
        initial_state_mode = resolved_options.get("initial_state_mode", "cash")
        if initial_state_mode != "cash":
            raise ValueError("U1 environment reset is cash-only")
        if "initial_book" in resolved_options:
            raise ValueError("U1 environment reset is cash-only")
        resolved_options["initial_state_mode"] = "cash"
        super().reset(seed=seed)
        _base_observation, info = self._base_env.reset(
            seed=seed,
            options=resolved_options,
        )
        return self._policy_observation(), info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        parsed = parse_normalized_target_exposure(
            action,
            policy_weight_scale=self.contract.policy_weight_scale,
        )
        before_value = float(self._base_env.hybrid.portfolio_value)
        base_action = np.asarray([parsed.policy_requested_weight], dtype=np.float32)
        _base_observation, delegated_reward, terminated, truncated, info = (
            self._base_env.step(base_action)
        )
        after_value = float(self._base_env.hybrid.portfolio_value)
        reward = universal_net_log_growth_reward(
            before_value=before_value,
            after_value=after_value,
        )
        if not math.isclose(
            float(delegated_reward),
            reward,
            rel_tol=0.0,
            abs_tol=_U1_REWARD_RECONCILIATION_ATOL,
        ):
            raise RuntimeError(
                "U1 reward drift: delegated base reward does not match realized wealth"
            )
        return self._policy_observation(), reward, terminated, truncated, info

    def close(self) -> None:
        self._base_env.close()


__all__ = ["UniversalTradeEnvironment", "UniversalTradeMarketEnv"]
