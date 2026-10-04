"""Resolved candidate-run contract used by controlled experiment evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.experiments.contracts._common import (
    contract_finite,
    contract_non_negative_int,
    contract_positive_int,
    contract_text,
    contract_unique_texts,
)
from trade_rl.evaluation.experiments.errors import ContractViolationError
from trade_rl.risk import PreTradeRiskConfig
from trade_rl.strategies.rl.intent import (
    PPO_GLOBAL_FEATURE_NAMES,
    PPO_OBSERVATION_SCHEMA_V3,
    PPO_OBSERVATION_SCHEMAS,
)
from trade_rl.strategies.rl.ppo_training import (
    PPO_TRAINING_LAYOUT_INTERLEAVED,
    PPO_TRAINING_LAYOUT_SEQUENTIAL,
)

_RESOLVED_RUN_CONFIG_V1 = "resolved_run_config_v1"
_RESOLVED_RUN_CONFIG_V2 = "resolved_run_config_v2"
_RESOLVED_RUN_CONFIG_V3 = "resolved_run_config_v3"
_RESOLVED_RUN_CONFIG_V4 = "resolved_run_config_v4"
_RESOLVED_RUN_CONFIG_V5 = "resolved_run_config_v5"
_RESOLVED_RUN_CONFIG_V7 = "resolved_run_config_v7"

if TYPE_CHECKING:
    from trade_rl.evaluation.runs import ResolvedCandidateRunSpec


def _index_tuple(values: object, *, field: str) -> tuple[int, ...]:
    if not isinstance(values, tuple) or not values:
        raise ContractViolationError(f"{field} must be a non-empty tuple")
    result = tuple(contract_non_negative_int(value, field=field) for value in values)
    if len(set(result)) != len(result):
        raise ContractViolationError(f"{field} must contain unique values")
    return result


@dataclass(frozen=True, slots=True)
class ResolvedRunConfig:
    """Canonical, dataset-resolved semantic configuration for one Run."""

    signal_name: str
    signal_index: int
    feature_names: tuple[str, ...]
    feature_indices: tuple[int, ...]
    fit_symbol_names: tuple[str, ...]
    fit_symbol_indices: tuple[int, ...]
    fit_cutoff: str
    rule_entry_threshold: float
    rule_exit_threshold: float
    forecast_entry_threshold: float
    forecast_exit_threshold: float
    ppo_total_timesteps: int
    ppo_seed: int
    evaluation_start: str
    evaluation_stop_exclusive: str
    gross_budget: float
    initial_capital: float
    execution_overlay: str
    ppo_observation_schema: str | None = None
    ppo_global_feature_names: tuple[str, ...] = ()
    schema_version: str = _RESOLVED_RUN_CONFIG_V1
    ppo_training_layout: str = PPO_TRAINING_LAYOUT_SEQUENTIAL
    ppo_rollout_steps_per_env: int | None = None
    ppo_minimum_hold_bars: int = 0
    ppo_settle_terminal_position: bool = False
    pretrade_risk_config: PreTradeRiskConfig | None = None
    forecast_switch_cost: float | None = None

    def __post_init__(self) -> None:
        signal_name = contract_text(self.signal_name, field="signal_name")
        signal_index = contract_non_negative_int(
            self.signal_index, field="signal_index"
        )
        feature_names = contract_unique_texts(self.feature_names, field="feature_names")
        feature_indices = _index_tuple(self.feature_indices, field="feature_indices")
        fit_symbol_names = contract_unique_texts(
            self.fit_symbol_names,
            field="fit_symbol_names",
        )
        fit_symbol_indices = _index_tuple(
            self.fit_symbol_indices,
            field="fit_symbol_indices",
        )
        if len(feature_names) != len(feature_indices):
            raise ContractViolationError(
                "feature_names and feature_indices must have the same length"
            )
        if len(fit_symbol_names) != len(fit_symbol_indices):
            raise ContractViolationError(
                "fit_symbol_names and fit_symbol_indices must have the same length"
            )
        fit_cutoff = contract_text(self.fit_cutoff, field="fit_cutoff")
        evaluation_start = contract_text(
            self.evaluation_start,
            field="evaluation_start",
        )
        evaluation_stop = contract_text(
            self.evaluation_stop_exclusive,
            field="evaluation_stop_exclusive",
        )

        rule_entry = contract_finite(
            self.rule_entry_threshold,
            field="rule_entry_threshold",
        )
        rule_exit = contract_finite(
            self.rule_exit_threshold,
            field="rule_exit_threshold",
        )
        forecast_entry = contract_finite(
            self.forecast_entry_threshold,
            field="forecast_entry_threshold",
        )
        forecast_exit = contract_finite(
            self.forecast_exit_threshold,
            field="forecast_exit_threshold",
        )
        if rule_entry <= 0.0:
            raise ContractViolationError(
                "rule_entry_threshold must be finite and positive"
            )
        if rule_exit < 0.0:
            raise ContractViolationError(
                "rule_exit_threshold must be finite and non-negative"
            )
        if rule_exit >= rule_entry:
            raise ContractViolationError(
                "rule exit threshold must be below entry threshold"
            )
        if forecast_entry <= 0.0:
            raise ContractViolationError(
                "forecast_entry_threshold must be finite and positive"
            )
        if forecast_exit < 0.0:
            raise ContractViolationError(
                "forecast_exit_threshold must be finite and non-negative"
            )
        if forecast_exit >= forecast_entry:
            raise ContractViolationError(
                "forecast exit threshold must be below entry threshold"
            )

        ppo_total_timesteps = contract_positive_int(
            self.ppo_total_timesteps,
            field="ppo_total_timesteps",
        )
        ppo_seed = contract_non_negative_int(self.ppo_seed, field="ppo_seed")
        gross_budget = contract_finite(self.gross_budget, field="gross_budget")
        initial_capital = contract_finite(
            self.initial_capital,
            field="initial_capital",
        )
        execution_overlay = contract_text(
            self.execution_overlay,
            field="execution_overlay",
        )
        schema_version = contract_text(self.schema_version, field="schema_version")
        forecast_switch_cost = self.forecast_switch_cost
        if schema_version == _RESOLVED_RUN_CONFIG_V7:
            if forecast_switch_cost is not None:
                if isinstance(forecast_switch_cost, bool):
                    raise ContractViolationError(
                        "forecast_switch_cost must be finite and non-negative"
                    )
                forecast_switch_cost = contract_finite(
                    forecast_switch_cost,
                    field="forecast_switch_cost",
                )
                if forecast_switch_cost < 0.0:
                    raise ContractViolationError(
                        "forecast_switch_cost must be finite and non-negative"
                    )
        elif forecast_switch_cost is not None:
            raise ContractViolationError(
                "legacy resolved-run config cannot define forecast_switch_cost"
            )
        ppo_training_layout = self.ppo_training_layout
        ppo_rollout_steps_per_env = self.ppo_rollout_steps_per_env
        ppo_minimum_hold_bars = 0
        ppo_settle_terminal_position = False
        pretrade_risk_config: PreTradeRiskConfig | None = None
        if schema_version == _RESOLVED_RUN_CONFIG_V1:
            if self.ppo_observation_schema is not None or self.ppo_global_feature_names:
                raise ContractViolationError(
                    "resolved_run_config_v1 must not define a PPO observation contract"
                )
            if (
                ppo_training_layout != PPO_TRAINING_LAYOUT_SEQUENTIAL
                or ppo_rollout_steps_per_env is not None
            ):
                raise ContractViolationError(
                    "resolved_run_config_v1 must not define a PPO training layout"
                )
            ppo_observation_schema: str | None = None
            ppo_global_feature_names: tuple[str, ...] = ()
        elif schema_version in {
            _RESOLVED_RUN_CONFIG_V2,
            _RESOLVED_RUN_CONFIG_V3,
            _RESOLVED_RUN_CONFIG_V4,
            _RESOLVED_RUN_CONFIG_V5,
            _RESOLVED_RUN_CONFIG_V7,
        }:
            ppo_observation_schema = contract_text(
                self.ppo_observation_schema,
                field="ppo_observation_schema",
            )
            if not isinstance(self.ppo_global_feature_names, tuple):
                raise ContractViolationError("ppo_global_feature_names must be a tuple")
            ppo_global_feature_names = tuple(
                contract_text(value, field="ppo_global_feature_names")
                for value in self.ppo_global_feature_names
            )
            if len(set(ppo_global_feature_names)) != len(ppo_global_feature_names):
                raise ContractViolationError(
                    "ppo_global_feature_names must contain unique values"
                )
            if ppo_observation_schema not in PPO_OBSERVATION_SCHEMAS:
                raise ContractViolationError("unsupported PPO observation schema")
            if ppo_global_feature_names != PPO_GLOBAL_FEATURE_NAMES:
                raise ContractViolationError(
                    "PPO global feature names do not match the frozen observation contract"
                )
            if schema_version == _RESOLVED_RUN_CONFIG_V2:
                if (
                    ppo_training_layout != PPO_TRAINING_LAYOUT_SEQUENTIAL
                    or ppo_rollout_steps_per_env is not None
                ):
                    raise ContractViolationError(
                        "resolved_run_config_v2 must not define a PPO training layout"
                    )
            else:
                ppo_training_layout = contract_text(
                    ppo_training_layout,
                    field="ppo_training_layout",
                )
                if ppo_training_layout not in {
                    PPO_TRAINING_LAYOUT_SEQUENTIAL,
                    PPO_TRAINING_LAYOUT_INTERLEAVED,
                }:
                    raise ContractViolationError("unsupported PPO training layout")
            if schema_version in {
                _RESOLVED_RUN_CONFIG_V3,
                _RESOLVED_RUN_CONFIG_V4,
                _RESOLVED_RUN_CONFIG_V5,
                _RESOLVED_RUN_CONFIG_V7,
            }:
                if ppo_training_layout == PPO_TRAINING_LAYOUT_SEQUENTIAL:
                    if ppo_rollout_steps_per_env is not None:
                        raise ContractViolationError(
                            "sequential PPO layout cannot define rollout steps"
                        )
                elif (
                    isinstance(ppo_rollout_steps_per_env, bool)
                    or not isinstance(ppo_rollout_steps_per_env, int)
                    or ppo_rollout_steps_per_env <= 0
                ):
                    raise ContractViolationError(
                        "interleaved PPO layout requires positive rollout steps"
                    )
            if schema_version in {
                _RESOLVED_RUN_CONFIG_V4,
                _RESOLVED_RUN_CONFIG_V5,
                _RESOLVED_RUN_CONFIG_V7,
            }:
                ppo_minimum_hold_bars = contract_non_negative_int(
                    self.ppo_minimum_hold_bars,
                    field="ppo_minimum_hold_bars",
                )
                if not isinstance(self.ppo_settle_terminal_position, bool):
                    raise ContractViolationError(
                        "ppo_settle_terminal_position must be boolean"
                    )
                ppo_settle_terminal_position = self.ppo_settle_terminal_position
                if (
                    ppo_minimum_hold_bars > 0
                    and ppo_observation_schema != PPO_OBSERVATION_SCHEMA_V3
                ):
                    raise ContractViolationError(
                        "PPO minimum hold requires the age-aware observation"
                    )
            elif self.ppo_minimum_hold_bars != 0 or self.ppo_settle_terminal_position:
                raise ContractViolationError(
                    "legacy resolved-run config cannot define duration semantics"
                )
            if schema_version in {_RESOLVED_RUN_CONFIG_V5, _RESOLVED_RUN_CONFIG_V7}:
                if self.pretrade_risk_config is not None and not isinstance(
                    self.pretrade_risk_config, PreTradeRiskConfig
                ):
                    raise ContractViolationError(
                        "pretrade_risk_config must be a PreTradeRiskConfig or null"
                    )
                pretrade_risk_config = self.pretrade_risk_config
                if ppo_observation_schema == PPO_OBSERVATION_SCHEMA_V3:
                    if not ppo_settle_terminal_position:
                        raise ContractViolationError(
                            "age-aware PPO comparison requires terminal settlement"
                        )
                    if pretrade_risk_config is None:
                        raise ContractViolationError(
                            "age-aware PPO comparison requires explicit pre-trade risk config"
                        )
                    if pretrade_risk_config.drawdown_stop > 0.20:
                        raise ContractViolationError(
                            "PPO drawdown stop must not exceed 20%"
                        )
            elif self.pretrade_risk_config is not None:
                raise ContractViolationError(
                    "legacy resolved-run config cannot define pre-trade risk"
                )
        else:
            raise ContractViolationError("unsupported resolved-run config schema")

        object.__setattr__(self, "signal_name", signal_name)
        object.__setattr__(self, "signal_index", signal_index)
        object.__setattr__(self, "feature_names", feature_names)
        object.__setattr__(self, "feature_indices", feature_indices)
        object.__setattr__(self, "fit_symbol_names", fit_symbol_names)
        object.__setattr__(self, "fit_symbol_indices", fit_symbol_indices)
        object.__setattr__(self, "fit_cutoff", fit_cutoff)
        object.__setattr__(self, "rule_entry_threshold", rule_entry)
        object.__setattr__(self, "rule_exit_threshold", rule_exit)
        object.__setattr__(self, "forecast_entry_threshold", forecast_entry)
        object.__setattr__(self, "forecast_exit_threshold", forecast_exit)
        object.__setattr__(self, "ppo_total_timesteps", ppo_total_timesteps)
        object.__setattr__(self, "ppo_seed", ppo_seed)
        object.__setattr__(self, "evaluation_start", evaluation_start)
        object.__setattr__(self, "evaluation_stop_exclusive", evaluation_stop)
        object.__setattr__(self, "gross_budget", gross_budget)
        object.__setattr__(self, "initial_capital", initial_capital)
        object.__setattr__(self, "execution_overlay", execution_overlay)
        object.__setattr__(self, "ppo_observation_schema", ppo_observation_schema)
        object.__setattr__(self, "ppo_global_feature_names", ppo_global_feature_names)
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "ppo_training_layout", ppo_training_layout)
        object.__setattr__(
            self,
            "ppo_rollout_steps_per_env",
            ppo_rollout_steps_per_env,
        )
        object.__setattr__(self, "ppo_minimum_hold_bars", ppo_minimum_hold_bars)
        object.__setattr__(
            self,
            "ppo_settle_terminal_position",
            ppo_settle_terminal_position,
        )
        object.__setattr__(self, "pretrade_risk_config", pretrade_risk_config)
        object.__setattr__(self, "forecast_switch_cost", forecast_switch_cost)

    @classmethod
    def from_candidate_spec(
        cls,
        spec: ResolvedCandidateRunSpec,
    ) -> ResolvedRunConfig:
        """Project one Run Core spec into the controlled-experiment contract."""

        config = spec.config
        lean = spec.lean_config
        return cls(
            signal_name=config.signal_name,
            signal_index=lean.signal_index,
            feature_names=config.feature_names,
            feature_indices=lean.feature_indices,
            fit_symbol_names=config.fit_symbol_names,
            fit_symbol_indices=lean.fit_symbol_indices,
            fit_cutoff=str(lean.fit_cutoff),
            rule_entry_threshold=lean.rule_entry_threshold,
            rule_exit_threshold=lean.rule_exit_threshold,
            forecast_entry_threshold=lean.forecast_entry_threshold,
            forecast_exit_threshold=lean.forecast_exit_threshold,
            ppo_total_timesteps=lean.ppo_total_timesteps,
            ppo_seed=lean.ppo_seed,
            evaluation_start=str(config.evaluation_start),
            evaluation_stop_exclusive=str(config.evaluation_stop_exclusive),
            gross_budget=config.gross_budget,
            initial_capital=config.initial_capital,
            execution_overlay=spec.execution_overlay,
            ppo_observation_schema=config.ppo_observation_schema,
            ppo_global_feature_names=PPO_GLOBAL_FEATURE_NAMES,
            schema_version=_RESOLVED_RUN_CONFIG_V7,
            ppo_training_layout=lean.ppo_training_layout,
            ppo_rollout_steps_per_env=lean.ppo_rollout_steps_per_env,
            ppo_minimum_hold_bars=lean.ppo_minimum_hold_bars,
            ppo_settle_terminal_position=lean.ppo_settle_terminal_position,
            pretrade_risk_config=config.pretrade_risk_config,
            forecast_switch_cost=config.forecast_switch_cost,
        )

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "signal_name": self.signal_name,
            "signal_index": self.signal_index,
            "feature_names": list(self.feature_names),
            "feature_indices": list(self.feature_indices),
            "fit_symbol_names": list(self.fit_symbol_names),
            "fit_symbol_indices": list(self.fit_symbol_indices),
            "fit_cutoff": self.fit_cutoff,
            "rule_entry_threshold": self.rule_entry_threshold,
            "rule_exit_threshold": self.rule_exit_threshold,
            "forecast_entry_threshold": self.forecast_entry_threshold,
            "forecast_exit_threshold": self.forecast_exit_threshold,
            "ppo_total_timesteps": self.ppo_total_timesteps,
            "ppo_seed": self.ppo_seed,
            "evaluation_start": self.evaluation_start,
            "evaluation_stop_exclusive": self.evaluation_stop_exclusive,
            "gross_budget": self.gross_budget,
            "initial_capital": self.initial_capital,
            "execution_overlay": self.execution_overlay,
        }
        if self.schema_version in {
            _RESOLVED_RUN_CONFIG_V2,
            _RESOLVED_RUN_CONFIG_V3,
            _RESOLVED_RUN_CONFIG_V4,
            _RESOLVED_RUN_CONFIG_V5,
            _RESOLVED_RUN_CONFIG_V7,
        }:
            payload["ppo_observation_schema"] = self.ppo_observation_schema
            payload["ppo_global_feature_names"] = list(self.ppo_global_feature_names)
        if self.schema_version in {
            _RESOLVED_RUN_CONFIG_V3,
            _RESOLVED_RUN_CONFIG_V4,
            _RESOLVED_RUN_CONFIG_V5,
            _RESOLVED_RUN_CONFIG_V7,
        }:
            payload["ppo_training_layout"] = self.ppo_training_layout
            payload["ppo_rollout_steps_per_env"] = self.ppo_rollout_steps_per_env
        if self.schema_version in {
            _RESOLVED_RUN_CONFIG_V4,
            _RESOLVED_RUN_CONFIG_V5,
            _RESOLVED_RUN_CONFIG_V7,
        }:
            payload["ppo_minimum_hold_bars"] = self.ppo_minimum_hold_bars
            payload["ppo_settle_terminal_position"] = self.ppo_settle_terminal_position
        if self.schema_version in {_RESOLVED_RUN_CONFIG_V5, _RESOLVED_RUN_CONFIG_V7}:
            risk = self.pretrade_risk_config
            payload["pretrade_risk_config"] = (
                None
                if risk is None
                else {
                    "max_gross": risk.max_gross,
                    "max_abs_weight": risk.max_abs_weight,
                    "max_turnover": risk.max_turnover,
                    "drawdown_start": risk.drawdown_start,
                    "drawdown_stop": risk.drawdown_stop,
                    "emergency_turnover_override": risk.emergency_turnover_override,
                    "fail_closed_tolerance": risk.fail_closed_tolerance,
                }
            )
        if self.schema_version == _RESOLVED_RUN_CONFIG_V7:
            payload["forecast_switch_cost"] = self.forecast_switch_cost
        return payload

    @property
    def digest(self) -> str:
        return content_digest(self.to_payload())


__all__ = ["ResolvedRunConfig"]
