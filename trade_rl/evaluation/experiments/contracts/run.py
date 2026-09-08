"""Resolved candidate-run contract used by controlled experiment evidence."""

from __future__ import annotations

from dataclasses import dataclass

from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.experiments.contracts._common import (
    contract_finite,
    contract_non_negative_int,
    contract_positive_int,
    contract_text,
    contract_unique_texts,
)
from trade_rl.evaluation.experiments.errors import ContractViolationError


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
    schema_version: str = "resolved_run_config_v1"

    def __post_init__(self) -> None:
        signal_name = contract_text(self.signal_name, field="signal_name")
        signal_index = contract_non_negative_int(self.signal_index, field="signal_index")
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
        object.__setattr__(self, "schema_version", schema_version)

    def to_payload(self) -> dict[str, object]:
        return {
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

    @property
    def digest(self) -> str:
        return content_digest(self.to_payload())


__all__ = ["ResolvedRunConfig"]
