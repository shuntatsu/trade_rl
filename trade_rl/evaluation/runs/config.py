"""Shared immutable configuration and dataset resolution for candidate runs."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from trade_rl._validation import require_sha256
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.runs.candidate_suite import LeanCandidateConfig

_ALLOWED_CONFIG_KEYS = frozenset(
    {
        "signal_name",
        "feature_names",
        "fit_symbol_names",
        "fit_cutoff",
        "evaluation_start",
        "evaluation_stop_exclusive",
        "rule_entry_threshold",
        "rule_exit_threshold",
        "forecast_entry_threshold",
        "forecast_exit_threshold",
        "ppo_total_timesteps",
        "ppo_seed",
        "gross_budget",
        "initial_capital",
    }
)


@dataclass(frozen=True, slots=True)
class CandidateRunConfig:
    """Raw semantic configuration for one candidate-suite run."""

    signal_name: str
    feature_names: tuple[str, ...]
    fit_symbol_names: tuple[str, ...]
    fit_cutoff: np.datetime64
    evaluation_start: np.datetime64
    evaluation_stop_exclusive: np.datetime64
    rule_entry_threshold: float
    rule_exit_threshold: float
    forecast_entry_threshold: float
    forecast_exit_threshold: float
    ppo_total_timesteps: int
    ppo_seed: int
    gross_budget: float
    initial_capital: float

    def __post_init__(self) -> None:
        signal_name = _validated_text(self.signal_name, field="signal_name")
        feature_names = _validated_names(self.feature_names, field="feature_names")
        fit_symbol_names = _validated_names(
            self.fit_symbol_names,
            field="fit_symbol_names",
        )
        fit_cutoff = _normalized_timestamp(self.fit_cutoff, field="fit_cutoff")
        evaluation_start = _normalized_timestamp(
            self.evaluation_start,
            field="evaluation_start",
        )
        evaluation_stop = _normalized_timestamp(
            self.evaluation_stop_exclusive,
            field="evaluation_stop_exclusive",
        )

        _require_positive_finite(
            self.rule_entry_threshold,
            field="rule_entry_threshold",
        )
        _require_non_negative_finite(
            self.rule_exit_threshold,
            field="rule_exit_threshold",
        )
        _require_positive_finite(
            self.forecast_entry_threshold,
            field="forecast_entry_threshold",
        )
        _require_non_negative_finite(
            self.forecast_exit_threshold,
            field="forecast_exit_threshold",
        )
        if self.rule_exit_threshold >= self.rule_entry_threshold:
            raise ValueError("rule exit threshold must be below entry threshold")
        if self.forecast_exit_threshold >= self.forecast_entry_threshold:
            raise ValueError("forecast exit threshold must be below entry threshold")
        if (
            isinstance(self.ppo_total_timesteps, bool)
            or not isinstance(self.ppo_total_timesteps, int)
            or self.ppo_total_timesteps <= 0
        ):
            raise ValueError("ppo_total_timesteps must be a positive integer")
        if (
            isinstance(self.ppo_seed, bool)
            or not isinstance(self.ppo_seed, int)
            or self.ppo_seed < 0
        ):
            raise ValueError("ppo_seed must be a non-negative integer")
        gross_budget = _require_finite(self.gross_budget, field="gross_budget")
        initial_capital = _require_finite(
            self.initial_capital,
            field="initial_capital",
        )

        object.__setattr__(self, "signal_name", signal_name)
        object.__setattr__(self, "feature_names", feature_names)
        object.__setattr__(self, "fit_symbol_names", fit_symbol_names)
        object.__setattr__(self, "fit_cutoff", fit_cutoff)
        object.__setattr__(self, "evaluation_start", evaluation_start)
        object.__setattr__(self, "evaluation_stop_exclusive", evaluation_stop)
        object.__setattr__(self, "gross_budget", gross_budget)
        object.__setattr__(self, "initial_capital", initial_capital)


@dataclass(frozen=True, slots=True)
class ResolvedCandidateRunSpec:
    """Dataset-bound executable candidate-run contract."""

    dataset_id: str
    dataset_artifact_schema: str
    dataset_artifact_digest: str
    config: CandidateRunConfig
    lean_config: LeanCandidateConfig
    evaluation_start_index: int
    evaluation_stop_index: int

    def __post_init__(self) -> None:
        require_sha256(self.dataset_id, field="dataset_id")
        if not self.dataset_artifact_schema:
            raise ValueError("dataset_artifact_schema must be non-empty")
        require_sha256(
            self.dataset_artifact_digest,
            field="dataset_artifact_digest",
        )


def _validated_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _validated_names(values: tuple[str, ...], *, field: str) -> tuple[str, ...]:
    if not values:
        raise ValueError(f"{field} must be a non-empty string array")
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{field} must be a non-empty string array")
    normalized = tuple(values)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field} must not contain duplicates")
    return normalized


def _normalized_timestamp(
    value: str | np.datetime64,
    *,
    field: str,
) -> np.datetime64:
    try:
        resolved = np.datetime64(value, "ns")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be an ISO datetime") from error
    if np.isnat(resolved):
        raise ValueError(f"{field} must not be NaT")
    return resolved


def _require_finite(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{field} must be a finite number")
    return resolved


def _require_positive_finite(value: object, *, field: str) -> float:
    resolved = _require_finite(value, field=field)
    if resolved <= 0.0:
        raise ValueError(f"{field} must be finite and positive")
    return resolved


def _require_non_negative_finite(value: object, *, field: str) -> float:
    resolved = _require_finite(value, field=field)
    if resolved < 0.0:
        raise ValueError(f"{field} must be finite and non-negative")
    return resolved


def _required_string(raw: Mapping[str, object], name: str) -> str:
    return _validated_text(raw.get(name), field=name)


def _required_float(raw: Mapping[str, object], name: str) -> float:
    return _require_finite(raw.get(name), field=name)


def _required_int(raw: Mapping[str, object], name: str) -> int:
    value = raw.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _required_string_tuple(
    raw: Mapping[str, object],
    name: str,
) -> tuple[str, ...]:
    value = raw.get(name)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty string array")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{name} must be a non-empty string array")
    result = tuple(cast(str, item) for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _required_timestamp(raw: Mapping[str, object], name: str) -> np.datetime64:
    return _normalized_timestamp(_required_string(raw, name), field=name)


def parse_candidate_run_config(raw: Mapping[str, object]) -> CandidateRunConfig:
    """Validate one JSON-like candidate-run configuration."""

    unknown = sorted(set(raw) - _ALLOWED_CONFIG_KEYS)
    if unknown:
        raise ValueError(f"unknown config keys: {', '.join(unknown)}")
    return CandidateRunConfig(
        signal_name=_required_string(raw, "signal_name"),
        feature_names=_required_string_tuple(raw, "feature_names"),
        fit_symbol_names=_required_string_tuple(raw, "fit_symbol_names"),
        fit_cutoff=_required_timestamp(raw, "fit_cutoff"),
        evaluation_start=_required_timestamp(raw, "evaluation_start"),
        evaluation_stop_exclusive=_required_timestamp(
            raw,
            "evaluation_stop_exclusive",
        ),
        rule_entry_threshold=_required_float(raw, "rule_entry_threshold"),
        rule_exit_threshold=_required_float(raw, "rule_exit_threshold"),
        forecast_entry_threshold=_required_float(raw, "forecast_entry_threshold"),
        forecast_exit_threshold=_required_float(raw, "forecast_exit_threshold"),
        ppo_total_timesteps=_required_int(raw, "ppo_total_timesteps"),
        ppo_seed=_required_int(raw, "ppo_seed"),
        gross_budget=_required_float(raw, "gross_budget"),
        initial_capital=_required_float(raw, "initial_capital"),
    )


def load_candidate_run_config(path: str | Path) -> CandidateRunConfig:
    """Load and validate a candidate-run JSON object from disk."""

    raw_object = cast(object, json.loads(Path(path).read_text(encoding="utf-8")))
    if not isinstance(raw_object, dict) or any(
        not isinstance(key, str) for key in raw_object
    ):
        raise ValueError("candidate run config must be a JSON object")
    return parse_candidate_run_config(cast(dict[str, object], raw_object))


def _feature_index(dataset: MarketDataset, name: str) -> int:
    try:
        return dataset.feature_names.index(name)
    except ValueError as error:
        raise ValueError(f"unknown feature name: {name}") from error


def _fit_symbol_index(dataset: MarketDataset, name: str) -> int:
    try:
        return dataset.symbols.index(name)
    except ValueError as error:
        raise ValueError(f"unknown fit symbol name: {name}") from error


def _exact_timestamp_index(
    dataset: MarketDataset,
    timestamp: np.datetime64,
    *,
    field: str,
) -> int:
    values = dataset.timestamps.astype("datetime64[ns]")
    matches = np.flatnonzero(values == np.datetime64(timestamp, "ns"))
    if matches.size != 1:
        raise ValueError(f"{field} must exactly match one dataset timestamp")
    return int(matches[0])


def resolve_candidate_run_spec(
    dataset: MarketDataset,
    *,
    dataset_artifact_schema: str,
    dataset_artifact_digest: str,
    config: CandidateRunConfig,
) -> ResolvedCandidateRunSpec:
    """Bind one validated candidate configuration to an exact dataset artifact."""

    feature_indices = tuple(
        _feature_index(dataset, name) for name in config.feature_names
    )
    fit_symbol_indices = tuple(
        _fit_symbol_index(dataset, name) for name in config.fit_symbol_names
    )
    start_index = _exact_timestamp_index(
        dataset,
        config.evaluation_start,
        field="evaluation_start",
    )
    stop_index = _exact_timestamp_index(
        dataset,
        config.evaluation_stop_exclusive,
        field="evaluation_stop_exclusive",
    )
    if dataset.timestamps[start_index] < config.fit_cutoff:
        raise ValueError("evaluation must not start before fit_cutoff")

    lean_config = LeanCandidateConfig(
        signal_index=_feature_index(dataset, config.signal_name),
        feature_indices=feature_indices,
        fit_symbol_indices=fit_symbol_indices,
        fit_cutoff=config.fit_cutoff,
        rule_entry_threshold=config.rule_entry_threshold,
        rule_exit_threshold=config.rule_exit_threshold,
        forecast_entry_threshold=config.forecast_entry_threshold,
        forecast_exit_threshold=config.forecast_exit_threshold,
        ppo_total_timesteps=config.ppo_total_timesteps,
        ppo_seed=config.ppo_seed,
    )
    return ResolvedCandidateRunSpec(
        dataset_id=dataset.dataset_id,
        dataset_artifact_schema=dataset_artifact_schema,
        dataset_artifact_digest=dataset_artifact_digest,
        config=config,
        lean_config=lean_config,
        evaluation_start_index=start_index,
        evaluation_stop_index=stop_index,
    )


__all__ = [
    "CandidateRunConfig",
    "ResolvedCandidateRunSpec",
    "load_candidate_run_config",
    "parse_candidate_run_config",
    "resolve_candidate_run_spec",
]
