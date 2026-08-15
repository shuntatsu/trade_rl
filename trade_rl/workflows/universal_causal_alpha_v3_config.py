"""Strict authored configuration for the research-only causal alpha V3 runner."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.config_fields import require_exact_fields
from trade_rl.learning.causal_alpha_v3 import (
    CausalAlphaV3FitConfig,
    CausalAlphaV3TargetConfig,
)

CAUSAL_ALPHA_V3_RESEARCH_CONFIG_SCHEMA: Final = (
    "universal_causal_alpha_v3_research_config_v1"
)
_MAX_CANDIDATES: Final = 8


def _mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a JSON object")
    return dict(value)


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be numeric")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{field} must be finite")
    return resolved


def _non_negative_int(value: object, *, field: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if value < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{field} must be {qualifier}")
    return value


@dataclass(frozen=True, slots=True)
class CausalAlphaV3NestedSelectionConfig:
    signal_contract_count: int
    minimum_economic_contract_count: int

    def __post_init__(self) -> None:
        _non_negative_int(
            self.signal_contract_count,
            field="signal_contract_count",
            positive=True,
        )
        _non_negative_int(
            self.minimum_economic_contract_count,
            field="minimum_economic_contract_count",
            positive=True,
        )

    @property
    def digest(self) -> str:
        return content_digest(self)


@dataclass(frozen=True, slots=True)
class CausalAlphaV3SignalGate:
    minimum_scope_count: int
    minimum_scope_coverage: float
    minimum_rank_ic_lower_ci: float
    minimum_top_bottom_spread_lower_ci: float
    minimum_direction_accuracy_excess_lower_ci: float
    bootstrap_resamples: int
    bootstrap_seed: int
    bootstrap_block_size: int

    def __post_init__(self) -> None:
        _non_negative_int(
            self.minimum_scope_count, field="minimum_scope_count", positive=True
        )
        coverage = _finite(self.minimum_scope_coverage, field="minimum_scope_coverage")
        if not 0.0 < coverage <= 1.0:
            raise ValueError("minimum_scope_coverage must be within (0, 1]")
        for field in (
            "minimum_rank_ic_lower_ci",
            "minimum_top_bottom_spread_lower_ci",
            "minimum_direction_accuracy_excess_lower_ci",
        ):
            _finite(getattr(self, field), field=field)
        _non_negative_int(
            self.bootstrap_resamples, field="bootstrap_resamples", positive=True
        )
        _non_negative_int(self.bootstrap_seed, field="bootstrap_seed")
        _non_negative_int(
            self.bootstrap_block_size,
            field="bootstrap_block_size",
            positive=True,
        )

    @property
    def digest(self) -> str:
        return content_digest(self)


@dataclass(frozen=True, slots=True)
class CausalAlphaV3SelectionGate:
    minimum_mean_gross_return: float
    minimum_mean_net_return: float
    minimum_symbol_episode_net_return: float
    maximum_mean_turnover_per_day: float
    maximum_unexplained_execution_rejections: int
    minimum_positive_gross_episode_fraction: float

    def __post_init__(self) -> None:
        for field in (
            "minimum_mean_gross_return",
            "minimum_mean_net_return",
            "minimum_symbol_episode_net_return",
            "maximum_mean_turnover_per_day",
            "minimum_positive_gross_episode_fraction",
        ):
            _finite(getattr(self, field), field=field)
        if self.maximum_mean_turnover_per_day < 0.0:
            raise ValueError("maximum_mean_turnover_per_day must be non-negative")
        if not 0.0 <= self.minimum_positive_gross_episode_fraction <= 1.0:
            raise ValueError(
                "minimum_positive_gross_episode_fraction must be within [0, 1]"
            )
        _non_negative_int(
            self.maximum_unexplained_execution_rejections,
            field="maximum_unexplained_execution_rejections",
        )

    @property
    def digest(self) -> str:
        return content_digest(self)


@dataclass(frozen=True, slots=True)
class CausalAlphaV3Candidate:
    name: str
    fit: CausalAlphaV3FitConfig
    target: CausalAlphaV3TargetConfig

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("V3 candidate name must be non-empty")
        if not isinstance(self.fit, CausalAlphaV3FitConfig):
            raise TypeError("V3 candidate fit config is invalid")
        if not isinstance(self.target, CausalAlphaV3TargetConfig):
            raise TypeError("V3 candidate target config is invalid")
        object.__setattr__(self, "name", self.name.strip())

    @property
    def semantic_digest(self) -> str:
        return content_digest(
            {
                "fit_config_digest": self.fit.digest,
                "schema_version": "causal_alpha_v3_candidate_semantics_v1",
                "target_config_digest": self.target.digest,
            }
        )

    @property
    def digest(self) -> str:
        return content_digest(
            {
                "name": self.name,
                "schema_version": "causal_alpha_v3_candidate_v1",
                "semantic_digest": self.semantic_digest,
            }
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "artifact_digest": self.digest,
            "fit_config_digest": self.fit.digest,
            "name": self.name,
            "semantic_digest": self.semantic_digest,
            "target_config_digest": self.target.digest,
        }


@dataclass(frozen=True, slots=True)
class CausalAlphaV3ResearchConfig:
    nested_selection: CausalAlphaV3NestedSelectionConfig
    signal_gate: CausalAlphaV3SignalGate
    selection_gate: CausalAlphaV3SelectionGate
    candidates: tuple[CausalAlphaV3Candidate, ...]
    schema_version: str = CAUSAL_ALPHA_V3_RESEARCH_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != CAUSAL_ALPHA_V3_RESEARCH_CONFIG_SCHEMA:
            raise ValueError("unsupported causal alpha V3 research config schema")
        values = tuple(self.candidates)
        if not values or len(values) > _MAX_CANDIDATES:
            raise ValueError(
                "V3 research candidates must contain between 1 and 8 items"
            )
        names = tuple(item.name for item in values)
        semantics = tuple(item.semantic_digest for item in values)
        if len(set(names)) != len(names):
            raise ValueError("V3 research candidate names must be unique")
        if len(set(semantics)) != len(semantics):
            raise ValueError("V3 research candidate semantic configs must be unique")
        object.__setattr__(self, "candidates", values)

    @classmethod
    def from_mapping(cls, raw: object) -> CausalAlphaV3ResearchConfig:
        payload = require_exact_fields(
            _mapping(raw, field="causal alpha V3 research config"),
            required={
                "schema_version",
                "nested_selection",
                "signal_gate",
                "selection_gate",
                "candidates",
            },
            optional=set(),
            field="causal alpha V3 research config",
        )
        nested_raw = require_exact_fields(
            _mapping(payload["nested_selection"], field="nested_selection"),
            required={"signal_contract_count", "minimum_economic_contract_count"},
            optional=set(),
            field="nested_selection",
        )
        signal_raw = require_exact_fields(
            _mapping(payload["signal_gate"], field="signal_gate"),
            required={
                "minimum_scope_count",
                "minimum_scope_coverage",
                "minimum_rank_ic_lower_ci",
                "minimum_top_bottom_spread_lower_ci",
                "minimum_direction_accuracy_excess_lower_ci",
                "bootstrap_resamples",
                "bootstrap_seed",
                "bootstrap_block_size",
            },
            optional=set(),
            field="signal_gate",
        )
        selection_raw = require_exact_fields(
            _mapping(payload["selection_gate"], field="selection_gate"),
            required={
                "minimum_mean_gross_return",
                "minimum_mean_net_return",
                "minimum_symbol_episode_net_return",
                "maximum_mean_turnover_per_day",
                "maximum_unexplained_execution_rejections",
                "minimum_positive_gross_episode_fraction",
            },
            optional=set(),
            field="selection_gate",
        )
        raw_candidates = payload["candidates"]
        if not isinstance(raw_candidates, list | tuple):
            raise ValueError("candidates must be a JSON array")
        candidates: list[CausalAlphaV3Candidate] = []
        for index, value in enumerate(raw_candidates):
            candidate_raw = require_exact_fields(
                _mapping(value, field=f"candidates[{index}]"),
                required={"name", "fit", "target"},
                optional=set(),
                field=f"candidates[{index}]",
            )
            fit_raw = require_exact_fields(
                _mapping(candidate_raw["fit"], field=f"candidates[{index}].fit"),
                required={"ridge_strength"},
                optional=set(),
                field=f"candidates[{index}].fit",
            )
            target_raw = require_exact_fields(
                _mapping(candidate_raw["target"], field=f"candidates[{index}].target"),
                required={
                    "target_magnitudes",
                    "uncertainty_multiplier",
                    "execution_cost_multiplier",
                    "edge_margin",
                    "alpha_rebalance_decisions",
                    "strong_reversal_threshold",
                    "max_target_delta",
                },
                optional=set(),
                field=f"candidates[{index}].target",
            )
            magnitudes = target_raw["target_magnitudes"]
            if not isinstance(magnitudes, list | tuple):
                raise ValueError("target_magnitudes must be a JSON array")
            candidates.append(
                CausalAlphaV3Candidate(
                    name=str(candidate_raw["name"]),
                    fit=CausalAlphaV3FitConfig(
                        ridge_strength=_finite(
                            fit_raw["ridge_strength"],
                            field=f"candidates[{index}].fit.ridge_strength",
                        )
                    ),
                    target=CausalAlphaV3TargetConfig(
                        target_magnitudes=tuple(float(item) for item in magnitudes),
                        uncertainty_multiplier=_finite(
                            target_raw["uncertainty_multiplier"],
                            field=f"candidates[{index}].target.uncertainty_multiplier",
                        ),
                        execution_cost_multiplier=_finite(
                            target_raw["execution_cost_multiplier"],
                            field=f"candidates[{index}].target.execution_cost_multiplier",
                        ),
                        edge_margin=_finite(
                            target_raw["edge_margin"],
                            field=f"candidates[{index}].target.edge_margin",
                        ),
                        alpha_rebalance_decisions=_non_negative_int(
                            target_raw["alpha_rebalance_decisions"],
                            field=f"candidates[{index}].target.alpha_rebalance_decisions",
                            positive=True,
                        ),
                        strong_reversal_threshold=_finite(
                            target_raw["strong_reversal_threshold"],
                            field=f"candidates[{index}].target.strong_reversal_threshold",
                        ),
                        max_target_delta=_finite(
                            target_raw["max_target_delta"],
                            field=f"candidates[{index}].target.max_target_delta",
                        ),
                    ),
                )
            )
        return cls(
            nested_selection=CausalAlphaV3NestedSelectionConfig(
                signal_contract_count=_non_negative_int(
                    nested_raw["signal_contract_count"],
                    field="nested_selection.signal_contract_count",
                    positive=True,
                ),
                minimum_economic_contract_count=_non_negative_int(
                    nested_raw["minimum_economic_contract_count"],
                    field="nested_selection.minimum_economic_contract_count",
                    positive=True,
                ),
            ),
            signal_gate=CausalAlphaV3SignalGate(
                minimum_scope_count=_non_negative_int(
                    signal_raw["minimum_scope_count"],
                    field="signal_gate.minimum_scope_count",
                    positive=True,
                ),
                minimum_scope_coverage=_finite(
                    signal_raw["minimum_scope_coverage"],
                    field="signal_gate.minimum_scope_coverage",
                ),
                minimum_rank_ic_lower_ci=_finite(
                    signal_raw["minimum_rank_ic_lower_ci"],
                    field="signal_gate.minimum_rank_ic_lower_ci",
                ),
                minimum_top_bottom_spread_lower_ci=_finite(
                    signal_raw["minimum_top_bottom_spread_lower_ci"],
                    field="signal_gate.minimum_top_bottom_spread_lower_ci",
                ),
                minimum_direction_accuracy_excess_lower_ci=_finite(
                    signal_raw["minimum_direction_accuracy_excess_lower_ci"],
                    field="signal_gate.minimum_direction_accuracy_excess_lower_ci",
                ),
                bootstrap_resamples=_non_negative_int(
                    signal_raw["bootstrap_resamples"],
                    field="signal_gate.bootstrap_resamples",
                    positive=True,
                ),
                bootstrap_seed=_non_negative_int(
                    signal_raw["bootstrap_seed"],
                    field="signal_gate.bootstrap_seed",
                ),
                bootstrap_block_size=_non_negative_int(
                    signal_raw["bootstrap_block_size"],
                    field="signal_gate.bootstrap_block_size",
                    positive=True,
                ),
            ),
            selection_gate=CausalAlphaV3SelectionGate(
                minimum_mean_gross_return=_finite(
                    selection_raw["minimum_mean_gross_return"],
                    field="selection_gate.minimum_mean_gross_return",
                ),
                minimum_mean_net_return=_finite(
                    selection_raw["minimum_mean_net_return"],
                    field="selection_gate.minimum_mean_net_return",
                ),
                minimum_symbol_episode_net_return=_finite(
                    selection_raw["minimum_symbol_episode_net_return"],
                    field="selection_gate.minimum_symbol_episode_net_return",
                ),
                maximum_mean_turnover_per_day=_finite(
                    selection_raw["maximum_mean_turnover_per_day"],
                    field="selection_gate.maximum_mean_turnover_per_day",
                ),
                maximum_unexplained_execution_rejections=_non_negative_int(
                    selection_raw["maximum_unexplained_execution_rejections"],
                    field="selection_gate.maximum_unexplained_execution_rejections",
                ),
                minimum_positive_gross_episode_fraction=_finite(
                    selection_raw["minimum_positive_gross_episode_fraction"],
                    field="selection_gate.minimum_positive_gross_episode_fraction",
                ),
            ),
            candidates=tuple(candidates),
            schema_version=str(payload["schema_version"]),
        )

    @classmethod
    def from_json(cls, path: Path) -> CausalAlphaV3ResearchConfig:
        source = Path(path)
        raw = json.loads(source.read_text(encoding="utf-8"))
        return cls.from_mapping(raw)

    @property
    def digest(self) -> str:
        return content_digest(
            {
                "candidate_digests": tuple(item.digest for item in self.candidates),
                "nested_selection_digest": self.nested_selection.digest,
                "schema_version": self.schema_version,
                "selection_gate_digest": self.selection_gate.digest,
                "signal_gate_digest": self.signal_gate.digest,
            }
        )


__all__ = [
    "CAUSAL_ALPHA_V3_RESEARCH_CONFIG_SCHEMA",
    "CausalAlphaV3Candidate",
    "CausalAlphaV3NestedSelectionConfig",
    "CausalAlphaV3ResearchConfig",
    "CausalAlphaV3SelectionGate",
    "CausalAlphaV3SignalGate",
]
