"""Validated configuration for concrete market walk-forward workflows."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from trade_rl.artifacts.hashing import content_digest
from trade_rl.workflows.fold_runner import BASELINE_CONFIGURATION
from trade_rl.workflows.training_run import TrainingRunConfig
from trade_rl.workflows.walk_forward import WalkForwardWorkflowConfig


def _mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    return dict(value)


def _environment_dynamics_payload(run: TrainingRunConfig) -> dict[str, object]:
    payload = asdict(run.environment)
    payload.pop("structured_sequence_observation", None)
    payload.pop("sequence_windows", None)
    return payload


@dataclass(frozen=True, slots=True)
class NamedCandidateRun:
    name: str
    run: TrainingRunConfig

    def __post_init__(self) -> None:
        if not self.name or self.name == BASELINE_CONFIGURATION:
            raise ValueError("candidate name is empty or reserved")


@dataclass(frozen=True, slots=True)
class MarketWalkForwardConfig:
    workflow: WalkForwardWorkflowConfig
    candidates: tuple[NamedCandidateRun, ...]
    minimum_selection_uplift: float = 0.0
    minimum_selection_score: float = 0.0
    minimum_seed_success_fraction: float = 0.0
    minimum_worst_seed_uplift: float | None = None
    maximum_seed_score_std: float | None = None
    maximum_selection_turnover_per_day: float | None = None
    maximum_selection_cost_fraction: float | None = None
    maximum_selection_drawdown: float | None = None
    checkpoint_finalists_per_seed: int = 1
    signal_digest: str = ""
    schema_version: str = "market_walk_forward_config_v1"

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ValueError("walk-forward requires at least one candidate")
        names = tuple(item.name for item in self.candidates)
        if len(set(names)) != len(names):
            raise ValueError("walk-forward candidate names must be unique")
        if (
            not math.isfinite(self.minimum_selection_uplift)
            or self.minimum_selection_uplift < 0.0
        ):
            raise ValueError("minimum_selection_uplift must be non-negative")
        if not math.isfinite(self.minimum_selection_score):
            raise ValueError("minimum_selection_score must be finite")
        if (
            not math.isfinite(self.minimum_seed_success_fraction)
            or not 0.0 <= self.minimum_seed_success_fraction <= 1.0
        ):
            raise ValueError("minimum_seed_success_fraction must be within [0, 1]")
        for field_name, value in (
            ("minimum_worst_seed_uplift", self.minimum_worst_seed_uplift),
            ("maximum_seed_score_std", self.maximum_seed_score_std),
            (
                "maximum_selection_turnover_per_day",
                self.maximum_selection_turnover_per_day,
            ),
            (
                "maximum_selection_cost_fraction",
                self.maximum_selection_cost_fraction,
            ),
            ("maximum_selection_drawdown", self.maximum_selection_drawdown),
        ):
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
            if (
                value is not None
                and field_name != "minimum_worst_seed_uplift"
                and value < 0.0
            ):
                raise ValueError(f"{field_name} must be non-negative")
        if (
            isinstance(self.checkpoint_finalists_per_seed, bool)
            or not isinstance(self.checkpoint_finalists_per_seed, int)
            or self.checkpoint_finalists_per_seed <= 0
        ):
            raise ValueError("checkpoint_finalists_per_seed must be a positive integer")
        if not self.signal_digest:
            object.__setattr__(
                self,
                "signal_digest",
                content_digest(
                    {
                        "schema_version": "trend_baseline_signal_v1",
                        "trend": asdict(self.candidates[0].run.trend),
                    }
                ),
            )
        if len(self.signal_digest) != 64:
            raise ValueError("signal_digest must be a SHA-256 digest")
        common = {
            content_digest(
                {
                    "action": asdict(item.run.action),
                    "alpha_contract": asdict(item.run.alpha_contract),
                    "environment_dynamics": _environment_dynamics_payload(item.run),
                    "portfolio_risk": asdict(item.run.portfolio_risk),
                    "risk": asdict(item.run.risk),
                    "reward": asdict(item.run.reward),
                    "trend": asdict(item.run.trend),
                }
            )
            for item in self.candidates
        }
        if len(common) != 1:
            raise ValueError(
                "walk-forward candidates must share environment dynamics, action, risk, "
                "reward, and trend contracts"
            )
        if self.schema_version != "market_walk_forward_config_v1":
            raise ValueError("unsupported market walk-forward configuration schema")

    @classmethod
    def from_json(
        cls,
        path: Path,
        *,
        n_bars: int,
    ) -> MarketWalkForwardConfig:
        payload = _mapping(
            json.loads(path.read_text(encoding="utf-8")),
            field="walk-forward config",
        )
        workflow_payload = _mapping(payload.get("workflow"), field="workflow")
        workflow_payload.pop("n_bars", None)
        raw_candidates = payload.get("candidates")
        if not isinstance(raw_candidates, list) or not raw_candidates:
            raise ValueError("candidates must be a non-empty list")
        candidates: list[NamedCandidateRun] = []
        for index, raw_candidate in enumerate(raw_candidates):
            candidate = _mapping(raw_candidate, field=f"candidates[{index}]")
            name = candidate.get("name")
            if not isinstance(name, str):
                raise ValueError(f"candidates[{index}].name must be a string")
            candidates.append(
                NamedCandidateRun(
                    name=name,
                    run=TrainingRunConfig.from_mapping(
                        candidate.get("run")
                    ).resolve_artifact_paths(path.parent),
                )
            )
        signal = payload.get("signal_digest", "")
        if not isinstance(signal, str):
            raise ValueError("signal_digest must be a string")
        finalists_per_seed = payload.get("checkpoint_finalists_per_seed", 1)
        if (
            isinstance(finalists_per_seed, bool)
            or not isinstance(finalists_per_seed, int)
            or finalists_per_seed <= 0
        ):
            raise ValueError("checkpoint_finalists_per_seed must be a positive integer")
        return cls(
            workflow=WalkForwardWorkflowConfig(
                n_bars=n_bars,
                **workflow_payload,
            ),
            candidates=tuple(candidates),
            minimum_selection_uplift=float(
                payload.get("minimum_selection_uplift", 0.0)
            ),
            minimum_selection_score=float(payload.get("minimum_selection_score", 0.0)),
            minimum_seed_success_fraction=float(
                payload.get("minimum_seed_success_fraction", 0.0)
            ),
            minimum_worst_seed_uplift=(
                None
                if payload.get("minimum_worst_seed_uplift") is None
                else float(payload["minimum_worst_seed_uplift"])
            ),
            maximum_seed_score_std=(
                None
                if payload.get("maximum_seed_score_std") is None
                else float(payload["maximum_seed_score_std"])
            ),
            maximum_selection_turnover_per_day=(
                None
                if payload.get("maximum_selection_turnover_per_day") is None
                else float(payload["maximum_selection_turnover_per_day"])
            ),
            maximum_selection_cost_fraction=(
                None
                if payload.get("maximum_selection_cost_fraction") is None
                else float(payload["maximum_selection_cost_fraction"])
            ),
            maximum_selection_drawdown=(
                None
                if payload.get("maximum_selection_drawdown") is None
                else float(payload["maximum_selection_drawdown"])
            ),
            checkpoint_finalists_per_seed=finalists_per_seed,
            signal_digest=signal,
            schema_version=str(
                payload.get("schema_version", "market_walk_forward_config_v1")
            ),
        )

    def digest_payload(self) -> dict[str, object]:
        return {
            "candidates": tuple(
                {"name": item.name, "run": item.run.digest_payload()}
                for item in self.candidates
            ),
            "checkpoint_finalists_per_seed": self.checkpoint_finalists_per_seed,
            "minimum_selection_uplift": self.minimum_selection_uplift,
            "minimum_selection_score": self.minimum_selection_score,
            "minimum_seed_success_fraction": self.minimum_seed_success_fraction,
            "minimum_worst_seed_uplift": self.minimum_worst_seed_uplift,
            "maximum_seed_score_std": self.maximum_seed_score_std,
            "maximum_selection_turnover_per_day": (
                self.maximum_selection_turnover_per_day
            ),
            "maximum_selection_cost_fraction": (self.maximum_selection_cost_fraction),
            "maximum_selection_drawdown": self.maximum_selection_drawdown,
            "schema_version": self.schema_version,
            "signal_digest": self.signal_digest,
            "workflow": asdict(self.workflow),
        }


__all__ = ["MarketWalkForwardConfig", "NamedCandidateRun"]
