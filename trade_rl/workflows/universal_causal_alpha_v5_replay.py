"""Simulator-authoritative replay attribution for Causal Alpha V5."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.learning.causal_alpha_v5 import CausalAlphaV5TargetPath
from trade_rl.learning.rollout_evaluation import ActionPathEvaluation

CAUSAL_ALPHA_V5_REPLAY_SCHEMA: Final = "causal_alpha_v5_replay_metric_v1"
_ACTION_CHANGE_TOLERANCE: Final = 1e-6


def _reason_counts(value: Any, *, field: str) -> tuple[tuple[str, int], ...]:
    try:
        resolved = tuple((str(reason), int(count)) for reason, count in value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must contain reason/count pairs") from error
    if (
        any(not reason or count <= 0 for reason, count in resolved)
        or len({reason for reason, _ in resolved}) != len(resolved)
        or tuple(sorted(resolved)) != resolved
    ):
        raise ValueError(f"{field} contains malformed reasons")
    return resolved


def _holding_attribution(
    *, initial_weight: float, targets: np.ndarray, step_hours: float
) -> tuple[tuple[float, ...], bool]:
    current_sign = int(np.sign(initial_weight))
    duration = 0.0
    completed: list[float] = []
    for target in targets:
        next_sign = int(np.sign(target))
        if current_sign != 0 and next_sign != current_sign:
            if duration > 0.0:
                completed.append(duration)
            duration = 0.0
        if next_sign != 0:
            duration = (
                step_hours if next_sign != current_sign else duration + step_hours
            )
        current_sign = next_sign
    return tuple(completed), current_sign != 0


@dataclass(frozen=True, slots=True)
class CausalAlphaV5ReplayMetric:
    run_manifest_digest: str
    v4_context_manifest_digest: str
    config_digest: str
    symbol: str
    episode_index: int
    contract_digest: str
    fit_digest: str
    forecast_digest: str
    calibration_fit_digest: str
    target_path_digest: str
    gross_return: float
    net_return: float
    turnover_per_day: float
    total_execution_cost: float
    submitted_change_count: int
    downstream_no_trade_suppression_count: int
    executed_change_count: int
    closed_trade_count: int
    sign_flip_count: int
    maximum_drawdown: float
    active_coverage: float
    flat_time_fraction: float
    time_weighted_absolute_exposure: float
    completed_holding_durations_hours: tuple[float, ...]
    has_unclosed_position: bool
    execution_rejection_reason_counts: tuple[tuple[str, int], ...]
    risk_projection_reason_counts: tuple[tuple[str, int], ...]
    target_reason_counts: tuple[tuple[str, int], ...]
    hard_risk_violation: bool
    has_meaningful_execution: bool
    schema_version: str = CAUSAL_ALPHA_V5_REPLAY_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        for name in (
            "run_manifest_digest",
            "v4_context_manifest_digest",
            "config_digest",
            "contract_digest",
            "fit_digest",
            "forecast_digest",
            "calibration_fit_digest",
            "target_path_digest",
        ):
            require_sha256(getattr(self, name), field=f"V5 replay {name}")
        if (
            not self.symbol
            or isinstance(self.episode_index, bool)
            or self.episode_index < 0
        ):
            raise ValueError("V5 replay scope identity is invalid")
        for name in (
            "submitted_change_count",
            "downstream_no_trade_suppression_count",
            "executed_change_count",
            "closed_trade_count",
            "sign_flip_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"V5 replay {name} is invalid")
        if self.downstream_no_trade_suppression_count > self.submitted_change_count:
            raise ValueError("V5 replay suppression exceeds submitted changes")
        for name in (
            "gross_return",
            "net_return",
            "turnover_per_day",
            "total_execution_cost",
            "maximum_drawdown",
            "active_coverage",
            "flat_time_fraction",
            "time_weighted_absolute_exposure",
        ):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"V5 replay {name} must be finite")
        if (
            self.turnover_per_day < 0.0
            or self.total_execution_cost < 0.0
            or self.maximum_drawdown < 0.0
        ):
            raise ValueError("V5 replay cost/risk values must be non-negative")
        if (
            not 0.0 <= self.active_coverage <= 1.0
            or not 0.0 <= self.flat_time_fraction <= 1.0
        ):
            raise ValueError("V5 replay fractions are invalid")
        if self.time_weighted_absolute_exposure < 0.0:
            raise ValueError("V5 replay exposure must be non-negative")
        durations = tuple(
            float(value) for value in self.completed_holding_durations_hours
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in durations):
            raise ValueError("V5 replay completed holding durations are invalid")
        if not isinstance(self.has_unclosed_position, bool):
            raise ValueError("V5 replay unclosed state must be boolean")
        execution = _reason_counts(
            self.execution_rejection_reason_counts, field="V5 execution rejections"
        )
        risk = _reason_counts(
            self.risk_projection_reason_counts, field="V5 risk projections"
        )
        target = _reason_counts(self.target_reason_counts, field="V5 target reasons")
        if not isinstance(self.hard_risk_violation, bool) or not isinstance(
            self.has_meaningful_execution, bool
        ):
            raise ValueError("V5 replay boolean evidence is invalid")
        if self.executed_change_count > 0 and not self.has_meaningful_execution:
            raise ValueError("V5 executed changes require meaningful execution")
        if self.schema_version != CAUSAL_ALPHA_V5_REPLAY_SCHEMA:
            raise ValueError("unsupported V5 replay schema")
        object.__setattr__(self, "completed_holding_durations_hours", durations)
        object.__setattr__(self, "execution_rejection_reason_counts", execution)
        object.__setattr__(self, "risk_projection_reason_counts", risk)
        object.__setattr__(self, "target_reason_counts", target)
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V5 replay metric digest mismatch")
        object.__setattr__(self, "digest", expected)

    @property
    def identity(self) -> tuple[str, int]:
        return (self.symbol, self.episode_index)

    @property
    def execution_rejection_count(self) -> int:
        return sum(count for _, count in self.execution_rejection_reason_counts)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "digest"
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def build_causal_alpha_v5_replay_metric(
    *,
    run_manifest_digest: str,
    v4_context_manifest_digest: str,
    config_digest: str,
    symbol: str,
    episode_index: int,
    contract_digest: str,
    fit_digest: str,
    forecast_digest: str,
    calibration_fit_digest: str,
    target_path: CausalAlphaV5TargetPath,
    evaluation: ActionPathEvaluation,
    episode_hours: float,
    action_change_tolerance: float = _ACTION_CHANGE_TOLERANCE,
) -> CausalAlphaV5ReplayMetric:
    """Add selective attribution without reconstructing simulator PnL."""

    if not isinstance(target_path, CausalAlphaV5TargetPath):
        raise TypeError("V5 replay target path is invalid")
    if not isinstance(evaluation, ActionPathEvaluation):
        raise TypeError("V5 replay evaluation is invalid")
    if not math.isfinite(episode_hours) or episode_hours <= 0.0:
        raise ValueError("V5 replay episode_hours must be positive")
    if not math.isfinite(action_change_tolerance) or action_change_tolerance < 0.0:
        raise ValueError("V5 replay action change tolerance is invalid")
    performance = evaluation.performance
    collapse = evaluation.collapse_evidence
    if len(target_path.targets) != performance.step_count:
        raise ValueError("V5 replay target path does not cover the evaluation")
    meaningful = bool(
        collapse.executed_change_count > 0
        or performance.turnover_total > action_change_tolerance
    )
    episode_days = episode_hours / 24.0
    step_hours = episode_hours / performance.step_count
    durations, unclosed = _holding_attribution(
        initial_weight=target_path.initial_weight,
        targets=np.asarray(target_path.targets),
        step_hours=step_hours,
    )
    return CausalAlphaV5ReplayMetric(
        run_manifest_digest=run_manifest_digest,
        v4_context_manifest_digest=v4_context_manifest_digest,
        config_digest=config_digest,
        symbol=symbol,
        episode_index=episode_index,
        contract_digest=contract_digest,
        fit_digest=fit_digest,
        forecast_digest=forecast_digest,
        calibration_fit_digest=calibration_fit_digest,
        target_path_digest=target_path.digest,
        gross_return=float(performance.gross_return),
        net_return=float(performance.net_return),
        turnover_per_day=float(performance.turnover_total) / episode_days,
        total_execution_cost=float(performance.cost_total),
        submitted_change_count=int(collapse.submitted_change_count),
        downstream_no_trade_suppression_count=int(
            collapse.downstream_no_trade_suppression_count
        ),
        executed_change_count=int(collapse.executed_change_count),
        closed_trade_count=int(performance.trade_count),
        sign_flip_count=int(target_path.sign_flip_count),
        maximum_drawdown=float(performance.maximum_drawdown),
        active_coverage=float(np.mean(target_path.active_mask)),
        flat_time_fraction=float(
            np.mean(np.abs(target_path.targets) <= action_change_tolerance)
        ),
        time_weighted_absolute_exposure=float(np.mean(np.abs(target_path.targets))),
        completed_holding_durations_hours=durations,
        has_unclosed_position=unclosed,
        execution_rejection_reason_counts=tuple(
            collapse.execution_rejection_reason_counts
        ),
        risk_projection_reason_counts=tuple(collapse.risk_projection_reason_counts),
        target_reason_counts=tuple(sorted(Counter(target_path.reasons).items())),
        hard_risk_violation=bool(collapse.hard_risk_violation),
        has_meaningful_execution=meaningful,
    )


__all__ = [
    "CAUSAL_ALPHA_V5_REPLAY_SCHEMA",
    "CausalAlphaV5ReplayMetric",
    "build_causal_alpha_v5_replay_metric",
]
