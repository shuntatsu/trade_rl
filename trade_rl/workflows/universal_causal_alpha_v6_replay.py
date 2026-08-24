"""Simulator-authoritative economic replay evidence for Causal Alpha V6."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.learning.causal_alpha_v6 import (
    CausalAlphaV6Candidate,
    CausalAlphaV6TargetPath,
)
from trade_rl.learning.rollout_evaluation import ActionPathEvaluation

CAUSAL_ALPHA_V6_REPLAY_SCHEMA: Final = "causal_alpha_v6_replay_metric_v1"
_ACTION_CHANGE_TOLERANCE: Final = 1e-6


def _reason_counts(
    value: Iterable[tuple[str, int]], *, field: str
) -> tuple[tuple[str, int], ...]:
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
) -> tuple[tuple[float, ...], float]:
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
            duration = step_hours if next_sign != current_sign else duration + step_hours
        current_sign = next_sign
    return tuple(completed), duration if current_sign != 0 else 0.0


def _non_negative_count(value: object, *, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"V6 replay {field} is invalid")


@dataclass(frozen=True, slots=True)
class CausalAlphaV6ReplayMetric:
    """One independent symbol/episode replay with complete economics."""

    run_manifest_digest: str
    v4_context_manifest_digest: str
    config_digest: str
    candidate: CausalAlphaV6Candidate
    symbol: str
    episode_index: int
    contract_digest: str
    fit_digest: str
    forecast_digest: str
    target_path_digest: str
    decision_count: int
    gross_return: float
    gross_wealth: float
    net_return: float
    net_wealth: float
    reward_total: float
    reward_scale: float
    turnover_per_day: float
    total_execution_cost: float
    target_change_count: int
    submitted_change_count: int
    downstream_no_trade_suppression_count: int
    executed_change_count: int
    closed_trade_count: int
    sign_flip_count: int
    maximum_drawdown: float
    actionable_coverage: float
    flat_time_fraction: float
    time_weighted_absolute_exposure: float
    completed_holding_durations_hours: tuple[float, ...]
    open_holding_duration_hours: float
    execution_rejection_reason_counts: tuple[tuple[str, int], ...]
    risk_projection_reason_counts: tuple[tuple[str, int], ...]
    target_reason_counts: tuple[tuple[str, int], ...]
    hard_risk_violation: bool
    has_meaningful_execution: bool
    schema_version: str = CAUSAL_ALPHA_V6_REPLAY_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        for name in (
            "run_manifest_digest",
            "v4_context_manifest_digest",
            "config_digest",
            "contract_digest",
            "fit_digest",
            "forecast_digest",
            "target_path_digest",
        ):
            require_sha256(getattr(self, name), field=f"V6 replay {name}")
        candidate = CausalAlphaV6Candidate(self.candidate)
        if not self.symbol or isinstance(self.episode_index, bool) or self.episode_index < 0:
            raise ValueError("V6 replay scope identity is invalid")
        for name in (
            "decision_count",
            "target_change_count",
            "submitted_change_count",
            "downstream_no_trade_suppression_count",
            "executed_change_count",
            "closed_trade_count",
            "sign_flip_count",
        ):
            _non_negative_count(getattr(self, name), field=name)
        if self.decision_count <= 0:
            raise ValueError("V6 replay decision count must be positive")
        if self.downstream_no_trade_suppression_count > self.submitted_change_count:
            raise ValueError("V6 replay suppression exceeds submitted changes")
        finite_names = (
            "gross_return",
            "gross_wealth",
            "net_return",
            "net_wealth",
            "reward_total",
            "reward_scale",
            "turnover_per_day",
            "total_execution_cost",
            "maximum_drawdown",
            "actionable_coverage",
            "flat_time_fraction",
            "time_weighted_absolute_exposure",
            "open_holding_duration_hours",
        )
        if any(not math.isfinite(getattr(self, name)) for name in finite_names):
            raise ValueError("V6 replay numeric evidence must be finite")
        if self.reward_scale <= 0.0:
            raise ValueError("V6 replay reward scale must be positive")
        if not math.isclose(
            self.gross_wealth, math.exp(self.gross_return), rel_tol=1e-12, abs_tol=1e-12
        ) or not math.isclose(
            self.net_wealth, math.exp(self.net_return), rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError("V6 replay wealth does not match log return")
        if not math.isclose(
            self.reward_total,
            self.net_return * self.reward_scale,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError("V6 replay reward does not match scaled net log return")
        if any(
            getattr(self, name) < 0.0
            for name in (
                "turnover_per_day",
                "total_execution_cost",
                "maximum_drawdown",
                "time_weighted_absolute_exposure",
                "open_holding_duration_hours",
            )
        ):
            raise ValueError("V6 replay cost/risk/holding evidence became negative")
        if not 0.0 <= self.actionable_coverage <= 1.0 or not 0.0 <= self.flat_time_fraction <= 1.0:
            raise ValueError("V6 replay fractions are invalid")
        durations = tuple(float(value) for value in self.completed_holding_durations_hours)
        if any(not math.isfinite(value) or value <= 0.0 for value in durations):
            raise ValueError("V6 replay completed holding durations are invalid")
        execution = _reason_counts(
            self.execution_rejection_reason_counts,
            field="V6 execution rejections",
        )
        risk = _reason_counts(
            self.risk_projection_reason_counts,
            field="V6 risk projections",
        )
        target = _reason_counts(self.target_reason_counts, field="V6 target reasons")
        if sum(count for _, count in target) != self.decision_count:
            raise ValueError("V6 replay target reasons do not cover decisions")
        if not isinstance(self.hard_risk_violation, bool) or not isinstance(
            self.has_meaningful_execution, bool
        ):
            raise ValueError("V6 replay boolean evidence is invalid")
        if self.executed_change_count > 0 and not self.has_meaningful_execution:
            raise ValueError("V6 executed changes require meaningful execution")
        if self.schema_version != CAUSAL_ALPHA_V6_REPLAY_SCHEMA:
            raise ValueError("unsupported V6 replay schema")
        object.__setattr__(self, "candidate", candidate)
        object.__setattr__(self, "completed_holding_durations_hours", durations)
        object.__setattr__(self, "execution_rejection_reason_counts", execution)
        object.__setattr__(self, "risk_projection_reason_counts", risk)
        object.__setattr__(self, "target_reason_counts", target)
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V6 replay metric digest mismatch")
        object.__setattr__(self, "digest", expected)

    @property
    def identity(self) -> tuple[str, str, int]:
        return (self.candidate.value, self.symbol, self.episode_index)

    @property
    def paired_identity(self) -> tuple[str, int, str, str, str]:
        return (
            self.symbol,
            self.episode_index,
            self.contract_digest,
            self.fit_digest,
            self.forecast_digest,
        )

    @property
    def execution_rejection_count(self) -> int:
        return sum(count for _, count in self.execution_rejection_reason_counts)

    @property
    def risk_projection_count(self) -> int:
        return sum(count for _, count in self.risk_projection_reason_counts)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in {"candidate", "digest"}
        }
        payload["candidate"] = self.candidate.value
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def build_causal_alpha_v6_replay_metric(
    *,
    run_manifest_digest: str,
    v4_context_manifest_digest: str,
    symbol: str,
    episode_index: int,
    contract_digest: str,
    fit_digest: str,
    forecast_digest: str,
    target_path: CausalAlphaV6TargetPath,
    evaluation: ActionPathEvaluation,
    episode_hours: float,
    reward_scale: float,
    action_change_tolerance: float = _ACTION_CHANGE_TOLERANCE,
) -> CausalAlphaV6ReplayMetric:
    """Bind V6 attribution to maintained simulator PnL and reward fields."""

    if not isinstance(target_path, CausalAlphaV6TargetPath):
        raise TypeError("V6 replay target path is invalid")
    if not isinstance(evaluation, ActionPathEvaluation):
        raise TypeError("V6 replay evaluation is invalid")
    if target_path.forecast_digest != forecast_digest:
        raise ValueError("V6 replay forecast identity drifted")
    if not math.isfinite(episode_hours) or episode_hours <= 0.0:
        raise ValueError("V6 replay episode_hours must be positive")
    if not math.isfinite(reward_scale) or reward_scale <= 0.0:
        raise ValueError("V6 replay reward_scale must be positive")
    if not math.isfinite(action_change_tolerance) or action_change_tolerance < 0.0:
        raise ValueError("V6 replay action change tolerance is invalid")
    performance = evaluation.performance
    collapse = evaluation.collapse_evidence
    rows = int(target_path.targets.size)
    if rows != performance.step_count or rows != collapse.decision_count:
        raise ValueError("V6 replay target path does not cover the evaluation")
    if not math.isclose(
        performance.reward_total,
        performance.net_return * reward_scale,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise ValueError("V6 replay evaluator reward is not pure scaled net log return")
    meaningful = bool(
        collapse.executed_change_count > 0
        or performance.turnover_total > action_change_tolerance
    )
    step_hours = episode_hours / rows
    durations, open_duration = _holding_attribution(
        initial_weight=target_path.initial_weight,
        targets=np.asarray(target_path.targets),
        step_hours=step_hours,
    )
    return CausalAlphaV6ReplayMetric(
        run_manifest_digest=run_manifest_digest,
        v4_context_manifest_digest=v4_context_manifest_digest,
        config_digest=target_path.config_digest,
        candidate=target_path.candidate,
        symbol=symbol,
        episode_index=episode_index,
        contract_digest=contract_digest,
        fit_digest=fit_digest,
        forecast_digest=forecast_digest,
        target_path_digest=target_path.digest,
        decision_count=rows,
        gross_return=float(performance.gross_return),
        gross_wealth=math.exp(float(performance.gross_return)),
        net_return=float(performance.net_return),
        net_wealth=math.exp(float(performance.net_return)),
        reward_total=float(performance.reward_total),
        reward_scale=float(reward_scale),
        turnover_per_day=float(performance.turnover_total) / (episode_hours / 24.0),
        total_execution_cost=float(performance.cost_total),
        target_change_count=target_path.submitted_change_count,
        submitted_change_count=int(collapse.submitted_change_count),
        downstream_no_trade_suppression_count=int(
            collapse.downstream_no_trade_suppression_count
        ),
        executed_change_count=int(collapse.executed_change_count),
        closed_trade_count=int(performance.trade_count),
        sign_flip_count=target_path.sign_flip_count,
        maximum_drawdown=float(performance.maximum_drawdown),
        actionable_coverage=float(np.mean(target_path.actionable_mask)),
        flat_time_fraction=float(
            np.mean(np.abs(target_path.targets) <= action_change_tolerance)
        ),
        time_weighted_absolute_exposure=float(np.mean(np.abs(target_path.targets))),
        completed_holding_durations_hours=durations,
        open_holding_duration_hours=open_duration,
        execution_rejection_reason_counts=tuple(
            collapse.execution_rejection_reason_counts
        ),
        risk_projection_reason_counts=tuple(collapse.risk_projection_reason_counts),
        target_reason_counts=target_path.reason_counts,
        hard_risk_violation=bool(collapse.hard_risk_violation),
        has_meaningful_execution=meaningful,
    )


__all__ = [
    "CAUSAL_ALPHA_V6_REPLAY_SCHEMA",
    "CausalAlphaV6ReplayMetric",
    "build_causal_alpha_v6_replay_metric",
]
