"""Economic replay evidence for the research-only Causal Alpha V4 lane."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Final

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.learning.causal_alpha_v4 import CausalAlphaV4TargetPath
from trade_rl.learning.rollout_evaluation import ActionPathEvaluation

CAUSAL_ALPHA_V4_REPLAY_SCHEMA: Final = "causal_alpha_v4_replay_metric_v1"
_V4_ACTION_CHANGE_TOLERANCE: Final = 1e-6


def _non_negative_count(value: int, *, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")


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
        raise ValueError(f"{field} contains invalid reason counts")
    return resolved


@dataclass(frozen=True, slots=True)
class CausalAlphaV4ReplayMetric:
    run_manifest_digest: str
    v4_context_manifest_digest: str
    config_digest: str
    symbol: str
    episode_index: int
    contract_digest: str
    fit_digest: str
    forecast_digest: str
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
    execution_rejection_reason_counts: tuple[tuple[str, int], ...]
    risk_projection_reason_counts: tuple[tuple[str, int], ...]
    target_reason_counts: tuple[tuple[str, int], ...]
    hard_risk_violation: bool
    has_meaningful_execution: bool
    schema_version: str = CAUSAL_ALPHA_V4_REPLAY_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        for field_name in (
            "run_manifest_digest",
            "v4_context_manifest_digest",
            "config_digest",
            "contract_digest",
            "fit_digest",
            "forecast_digest",
            "target_path_digest",
        ):
            require_sha256(getattr(self, field_name), field=f"V4 replay {field_name}")
        if not isinstance(self.symbol, str) or not self.symbol:
            raise ValueError("V4 replay symbol must be non-empty")
        _non_negative_count(self.episode_index, field="V4 replay episode_index")
        for field_name in (
            "submitted_change_count",
            "downstream_no_trade_suppression_count",
            "executed_change_count",
            "closed_trade_count",
            "sign_flip_count",
        ):
            _non_negative_count(getattr(self, field_name), field=f"V4 replay {field_name}")
        if self.downstream_no_trade_suppression_count > self.submitted_change_count:
            raise ValueError("V4 replay suppression count exceeds submitted changes")
        for field_name in (
            "gross_return",
            "net_return",
            "turnover_per_day",
            "total_execution_cost",
            "maximum_drawdown",
        ):
            value = getattr(self, field_name)
            if not math.isfinite(value):
                raise ValueError(f"V4 replay {field_name} must be finite")
        if self.turnover_per_day < 0.0 or self.total_execution_cost < 0.0:
            raise ValueError("V4 replay turnover/cost must be non-negative")
        if self.maximum_drawdown < 0.0:
            raise ValueError("V4 replay maximum drawdown must be non-negative")
        execution = _reason_counts(
            self.execution_rejection_reason_counts,
            field="V4 replay execution rejections",
        )
        risk = _reason_counts(
            self.risk_projection_reason_counts,
            field="V4 replay risk projections",
        )
        target = _reason_counts(
            self.target_reason_counts,
            field="V4 replay target reasons",
        )
        if not isinstance(self.hard_risk_violation, bool):
            raise ValueError("V4 replay hard_risk_violation must be boolean")
        if not isinstance(self.has_meaningful_execution, bool):
            raise ValueError("V4 replay has_meaningful_execution must be boolean")
        if self.executed_change_count > 0 and not self.has_meaningful_execution:
            raise ValueError("V4 executed changes require meaningful execution")
        if self.schema_version != CAUSAL_ALPHA_V4_REPLAY_SCHEMA:
            raise ValueError("unsupported V4 replay metric schema")
        object.__setattr__(self, "execution_rejection_reason_counts", execution)
        object.__setattr__(self, "risk_projection_reason_counts", risk)
        object.__setattr__(self, "target_reason_counts", target)
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V4 replay metric digest mismatch")
        object.__setattr__(self, "digest", expected)

    @property
    def execution_rejection_count(self) -> int:
        return sum(count for _, count in self.execution_rejection_reason_counts)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "closed_trade_count": self.closed_trade_count,
            "config_digest": self.config_digest,
            "contract_digest": self.contract_digest,
            "downstream_no_trade_suppression_count": (
                self.downstream_no_trade_suppression_count
            ),
            "episode_index": self.episode_index,
            "executed_change_count": self.executed_change_count,
            "execution_rejection_reason_counts": self.execution_rejection_reason_counts,
            "fit_digest": self.fit_digest,
            "forecast_digest": self.forecast_digest,
            "gross_return": self.gross_return,
            "hard_risk_violation": self.hard_risk_violation,
            "has_meaningful_execution": self.has_meaningful_execution,
            "maximum_drawdown": self.maximum_drawdown,
            "net_return": self.net_return,
            "risk_projection_reason_counts": self.risk_projection_reason_counts,
            "run_manifest_digest": self.run_manifest_digest,
            "schema_version": self.schema_version,
            "sign_flip_count": self.sign_flip_count,
            "submitted_change_count": self.submitted_change_count,
            "symbol": self.symbol,
            "target_path_digest": self.target_path_digest,
            "target_reason_counts": self.target_reason_counts,
            "total_execution_cost": self.total_execution_cost,
            "turnover_per_day": self.turnover_per_day,
            "v4_context_manifest_digest": self.v4_context_manifest_digest,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def build_causal_alpha_v4_replay_metric(
    *,
    run_manifest_digest: str,
    v4_context_manifest_digest: str,
    config_digest: str,
    symbol: str,
    episode_index: int,
    contract_digest: str,
    fit_digest: str,
    forecast_digest: str,
    target_path: CausalAlphaV4TargetPath,
    evaluation: ActionPathEvaluation,
    episode_hours: float,
    action_change_tolerance: float = _V4_ACTION_CHANGE_TOLERANCE,
) -> CausalAlphaV4ReplayMetric:
    """Bind V4 target attribution to authoritative simulator replay accounting."""

    if not isinstance(target_path, CausalAlphaV4TargetPath):
        raise TypeError("V4 replay target_path is invalid")
    if not isinstance(evaluation, ActionPathEvaluation):
        raise TypeError("V4 replay evaluation is invalid")
    if not math.isfinite(episode_hours) or episode_hours <= 0.0:
        raise ValueError("V4 replay episode_hours must be positive")
    if not math.isfinite(action_change_tolerance) or action_change_tolerance < 0.0:
        raise ValueError("V4 replay action_change_tolerance must be non-negative")
    if len(target_path.targets) != evaluation.performance.step_count:
        raise ValueError("V4 replay target path does not cover the evaluation")

    performance = evaluation.performance
    collapse = evaluation.collapse_evidence
    episode_days = episode_hours / 24.0
    meaningful = bool(
        collapse.executed_change_count > 0
        or performance.turnover_total > action_change_tolerance
    )
    target_reason_counts = tuple(sorted(Counter(target_path.reasons).items()))
    return CausalAlphaV4ReplayMetric(
        run_manifest_digest=run_manifest_digest,
        v4_context_manifest_digest=v4_context_manifest_digest,
        config_digest=config_digest,
        symbol=symbol,
        episode_index=episode_index,
        contract_digest=contract_digest,
        fit_digest=fit_digest,
        forecast_digest=forecast_digest,
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
        execution_rejection_reason_counts=tuple(
            collapse.execution_rejection_reason_counts
        ),
        risk_projection_reason_counts=tuple(collapse.risk_projection_reason_counts),
        target_reason_counts=target_reason_counts,
        hard_risk_violation=bool(collapse.hard_risk_violation),
        has_meaningful_execution=meaningful,
    )


__all__ = [
    "CAUSAL_ALPHA_V4_REPLAY_SCHEMA",
    "CausalAlphaV4ReplayMetric",
    "build_causal_alpha_v4_replay_metric",
]
