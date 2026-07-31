"""Strict episode result contracts for Stage A production evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Self

from trade_rl.domain.common import require_sha256
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.orders import OrderBookState, OrderEvent
from trade_rl.workflows.stage_a_zero_shot_runner_contracts import (
    StageAEvaluationCellRequest,
)


def _finite_float(value: float | int, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number")
    try:
        resolved = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a finite number") from error
    if not math.isfinite(resolved):
        raise ValueError(f"{field} must be finite")
    return resolved


def _optional_sha256(value: str | None, *, field: str) -> str | None:
    if value is None:
        return None
    require_sha256(value, field=field)
    return value


@dataclass(frozen=True, slots=True)
class StageAEvaluationEpisodeResult:
    """One validated environment episode before artifact derivation."""

    request_digest: str
    policy_source_digest: str | None
    candidate_config_digest: str
    actions: tuple[tuple[float, ...], ...]
    observation_digests: tuple[str, ...]
    equity_curve: tuple[float, ...]
    order_events: tuple[OrderEvent, ...]
    terminal_book: BookState
    terminal_order_book: OrderBookState

    def __post_init__(self) -> None:
        require_sha256(
            self.request_digest,
            field="stage_a_episode_result.request_digest",
        )
        policy_source_digest = _optional_sha256(
            self.policy_source_digest,
            field="stage_a_episode_result.policy_source_digest",
        )
        require_sha256(
            self.candidate_config_digest,
            field="stage_a_episode_result.candidate_config_digest",
        )

        if not self.actions:
            raise ValueError("Stage A episode actions must not be empty")
        actions: list[tuple[float, ...]] = []
        for step_index, row in enumerate(self.actions):
            if not row:
                raise ValueError("Stage A episode actions must not be empty")
            actions.append(
                tuple(
                    _finite_float(
                        value,
                        field=f"Stage A episode actions[{step_index}]",
                    )
                    for value in row
                )
            )

        observations = tuple(self.observation_digests)
        if not observations:
            raise ValueError("Stage A episode observations must not be empty")
        if len(observations) != len(actions) + 1:
            raise ValueError("Stage A episode observation closure mismatch")
        for index, digest in enumerate(observations):
            require_sha256(
                digest,
                field=f"stage_a_episode_result.observation_digests[{index}]",
            )

        equity = tuple(
            _finite_float(value, field="Stage A episode equity curve")
            for value in self.equity_curve
        )
        if len(equity) != len(observations):
            raise ValueError("Stage A episode equity closure mismatch")
        if any(value <= 0.0 for value in equity):
            raise ValueError("Stage A episode equity curve must be positive")

        events = tuple(self.order_events)
        if not events:
            raise ValueError("Stage A episode order events must not be empty")
        if any(not isinstance(event, OrderEvent) for event in events):
            raise ValueError(
                "Stage A episode order events must contain OrderEvent values"
            )
        if not isinstance(self.terminal_book, BookState):
            raise ValueError("Stage A episode terminal book must be BookState")
        if not isinstance(self.terminal_order_book, OrderBookState):
            raise ValueError(
                "Stage A episode terminal order book must be OrderBookState"
            )
        terminal_equity = _finite_float(
            self.terminal_book.portfolio_value,
            field="Stage A episode terminal book equity",
        )
        tolerance = max(1e-9, abs(equity[-1]) * 1e-12)
        if not math.isclose(
            terminal_equity,
            equity[-1],
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            raise ValueError("Stage A episode terminal equity mismatch")

        object.__setattr__(self, "policy_source_digest", policy_source_digest)
        object.__setattr__(self, "actions", tuple(actions))
        object.__setattr__(self, "observation_digests", observations)
        object.__setattr__(self, "equity_curve", equity)
        object.__setattr__(self, "order_events", events)

    def validate_against(
        self,
        request: StageAEvaluationCellRequest,
        *,
        expected_policy_source_digest: str | None,
        expected_candidate_config_digest: str,
    ) -> Self:
        """Reconcile the result with the exact request and policy source."""

        if self.request_digest != request.digest:
            raise ValueError("Stage A episode request digest mismatch")
        require_sha256(
            expected_candidate_config_digest,
            field="expected_stage_a_candidate_config_digest",
        )
        if self.candidate_config_digest != expected_candidate_config_digest:
            raise ValueError("Stage A episode candidate config digest mismatch")

        expected_source = _optional_sha256(
            expected_policy_source_digest,
            field="expected_stage_a_policy_source_digest",
        )
        if request.is_baseline:
            if self.policy_source_digest is not None:
                raise ValueError(
                    "Stage A baseline episode must not define a policy source"
                )
            if expected_source is not None:
                raise ValueError(
                    "Stage A baseline request must not expect a policy source"
                )
        else:
            if self.policy_source_digest is None:
                raise ValueError("Stage A policy episode requires a policy source")
            if expected_source is None:
                raise ValueError("Stage A policy request requires a policy source")
        if self.policy_source_digest != expected_source:
            raise ValueError("Stage A episode policy source digest mismatch")

        for event in self.order_events:
            if event.dataset_id != request.dataset_identity:
                raise ValueError("Stage A episode order event dataset mismatch")
            if event.execution_policy_digest != request.execution_identity:
                raise ValueError("Stage A episode order event execution mismatch")
        return self


class StageAEvaluationEpisodeExecutor(Protocol):
    """Execute one exact Stage A request against an optional policy."""

    def execute(
        self,
        request: StageAEvaluationCellRequest,
        *,
        policy: object | None,
        policy_source_digest: str | None,
        candidate_config_digest: str,
    ) -> StageAEvaluationEpisodeResult: ...


__all__ = [
    "StageAEvaluationEpisodeExecutor",
    "StageAEvaluationEpisodeResult",
]
