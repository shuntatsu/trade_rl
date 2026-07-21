"""Canonical deterministic replay evidence for stateful execution episodes."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.simulation.orders import OrderEvent

REPLAY_EVIDENCE_SCHEMA = "stateful_execution_replay_v1"


@dataclass(frozen=True, slots=True)
class StatefulReplayEvidence:
    """Identity-bound digests for one deterministic execution replay."""

    dataset_id: str
    seed: int
    execution_policy_digest: str
    step_count: int
    order_event_count: int
    action_digest: str
    order_event_digest: str
    equity_curve_digest: str
    observation_trace_digest: str
    schema_version: str = REPLAY_EVIDENCE_SCHEMA

    def __post_init__(self) -> None:
        require_sha256(self.dataset_id, field="replay.dataset_id")
        require_sha256(
            self.execution_policy_digest,
            field="replay.execution_policy_digest",
        )
        for field_name, integer_value in (
            ("seed", self.seed),
            ("step_count", self.step_count),
            ("order_event_count", self.order_event_count),
        ):
            if (
                isinstance(integer_value, bool)
                or not isinstance(integer_value, int)
                or integer_value < 0
            ):
                raise ValueError(f"{field_name} must be a non-negative integer")
        for field_name, digest_value in (
            ("action_digest", self.action_digest),
            ("order_event_digest", self.order_event_digest),
            ("equity_curve_digest", self.equity_curve_digest),
            ("observation_trace_digest", self.observation_trace_digest),
        ):
            require_sha256(digest_value, field=f"replay.{field_name}")
        if self.schema_version != REPLAY_EVIDENCE_SCHEMA:
            raise ValueError("unsupported replay evidence schema")

    def to_mapping(self) -> dict[str, object]:
        return {
            "action_digest": self.action_digest,
            "dataset_id": self.dataset_id,
            "equity_curve_digest": self.equity_curve_digest,
            "execution_policy_digest": self.execution_policy_digest,
            "observation_trace_digest": self.observation_trace_digest,
            "order_event_count": self.order_event_count,
            "order_event_digest": self.order_event_digest,
            "schema_version": self.schema_version,
            "seed": self.seed,
            "step_count": self.step_count,
        }

    @property
    def digest(self) -> str:
        return content_digest(self.to_mapping())


def _finite_vector(values: Sequence[float], *, field: str) -> tuple[float, ...]:
    normalized = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in normalized):
        raise ValueError(f"{field} must contain only finite values")
    return normalized


def build_stateful_replay_evidence(
    *,
    dataset_id: str,
    seed: int,
    execution_policy_digest: str,
    actions: Sequence[Sequence[float]],
    order_events: Sequence[OrderEvent],
    equity_curve: Sequence[float],
    observation_digests: Sequence[str],
) -> StatefulReplayEvidence:
    """Build canonical replay evidence from one completed action trace."""

    require_sha256(dataset_id, field="dataset_id")
    require_sha256(execution_policy_digest, field="execution_policy_digest")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    normalized_actions = tuple(
        _finite_vector(action, field=f"actions[{index}]")
        for index, action in enumerate(actions)
    )
    step_count = len(normalized_actions)
    normalized_equity = _finite_vector(equity_curve, field="equity_curve")
    if len(normalized_equity) != step_count + 1:
        raise ValueError("equity_curve must include initial equity and every step")
    if len(observation_digests) != step_count + 1:
        raise ValueError(
            "observation_digests must include initial observation and every step"
        )
    normalized_observations = tuple(observation_digests)
    for index, digest in enumerate(normalized_observations):
        require_sha256(digest, field=f"observation_digests[{index}]")

    event_payloads: list[dict[str, object]] = []
    for index, event in enumerate(order_events):
        if event.dataset_id != dataset_id:
            raise ValueError(f"order_events[{index}] dataset identity mismatch")
        if event.execution_policy_digest != execution_policy_digest:
            raise ValueError(
                f"order_events[{index}] execution policy identity mismatch"
            )
        event_payloads.append(event.canonical_payload())

    return StatefulReplayEvidence(
        dataset_id=dataset_id,
        seed=seed,
        execution_policy_digest=execution_policy_digest,
        step_count=step_count,
        order_event_count=len(event_payloads),
        action_digest=content_digest(normalized_actions),
        order_event_digest=content_digest(event_payloads),
        equity_curve_digest=content_digest(normalized_equity),
        observation_trace_digest=content_digest(normalized_observations),
    )
