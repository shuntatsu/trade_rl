"""Identity-bound serving account state and monotonic decision reconciliation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trade_rl.data.identity import content_and_arrays_digest
from trade_rl.domain.common import require_sha256

SERVING_STATE_SCHEMA = "serving_state_snapshot_v1"


def _vector(value: np.ndarray, *, field: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{field} must be a non-empty finite vector")
    return array


def _array_digest(field: str, value: np.ndarray) -> str:
    array = _vector(value, field=field)
    return content_and_arrays_digest(
        {"field": field, "schema_version": SERVING_STATE_SCHEMA},
        ((field, array),),
    )


@dataclass(frozen=True, slots=True)
class ServingStateSnapshot:
    snapshot_digest: str
    dataset_id: str
    decision_index: int
    portfolio_state_digest: str
    pending_target_digest: str
    observation_digest: str
    schema_version: str = SERVING_STATE_SCHEMA

    def __post_init__(self) -> None:
        for name, value in (
            ("snapshot_digest", self.snapshot_digest),
            ("dataset_id", self.dataset_id),
            ("portfolio_state_digest", self.portfolio_state_digest),
            ("pending_target_digest", self.pending_target_digest),
            ("observation_digest", self.observation_digest),
        ):
            require_sha256(value, field=name)
        if (
            isinstance(self.decision_index, bool)
            or not isinstance(self.decision_index, int)
            or self.decision_index < 0
        ):
            raise ValueError("decision_index must be a non-negative integer")
        if self.schema_version != SERVING_STATE_SCHEMA:
            raise ValueError("unsupported serving state schema")
        if self.snapshot_digest != self.recomputed_digest():
            raise ValueError("serving state snapshot digest mismatch")

    def digest_payload(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "decision_index": self.decision_index,
            "observation_digest": self.observation_digest,
            "pending_target_digest": self.pending_target_digest,
            "portfolio_state_digest": self.portfolio_state_digest,
            "schema_version": self.schema_version,
        }

    def recomputed_digest(self) -> str:
        from trade_rl.artifacts.hashing import content_digest

        return content_digest(self.digest_payload())

    @staticmethod
    def portfolio_digest(value: np.ndarray) -> str:
        return _array_digest("portfolio_state", value)

    @staticmethod
    def pending_digest(value: np.ndarray) -> str:
        return _array_digest("pending_target", value)

    @staticmethod
    def observation_digest_for(value: np.ndarray) -> str:
        return _array_digest("current_flat_observation", value)

    @classmethod
    def create(
        cls,
        *,
        dataset_id: str,
        decision_index: int,
        portfolio_state: np.ndarray,
        pending_target: np.ndarray,
        observation_digest: str,
    ) -> ServingStateSnapshot:
        from trade_rl.artifacts.hashing import content_digest

        payload = {
            "dataset_id": dataset_id,
            "decision_index": decision_index,
            "observation_digest": observation_digest,
            "pending_target_digest": cls.pending_digest(pending_target),
            "portfolio_state_digest": cls.portfolio_digest(portfolio_state),
            "schema_version": SERVING_STATE_SCHEMA,
        }
        return cls(
            snapshot_digest=content_digest(payload),
            dataset_id=dataset_id,
            decision_index=decision_index,
            portfolio_state_digest=str(payload["portfolio_state_digest"]),
            pending_target_digest=str(payload["pending_target_digest"]),
            observation_digest=observation_digest,
        )

    def require_matches(
        self,
        *,
        dataset_id: str,
        decision_index: int,
        portfolio_state: np.ndarray | None = None,
        pending_target: np.ndarray,
        current_flat: np.ndarray | None = None,
    ) -> None:
        if self.dataset_id != dataset_id or self.decision_index != decision_index:
            raise ValueError("serving state dataset or decision index mismatch")
        if self.pending_target_digest != self.pending_digest(pending_target):
            raise ValueError("serving state pending target mismatch")
        if portfolio_state is None:
            raise ValueError("serving portfolio state is required")
        if self.portfolio_state_digest != self.portfolio_digest(portfolio_state):
            raise ValueError("serving portfolio state mismatch")
        if (
            current_flat is not None
            and self.observation_digest != self.observation_digest_for(current_flat)
        ):
            raise ValueError("serving state observation mismatch")


class ServingStateGuard:
    """Reject duplicate or non-monotonic state snapshots per dataset."""

    def __init__(self) -> None:
        self._last_index: dict[str, int] = {}

    def require_fresh(self, snapshot: ServingStateSnapshot) -> None:
        previous = self._last_index.get(snapshot.dataset_id)
        if previous is not None and snapshot.decision_index <= previous:
            raise ValueError("serving state decision index is stale or non-monotonic")

    def accept(self, snapshot: ServingStateSnapshot) -> None:
        self.require_fresh(snapshot)
        self._last_index[snapshot.dataset_id] = snapshot.decision_index

    def require_matches(
        self,
        snapshot: ServingStateSnapshot,
        *,
        dataset_id: str,
        decision_index: int,
        portfolio_state: np.ndarray | None = None,
        pending_target: np.ndarray,
        current_flat: np.ndarray | None = None,
    ) -> None:
        self.require_fresh(snapshot)
        snapshot.require_matches(
            dataset_id=dataset_id,
            decision_index=decision_index,
            portfolio_state=portfolio_state,
            pending_target=pending_target,
            current_flat=current_flat,
        )


__all__ = ["SERVING_STATE_SCHEMA", "ServingStateGuard", "ServingStateSnapshot"]
