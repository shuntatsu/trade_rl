"""Canonical deterministic replay and immutable execution-event artifacts."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from trade_rl.artifacts.codec import canonical_json_bytes, to_json_value
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.simulation.accounting import BookState, EconomicTerminationReason
from trade_rl.simulation.orders import OrderBookState, OrderEvent, PendingOrder

REPLAY_EVIDENCE_SCHEMA = "stateful_execution_replay_v1"
EXECUTION_EVENT_ARTIFACT_FILE_NAME = "order-events.json"
EXECUTION_EVENT_ARTIFACT_SCHEMA = "execution_order_event_artifact_v1"
_TERMINAL_BOOK_SCHEMA = "execution_terminal_book_v1"
_TERMINAL_ORDER_BOOK_SCHEMA = "execution_terminal_order_book_v1"


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


@dataclass(frozen=True, slots=True)
class ExecutionEventArtifact:
    """Canonical order-event stream plus its terminal accounting states."""

    dataset_id: str
    execution_policy_digest: str
    order_event_schema: str
    events: tuple[dict[str, object], ...]
    terminal_book: dict[str, object]
    terminal_order_book: dict[str, object]
    schema_version: str = EXECUTION_EVENT_ARTIFACT_SCHEMA

    def __post_init__(self) -> None:
        require_sha256(self.dataset_id, field="event_artifact.dataset_id")
        require_sha256(
            self.execution_policy_digest,
            field="event_artifact.execution_policy_digest",
        )
        if not isinstance(self.order_event_schema, str) or not self.order_event_schema:
            raise ValueError("order_event_schema must be a non-empty string")
        events = tuple(dict(event) for event in self.events)
        if not events:
            raise ValueError("execution event artifact requires at least one event")
        for index, event in enumerate(events):
            if event.get("schema_version") != self.order_event_schema:
                raise ValueError(f"events[{index}] schema identity mismatch")
            if event.get("dataset_id") != self.dataset_id:
                raise ValueError(f"events[{index}] dataset identity mismatch")
            if event.get("execution_policy_digest") != self.execution_policy_digest:
                raise ValueError(f"events[{index}] execution policy identity mismatch")
        terminal_book = dict(self.terminal_book)
        terminal_order_book = dict(self.terminal_order_book)
        if terminal_book.get("schema_version") != _TERMINAL_BOOK_SCHEMA:
            raise ValueError("terminal book schema identity mismatch")
        if terminal_order_book.get("schema_version") != _TERMINAL_ORDER_BOOK_SCHEMA:
            raise ValueError("terminal order book schema identity mismatch")
        if self.schema_version != EXECUTION_EVENT_ARTIFACT_SCHEMA:
            raise ValueError("unsupported execution event artifact schema")
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "terminal_book", terminal_book)
        object.__setattr__(self, "terminal_order_book", terminal_order_book)

    @property
    def order_event_count(self) -> int:
        return len(self.events)

    @property
    def terminal_book_digest(self) -> str:
        return content_digest(self.terminal_book)

    @property
    def terminal_order_book_digest(self) -> str:
        return content_digest(self.terminal_order_book)

    def to_mapping(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "events": self.events,
            "execution_policy_digest": self.execution_policy_digest,
            "order_event_schema": self.order_event_schema,
            "schema_version": self.schema_version,
            "terminal_book": self.terminal_book,
            "terminal_order_book": self.terminal_order_book,
        }

    @property
    def raw_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_mapping()) + b"\n"

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ExecutionEventArtifact:
        required = {
            "dataset_id",
            "events",
            "execution_policy_digest",
            "order_event_schema",
            "schema_version",
            "terminal_book",
            "terminal_order_book",
        }
        if set(value) != required:
            raise ValueError("execution event artifact field closure mismatch")
        raw_events = value["events"]
        raw_terminal_book = value["terminal_book"]
        raw_terminal_order_book = value["terminal_order_book"]
        if not isinstance(raw_events, list) or any(
            not isinstance(event, dict) for event in raw_events
        ):
            raise ValueError("execution event artifact events must be objects")
        if not isinstance(raw_terminal_book, dict):
            raise ValueError("terminal_book must be an object")
        if not isinstance(raw_terminal_order_book, dict):
            raise ValueError("terminal_order_book must be an object")
        dataset_id = value["dataset_id"]
        execution_policy_digest = value["execution_policy_digest"]
        order_event_schema = value["order_event_schema"]
        schema_version = value["schema_version"]
        if not all(
            isinstance(item, str)
            for item in (
                dataset_id,
                execution_policy_digest,
                order_event_schema,
                schema_version,
            )
        ):
            raise ValueError("execution event artifact identities must be strings")
        return cls(
            dataset_id=cast(str, dataset_id),
            execution_policy_digest=cast(str, execution_policy_digest),
            order_event_schema=cast(str, order_event_schema),
            events=tuple(dict(event) for event in raw_events),
            terminal_book=dict(raw_terminal_book),
            terminal_order_book=dict(raw_terminal_order_book),
            schema_version=cast(str, schema_version),
        )


def _finite_vector(values: Sequence[float], *, field: str) -> tuple[float, ...]:
    normalized = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in normalized):
        raise ValueError(f"{field} must contain only finite values")
    return normalized


def _book_payload(book: BookState) -> dict[str, object]:
    reason = book.termination_reason
    if isinstance(reason, EconomicTerminationReason):
        termination_reason: str | None = reason.value
    elif reason is None:
        termination_reason = None
    else:  # pragma: no cover - BookState normalizes this defensively
        termination_reason = str(reason)
    multipliers = book.contract_multipliers
    if multipliers is None:  # pragma: no cover - BookState normalizes this
        raise RuntimeError("book contract multipliers were not normalized")
    return {
        "borrow_cost": float(book.borrow_cost),
        "cash": float(book.cash),
        "contract_multipliers": multipliers.tolist(),
        "fill_count": book.fill_count,
        "funding_pnl": float(book.funding_pnl),
        "insolvent": book.insolvent,
        "maintenance_margin": float(book.maintenance_margin),
        "maintenance_requirement": float(book.maintenance_requirement),
        "margin_deficit": float(book.margin_deficit),
        "margin_used": float(book.margin_used),
        "mark_prices": book.mark_prices.tolist(),
        "max_drawdown": float(book.max_drawdown),
        "peak_value": float(book.peak_value),
        "quantities": book.quantities.tolist(),
        "rebalance_events": book.rebalance_events,
        "returns_history": tuple(float(value) for value in book.returns_history),
        "schema_version": _TERMINAL_BOOK_SCHEMA,
        "termination_reason": termination_reason,
        "total_cost": float(book.total_cost),
        "turnover_total": float(book.turnover_total),
    }


def _pending_order_payload(order: PendingOrder) -> dict[str, object]:
    payload = to_json_value(order)
    if not isinstance(payload, dict):  # pragma: no cover - dataclass conversion
        raise TypeError("pending order did not serialize as an object")
    return cast(dict[str, object], payload)


def _order_book_payload(order_book: OrderBookState) -> dict[str, object]:
    return {
        "active_orders": tuple(
            _pending_order_payload(order) for order in order_book.active_orders
        ),
        "schema_version": _TERMINAL_ORDER_BOOK_SCHEMA,
        "terminal_orders": tuple(
            _pending_order_payload(order) for order in order_book.terminal_orders
        ),
    }


def build_execution_event_artifact(
    *,
    dataset_id: str,
    execution_policy_digest: str,
    order_events: Sequence[OrderEvent],
    terminal_book: BookState,
    terminal_order_book: OrderBookState,
) -> ExecutionEventArtifact:
    """Build one canonical artifact from an actual execution result."""

    events = tuple(event.canonical_payload() for event in order_events)
    if not events:
        raise ValueError("execution event artifact requires at least one event")
    schemas = {event.schema_version for event in order_events}
    if len(schemas) != 1:
        raise ValueError("execution event artifact requires one event schema")
    return ExecutionEventArtifact(
        dataset_id=dataset_id,
        execution_policy_digest=execution_policy_digest,
        order_event_schema=next(iter(schemas)),
        events=events,
        terminal_book=_book_payload(terminal_book),
        terminal_order_book=_order_book_payload(terminal_order_book),
    )


def write_execution_event_artifact(
    path: str | Path,
    artifact: ExecutionEventArtifact,
) -> Path:
    output = Path(path)
    if output.exists():
        raise FileExistsError("execution event artifact already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(artifact.raw_bytes)
    return output


def load_execution_event_artifact_bytes(raw: bytes) -> ExecutionEventArtifact:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("execution event artifact must be valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("execution event artifact must be an object")
    artifact = ExecutionEventArtifact.from_mapping(value)
    if raw != artifact.raw_bytes:
        raise ValueError("execution event artifact must use canonical encoding")
    return artifact


def load_execution_event_artifact(path: str | Path) -> ExecutionEventArtifact:
    return load_execution_event_artifact_bytes(Path(path).read_bytes())


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


__all__ = [
    "EXECUTION_EVENT_ARTIFACT_FILE_NAME",
    "EXECUTION_EVENT_ARTIFACT_SCHEMA",
    "REPLAY_EVIDENCE_SCHEMA",
    "ExecutionEventArtifact",
    "StatefulReplayEvidence",
    "build_execution_event_artifact",
    "build_stateful_replay_evidence",
    "load_execution_event_artifact",
    "load_execution_event_artifact_bytes",
    "write_execution_event_artifact",
]
