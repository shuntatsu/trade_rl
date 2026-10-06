"""One-shot continuous-account handoff between contiguous allocation OOS windows."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from trade_rl._validation import require_sha256
from trade_rl.artifacts import content_digest
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.allocation_snapshot import snapshot_allocation_account
from trade_rl.evaluation.rl_allocation.env import AllocationTradingEnv
from trade_rl.risk import PreTradeRisk
from trade_rl.simulation import BookState, MarketExecutor
from trade_rl.simulation.orders.model import OrderBookState


@dataclass(frozen=True, slots=True)
class AllocationAccountContinuation:
    """Ephemeral one-shot runtime handoff; not a serialized restart artifact."""

    account_id: str
    dataset_id: str
    execution_policy_digest: str
    risk_digest: str
    symbol_index: int
    next_index: int
    initial_capital: float
    state_digest: str
    _dataset: MarketDataset = field(repr=False)
    _executor: MarketExecutor = field(repr=False)
    _book: BookState = field(repr=False)
    _order_book: OrderBookState = field(repr=False)
    _consumed: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.account_id, str) or not self.account_id.strip():
            raise ValueError("continuation account_id must be nonempty")
        require_sha256(self.dataset_id, field="dataset_id")
        require_sha256(self.execution_policy_digest, field="execution_policy_digest")
        require_sha256(self.risk_digest, field="risk_digest")
        require_sha256(self.state_digest, field="state_digest")
        if (
            isinstance(self.symbol_index, bool)
            or not isinstance(self.symbol_index, int)
            or self.symbol_index < 0
        ):
            raise ValueError("continuation symbol_index must be non-negative")
        if (
            isinstance(self.next_index, bool)
            or not isinstance(self.next_index, int)
            or self.next_index < 0
        ):
            raise ValueError("continuation next_index must be non-negative")
        if (
            isinstance(self.initial_capital, bool)
            or not isinstance(self.initial_capital, (int, float))
            or not math.isfinite(self.initial_capital)
            or self.initial_capital <= 0
        ):
            raise ValueError("continuation initial_capital must be finite and positive")

    @property
    def consumed(self) -> bool:
        return self._consumed

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema": "allocation_account_continuation_v1",
            "account_id": self.account_id,
            "dataset_id": self.dataset_id,
            "execution_policy_digest": self.execution_policy_digest,
            "risk_digest": self.risk_digest,
            "symbol_index": self.symbol_index,
            "next_index": self.next_index,
            "initial_capital": float(self.initial_capital),
            "state_digest": self.state_digest,
        }

    @property
    def digest(self) -> str:
        return content_digest(self.identity_payload())


def _require_live_state(env: AllocationTradingEnv) -> None:
    for name in ("book", "order_book", "index", "risk"):
        if not hasattr(env, name):
            raise RuntimeError("allocation environment has not been initialized")


def allocation_state_digest(env: AllocationTradingEnv) -> str:
    """Return the existing canonical account/order context digest at env.index."""
    if type(env) is not AllocationTradingEnv:
        raise ValueError("state digest requires AllocationTradingEnv")
    _require_live_state(env)
    env.validate_binding()
    snapshot = snapshot_allocation_account(
        env.executor,
        env.book,
        env.order_book,
        account_id=env.account_id,
        pretrade_risk=env.risk,
        symbol_index=env.symbol_index,
        start_index=env.index,
    )
    return snapshot.source_state_digest


def export_allocation_continuation(
    env: AllocationTradingEnv,
) -> AllocationAccountContinuation:
    """Close one horizon without settling it and export exact runtime state once."""
    if type(env) is not AllocationTradingEnv:
        raise ValueError("continuation export requires AllocationTradingEnv")
    _require_live_state(env)
    env.validate_binding()
    if env.book.termination_reason is not None:
        raise ValueError("economically terminated accounts cannot be continued")
    if not env._terminated or env.index != env.stop_index:
        raise RuntimeError("continuation export requires a true horizon terminal")
    if env._continuation_exported:
        raise RuntimeError("allocation horizon continuation was already exported")
    state_digest = allocation_state_digest(env)
    continuation = AllocationAccountContinuation(
        account_id=env.account_id,
        dataset_id=env.dataset.dataset_id,
        execution_policy_digest=env.executor.execution_policy_digest,
        risk_digest=content_digest(env.risk.config),
        symbol_index=env.symbol_index,
        next_index=env.index,
        initial_capital=env.initial_capital,
        state_digest=state_digest,
        _dataset=env.dataset,
        _executor=env.executor,
        _book=env.book.clone(),
        _order_book=env.order_book,
    )
    env._continuation_exported = True
    return continuation


def _validate_resume_target(
    env: AllocationTradingEnv, continuation: AllocationAccountContinuation
) -> PreTradeRisk:
    if continuation.consumed:
        raise RuntimeError("allocation continuation has already been consumed")
    if not env._terminated:
        raise RuntimeError("resume target must not already be active")
    if env.start_index != continuation.next_index:
        raise ValueError("continuous allocation windows must be contiguous")
    if env.dataset is not continuation._dataset:
        raise ValueError("continuous allocation requires the same Dataset instance")
    if env.dataset.dataset_id != continuation.dataset_id:
        raise ValueError("continuous allocation Dataset identity mismatch")
    if env.account_id != continuation.account_id:
        raise ValueError("continuous allocation account identity mismatch")
    if env.symbol_index != continuation.symbol_index:
        raise ValueError("continuous allocation symbol identity mismatch")
    if env.initial_capital != continuation.initial_capital:
        raise ValueError("continuous allocation capital denominator mismatch")
    if env.executor.execution_policy_digest != continuation.execution_policy_digest:
        raise ValueError("continuous allocation execution policy mismatch")
    risk = PreTradeRisk(env.risk_config)
    if content_digest(risk.config) != continuation.risk_digest:
        raise ValueError("continuous allocation risk profile mismatch")
    if continuation._executor.dataset is not env.dataset:
        raise ValueError("continuation executor belongs to another Dataset")
    if (
        continuation._executor.execution_policy_digest
        != env.executor.execution_policy_digest
    ):
        raise ValueError("continuation executor policy differs from resume target")
    return risk


def resume_allocation_continuation(
    env: AllocationTradingEnv,
    continuation: AllocationAccountContinuation,
) -> tuple[np.ndarray, dict[str, object]]:
    """Resume a contiguous fold without resetting book, orders or execution RNG."""
    if type(env) is not AllocationTradingEnv:
        raise ValueError("continuation resume requires AllocationTradingEnv")
    if type(continuation) is not AllocationAccountContinuation:
        raise ValueError("resume requires AllocationAccountContinuation")
    risk = _validate_resume_target(env, continuation)
    book = continuation._book.clone()
    order_book = continuation._order_book
    if book.termination_reason is not None:
        raise ValueError("economically terminated continuation cannot be resumed")
    book.validate_processing_clock(
        dataset_id=env.dataset.dataset_id,
        index=env.start_index,
        require_known=True,
    )

    opening = snapshot_allocation_account(
        continuation._executor,
        book,
        order_book,
        account_id=env.account_id,
        pretrade_risk=risk,
        symbol_index=env.symbol_index,
        start_index=env.start_index,
    )
    if opening.source_state_digest != continuation.state_digest:
        raise ValueError("continuous allocation opening state digest mismatch")

    original_executor, original_risk = env.executor, env.risk
    original_terminated = env._terminated
    existing = {
        name: getattr(env, name)
        for name in ("book", "order_book", "index", "decision")
        if hasattr(env, name)
    }
    try:
        env.executor = continuation._executor
        env.risk = risk
        env.book = book
        env.order_book = order_book
        env.index = env.start_index
        env._terminated = False
        env._continuation_exported = False
        env.validate_binding()
        observation = env._observation()
    except Exception:
        env.executor = original_executor
        env.risk = original_risk
        env._terminated = original_terminated
        for name in ("book", "order_book", "index", "decision"):
            if name in existing:
                setattr(env, name, existing[name])
            elif hasattr(env, name):
                delattr(env, name)
        raise

    object.__setattr__(continuation, "_consumed", True)
    return observation, {
        "symbol": env.dataset.symbols[env.symbol_index],
        "opening_state_digest": opening.source_state_digest,
        "continuation_digest": continuation.digest,
    }


__all__ = [
    "AllocationAccountContinuation",
    "allocation_state_digest",
    "export_allocation_continuation",
    "resume_allocation_continuation",
]
