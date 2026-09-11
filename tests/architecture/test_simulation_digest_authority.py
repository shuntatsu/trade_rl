from __future__ import annotations

import pytest

from trade_rl.simulation import liquidity
from trade_rl.simulation.orders import model, reconciliation


def _reject(value: str, *, field: str) -> str:
    raise ValueError(f"{field} rejected by canonical SHA authority")


def test_order_model_digest_adapter_delegates_and_preserves_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    def accept(value: str, *, field: str) -> str:
        calls.append((value, field))
        return value

    monkeypatch.setattr(model, "require_sha256", accept)
    model._validate_digest("execution_policy_digest", "a" * 64)
    assert calls == [("a" * 64, "execution_policy_digest")]

    monkeypatch.setattr(model, "require_sha256", _reject)
    with pytest.raises(model.OrderDomainError, match="canonical SHA authority"):
        model._validate_digest("execution_policy_digest", "a" * 64)


def test_reconciliation_digest_adapter_delegates_and_preserves_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    def accept(value: str, *, field: str) -> str:
        calls.append((value, field))
        return value

    monkeypatch.setattr(reconciliation, "require_sha256", accept)
    reconciliation._validate_digest("a" * 64)
    assert calls == [("a" * 64, "execution_policy_digest")]

    monkeypatch.setattr(reconciliation, "require_sha256", _reject)
    with pytest.raises(
        reconciliation.OrderReconciliationError,
        match="canonical SHA authority",
    ):
        reconciliation._validate_digest("a" * 64)


def test_liquidity_digest_adapter_delegates_and_preserves_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    def accept(value: str, *, field: str) -> str:
        calls.append((value, field))
        return value

    monkeypatch.setattr(liquidity, "require_sha256", accept)
    liquidity._validate_digest("order_id", "a" * 64)
    assert calls == [("a" * 64, "order_id")]

    monkeypatch.setattr(liquidity, "require_sha256", _reject)
    with pytest.raises(liquidity.LiquidityAllocationError, match="canonical SHA authority"):
        liquidity._validate_digest("order_id", "a" * 64)
