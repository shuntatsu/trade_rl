"""Live guardrails and emergency risk-reduction execution.

The policy guardrails are pure calculations. Emergency flattening is deliberately
separate and requires an execution adapter. A CLI invocation without an adapter
fails closed and never claims that positions were flattened.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence

import numpy as np


@dataclass
class GuardrailConfig:
    max_data_age_hours: float = 2.0
    max_daily_loss: float = 0.05
    max_drawdown: float = 0.20
    max_consecutive_losses: int = 12
    max_turnover_z: float = 3.0
    max_abs_weight: float = 0.5


@dataclass
class GuardrailState:
    day_start_value: float = 1.0
    peak_value: float = 1.0
    consecutive_losses: int = 0
    turnover_mean: float = 0.0
    turnover_std: float = 1.0


@dataclass
class GuardrailResult:
    action: str = "proceed"
    scale: float = 1.0
    triggered: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "scale": self.scale,
            "triggered": self.triggered,
            "warnings": self.warnings,
        }


@dataclass(frozen=True)
class PositionSnapshot:
    symbol: str
    quantity: float


@dataclass(frozen=True)
class FlattenExecutionReport:
    success: bool
    idempotency_key: str
    blocked_new_risk: bool
    cancelled_order_ids: tuple[str, ...]
    submitted_order_ids: tuple[str, ...]
    remaining_open_order_ids: tuple[str, ...]
    remaining_positions: Mapping[str, float]
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["cancelled_order_ids"] = list(self.cancelled_order_ids)
        payload["submitted_order_ids"] = list(self.submitted_order_ids)
        payload["remaining_open_order_ids"] = list(self.remaining_open_order_ids)
        payload["remaining_positions"] = dict(self.remaining_positions)
        payload["errors"] = list(self.errors)
        return payload


class EmergencyExecutionAdapter(Protocol):
    """Exchange/platform contract required for a real emergency flatten."""

    def block_new_risk(self, reason: str, idempotency_key: str) -> None: ...

    def cancel_all_orders(self, reason: str, idempotency_key: str) -> Sequence[str]: ...

    def list_open_order_ids(self) -> Sequence[str]: ...

    def list_positions(self) -> Sequence[PositionSnapshot]: ...

    def submit_reduce_only_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        client_order_id: str,
    ) -> str: ...

    def reconcile(self) -> None: ...


def evaluate_guardrails(
    weights: np.ndarray,
    portfolio_value: float,
    turnover: float,
    data_age_hours: float,
    features: Optional[np.ndarray] = None,
    state: Optional[GuardrailState] = None,
    config: Optional[GuardrailConfig] = None,
) -> GuardrailResult:
    cfg = config or GuardrailConfig()
    st = state or GuardrailState()
    res = GuardrailResult()

    if data_age_hours > cfg.max_data_age_hours:
        res.action = "flatten"
        res.triggered.append(f"stale data ({data_age_hours:.1f}h old)")
        res.scale = 0.0
        return res

    if features is not None:
        if np.isnan(features).any():
            res.action = "flatten"
            res.triggered.append("NaN in features")
            res.scale = 0.0
            return res
        if np.all(features == 0):
            res.action = "flatten"
            res.triggered.append("all-zero features (feed likely down)")
            res.scale = 0.0
            return res

    daily_loss = 1.0 - portfolio_value / max(st.day_start_value, 1e-9)
    if daily_loss > cfg.max_daily_loss:
        res.action = "flatten"
        res.triggered.append(f"daily loss {daily_loss:.1%} > {cfg.max_daily_loss:.0%}")
        res.scale = 0.0
        return res

    drawdown = 1.0 - portfolio_value / max(st.peak_value, 1e-9)
    if drawdown > cfg.max_drawdown:
        res.action = "flatten"
        res.triggered.append(f"drawdown {drawdown:.1%} > {cfg.max_drawdown:.0%}")
        res.scale = 0.0
        return res

    if st.consecutive_losses > cfg.max_consecutive_losses:
        res.action = "scale"
        res.scale = min(res.scale, 0.5)
        res.triggered.append(f"{st.consecutive_losses} consecutive losing bars")

    if st.turnover_std > 0:
        turnover_z = (turnover - st.turnover_mean) / st.turnover_std
        if turnover_z > cfg.max_turnover_z:
            res.action = "scale"
            res.scale = min(res.scale, 0.5)
            res.triggered.append(
                f"turnover z-score {turnover_z:.1f} > {cfg.max_turnover_z}"
            )

    if len(weights) > 0 and np.abs(weights).max() > cfg.max_abs_weight:
        res.warnings.append(
            f"weight cap exceeded ({np.abs(weights).max():.2f} > {cfg.max_abs_weight})"
        )
    return res


def apply_guardrails(weights: np.ndarray, result: GuardrailResult) -> np.ndarray:
    if result.action == "flatten":
        return np.zeros_like(weights)
    if result.action == "scale":
        return weights * result.scale
    return weights


def execute_emergency_flatten(
    adapter: EmergencyExecutionAdapter,
    *,
    reason: str,
    idempotency_key: str,
    position_tolerance: float = 1e-9,
    max_reconcile_rounds: int = 3,
) -> FlattenExecutionReport:
    """Block risk, cancel orders, reduce residual positions, and reconcile.

    The function is idempotent when the adapter honors the supplied client order
    IDs. It never reports success unless both open orders and positions are flat.
    """

    if not re.fullmatch(r"[A-Za-z0-9_.:-]{8,128}", idempotency_key):
        raise ValueError("idempotency_key must be 8-128 safe characters")
    if position_tolerance < 0:
        raise ValueError("position_tolerance must be non-negative")
    if max_reconcile_rounds <= 0:
        raise ValueError("max_reconcile_rounds must be positive")

    cancelled: list[str] = []
    submitted: list[str] = []
    errors: list[str] = []
    blocked = False

    try:
        adapter.block_new_risk(reason, idempotency_key)
        blocked = True
        cancelled.extend(adapter.cancel_all_orders(reason, idempotency_key))
        adapter.reconcile()

        remaining_orders = tuple(adapter.list_open_order_ids())
        if remaining_orders:
            return FlattenExecutionReport(
                success=False,
                idempotency_key=idempotency_key,
                blocked_new_risk=blocked,
                cancelled_order_ids=tuple(cancelled),
                submitted_order_ids=tuple(submitted),
                remaining_open_order_ids=remaining_orders,
                remaining_positions=_position_map(adapter.list_positions()),
                errors=("open orders remain after cancellation",),
            )

        for round_index in range(max_reconcile_rounds):
            positions = [
                position
                for position in adapter.list_positions()
                if abs(float(position.quantity)) > position_tolerance
            ]
            if not positions:
                break
            for position in positions:
                side = "sell" if position.quantity > 0 else "buy"
                client_order_id = _client_order_id(
                    idempotency_key, position.symbol, round_index
                )
                order_id = adapter.submit_reduce_only_market_order(
                    symbol=position.symbol,
                    side=side,
                    quantity=abs(float(position.quantity)),
                    client_order_id=client_order_id,
                )
                submitted.append(str(order_id))
            adapter.reconcile()
            round_open_orders = tuple(adapter.list_open_order_ids())
            if round_open_orders:
                return FlattenExecutionReport(
                    success=False,
                    idempotency_key=idempotency_key,
                    blocked_new_risk=blocked,
                    cancelled_order_ids=tuple(cancelled),
                    submitted_order_ids=tuple(submitted),
                    remaining_open_order_ids=round_open_orders,
                    remaining_positions=_position_map(adapter.list_positions()),
                    errors=("reduce-only orders remain open after reconciliation",),
                )

        remaining_orders = tuple(adapter.list_open_order_ids())
        remaining_positions = _position_map(adapter.list_positions())
        nonflat_positions = {
            symbol: quantity
            for symbol, quantity in remaining_positions.items()
            if abs(quantity) > position_tolerance
        }
        success = blocked and not remaining_orders and not nonflat_positions
        if remaining_orders:
            errors.append("open orders remain after flatten execution")
        if nonflat_positions:
            errors.append("positions remain after flatten execution")
        return FlattenExecutionReport(
            success=success,
            idempotency_key=idempotency_key,
            blocked_new_risk=blocked,
            cancelled_order_ids=tuple(cancelled),
            submitted_order_ids=tuple(submitted),
            remaining_open_order_ids=remaining_orders,
            remaining_positions=remaining_positions,
            errors=tuple(errors),
        )
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        try:
            remaining_orders = tuple(adapter.list_open_order_ids())
            remaining_positions = _position_map(adapter.list_positions())
        except Exception as reconciliation_exc:
            remaining_orders = ()
            remaining_positions = {}
            errors.append(
                f"reconciliation failed: {type(reconciliation_exc).__name__}: "
                f"{reconciliation_exc}"
            )
        return FlattenExecutionReport(
            success=False,
            idempotency_key=idempotency_key,
            blocked_new_risk=blocked,
            cancelled_order_ids=tuple(cancelled),
            submitted_order_ids=tuple(submitted),
            remaining_open_order_ids=remaining_orders,
            remaining_positions=remaining_positions,
            errors=tuple(errors),
        )


def _position_map(positions: Sequence[PositionSnapshot]) -> dict[str, float]:
    return {position.symbol: float(position.quantity) for position in positions}


def _client_order_id(idempotency_key: str, symbol: str, round_index: int) -> str:
    safe_symbol = re.sub(r"[^A-Za-z0-9_.-]", "_", symbol)
    value = f"{idempotency_key}:{safe_symbol}:{round_index}"
    return value[:128]


def _load_executor(spec: str) -> EmergencyExecutionAdapter:
    if ":" not in spec:
        raise ValueError("executor must be specified as module:factory")
    module_name, factory_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name)
    executor = factory()
    return executor


def main(
    argv: Optional[List[str]] = None,
    *,
    executor: EmergencyExecutionAdapter | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="Mars Lite live guardrails CLI")
    parser.add_argument(
        "--action",
        choices=["flatten", "scale", "evaluate"],
        default="evaluate",
    )
    parser.add_argument("--scale", type=float, default=0.0)
    parser.add_argument(
        "--reason", default="manual emergency intervention via CLI"
    )
    parser.add_argument(
        "--executor",
        help="Execution adapter factory as module:factory; required for flatten",
    )
    parser.add_argument("--idempotency-key")
    parser.add_argument("--position-tolerance", type=float, default=1e-9)
    parser.add_argument("--max-reconcile-rounds", type=int, default=3)
    parser.add_argument(
        "--output-format", choices=["json", "text"], default="json"
    )
    args = parser.parse_args(argv)

    if args.action == "flatten":
        if not args.idempotency_key:
            payload = {
                "success": False,
                "action": "flatten_not_executed",
                "errors": ["--idempotency-key is required for flatten"],
            }
            _print_payload(payload, args.output_format)
            return 2
        if executor is None and args.executor:
            try:
                executor = _load_executor(args.executor)
            except Exception as exc:
                payload = {
                    "success": False,
                    "action": "flatten_not_executed",
                    "errors": [f"failed to load executor: {type(exc).__name__}: {exc}"],
                }
                _print_payload(payload, args.output_format)
                return 2
        if executor is None:
            payload = {
                "success": False,
                "action": "flatten_not_executed",
                "errors": [
                    "real emergency flatten requires --executor module:factory"
                ],
            }
            _print_payload(payload, args.output_format)
            return 2

        report = execute_emergency_flatten(
            executor,
            reason=args.reason,
            idempotency_key=args.idempotency_key,
            position_tolerance=args.position_tolerance,
            max_reconcile_rounds=args.max_reconcile_rounds,
        )
        _print_payload(report.to_dict(), args.output_format)
        return 0 if report.success else 1

    if args.action == "scale":
        result = GuardrailResult(
            action="scale",
            scale=args.scale,
            triggered=[args.reason],
            warnings=[
                "scale is an advisory guardrail result; execution must be applied by the platform"
            ],
        )
    else:
        result = GuardrailResult(action="proceed", scale=1.0)
    _print_payload(result.to_dict(), args.output_format)
    return 0


def _print_payload(payload: Mapping[str, Any], output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(dict(payload), ensure_ascii=False, indent=2))
        return
    for key, value in payload.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    raise SystemExit(main())
