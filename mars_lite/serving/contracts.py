"""Typed contracts for authenticated stateful signal inference."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping, Sequence

import numpy as np

OrderSide = Literal["buy", "sell"]
InferenceStatus = Literal["ok", "no_signal", "rejected"]


@dataclass(frozen=True)
class PendingOrderInput:
    symbol: str
    side: OrderSide
    notional: float
    reduce_only: bool = False

    def validate(self, allowed_symbols: set[str]) -> None:
        if self.symbol not in allowed_symbols:
            raise ValueError(f"pending order symbol not in bundle: {self.symbol}")
        if self.side not in ("buy", "sell"):
            raise ValueError(f"invalid pending order side: {self.side}")
        if not math.isfinite(self.notional) or self.notional <= 0:
            raise ValueError("pending order notional must be finite and positive")
        if not isinstance(self.reduce_only, bool):
            raise ValueError("pending order reduce_only must be a boolean")


@dataclass(frozen=True)
class InferenceState:
    current_weights: Mapping[str, float]
    portfolio_value: float
    day_start_value: float
    peak_value: float
    consecutive_losses: int
    turnover_mean: float
    turnover_std: float
    pending_orders: tuple[PendingOrderInput, ...] = field(default_factory=tuple)
    disagreement: float = 0.0
    vol_scale: float | None = None
    dd_scale: float | None = None
    disagreement_scale: float | None = None
    est_port_vol: float | None = None

    def validate(
        self,
        symbols: Sequence[str],
        *,
        require_observation_risk_state: bool = False,
    ) -> None:
        ordered_symbols = tuple(symbols)
        if not ordered_symbols or len(set(ordered_symbols)) != len(ordered_symbols):
            raise ValueError("bundle symbols must be unique and non-empty")
        if set(self.current_weights) != set(ordered_symbols):
            raise ValueError("current_weights must exactly match bundle symbols")
        weights = np.asarray(
            [self.current_weights[symbol] for symbol in ordered_symbols],
            dtype=np.float64,
        )
        if not np.isfinite(weights).all():
            raise ValueError("current_weights contain non-finite values")

        scalar_values = {
            "portfolio_value": self.portfolio_value,
            "day_start_value": self.day_start_value,
            "peak_value": self.peak_value,
            "turnover_mean": self.turnover_mean,
            "turnover_std": self.turnover_std,
            "disagreement": self.disagreement,
        }
        risk_state_values = {
            "vol_scale": self.vol_scale,
            "dd_scale": self.dd_scale,
            "disagreement_scale": self.disagreement_scale,
            "est_port_vol": self.est_port_vol,
        }
        if require_observation_risk_state:
            for name, value in risk_state_values.items():
                if value is None:
                    raise ValueError(f"{name} is required by the observation schema")
        for name, value in scalar_values.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        for name in ("vol_scale", "dd_scale", "disagreement_scale"):
            value = risk_state_values[name]
            if value is not None and (
                not math.isfinite(value) or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{name} must be finite and between 0 and 1")
        if self.est_port_vol is not None and (
            not math.isfinite(self.est_port_vol) or self.est_port_vol < 0
        ):
            raise ValueError("est_port_vol must be finite and non-negative")
        if self.portfolio_value <= 0:
            raise ValueError("portfolio_value must be positive")
        if self.day_start_value <= 0:
            raise ValueError("day_start_value must be positive")
        if self.peak_value < max(self.portfolio_value, self.day_start_value):
            raise ValueError("peak_value must be at least current and day-start value")
        if (
            isinstance(self.consecutive_losses, bool)
            or not isinstance(self.consecutive_losses, int)
            or self.consecutive_losses < 0
        ):
            raise ValueError("consecutive_losses must be a non-negative integer")
        if self.turnover_mean < 0:
            raise ValueError("turnover_mean must be non-negative")
        if self.turnover_std < 0:
            raise ValueError("turnover_std must be non-negative")
        if not 0.0 <= self.disagreement <= 1.0:
            raise ValueError("disagreement must be between 0 and 1")
        allowed = set(ordered_symbols)
        for order in self.pending_orders:
            order.validate(allowed)

    def weights_array(
        self,
        symbols: Sequence[str],
        *,
        require_observation_risk_state: bool = False,
    ) -> np.ndarray:
        self.validate(
            symbols,
            require_observation_risk_state=require_observation_risk_state,
        )
        return np.asarray(
            [self.current_weights[symbol] for symbol in symbols], dtype=np.float64
        )


@dataclass(frozen=True)
class InferenceRequest:
    request_id: str
    market_snapshot_id: str
    state: InferenceState
    idempotency_key: str | None = None

    def validate(
        self,
        symbols: Sequence[str],
        *,
        require_observation_risk_state: bool = False,
    ) -> None:
        if not self.request_id.strip():
            raise ValueError("request_id is required")
        if not self.market_snapshot_id.strip():
            raise ValueError("market_snapshot_id is required")
        if self.idempotency_key is not None and not self.idempotency_key.strip():
            raise ValueError("idempotency_key must be non-empty when supplied")
        self.state.validate(
            symbols,
            require_observation_risk_state=require_observation_risk_state,
        )

    def payload_hash(self) -> str:
        payload = asdict(self)
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class InferenceResponse:
    status: InferenceStatus
    request_id: str
    market_snapshot_id: str
    model_version: str
    bundle_digest: str
    target_weights: Mapping[str, float] | None
    reasons: tuple[str, ...] = field(default_factory=tuple)
    guardrail: Mapping[str, Any] = field(default_factory=dict)
    pre_trade_risk: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
