"""Authenticated read-only online serving application."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping, Protocol, Sequence, cast

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from mars_lite.server.auth import bearer_dependency
from mars_lite.serving.contracts import (
    InferenceRequest,
    InferenceState,
    OrderSide,
    PendingOrderInput,
)
from mars_lite.serving.runtime import FeatureSnapshot, ServingRuntime


class FeatureProvider(Protocol):
    def get_snapshot(self) -> FeatureSnapshot: ...


class RuntimeLike(Protocol):
    def refresh(self) -> bool: ...

    def readiness(self): ...

    def infer(self, request: InferenceRequest, snapshot: FeatureSnapshot): ...


def _parse_pending_order(value: Mapping[str, Any]) -> PendingOrderInput:
    try:
        side = value["side"]
        if side not in ("buy", "sell"):
            raise ValueError("invalid pending order side")
        reduce_only = value.get("reduce_only", False)
        if not isinstance(reduce_only, bool):
            raise ValueError("reduce_only must be a boolean")
        return PendingOrderInput(
            symbol=str(value["symbol"]),
            side=cast(OrderSide, side),
            notional=float(value["notional"]),
            reduce_only=reduce_only,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid pending order payload") from exc


def parse_inference_request(
    payload: Mapping[str, Any], *, resolved_market_snapshot_id: str | None = None
) -> InferenceRequest:
    try:
        state_payload = payload["state"]
        if not isinstance(state_payload, Mapping):
            raise TypeError("state must be an object")
        raw_weights = state_payload["current_weights"]
        if not isinstance(raw_weights, Mapping):
            raise TypeError("current_weights must be an object")
        pending_payload = state_payload.get("pending_orders", ())
        if not isinstance(pending_payload, Sequence) or isinstance(
            pending_payload, (str, bytes)
        ):
            raise TypeError("pending_orders must be an array")
        pending = tuple(
            _parse_pending_order(item)
            for item in pending_payload
            if isinstance(item, Mapping)
        )
        if len(pending) != len(pending_payload):
            raise ValueError("each pending order must be an object")
        state = InferenceState(
            current_weights={
                str(symbol): float(weight) for symbol, weight in raw_weights.items()
            },
            portfolio_value=float(state_payload["portfolio_value"]),
            day_start_value=float(state_payload["day_start_value"]),
            peak_value=float(state_payload["peak_value"]),
            consecutive_losses=int(state_payload["consecutive_losses"]),
            turnover_mean=float(state_payload["turnover_mean"]),
            turnover_std=float(state_payload["turnover_std"]),
            pending_orders=pending,
            disagreement=float(state_payload.get("disagreement", 0.0)),
            vol_scale=_optional_float(state_payload.get("vol_scale")),
            dd_scale=_optional_float(state_payload.get("dd_scale")),
            disagreement_scale=_optional_float(state_payload.get("disagreement_scale")),
            est_port_vol=_optional_float(state_payload.get("est_port_vol")),
        )
        supplied_snapshot_id = payload.get("market_snapshot_id")
        if resolved_market_snapshot_id is None:
            if supplied_snapshot_id is None:
                raise ValueError("market_snapshot_id is required")
            market_snapshot_id = str(supplied_snapshot_id)
        else:
            if supplied_snapshot_id not in (
                None,
                "latest",
                resolved_market_snapshot_id,
            ):
                raise ValueError("market_snapshot_id does not match serving snapshot")
            market_snapshot_id = resolved_market_snapshot_id
        return InferenceRequest(
            request_id=str(payload["request_id"]),
            market_snapshot_id=market_snapshot_id,
            state=state,
            idempotency_key=(
                str(payload["idempotency_key"])
                if payload.get("idempotency_key") is not None
                else None
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid inference request: {exc}") from exc


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def create_app(
    *,
    runtime: RuntimeLike,
    feature_provider: FeatureProvider,
    auth_token: str,
    allowed_origins: Sequence[str] = (),
) -> FastAPI:
    """Create the read-only serving app; no control-plane routes are registered."""
    app = FastAPI(
        title="Trade RL Serving Plane",
        description="Authenticated read-only portfolio signal serving",
        version="1.0.0",
    )
    origins = tuple(origin for origin in allowed_origins if origin and origin != "*")
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(origins),
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type"],
        )
    require_token = bearer_dependency(auth_token)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready() -> JSONResponse:
        runtime.refresh()
        readiness = runtime.readiness()
        status_code = 200 if readiness.status in {"ready", "degraded"} else 503
        return JSONResponse(asdict(readiness), status_code=status_code)

    @app.post("/api/signal/latest", dependencies=[Depends(require_token)])
    async def signal_latest(payload: dict[str, Any]) -> JSONResponse:
        runtime.refresh()
        try:
            snapshot = feature_provider.get_snapshot()
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(
                status_code=503, detail=f"feature snapshot unavailable: {exc}"
            ) from exc
        try:
            request = parse_inference_request(
                payload, resolved_market_snapshot_id=snapshot.snapshot_id
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        response = runtime.infer(request, snapshot)
        status_code = 503 if response.status == "no_signal" else 200
        return JSONResponse(response.to_dict(), status_code=status_code)

    return app


__all__ = ["FeatureProvider", "ServingRuntime", "create_app", "parse_inference_request"]
