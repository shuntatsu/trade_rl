from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_REQUIRED_FIELDS = {
    "max_leverage",
    "max_single_weight",
    "max_net_exposure",
    "max_worst_case_notional",
    "min_order_notional",
    "symbol_liquidity_caps",
    "forbidden_symbols",
}


@dataclass(frozen=True)
class ReleaseRiskPolicy:
    """Complete pre-trade limits required for a release-capable bundle."""

    max_leverage: float
    max_single_weight: float
    max_net_exposure: float
    max_worst_case_notional: float
    min_order_notional: float
    symbol_liquidity_caps: dict[str, float]
    forbidden_symbols: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return the validated policy for immutable bundle serialization."""

        return asdict(self)


def _positive_finite(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and positive")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and positive") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _load_object(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid release risk JSON: {source}") from exc
    if not isinstance(payload, dict):
        raise ValueError("release risk document must be a JSON object")
    return payload


def load_release_risk_policy(
    path: str | Path,
    *,
    symbols: tuple[str, ...],
) -> ReleaseRiskPolicy:
    """Load and strictly validate a release-only pre-trade risk policy."""

    ordered_symbols = tuple(symbols)
    if not ordered_symbols or len(set(ordered_symbols)) != len(ordered_symbols):
        raise ValueError("symbols must be unique and non-empty")

    payload = _load_object(path)
    missing = sorted(_REQUIRED_FIELDS - payload.keys())
    if missing:
        raise ValueError(f"missing release risk fields: {', '.join(missing)}")
    unknown = sorted(payload.keys() - _REQUIRED_FIELDS)
    if unknown:
        raise ValueError(f"unknown release risk fields: {', '.join(unknown)}")

    max_leverage = _positive_finite("max_leverage", payload["max_leverage"])
    max_single_weight = _positive_finite(
        "max_single_weight", payload["max_single_weight"]
    )
    max_net_exposure = _positive_finite(
        "max_net_exposure", payload["max_net_exposure"]
    )
    if max_single_weight > 1.0:
        raise ValueError("max_single_weight must be <= 1.0")
    if max_single_weight > max_leverage:
        raise ValueError("max_single_weight must be <= max_leverage")
    if max_net_exposure > max_leverage:
        raise ValueError("max_net_exposure must be <= max_leverage")

    raw_caps = payload["symbol_liquidity_caps"]
    if not isinstance(raw_caps, dict):
        raise ValueError("symbol_liquidity_caps must be an object")
    if not all(isinstance(key, str) for key in raw_caps):
        raise ValueError("symbol_liquidity_caps keys must be strings")
    symbol_set = set(ordered_symbols)
    missing_caps = sorted(symbol_set - set(raw_caps))
    if missing_caps:
        raise ValueError(f"missing liquidity caps for: {', '.join(missing_caps)}")
    unknown_caps = sorted(set(raw_caps) - symbol_set)
    if unknown_caps:
        raise ValueError(f"unknown liquidity caps for: {', '.join(unknown_caps)}")
    caps = {
        symbol: _positive_finite(f"liquidity cap {symbol}", raw_caps[symbol])
        for symbol in ordered_symbols
    }

    raw_forbidden = payload["forbidden_symbols"]
    if not isinstance(raw_forbidden, list) or not all(
        isinstance(item, str) for item in raw_forbidden
    ):
        raise ValueError("forbidden_symbols must be a list of strings")
    if len(set(raw_forbidden)) != len(raw_forbidden):
        raise ValueError("forbidden_symbols must not contain duplicates")
    unknown_forbidden = sorted(set(raw_forbidden) - symbol_set)
    if unknown_forbidden:
        raise ValueError(
            f"unknown forbidden symbols: {', '.join(unknown_forbidden)}"
        )

    max_worst_case_notional = _positive_finite(
        "max_worst_case_notional", payload["max_worst_case_notional"]
    )
    min_order_notional = _positive_finite(
        "min_order_notional", payload["min_order_notional"]
    )
    if min_order_notional > min(caps.values()):
        raise ValueError("min_order_notional must not exceed any liquidity cap")

    return ReleaseRiskPolicy(
        max_leverage=max_leverage,
        max_single_weight=max_single_weight,
        max_net_exposure=max_net_exposure,
        max_worst_case_notional=max_worst_case_notional,
        min_order_notional=min_order_notional,
        symbol_liquidity_caps=caps,
        forbidden_symbols=tuple(raw_forbidden),
    )
