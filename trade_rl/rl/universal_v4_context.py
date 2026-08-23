"""Artifact-backed policy context for the research-only Causal Alpha V4 lane."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.v4_context import (
    CROSS_MARKET_CORE_NAMES,
    CROSS_MARKET_DERIVATIVE_NAMES,
    GLOBAL_MARKET_CORE_NAMES,
    GLOBAL_MARKET_DERIVATIVE_NAMES,
    V4TargetContext,
)

V4_POLICY_CONTEXT_SCHEMA: Final = "universal_v4_policy_context_v1"
_PROFILE_SCHEMAS: Final = {
    "cross_market_core_v1": (
        CROSS_MARKET_CORE_NAMES,
        GLOBAL_MARKET_CORE_NAMES,
    ),
    "cross_market_derivatives_v1": (
        (*CROSS_MARKET_CORE_NAMES, *CROSS_MARKET_DERIVATIVE_NAMES),
        (*GLOBAL_MARKET_CORE_NAMES, *GLOBAL_MARKET_DERIVATIVE_NAMES),
    ),
}


def _row_matrix(value: object, *, field: str, width: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(1, -1).copy(order="C")
    if array.shape != (1, width) or not np.isfinite(array).all():
        raise ValueError(f"{field} must be a finite (1, {width}) matrix")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class V4PolicyContext:
    local_values: np.ndarray
    local_available: np.ndarray
    local_staleness_hours: np.ndarray
    global_values: np.ndarray
    global_available: np.ndarray
    global_staleness_hours: np.ndarray
    beta: np.ndarray
    beta_available: np.ndarray
    digest: str

    def __post_init__(self) -> None:
        local = np.asarray(self.local_values, dtype=np.float32)
        global_values = np.asarray(self.global_values, dtype=np.float32)
        if local.ndim != 2 or local.shape[0] != 1 or local.shape[1] == 0:
            raise ValueError("V4 local policy context must be one non-empty row")
        if (
            global_values.ndim != 2
            or global_values.shape[0] != 1
            or global_values.shape[1] == 0
        ):
            raise ValueError("V4 global policy context must be one non-empty row")
        local_width = int(local.shape[1])
        global_width = int(global_values.shape[1])
        arrays = {
            "local_values": _row_matrix(
                self.local_values, field="local_values", width=local_width
            ),
            "local_available": _row_matrix(
                self.local_available, field="local_available", width=local_width
            ),
            "local_staleness_hours": _row_matrix(
                self.local_staleness_hours,
                field="local_staleness_hours",
                width=local_width,
            ),
            "global_values": _row_matrix(
                self.global_values, field="global_values", width=global_width
            ),
            "global_available": _row_matrix(
                self.global_available, field="global_available", width=global_width
            ),
            "global_staleness_hours": _row_matrix(
                self.global_staleness_hours,
                field="global_staleness_hours",
                width=global_width,
            ),
            "beta": _row_matrix(self.beta, field="beta", width=1),
            "beta_available": _row_matrix(
                self.beta_available, field="beta_available", width=1
            ),
        }
        if np.any(arrays["local_staleness_hours"] < 0.0) or np.any(
            arrays["global_staleness_hours"] < 0.0
        ):
            raise ValueError("V4 policy context staleness must be non-negative")
        if np.any(
            (arrays["local_available"] != 0.0)
            & (arrays["local_available"] != 1.0)
        ) or np.any(
            (arrays["global_available"] != 0.0)
            & (arrays["global_available"] != 1.0)
        ):
            raise ValueError("V4 policy availability masks must be binary")
        if np.any(
            (arrays["beta_available"] != 0.0)
            & (arrays["beta_available"] != 1.0)
        ):
            raise ValueError("V4 beta availability mask must be binary")
        if not isinstance(self.digest, str) or len(self.digest) != 64:
            raise ValueError("V4 policy context digest must be a SHA-256 digest")
        for field, array in arrays.items():
            object.__setattr__(self, field, array)


@dataclass(frozen=True, slots=True)
class V4ContextProvider:
    contexts: Mapping[str, V4TargetContext]

    def __post_init__(self) -> None:
        contexts = dict(self.contexts)
        if not contexts:
            raise ValueError("V4 context provider requires at least one target context")
        if any(
            not isinstance(symbol, str)
            or not symbol
            or not isinstance(context, V4TargetContext)
            or context.symbol != symbol
            for symbol, context in contexts.items()
        ):
            raise ValueError("V4 context provider symbol/context identity is invalid")
        profiles = {context.profile_name for context in contexts.values()}
        if len(profiles) != 1:
            raise ValueError("V4 context provider profile schema drifted across symbols")
        profile = next(iter(profiles))
        expected = _PROFILE_SCHEMAS.get(profile)
        if expected is None:
            raise ValueError("V4 context provider profile is unsupported")
        expected_local, expected_global = expected
        for context in contexts.values():
            if context.local.feature_names != expected_local:
                raise ValueError("V4 local feature schema/order drifted")
            if context.global_market.feature_names != expected_global:
                raise ValueError("V4 global feature schema/order drifted")
        object.__setattr__(self, "contexts", contexts)

    @property
    def profile_name(self) -> str:
        return next(iter(self.contexts.values())).profile_name

    @property
    def local_width(self) -> int:
        return len(next(iter(self.contexts.values())).local.feature_names)

    @property
    def global_width(self) -> int:
        return len(next(iter(self.contexts.values())).global_market.feature_names)

    @property
    def schema_digest(self) -> str:
        first = next(iter(self.contexts.values()))
        return content_digest(
            {
                "global_feature_names": first.global_market.feature_names,
                "local_feature_names": first.local.feature_names,
                "profile_name": first.profile_name,
                "schema_version": V4_POLICY_CONTEXT_SCHEMA,
            }
        )

    @property
    def digest(self) -> str:
        return content_digest(
            {
                "context_digests": tuple(
                    (symbol, context.digest)
                    for symbol, context in self.contexts.items()
                ),
                "schema_digest": self.schema_digest,
                "schema_version": "universal_v4_context_provider_v1",
            }
        )

    def resolve(self, *, symbol: str, decision_index: int) -> V4PolicyContext:
        if not isinstance(symbol, str) or not symbol:
            raise ValueError("V4 context symbol must be non-empty")
        if isinstance(decision_index, bool) or not isinstance(decision_index, int):
            raise TypeError("V4 decision_index must be an integer")
        try:
            context = self.contexts[symbol]
        except KeyError as error:
            raise ValueError("V4 context provider does not contain routed symbol") from error
        decisions = context.local.decision_indices
        row = int(np.searchsorted(decisions, decision_index, side="left"))
        if row >= len(decisions) or int(decisions[row]) != decision_index:
            raise ValueError("V4 context provider does not contain decision index")
        return V4PolicyContext(
            local_values=context.local.values[row : row + 1],
            local_available=context.local.available[row : row + 1].astype(
                np.float32, copy=False
            ),
            local_staleness_hours=context.local.staleness_hours[row : row + 1],
            global_values=context.global_market.values[row : row + 1],
            global_available=context.global_market.available[row : row + 1].astype(
                np.float32, copy=False
            ),
            global_staleness_hours=context.global_market.staleness_hours[
                row : row + 1
            ],
            beta=context.beta[row : row + 1, None],
            beta_available=context.beta_available[row : row + 1, None].astype(
                np.float32, copy=False
            ),
            digest=context.policy_row_digest(row),
        )


__all__ = [
    "V4_POLICY_CONTEXT_SCHEMA",
    "V4ContextProvider",
    "V4PolicyContext",
]
