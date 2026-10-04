"""Immutable source endpoints for the actual supervised fit rows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trade_rl.artifacts.hashing import content_digest


def _timestamp(value: np.datetime64, *, field: str) -> np.datetime64:
    if not isinstance(value, np.datetime64) or np.isnat(value):
        raise ValueError(f"{field} must be a non-NaT timestamp")
    normalized = value.astype("datetime64[ns]")
    if np.isnat(normalized) or normalized.astype(value.dtype) != value:
        raise ValueError(f"{field} must be exactly representable in nanoseconds")
    return normalized


def _vector(value: np.ndarray, *, dtype: str, count: int) -> np.ndarray:
    original = np.asarray(value)
    if original.shape != (count,):
        raise ValueError("trace arrays must match row_symbols")
    if dtype == "datetime64[ns]":
        if original.dtype.kind != "M":
            raise ValueError("trace clocks must contain timestamps")
        for timestamp in original:
            _timestamp(timestamp, field="trace clock")
    normalized = np.asarray(original, dtype=dtype)
    return np.frombuffer(normalized.tobytes(), dtype=normalized.dtype)


@dataclass(frozen=True, slots=True)
class ForecastTrainingTrace:
    """Recorded endpoints, not an assumption that label-end equals publication."""

    row_symbols: tuple[str, ...]
    start_times: np.ndarray
    end_times: np.ndarray
    start_available_at: np.ndarray
    end_available_at: np.ndarray
    start_close: np.ndarray
    end_close: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.row_symbols, tuple) or not self.row_symbols:
            raise ValueError("row_symbols must be a non-empty tuple")
        if any(not isinstance(s, str) or not s.strip() for s in self.row_symbols):
            raise ValueError("row_symbols must contain non-empty symbols")
        for field in (
            "start_times",
            "end_times",
            "start_available_at",
            "end_available_at",
            "start_close",
            "end_close",
        ):
            dtype = "float64" if field.endswith("close") else "datetime64[ns]"
            object.__setattr__(
                self,
                field,
                _vector(getattr(self, field), dtype=dtype, count=len(self.row_symbols)),
            )
        if np.any(self.end_times <= self.start_times):
            raise ValueError("label endpoints must follow feature rows")
        if np.any(self.start_available_at < self.start_times) or np.any(
            self.end_available_at < self.end_times
        ):
            raise ValueError("source publication cannot precede its event")
        for prices in (self.start_close, self.end_close):
            if not np.isfinite(prices).all() or np.any(prices <= 0):
                raise ValueError("trace endpoint prices must be finite and positive")

    @property
    def label_available_times(self) -> np.ndarray:
        values = np.maximum(self.start_available_at, self.end_available_at)
        return np.frombuffer(values.tobytes(), dtype=values.dtype)

    def payload(self) -> dict[str, object]:
        return {
            "schema": "forecast_training_trace_v1",
            "row_symbols": list(self.row_symbols),
            **{
                field: getattr(self, field).astype(np.int64).tolist()
                for field in (
                    "start_times",
                    "end_times",
                    "start_available_at",
                    "end_available_at",
                )
            },
            "start_close": self.start_close.tolist(),
            "end_close": self.end_close.tolist(),
        }

    @property
    def digest(self) -> str:
        return content_digest(self.payload())
