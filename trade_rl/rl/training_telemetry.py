"""Sample rollout transitions without coupling storage to Studio."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np

from trade_rl.telemetry.training import (
    TelemetryEventType,
    TrainingTelemetryRecord,
    TrainingTelemetryWriter,
)

_DEFAULT_SAMPLE_EVERY = 32
_DEFAULT_POSITION_THRESHOLD = 0.02


class _BookLike(Protocol):
    @property
    def weights(self) -> np.ndarray: ...

    @property
    def portfolio_value(self) -> float: ...

    mark_prices: np.ndarray


def _vector(
    value: object,
    *,
    fallback: tuple[float, ...] = (),
) -> tuple[float, ...]:
    if value is None:
        return fallback
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return fallback
    if not np.isfinite(array).all():
        return fallback
    return tuple(float(item) for item in array)


def _number(value: object, *, fallback: float | None = None) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        return fallback
    resolved = float(value)
    return resolved if np.isfinite(resolved) else fallback


def _risk_reasons(info: dict[str, object]) -> tuple[str, ...]:
    explicit = info.get("telemetry_risk_reasons")
    if isinstance(explicit, (tuple, list)):
        return tuple(str(item) for item in explicit if str(item))
    risk = info.get("hybrid_risk")
    reasons = getattr(risk, "reasons", ())
    return tuple(str(getattr(item, "value", item)) for item in reasons if str(item))


def _execution_book(info: dict[str, object], name: str) -> _BookLike | None:
    execution = info.get(name)
    book = getattr(execution, "book", None)
    return cast(_BookLike, book) if book is not None else None


def _market_values(
    info: dict[str, object],
    *,
    previous_close: float | None,
) -> tuple[
    str,
    int | None,
    str | None,
    float | None,
    float | None,
    float | None,
    float | None,
]:
    symbol = str(info.get("telemetry_symbol") or "ASSET-0")
    raw_index = info.get("telemetry_market_index")
    market_index = (
        int(raw_index)
        if isinstance(raw_index, (int, np.integer)) and not isinstance(raw_index, bool)
        else None
    )
    raw_time = info.get("telemetry_market_time")
    market_time = str(raw_time) if raw_time is not None and str(raw_time) else None
    values = _vector(info.get("telemetry_ohlc"))
    if len(values) == 4:
        return (
            symbol,
            market_index,
            market_time,
            values[0],
            values[1],
            values[2],
            values[3],
        )
    book = _execution_book(info, "hybrid_execution")
    marks = _vector(getattr(book, "mark_prices", None)) if book is not None else ()
    close = marks[0] if marks else None
    open_value = previous_close if previous_close is not None else close
    if open_value is None or close is None:
        return symbol, market_index, market_time, None, None, None, None
    return (
        symbol,
        market_index,
        market_time,
        open_value,
        max(open_value, close),
        min(open_value, close),
        close,
    )


class TrainingTelemetrySampler:
    """Reduce vector rollout data into append-only visualization records."""

    def __init__(
        self,
        path: Path,
        *,
        seed: int,
        sample_every: int = _DEFAULT_SAMPLE_EVERY,
        position_threshold: float = _DEFAULT_POSITION_THRESHOLD,
    ) -> None:
        if isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be non-negative")
        if isinstance(sample_every, bool) or sample_every <= 0:
            raise ValueError("sample_every must be positive")
        if not np.isfinite(position_threshold) or position_threshold < 0.0:
            raise ValueError("position_threshold must be finite and non-negative")
        self.seed = int(seed)
        self.sample_every = int(sample_every)
        self.position_threshold = float(position_threshold)
        self.writer = TrainingTelemetryWriter(path)
        self.sequence = 0
        self.disabled = False
        self._previous_weights: dict[int, tuple[float, ...]] = {}
        self._previous_close: dict[int, float] = {}

    def _weights(
        self,
        info: dict[str, object],
        environment_id: int,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        fallback_before = self._previous_weights.get(environment_id, ())
        before = _vector(
            info.get("telemetry_weights_before"),
            fallback=fallback_before,
        )
        book = _execution_book(info, "hybrid_execution")
        book_weights = getattr(book, "weights", None) if book is not None else None
        after = _vector(
            info.get("telemetry_weights_after"),
            fallback=_vector(book_weights, fallback=before),
        )
        if not before or len(before) != len(after):
            before = tuple(0.0 for _ in after)
        return before, after

    def _event_type(
        self,
        *,
        info: dict[str, object],
        environment_step: int,
        weights_before: tuple[float, ...],
        weights_after: tuple[float, ...],
        done: bool,
        reasons: tuple[str, ...],
    ) -> TelemetryEventType | None:
        if done or bool(info.get("hybrid_terminated")):
            return "episode_end"
        if reasons or bool(info.get("emergency_deleverage")):
            return "risk"
        position_delta = sum(
            abs(after - before)
            for before, after in zip(
                weights_before,
                weights_after,
                strict=True,
            )
        )
        if position_delta >= self.position_threshold:
            return "position"
        if environment_step % self.sample_every == 0:
            return "rollout"
        return None

    def consume(
        self,
        *,
        global_step: int,
        actions: object,
        rewards: object,
        dones: object,
        infos: object,
    ) -> int:
        if self.disabled:
            return 0
        try:
            action_rows = np.asarray(actions, dtype=np.float64)
            if action_rows.ndim == 1:
                action_rows = action_rows.reshape(1, -1)
            reward_rows = np.asarray(rewards, dtype=np.float64).reshape(-1)
            done_rows = np.asarray(dones, dtype=np.bool_).reshape(-1)
            if not isinstance(infos, (tuple, list)):
                raise ValueError("infos must be a vector sequence")
            emitted = 0
            for environment_id, raw_info in enumerate(infos):
                if not isinstance(raw_info, dict):
                    continue
                info = cast(dict[str, object], raw_info)
                raw_step = info.get("decision_step_index", global_step)
                environment_step = (
                    int(raw_step)
                    if isinstance(raw_step, (int, np.integer))
                    and not isinstance(raw_step, bool)
                    else int(global_step)
                )
                weights_before, weights_after = self._weights(
                    info,
                    environment_id,
                )
                reasons = _risk_reasons(info)
                done = (
                    bool(done_rows[environment_id])
                    if environment_id < done_rows.size
                    else False
                )
                event_type = self._event_type(
                    info=info,
                    environment_step=environment_step,
                    weights_before=weights_before,
                    weights_after=weights_after,
                    done=done,
                    reasons=reasons,
                )
                self._previous_weights[environment_id] = weights_after
                if event_type is None:
                    continue
                market = _market_values(
                    info,
                    previous_close=self._previous_close.get(environment_id),
                )
                symbol, market_index, market_time, open_value, high, low, close = market
                if close is not None:
                    self._previous_close[environment_id] = close
                hybrid_book = _execution_book(info, "hybrid_execution")
                shadow_book = _execution_book(info, "shadow_execution")
                hybrid_fallback = (
                    float(hybrid_book.portfolio_value)
                    if hybrid_book is not None
                    else None
                )
                shadow_fallback = (
                    float(shadow_book.portfolio_value)
                    if shadow_book is not None
                    else None
                )
                portfolio_value = _number(
                    info.get("portfolio_value_after"),
                    fallback=hybrid_fallback,
                )
                baseline_value = _number(
                    info.get("baseline_portfolio_value_after"),
                    fallback=shadow_fallback,
                )
                self.sequence += 1
                action = (
                    tuple(
                        float(item) for item in action_rows[environment_id].reshape(-1)
                    )
                    if environment_id < action_rows.shape[0]
                    else ()
                )
                reward_fallback = (
                    float(reward_rows[environment_id])
                    if environment_id < reward_rows.size
                    else None
                )
                self.writer.append(
                    TrainingTelemetryRecord(
                        sequence=self.sequence,
                        recorded_at=datetime.now(UTC).isoformat(),
                        global_step=int(global_step),
                        environment_step=environment_step,
                        seed=self.seed,
                        environment_id=environment_id,
                        event_type=event_type,
                        market_index=market_index,
                        market_time=market_time,
                        symbol=symbol,
                        open=open_value,
                        high=high,
                        low=low,
                        close=close,
                        action=action,
                        executed_target=_vector(
                            info.get("executed_target"),
                            fallback=weights_after,
                        ),
                        weights_before=weights_before,
                        weights_after=weights_after,
                        portfolio_value=portfolio_value,
                        baseline_portfolio_value=baseline_value,
                        reward=_number(
                            info.get("reward_total_scaled"),
                            fallback=reward_fallback,
                        ),
                        drawdown=_number(info.get("drawdown_after")),
                        interval_cost=_number(info.get("interval_cost")),
                        interval_return=_number(info.get("interval_net_return")),
                        risk_reasons=reasons,
                        emergency_deleverage=bool(info.get("emergency_deleverage")),
                        terminated=bool(info.get("hybrid_terminated")) or done,
                        truncated=bool(info.get("TimeLimit.truncated")),
                    )
                )
                emitted += 1
            return emitted
        except Exception:
            self.close()
            return 0

    def flush(self) -> None:
        if not self.disabled:
            self.writer.flush()

    def close(self) -> None:
        if self.disabled:
            return
        try:
            self.writer.close()
        finally:
            self.disabled = True


def build_training_telemetry_callback(
    *,
    path: Path,
    seed: int,
    sample_every: int = _DEFAULT_SAMPLE_EVERY,
) -> Any:
    """Build an SB3 callback lazily while keeping the sampler testable."""

    from stable_baselines3.common.callbacks import BaseCallback

    sampler = TrainingTelemetrySampler(
        path,
        seed=seed,
        sample_every=sample_every,
    )

    class TrainingTelemetryCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__(verbose=0)

        def _on_step(self) -> bool:
            sampler.consume(
                global_step=int(self.model.num_timesteps),
                actions=self.locals.get("actions", ()),
                rewards=self.locals.get("rewards", ()),
                dones=self.locals.get("dones", ()),
                infos=self.locals.get("infos", ()),
            )
            return True

        def _on_rollout_end(self) -> None:
            sampler.flush()

        def _on_training_end(self) -> None:
            sampler.close()

    return TrainingTelemetryCallback()


__all__ = [
    "TrainingTelemetrySampler",
    "build_training_telemetry_callback",
]
