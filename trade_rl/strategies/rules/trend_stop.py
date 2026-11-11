"""Causal Donchian breakout with ATR stops as one shared-portfolio policy.

The rule lives entirely in the decision layer. Orders, fills, costs, funding,
hard risk and accounting stay with the existing shared portfolio session: this
policy only proposes target weights (and exact-quantity HOLD) once per native bar.

Decision at row ``t`` uses only completed bars ``<= t``:

* entry: ``close[t]`` beyond the prior ``entry_bars`` extremum (row ``t`` excluded);
* signal exit: ``close[t]`` beyond the prior ``exit_bars`` opposite extremum;
* stop: an initial stop ``initial_stop_atr * ATR`` from the observed entry price
  and a trailing stop ``trailing_stop_atr * ATR`` from the best extreme since
  entry, both with the ATR of the entry signal bar (frozen at entry).

Stop model (``bar_extreme_touch_next_open_v1``): the stop is *triggered* when the
completed bar's low (long) or high (short) touches the stop level, evaluated with
the bar's favourable extreme applied first (adverse ordering). The exit is then a
normal order that fills at the next bar's open under the engine's cost model, so a
gap through the stop is taken in full and no fill is invented at the stop price.
The runner separately reports any fill that was *better* than the stop level, so a
stop-price haircut can be applied (see ``trade_rl.evaluation.trend_stop_study``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.strategies.portfolio import (
    PortfolioDecisionSnapshot,
    PortfolioTargetProposal,
)

_TOLERANCE = 1e-12


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


@dataclass(frozen=True, slots=True)
class TrendStopConfig:
    """Fixed parameters of one trend-with-stops variant."""

    entry_bars: int
    exit_bars: int
    atr_period: int = 14
    initial_stop_atr: float = 2.0
    trailing_stop_atr: float = 3.0
    risk_per_trade: float = 0.005
    max_symbol_weight: float = 0.25
    gross_cap: float = 0.95
    allow_short: bool = True

    def __post_init__(self) -> None:
        entry = _positive_int(self.entry_bars, name="entry_bars")
        exit_ = _positive_int(self.exit_bars, name="exit_bars")
        _positive_int(self.atr_period, name="atr_period")
        if exit_ >= entry:
            raise ValueError("exit_bars must be smaller than entry_bars")
        initial = _finite_real(self.initial_stop_atr, name="initial_stop_atr")
        trailing = _finite_real(self.trailing_stop_atr, name="trailing_stop_atr")
        if initial <= 0.0 or trailing <= 0.0:
            raise ValueError("stop distances must be positive")
        risk = _finite_real(self.risk_per_trade, name="risk_per_trade")
        if not 0.0 < risk < 1.0:
            raise ValueError("risk_per_trade must be within (0, 1)")
        symbol_cap = _finite_real(self.max_symbol_weight, name="max_symbol_weight")
        gross = _finite_real(self.gross_cap, name="gross_cap")
        if not 0.0 < symbol_cap <= gross <= 1.0:
            raise ValueError(
                "weights must satisfy 0 < max_symbol_weight <= gross_cap <= 1"
            )
        if not isinstance(self.allow_short, bool):
            raise TypeError("allow_short must be a boolean")


@dataclass(frozen=True, slots=True)
class TrendStopSignals:
    """Causal per-row, per-symbol rule inputs (NaN where unavailable)."""

    entry_high: np.ndarray
    entry_low: np.ndarray
    entry_valid: np.ndarray
    exit_high: np.ndarray
    exit_low: np.ndarray
    exit_valid: np.ndarray
    atr: np.ndarray
    atr_valid: np.ndarray


@dataclass(frozen=True, slots=True)
class StopExitEvent:
    """One stop decision: the exit order is submitted at ``decision_index``."""

    decision_index: int
    symbol_index: int
    side: int
    stop_level: float
    trailing: bool


def usable_rows(dataset: MarketDataset) -> np.ndarray:
    """Rows whose bar may feed a rule: tradable, active and available on time."""

    return np.asarray(
        dataset.resolved_array("information_available")
        & dataset.resolved_array("symbol_active")
        & dataset.tradable
        & (dataset.resolved_array("available_at") <= dataset.timestamps[:, None]),
        dtype=np.bool_,
    )


def donchian_extrema(
    high: np.ndarray, low: np.ndarray, usable: np.ndarray, bars: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Highest high / lowest low of the prior ``bars`` rows; row ``t`` is excluded.

    A value is valid only when the whole prior window and the current row are usable.
    """

    bars = _positive_int(bars, name="bars")
    n_rows, n_symbols = high.shape
    upper = np.full((n_rows, n_symbols), np.nan, dtype=np.float64)
    lower = np.full((n_rows, n_symbols), np.nan, dtype=np.float64)
    valid = np.zeros((n_rows, n_symbols), dtype=np.bool_)
    if n_rows <= bars:
        return upper, lower, valid
    windows_high = np.lib.stride_tricks.sliding_window_view(high[:-1], bars, axis=0)
    windows_low = np.lib.stride_tricks.sliding_window_view(low[:-1], bars, axis=0)
    windows_ok = np.lib.stride_tricks.sliding_window_view(usable[:-1], bars, axis=0)
    ok = windows_ok.all(axis=-1) & usable[bars:]
    upper[bars:] = np.where(ok, windows_high.max(axis=-1), np.nan)
    lower[bars:] = np.where(ok, windows_low.min(axis=-1), np.nan)
    valid[bars:] = ok
    return upper, lower, valid


def wilder_atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    usable: np.ndarray,
    period: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Wilder ATR in price units, seeded by the mean of the first ``period`` TRs.

    The state restarts after any unusable row, so a value never carries information
    across a data gap. ``atr[t]`` uses rows ``<= t`` only.
    """

    period = _positive_int(period, name="period")
    n_rows, n_symbols = close.shape
    atr = np.full((n_rows, n_symbols), np.nan, dtype=np.float64)
    valid = np.zeros((n_rows, n_symbols), dtype=np.bool_)
    for symbol in range(n_symbols):
        previous_ok = False
        previous_close = 0.0
        seen = 0
        seed = 0.0
        value = math.nan
        for row in range(n_rows):
            if not usable[row, symbol]:
                previous_ok = False
                seen = 0
                seed = 0.0
                value = math.nan
                continue
            high_value = float(high[row, symbol])
            low_value = float(low[row, symbol])
            if previous_ok:
                true_range = max(
                    high_value - low_value,
                    abs(high_value - previous_close),
                    abs(low_value - previous_close),
                )
                if seen < period:
                    seed += true_range
                    seen += 1
                    if seen == period:
                        value = seed / period
                else:
                    value = (value * (period - 1) + true_range) / period
                if seen >= period:
                    atr[row, symbol] = value
                    valid[row, symbol] = True
            previous_ok = True
            previous_close = float(close[row, symbol])
    return atr, valid


def build_trend_stop_signals(
    dataset: MarketDataset, config: TrendStopConfig
) -> TrendStopSignals:
    """Compute every causal rule input once from the Dataset's completed bars."""

    usable = usable_rows(dataset)
    high = np.asarray(dataset.high, dtype=np.float64)
    low = np.asarray(dataset.low, dtype=np.float64)
    close = np.asarray(dataset.close, dtype=np.float64)
    entry_high, entry_low, entry_valid = donchian_extrema(
        high, low, usable, config.entry_bars
    )
    exit_high, exit_low, exit_valid = donchian_extrema(
        high, low, usable, config.exit_bars
    )
    atr, atr_valid = wilder_atr(high, low, close, usable, config.atr_period)
    return TrendStopSignals(
        entry_high=entry_high,
        entry_low=entry_low,
        entry_valid=entry_valid,
        exit_high=exit_high,
        exit_low=exit_low,
        exit_valid=exit_valid,
        atr=atr,
        atr_valid=atr_valid,
    )


class _SymbolState:
    __slots__ = (
        "side",
        "anchor",
        "extreme",
        "atr0",
        "pending_atr",
        "blocked_side",
        "stop_pending",
    )

    def __init__(self) -> None:
        self.side = 0
        self.anchor = math.nan
        self.extreme = math.nan
        self.atr0 = math.nan
        self.pending_atr = math.nan
        self.blocked_side = 0
        self.stop_pending = False


class TrendStopPortfolioPolicy:
    """One ``PortfolioPolicy`` for a fixed Dataset roster; instances are single-use."""

    def __init__(self, dataset: MarketDataset, config: TrendStopConfig) -> None:
        if not isinstance(dataset, MarketDataset):
            raise TypeError("dataset must be a MarketDataset")
        if not isinstance(config, TrendStopConfig):
            raise TypeError("config must be a TrendStopConfig")
        self._config = config
        self._symbols = tuple(dataset.symbols)
        self._open = np.asarray(dataset.open, dtype=np.float64)
        self._high = np.asarray(dataset.high, dtype=np.float64)
        self._low = np.asarray(dataset.low, dtype=np.float64)
        self._close = np.asarray(dataset.close, dtype=np.float64)
        self._usable = usable_rows(dataset)
        self._timestamps_ns = np.asarray(
            dataset.timestamps.astype("datetime64[ns]").astype(np.int64)
        )
        self._signals = build_trend_stop_signals(dataset, config)
        self._states = [_SymbolState() for _ in self._symbols]
        self._stop_events: list[StopExitEvent] = []
        self._last_index: int | None = None
        self._counters = {
            "entries": 0,
            "reversals": 0,
            "stop_exits": 0,
            "signal_exits": 0,
            "capacity_scaled": 0,
            "capacity_blocked": 0,
        }

    @property
    def config(self) -> TrendStopConfig:
        return self._config

    @property
    def symbols(self) -> tuple[str, ...]:
        return self._symbols

    @property
    def stop_events(self) -> tuple[StopExitEvent, ...]:
        return tuple(self._stop_events)

    def summary(self) -> dict[str, int]:
        return dict(self._counters)

    # -- decision -----------------------------------------------------------------

    def decide_portfolio(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> PortfolioTargetProposal:
        if type(snapshot) is not PortfolioDecisionSnapshot:
            raise TypeError("snapshot must be a PortfolioDecisionSnapshot")
        if snapshot.symbols != self._symbols:
            raise ValueError("snapshot symbols do not match the policy roster")
        row = snapshot.index
        if row >= self._timestamps_ns.size or (
            snapshot.timestamp_ns != int(self._timestamps_ns[row])
        ):
            raise ValueError("snapshot is not bound to this Dataset row")
        if self._last_index is not None and row != self._last_index + 1:
            raise ValueError("snapshot index must advance by exactly one decision")
        self._last_index = row

        account = snapshot.account
        n_symbols = len(self._symbols)
        if account.insolvent or account.termination_reason is not None:
            return PortfolioTargetProposal(
                target_weights=account.weights,
                hold_current_quantity_symbols=tuple(range(n_symbols)),
            )

        # (kind, weight): "hold" keeps the exact quantity, "exit" targets zero and
        # "enter" carries the signed risk-sized weight before the gross-cap scale.
        decisions: list[tuple[str, float]] = []
        for index in range(n_symbols):
            decisions.append(self._decide_symbol(index, row, account.quantities[index]))

        held_gross = sum(
            abs(account.weights[index])
            for index, (kind, _) in enumerate(decisions)
            if kind == "hold"
        )
        requested = sum(abs(weight) for kind, weight in decisions if kind == "enter")
        scale = 1.0
        if requested > _TOLERANCE:
            capacity = max(0.0, self._config.gross_cap - held_gross)
            if capacity < requested:
                scale = capacity / requested
                if scale <= _TOLERANCE:
                    self._counters["capacity_blocked"] += sum(
                        kind == "enter" for kind, _ in decisions
                    )
                    scale = 0.0
                else:
                    self._counters["capacity_scaled"] += sum(
                        kind == "enter" for kind, _ in decisions
                    )

        targets: list[float] = []
        holds: list[int] = []
        for index, (kind, weight) in enumerate(decisions):
            if kind == "hold":
                targets.append(account.weights[index])
                holds.append(index)
            elif kind == "exit":
                targets.append(0.0)
            else:
                targets.append(weight * scale)
        return PortfolioTargetProposal(
            target_weights=tuple(targets),
            hold_current_quantity_symbols=tuple(holds),
        )

    def _decide_symbol(
        self, index: int, row: int, quantity: float
    ) -> tuple[str, float]:
        config = self._config
        signals = self._signals
        state = self._states[index]
        side = 0 if abs(quantity) <= _TOLERANCE else (1 if quantity > 0.0 else -1)

        if side == 0:
            state.side = 0
            state.stop_pending = False
        elif side != state.side:
            state.side = side
            state.stop_pending = False
            state.anchor = float(self._open[row, index])
            state.atr0 = (
                state.pending_atr
                if math.isfinite(state.pending_atr)
                else float(signals.atr[row, index])
            )
            state.extreme = state.anchor

        usable = bool(self._usable[row, index])
        close = float(self._close[row, index])
        long_signal = short_signal = False
        if usable and bool(signals.entry_valid[row, index]):
            long_signal = close > float(signals.entry_high[row, index])
            short_signal = config.allow_short and (
                close < float(signals.entry_low[row, index])
            )

        if side == 0:
            if state.blocked_side == 1 and not long_signal:
                state.blocked_side = 0
            if state.blocked_side == -1 and not short_signal:
                state.blocked_side = 0
            if long_signal and state.blocked_side != 1:
                return self._enter(index, row, 1, close)
            if short_signal and state.blocked_side != -1:
                return self._enter(index, row, -1, close)
            return ("exit", 0.0)

        if not usable:
            return ("hold", 0.0)

        if not math.isfinite(state.atr0) or state.atr0 <= 0.0:
            # Fail closed: a position without a defined stop distance is closed.
            return ("exit", 0.0)

        stop_hit, level, trailing = self._stop_state(state, row, index)
        opposite = short_signal if side == 1 else long_signal
        if opposite:
            if stop_hit:
                self._record_stop(row, index, side, level, trailing)
            self._counters["reversals"] += 1
            return self._enter(index, row, -side, close, reversal=True)
        if stop_hit:
            self._record_stop(row, index, side, level, trailing)
            state.blocked_side = side
            return ("exit", 0.0)
        if bool(signals.exit_valid[row, index]):
            if side == 1 and close < float(signals.exit_low[row, index]):
                self._counters["signal_exits"] += 1
                return ("exit", 0.0)
            if side == -1 and close > float(signals.exit_high[row, index]):
                self._counters["signal_exits"] += 1
                return ("exit", 0.0)
        return ("hold", 0.0)

    def _stop_state(
        self, state: _SymbolState, row: int, index: int
    ) -> tuple[bool, float, bool]:
        config = self._config
        high = float(self._high[row, index])
        low = float(self._low[row, index])
        if state.side == 1:
            state.extreme = max(state.extreme, high)
            initial = state.anchor - config.initial_stop_atr * state.atr0
            trail = state.extreme - config.trailing_stop_atr * state.atr0
            level = max(initial, trail)
            return low <= level, level, trail > initial
        state.extreme = min(state.extreme, low)
        initial = state.anchor + config.initial_stop_atr * state.atr0
        trail = state.extreme + config.trailing_stop_atr * state.atr0
        level = min(initial, trail)
        return high >= level, level, trail < initial

    def _record_stop(
        self, row: int, index: int, side: int, level: float, trailing: bool
    ) -> None:
        state = self._states[index]
        if state.stop_pending:
            # The exit order is still working (partial fill); one event per position.
            return
        state.stop_pending = True
        self._counters["stop_exits"] += 1
        self._stop_events.append(
            StopExitEvent(
                decision_index=row,
                symbol_index=index,
                side=side,
                stop_level=level,
                trailing=trailing,
            )
        )

    def _enter(
        self,
        index: int,
        row: int,
        side: int,
        close: float,
        *,
        reversal: bool = False,
    ) -> tuple[str, float]:
        config = self._config
        atr = float(self._signals.atr[row, index])
        if not bool(self._signals.atr_valid[row, index]) or atr <= 0.0:
            return ("exit", 0.0)
        distance_fraction = config.initial_stop_atr * atr / close
        weight = min(
            config.risk_per_trade / distance_fraction, config.max_symbol_weight
        )
        self._states[index].pending_atr = atr
        if not reversal:
            self._counters["entries"] += 1
        return ("enter", side * weight)


__all__ = [
    "StopExitEvent",
    "TrendStopConfig",
    "TrendStopPortfolioPolicy",
    "TrendStopSignals",
    "build_trend_stop_signals",
    "donchian_extrema",
    "usable_rows",
    "wilder_atr",
]
