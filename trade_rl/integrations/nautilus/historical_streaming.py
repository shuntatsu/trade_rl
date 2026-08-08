"""Persistent child-process streaming runtime for historical Nautilus replay."""

from __future__ import annotations

import json
import math
import multiprocessing
import os
from dataclasses import asdict, dataclass
from decimal import Decimal
from multiprocessing.connection import Connection
from typing import Any

from trade_rl.integrations.nautilus.event_projection import (
    MarketPhase,
    SourceBar,
    project_bar_events,
)
from trade_rl.integrations.nautilus.historical_execution import (
    NautilusHistoricalExecutionResult,
    NautilusHistoricalPositionSnapshot,
    NautilusHistoricalTargetInterval,
)
from trade_rl.integrations.nautilus.instrument import build_maintained_btcusdt_perpetual
from trade_rl.integrations.nautilus.order_adapter import submit_target_exposure_plan
from trade_rl.integrations.nautilus.quote_projection import build_quote_tick
from trade_rl.integrations.nautilus.runtime_identity import require_nautilus_runtime
from trade_rl.integrations.nautilus.trace_adapter import (
    canonicalize_nautilus_fill_events,
)
from trade_rl.simulation.execution_canonicalization import CanonicalFillSignature
from trade_rl.simulation.target_exposure_controller import (
    TargetExposureController,
    TargetExposureInput,
)

_PRICE_PHASES = frozenset(
    {
        MarketPhase.OPEN_QUOTE,
        MarketPhase.HIGH,
        MarketPhase.LOW,
        MarketPhase.CLOSE_QUOTE,
    }
)


@dataclass(frozen=True, slots=True)
class NautilusHistoricalStreamingResult:
    """Cumulative execution emitted by one persistent child worker."""

    worker_pid: int
    execution: NautilusHistoricalExecutionResult


class NautilusHistoricalStreamingWorker:
    """Own one spawned child and one BacktestEngine for an episode-like stream."""

    def __init__(
        self,
        *,
        starting_balance: Decimal = Decimal("100000"),
        no_trade_band: float = 0.05,
        timeout_seconds: float = 60.0,
    ) -> None:
        if not starting_balance.is_finite() or starting_balance <= 0:
            raise ValueError("streaming starting_balance must be finite and positive")
        if not math.isfinite(no_trade_band) or no_trade_band < 0.0:
            raise ValueError("streaming no_trade_band must be finite and non-negative")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
            raise ValueError("streaming timeout_seconds must be finite and positive")

        context = multiprocessing.get_context("spawn")
        parent_connection, child_connection = context.Pipe(duplex=True)
        process = context.Process(
            target=_streaming_worker_main,
            args=(child_connection,),
            daemon=True,
        )
        process.start()
        child_connection.close()

        self._connection = parent_connection
        self._process = process
        self._timeout_seconds = float(timeout_seconds)
        self._closed = False
        try:
            _send_message(
                self._connection,
                {
                    "command": "initialize",
                    "starting_balance": str(starting_balance),
                    "no_trade_band": float(no_trade_band),
                },
            )
            response = self._receive_response(expected_event="ready")
            worker_pid = _required_int(response, "worker_pid")
            if process.pid is None or worker_pid != process.pid:
                raise RuntimeError("streaming Nautilus child identity mismatch")
            self._worker_pid = worker_pid
        except BaseException:
            self._terminate()
            raise

    @property
    def worker_pid(self) -> int:
        return self._worker_pid

    def __enter__(self) -> NautilusHistoricalStreamingWorker:
        return self

    def __exit__(self, _exc_type: object, exc: object, _traceback: object) -> None:
        self.close()

    def execute(
        self,
        interval: NautilusHistoricalTargetInterval,
    ) -> NautilusHistoricalStreamingResult:
        """Advance the persistent engine by exactly one target interval."""

        if self._closed:
            raise RuntimeError("streaming Nautilus worker is closed")
        if not isinstance(interval, NautilusHistoricalTargetInterval):
            raise TypeError("streaming interval has an invalid type")
        _send_message(
            self._connection,
            {
                "command": "execute",
                "interval": _interval_to_payload(interval),
            },
        )
        response = self._receive_response(expected_event="execution")
        worker_pid = _required_int(response, "worker_pid")
        if worker_pid != self._worker_pid:
            raise RuntimeError("streaming Nautilus worker identity changed")
        execution_payload = response.get("execution")
        if not isinstance(execution_payload, dict):
            raise RuntimeError("streaming Nautilus child returned invalid execution")
        return NautilusHistoricalStreamingResult(
            worker_pid=worker_pid,
            execution=_execution_from_payload(execution_payload),
        )

    def close(self) -> None:
        """Finalize the child engine and release the process deterministically."""

        if self._closed:
            return
        self._closed = True
        try:
            if self._process.is_alive():
                try:
                    _send_message(self._connection, {"command": "close"})
                    self._receive_response(expected_event="closed")
                except (BrokenPipeError, ConnectionResetError, EOFError, OSError):
                    pass
                except RuntimeError as exc:
                    if not isinstance(exc.__cause__, EOFError) and self._process.is_alive():
                        raise
                else:
                    self._process.join(timeout=self._timeout_seconds)
                    if self._process.is_alive():
                        raise RuntimeError(
                            "streaming Nautilus child did not exit after close"
                        )
                    if self._process.exitcode != 0:
                        raise RuntimeError(
                            "streaming Nautilus child exited unsuccessfully: "
                            f"{self._process.exitcode}"
                        )
        finally:
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=self._timeout_seconds)
            self._connection.close()

    def _receive_response(self, *, expected_event: str) -> dict[str, Any]:
        if not self._connection.poll(self._timeout_seconds):
            if not self._process.is_alive():
                raise RuntimeError(
                    "streaming Nautilus child exited before returning a response: "
                    f"{self._process.exitcode}"
                )
            raise TimeoutError(
                f"streaming Nautilus child timed out waiting for {expected_event}"
            )
        try:
            response = _receive_message(self._connection)
        except EOFError as exc:
            raise RuntimeError(
                "streaming Nautilus child closed its response channel unexpectedly"
            ) from exc
        if response.get("ok") is not True:
            error = response.get("error")
            raise RuntimeError(f"streaming Nautilus child failed: {error}")
        if response.get("event") != expected_event:
            raise RuntimeError(
                "streaming Nautilus child returned an unexpected event: "
                f"{response.get('event')!r}"
            )
        return response

    def _terminate(self) -> None:
        self._closed = True
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=self._timeout_seconds)
        self._connection.close()


class _StreamingSession:
    def __init__(self, *, starting_balance: Decimal, no_trade_band: float) -> None:
        runtime = require_nautilus_runtime()
        controller = TargetExposureController(no_trade_band=no_trade_band)

        from nautilus_trader.adapters.binance import BINANCE_VENUE
        from nautilus_trader.backtest.engine import BacktestEngine
        from nautilus_trader.model import Money
        from nautilus_trader.model.currencies import USDT
        from nautilus_trader.model.enums import AccountType, OmsType
        from nautilus_trader.trading.strategy import Strategy

        engine = BacktestEngine()
        engine.add_venue(
            venue=BINANCE_VENUE,
            oms_type=OmsType.NETTING,
            account_type=AccountType.MARGIN,
            base_currency=USDT,
            starting_balances=[Money(starting_balance, USDT)],
        )
        instrument = build_maintained_btcusdt_perpetual()
        engine.add_instrument(instrument)
        quantity_increment = float(instrument.size_increment.as_decimal())

        class StreamingTargetStrategy(Strategy):
            def __init__(self) -> None:
                super().__init__()
                self.realized_quantity = Decimal("0")
                self.queued_interval: NautilusHistoricalTargetInterval | None = None
                self.pending_interval: NautilusHistoricalTargetInterval | None = None
                self.filled_events: list[object] = []

            def on_start(self) -> None:
                self.subscribe_quote_ticks(instrument.id)

            def queue_interval(
                self, interval: NautilusHistoricalTargetInterval
            ) -> None:
                if (
                    self.queued_interval is not None
                    or self.pending_interval is not None
                ):
                    raise RuntimeError(
                        "streaming target strategy already has pending work"
                    )
                self.queued_interval = interval

            def on_quote_tick(self, tick: object) -> None:
                if self.pending_interval is not None:
                    self._reconcile_pending()
                if self.queued_interval is None:
                    return
                if self.pending_interval is not None:
                    raise RuntimeError(
                        "previous streaming target was not reconciled before activation"
                    )
                self.pending_interval = self.queued_interval
                self.queued_interval = None
                self._reconcile_pending()

            def on_order_filled(self, event: object) -> None:
                self.filled_events.append(event)
                self.realized_quantity += _signed_fill_quantity(event)

            def _reconcile_pending(self) -> None:
                interval = self.pending_interval
                if interval is None:
                    return
                plan = controller.plan(
                    TargetExposureInput(
                        target_exposure=interval.target_exposure,
                        allocated_equity=interval.allocated_equity,
                        reference_price=interval.source_bars[0].open_price,
                        contract_multiplier=1.0,
                        realized_quantity=float(self.realized_quantity),
                        working_remaining_quantities=(),
                        quantity_increment=quantity_increment,
                    )
                )
                if plan.cancel_working_orders:
                    raise RuntimeError(
                        "streaming Market IOC replay must not retain stale orders"
                    )
                if plan.child_order is None:
                    self.pending_interval = None
                    return
                deferred_replan = plan.deferred_target_quantity is not None
                submit_target_exposure_plan(
                    strategy=self,
                    instrument=instrument,
                    plan=plan,
                )
                if not deferred_replan:
                    self.pending_interval = None

        strategy = StreamingTargetStrategy()
        engine.add_strategy(strategy)

        self.runtime_version = runtime.package_version or ""
        self.engine = engine
        self.instrument = instrument
        self.strategy = strategy
        self.venue = BINANCE_VENUE
        self.currency = USDT
        self._next_sequence = 1
        self._closed = False

    def execute(
        self,
        interval: NautilusHistoricalTargetInterval,
    ) -> NautilusHistoricalExecutionResult:
        if self._closed:
            raise RuntimeError("streaming Nautilus session is closed")
        _validate_interval(interval, expected_sequence=self._next_sequence)

        quotes: list[Any] = []
        first_phase: MarketPhase | None = None
        for bar in interval.source_bars:
            for event in project_bar_events(bar, activate_queued_target=False):
                if event.phase not in _PRICE_PHASES:
                    continue
                if first_phase is None:
                    first_phase = event.phase
                quotes.append(
                    build_quote_tick(
                        event,
                        instrument=self.instrument,
                        half_spread_ticks=1,
                        displayed_size=1_000_000.0,
                    )
                )
        if first_phase is not MarketPhase.OPEN_QUOTE or not quotes:
            raise RuntimeError("streaming interval must begin with an open quote")

        self.strategy.queue_interval(interval)
        self.engine.add_data(quotes, sort=True)
        try:
            self.engine.run(streaming=True)
        finally:
            self.engine.clear_data()
        if self.strategy.queued_interval is not None:
            raise RuntimeError("streaming target was not activated")
        if self.strategy.pending_interval is not None:
            raise RuntimeError("streaming target remained pending after its interval")

        self._next_sequence += 1
        canonical = canonicalize_nautilus_fill_events(
            self.strategy.filled_events,
            price_tick=self.instrument.price_increment.as_decimal(),
            lot_size=self.instrument.size_increment.as_decimal(),
            currency_precision=self.currency.precision,
        )
        account = self.engine.cache.account_for_venue(self.venue)
        if account is None:
            raise RuntimeError("expected Binance streaming backtest account")
        balance = account.balance_total(self.currency)
        if balance is None:
            raise RuntimeError("expected streaming USDT account balance")
        terminal_position_lots = (
            canonical.fills[-1].position_lots if canonical.fills else 0
        )
        return NautilusHistoricalExecutionResult(
            runtime_version=self.runtime_version,
            fills=canonical.fills,
            fee_minor=canonical.fee_minor,
            final_balance_minor=_minor_units(
                balance.as_decimal(),
                currency_precision=self.currency.precision,
                name="final_balance",
            ),
            terminal_position_lots=terminal_position_lots,
            terminal_open_orders=len(
                self.engine.cache.orders_open(instrument_id=self.instrument.id)
            ),
            position_snapshots=(),
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.engine.end()
        finally:
            self.engine.dispose()


def _streaming_worker_main(connection: Connection) -> None:
    session: _StreamingSession | None = None
    try:
        initialize = _receive_message(connection)
        if initialize.get("command") != "initialize":
            raise RuntimeError("streaming worker expected initialize command")
        session = _StreamingSession(
            starting_balance=Decimal(_required_str(initialize, "starting_balance")),
            no_trade_band=_required_float(initialize, "no_trade_band"),
        )
        _send_message(
            connection,
            {
                "ok": True,
                "event": "ready",
                "worker_pid": os.getpid(),
                "runtime_version": session.runtime_version,
            },
        )
        while True:
            request = _receive_message(connection)
            command = request.get("command")
            if command == "execute":
                interval_payload = request.get("interval")
                if not isinstance(interval_payload, dict):
                    raise RuntimeError(
                        "streaming worker received invalid interval payload"
                    )
                execution = session.execute(_interval_from_payload(interval_payload))
                _send_message(
                    connection,
                    {
                        "ok": True,
                        "event": "execution",
                        "worker_pid": os.getpid(),
                        "execution": _execution_to_payload(execution),
                    },
                )
            elif command == "close":
                session.close()
                session = None
                _send_message(
                    connection,
                    {"ok": True, "event": "closed", "worker_pid": os.getpid()},
                )
                return
            else:
                raise RuntimeError(f"unsupported streaming worker command: {command!r}")
    except Exception as exc:
        try:
            _send_message(
                connection,
                {
                    "ok": False,
                    "event": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if session is not None:
            session.close()
        connection.close()


def _send_message(connection: Connection, payload: dict[str, Any]) -> None:
    connection.send_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def _receive_message(connection: Connection) -> dict[str, Any]:
    payload = json.loads(connection.recv_bytes().decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("streaming Nautilus message must be an object")
    return payload


def _interval_to_payload(interval: NautilusHistoricalTargetInterval) -> dict[str, Any]:
    return {
        "sequence": interval.sequence,
        "target_exposure": interval.target_exposure,
        "allocated_equity": interval.allocated_equity,
        "source_bars": [asdict(bar) for bar in interval.source_bars],
    }


def _interval_from_payload(payload: dict[str, Any]) -> NautilusHistoricalTargetInterval:
    return NautilusHistoricalTargetInterval(
        sequence=int(payload["sequence"]),
        target_exposure=float(payload["target_exposure"]),
        allocated_equity=float(payload["allocated_equity"]),
        source_bars=tuple(
            SourceBar(**bar_payload) for bar_payload in payload["source_bars"]
        ),
    )


def _execution_to_payload(
    execution: NautilusHistoricalExecutionResult,
) -> dict[str, Any]:
    return {
        "runtime_version": execution.runtime_version,
        "fills": [asdict(fill) for fill in execution.fills],
        "fee_minor": execution.fee_minor,
        "final_balance_minor": execution.final_balance_minor,
        "terminal_position_lots": execution.terminal_position_lots,
        "terminal_open_orders": execution.terminal_open_orders,
        "position_snapshots": [
            {
                "timestamp_ns": snapshot.timestamp_ns,
                "signed_quantity": str(snapshot.signed_quantity),
            }
            for snapshot in execution.position_snapshots
        ],
    }


def _execution_from_payload(
    payload: dict[str, Any],
) -> NautilusHistoricalExecutionResult:
    return NautilusHistoricalExecutionResult(
        runtime_version=str(payload["runtime_version"]),
        fills=tuple(CanonicalFillSignature(**fill) for fill in payload["fills"]),
        fee_minor=int(payload["fee_minor"]),
        final_balance_minor=int(payload["final_balance_minor"]),
        terminal_position_lots=int(payload["terminal_position_lots"]),
        terminal_open_orders=int(payload["terminal_open_orders"]),
        position_snapshots=tuple(
            NautilusHistoricalPositionSnapshot(
                timestamp_ns=int(snapshot["timestamp_ns"]),
                signed_quantity=Decimal(str(snapshot["signed_quantity"])),
            )
            for snapshot in payload["position_snapshots"]
        ),
    )


def _validate_interval(
    interval: NautilusHistoricalTargetInterval,
    *,
    expected_sequence: int,
) -> None:
    if interval.sequence != expected_sequence:
        raise ValueError("streaming historical interval sequence must be contiguous")
    if (
        not math.isfinite(interval.target_exposure)
        or not -1.0 <= interval.target_exposure <= 1.0
    ):
        raise ValueError("streaming target exposure must be within [-1, 1]")
    if not math.isfinite(interval.allocated_equity) or interval.allocated_equity <= 0.0:
        raise ValueError("streaming allocated equity must be finite and positive")
    if not interval.source_bars:
        raise ValueError("streaming target interval must contain source bars")


def _signed_fill_quantity(event: object) -> Decimal:
    quantity = _as_decimal(getattr(event, "last_qty", None), name="last_qty")
    side = getattr(event, "order_side", None)
    side_name = getattr(side, "name", str(side)).upper()
    if side_name.endswith("BUY"):
        return quantity
    if side_name.endswith("SELL"):
        return -quantity
    raise ValueError(f"unsupported Nautilus order side: {side!r}")


def _as_decimal(value: object, *, name: str) -> Decimal:
    if value is None:
        raise ValueError(f"{name} is missing")
    if hasattr(value, "as_decimal"):
        value = value.as_decimal()
    result = value if isinstance(value, Decimal) else Decimal(str(value))
    if not result.is_finite():
        raise ValueError(f"{name} must be finite")
    return result


def _minor_units(value: Decimal, *, currency_precision: int, name: str) -> int:
    scaled = value * (Decimal(10) ** currency_precision)
    integral = scaled.to_integral_value()
    if scaled != integral:
        raise RuntimeError(f"{name} is not aligned to settlement minor unit")
    return int(integral)


def _required_float(payload: dict[str, Any], field: str) -> float:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"streaming Nautilus response field {field!r} is invalid")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"streaming Nautilus response field {field!r} is not finite")
    return result


def _required_str(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"streaming Nautilus response field {field!r} is invalid")
    return value


def _required_int(payload: dict[str, Any], field: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"streaming Nautilus response field {field!r} is invalid")
    return value


__all__ = [
    "NautilusHistoricalStreamingResult",
    "NautilusHistoricalStreamingWorker",
]
