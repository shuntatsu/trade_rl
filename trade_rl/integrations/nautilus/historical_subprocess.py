"""Fresh-process boundary for historical Nautilus replay execution."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from trade_rl.integrations.nautilus.event_projection import SourceBar
from trade_rl.integrations.nautilus.historical_execution import (
    NautilusHistoricalExecutionResult,
    NautilusHistoricalPositionSnapshot,
    NautilusHistoricalTargetInterval,
    run_historical_target_intervals,
)
from trade_rl.simulation.execution_canonicalization import CanonicalFillSignature

_WORKER_MODULE = "trade_rl.integrations.nautilus.historical_subprocess"


@dataclass(frozen=True, slots=True)
class NautilusHistoricalSubprocessResult:
    """One historical execution plus the child process identity that produced it."""

    worker_pid: int
    execution: NautilusHistoricalExecutionResult


def run_historical_target_intervals_subprocess(
    intervals: tuple[NautilusHistoricalTargetInterval, ...],
    *,
    snapshot_timestamps_ns: tuple[int, ...] = (),
    starting_balance: Decimal = Decimal("100000"),
    no_trade_band: float = 0.05,
    timeout_seconds: float = 60.0,
) -> NautilusHistoricalSubprocessResult:
    """Execute one complete historical replay in a fresh Python child process."""

    if timeout_seconds <= 0.0:
        raise ValueError("historical subprocess timeout must be positive")
    request = {
        "intervals": [_interval_to_payload(interval) for interval in intervals],
        "snapshot_timestamps_ns": list(snapshot_timestamps_ns),
        "starting_balance": str(starting_balance),
        "no_trade_band": no_trade_band,
    }
    with tempfile.TemporaryDirectory(prefix="trade-rl-nautilus-") as directory:
        root = Path(directory)
        request_path = root / "request.json"
        result_path = root / "result.json"
        request_path.write_text(
            json.dumps(request, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        completed = subprocess.run(
            (
                sys.executable,
                "-m",
                _WORKER_MODULE,
                "--worker",
                str(request_path),
                str(result_path),
            ),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        if completed.returncode != 0:
            details = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(
                "historical Nautilus subprocess failed "
                f"with exit code {completed.returncode}: {details[-4000:]}"
            )
        if not result_path.is_file():
            raise RuntimeError("historical Nautilus subprocess did not write a result")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    return _subprocess_result_from_payload(payload)


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


def _execution_from_payload(payload: dict[str, Any]) -> NautilusHistoricalExecutionResult:
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


def _subprocess_result_from_payload(payload: dict[str, Any]) -> NautilusHistoricalSubprocessResult:
    return NautilusHistoricalSubprocessResult(
        worker_pid=int(payload["worker_pid"]),
        execution=_execution_from_payload(payload["execution"]),
    )


def _run_worker(request_path: Path, result_path: Path) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    intervals = tuple(_interval_from_payload(value) for value in request["intervals"])
    execution = run_historical_target_intervals(
        intervals,
        snapshot_timestamps_ns=tuple(
            int(value) for value in request["snapshot_timestamps_ns"]
        ),
        starting_balance=Decimal(str(request["starting_balance"])),
        no_trade_band=float(request["no_trade_band"]),
    )
    result_path.write_text(
        json.dumps(
            {
                "worker_pid": os.getpid(),
                "execution": _execution_to_payload(execution),
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )


def _main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("request_path", nargs="?")
    parser.add_argument("result_path", nargs="?")
    args = parser.parse_args()
    if not args.worker or args.request_path is None or args.result_path is None:
        parser.error("worker mode requires request and result paths")
    _run_worker(Path(args.request_path), Path(args.result_path))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through child process
    raise SystemExit(_main())


__all__ = [
    "NautilusHistoricalSubprocessResult",
    "run_historical_target_intervals_subprocess",
]
