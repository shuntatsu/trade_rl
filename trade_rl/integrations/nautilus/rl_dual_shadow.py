"""Nautilus dual-shadow observer for authoritative RL execution steps."""

from __future__ import annotations

import math
from decimal import Decimal

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketDataset
from trade_rl.integrations.nautilus.historical_execution import (
    NautilusHistoricalTargetInterval,
)
from trade_rl.integrations.nautilus.historical_projection import (
    project_historical_interval_source_bars,
)
from trade_rl.integrations.nautilus.historical_streaming import (
    NautilusHistoricalStreamingWorker,
)
from trade_rl.integrations.nautilus.instrument import MAINTAINED_BTCUSDT_PERPETUAL
from trade_rl.rl.environment_execution import (
    ExecutionDualShadowRequest,
    ExecutionDualShadowSnapshot,
)

_QUANTITY_TOLERANCE = 1e-12
_MAINTAINED_DATASET_SYMBOL = "BTCUSDT"


class NautilusEnvironmentDualShadow:
    """Stream authoritative hybrid targets through one Nautilus child per episode."""

    def __init__(
        self,
        dataset: MarketDataset,
        *,
        no_trade_band: float = 0.05,
        timeout_seconds: float = 60.0,
    ) -> None:
        if dataset.symbols != (_MAINTAINED_DATASET_SYMBOL,):
            raise ValueError(
                "Nautilus RL dual shadow requires maintained symbol BTCUSDT"
            )
        if not math.isfinite(no_trade_band) or no_trade_band < 0.0:
            raise ValueError("no_trade_band must be finite and non-negative")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be finite and positive")
        self.dataset = dataset
        self.no_trade_band = float(no_trade_band)
        self.timeout_seconds = float(timeout_seconds)
        self._identity_digest = content_digest(
            {
                "dataset_id": dataset.dataset_id,
                "instrument_id": MAINTAINED_BTCUSDT_PERPETUAL.instrument_id,
                "no_trade_band": self.no_trade_band,
                "runtime_contract": "nautilus_rl_dual_shadow_streaming_v2",
            }
        )
        self._next_start_index: int | None = None
        self._worker: NautilusHistoricalStreamingWorker | None = None
        self._step_count = 0

    @property
    def identity_digest(self) -> str:
        return self._identity_digest

    @property
    def step_count(self) -> int:
        return self._step_count

    def reset(
        self,
        *,
        start_index: int,
        initial_capital: float,
        initial_quantities: tuple[float, ...],
    ) -> None:
        if isinstance(start_index, bool) or not isinstance(start_index, int):
            raise TypeError("dual-shadow start_index must be an integer")
        if start_index < 0 or start_index >= len(self.dataset.timestamps) - 1:
            raise ValueError("dual-shadow start_index is outside replayable data")
        if not math.isfinite(initial_capital) or initial_capital <= 0.0:
            raise ValueError("dual-shadow initial_capital must be finite and positive")
        if len(initial_quantities) != 1:
            raise ValueError("Nautilus RL dual shadow requires one initial quantity")
        if abs(initial_quantities[0]) > _QUANTITY_TOLERANCE:
            raise ValueError(
                "Nautilus RL dual shadow currently requires a cash-only reset"
            )

        self.close()
        self._worker = NautilusHistoricalStreamingWorker(
            starting_balance=Decimal(str(initial_capital)),
            no_trade_band=self.no_trade_band,
            timeout_seconds=self.timeout_seconds,
        )
        self._next_start_index = start_index
        self._step_count = 0

    def observe(
        self, request: ExecutionDualShadowRequest
    ) -> ExecutionDualShadowSnapshot:
        worker = self._worker
        if worker is None or self._next_start_index is None:
            raise RuntimeError(
                "Nautilus RL dual shadow must be reset before observation"
            )
        if len(request.target) != 1 or len(request.legacy_terminal_quantities) != 1:
            raise ValueError(
                "Nautilus RL dual shadow requires single-instrument evidence"
            )
        if request.start_index != self._next_start_index:
            raise ValueError("Nautilus RL dual-shadow step boundary is not contiguous")
        if request.end_index <= request.start_index:
            raise ValueError("Nautilus RL dual-shadow interval must advance")
        if request.end_index >= len(self.dataset.timestamps):
            raise ValueError("Nautilus RL dual-shadow interval exceeds the dataset")

        interval = NautilusHistoricalTargetInterval(
            sequence=self._step_count + 1,
            target_exposure=request.target[0],
            allocated_equity=request.allocated_equity,
            source_bars=project_historical_interval_source_bars(
                self.dataset,
                start_index=request.start_index,
                end_index=request.end_index,
            ),
        )
        streaming_result = worker.execute(interval)
        lot_size = Decimal(MAINTAINED_BTCUSDT_PERPETUAL.size_increment)
        candidate_quantity = (
            Decimal(streaming_result.execution.terminal_position_lots) * lot_size
        )
        legacy_quantity = Decimal(str(request.legacy_terminal_quantities[0]))
        structural_parity = math.isclose(
            float(candidate_quantity),
            float(legacy_quantity),
            rel_tol=1e-12,
            abs_tol=max(float(lot_size) / 2.0, _QUANTITY_TOLERANCE),
        )

        self._next_start_index = request.end_index
        self._step_count += 1
        return ExecutionDualShadowSnapshot(
            runtime_identity=(
                f"nautilus_trader=={streaming_result.execution.runtime_version}/"
                "historical_streaming_v1"
            ),
            worker_pid=streaming_result.worker_pid,
            structural_parity=structural_parity,
            candidate_terminal_quantities=(float(candidate_quantity),),
            legacy_terminal_quantities=request.legacy_terminal_quantities,
        )

    def close(self) -> None:
        """Release the episode-owned child process if one is active."""

        worker = self._worker
        self._worker = None
        self._next_start_index = None
        if worker is not None:
            worker.close()


__all__ = ["NautilusEnvironmentDualShadow"]
