"""Out-of-sample test of ``TS_TREND_LONG_SPOT`` (``trend_long_spot_oos_v2``)."""

from __future__ import annotations

from trade_rl.evaluation.trend_oos.data import (
    StageDataError,
    StageEmbargoError,
    StageWindow,
    assemble_stage_dataset,
    frozen_window,
    read_limit,
)
from trade_rl.evaluation.trend_oos.decision import (
    OosCell,
    decide,
    stationary_bootstrap,
)
from trade_rl.evaluation.trend_oos.protocol import (
    PROTOCOL_ID,
    protocol_digest,
    protocol_payload,
)
from trade_rl.evaluation.trend_oos.runner import run_stage
from trade_rl.evaluation.trend_oos.signals import build_oos_signals

__all__ = [
    "PROTOCOL_ID",
    "OosCell",
    "StageDataError",
    "StageEmbargoError",
    "StageWindow",
    "assemble_stage_dataset",
    "build_oos_signals",
    "decide",
    "frozen_window",
    "protocol_digest",
    "protocol_payload",
    "read_limit",
    "run_stage",
    "stationary_bootstrap",
]
