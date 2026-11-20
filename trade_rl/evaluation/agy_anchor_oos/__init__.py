"""Out-of-sample test of the agy anchor (``agy_anchor_realistic_v1``)."""

from __future__ import annotations

from trade_rl.evaluation.agy_anchor_oos.data import (
    HourlyLeg,
    StageData,
    StageWindow,
    assemble_stage_dataset,
    build_anchor_inputs,
    frozen_window,
    read_limit,
)
from trade_rl.evaluation.agy_anchor_oos.decision import (
    OosCell,
    combine_label,
    decide_stage,
)
from trade_rl.evaluation.agy_anchor_oos.protocol import (
    PROTOCOL_ID,
    protocol_digest,
    protocol_payload,
)
from trade_rl.evaluation.agy_anchor_oos.runner import run_stage

__all__ = [
    "PROTOCOL_ID",
    "HourlyLeg",
    "OosCell",
    "StageData",
    "StageWindow",
    "assemble_stage_dataset",
    "build_anchor_inputs",
    "combine_label",
    "decide_stage",
    "frozen_window",
    "protocol_digest",
    "protocol_payload",
    "read_limit",
    "run_stage",
]
