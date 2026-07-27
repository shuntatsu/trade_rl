"""Validated market walk-forward configuration with explicit ledger trust mode."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import trade_rl.workflows._market_walk_forward_config_base as _base

ExecutionSensitivityConfig = _base.ExecutionSensitivityConfig
ExecutionSensitivityScenario = _base.ExecutionSensitivityScenario
NamedCandidateRun = _base.NamedCandidateRun


class SealedTestLedgerMode(StrEnum):
    """Explicit trust semantics for outer-test access authorization."""

    LOCAL_EXPLORATORY = "local_exploratory"
    DURABLE_POSTGRES = "durable_postgres"


@dataclass(frozen=True, slots=True)
class MarketWalkForwardConfig(_base.MarketWalkForwardConfig):
    sealed_test_ledger_mode: SealedTestLedgerMode = (
        SealedTestLedgerMode.LOCAL_EXPLORATORY
    )

    def __post_init__(self) -> None:
        _base.MarketWalkForwardConfig.__post_init__(self)
        if not isinstance(self.sealed_test_ledger_mode, SealedTestLedgerMode):
            raise ValueError("sealed_test_ledger_mode must be a supported mode")

    @classmethod
    def from_json(
        cls,
        path: Path,
        *,
        n_bars: int,
    ) -> "MarketWalkForwardConfig":
        base = _base.MarketWalkForwardConfig.from_json(path, n_bars=n_bars)
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_mode = payload.get(
            "sealed_test_ledger_mode",
            SealedTestLedgerMode.LOCAL_EXPLORATORY.value,
        )
        if not isinstance(raw_mode, str):
            raise ValueError("sealed_test_ledger_mode must be a string")
        try:
            mode = SealedTestLedgerMode(raw_mode)
        except ValueError as error:
            raise ValueError("sealed_test_ledger_mode is unsupported") from error
        return cls(
            workflow=base.workflow,
            candidates=base.candidates,
            minimum_selection_uplift=base.minimum_selection_uplift,
            minimum_selection_score=base.minimum_selection_score,
            minimum_seed_success_fraction=base.minimum_seed_success_fraction,
            minimum_worst_seed_uplift=base.minimum_worst_seed_uplift,
            maximum_seed_score_std=base.maximum_seed_score_std,
            maximum_selection_turnover_per_day=(
                base.maximum_selection_turnover_per_day
            ),
            maximum_selection_cost_fraction=base.maximum_selection_cost_fraction,
            maximum_selection_drawdown=base.maximum_selection_drawdown,
            checkpoint_finalists_per_seed=base.checkpoint_finalists_per_seed,
            execution_sensitivity=base.execution_sensitivity,
            signal_digest=base.signal_digest,
            schema_version=base.schema_version,
            sealed_test_ledger_mode=mode,
        )

    def digest_payload(self) -> dict[str, object]:
        payload = _base.MarketWalkForwardConfig.digest_payload(self)
        payload["sealed_test_ledger_mode"] = self.sealed_test_ledger_mode.value
        return payload


__all__ = [
    "ExecutionSensitivityConfig",
    "ExecutionSensitivityScenario",
    "MarketWalkForwardConfig",
    "NamedCandidateRun",
    "SealedTestLedgerMode",
]
