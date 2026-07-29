"""Validated market walk-forward configuration with explicit ledger trust mode."""

from __future__ import annotations

import copy
import json
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import trade_rl.workflows._market_walk_forward_config_base as _base

ExecutionSensitivityConfig = _base.ExecutionSensitivityConfig
ExecutionSensitivityScenario = _base.ExecutionSensitivityScenario
NamedCandidateRun = _base.NamedCandidateRun


def _absolute_config_path(value: object, *, base: Path, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty path string")
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (base / path).resolve()
    return str(resolved)


def _normalize_run_artifact_paths(
    raw: object,
    *,
    base: Path,
    field: str,
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"{field} must be a JSON object")
    run = copy.deepcopy(raw)
    for name in ("alpha_artifact", "factor_artifact"):
        if run.get(name) is not None:
            run[name] = _absolute_config_path(
                run[name],
                base=base,
                field=f"{field}.{name}",
            )
    resume = run.get("resume_checkpoints")
    if resume is not None:
        if not isinstance(resume, dict):
            raise ValueError(f"{field}.resume_checkpoints must be a JSON object")
        run["resume_checkpoints"] = {
            key: _absolute_config_path(
                value,
                base=base,
                field=f"{field}.resume_checkpoints[{key!r}]",
            )
            for key, value in resume.items()
        }
    return run


def _expand_candidate_run_files(path: Path, payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("walk-forward config must be a JSON object")
    expanded = copy.deepcopy(payload)
    raw_candidates = expanded.get("candidates")
    if not isinstance(raw_candidates, list):
        return expanded
    for index, raw_candidate in enumerate(raw_candidates):
        if not isinstance(raw_candidate, dict):
            continue
        has_run = "run" in raw_candidate
        has_run_file = "run_file" in raw_candidate
        if has_run == has_run_file:
            raise ValueError(
                f"candidates[{index}] must define exactly one of run or run_file"
            )
        if has_run_file:
            run_path = Path(
                _absolute_config_path(
                    raw_candidate["run_file"],
                    base=path.parent,
                    field=f"candidates[{index}].run_file",
                )
            )
            raw_run = json.loads(run_path.read_text(encoding="utf-8"))
            raw_candidate["run"] = _normalize_run_artifact_paths(
                raw_run,
                base=run_path.parent,
                field=f"candidates[{index}].run",
            )
            del raw_candidate["run_file"]
        else:
            raw_candidate["run"] = _normalize_run_artifact_paths(
                raw_candidate["run"],
                base=path.parent,
                field=f"candidates[{index}].run",
            )
    return expanded


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
        payload = json.loads(path.read_text(encoding="utf-8"))
        expanded = _expand_candidate_run_files(path, payload)
        with tempfile.TemporaryDirectory(prefix="trade-rl-walk-forward-") as directory:
            expanded_path = Path(directory) / path.name
            expanded_path.write_text(
                json.dumps(expanded, sort_keys=True),
                encoding="utf-8",
            )
            base = _base.MarketWalkForwardConfig.from_json(
                expanded_path,
                n_bars=n_bars,
            )
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
