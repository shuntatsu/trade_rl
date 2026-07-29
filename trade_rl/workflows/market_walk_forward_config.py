"""Validated market walk-forward configuration with explicit ledger trust mode."""

from __future__ import annotations

import copy
import json
import math
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import trade_rl.workflows._market_walk_forward_config_base as _base

ExecutionSensitivityScenario = _base.ExecutionSensitivityScenario
NamedCandidateRun = _base.NamedCandidateRun

_STANDARD_EXECUTION_SCENARIOS = frozenset(
    {
        "nominal",
        "tick_2x",
        "lot_2x",
        "minimum_notional_2x",
        "joint_2x",
        "joint_5x",
    }
)


def _finite_number(value: object, *, field: str, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{field} must be a finite number")
    return resolved


def _boolean(value: object, *, field: str, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


@dataclass(frozen=True, slots=True)
class ExecutionSensitivityConfig(_base.ExecutionSensitivityConfig):
    """Canonical stress pack plus optional report-only extensions."""

    def __post_init__(self) -> None:
        names = tuple(item.name for item in self.scenarios)
        if len(set(names)) != len(names):
            raise ValueError("execution sensitivity scenario names must be unique")
        standard = tuple(
            item
            for item in self.scenarios
            if item.name in _STANDARD_EXECUTION_SCENARIOS
        )
        _base.ExecutionSensitivityConfig(
            scenarios=standard,
            required_scenario=self.required_scenario,
            minimum_selected_return=self.minimum_selected_return,
            minimum_baseline_uplift=self.minimum_baseline_uplift,
            maximum_drawdown=self.maximum_drawdown,
            schema_version=self.schema_version,
        )
        for scenario in self.scenarios:
            if (
                scenario.name not in _STANDARD_EXECUTION_SCENARIOS
                and not scenario.report_only
            ):
                raise ValueError(
                    "additional execution sensitivity scenarios must be report-only"
                )
        if self.scenarios:
            required = next(
                item for item in self.scenarios if item.name == self.required_scenario
            )
            if required.report_only:
                raise ValueError(
                    "required execution sensitivity scenario cannot be report-only"
                )


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


def _base_compatible_payload(expanded: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(expanded)
    sensitivity = payload.get("execution_sensitivity")
    if not isinstance(sensitivity, dict):
        return payload
    scenarios = sensitivity.get("scenarios")
    if not isinstance(scenarios, list):
        return payload
    sensitivity["scenarios"] = [
        scenario
        for scenario in scenarios
        if not isinstance(scenario, dict)
        or scenario.get("name") in _STANDARD_EXECUTION_SCENARIOS
    ]
    return payload


def _extended_execution_sensitivity(
    expanded: dict[str, Any],
    base: _base.ExecutionSensitivityConfig,
) -> ExecutionSensitivityConfig:
    sensitivity = expanded.get("execution_sensitivity")
    if not isinstance(sensitivity, dict):
        return ExecutionSensitivityConfig(
            scenarios=base.scenarios,
            required_scenario=base.required_scenario,
            minimum_selected_return=base.minimum_selected_return,
            minimum_baseline_uplift=base.minimum_baseline_uplift,
            maximum_drawdown=base.maximum_drawdown,
            schema_version=base.schema_version,
        )
    raw_scenarios = sensitivity.get("scenarios")
    if not isinstance(raw_scenarios, list):
        raise ValueError("execution_sensitivity.scenarios must be a list")
    additional: list[ExecutionSensitivityScenario] = []
    for index, raw_scenario in enumerate(raw_scenarios):
        if not isinstance(raw_scenario, dict):
            raise ValueError(
                f"execution_sensitivity.scenarios[{index}] must be a JSON object"
            )
        name = raw_scenario.get("name")
        if not isinstance(name, str):
            raise ValueError(
                f"execution_sensitivity.scenarios[{index}].name must be a string"
            )
        if name in _STANDARD_EXECUTION_SCENARIOS:
            continue
        field = f"execution_sensitivity.scenarios[{index}]"
        additional.append(
            ExecutionSensitivityScenario(
                name=name,
                tick_size_factor=_finite_number(
                    raw_scenario.get("tick_size_factor"),
                    field=f"{field}.tick_size_factor",
                    default=1.0,
                ),
                lot_size_factor=_finite_number(
                    raw_scenario.get("lot_size_factor"),
                    field=f"{field}.lot_size_factor",
                    default=1.0,
                ),
                minimum_notional_factor=_finite_number(
                    raw_scenario.get("minimum_notional_factor"),
                    field=f"{field}.minimum_notional_factor",
                    default=1.0,
                ),
                adverse_tick_rounding=_boolean(
                    raw_scenario.get("adverse_tick_rounding"),
                    field=f"{field}.adverse_tick_rounding",
                    default=True,
                ),
                report_only=_boolean(
                    raw_scenario.get("report_only"),
                    field=f"{field}.report_only",
                    default=False,
                ),
            )
        )
    return ExecutionSensitivityConfig(
        scenarios=base.scenarios + tuple(additional),
        required_scenario=base.required_scenario,
        minimum_selected_return=base.minimum_selected_return,
        minimum_baseline_uplift=base.minimum_baseline_uplift,
        maximum_drawdown=base.maximum_drawdown,
        schema_version=base.schema_version,
    )


class SealedTestLedgerMode(StrEnum):
    """Explicit trust semantics for outer-test access authorization."""

    LOCAL_EXPLORATORY = "local_exploratory"
    DURABLE_POSTGRES = "durable_postgres"


@dataclass(frozen=True, slots=True)
class MarketWalkForwardConfig(_base.MarketWalkForwardConfig):
    execution_sensitivity: ExecutionSensitivityConfig = ExecutionSensitivityConfig()
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
        base_payload = _base_compatible_payload(expanded)
        with tempfile.TemporaryDirectory(prefix="trade-rl-walk-forward-") as directory:
            expanded_path = Path(directory) / path.name
            expanded_path.write_text(
                json.dumps(base_payload, sort_keys=True),
                encoding="utf-8",
            )
            base = _base.MarketWalkForwardConfig.from_json(
                expanded_path,
                n_bars=n_bars,
            )
        sensitivity = _extended_execution_sensitivity(
            expanded, base.execution_sensitivity
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
            execution_sensitivity=sensitivity,
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
