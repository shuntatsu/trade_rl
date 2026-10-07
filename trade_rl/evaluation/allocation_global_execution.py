"""Closed declared context and received/native facts for one global account.

These are software consistency records, not research validity or proof of
historical fitting, Dataset getter use or backend prediction tensors.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from hashlib import sha256
from typing import Any

import numpy as np

from trade_rl._validation import require_non_empty, require_sha256
from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.artifacts.verified_file import file_digest
from trade_rl.evaluation.allocation_cash_control import (
    GlobalAllocationCashControlResult,
    _cash_policy_digest_for_recipe,
    run_global_allocation_cash_control,
)
from trade_rl.evaluation.allocation_comparison import (
    AllocationCandidateKind,
    AllocationComparisonContract,
)
from trade_rl.evaluation.allocation_comparison_evidence import (
    allocation_business_objective_digest,
    allocation_comparison_scenario_digest,
    allocation_economic_clock_digest,
    allocation_fold_plan_digest,
)
from trade_rl.evaluation.allocation_nonrl_walk_forward import (
    GlobalNonRLAllocationResult,
    _carriers,
    _validate_global_control_roster,
    allocation_nonrl_rule_digest,
    run_global_nonrl_allocation,
)
from trade_rl.evaluation.allocation_scenario_identity import (
    allocation_candidate_recipe_digest,
)
from trade_rl.evaluation.evidence import ExecutionDiagnostics
from trade_rl.evaluation.rl_allocation.continuous_walk_forward import (
    AllocationFoldPolicy,
)
from trade_rl.evaluation.rl_allocation.env import AllocationTradingEnv
from trade_rl.evaluation.rl_allocation.fee_stress_admission import (
    FeeStressedGlobalAllocationResult,
    _comparison,
    run_fee_stressed_global_allocation,
)
from trade_rl.evaluation.rl_allocation.global_execution_context import (
    GlobalAllocationExecutionCollector,
    allocation_array_pin,
    allocation_execution_runtime,
    global_declared_source_digest,
    validate_global_collector,
)
from trade_rl.evaluation.rl_allocation.global_walk_forward import (
    GlobalAllocationWalkForwardResult,
    run_artifact_bound_global_allocation_walk_forward,
    validate_global_allocation_walk_forward,
)
from trade_rl.evaluation.rl_allocation.policy_admission import (
    AllocationFoldPolicyArtifact,
)
from trade_rl.evaluation.rl_allocation.transition_facts import (
    validate_allocation_execution_facts,
)
from trade_rl.evaluation.robustness.closed_trades import ClosedTradeDiagnostics
from trade_rl.evaluation.robustness.walk_forward.folds import (
    IndexRange,
    WalkForwardFold,
)
from trade_rl.evaluation.runs.provenance import build_candidate_run_provenance
from trade_rl.strategies.rl.allocation_artifact import read_allocation_policy_manifest
from trade_rl.strategies.rl.allocation_fee_stress import (
    allocation_policy_state_digest,
    retained_debt_economics_digest,
)
from trade_rl.strategies.rl.allocation_preprocessing import _native_json
from trade_rl.strategies.rl.allocation_recipe_validation import (
    validate_allocation_recipe,
)

_KINDS = {"nonrl", "cash", "residual_ppo", "direct_ppo"}
_COMMON = {
    "source_digest",
    "dataset_id",
    "account_id",
    "symbol_index",
    "range",
    "source_clocks_ns",
    "horizon_steps",
    "initial_capital",
    "fold_plan_digest",
    "fold_plan",
    "objective_digest",
    "clock_digest",
    "forecast_context_digest",
    "cost_beliefs_digest",
    "economics_digest",
    "risk_digest",
    "scenario",
    "implementation_digest",
    "runtime_environment_digest",
    "source_semantics",
}
_CELL = {
    "kind",
    "seed",
    "reset_seed",
    "original_recipe",
    "original_recipe_digest",
    "original_candidate_digest",
    "runtime",
    "runtime_recipe_digest",
    "artifacts",
    "control_policy_digest",
    "fee",
}
_RECEIPT = {
    "schema",
    "plan",
    "common_digest",
    "cell_plan_digest",
    "status",
    "reset_count",
    "opening_state_digest",
    "closing_state_digest",
    "rows",
    "segments",
    "native_terminal",
    "termination_reason",
    "economic_stop_consistent",
    "legacy_fee_receipt_digest",
    "failure",
    "model_states_before",
    "model_states_after",
    "model_state_failures_after",
    "provenance_before",
    "provenance_after",
}
_ROW = {
    "fold_index",
    "decision_index",
    "decision_time_ns",
    "policy_digest",
    "runtime_recipe_digest",
    "before_state_digest",
    "input",
    "call_outcome",
    "native",
    "native_validation_failure",
    "after_state_digest",
    "terminated",
    "truncated",
}
_SOURCE_SEMANTICS = "declared_finite_envelope_not_getter_usage_or_origin"


def _closed(value: object, keys: set[str], field: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"{field} must contain exactly its declared fields")
    return value


def _digest(value: object, field: str) -> None:
    if type(value) is not str:
        raise ValueError(f"{field} must be a native SHA-256 string")
    require_sha256(value, field=field)


def _fold_payload(folds: tuple[WalkForwardFold, ...]) -> list[dict[str, object]]:
    return [
        {
            "fold_index": f.fold_index,
            "train": [f.train.start, f.train.stop],
            "checkpoint_validation": [
                f.checkpoint_validation.start,
                f.checkpoint_validation.stop,
            ],
            "configuration_selection": [
                f.configuration_selection.start,
                f.configuration_selection.stop,
            ],
            "test": [f.test.start, f.test.stop],
            "purge_bars": f.purge_bars,
        }
        for f in folds
    ]


def _declared_folds(common: dict[str, Any]) -> tuple[WalkForwardFold, ...]:
    if type(common["fold_plan"]) is not list or not common["fold_plan"]:
        raise ValueError("global plan requires its complete fold roster")
    folds = []
    for raw in common["fold_plan"]:
        _closed(
            raw,
            {
                "fold_index",
                "train",
                "checkpoint_validation",
                "configuration_selection",
                "test",
                "purge_bars",
            },
            "global declared fold",
        )
        for key in ("fold_index", "purge_bars"):
            if type(raw[key]) is not int or raw[key] < 0:
                raise ValueError(
                    "global fold scalars must be native non-negative integers"
                )
        intervals = []
        for key in (
            "train",
            "checkpoint_validation",
            "configuration_selection",
            "test",
        ):
            value = raw[key]
            if (
                type(value) is not list
                or len(value) != 2
                or any(type(x) is not int for x in value)
            ):
                raise ValueError("global fold ranges must be native integer pairs")
            intervals.append(IndexRange(*value))
        folds.append(
            WalkForwardFold(
                raw["fold_index"],
                intervals[0],
                intervals[1],
                intervals[2],
                intervals[3],
                raw["purge_bars"],
            )
        )
    result = tuple(folds)
    if (
        allocation_fold_plan_digest(result) != common["fold_plan_digest"]
        or result[0].test.start != common["range"][0]
        or result[-1].test.stop != common["range"][1]
        or any(a.test.stop != b.test.start for a, b in zip(result, result[1:]))
    ):
        raise ValueError("global fold roster must cover the one declared horizon")
    return result


def _plan(value: object) -> dict[str, Any]:
    _native_json(value)
    raw = _closed(value, {"schema", "common", "cell"}, "global execution plan")
    if raw["schema"] != "allocation_global_execution_plan_v1":
        raise ValueError("unknown global execution plan schema")
    common, cell = (
        _closed(raw["common"], _COMMON, "common global context"),
        _closed(raw["cell"], _CELL, "global cell plan"),
    )
    for key, item in common.items():
        if key.endswith("digest") or key == "dataset_id":
            _digest(item, key)
    if common["source_semantics"] != _SOURCE_SEMANTICS:
        raise ValueError("declared source must not claim observed use or origin")
    require_non_empty(common["account_id"], field="account_id")
    require_non_empty(common["scenario"], field="scenario")
    if (
        type(common["symbol_index"]) is not int
        or common["symbol_index"] < 0
        or type(common["horizon_steps"]) is not int
        or common["horizon_steps"] <= 0
        or type(common["initial_capital"]) not in (int, float)
        or common["initial_capital"] <= 0
        or type(common["range"]) is not list
        or len(common["range"]) != 2
        or any(type(x) is not int for x in common["range"])
        or common["range"][0] < 0
        or common["range"][1] - common["range"][0] != common["horizon_steps"]
        or type(common["source_clocks_ns"]) is not list
        or len(common["source_clocks_ns"]) != common["horizon_steps"] + 1
        or any(type(x) is not int for x in common["source_clocks_ns"])
        or any(
            b <= a
            for a, b in zip(common["source_clocks_ns"], common["source_clocks_ns"][1:])
        )
    ):
        raise ValueError("global plan requires complete native H/C/range/source clocks")
    folds = _declared_folds(common)
    if type(cell["kind"]) is not str or cell["kind"] not in _KINDS:
        raise ValueError("global execution kind is not declared")
    control = cell["kind"] in {"nonrl", "cash"}
    if (control and cell["seed"] is not None) or (
        not control and (type(cell["seed"]) is not int or cell["seed"] < 0)
    ):
        raise ValueError("global cell requires native RL seed or seedless control")
    if (
        (
            cell["kind"] == "cash"
            and (type(cell["reset_seed"]) is not int or cell["reset_seed"] != 0)
        )
        or (cell["kind"] == "nonrl" and cell["reset_seed"] is not None)
        or (
            not control
            and (type(cell["reset_seed"]) is not int or cell["reset_seed"] < 0)
        )
    ):
        raise ValueError("global actual reset seed differs from closed declaration")
    for key in (
        "original_recipe_digest",
        "original_candidate_digest",
        "runtime_recipe_digest",
    ):
        _digest(cell[key], key)
    validate_allocation_recipe(cell["original_recipe"])
    if (
        content_digest(cell["original_recipe"]) != cell["original_recipe_digest"]
        or allocation_candidate_recipe_digest(cell["original_recipe"])
        != cell["original_candidate_digest"]
    ):
        raise ValueError("original global cell recipe differs from its pins")
    runtime = _closed(
        cell["runtime"],
        {"recipe", "cost_payload", "source_context_digest"},
        "global actual runtime",
    )
    validate_allocation_recipe(runtime["recipe"])
    if runtime["recipe"]["schema"] not in {
        "allocation_ppo_recipe_v2",
        "allocation_ppo_recipe_v3",
    }:
        raise ValueError("global receipt requires the native global observation recipe")
    if (
        runtime["recipe"]["action"]["mode"]
        != ("residual" if cell["kind"] in {"nonrl", "residual_ppo"} else "direct")
        or runtime["recipe"]["runtime_profile"]["initial_capital"]
        != common["initial_capital"]
        or runtime["recipe"]["runtime_profile"]["economics_digest"]
        != common["economics_digest"]
        or runtime["recipe"]["runtime_profile"]["risk_digest"] != common["risk_digest"]
    ):
        raise ValueError("global runtime recipe differs from shared native context")
    if (
        retained_debt_economics_digest(runtime["cost_payload"])
        != common["economics_digest"]
    ):
        raise ValueError(
            "global actual cost payload differs from native recipe economics"
        )
    profile = runtime["recipe"]["runtime_profile"]
    if (
        profile["economic_horizon_seconds"]
        != common["horizon_steps"] * profile["decision_interval_seconds"]
        or runtime["recipe"]["observation"]["episode_steps"] != common["horizon_steps"]
        or any(
            b - a != profile["decision_interval_seconds"] * 10**9
            for a, b in zip(common["source_clocks_ns"], common["source_clocks_ns"][1:])
        )
    ):
        raise ValueError("global source clocks/H differ from the native runtime recipe")
    _digest(runtime["source_context_digest"], "cell source context")
    if content_digest(runtime["recipe"]) != cell["runtime_recipe_digest"]:
        raise ValueError("global runtime recipe differs from its pin")
    if (
        type(cell["artifacts"]) is not list
        or (control and cell["artifacts"])
        or (not control and not cell["artifacts"])
    ):
        raise ValueError("global cell original artifact roster differs from its kind")
    for artifact in cell["artifacts"]:
        _closed(
            artifact,
            {"fold_index", "policy_digest", "recipe_digest", "policy_sha256"},
            "original artifact",
        )
        if type(artifact["fold_index"]) is not int or artifact["fold_index"] < 0:
            raise ValueError("artifact fold must be native non-negative integer")
        for key in ("policy_digest", "recipe_digest", "policy_sha256"):
            _digest(artifact[key], key)
        if artifact["recipe_digest"] != cell["original_recipe_digest"]:
            raise ValueError("original artifact recipe differs from cell")
    if not control and [a["fold_index"] for a in cell["artifacts"]] != [
        f.fold_index for f in folds
    ]:
        raise ValueError("global artifact roster differs from complete fold plan")
    if control:
        expected = (
            allocation_nonrl_rule_digest(cell["original_candidate_digest"])
            if cell["kind"] == "nonrl"
            else _cash_policy_digest_for_recipe(cell["original_candidate_digest"])
        )
        if cell["control_policy_digest"] != expected or cell["fee"] is not None:
            raise ValueError("global control identity differs from its fixed rule")
    elif cell["control_policy_digest"] is not None:
        raise ValueError("RL cell cannot claim a control identity")
    if cell["fee"] is not None:
        fee = _closed(cell["fee"], {"factor", "contract_digest"}, "fee intervention")
        _digest(fee["contract_digest"], "fee contract")
        if cell["reset_seed"] != cell["seed"]:
            raise ValueError("fee original model seed differs from comparison seed")
        if type(fee["factor"]) not in (float, int) or fee["factor"] <= 1:
            raise ValueError("fee intervention requires factor greater than one")
    return raw


@dataclass(frozen=True, slots=True)
class GlobalAllocationExecutionPlan:
    _bytes: bytes

    def __post_init__(self) -> None:
        if type(self._bytes) is not bytes:
            raise ValueError("global plan requires detached canonical bytes")
        raw = _plan(json.loads(self._bytes))
        if canonical_json_bytes(raw) != self._bytes:
            raise ValueError("global execution plan is not canonical")

    @property
    def payload(self) -> dict[str, Any]:
        return json.loads(self._bytes)

    @property
    def common_digest(self) -> str:
        return content_digest(self.payload["common"])

    @property
    def cell_plan_digest(self) -> str:
        return content_digest(self.payload["cell"])

    @classmethod
    def from_payload(cls, value: object) -> GlobalAllocationExecutionPlan:
        return cls(canonical_json_bytes(_plan(value)))


def _provenance(common: dict[str, Any]) -> dict[str, object]:
    actual = build_candidate_run_provenance()
    if (
        actual["implementation_digest"] != common["implementation_digest"]
        or actual["runtime_environment_digest"] != common["runtime_environment_digest"]
    ):
        raise ValueError(
            "global implementation/runtime provenance differs from frozen pins"
        )
    return actual


def _artifact_roster(
    env: AllocationTradingEnv,
    artifacts: tuple[AllocationFoldPolicyArtifact, ...],
    original_recipe: str,
    seed: int,
) -> list[dict[str, Any]]:
    if (
        type(artifacts) is not tuple
        or not artifacts
        or any(type(a) is not AllocationFoldPolicyArtifact for a in artifacts)
    ):
        raise ValueError("global cell requires native immutable policy artifacts")
    cutoff = int(
        env.dataset.timestamps[env.start_index].astype("datetime64[ns]").astype("int64")
    )
    manifests = tuple(
        read_allocation_policy_manifest(
            a.bundle_root,
            expected_digest=a.expected_digest,
            expected_recipe_digest=original_recipe,
            training_cutoff_ns=cutoff,
        )
        for a in artifacts
    )
    result = []
    for artifact, manifest in zip(artifacts, manifests, strict=True):
        if (
            artifact.expected_recipe_digest != original_recipe
            or manifest["training"]["seed"] != seed
        ):
            raise ValueError("global original artifact recipe/seed differs from cell")
        if (
            file_digest(
                artifact.bundle_root / "policy.zip",
                field="global original allocation policy",
            )
            != manifest["policy_sha256"]
        ):
            raise ValueError(
                "global original allocation policy archive digest mismatch"
            )
        result.append(
            {
                "fold_index": artifact.fold_index,
                "policy_digest": artifact.expected_digest,
                "recipe_digest": original_recipe,
                "policy_sha256": manifest["policy_sha256"],
            }
        )
    return result


def declare_global_allocation_execution(
    folds: tuple[WalkForwardFold, ...],
    env: AllocationTradingEnv,
    *,
    kind: str,
    scenario: str,
    expected_implementation_digest: str,
    expected_runtime_digest: str,
    seed: int | None = None,
    reset_seed: int | None = None,
    artifacts: tuple[AllocationFoldPolicyArtifact, ...] = (),
    base_env: AllocationTradingEnv | None = None,
    contract: AllocationComparisonContract | None = None,
    fee_factor: float | None = None,
) -> GlobalAllocationExecutionPlan:
    if type(kind) is not str or kind not in _KINDS:
        raise ValueError("global execution kind is not declared")
    control = kind in {"nonrl", "cash"}
    if (
        control
        and (
            seed is not None
            or reset_seed is not None
            or artifacts
            or base_env is not None
            or contract is not None
            or fee_factor is not None
        )
    ) or (not control and (type(seed) is not int or seed < 0)):
        raise ValueError("global seedless control or RL declaration differs from kind")
    actual_reset_seed = (
        (0 if kind == "cash" else None)
        if control
        else (seed if reset_seed is None else reset_seed)
    )
    if actual_reset_seed is not None and (
        type(actual_reset_seed) is not int or actual_reset_seed < 0
    ):
        raise ValueError(
            "global reset seed requires a native non-negative integer or None"
        )
    mode = "residual" if kind in {"nonrl", "residual_ppo"} else "direct"
    _carriers((env,), required_action_mode=mode)
    runtime = allocation_execution_runtime(env)
    original = env if base_env is None else base_env
    allocation_execution_runtime(original)
    candidate = allocation_candidate_recipe_digest(original.recipe)
    control_digest = (
        (
            allocation_nonrl_rule_digest(candidate)
            if kind == "nonrl"
            else _cash_policy_digest_for_recipe(candidate)
        )
        if control
        else None
    )
    roster: list[dict[str, Any]] = []
    if not control:
        if type(seed) is not int:
            raise ValueError("global RL cell requires a native seed")
        roster = _artifact_roster(env, artifacts, original.recipe_digest, seed)
    placeholders = tuple(
        AllocationFoldPolicy(
            control_digest or a["policy_digest"], env.recipe_digest, lambda *_: 0
        )
        for a in (roster if roster else [{} for _ in folds])
    )
    validate_global_allocation_walk_forward(folds, env, placeholders)
    if roster and [a["fold_index"] for a in roster] != [f.fold_index for f in folds]:
        raise ValueError("global original artifact roster differs from segments")
    if base_env is not None:
        if contract is None or fee_factor is None:
            raise ValueError(
                "fee declaration requires its original comparison contract"
            )
        _comparison(
            contract,
            AllocationCandidateKind.RESIDUAL_PPO
            if mode == "residual"
            else AllocationCandidateKind.DIRECT_PPO,
            scenario,
            folds,
            base_env,
            env,
        )
    elif contract is not None or fee_factor is not None:
        raise ValueError("fee declaration requires an original base environment")
    common = {
        "source_digest": global_declared_source_digest(env),
        "dataset_id": env.dataset.dataset_id,
        "account_id": env.account_id,
        "symbol_index": env.symbol_index,
        "range": [env.start_index, env.stop_index],
        "source_clocks_ns": env.dataset.timestamps[env.start_index : env.stop_index + 1]
        .astype("datetime64[ns]")
        .astype("int64")
        .tolist(),
        "horizon_steps": env.stop_index - env.start_index,
        "initial_capital": env.initial_capital,
        "fold_plan_digest": allocation_fold_plan_digest(folds),
        "fold_plan": _fold_payload(folds),
        "objective_digest": allocation_business_objective_digest(env.bound.objective),
        "clock_digest": allocation_economic_clock_digest(env.bound.clock),
        "forecast_context_digest": env.stream.digest,
        "cost_beliefs_digest": content_digest(
            [env._estimates[k].payload() for k in sorted(env._estimates)]
        ),
        "economics_digest": env.executor.execution_policy_digest,
        "risk_digest": content_digest(env.risk_config),
        "scenario": scenario,
        "implementation_digest": expected_implementation_digest,
        "runtime_environment_digest": expected_runtime_digest,
        "source_semantics": _SOURCE_SEMANTICS,
    }
    _provenance(common)
    return GlobalAllocationExecutionPlan.from_payload(
        {
            "schema": "allocation_global_execution_plan_v1",
            "common": common,
            "cell": {
                "kind": kind,
                "seed": seed,
                "reset_seed": actual_reset_seed,
                "original_recipe": original.recipe,
                "original_recipe_digest": original.recipe_digest,
                "original_candidate_digest": candidate,
                "runtime": runtime,
                "runtime_recipe_digest": env.recipe_digest,
                "artifacts": roster,
                "control_policy_digest": control_digest,
                "fee": None
                if base_env is None
                else {"factor": fee_factor, "contract_digest": contract.digest},  # type: ignore[union-attr]
            },
        }
    )


def _check_failure(value: object, *, outcome: bool = False) -> None:
    keys = {"error_type", "message"} | ({"status"} if outcome else set())
    raw = _closed(value, keys, "original failure")
    require_non_empty(raw["error_type"], field="error_type")
    if type(raw["message"]) is not str:
        raise ValueError("failure message must be native text")


def _check_provenance(value: object, common: dict[str, Any], *, bind: bool) -> None:
    raw = _closed(
        value,
        {
            "schema_version",
            "implementation",
            "implementation_digest",
            "runtime_environment",
            "runtime_environment_digest",
            "research_context_digest",
        },
        "global provenance",
    )
    if (
        raw["schema_version"] != "candidate_run_provenance_v1"
        or raw["research_context_digest"] is not None
        or content_digest(raw["implementation"]) != raw["implementation_digest"]
        or content_digest(raw["runtime_environment"])
        != raw["runtime_environment_digest"]
    ):
        raise ValueError("global provenance contents differ from original pins")
    if bind and any(
        raw[k] != common[k]
        for k in ("implementation_digest", "runtime_environment_digest")
    ):
        raise ValueError("global provenance differs from frozen common context")


def _check_model_states(raw: dict[str, Any], count: int) -> None:
    before, after, failures = (
        raw["model_states_before"],
        raw["model_states_after"],
        raw["model_state_failures_after"],
    )
    if (
        type(before) is not list
        or type(after) is not list
        or type(failures) is not list
        or len(before) > count
        or len(after) > count
    ):
        raise ValueError("observed model state roster differs from original artifacts")
    for pin in before:
        _digest(pin, "original model state")
    unavailable = []
    for index, pin in enumerate(after):
        if pin is None:
            unavailable.append(index)
        else:
            _digest(pin, "final model state")
    indices = []
    for failure in failures:
        _closed(failure, {"index", "error_type", "message"}, "unavailable model state")
        if type(failure["index"]) is not int:
            raise ValueError("model state failure index must be native")
        _check_failure({k: failure[k] for k in ("error_type", "message")})
        indices.append(failure["index"])
    if indices != unavailable or (
        raw["status"] == "completed"
        and (len(before) != count or before != after or failures)
    ):
        raise ValueError(
            "completed model state requires complete unchanged original roster"
        )


def _check_segments(raw: dict[str, Any], folds: tuple[WalkForwardFold, ...]) -> None:
    segments = raw["segments"]
    if type(segments) is not list or len(segments) != len(folds):
        raise ValueError("completed receipt requires every native segment")
    for segment, fold in zip(segments, folds, strict=True):
        _closed(
            segment,
            {
                "fold_index",
                "range",
                "opening_state_digest",
                "closing_state_digest",
                "returns",
                "diagnostics",
            },
            "native global segment",
        )
        rows = [r for r in raw["rows"] if r["fold_index"] == fold.fold_index]
        if (
            segment["fold_index"] != fold.fold_index
            or segment["range"] != [fold.test.start, fold.test.stop]
            or segment["opening_state_digest"] != rows[0]["before_state_digest"]
            or segment["closing_state_digest"] != rows[-1]["after_state_digest"]
            or segment["returns"]
            != [r["native"]["execution"]["interval_net_return"] for r in rows]
        ):
            raise ValueError(
                "native segments differ from actual received/state/return coverage"
            )
        diag = _closed(
            segment["diagnostics"],
            set(ExecutionDiagnostics().digest_payload()),
            "native segment diagnostics",
        )
        closed = _closed(
            diag["closed_trades"],
            set(ClosedTradeDiagnostics().digest_payload()),
            "closed trade diagnostics",
        )
        trade = ClosedTradeDiagnostics(
            **{f.name: closed[f.name] for f in fields(ClosedTradeDiagnostics)}
        )
        if canonical_json_bytes(trade.digest_payload()) != canonical_json_bytes(closed):
            raise ValueError("closed trade diagnostics differ")
        diagnostics = ExecutionDiagnostics(
            **{
                k: v
                for k, v in diag.items()
                if k not in {"closed_trades", "termination_reasons"}
            },
            closed_trades=trade,
            termination_reasons=tuple(diag["termination_reasons"]),
        )
        if canonical_json_bytes(diagnostics.digest_payload()) != canonical_json_bytes(
            diag
        ):
            raise ValueError("native diagnostics differ")


def _check_invalid_return(value: object) -> None:
    raw = _closed(
        value,
        {
            "status",
            "scalar_type",
            "numpy_dtype",
            "encoding",
            "value",
            "validation_failure",
        },
        "invalid public return",
    )
    kinds = {
        "python_bool",
        "python_int",
        "python_float",
        "numpy_bool",
        "numpy_int",
        "numpy_float",
        "unsupported",
    }
    if raw["scalar_type"] not in kinds:
        raise ValueError("invalid return must name its closed scalar type")
    _check_failure(raw["validation_failure"])
    if (
        raw["validation_failure"]["error_type"] != "ValueError"
        or raw["validation_failure"]["message"]
        != "global public action result must be an integer within 0..3"
    ):
        raise ValueError("invalid public return differs from native action rejection")
    kind = raw["scalar_type"]
    if kind.startswith("numpy_"):
        if type(raw["numpy_dtype"]) is not str:
            raise ValueError("numpy invalid return requires its scalar dtype")
        dtype = np.dtype(raw["numpy_dtype"])
        if dtype.kind not in (
            {"b"}
            if kind == "numpy_bool"
            else {"i", "u"}
            if kind == "numpy_int"
            else {"f"}
        ) or dtype.itemsize not in (1, 2, 4, 8):
            raise ValueError("invalid return scalar dtype differs from its type")
    elif raw["numpy_dtype"] is not None:
        raise ValueError("native/unsupported return cannot invent numpy dtype")
    encoding = raw["encoding"]
    if encoding == "native_scalar":
        expected = (
            bool if kind.endswith("bool") else int if kind.endswith("int") else float
        )
        if (
            kind == "unsupported"
            or type(raw["value"]) is not expected
            or (expected is int and 0 <= raw["value"] <= 3)
        ):
            raise ValueError(
                "invalid scalar return cannot encode an admissible integer"
            )
    elif encoding == "unavailable_nonfinite":
        if kind not in {"python_float", "numpy_float"} or raw["value"] is not None:
            raise ValueError(
                "nonfinite return must explicitly retain unavailable scalar value"
            )
    elif encoding == "unavailable_unsupported":
        if kind != "unsupported" or raw["value"] is not None:
            raise ValueError(
                "unsupported return must explicitly retain unavailable encoding"
            )
    else:
        raise ValueError("unknown invalid-return encoding")


def _receipt(value: object) -> dict[str, Any]:
    _native_json(value)
    raw = _closed(value, _RECEIPT, "global execution receipt")
    if raw["schema"] != "allocation_global_execution_receipt_v1":
        raise ValueError("unknown global execution receipt schema")
    plan = GlobalAllocationExecutionPlan.from_payload(raw["plan"])
    common, cell = plan.payload["common"], plan.payload["cell"]
    if (
        raw["common_digest"] != plan.common_digest
        or raw["cell_plan_digest"] != plan.cell_plan_digest
    ):
        raise ValueError("global receipt differs from plan digests")
    if (
        raw["status"] not in {"completed", "economic_stop", "integrity_failure"}
        or type(raw["reset_count"]) is not int
        or not 0 <= raw["reset_count"] <= 1
    ):
        raise ValueError("global receipt has invalid completion/reset state")
    if type(raw["rows"]) is not list or len(raw["rows"]) > common["horizon_steps"]:
        raise ValueError("global receipt exceeds its declared horizon")
    folds = _declared_folds(common)
    for field in ("opening_state_digest", "closing_state_digest"):
        if raw[field] is not None:
            _digest(raw[field], field)
    if type(raw["native_terminal"]) is not bool or (
        raw["rows"] and raw["reset_count"] != 1
    ):
        raise ValueError("global receipt terminal/reset must reflect actual coverage")
    _check_provenance(raw["provenance_before"], common, bind=True)
    _check_provenance(
        raw["provenance_after"], common, bind=raw["status"] == "completed"
    )
    _check_model_states(raw, len(cell["artifacts"]))
    if type(raw["economic_stop_consistent"]) is not bool or raw[
        "economic_stop_consistent"
    ] != (raw["status"] == "economic_stop"):
        raise ValueError(
            "economic stop must follow a consistent native after-step check"
        )
    fee_digest = raw["legacy_fee_receipt_digest"]
    if fee_digest is not None:
        _digest(fee_digest, "legacy fee receipt")
        if cell["fee"] is None:
            raise ValueError("nonfee route cannot claim a legacy fee receipt")
    elif raw["status"] == "completed" and cell["fee"] is not None:
        raise ValueError(
            "completed fee execution requires original returned receipt bytes"
        )

    previous = raw["opening_state_digest"]
    for offset, item in enumerate(raw["rows"]):
        row = _closed(item, _ROW, "received global row")
        expected_fold = next(
            f
            for f in folds
            if f.test.start <= common["range"][0] + offset < f.test.stop
        )
        roster_index = folds.index(expected_fold)
        expected_policy = (
            cell["control_policy_digest"]
            if cell["kind"] in {"nonrl", "cash"}
            else cell["artifacts"][roster_index]["policy_digest"]
        )
        _digest(row["before_state_digest"], "received opening state")
        if row["after_state_digest"] is not None:
            _digest(row["after_state_digest"], "received closing state")
        if (
            type(row["fold_index"]) is not int
            or row["fold_index"] != expected_fold.fold_index
            or row["policy_digest"] != expected_policy
        ):
            raise ValueError(
                "received global row differs from original fold/policy roster"
            )
        if (
            type(row["decision_index"]) is not int
            or type(row["decision_time_ns"]) is not int
            or row["decision_index"] != common["range"][0] + offset
            or row["decision_time_ns"] != common["source_clocks_ns"][offset]
            or row["runtime_recipe_digest"] != cell["runtime_recipe_digest"]
        ):
            raise ValueError("received global row clock/recipe differs")
        if previous is not None and row["before_state_digest"] != previous:
            raise ValueError("received global account state chain differs")
        input_pin = _closed(
            row["input"],
            {"seam", "dtype", "shape", "sha256", "hex"},
            "received action input",
        )
        if input_pin["seam"] != "public_policy_action_argument":
            raise ValueError("global input must name its actual received public seam")
        values = np.frombuffer(
            bytes.fromhex(input_pin["hex"]), dtype=np.dtype(input_pin["dtype"])
        )
        if (
            values.dtype != np.dtype(np.float32)
            or input_pin["shape"]
            != [len(cell["runtime"]["recipe"]["observation"]["fields"])]
            or input_pin["shape"] != [len(values)]
            or not np.isfinite(values).all()
            or allocation_array_pin(values)["sha256"] != input_pin["sha256"]
        ):
            raise ValueError(
                "global received input bytes differ from finite native vector"
            )
        outcome = row["call_outcome"]
        if outcome is not None:
            if type(outcome) is not dict:
                raise ValueError("public outcome must be a native mapping")
            if outcome.get("status") == "returned":
                _closed(outcome, {"status", "action"}, "public returned action")
                if (
                    type(outcome["action"]) is not int
                    or not 0 <= outcome["action"] <= 3
                ):
                    raise ValueError("public action must be native code within 0..3")
                if cell["kind"] in {"nonrl", "cash"} and outcome["action"] != (
                    2 if cell["kind"] == "nonrl" else 0
                ):
                    raise ValueError("control action differs from fixed native rule")
            elif outcome.get("status") == "invalid_return":
                _check_invalid_return(outcome)
            elif outcome.get("status") == "raised":
                _closed(
                    outcome,
                    {"status", "error_type", "message"},
                    "public action exception",
                )
                _check_failure(outcome, outcome=True)
            else:
                raise ValueError("unknown public call outcome")
        if row["native"] is not None:
            if outcome is None or outcome["status"] != "returned":
                raise ValueError(
                    "native event requires an actually returned public action"
                )
            native = row["native"]
            rejected = row["native_validation_failure"]
            if rejected is None:
                validate_allocation_execution_facts(
                    native,
                    cell["runtime"]["recipe"],
                    dataset_id=common["dataset_id"],
                    action_code=outcome["action"],
                )
                if (
                    native["decision_index"] != row["decision_index"]
                    or native["processing_time_ns"]
                    != common["source_clocks_ns"][offset + 1]
                    or native["proposal"]["decision"]["baseline"]["context"][
                        "state_digest"
                    ]
                    != row["before_state_digest"]
                ):
                    raise ValueError(
                        "native event differs from received decision/account"
                    )
            else:
                _check_failure(rejected)
                if (
                    raw["status"] != "integrity_failure"
                    or offset != len(raw["rows"]) - 1
                    or row["after_state_digest"] is not None
                    or row["terminated"] is not None
                    or row["truncated"] is not None
                ):
                    raise ValueError(
                        "rejected native facts cannot claim completed suffix"
                    )
                try:
                    validate_allocation_execution_facts(
                        native,
                        cell["runtime"]["recipe"],
                        dataset_id=common["dataset_id"],
                        action_code=outcome["action"],
                    )
                except ValueError as rejected_error:
                    if rejected != {
                        "error_type": type(rejected_error).__name__,
                        "message": str(rejected_error),
                    }:
                        raise ValueError(
                            "rejected native failure differs from original reader"
                        ) from rejected_error
                else:
                    raise ValueError("native rejection requires actual reader failure")
        elif row["native_validation_failure"] is not None:
            raise ValueError(
                "native failure requires its actual detached rejected facts"
            )
        if raw["status"] == "completed" and (
            type(row["terminated"]) is not bool
            or row["terminated"] != (offset + 1 == common["horizon_steps"])
            or row["truncated"] is not False
        ):
            raise ValueError("completed rows require actual native terminal geometry")
        if row["native"] is None and (
            row["after_state_digest"] is not None
            or row["terminated"] is not None
            or row["truncated"] is not None
            or offset != len(raw["rows"]) - 1
        ):
            raise ValueError("unexecuted call cannot claim a native suffix")
        if offset < len(raw["rows"]) - 1 and (
            row["native"] is None
            or row["native_validation_failure"] is not None
            or row["after_state_digest"] is None
            or row["terminated"] is not False
            or row["truncated"] is not False
        ):
            raise ValueError(
                "observed prefix requires real contiguous live native states"
            )
        previous = row["after_state_digest"]
    if raw["closing_state_digest"] != previous:
        raise ValueError("global closing state differs from actual available suffix")
    if raw["status"] == "completed":
        if (
            raw["reset_count"] != 1
            or raw["opening_state_digest"] is None
            or not raw["rows"]
            or raw["opening_state_digest"] != raw["rows"][0]["before_state_digest"]
            or len(raw["rows"]) != common["horizon_steps"]
            or any(
                r["native"] is None
                or r["native_validation_failure"] is not None
                or r["after_state_digest"] is None
                or r["call_outcome"] is None
                for r in raw["rows"]
            )
            or raw["native_terminal"] is not True
            or raw["termination_reason"] is not None
            or raw["failure"] is not None
            or raw["closing_state_digest"] != previous
            or raw["model_states_before"] != raw["model_states_after"]
        ):
            raise ValueError(
                "global completion requires full actual H/terminal/state consistency"
            )
        _check_segments(raw, folds)
    elif raw["failure"] is None:
        raise ValueError("partial global receipt must retain its failure")
    else:
        _check_failure(raw["failure"])
        if raw["segments"]:
            if len(raw["rows"]) != common["horizon_steps"] or any(
                r["native"] is None for r in raw["rows"]
            ):
                raise ValueError("partial segments require actual full native coverage")
            _check_segments(raw, folds)
        if raw["status"] == "economic_stop":
            if (
                not raw["rows"]
                or raw["rows"][-1]["native"] is None
                or raw["termination_reason"]
                != raw["rows"][-1]["native"]["book"]["termination_reason"]
                or raw["termination_reason"] is None
                or raw["rows"][-1]["native_validation_failure"] is not None
            ):
                raise ValueError(
                    "economic stop requires the actual native signed prefix/reason"
                )
        elif raw["termination_reason"] is not None:
            if (
                not raw["rows"]
                or raw["rows"][-1]["native"] is None
                or raw["rows"][-1]["native_validation_failure"] is not None
                or raw["rows"][-1]["native"]["book"]["termination_reason"]
                != raw["termination_reason"]
            ):
                raise ValueError(
                    "integrity failure reason must come from its actual native signed event"
                )
    return raw


@dataclass(frozen=True, slots=True)
class GlobalAllocationExecutionReceipt:
    _bytes: bytes

    def __post_init__(self) -> None:
        if (
            type(self._bytes) is not bytes
            or canonical_json_bytes(_receipt(json.loads(self._bytes))) != self._bytes
        ):
            raise ValueError("global receipt requires canonical detached bytes")

    @property
    def payload(self) -> dict[str, Any]:
        return json.loads(self._bytes)

    @property
    def digest(self) -> str:
        return content_digest(self.payload)

    @classmethod
    def from_payload(cls, value: object) -> GlobalAllocationExecutionReceipt:
        return cls(canonical_json_bytes(_receipt(value)))


def _check_legacy_fee_result(
    result: FeeStressedGlobalAllocationResult, raw: dict[str, Any]
) -> None:
    common, cell = raw["plan"]["common"], raw["plan"]["cell"]
    legacy = _closed(
        result.receipt,
        {
            "schema",
            "contract_digest",
            "scenario",
            "scenario_digest",
            "candidate",
            "seed",
            "base_recipe_digest",
            "runtime_recipe_digest",
            "runtime",
            "binding_digests",
            "original_policy_digests",
            "original_policy_sha256",
            "model_states_before",
            "model_states_after",
            "fold_plan_digest",
            "actor_rows",
            "opening_state_digest",
            "closing_state_digest",
            "native_execution_summary",
            "source_semantics",
        },
        "original fee route receipt",
    )
    if (
        type(result._receipt) is not bytes
        or canonical_json_bytes(legacy) != result._receipt
        or sha256(result._receipt).hexdigest() != raw["legacy_fee_receipt_digest"]
    ):
        raise ValueError(
            "original fee route receipt bytes differ from observed execution"
        )
    expected = {
        "schema": "allocation_fee_global_execution_receipt_v1",
        "contract_digest": cell["fee"]["contract_digest"],
        "scenario": common["scenario"],
        "scenario_digest": allocation_comparison_scenario_digest(
            name=common["scenario"],
            dataset_id=common["dataset_id"],
            forecast_context_digest=common["forecast_context_digest"],
            economics_digest=common["economics_digest"],
            risk_digest=common["risk_digest"],
        ),
        "candidate": cell["kind"],
        "seed": cell["reset_seed"],
        "base_recipe_digest": cell["original_recipe_digest"],
        "runtime_recipe_digest": cell["runtime_recipe_digest"],
        "runtime": cell["runtime"],
        "original_policy_digests": [a["policy_digest"] for a in cell["artifacts"]],
        "original_policy_sha256": [a["policy_sha256"] for a in cell["artifacts"]],
        "model_states_before": raw["model_states_before"],
        "model_states_after": raw["model_states_after"],
        "fold_plan_digest": common["fold_plan_digest"],
        "opening_state_digest": raw["opening_state_digest"],
        "closing_state_digest": raw["closing_state_digest"],
        "native_execution_summary": result.walk_forward.stitched.diagnostics.digest_payload(),
        "source_semantics": "declared_envelope_plus_actual_actor_rows_not_authenticated_usage",
    }
    if canonical_json_bytes({k: legacy[k] for k in expected}) != canonical_json_bytes(
        expected
    ):
        raise ValueError(
            "original fee route metadata differs from observed plan/context"
        )
    if type(legacy["binding_digests"]) is not list or len(
        legacy["binding_digests"]
    ) != len(cell["artifacts"]):
        raise ValueError("original fee binding roster differs")
    for digest in legacy["binding_digests"]:
        _digest(digest, "original fee binding")
    if type(legacy["actor_rows"]) is not list or len(legacy["actor_rows"]) != len(
        raw["rows"]
    ):
        raise ValueError("original fee action roster differs")
    for actor, row in zip(legacy["actor_rows"], raw["rows"], strict=True):
        _closed(
            actor,
            {
                "fold_index",
                "decision_index",
                "decision_time_ns",
                "state_digest",
                "policy_digest",
                "runtime_recipe_digest",
                "source_context_digest",
                "input",
                "input_hex",
                "cash",
                "equity",
                "action",
            },
            "original fee actor row",
        )
        expected_actor = {
            k: row[k]
            for k in (
                "fold_index",
                "decision_index",
                "decision_time_ns",
                "policy_digest",
                "runtime_recipe_digest",
            )
        }
        expected_actor.update(
            state_digest=row["before_state_digest"],
            source_context_digest=cell["runtime"]["source_context_digest"],
            input={k: row["input"][k] for k in ("dtype", "shape", "sha256")},
            input_hex=row["input"]["hex"],
            action=row["call_outcome"]["action"],
        )
        if canonical_json_bytes(
            {k: actor[k] for k in expected_actor}
        ) != canonical_json_bytes(expected_actor):
            raise ValueError(
                "original fee actor rows differ from actually received calls"
            )


@dataclass(frozen=True, slots=True)
class ObservedGlobalAllocationExecution:
    native_result: (
        GlobalAllocationWalkForwardResult
        | GlobalNonRLAllocationResult
        | GlobalAllocationCashControlResult
        | FeeStressedGlobalAllocationResult
    )
    receipt: GlobalAllocationExecutionReceipt

    def __post_init__(self) -> None:
        if type(self.receipt) is not GlobalAllocationExecutionReceipt:
            raise ValueError("observed result requires exact completed receipt")
        self.receipt.__post_init__()
        raw = self.receipt.payload
        if raw["status"] != "completed":
            raise ValueError("partial receipt cannot wrap completed execution")
        cell = raw["plan"]["cell"]
        expected = (
            GlobalNonRLAllocationResult
            if cell["kind"] == "nonrl"
            else GlobalAllocationCashControlResult
            if cell["kind"] == "cash"
            else FeeStressedGlobalAllocationResult
            if cell["fee"] is not None
            else GlobalAllocationWalkForwardResult
        )
        if type(self.native_result) is not expected:
            raise ValueError(
                "observed result requires exact native global route result"
            )
        self.native_result.__post_init__()
        if isinstance(
            self.native_result,
            (GlobalNonRLAllocationResult, GlobalAllocationCashControlResult),
        ):
            candidate = (
                self.native_result.candidate_recipe_digest
                if isinstance(self.native_result, GlobalNonRLAllocationResult)
                else self.native_result.carrier_recipe_digest
            )
            if (
                candidate != cell["original_candidate_digest"]
                or self.native_result.forecast_context_digest
                != raw["plan"]["common"]["forecast_context_digest"]
                or self.native_result.runtime_recipe_digest
                != cell["runtime_recipe_digest"]
            ):
                raise ValueError(
                    "native control wrapper metadata differs from observed plan/context"
                )
        elif isinstance(self.native_result, FeeStressedGlobalAllocationResult):
            _check_legacy_fee_result(self.native_result, raw)
        result = (
            self.native_result
            if isinstance(self.native_result, GlobalAllocationWalkForwardResult)
            else self.native_result.walk_forward
        )
        _validate_global_control_roster(result)
        expected_policies = (
            [cell["control_policy_digest"]] * len(result.folds)
            if cell["kind"] in {"nonrl", "cash"}
            else [a["policy_digest"] for a in cell["artifacts"]]
        )
        if list(result.policy_digests) != expected_policies:
            raise ValueError("native result policy roster differs from observed cell")
        actual = [
            {
                "fold_index": f.fold_index,
                "range": [f.start, f.stop],
                "opening_state_digest": f.opening_state_digest,
                "closing_state_digest": f.closing_state_digest,
                "returns": list(f.returns.values),
                "diagnostics": f.diagnostics.digest_payload(),
            }
            for f in result.folds
        ]
        if canonical_json_bytes(actual) != canonical_json_bytes(raw["segments"]):
            raise ValueError("native result segments differ from observed receipt")


def _make_receipt(
    plan: GlobalAllocationExecutionPlan,
    collector: GlobalAllocationExecutionCollector,
    provenance_before: dict[str, object],
    error: Exception | None = None,
    native_result: object = None,
) -> GlobalAllocationExecutionReceipt:
    after = build_candidate_run_provenance()
    after_states: list[str | None] = []
    state_failures = []
    for index, model in enumerate(collector.models):
        try:
            after_states.append(allocation_policy_state_digest(model))
        except ValueError as state_error:
            after_states.append(None)
            state_failures.append(
                {
                    "index": index,
                    "error_type": type(state_error).__name__,
                    "message": str(state_error),
                }
            )
    reason = getattr(getattr(collector.env, "book", None), "termination_reason", None)
    return GlobalAllocationExecutionReceipt.from_payload(
        json.loads(
            canonical_json_bytes(
                {
                    "schema": "allocation_global_execution_receipt_v1",
                    "plan": plan.payload,
                    "common_digest": plan.common_digest,
                    "cell_plan_digest": plan.cell_plan_digest,
                    "status": "completed"
                    if error is None
                    else "economic_stop"
                    if reason is not None and collector.economic_stop_consistent
                    else "integrity_failure",
                    "reset_count": collector.resets,
                    "opening_state_digest": collector.opening,
                    "closing_state_digest": collector.rows[-1]["after_state_digest"]
                    if collector.rows
                    else None,
                    "rows": collector.rows,
                    "segments": collector.segments,
                    "native_terminal": bool(
                        getattr(collector.env, "_terminated", False)
                        and getattr(collector.env, "index", None)
                        == collector.env.stop_index
                    ),
                    "termination_reason": reason,
                    "economic_stop_consistent": bool(
                        error is not None and collector.economic_stop_consistent
                    ),
                    "legacy_fee_receipt_digest": sha256(
                        native_result._receipt
                    ).hexdigest()
                    if type(native_result) is FeeStressedGlobalAllocationResult
                    else None,
                    "failure": None
                    if error is None
                    else {
                        "error_type": type(error).__name__,
                        "message": str(error),
                    },
                    "model_states_before": list(collector.states),
                    "model_states_after": after_states,
                    "model_state_failures_after": state_failures,
                    "provenance_before": provenance_before,
                    "provenance_after": after,
                }
            )
        )
    )


def run_declared_global_allocation_execution(
    folds: tuple[WalkForwardFold, ...],
    env: AllocationTradingEnv,
    plan: GlobalAllocationExecutionPlan,
    *,
    artifacts: tuple[AllocationFoldPolicyArtifact, ...] = (),
    base_env: AllocationTradingEnv | None = None,
    contract: AllocationComparisonContract | None = None,
) -> ObservedGlobalAllocationExecution:
    if type(plan) is not GlobalAllocationExecutionPlan:
        raise ValueError("global execution requires its exact closed plan")
    plan.__post_init__()
    cell, common = plan.payload["cell"], plan.payload["common"]
    provenance_before = _provenance(common)
    collector = GlobalAllocationExecutionCollector(env, common, cell)
    validate_global_collector(env, collector)
    declared = declare_global_allocation_execution(
        folds,
        env,
        kind=cell["kind"],
        scenario=common["scenario"],
        expected_implementation_digest=common["implementation_digest"],
        expected_runtime_digest=common["runtime_environment_digest"],
        seed=cell["seed"],
        reset_seed=None if cell["kind"] in {"nonrl", "cash"} else cell["reset_seed"],
        artifacts=artifacts,
        base_env=base_env,
        contract=contract,
        fee_factor=None if cell["fee"] is None else cell["fee"]["factor"],
    )
    if declared.payload != plan.payload:
        raise ValueError("global current declaration differs from frozen plan")
    result: (
        GlobalAllocationWalkForwardResult
        | GlobalNonRLAllocationResult
        | GlobalAllocationCashControlResult
        | FeeStressedGlobalAllocationResult
    ) | None = None
    try:
        if cell["kind"] == "nonrl":
            result = run_global_nonrl_allocation(folds, env, collector=collector)
        elif cell["kind"] == "cash":
            result = run_global_allocation_cash_control(folds, env, collector=collector)
        elif cell["fee"] is not None:
            if base_env is None or contract is None:
                raise ValueError("global fee execution requires original base/contract")
            result = run_fee_stressed_global_allocation(
                folds,
                base_env=base_env,
                stress_env=env,
                artifacts=artifacts,
                contract=contract,
                candidate=AllocationCandidateKind.RESIDUAL_PPO
                if cell["kind"] == "residual_ppo"
                else AllocationCandidateKind.DIRECT_PPO,
                scenario=common["scenario"],
                fee_factor=cell["fee"]["factor"],
                reset_seed=cell["reset_seed"],
                collector=collector,
            )
        else:
            result = run_artifact_bound_global_allocation_walk_forward(
                folds,
                env,
                artifacts,
                reset_seed=cell["reset_seed"],
                collector=collector,
            )
        collector._consistent()
        _provenance(common)
        return ObservedGlobalAllocationExecution(
            result,
            _make_receipt(plan, collector, provenance_before, native_result=result),
        )
    except Exception as error:
        collector.failed(error)
        try:
            setattr(
                error,
                "global_execution_receipt",
                _make_receipt(
                    plan, collector, provenance_before, error, native_result=result
                ),
            )
        except Exception as receipt_error:
            # Never replace the actual execution failure with a recorder failure.
            setattr(
                error,
                "global_execution_receipt_error",
                {
                    "error_type": type(receipt_error).__name__,
                    "message": str(receipt_error),
                },
            )
        raise


__all__ = [
    "GlobalAllocationExecutionPlan",
    "GlobalAllocationExecutionReceipt",
    "ObservedGlobalAllocationExecution",
    "declare_global_allocation_execution",
    "run_declared_global_allocation_execution",
]
