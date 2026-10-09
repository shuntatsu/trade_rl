"""Declared global envelopes and typed observers; no fitting or research authority."""

from __future__ import annotations

import json
from hashlib import sha256
from numbers import Integral
from typing import Any

import numpy as np

from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.evaluation.allocation_comparison_evidence import (
    allocation_fold_plan_digest,
)
from trade_rl.evaluation.allocation_decision import AllocationActionExecutionResult
from trade_rl.evaluation.rl_allocation.continuation import allocation_state_digest
from trade_rl.evaluation.rl_allocation.continuous_walk_forward import (
    AllocationFoldPolicy,
)
from trade_rl.evaluation.rl_allocation.env import (
    AllocationTradingEnv,
)
from trade_rl.evaluation.rl_allocation.transition_facts import (
    freeze_allocation_execution,
    validate_allocation_execution_cost_payload,
    validate_allocation_execution_facts_v2,
)
from trade_rl.evaluation.robustness.walk_forward.folds import WalkForwardFold
from trade_rl.evaluation.robustness.walk_forward.stitching import FoldOOSResult
from trade_rl.simulation import MarketExecutor
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.rl.allocation_fee_stress import (
    allocation_policy_state_digest,
    retained_debt_economics_digest,
)
from trade_rl.strategies.rl.allocation_model import AllocationPPOPolicy


def allocation_array_pin(values: np.ndarray) -> dict[str, Any]:
    return {
        "dtype": values.dtype.str,
        "shape": list(values.shape),
        "sha256": sha256(np.ascontiguousarray(values).tobytes()).hexdigest(),
    }


def allocation_source_envelope_digest(env: AllocationTradingEnv) -> str:
    """Declared finite envelope, explicitly not an authenticated usage receipt."""
    dataset = env.dataset
    arrays = {}
    for name, values in dataset.identity_arrays().items():
        if name == "global_features":
            continue
        if name in ("features", "feature_available"):
            selected = values[env.start_index : env.stop_index, env.symbol_index][
                :, list(env.feature_indices)
            ]
        elif values.ndim > 0 and values.shape[0] == dataset.n_bars:
            selected = values[env.start_index : env.stop_index + 1]
        else:
            selected = values
        arrays[name] = allocation_array_pin(selected)
    return content_digest(
        {
            "schema": "allocation_fee_global_source_envelope_v1",
            "dataset_id": dataset.dataset_id,
            "dataset_contract": dataset.identity_contract_payload(),
            "symbol_index": env.symbol_index,
            "feature_indices": list(env.feature_indices),
            "feature_names": list(dataset.feature_names),
            "range": [env.start_index, env.stop_index],
            "account_id": env.account_id,
            "forecast_context_digest": env.stream.digest,
            "cost_beliefs": [
                env._estimates[key].payload() for key in sorted(env._estimates)
            ],
            "arrays": arrays,
        }
    )


def allocation_execution_runtime(env: AllocationTradingEnv) -> dict[str, Any]:
    if (
        type(env) is not AllocationTradingEnv
        or type(env.executor) is not MarketExecutor
    ):
        raise ValueError("fee view requires native allocation execution")
    cost = env.execution_cost
    if (
        type(cost) is not ExecutionCostConfig
        or type(env.executor.cost) is not ExecutionCostConfig
    ):
        raise ValueError("fee view requires native cost configuration")
    cost.__post_init__()
    env.executor.cost.__post_init__()
    actual = env.executor.cost.execution_policy_payload()
    if canonical_json_bytes(actual) != canonical_json_bytes(
        cost.execution_policy_payload()
    ):
        raise ValueError("fee executor actual cost differs from configured cost")
    if (
        env.executor.market_order_profile is not None
        or env.executor.rule_stress.enabled
        or env.executor.insolvency_valuation != "retain_debt"
    ):
        raise ValueError("fee view rejects execution profiles or rule stress")
    economics = retained_debt_economics_digest(actual)
    if economics != env.executor.execution_policy_digest:
        raise ValueError("fee native cost contents changed behind cached economics")
    env.validate_binding()
    return {
        "recipe": env.recipe,
        "cost_payload": actual,
        "source_context_digest": allocation_source_envelope_digest(env),
    }


def global_declared_source_digest(env: AllocationTradingEnv) -> str:
    """Full shared raw scope, independent of actor-selected columns/preprocessing."""
    arrays = {}
    for name, values in env.dataset.identity_arrays().items():
        if name == "global_features":
            continue
        selected = values
        if values.ndim > 0 and values.shape[0] == env.dataset.n_bars:
            selected = values[env.start_index : env.stop_index + 1]
        if not np.isfinite(selected).all():
            raise ValueError("declared global source arrays must be finite")
        arrays[name] = allocation_array_pin(selected)
    return content_digest(
        {
            "schema": "allocation_global_declared_source_v1",
            "dataset_contract": env.dataset.identity_contract_payload(),
            "dataset_id": env.dataset.dataset_id,
            "range": [env.start_index, env.stop_index],
            "arrays": arrays,
        }
    )


def _invalid_return(action: object, error: ValueError) -> dict[str, Any]:
    scalar_type, numpy_dtype, encoding, value = (
        "unsupported",
        None,
        "unavailable_unsupported",
        None,
    )
    actual_type = type(action)
    if actual_type in (bool, int, float):
        scalar_type = {bool: "python_bool", int: "python_int", float: "python_float"}[
            actual_type
        ]
        value = action
        encoding = "native_scalar"
    elif actual_type is np.bool_ and isinstance(action, np.bool_):
        scalar_type, numpy_dtype, value, encoding = (
            "numpy_bool",
            action.dtype.str,
            bool(action),
            "native_scalar",
        )
    elif actual_type in (
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.longlong,
        np.ulonglong,
    ) and isinstance(action, np.integer):
        scalar_type, numpy_dtype, value, encoding = (
            "numpy_int",
            action.dtype.str,
            int(action),
            "native_scalar",
        )
    elif actual_type in (np.float16, np.float32, np.float64) and isinstance(
        action, np.floating
    ):
        scalar_type, numpy_dtype, value, encoding = (
            "numpy_float",
            action.dtype.str,
            float(action),
            "native_scalar",
        )
    if type(value) is float and not np.isfinite(value):
        value, encoding = None, "unavailable_nonfinite"
    return {
        "status": "invalid_return",
        "scalar_type": scalar_type,
        "numpy_dtype": numpy_dtype,
        "encoding": encoding,
        "value": value,
        "validation_failure": {
            "error_type": type(error).__name__,
            "message": str(error),
        },
    }


class GlobalAllocationExecutionCollector:
    def __init__(
        self, env: AllocationTradingEnv, common: dict[str, Any], cell: dict[str, Any]
    ):
        self.env, self.dataset = env, env.dataset
        self.common, self.cell = json.loads(canonical_json_bytes([common, cell]))
        validate_allocation_execution_cost_payload(
            self.cell["runtime"]["cost_payload"],
            economics_digest=self.common["economics_digest"],
        )
        self.rows: list[dict[str, Any]] = []
        self.models: tuple[AllocationPPOPolicy, ...] = ()
        self.states: tuple[str, ...] = ()
        self.resets = 0
        self.opening: str | None = None
        self.segments: list[dict[str, Any]] = []
        self.error: Exception | None = None
        self.economic_stop_consistent = False

    def _consistent(self) -> None:
        if (
            self.env.dataset is not self.dataset
            or global_declared_source_digest(self.env) != self.common["source_digest"]
            or canonical_json_bytes(allocation_execution_runtime(self.env))
            != canonical_json_bytes(self.cell["runtime"])
        ):
            raise ValueError("global declared source/runtime contents changed")
        if tuple(allocation_policy_state_digest(m) for m in self.models) != self.states:
            raise ValueError("global original policy tensor state changed")

    def loaded_policies(self, policies: tuple[AllocationPPOPolicy, ...]) -> None:
        if (
            type(policies) is not tuple
            or any(type(model) is not AllocationPPOPolicy for model in policies)
            or self.models
            or len(policies) != len(self.cell["artifacts"])
        ):
            raise ValueError("global original loaded policy roster differs")
        for model, artifact in zip(policies, self.cell["artifacts"], strict=True):
            if (
                content_digest(model.manifest) != artifact["policy_digest"]
                or model.manifest["policy_sha256"] != artifact["policy_sha256"]
            ):
                raise ValueError("global loaded policy differs from original pins")
        self.models = policies
        self.states = tuple(allocation_policy_state_digest(m) for m in policies)

    def before_reset(
        self,
        folds: tuple[WalkForwardFold, ...],
        env: AllocationTradingEnv,
        policies: tuple[AllocationFoldPolicy, ...],
        seed: int | None,
    ) -> None:
        if (
            env is not self.env
            or self.resets
            or allocation_fold_plan_digest(folds) != self.common["fold_plan_digest"]
            or seed != self.cell["reset_seed"]
        ):
            raise ValueError("global reset/fold/seed declaration differs")
        expected = (
            [self.cell["control_policy_digest"]] * len(folds)
            if self.cell["kind"] in {"nonrl", "cash"}
            else [a["policy_digest"] for a in self.cell["artifacts"]]
        )
        if [p.policy_digest for p in policies] != expected or (
            self.cell["artifacts"] and not self.models
        ):
            raise ValueError("global admitted control/model roster differs")
        self._consistent()

    def after_reset(self) -> None:
        self.resets += 1
        self.opening = allocation_state_digest(self.env)
        self._consistent()

    def before_action(
        self,
        fold_index: int,
        policy_digest: str,
        observation: np.ndarray,
        recipe_digest: str,
    ) -> None:
        values = np.asarray(observation)
        if (
            values.dtype != np.dtype(np.float32)
            or values.shape != self.env.observation_space.shape
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                "global public action input differs from native finite float32 vector"
            )
        self._consistent()
        self.rows.append(
            {
                "fold_index": fold_index,
                "decision_index": self.env.index,
                "decision_time_ns": int(
                    self.env.dataset.timestamps[self.env.index]
                    .astype("datetime64[ns]")
                    .astype("int64")
                ),
                "policy_digest": policy_digest,
                "runtime_recipe_digest": recipe_digest,
                "before_state_digest": allocation_state_digest(self.env),
                "input": allocation_array_pin(values)
                | {
                    "seam": "public_policy_action_argument",
                    "hex": np.ascontiguousarray(values).tobytes().hex(),
                },
                "call_outcome": None,
                "native": None,
                "native_validation_failure": None,
                "after_state_digest": None,
                "terminated": None,
                "truncated": None,
            }
        )

    def after_action(self, action: object) -> None:
        if (
            isinstance(action, (bool, np.bool_))
            or not isinstance(action, Integral)
            or not 0 <= int(action) <= 3
        ):
            error = ValueError(
                "global public action result must be an integer within 0..3"
            )
            self.rows[-1]["call_outcome"] = _invalid_return(action, error)
            raise error
        self.rows[-1]["call_outcome"] = {"status": "returned", "action": int(action)}
        self._consistent()
        if allocation_state_digest(self.env) != self.rows[-1]["before_state_digest"]:
            raise ValueError("global account changed during public action invocation")

    def action_failed(self, error: Exception) -> None:
        self.rows[-1]["call_outcome"] = {
            "status": "raised",
            "error_type": type(error).__name__,
            "message": str(error),
        }

    def freeze_execution(self, result: AllocationActionExecutionResult) -> bytes:
        raw = freeze_allocation_execution(self.env, result)
        self.rows[-1]["native"] = json.loads(raw)
        try:
            validate_allocation_execution_facts_v2(
                self.rows[-1]["native"],
                self.cell["runtime"]["recipe"],
                dataset_id=self.common["dataset_id"],
                action_code=self.rows[-1]["call_outcome"]["action"],
                actual_cost_payload=self.cell["runtime"]["cost_payload"],
            )
        except ValueError as error:
            self.rows[-1]["native_validation_failure"] = {
                "error_type": type(error).__name__,
                "message": str(error),
            }
            raise
        return raw

    def after_step(
        self, terminated: bool, truncated: bool, info: dict[str, Any]
    ) -> None:
        row = self.rows[-1]
        self.economic_stop_consistent = False
        if canonical_json_bytes(row["native"]) != info.get("transition_trace"):
            raise ValueError(
                "global returned native trace differs from original pre-overwrite facts"
            )
        row["terminated"], row["truncated"] = terminated, truncated
        # The existing account snapshot is defined only for solvent live books.
        # An economic stop keeps the actual detached signed book, not a forged
        # live-state digest or another account transition.
        if self.env.book.termination_reason is None:
            row["after_state_digest"] = allocation_state_digest(self.env)
        self._consistent()
        self.economic_stop_consistent = self.env.book.termination_reason is not None

    def completed(self, folds: tuple[FoldOOSResult, ...]) -> None:
        if type(folds) is not tuple or any(type(f) is not FoldOOSResult for f in folds):
            raise ValueError(
                "global observed completion requires exact native segments"
            )
        self.segments = [
            {
                "fold_index": f.fold_index,
                "range": [f.start, f.stop],
                "opening_state_digest": f.opening_state_digest,
                "closing_state_digest": f.closing_state_digest,
                "returns": list(f.returns.values),
                "diagnostics": f.diagnostics.digest_payload(),
            }
            for f in folds
        ]

    def failed(self, error: Exception) -> None:
        if self.error is None:
            self.error = error


def validate_global_collector(
    env: AllocationTradingEnv,
    collector: GlobalAllocationExecutionCollector | None,
) -> None:
    if collector is None:
        return
    if type(collector) is not GlobalAllocationExecutionCollector:
        raise ValueError(
            "global collector requires the exact native observational collector"
        )
    if type(env) is not AllocationTradingEnv:
        raise ValueError("global collector requires the native allocation environment")
    if env._transition_recorder is not None:
        raise ValueError("global execution observer slot is occupied")
