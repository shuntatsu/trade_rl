"""V10-owned execution-boundary diagnostics for replay forensics."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.identity import content_and_arrays_digest
from trade_rl.learning.rollout_evaluation import (
    ActionPathEvaluation,
    ActionPathExecutionTrace,
)

EXECUTION_TRACE_SCHEMA = "causal_alpha_v10_execution_trace_v1"
EXECUTION_DIAGNOSTICS_SCHEMA = "causal_alpha_v10_execution_diagnostics_v1"


def _trace_digest(trace: ActionPathExecutionTrace) -> str:
    return content_and_arrays_digest(
        {"schema_version": EXECUTION_TRACE_SCHEMA},
        (
            ("pre_action_weights", trace.pre_action_weights),
            ("risk_constrained_weights", trace.risk_constrained_weights),
            ("post_step_weights", trace.post_step_weights),
            ("applied_risk_scales", trace.applied_risk_scales),
            ("strategy_intent_changes", trace.strategy_intent_changes),
            ("realized_state_follows", trace.realized_state_follows),
            ("rebalance_reassertions", trace.rebalance_reassertions),
            ("hard_risk_violations", trace.hard_risk_violations),
        ),
    )


def _trace_payload(trace: ActionPathExecutionTrace) -> dict[str, object]:
    return {
        "schema_version": EXECUTION_TRACE_SCHEMA,
        "pre_action_weights": trace.pre_action_weights.tolist(),
        "risk_constrained_weights": trace.risk_constrained_weights.tolist(),
        "post_step_weights": trace.post_step_weights.tolist(),
        "applied_risk_scales": trace.applied_risk_scales.tolist(),
        "strategy_intent_changes": trace.strategy_intent_changes.tolist(),
        "realized_state_follows": trace.realized_state_follows.tolist(),
        "rebalance_reassertions": trace.rebalance_reassertions.tolist(),
        "hard_risk_violations": trace.hard_risk_violations.tolist(),
        "artifact_digest": _trace_digest(trace),
    }


def execution_trace_payload(evaluation: ActionPathEvaluation) -> dict[str, object]:
    """Serialize the exact decision-boundary trace retained by the evaluator."""

    trace = evaluation.execution_trace
    if not isinstance(trace, ActionPathExecutionTrace):
        raise ValueError("V10 replay requires execution-boundary trace evidence")
    return _trace_payload(trace)


def _trace_from_payload(value: object) -> ActionPathExecutionTrace:
    if not isinstance(value, Mapping):
        raise ValueError("V10 replay execution trace is invalid")
    if value.get("schema_version") != EXECUTION_TRACE_SCHEMA:
        raise ValueError("V10 replay execution trace schema drifted")
    for field in (
        "strategy_intent_changes",
        "realized_state_follows",
        "rebalance_reassertions",
        "hard_risk_violations",
    ):
        if field not in value:
            raise ValueError("V10 replay execution trace is invalid")
        raw_boolean = np.asarray(value[field])
        if raw_boolean.dtype.kind != "b":
            raise ValueError("V10 replay execution trace boolean fields are invalid")
    try:
        trace = ActionPathExecutionTrace(
            pre_action_weights=value["pre_action_weights"],
            risk_constrained_weights=value["risk_constrained_weights"],
            post_step_weights=value["post_step_weights"],
            applied_risk_scales=value["applied_risk_scales"],
            strategy_intent_changes=value["strategy_intent_changes"],
            realized_state_follows=value["realized_state_follows"],
            rebalance_reassertions=value["rebalance_reassertions"],
            hard_risk_violations=value["hard_risk_violations"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("V10 replay execution trace is invalid") from error
    if value.get("artifact_digest") != _trace_digest(trace):
        raise ValueError("V10 replay execution trace digest drifted")
    return trace


def validate_execution_trace_payload(value: object) -> dict[str, object]:
    """Validate a persisted trace and return its canonical JSON-safe payload."""

    return _trace_payload(_trace_from_payload(value))


def _diagnostics_body(trace: ActionPathExecutionTrace) -> dict[str, object]:
    """Derive the only valid compact diagnostics for one execution trace."""

    return {
        "schema_version": EXECUTION_DIAGNOSTICS_SCHEMA,
        "trace_digest": _trace_digest(trace),
        "decision_count": int(trace.pre_action_weights.shape[0]),
        "strategy_intent_change_count": int(
            np.count_nonzero(trace.strategy_intent_changes)
        ),
        "realized_state_follow_count": int(
            np.count_nonzero(trace.realized_state_follows)
        ),
        "rebalance_reassertion_count": int(
            np.count_nonzero(trace.rebalance_reassertions)
        ),
        "hard_risk_violation": bool(np.any(trace.hard_risk_violations)),
        "minimum_applied_risk_scale": float(np.min(trace.applied_risk_scales)),
        "pre_action_mean_abs_weight": float(np.mean(np.abs(trace.pre_action_weights))),
        "risk_constrained_mean_abs_weight": float(
            np.mean(np.abs(trace.risk_constrained_weights))
        ),
        "post_step_mean_abs_weight": float(np.mean(np.abs(trace.post_step_weights))),
        "maximum_post_step_abs_weight": float(np.max(np.abs(trace.post_step_weights))),
    }


def execution_diagnostics(
    evaluation: ActionPathEvaluation,
    trace_payload: Mapping[str, object],
) -> dict[str, object]:
    """Summarize V10 execution-boundary facts without changing generic evidence."""

    trace = evaluation.execution_trace
    if not isinstance(trace, ActionPathExecutionTrace):
        raise ValueError("V10 replay requires execution-boundary trace evidence")
    trace_digest = _trace_digest(trace)
    if trace_payload.get("artifact_digest") != trace_digest:
        raise ValueError("V10 replay trace payload does not match evaluation")
    body = _diagnostics_body(trace)
    hard_risk = body["hard_risk_violation"]
    if (
        getattr(evaluation.collapse_evidence, "hard_risk_violation", None)
        is not hard_risk
    ):
        raise ValueError(
            "V10 hard-risk diagnostics do not reconcile with collapse evidence"
        )
    return {**body, "artifact_digest": content_digest(body)}


def validate_execution_diagnostics(
    value: object,
    trace_payload: Mapping[str, object],
) -> dict[str, object]:
    """Validate compact diagnostics and prove every derived value matches its trace."""

    if not isinstance(value, Mapping):
        raise ValueError("V10 replay execution diagnostics are invalid")
    payload = {str(key): item for key, item in value.items()}
    if payload.get("schema_version") != EXECUTION_DIAGNOSTICS_SCHEMA:
        raise ValueError("V10 replay execution diagnostics schema drifted")
    artifact_digest = payload.pop("artifact_digest", None)
    if artifact_digest != content_digest(payload):
        raise ValueError("V10 replay execution diagnostics digest drifted")
    if payload.get("trace_digest") != trace_payload.get("artifact_digest"):
        raise ValueError("V10 replay execution diagnostics trace identity drifted")
    for field in (
        "decision_count",
        "strategy_intent_change_count",
        "realized_state_follow_count",
        "rebalance_reassertion_count",
    ):
        count = payload.get(field)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("V10 replay execution diagnostics counts are invalid")
    if not isinstance(payload.get("hard_risk_violation"), bool):
        raise ValueError("V10 replay hard-risk diagnostics are invalid")
    bounded_scale = payload.get("minimum_applied_risk_scale")
    if (
        isinstance(bounded_scale, bool)
        or not isinstance(bounded_scale, int | float)
        or not np.isfinite(float(bounded_scale))
        or not 0.0 <= float(bounded_scale) <= 1.0
    ):
        raise ValueError("V10 replay applied risk scale diagnostics are invalid")
    for field in (
        "pre_action_mean_abs_weight",
        "risk_constrained_mean_abs_weight",
        "post_step_mean_abs_weight",
        "maximum_post_step_abs_weight",
    ):
        metric = payload.get(field)
        if (
            isinstance(metric, bool)
            or not isinstance(metric, int | float)
            or not np.isfinite(float(metric))
            or float(metric) < 0.0
        ):
            raise ValueError("V10 replay execution diagnostics weights are invalid")
    expected = _diagnostics_body(_trace_from_payload(trace_payload))
    if payload != expected:
        raise ValueError(
            "V10 replay execution diagnostics do not reconcile with execution trace"
        )
    return {**expected, "artifact_digest": artifact_digest}


__all__ = [
    "EXECUTION_DIAGNOSTICS_SCHEMA",
    "EXECUTION_TRACE_SCHEMA",
    "execution_diagnostics",
    "execution_trace_payload",
    "validate_execution_diagnostics",
    "validate_execution_trace_payload",
]
