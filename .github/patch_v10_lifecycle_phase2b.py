from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    file = Path(path)
    text = file.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one match, got {count}: {old[:120]!r}")
    file.write_text(text.replace(old, new, 1), encoding="utf-8")


path = "trade_rl/learning/rollout_evaluation.py"
replace_once(
    path,
    '_STEP_TRACE_SCHEMA = "action_path_step_trace_v1"\n_STEP_TRACE_TOLERANCE = 1e-12\n',
    '_STEP_TRACE_SCHEMA = "action_path_step_trace_v1"\n_LIFECYCLE_TRACE_SCHEMA = "action_path_lifecycle_trace_v1"\n_STEP_TRACE_TOLERANCE = 1e-12\n',
)
replace_once(
    path,
    """\n\n@dataclass(frozen=True, slots=True)\nclass ActionPathStepEconomics:\n""",
    '''\n\n@dataclass(frozen=True, slots=True)\nclass ActionPathLifecycleTrace:\n    """Immutable execution lifecycle evidence separate from legacy step attribution."""\n\n    submitted_targets: np.ndarray\n    execution_intent_targets: np.ndarray\n    final_risk_targets: np.ndarray\n    applied_risk_scales: np.ndarray\n    hard_risk_evidence_available: np.ndarray\n    hard_risk_violations: np.ndarray\n    risk_reasons: tuple[tuple[str, ...], ...]\n    transition_classes: tuple[str, ...]\n    flatten_initiators: tuple[str, ...]\n    schema_version: str = _LIFECYCLE_TRACE_SCHEMA\n    digest: str = ""\n\n    def __post_init__(self) -> None:\n        submitted = np.asarray(self.submitted_targets, dtype=np.float64).copy()\n        rows = submitted.shape[0] if submitted.ndim == 2 else 0\n        matrices = {\n            "submitted_targets": _trace_matrix(\n                submitted, rows=rows, field="lifecycle submitted_targets"\n            ),\n            "execution_intent_targets": _trace_matrix(\n                self.execution_intent_targets,\n                rows=rows,\n                field="lifecycle execution_intent_targets",\n            ),\n            "final_risk_targets": _trace_matrix(\n                self.final_risk_targets, rows=rows, field="lifecycle final_risk_targets"\n            ),\n        }\n        shape = matrices["submitted_targets"].shape\n        if any(array.shape != shape for array in matrices.values()):\n            raise ValueError("lifecycle target matrices are not aligned")\n        risk_scales = _trace_vector(\n            self.applied_risk_scales, rows=rows, field="lifecycle applied_risk_scales"\n        )\n        if np.any((risk_scales < 0.0) | (risk_scales > 1.0)):\n            raise ValueError("lifecycle applied risk scales are invalid")\n        bools: dict[str, np.ndarray] = {}\n        for name in ("hard_risk_evidence_available", "hard_risk_violations"):\n            raw = np.asarray(getattr(self, name))\n            if raw.dtype.kind != "b":\n                raise ValueError(f"lifecycle {name} must be boolean")\n            value = raw.reshape(-1).astype(np.bool_, copy=True)\n            if value.shape != (rows,):\n                raise ValueError(f"lifecycle {name} is not step-aligned")\n            value.setflags(write=False)\n            bools[name] = value\n        reasons = tuple(tuple(item) for item in self.risk_reasons)\n        transitions = tuple(self.transition_classes)\n        initiators = tuple(self.flatten_initiators)\n        if not (len(reasons) == len(transitions) == len(initiators) == rows):\n            raise ValueError("lifecycle string evidence is not step-aligned")\n        if any(\n            not isinstance(reason, str) or not reason.strip()\n            for row in reasons\n            for reason in row\n        ):\n            raise ValueError("lifecycle risk reasons are invalid")\n        allowed_transitions = {"flat", "entry", "hold", "rebalance", "exit", "flip", "mixed"}\n        if any(value not in allowed_transitions for value in transitions):\n            raise ValueError("lifecycle transition class is invalid")\n        if any(not isinstance(value, str) or not value for value in initiators):\n            raise ValueError("lifecycle flatten initiator is invalid")\n        if self.schema_version != _LIFECYCLE_TRACE_SCHEMA:\n            raise ValueError("unsupported lifecycle trace schema")\n        for name, value in matrices.items():\n            object.__setattr__(self, name, value)\n        object.__setattr__(self, "applied_risk_scales", risk_scales)\n        for name, value in bools.items():\n            object.__setattr__(self, name, value)\n        object.__setattr__(self, "risk_reasons", reasons)\n        object.__setattr__(self, "transition_classes", transitions)\n        object.__setattr__(self, "flatten_initiators", initiators)\n        expected = content_and_arrays_digest(\n            {\n                "flatten_initiators": initiators,\n                "risk_reasons": reasons,\n                "schema_version": self.schema_version,\n                "transition_classes": transitions,\n            },\n            (\n                ("submitted_targets", matrices["submitted_targets"]),\n                ("execution_intent_targets", matrices["execution_intent_targets"]),\n                ("final_risk_targets", matrices["final_risk_targets"]),\n                ("applied_risk_scales", risk_scales),\n                ("hard_risk_evidence_available", bools["hard_risk_evidence_available"]),\n                ("hard_risk_violations", bools["hard_risk_violations"]),\n            ),\n        )\n        if self.digest and self.digest != expected:\n            raise ValueError("lifecycle trace digest mismatch")\n        object.__setattr__(self, "digest", expected)\n\n    @property\n    def decision_count(self) -> int:\n        return int(self.submitted_targets.shape[0])\n\n    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:\n        payload: dict[str, object] = {\n            "applied_risk_scales": self.applied_risk_scales.tolist(),\n            "execution_intent_targets": self.execution_intent_targets.tolist(),\n            "final_risk_targets": self.final_risk_targets.tolist(),\n            "flatten_initiators": self.flatten_initiators,\n            "hard_risk_evidence_available": self.hard_risk_evidence_available.tolist(),\n            "hard_risk_violations": self.hard_risk_violations.tolist(),\n            "risk_reasons": self.risk_reasons,\n            "schema_version": self.schema_version,\n            "submitted_targets": self.submitted_targets.tolist(),\n            "transition_classes": self.transition_classes,\n        }\n        if include_digest:\n            payload["artifact_digest"] = self.digest\n        return payload\n\n    @classmethod\n    def from_payload(cls, value: object) -> ActionPathLifecycleTrace:\n        if not isinstance(value, Mapping):\n            raise ValueError("lifecycle trace payload is invalid")\n        payload = dict(value)\n        digest = str(payload.pop("artifact_digest", ""))\n        payload["risk_reasons"] = tuple(\n            tuple(str(reason) for reason in row)\n            for row in payload["risk_reasons"]\n        )\n        payload["transition_classes"] = tuple(payload["transition_classes"])\n        payload["flatten_initiators"] = tuple(payload["flatten_initiators"])\n        return cls(**payload, digest=digest)\n\n\n@dataclass(frozen=True, slots=True)\nclass ActionPathStepEconomics:\n''',
)
replace_once(
    path,
    """    step_economics: ActionPathStepEconomics | None = None\n    step_trace: ActionPathStepTrace | None = None\n\n    def __post_init__(self) -> None:\n""",
    """    step_economics: ActionPathStepEconomics | None = None\n    step_trace: ActionPathStepTrace | None = None\n    lifecycle_trace: ActionPathLifecycleTrace | None = None\n\n    def __post_init__(self) -> None:\n""",
)
replace_once(
    path,
    """                        raise ValueError(f\"step trace {name} does not reconcile\")\n        actions.setflags(write=False)\n""",
    """                        raise ValueError(f\"step trace {name} does not reconcile\")\n        if self.lifecycle_trace is not None:\n            if not isinstance(self.lifecycle_trace, ActionPathLifecycleTrace):\n                raise TypeError(\"lifecycle_trace must be ActionPathLifecycleTrace\")\n            if self.lifecycle_trace.decision_count != self.performance.step_count:\n                raise ValueError(\"lifecycle trace does not cover evaluated path\")\n        actions.setflags(write=False)\n""",
)
replace_once(
    path,
    """def evaluate_action_path(\n""",
    '''def _optional_numeric_attr(value: object, name: str) -> float | None:\n    raw = getattr(value, name, None)\n    if raw is None:\n        return None\n    if isinstance(raw, bool) or not isinstance(raw, int | float):\n        raise ValueError(f"hybrid risk {name} is invalid")\n    result = float(raw)\n    if not np.isfinite(result):\n        raise ValueError(f"hybrid risk {name} is non-finite")\n    return result\n\n\ndef _hard_risk_observation(\n    risk: object | None,\n    *,\n    shape: tuple[int, ...],\n    fallback: np.ndarray,\n) -> tuple[np.ndarray, float, bool, bool, tuple[str, ...]]:\n    if risk is None:\n        return fallback.copy(), 1.0, False, False, ()\n    final_weights = _step_vector(\n        getattr(risk, "weights", None),\n        shape=shape,\n        field="final risk target",\n        fallback=fallback,\n    )\n    scale = _optional_numeric_attr(risk, "risk_scale")\n    max_abs = _optional_numeric_attr(risk, "max_abs_weight")\n    max_gross = _optional_numeric_attr(risk, "max_gross")\n    tolerance = _optional_numeric_attr(risk, "fail_closed_tolerance")\n    reasons = tuple(getattr(risk, "reasons", ()))\n    if any(not isinstance(reason, str) or not reason.strip() for reason in reasons):\n        raise ValueError("hybrid risk projection reason is invalid")\n    available = None not in (scale, max_abs, max_gross, tolerance)\n    if not available:\n        return final_weights, 1.0 if scale is None else scale, False, False, reasons\n    assert scale is not None and max_abs is not None and max_gross is not None\n    assert tolerance is not None\n    if not 0.0 <= scale <= 1.0 or max_abs <= 0.0 or max_gross <= 0.0 or tolerance < 0.0:\n        raise ValueError("hybrid risk hard-limit evidence is invalid")\n    absolute = np.abs(final_weights)\n    violation = bool(\n        np.max(absolute, initial=0.0) > max_abs * scale + tolerance\n        or float(np.sum(absolute)) > max_gross * scale + tolerance\n        or (scale == 0.0 and np.any(absolute > tolerance))\n    )\n    return final_weights, scale, True, violation, reasons\n\n\ndef _transition_class(\n    before: np.ndarray, after: np.ndarray, *, tolerance: float\n) -> str:\n    before_nonflat = np.abs(before) > tolerance\n    after_nonflat = np.abs(after) > tolerance\n    entry = (~before_nonflat) & after_nonflat\n    exit_ = before_nonflat & (~after_nonflat)\n    flip = before_nonflat & after_nonflat & (before * after < 0.0)\n    if np.any(flip):\n        return "flip"\n    if np.any(entry) and np.any(exit_):\n        return "mixed"\n    if np.any(exit_):\n        return "exit"\n    if np.any(entry):\n        return "entry"\n    if np.any(after_nonflat):\n        if np.any(np.abs(after - before) > tolerance):\n            return "rebalance"\n        return "hold"\n    return "flat"\n\n\ndef _flatten_initiator(\n    *,\n    transition: str,\n    before: np.ndarray,\n    after: np.ndarray,\n    execution_intent: np.ndarray,\n    risk_reasons: tuple[str, ...],\n    hierarchy_reason: str,\n    liquidation: object | None,\n    liquidation_terminal: object,\n    tolerance: float,\n) -> str:\n    if transition != "exit":\n        return "not_applicable"\n    if liquidation is not None or liquidation_terminal is True:\n        return "liquidation"\n    for reason in ("emergency_flatten", "drawdown_deleveraging"):\n        if reason in risk_reasons:\n            return f"risk:{reason}"\n    if hierarchy_reason in {"exit", "neutral_fast_expiry", "risk_cap_flatten"}:\n        return f"policy:{hierarchy_reason}"\n    exited = (np.abs(before) > tolerance) & (np.abs(after) <= tolerance)\n    if np.any(exited) and np.all(np.abs(execution_intent[exited]) <= tolerance):\n        return "execution_intent_flatten"\n    return "unexplained"\n\n\ndef evaluate_action_path(\n''',
)
replace_once(
    path,
    """    trace_executed: list[bool] = []\n    for offset in range(expected_count):\n""",
    """    trace_executed: list[bool] = []\n    lifecycle_submitted_targets: list[np.ndarray] = []\n    lifecycle_execution_targets: list[np.ndarray] = []\n    lifecycle_final_risk_targets: list[np.ndarray] = []\n    lifecycle_risk_scales: list[float] = []\n    lifecycle_hard_risk_available: list[bool] = []\n    lifecycle_hard_risk_violations: list[bool] = []\n    lifecycle_risk_reasons: list[tuple[str, ...]] = []\n    lifecycle_transitions: list[str] = []\n    lifecycle_flatten_initiators: list[str] = []\n    for offset in range(expected_count):\n""",
)
replace_once(
    path,
    """        risk = info.get(\"hybrid_risk\")\n        risk_pretrade = None if risk is None else getattr(risk, \"pretrade_weights\", None)\n""",
    """        risk = info.get(\"hybrid_risk\")\n        submitted_target = _step_vector(\n            info.get(\"submitted_target\"),\n            shape=action.shape,\n            field=\"submitted target\",\n            fallback=np.asarray(action, dtype=np.float64),\n        )\n        execution_intent_target = _step_vector(\n            info.get(\"executed_target\"),\n            shape=action.shape,\n            field=\"execution intent target\",\n            fallback=submitted_target,\n        )\n        risk_pretrade = None if risk is None else getattr(risk, \"pretrade_weights\", None)\n""",
)
replace_once(
    path,
    """        realized = _step_vector(\n            effective,\n            shape=action.shape,\n            field=\"realized weight\",\n            fallback=projected,\n        )\n        provider = getattr(model, \"last_step_trace_metadata\", None)\n""",
    """        realized = _step_vector(\n            effective,\n            shape=action.shape,\n            field=\"realized weight\",\n            fallback=projected,\n        )\n        (\n            final_risk_target,\n            applied_risk_scale,\n            hard_risk_available,\n            hard_risk_violation,\n            step_risk_reasons,\n        ) = _hard_risk_observation(\n            risk,\n            shape=action.shape,\n            fallback=projected,\n        )\n        provider = getattr(model, \"last_step_trace_metadata\", None)\n""",
)
replace_once(
    path,
    """        if risk is not None:\n            for reason in tuple(getattr(risk, \"reasons\", ())):\n                if not isinstance(reason, str) or not reason.strip():\n                    raise ValueError(\"hybrid risk projection reason is invalid\")\n                risk_projection_reasons[reason] += 1\n""",
    """        for reason in step_risk_reasons:\n            risk_projection_reasons[reason] += 1\n""",
)
replace_once(
    path,
    """        trace_executed.append(executed)\n    diagnostics = trades.diagnostics()\n""",
    """        trace_executed.append(executed)\n        transition = _transition_class(\n            current_before, realized, tolerance=action_change_tolerance\n        )\n        flatten_initiator = _flatten_initiator(\n            transition=transition,\n            before=current_before,\n            after=realized,\n            execution_intent=execution_intent_target,\n            risk_reasons=step_risk_reasons,\n            hierarchy_reason=hierarchy_reason,\n            liquidation=liquidation,\n            liquidation_terminal=info.get(\"liquidation_terminal\"),\n            tolerance=action_change_tolerance,\n        )\n        lifecycle_submitted_targets.append(submitted_target)\n        lifecycle_execution_targets.append(execution_intent_target)\n        lifecycle_final_risk_targets.append(final_risk_target)\n        lifecycle_risk_scales.append(applied_risk_scale)\n        lifecycle_hard_risk_available.append(hard_risk_available)\n        lifecycle_hard_risk_violations.append(hard_risk_violation)\n        lifecycle_risk_reasons.append(step_risk_reasons)\n        lifecycle_transitions.append(transition)\n        lifecycle_flatten_initiators.append(flatten_initiator)\n    diagnostics = trades.diagnostics()\n""",
)
replace_once(
    path,
    """    evidence = ActionPathCollapseEvidence(\n""",
    """    lifecycle_trace = ActionPathLifecycleTrace(\n        submitted_targets=np.stack(lifecycle_submitted_targets, axis=0),\n        execution_intent_targets=np.stack(lifecycle_execution_targets, axis=0),\n        final_risk_targets=np.stack(lifecycle_final_risk_targets, axis=0),\n        applied_risk_scales=np.asarray(lifecycle_risk_scales, dtype=np.float64),\n        hard_risk_evidence_available=np.asarray(\n            lifecycle_hard_risk_available, dtype=np.bool_\n        ),\n        hard_risk_violations=np.asarray(\n            lifecycle_hard_risk_violations, dtype=np.bool_\n        ),\n        risk_reasons=tuple(lifecycle_risk_reasons),\n        transition_classes=tuple(lifecycle_transitions),\n        flatten_initiators=tuple(lifecycle_flatten_initiators),\n    )\n    evidence = ActionPathCollapseEvidence(\n""",
)
replace_once(
    path,
    """        hard_risk_violation=False,\n    )\n""",
    """        hard_risk_violation=bool(np.any(lifecycle_trace.hard_risk_violations)),\n    )\n""",
)
replace_once(
    path,
    """        step_trace=step_trace,\n    )\n\n\n__all__ = [\n""",
    """        step_trace=step_trace,\n        lifecycle_trace=lifecycle_trace,\n    )\n\n\n__all__ = [\n""",
)
replace_once(
    path,
    """    \"ActionPathEvaluation\",\n    \"ActionPathStepTrace\",\n""",
    """    \"ActionPathEvaluation\",\n    \"ActionPathLifecycleTrace\",\n    \"ActionPathStepTrace\",\n""",
)

# Persist lifecycle evidence only when a caller provides it; legacy V8/V9 payloads remain unchanged.
path = "trade_rl/workflows/universal_causal_alpha_v8_replay.py"
replace_once(
    path,
    "from trade_rl.learning.rollout_evaluation import ActionPathStepTrace\n",
    "from trade_rl.learning.rollout_evaluation import (\n    ActionPathLifecycleTrace,\n    ActionPathStepTrace,\n)\n",
)
replace_once(
    path,
    """    digest: str = \"\"\n    step_trace: ActionPathStepTrace | None = None\n\n    def __post_init__(self) -> None:\n""",
    """    digest: str = \"\"\n    step_trace: ActionPathStepTrace | None = None\n    lifecycle_trace: ActionPathLifecycleTrace | None = None\n\n    def __post_init__(self) -> None:\n""",
)
replace_once(
    path,
    """            if self.step_trace.decision_count != self.v6_metric.decision_count:\n                raise ValueError(\"V8 replay step trace count drifted\")\n        for observed, expected in (\n""",
    """            if self.step_trace.decision_count != self.v6_metric.decision_count:\n                raise ValueError(\"V8 replay step trace count drifted\")\n        if self.lifecycle_trace is not None:\n            if not isinstance(self.lifecycle_trace, ActionPathLifecycleTrace):\n                raise TypeError(\"V8 replay lifecycle trace is invalid\")\n            if self.lifecycle_trace.decision_count != self.v6_metric.decision_count:\n                raise ValueError(\"V8 replay lifecycle trace count drifted\")\n        for observed, expected in (\n""",
)
replace_once(
    path,
    """        if self.step_trace is not None:\n            payload[\"step_trace\"] = self.step_trace.to_payload()\n        if include_digest:\n""",
    """        if self.step_trace is not None:\n            payload[\"step_trace\"] = self.step_trace.to_payload()\n        if self.lifecycle_trace is not None:\n            payload[\"lifecycle_trace\"] = self.lifecycle_trace.to_payload()\n        if include_digest:\n""",
)
replace_once(
    path,
    """        trace_payload = payload.pop(\"step_trace\", None)\n        trace = (\n            None\n            if trace_payload is None\n            else ActionPathStepTrace.from_payload(trace_payload)\n        )\n        root_digest = str(payload.pop(\"artifact_digest\"))\n""",
    """        trace_payload = payload.pop(\"step_trace\", None)\n        trace = (\n            None\n            if trace_payload is None\n            else ActionPathStepTrace.from_payload(trace_payload)\n        )\n        lifecycle_payload = payload.pop(\"lifecycle_trace\", None)\n        lifecycle = (\n            None\n            if lifecycle_payload is None\n            else ActionPathLifecycleTrace.from_payload(lifecycle_payload)\n        )\n        root_digest = str(payload.pop(\"artifact_digest\"))\n""",
)
replace_once(
    path,
    """            digest=root_digest,\n            step_trace=trace,\n        )\n""",
    """            digest=root_digest,\n            step_trace=trace,\n            lifecycle_trace=lifecycle,\n        )\n""",
)

# V10 owns the lifecycle requirement and bumps only its replay leaf schema.
path = "trade_rl/workflows/universal_causal_alpha_v10_stage_entry.py"
replace_once(
    path,
    '_REPLAY_LEAF_SCHEMA: Final = "causal_alpha_v10_replay_leaf_v2"\n',
    '_REPLAY_LEAF_SCHEMA: Final = "causal_alpha_v10_replay_leaf_v3"\n',
)
replace_once(
    path,
    """    if evaluation.step_trace is None:\n        raise ValueError(\"V10 replay requires per-step action trace\")\n    base = build_causal_alpha_v6_replay_metric(\n""",
    """    if evaluation.step_trace is None:\n        raise ValueError(\"V10 replay requires per-step action trace\")\n    if evaluation.lifecycle_trace is None:\n        raise ValueError(\"V10 replay requires execution lifecycle trace\")\n    if not np.all(evaluation.lifecycle_trace.hard_risk_evidence_available):\n        raise ValueError(\"V10 replay requires authoritative hard-risk evidence\")\n    if target.candidate is CausalAlphaV10Candidate.HIERARCHICAL_WAVE:\n        unexplained = tuple(\n            index\n            for index, (transition, initiator) in enumerate(\n                zip(\n                    evaluation.lifecycle_trace.transition_classes,\n                    evaluation.lifecycle_trace.flatten_initiators,\n                    strict=True,\n                )\n            )\n            if transition == \"exit\" and initiator == \"unexplained\"\n        )\n        if unexplained:\n            raise ValueError(\"V10 hierarchical replay has unexplained flatten transition\")\n    base = build_causal_alpha_v6_replay_metric(\n""",
)
replace_once(
    path,
    """        step_trace=evaluation.step_trace,\n    )\n""",
    """        step_trace=evaluation.step_trace,\n        lifecycle_trace=evaluation.lifecycle_trace,\n    )\n""",
)
replace_once(
    path,
    """        or getattr(metric, \"step_trace\", None) is None\n        or target_payload.get(\"artifact_digest\") != metric.v8_target_path_digest\n""",
    """        or getattr(metric, \"step_trace\", None) is None\n        or getattr(metric, \"lifecycle_trace\", None) is None\n        or not np.all(metric.lifecycle_trace.hard_risk_evidence_available)\n        or target_payload.get(\"artifact_digest\") != metric.v8_target_path_digest\n""",
)

print("phase2b patch applied")
