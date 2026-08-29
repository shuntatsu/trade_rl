from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    file = Path(path)
    text = file.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one match, got {count}: {old[:80]!r}")
    file.write_text(text.replace(old, new, 1), encoding="utf-8")


# PreTrade: add explicit, strictly validated reduce-only intent and expose hard-limit metadata.
path = "trade_rl/risk/pretrade.py"
replace_once(
    path,
    """    drawdown_budget: float | None = None\n\n    def __post_init__(self) -> None:\n""",
    """    drawdown_budget: float | None = None\n    max_abs_weight: float | None = None\n    fail_closed_tolerance: float | None = None\n\n    def __post_init__(self) -> None:\n""",
)
replace_once(
    path,
    """        if self.drawdown_budget is not None and (\n            not math.isfinite(self.drawdown_budget)\n            or not 0.0 <= self.drawdown_budget <= 1.0\n        ):\n            raise ValueError(\"drawdown_budget must be within [0, 1] when provided\")\n        object.__setattr__(self, \"weights\", weights)\n""",
    """        if self.drawdown_budget is not None and (\n            not math.isfinite(self.drawdown_budget)\n            or not 0.0 <= self.drawdown_budget <= 1.0\n        ):\n            raise ValueError(\"drawdown_budget must be within [0, 1] when provided\")\n        if self.max_abs_weight is not None and (\n            not math.isfinite(self.max_abs_weight) or self.max_abs_weight <= 0.0\n        ):\n            raise ValueError(\"max_abs_weight must be finite and positive when provided\")\n        if self.fail_closed_tolerance is not None and (\n            not math.isfinite(self.fail_closed_tolerance)\n            or self.fail_closed_tolerance < 0.0\n        ):\n            raise ValueError(\"fail_closed_tolerance must be non-negative when provided\")\n        object.__setattr__(self, \"weights\", weights)\n""",
)
replace_once(
    path,
    """        reasons: list[str],\n        emergency_mask: np.ndarray,\n    ) -> np.ndarray:\n""",
    """        reasons: list[str],\n        emergency_mask: np.ndarray,\n        reduce_only_mask: np.ndarray,\n    ) -> np.ndarray:\n""",
)
replace_once(
    path,
    """            if emergency_mask[index]:\n                controlled[index] = 0.0\n                continue\n            if abs(current) <= _TOLERANCE:\n""",
    """            if emergency_mask[index]:\n                controlled[index] = 0.0\n                continue\n            if reduce_only_mask[index]:\n                continue\n            if abs(current) <= _TOLERANCE:\n""",
)
replace_once(
    path,
    """        small_changes = (\n            np.abs(controlled - existing) < self.config.no_trade_band\n        ) & ~emergency_mask\n""",
    """        small_changes = (\n            np.abs(controlled - existing) < self.config.no_trade_band\n        ) & ~emergency_mask & ~reduce_only_mask\n""",
)
replace_once(
    path,
    """        drawdown: float,\n        emergency_flatten_mask: np.ndarray | None = None,\n    ) -> RiskConstrainedTarget:\n""",
    """        drawdown: float,\n        emergency_flatten_mask: np.ndarray | None = None,\n        reduce_only_mask: np.ndarray | None = None,\n    ) -> RiskConstrainedTarget:\n""",
)
replace_once(
    path,
    """        proposal_weights = requested.copy()\n        emergency_mask = (\n""",
    """        proposal_weights = requested.copy()\n        if reduce_only_mask is None:\n            reduce_mask = np.zeros(requested.shape, dtype=np.bool_)\n        else:\n            raw_reduce_mask = np.asarray(reduce_only_mask)\n            if raw_reduce_mask.dtype != np.dtype(np.bool_):\n                raise TypeError(\"reduce_only_mask must contain booleans\")\n            reduce_mask = raw_reduce_mask.reshape(-1).copy()\n            if reduce_mask.shape != requested.shape:\n                raise ValueError(\"reduce_only_mask must match target weights\")\n        for index in np.flatnonzero(reduce_mask):\n            target = float(requested[index])\n            current = float(existing[index])\n            if abs(current) <= _TOLERANCE:\n                raise ValueError(\"reduce-only target cannot start from flat exposure\")\n            if target * current < -_TOLERANCE:\n                raise ValueError(\"reduce-only target cannot change sign\")\n            if abs(target) > abs(current) + _TOLERANCE:\n                raise ValueError(\"reduce-only target cannot increase exposure\")\n        emergency_mask = (\n""",
)
replace_once(
    path,
    """        if emergency_mask.shape != requested.shape:\n            raise ValueError(\"emergency flatten mask must match target weights\")\n        scale = self.risk_scale(drawdown)\n        reasons: list[str] = []\n""",
    """        if emergency_mask.shape != requested.shape:\n            raise ValueError(\"emergency flatten mask must match target weights\")\n        scale = self.risk_scale(drawdown)\n        reasons: list[str] = []\n        if np.any(reduce_mask):\n            reasons.append(\"reduce_only\")\n""",
)
replace_once(
    path,
    """            reasons=reasons,\n            emergency_mask=emergency_mask,\n        )\n""",
    """            reasons=reasons,\n            emergency_mask=emergency_mask,\n            reduce_only_mask=reduce_mask,\n        )\n""",
)
replace_once(
    path,
    """            max_gross=self.config.max_gross,\n            drawdown_budget=self.config.drawdown_start,\n        )\n""",
    """            max_gross=self.config.max_gross,\n            drawdown_budget=self.config.drawdown_start,\n            max_abs_weight=self.config.max_abs_weight,\n            fail_closed_tolerance=self.config.fail_closed_tolerance,\n        )\n""",
)

# Decision planner: reduce-only metadata follows exactly the same one-decision delay as its target.
path = "trade_rl/rl/environment_decision.py"
replace_once(
    path,
    """    current_index: int\n    end_index: int\n\n\n@dataclass(frozen=True, slots=True)\nclass EnvironmentDecisionPlan:\n""",
    """    current_index: int\n    end_index: int\n    submitted_hybrid_reduce_only_mask: np.ndarray | None = None\n    pending_hybrid_reduce_only_mask: np.ndarray | None = None\n\n\n@dataclass(frozen=True, slots=True)\nclass EnvironmentDecisionPlan:\n""",
)
replace_once(
    path,
    """    next_pending_shadow_target: np.ndarray | None\n    execution_delay_warmup: bool\n""",
    """    next_pending_shadow_target: np.ndarray | None\n    executed_hybrid_reduce_only_mask: np.ndarray\n    next_pending_hybrid_reduce_only_mask: np.ndarray | None\n    execution_delay_warmup: bool\n""",
)
replace_once(
    path,
    """        execution_delay_warmup = False\n        next_pending_hybrid: np.ndarray | None = None\n        next_pending_shadow: np.ndarray | None = None\n        if self.signal_delay_decisions == 0:\n""",
    """        raw_reduce_mask = request.submitted_hybrid_reduce_only_mask\n        if raw_reduce_mask is None:\n            submitted_reduce_mask = np.zeros(submitted_hybrid.shape, dtype=np.bool_)\n        else:\n            raw_array = np.asarray(raw_reduce_mask)\n            if raw_array.dtype != np.dtype(np.bool_):\n                raise TypeError(\"submitted hybrid reduce-only mask must be boolean\")\n            submitted_reduce_mask = raw_array.reshape(-1).copy()\n            if submitted_reduce_mask.shape != submitted_hybrid.shape:\n                raise ValueError(\"submitted hybrid reduce-only mask shape is invalid\")\n        execution_delay_warmup = False\n        next_pending_hybrid: np.ndarray | None = None\n        next_pending_shadow: np.ndarray | None = None\n        next_pending_reduce_mask: np.ndarray | None = None\n        if self.signal_delay_decisions == 0:\n""",
)
replace_once(
    path,
    """            executed_hybrid = submitted_hybrid.copy()\n            executed_shadow = submitted_shadow.copy()\n        else:\n""",
    """            executed_hybrid = submitted_hybrid.copy()\n            executed_shadow = submitted_shadow.copy()\n            executed_reduce_mask = submitted_reduce_mask.copy()\n        else:\n""",
)
replace_once(
    path,
    """            next_pending_hybrid = submitted_hybrid.copy()\n            next_pending_shadow = submitted_shadow.copy()\n        return EnvironmentDecisionPlan(\n""",
    """            if request.pending_hybrid_target is None:\n                executed_reduce_mask = np.zeros(submitted_hybrid.shape, dtype=np.bool_)\n            elif request.pending_hybrid_reduce_only_mask is None:\n                executed_reduce_mask = np.zeros(submitted_hybrid.shape, dtype=np.bool_)\n            else:\n                raw_pending_reduce = np.asarray(request.pending_hybrid_reduce_only_mask)\n                if raw_pending_reduce.dtype != np.dtype(np.bool_):\n                    raise TypeError(\"pending hybrid reduce-only mask must be boolean\")\n                executed_reduce_mask = raw_pending_reduce.reshape(-1).copy()\n                if executed_reduce_mask.shape != submitted_hybrid.shape:\n                    raise ValueError(\"pending hybrid reduce-only mask shape is invalid\")\n            next_pending_hybrid = submitted_hybrid.copy()\n            next_pending_shadow = submitted_shadow.copy()\n            next_pending_reduce_mask = submitted_reduce_mask.copy()\n        return EnvironmentDecisionPlan(\n""",
)
replace_once(
    path,
    """            next_pending_hybrid_target=next_pending_hybrid,\n            next_pending_shadow_target=next_pending_shadow,\n            execution_delay_warmup=execution_delay_warmup,\n""",
    """            next_pending_hybrid_target=next_pending_hybrid,\n            next_pending_shadow_target=next_pending_shadow,\n            executed_hybrid_reduce_only_mask=executed_reduce_mask,\n            next_pending_hybrid_reduce_only_mask=next_pending_reduce_mask,\n            execution_delay_warmup=execution_delay_warmup,\n""",
)

# Risk projector: pass reduce-only intent into PreTrade and expose exact hard-limit metadata.
path = "trade_rl/rl/environment_risk.py"
replace_once(
    path,
    """    proposal: np.ndarray\n    book: BookState\n    current_index: int\n""",
    """    proposal: np.ndarray\n    book: BookState\n    current_index: int\n    reduce_only_mask: np.ndarray | None = None\n""",
)
replace_once(
    path,
    """            drawdown=self.drawdown(request.book),\n            emergency_flatten_mask=assessment.flatten_mask,\n        )\n""",
    """            drawdown=self.drawdown(request.book),\n            emergency_flatten_mask=assessment.flatten_mask,\n            reduce_only_mask=request.reduce_only_mask,\n        )\n""",
)
replace_once(
    path,
    """            max_gross=pretrade.max_gross,\n            drawdown_budget=pretrade.drawdown_budget,\n        )\n""",
    """            max_gross=pretrade.max_gross,\n            drawdown_budget=pretrade.drawdown_budget,\n            max_abs_weight=pretrade.max_abs_weight,\n            fail_closed_tolerance=pretrade.fail_closed_tolerance,\n        )\n""",
)

# V10 hierarchy: never turn a valid same-direction cap reduction into flat just because ordinary no-trade/hysteresis would suppress it.
path = "trade_rl/learning/causal_alpha_v10_hierarchy.py"
replace_once(
    path,
    """            \"position_origin\": trace_origin,\n            \"slow_direction\": int(self._slow_direction[offset]),\n""",
    """            \"position_origin\": trace_origin,\n            \"reduce_only\": hierarchy_reason in {\n                \"exit\",\n                \"neutral_fast_expiry\",\n                \"risk_cap_flatten\",\n                \"risk_cap_projection\",\n            },\n            \"slow_direction\": int(self._slow_direction[offset]),\n""",
)
replace_once(
    path,
    """            partial = float(np.sign(observed_current) * risk_cap)\n            if not self._partial_risk_reduction_executable(\n                observed_current,\n                partial,\n            ):\n                self._risk_flatten_latched = True\n                return self._record(\n                    offset=offset,\n                    observed_current=observed_current,\n                    requested=0.0,\n                    reason=\"risk_projection\",\n                    hierarchy_reason=\"risk_cap_flatten\",\n                )\n            decision_current = partial\n""",
    """            partial = float(np.sign(observed_current) * risk_cap)\n            decision_current = partial\n""",
)
replace_once(
    path,
    """                \"schema_version\": \"causal_alpha_v10_target_compiler_contract_v3\",\n""",
    """                \"reduce_only_execution_contract\": \"explicit_v1\",\n                \"schema_version\": \"causal_alpha_v10_target_compiler_contract_v4\",\n""",
)

# Update the existing regression oracle to the intentionally changed, now-explicit reduce-only behavior.
path = "tests/learning/test_causal_alpha_v10_closed_loop.py"
replace_once(
    path,
    """def test_v10_non_executable_hard_risk_reduction_flattens() -> None:\n""",
    """def test_v10_micro_hard_risk_reduction_projects_to_cap() -> None:\n""",
)
replace_once(
    path,
    """    assert path.targets[16] == 0.10\n    assert path.targets[32] == 0.0\n\n\ndef test_v10_risk_flatten_latch_releases_once_realized_exposure_is_within_cap() -> None:\n""",
    """    assert path.targets[16] == 0.10\n    assert path.targets[32] == 0.04\n    assert path.reasons[32] == \"risk_projection\"\n\n\ndef test_v10_risk_flatten_latch_releases_once_realized_exposure_is_within_cap() -> None:\n""",
)

print("phase1 patch applied")
