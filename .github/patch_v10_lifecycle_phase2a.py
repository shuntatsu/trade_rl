from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    file = Path(path)
    text = file.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one match, got {count}: {old[:100]!r}")
    file.write_text(text.replace(old, new, 1), encoding="utf-8")


path = "trade_rl/rl/environment.py"
replace_once(
    path,
    """        self._pending_hybrid_target = state.pending_hybrid_target\n        self._pending_shadow_target = state.pending_shadow_target\n        self._hybrid_order_book = state.hybrid_order_book\n""",
    """        self._pending_hybrid_target = state.pending_hybrid_target\n        self._pending_shadow_target = state.pending_shadow_target\n        self._pending_hybrid_reduce_only_mask: np.ndarray | None = None\n        self._next_hybrid_reduce_only_mask: np.ndarray | None = None\n        self._hybrid_order_book = state.hybrid_order_book\n""",
)
replace_once(
    path,
    """    def baseline_action(self) -> np.ndarray:\n        \"\"\"Encode the exact shadow baseline in the maintained action schema.\"\"\"\n""",
    """    def set_next_hybrid_reduce_only_mask(self, mask: np.ndarray) -> None:\n        \"\"\"Bind explicit reduce-only intent to the next submitted hybrid target.\"\"\"\n\n        raw = np.asarray(mask)\n        if raw.dtype != np.dtype(np.bool_):\n            raise TypeError(\"hybrid reduce-only mask must contain booleans\")\n        resolved = raw.reshape(-1).copy()\n        if resolved.shape != (self.dataset.n_symbols,):\n            raise ValueError(\"hybrid reduce-only mask must match dataset symbols\")\n        self._next_hybrid_reduce_only_mask = resolved\n\n    def baseline_action(self) -> np.ndarray:\n        \"\"\"Encode the exact shadow baseline in the maintained action schema.\"\"\"\n""",
)
replace_once(
    path,
    """        self._pending_hybrid_target = None\n        self._pending_shadow_target = None\n        self._hybrid_order_book = OrderBookState.empty()\n""",
    """        self._pending_hybrid_target = None\n        self._pending_shadow_target = None\n        self._pending_hybrid_reduce_only_mask = None\n        self._next_hybrid_reduce_only_mask = None\n        self._hybrid_order_book = OrderBookState.empty()\n""",
)
replace_once(
    path,
    """        trends, alpha, factor_basis = self._market_inputs()\n        decision = self._decision_planner.plan(\n            EnvironmentDecisionRequest(\n""",
    """        trends, alpha, factor_basis = self._market_inputs()\n        submitted_reduce_only_mask = (\n            np.zeros(self.dataset.n_symbols, dtype=np.bool_)\n            if self._next_hybrid_reduce_only_mask is None\n            else self._next_hybrid_reduce_only_mask.copy()\n        )\n        self._next_hybrid_reduce_only_mask = None\n        decision = self._decision_planner.plan(\n            EnvironmentDecisionRequest(\n""",
)
replace_once(
    path,
    """                pending_hybrid_target=self._pending_hybrid_target,\n                pending_shadow_target=self._pending_shadow_target,\n                current_index=self.current_index,\n""",
    """                pending_hybrid_target=self._pending_hybrid_target,\n                pending_shadow_target=self._pending_shadow_target,\n                current_index=self.current_index,\n                submitted_hybrid_reduce_only_mask=submitted_reduce_only_mask,\n                pending_hybrid_reduce_only_mask=self._pending_hybrid_reduce_only_mask,\n""",
)
replace_once(
    path,
    """        self._pending_hybrid_target = decision.next_pending_hybrid_target\n        self._pending_shadow_target = decision.next_pending_shadow_target\n        hybrid_risk = self._risk_projector.project(\n""",
    """        self._pending_hybrid_target = decision.next_pending_hybrid_target\n        self._pending_shadow_target = decision.next_pending_shadow_target\n        self._pending_hybrid_reduce_only_mask = (\n            decision.next_pending_hybrid_reduce_only_mask\n        )\n        hybrid_risk = self._risk_projector.project(\n""",
)
replace_once(
    path,
    """                proposal=decision.executed_hybrid_target,\n                book=self.hybrid,\n                current_index=self.current_index,\n            )\n""",
    """                proposal=decision.executed_hybrid_target,\n                book=self.hybrid,\n                current_index=self.current_index,\n                reduce_only_mask=decision.executed_hybrid_reduce_only_mask,\n            )\n""",
)
replace_once(
    path,
    """        if time_limit_reached:\n            self._pending_hybrid_target = None\n            self._pending_shadow_target = None\n""",
    """        if time_limit_reached:\n            self._pending_hybrid_target = None\n            self._pending_shadow_target = None\n            self._pending_hybrid_reduce_only_mask = None\n            self._next_hybrid_reduce_only_mask = None\n""",
)

path = "trade_rl/workflows/universal_causal_alpha_v10_stage_entry.py"
replace_once(
    path,
    """def _build_hierarchical_replay(\n""",
    """class _V10ReduceOnlyEnvironment:\n    \"\"\"Attach V10 hierarchy risk-reduction intent to the matching submission.\"\"\"\n\n    def __init__(self, environment: Any, policy: CausalAlphaV10HierarchyPolicy) -> None:\n        self._environment = environment\n        self._policy = policy\n\n    def __getattr__(self, name: str) -> Any:\n        return getattr(self._environment, name)\n\n    def reset(self, *, options: dict[str, object]) -> tuple[object, dict[str, object]]:\n        return self._environment.reset(options=options)\n\n    def step(self, action: np.ndarray):\n        metadata = self._policy.last_step_trace_metadata\n        raw_reduce_only = metadata.get(\"reduce_only\", False)\n        if not isinstance(raw_reduce_only, bool):\n            raise TypeError(\"V10 reduce-only metadata must be boolean\")\n        setter = getattr(self._environment, \"set_next_hybrid_reduce_only_mask\", None)\n        if not callable(setter):\n            raise TypeError(\"V10 replay environment cannot bind reduce-only intent\")\n        action_vector = np.asarray(action).reshape(-1)\n        setter(np.full(action_vector.shape, raw_reduce_only, dtype=np.bool_))\n        return self._environment.step(action)\n\n\ndef _build_hierarchical_replay(\n""",
)
replace_once(
    path,
    """        evaluation = evaluate_action_path(\n            _InitialStateEnvironment(environment, contract.initial_state_mode),\n            evaluation_range=(contract.start, contract.stop),\n            model=policy,\n            deterministic=True,\n        )\n""",
    """        replay_environment = _V10ReduceOnlyEnvironment(\n            _InitialStateEnvironment(environment, contract.initial_state_mode),\n            policy,\n        )\n        evaluation = evaluate_action_path(\n            replay_environment,\n            evaluation_range=(contract.start, contract.stop),\n            model=policy,\n            deterministic=True,\n        )\n""",
)

print("phase2a patch applied")
