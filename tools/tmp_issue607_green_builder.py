from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text()
    if text.count(old) != 1:
        raise SystemExit(f"anchor count differs for {path}: {old[:80]!r}")
    target.write_text(text.replace(old, new))


ppo_path = "trade_rl/strategies/rl/ppo.py"
replace_once(
    ppo_path,
    'PPO_OBSERVATION_SCHEMA = "ppo_observation_v2"\nPPO_GLOBAL_FEATURE_NAMES: tuple[str, ...] = ()\n',
    'PPO_OBSERVATION_SCHEMA = "ppo_observation_v2"\n'
    'PPO_GLOBAL_FEATURE_NAMES: tuple[str, ...] = ()\n'
    'PPO_GLOBAL_BTC_REGIME_CONTEXT = "ppo_global_btc_regime_context"\n'
    'PPO_GLOBAL_BTC_REGIME_OBSERVATION_SCHEMA = (\n'
    '    "ppo_observation_v3_global_btc_regime"\n'
    ')\n'
    '_PPO_GLOBAL_BTC_REGIME_REFERENCE_SYMBOL = "BTCUSDT"\n'
    '_PPO_GLOBAL_BTC_REGIME_REFERENCE_FEATURE = "1h__log_return_24bar"\n',
)
replace_once(
    ppo_path,
    '''def ppo_observation_contract_payload() -> dict[str, object]:\n    """Return the frozen semantic PPO observation contract for persisted evidence."""\n\n    return {\n        "schema_version": PPO_OBSERVATION_SCHEMA,\n        "global_feature_names": list(PPO_GLOBAL_FEATURE_NAMES),\n        "includes_local_feature_staleness": True,\n        "layout": [\n            "local_values",\n            "local_available",\n            "local_staleness",\n            "current_intent",\n            "current_weight",\n        ],\n    }\n''',
    '''def _validated_global_context(value: str | None) -> str | None:\n    if value is None:\n        return None\n    if value != PPO_GLOBAL_BTC_REGIME_CONTEXT:\n        raise ValueError(f"unsupported PPO global context: {value}")\n    return value\n\n\ndef ppo_observation_contract_payload(\n    *,\n    global_context: str | None = None,\n) -> dict[str, object]:\n    """Return the frozen semantic PPO observation contract for persisted evidence."""\n\n    context = _validated_global_context(global_context)\n    if context is None:\n        return {\n            "schema_version": PPO_OBSERVATION_SCHEMA,\n            "global_feature_names": list(PPO_GLOBAL_FEATURE_NAMES),\n            "includes_local_feature_staleness": True,\n            "layout": [\n                "local_values",\n                "local_available",\n                "local_staleness",\n                "current_intent",\n                "current_weight",\n            ],\n        }\n    return {\n        "schema_version": PPO_GLOBAL_BTC_REGIME_OBSERVATION_SCHEMA,\n        "global_feature_names": list(PPO_GLOBAL_FEATURE_NAMES),\n        "includes_local_feature_staleness": True,\n        "global_context": context,\n        "reference_symbol": _PPO_GLOBAL_BTC_REGIME_REFERENCE_SYMBOL,\n        "reference_feature": _PPO_GLOBAL_BTC_REGIME_REFERENCE_FEATURE,\n        "layout": [\n            "local_values",\n            "local_available",\n            "local_staleness",\n            "global_reference_value",\n            "global_reference_available_and_finite",\n            "global_reference_normalized_staleness",\n            "current_intent",\n            "current_weight",\n        ],\n    }\n''',
)
replace_once(
    ppo_path,
    '''def _intent_from_action(action: object) -> PositionIntent:\n''',
    '''def _global_btc_regime_indices(dataset: MarketDataset) -> tuple[int, int]:\n    try:\n        symbol_index = dataset.symbols.index(_PPO_GLOBAL_BTC_REGIME_REFERENCE_SYMBOL)\n    except ValueError as error:\n        raise ValueError(\n            "PPO global BTC regime context requires BTCUSDT in the dataset"\n        ) from error\n    try:\n        feature_index = dataset.feature_names.index(\n            _PPO_GLOBAL_BTC_REGIME_REFERENCE_FEATURE\n        )\n    except ValueError as error:\n        raise ValueError(\n            "PPO global BTC regime context requires 1h__log_return_24bar"\n        ) from error\n    return symbol_index, feature_index\n\n\ndef _global_btc_regime_channels(\n    dataset: MarketDataset,\n    *,\n    index: int,\n) -> np.ndarray:\n    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < dataset.n_bars:\n        raise ValueError("PPO global context index is outside the dataset")\n    symbol_index, feature_index = _global_btc_regime_indices(dataset)\n    value = float(dataset.features[index, symbol_index, feature_index])\n    available = bool(dataset.feature_available[index, symbol_index, feature_index])\n    staleness = float(\n        dataset.resolved_array("feature_staleness")[index, symbol_index, feature_index]\n    )\n    if not math.isfinite(staleness) or staleness < 0.0:\n        raise ValueError("PPO global reference staleness must be finite and non-negative")\n    usable = available and math.isfinite(value)\n    channels = np.asarray(\n        [value if usable else 0.0, float(usable), staleness],\n        dtype=np.float64,\n    )\n    channels.setflags(write=False)\n    return channels\n\n\ndef _encode_with_global_context(\n    observation: StrategyObservation,\n    feature_indices: tuple[int, ...],\n    *,\n    dataset: MarketDataset | None,\n    global_context: str | None,\n) -> np.ndarray:\n    context = _validated_global_context(global_context)\n    baseline = _encode_observation(observation, feature_indices)\n    if context is None:\n        return baseline\n    if dataset is None:\n        raise ValueError("PPO global context requires the canonical MarketDataset")\n    if not 0 <= observation.index < dataset.n_bars:\n        raise ValueError("PPO observation index is outside the canonical dataset")\n    if np.datetime64(observation.timestamp, "ns") != np.datetime64(\n        dataset.timestamps[observation.index], "ns"\n    ):\n        raise ValueError("PPO observation timestamp does not match the canonical dataset row")\n    reference = _global_btc_regime_channels(dataset, index=observation.index)\n    encoded = np.concatenate((baseline[:-2], reference, baseline[-2:])).astype(np.float32)\n    encoded.setflags(write=False)\n    return encoded\n\n\ndef _intent_from_action(action: object) -> PositionIntent:\n''',
)
replace_once(
    ppo_path,
    '''    def __init__(\n        self,\n        policy: _PredictPolicy,\n        *,\n        feature_indices: tuple[int, ...],\n    ) -> None:\n        self.policy = policy\n        self.feature_indices = _validated_indices(feature_indices)\n\n    def decide(self, observation: StrategyObservation) -> PositionIntent:\n        encoded = _encode_observation(\n            observation,\n            self.feature_indices,\n        )\n''',
    '''    def __init__(\n        self,\n        policy: _PredictPolicy,\n        *,\n        feature_indices: tuple[int, ...],\n        dataset: MarketDataset | None = None,\n        global_context: str | None = None,\n    ) -> None:\n        self.policy = policy\n        self.feature_indices = _validated_indices(feature_indices)\n        self.global_context = _validated_global_context(global_context)\n        if self.global_context is not None and dataset is None:\n            raise ValueError("PPO global context requires the canonical MarketDataset")\n        if self.global_context is not None:\n            assert dataset is not None\n            _global_btc_regime_indices(dataset)\n        self.dataset = dataset\n\n    def decide(self, observation: StrategyObservation) -> PositionIntent:\n        encoded = _encode_with_global_context(\n            observation,\n            self.feature_indices,\n            dataset=self.dataset,\n            global_context=self.global_context,\n        )\n''',
)
replace_once(
    ppo_path,
    '''        execution_cost: ExecutionCostConfig | None = None,\n    ) -> None:\n''',
    '''        execution_cost: ExecutionCostConfig | None = None,\n        global_context: str | None = None,\n    ) -> None:\n''',
)
replace_once(
    ppo_path,
    '''        self.execution_cost = execution_cost or ExecutionCostConfig.zero()\n\n        observation_size = 3 * len(self.feature_indices) + 2\n''',
    '''        self.execution_cost = execution_cost or ExecutionCostConfig.zero()\n        self.global_context = _validated_global_context(global_context)\n        if self.global_context is not None:\n            _global_btc_regime_indices(dataset)\n\n        observation_size = 3 * len(self.feature_indices) + 2\n        if self.global_context is not None:\n            observation_size += 3\n''',
)
replace_once(
    ppo_path,
    '''    def _encoded_observation(self) -> np.ndarray:\n        return _encode_observation(\n            self._strategy_observation(),\n            self.feature_indices,\n        ).copy()\n''',
    '''    def _encoded_observation(self) -> np.ndarray:\n        return _encode_with_global_context(\n            self._strategy_observation(),\n            self.feature_indices,\n            dataset=self.dataset,\n            global_context=self.global_context,\n        ).copy()\n''',
)
replace_once(
    ppo_path,
    '''    execution_cost: ExecutionCostConfig | None = None,\n) -> PPOIntentStrategy:\n''',
    '''    execution_cost: ExecutionCostConfig | None = None,\n    global_context: str | None = None,\n) -> PPOIntentStrategy:\n''',
)
replace_once(
    ppo_path,
    '''        initial_capital=initial_capital,\n        execution_cost=execution_cost,\n    )\n''',
    '''        initial_capital=initial_capital,\n        execution_cost=execution_cost,\n        global_context=global_context,\n    )\n''',
)
replace_once(
    ppo_path,
    '''    return PPOIntentStrategy(\n        cast(_PredictPolicy, model),\n        feature_indices=indices,\n    )\n''',
    '''    return PPOIntentStrategy(\n        cast(_PredictPolicy, model),\n        feature_indices=indices,\n        dataset=dataset if global_context is not None else None,\n        global_context=global_context,\n    )\n''',
)
replace_once(
    ppo_path,
    '''    "PPO_GLOBAL_FEATURE_NAMES",\n    "PPO_OBSERVATION_SCHEMA",\n''',
    '''    "PPO_GLOBAL_BTC_REGIME_CONTEXT",\n    "PPO_GLOBAL_BTC_REGIME_OBSERVATION_SCHEMA",\n    "PPO_GLOBAL_FEATURE_NAMES",\n    "PPO_OBSERVATION_SCHEMA",\n''',
)

candidate_path = "trade_rl/evaluation/runs/candidate_suite.py"
replace_once(
    candidate_path,
    'from trade_rl.strategies.rl.ppo import fit_ppo_strategy\n',
    'from trade_rl.strategies.rl.ppo import (\n'
    '    PPO_GLOBAL_BTC_REGIME_CONTEXT,\n'
    '    fit_ppo_strategy,\n'
    ')\n',
)
replace_once(
    candidate_path,
    '''    ppo_total_timesteps: int\n    ppo_seed: int = 0\n''',
    '''    ppo_total_timesteps: int\n    ppo_seed: int = 0\n    ppo_global_context: str | None = None\n''',
)
replace_once(
    candidate_path,
    '''        if self.ppo_seed < 0:\n            raise ValueError("ppo_seed must be a non-negative integer")\n        object.__setattr__(self, "feature_indices", indices)\n''',
    '''        if self.ppo_seed < 0:\n            raise ValueError("ppo_seed must be a non-negative integer")\n        if self.ppo_global_context not in (None, PPO_GLOBAL_BTC_REGIME_CONTEXT):\n            raise ValueError(\n                f"unsupported PPO global context: {self.ppo_global_context}"\n            )\n        object.__setattr__(self, "feature_indices", indices)\n''',
)
replace_once(
    candidate_path,
    '''        execution_cost=execution_cost,\n    )\n\n    strategies: dict[str, SingleSymbolStrategy] = {\n''',
    '''        execution_cost=execution_cost,\n        global_context=config.ppo_global_context,\n    )\n\n    strategies: dict[str, SingleSymbolStrategy] = {\n''',
)

config_path = "trade_rl/evaluation/runs/config.py"
replace_once(
    config_path,
    'from trade_rl.evaluation.runs.candidate_suite import LeanCandidateConfig\n',
    'from trade_rl.evaluation.runs.candidate_suite import LeanCandidateConfig\n'
    'from trade_rl.strategies.rl.ppo import PPO_GLOBAL_BTC_REGIME_CONTEXT\n',
)
replace_once(
    config_path,
    '''        "initial_capital",\n    )\n''',
    '''        "initial_capital",\n        "ppo_global_context",\n    )\n''',
)
replace_once(
    config_path,
    '''    gross_budget: float\n    initial_capital: float\n''',
    '''    gross_budget: float\n    initial_capital: float\n    ppo_global_context: str | None = None\n''',
)
replace_once(
    config_path,
    '''        object.__setattr__(self, "gross_budget", gross_budget)\n        object.__setattr__(self, "initial_capital", initial_capital)\n''',
    '''        context = self.ppo_global_context\n        if context is not None:\n            context = _validated_text(context, field="ppo_global_context")\n            if context != PPO_GLOBAL_BTC_REGIME_CONTEXT:\n                raise ValueError(f"unsupported PPO global context: {context}")\n        object.__setattr__(self, "gross_budget", gross_budget)\n        object.__setattr__(self, "initial_capital", initial_capital)\n        object.__setattr__(self, "ppo_global_context", context)\n''',
)
replace_once(
    config_path,
    '''        for name in self.JSON_FIELDS:\n            value = getattr(self, name)\n            if isinstance(value, np.datetime64):\n''',
    '''        for name in self.JSON_FIELDS:\n            value = getattr(self, name)\n            if name == "ppo_global_context" and value is None:\n                continue\n            if isinstance(value, np.datetime64):\n''',
)
replace_once(
    config_path,
    '''        initial_capital=_required_float(raw, "initial_capital"),\n    )\n''',
    '''        initial_capital=_required_float(raw, "initial_capital"),\n        ppo_global_context=(\n            None\n            if raw.get("ppo_global_context") is None\n            else _required_string(raw, "ppo_global_context")\n        ),\n    )\n''',
)
replace_once(
    config_path,
    '''        ppo_total_timesteps=config.ppo_total_timesteps,\n        ppo_seed=config.ppo_seed,\n    )\n''',
    '''        ppo_total_timesteps=config.ppo_total_timesteps,\n        ppo_seed=config.ppo_seed,\n        ppo_global_context=config.ppo_global_context,\n    )\n''',
)

run_path = "trade_rl/evaluation/experiments/contracts/run.py"
replace_once(
    run_path,
    '''from trade_rl.strategies.rl.ppo import (\n    PPO_GLOBAL_FEATURE_NAMES,\n    PPO_OBSERVATION_SCHEMA,\n)\n\n_RESOLVED_RUN_CONFIG_V1 = "resolved_run_config_v1"\n_RESOLVED_RUN_CONFIG_V2 = "resolved_run_config_v2"\n''',
    '''from trade_rl.strategies.rl.ppo import (\n    PPO_GLOBAL_BTC_REGIME_CONTEXT,\n    PPO_GLOBAL_BTC_REGIME_OBSERVATION_SCHEMA,\n    PPO_GLOBAL_FEATURE_NAMES,\n    PPO_OBSERVATION_SCHEMA,\n)\n\n_RESOLVED_RUN_CONFIG_V1 = "resolved_run_config_v1"\n_RESOLVED_RUN_CONFIG_V2 = "resolved_run_config_v2"\n_RESOLVED_RUN_CONFIG_V3 = "resolved_run_config_v3"\n''',
)
replace_once(
    run_path,
    '''    ppo_observation_schema: str | None = None\n    ppo_global_feature_names: tuple[str, ...] = ()\n    schema_version: str = _RESOLVED_RUN_CONFIG_V1\n''',
    '''    ppo_observation_schema: str | None = None\n    ppo_global_feature_names: tuple[str, ...] = ()\n    ppo_global_context: str | None = None\n    schema_version: str = _RESOLVED_RUN_CONFIG_V1\n''',
)
replace_once(
    run_path,
    '''        if schema_version == _RESOLVED_RUN_CONFIG_V1:\n            if self.ppo_observation_schema is not None or self.ppo_global_feature_names:\n                raise ContractViolationError(\n                    "resolved_run_config_v1 must not define a PPO observation contract"\n                )\n            ppo_observation_schema: str | None = None\n            ppo_global_feature_names: tuple[str, ...] = ()\n        elif schema_version == _RESOLVED_RUN_CONFIG_V2:\n''',
    '''        if schema_version == _RESOLVED_RUN_CONFIG_V1:\n            if (\n                self.ppo_observation_schema is not None\n                or self.ppo_global_feature_names\n                or self.ppo_global_context is not None\n            ):\n                raise ContractViolationError(\n                    "resolved_run_config_v1 must not define a PPO observation contract"\n                )\n            ppo_observation_schema: str | None = None\n            ppo_global_feature_names: tuple[str, ...] = ()\n            ppo_global_context: str | None = None\n        elif schema_version in {_RESOLVED_RUN_CONFIG_V2, _RESOLVED_RUN_CONFIG_V3}:\n''',
)
replace_once(
    run_path,
    '''            if ppo_observation_schema != PPO_OBSERVATION_SCHEMA:\n                raise ContractViolationError("unsupported PPO observation schema")\n            if ppo_global_feature_names != PPO_GLOBAL_FEATURE_NAMES:\n                raise ContractViolationError(\n                    "PPO global feature names do not match the frozen observation contract"\n                )\n        else:\n''',
    '''            if ppo_global_feature_names != PPO_GLOBAL_FEATURE_NAMES:\n                raise ContractViolationError(\n                    "PPO global feature names do not match the frozen observation contract"\n                )\n            if schema_version == _RESOLVED_RUN_CONFIG_V2:\n                if ppo_observation_schema != PPO_OBSERVATION_SCHEMA:\n                    raise ContractViolationError("unsupported PPO observation schema")\n                if self.ppo_global_context is not None:\n                    raise ContractViolationError(\n                        "resolved_run_config_v2 must not define PPO global context"\n                    )\n                ppo_global_context = None\n            else:\n                if (\n                    ppo_observation_schema\n                    != PPO_GLOBAL_BTC_REGIME_OBSERVATION_SCHEMA\n                ):\n                    raise ContractViolationError("unsupported PPO observation schema")\n                ppo_global_context = contract_text(\n                    self.ppo_global_context,\n                    field="ppo_global_context",\n                )\n                if ppo_global_context != PPO_GLOBAL_BTC_REGIME_CONTEXT:\n                    raise ContractViolationError("unsupported PPO global context")\n        else:\n''',
)
replace_once(
    run_path,
    '''        object.__setattr__(self, "ppo_global_feature_names", ppo_global_feature_names)\n        object.__setattr__(self, "schema_version", schema_version)\n''',
    '''        object.__setattr__(self, "ppo_global_feature_names", ppo_global_feature_names)\n        object.__setattr__(self, "ppo_global_context", ppo_global_context)\n        object.__setattr__(self, "schema_version", schema_version)\n''',
)
replace_once(
    run_path,
    '''        return cls(\n            signal_name=config.signal_name,\n''',
    '''        candidate_context = config.ppo_global_context\n        observation_schema = (\n            PPO_OBSERVATION_SCHEMA\n            if candidate_context is None\n            else PPO_GLOBAL_BTC_REGIME_OBSERVATION_SCHEMA\n        )\n        schema_version = (\n            _RESOLVED_RUN_CONFIG_V2\n            if candidate_context is None\n            else _RESOLVED_RUN_CONFIG_V3\n        )\n        return cls(\n            signal_name=config.signal_name,\n''',
)
replace_once(
    run_path,
    '''            ppo_observation_schema=PPO_OBSERVATION_SCHEMA,\n            ppo_global_feature_names=PPO_GLOBAL_FEATURE_NAMES,\n            schema_version=_RESOLVED_RUN_CONFIG_V2,\n''',
    '''            ppo_observation_schema=observation_schema,\n            ppo_global_feature_names=PPO_GLOBAL_FEATURE_NAMES,\n            ppo_global_context=candidate_context,\n            schema_version=schema_version,\n''',
)
replace_once(
    run_path,
    '''        if self.schema_version == _RESOLVED_RUN_CONFIG_V2:\n            payload["ppo_observation_schema"] = self.ppo_observation_schema\n            payload["ppo_global_feature_names"] = list(self.ppo_global_feature_names)\n''',
    '''        if self.schema_version in {_RESOLVED_RUN_CONFIG_V2, _RESOLVED_RUN_CONFIG_V3}:\n            payload["ppo_observation_schema"] = self.ppo_observation_schema\n            payload["ppo_global_feature_names"] = list(self.ppo_global_feature_names)\n        if self.schema_version == _RESOLVED_RUN_CONFIG_V3:\n            payload["ppo_global_context"] = self.ppo_global_context\n''',
)

delta_path = "trade_rl/evaluation/experiments/delta.py"
replace_once(
    delta_path,
    'from trade_rl.evaluation.runs import LoadedCandidateRun\n',
    'from trade_rl.evaluation.runs import LoadedCandidateRun\n'
    'from trade_rl.strategies.rl.ppo import PPO_GLOBAL_BTC_REGIME_CONTEXT\n',
)
replace_once(
    delta_path,
    '''                {\n                    ("feature_names",),\n                    ("feature_indices",),\n                }\n''',
    '''                {\n                    ("feature_names",),\n                    ("feature_indices",),\n                    ("schema_version",),\n                    ("ppo_observation_schema",),\n                    ("ppo_global_context",),\n                }\n''',
)
replace_once(
    delta_path,
    '''def _fixed_config_violations(\n    *,\n    plan: StudyPlan,\n    evidence: LoadedEvidenceSet,\n    label: str,\n) -> list[str]:\n''',
    '''def _fixed_config_violations(\n    *,\n    plan: StudyPlan,\n    evidence: LoadedEvidenceSet,\n    label: str,\n    allowed_paths: frozenset[tuple[str, ...]] = frozenset(),\n) -> list[str]:\n''',
)
replace_once(
    delta_path,
    '''    for field in plan.FIXED_RESOLVED_FIELDS:\n        if field not in evidence.semantic_config or (\n''',
    '''    for field in plan.FIXED_RESOLVED_FIELDS:\n        if (field,) in allowed_paths:\n            continue\n        if field not in evidence.semantic_config or (\n''',
)
replace_once(
    delta_path,
    '''def verify_controlled_delta(\n''',
    '''def _rule_for_candidate(\n    *,\n    factor: ControlledFactor,\n    candidate_semantic: Mapping[str, object],\n) -> FactorRule:\n    rule = FACTOR_RULES[factor]\n    if (\n        factor is ControlledFactor.FEATURE_SET\n        and candidate_semantic.get("ppo_global_context")\n        == PPO_GLOBAL_BTC_REGIME_CONTEXT\n    ):\n        return FactorRule(\n            allowed_paths=rule.allowed_paths,\n            unaffected_strategies=frozenset(\n                {\n                    "cash",\n                    "constant_long",\n                    "constant_short",\n                    "trend",\n                    "mean_reversion",\n                    "ridge24",\n                    "lightgbm24",\n                }\n            ),\n        )\n    return rule\n\n\ndef verify_controlled_delta(\n''',
)
replace_once(
    delta_path,
    '''    violations.extend(\n        _fixed_config_violations(plan=plan, evidence=baseline, label="baseline")\n    )\n    violations.extend(\n        _fixed_config_violations(plan=plan, evidence=candidate, label="candidate")\n    )\n\n    baseline_violations, baseline_matrices = _evidence_violations(\n''',
    '''    rule = _rule_for_candidate(\n        factor=definition.factor,\n        candidate_semantic=candidate.semantic_config,\n    )\n    violations.extend(\n        _fixed_config_violations(plan=plan, evidence=baseline, label="baseline")\n    )\n    violations.extend(\n        _fixed_config_violations(\n            plan=plan,\n            evidence=candidate,\n            label="candidate",\n            allowed_paths=rule.allowed_paths,\n        )\n    )\n\n    baseline_violations, baseline_matrices = _evidence_violations(\n''',
)
replace_once(
    delta_path,
    '''    rule = FACTOR_RULES[definition.factor]\n    changed_paths, forbidden_paths = _classify_resolved_delta(\n''',
    '''    changed_paths, forbidden_paths = _classify_resolved_delta(\n''',
)

print("ISSUE607_BUILDER_PATCHED=true")
