from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise SystemExit(f"patch target drifted in {path}: {old[:100]!r}")
    target.write_text(text.replace(old, new), encoding="utf-8")


contracts = "trade_rl/workflows/universal_causal_alpha_contracts.py"
replace_once(
    contracts,
    '''    CausalAlphaRidgeConfig,\n    CausalAlphaRidgeModel,\n)\n''',
    '''    CausalAlphaRidgeConfig,\n    CausalAlphaRidgeModel,\n    CausalAlphaTeacherAdmissionEvidence,\n)\n''',
)
replace_once(
    contracts,
    '''    selection: CausalAlphaSelectionEvidence\n    selected_candidate_digest: str\n''',
    '''    selection: CausalAlphaSelectionEvidence\n    teacher_admission: CausalAlphaTeacherAdmissionEvidence\n    selected_candidate_digest: str\n''',
)
replace_once(
    contracts,
    '''        if self.selection.selected_candidate_digest != self.selected_candidate_digest:\n            raise ValueError("causal alpha package selected candidate identity drifted")\n        if not np.isfinite(self.episode_hours) or self.episode_hours <= 0.0:\n''',
    '''        if self.selection.selected_candidate_digest != self.selected_candidate_digest:\n            raise ValueError("causal alpha package selected candidate identity drifted")\n        if not isinstance(self.teacher_admission, CausalAlphaTeacherAdmissionEvidence):\n            raise TypeError("causal alpha package teacher admission is invalid")\n        if tuple(metric.symbol for metric in self.teacher_admission.metrics) != symbols:\n            raise ValueError("causal alpha package teacher admission symbol scope drifted")\n        if not np.isfinite(self.episode_hours) or self.episode_hours <= 0.0:\n''',
)
replace_once(
    contracts,
    '''                "selection_digest": self.selection.digest,\n                "teacher_config_digest": self.teacher_config_digest,\n''',
    '''                "selection_digest": self.selection.digest,\n                "teacher_admission_digest": self.teacher_admission.digest,\n                "teacher_config_digest": self.teacher_config_digest,\n''',
)

facade = "trade_rl/workflows/universal_causal_alpha_teacher.py"
replace_once(
    facade,
    '''from trade_rl.artifacts.hashing import content_digest\nfrom trade_rl.learning.episode_oracle_bc import evaluate_episode_action_path\n''',
    '''from trade_rl.artifacts.hashing import content_digest\nfrom trade_rl.learning.causal_alpha_teacher import (\n    CausalAlphaTeacherAdmissionEvidence,\n    CausalAlphaTeacherHoldoutMetric,\n    evaluate_causal_alpha_teacher_admission,\n)\nfrom trade_rl.learning.episode_oracle_bc import evaluate_episode_action_path\n''',
)
helper = '''\n\ndef evaluate_causal_alpha_teacher_holdouts(\n    *,\n    train_symbols: tuple[str, ...],\n    batches: Mapping[str, EpisodeOracleBatch],\n    environment_factories: Mapping[str, Any],\n    episode_hours: float,\n) -> CausalAlphaTeacherAdmissionEvidence:\n    \"\"\"Replay each untouched teacher holdout exactly once and freeze admission.\"\"\"\n\n    symbols = tuple(train_symbols)\n    if not symbols or len(set(symbols)) != len(symbols):\n        raise ValueError("causal alpha teacher holdout symbols must be unique")\n    if set(batches) != set(symbols) or set(environment_factories) != set(symbols):\n        raise ValueError("causal alpha teacher holdout scope must match train_symbols")\n    if not np.isfinite(episode_hours) or episode_hours <= 0.0:\n        raise ValueError("causal alpha teacher holdout episode_hours must be positive")\n    episode_days = float(episode_hours) / 24.0\n    metrics: list[CausalAlphaTeacherHoldoutMetric] = []\n    for symbol in symbols:\n        batch = batches[symbol]\n        if not isinstance(batch, EpisodeOracleBatch):\n            raise TypeError("causal alpha teacher holdout batch type is invalid")\n        if not batch.contracts or len(batch.targets) != len(batch.contracts):\n            raise ValueError(f"causal alpha teacher holdout batch is invalid for {symbol}")\n        factory = environment_factories[symbol]\n        if not callable(factory):\n            raise TypeError("causal alpha teacher holdout factory must be callable")\n        evaluation = evaluate_episode_action_path(\n            factory,\n            batch.contracts[-1],\n            actions=batch.targets[-1],\n        )\n        performance = evaluation.performance\n        metrics.append(\n            CausalAlphaTeacherHoldoutMetric(\n                symbol=symbol,\n                gross_return=float(performance.gross_return),\n                net_return=float(performance.net_return),\n                turnover_per_day=float(performance.turnover_total) / episode_days,\n                total_execution_cost=float(performance.cost_total),\n                trade_count=int(performance.trade_count),\n                maximum_drawdown=float(performance.maximum_drawdown),\n            )\n        )\n    return evaluate_causal_alpha_teacher_admission(tuple(metrics))\n'''
replace_once(
    facade,
    '''\ndef build_universal_causal_alpha_teacher_package(\n''',
    helper + '''\n\ndef build_universal_causal_alpha_teacher_package(\n''',
)
replace_once(
    facade,
    '''    binding_by_symbol = {binding.concrete_symbol: binding for binding in binding_values}\n    selection = evaluate_causal_alpha_selection(\n        train_symbols=symbols,\n        samples=samples,\n        partitions=partitions,\n        candidates=candidate_values,\n        environment_factories={\n            symbol: partial(concrete_environment_factory, binding_by_symbol[symbol])\n            for symbol in symbols\n        },\n        episode_hours=resolved_episode_hours,\n    )\n''',
    '''    binding_by_symbol = {binding.concrete_symbol: binding for binding in binding_values}\n    environment_factories = {\n        symbol: partial(concrete_environment_factory, binding_by_symbol[symbol])\n        for symbol in symbols\n    }\n    selection = evaluate_causal_alpha_selection(\n        train_symbols=symbols,\n        samples=samples,\n        partitions=partitions,\n        candidates=candidate_values,\n        environment_factories=environment_factories,\n        episode_hours=resolved_episode_hours,\n    )\n''',
)
replace_once(
    facade,
    '''        batches[symbol] = batch\n        batch_evidence[symbol] = evidence\n    return UniversalCausalAlphaTeacherPackage(\n''',
    '''        batches[symbol] = batch\n        batch_evidence[symbol] = evidence\n    teacher_admission = evaluate_causal_alpha_teacher_holdouts(\n        train_symbols=symbols,\n        batches=batches,\n        environment_factories=environment_factories,\n        episode_hours=resolved_episode_hours,\n    )\n    return UniversalCausalAlphaTeacherPackage(\n''',
)
replace_once(
    facade,
    '''        selection=selection,\n        selected_candidate_digest=selected.digest,\n''',
    '''        selection=selection,\n        teacher_admission=teacher_admission,\n        selected_candidate_digest=selected.digest,\n''',
)
replace_once(
    facade,
    '''    "evaluate_causal_alpha_selection",\n    "fit_expanding_causal_alpha_models",\n''',
    '''    "evaluate_causal_alpha_selection",\n    "evaluate_causal_alpha_teacher_holdouts",\n    "fit_expanding_causal_alpha_models",\n''',
)

pretraining = "trade_rl/integrations/universal_pretraining.py"
replace_once(
    pretraining,
    '''from trade_rl.learning.causal_alpha_teacher import (\n    CausalAlphaTeacherHoldoutMetric,\n    evaluate_causal_alpha_teacher_admission,\n)\n''',
    '''''',
)
replace_once(
    pretraining,
    '''    aggregate_episode_behavior_cloning_holdouts,\n    evaluate_episode_action_path,\n    evaluate_episode_behavior_cloning_holdout,\n''',
    '''    aggregate_episode_behavior_cloning_holdouts,\n    evaluate_episode_behavior_cloning_holdout,\n''',
)
replace_once(
    pretraining,
    '''    causal_teacher_selection_evidence: Mapping[str, object] | None = None\n    causal_teacher_episode_hours: float | None = None\n''',
    '''    causal_teacher_selection_evidence: Mapping[str, object] | None = None\n    causal_teacher_admission_evidence: Mapping[str, object] | None = None\n    causal_teacher_episode_hours: float | None = None\n''',
)
replace_once(
    pretraining,
    '''        selection_evidence = self.causal_teacher_selection_evidence\n        episode_hours = self.causal_teacher_episode_hours\n        if selection_evidence is None:\n            if episode_hours is not None:\n                raise ValueError(\n                    "causal teacher episode hours require selection evidence"\n                )\n        else:\n            selection_evidence = dict(selection_evidence)\n            if (\n                selection_evidence.get("schema_version")\n                != "causal_alpha_selection_evidence_v1"\n            ):\n                raise ValueError("causal teacher selection evidence schema mismatch")\n            artifact_digest = selection_evidence.get("artifact_digest")\n            if not isinstance(artifact_digest, str) or len(artifact_digest) != 64:\n                raise ValueError("causal teacher selection evidence digest is invalid")\n            if (\n                episode_hours is None\n                or not math.isfinite(episode_hours)\n                or episode_hours <= 0.0\n            ):\n                raise ValueError("causal teacher episode hours must be positive")\n''',
    '''        selection_evidence = self.causal_teacher_selection_evidence\n        admission_evidence = self.causal_teacher_admission_evidence\n        episode_hours = self.causal_teacher_episode_hours\n        if selection_evidence is None:\n            if admission_evidence is not None or episode_hours is not None:\n                raise ValueError(\n                    "causal teacher admission/episode hours require selection evidence"\n                )\n        else:\n            selection_evidence = dict(selection_evidence)\n            if (\n                selection_evidence.get("schema_version")\n                != "causal_alpha_selection_evidence_v1"\n            ):\n                raise ValueError("causal teacher selection evidence schema mismatch")\n            artifact_digest = selection_evidence.get("artifact_digest")\n            if not isinstance(artifact_digest, str) or len(artifact_digest) != 64:\n                raise ValueError("causal teacher selection evidence digest is invalid")\n            if admission_evidence is None:\n                raise ValueError("causal teacher admission evidence is unavailable")\n            admission_evidence = dict(admission_evidence)\n            if admission_evidence.get("schema_version") != "causal_alpha_teacher_admission_v1":\n                raise ValueError("causal teacher admission evidence schema mismatch")\n            admission_digest = admission_evidence.get("artifact_digest")\n            if not isinstance(admission_digest, str) or len(admission_digest) != 64:\n                raise ValueError("causal teacher admission evidence digest is invalid")\n            metrics = admission_evidence.get("metrics")\n            if not isinstance(metrics, list) or tuple(\n                item.get("symbol") for item in metrics if isinstance(item, dict)\n            ) != symbols:\n                raise ValueError("causal teacher admission symbol scope mismatch")\n            if (\n                episode_hours is None\n                or not math.isfinite(episode_hours)\n                or episode_hours <= 0.0\n            ):\n                raise ValueError("causal teacher episode hours must be positive")\n''',
)
replace_once(
    pretraining,
    '''        object.__setattr__(\n            self, "causal_teacher_selection_evidence", selection_evidence\n        )\n        object.__setattr__(self, "causal_teacher_episode_hours", episode_hours)\n''',
    '''        object.__setattr__(\n            self, "causal_teacher_selection_evidence", selection_evidence\n        )\n        object.__setattr__(\n            self, "causal_teacher_admission_evidence", admission_evidence\n        )\n        object.__setattr__(self, "causal_teacher_episode_hours", episode_hours)\n''',
)
old_hook = '''        if config.behavior_cloning_teacher == "causal_alpha_ridge":\n            selection = bundle.causal_teacher_selection_evidence\n            episode_hours = bundle.causal_teacher_episode_hours\n            if selection is None or episode_hours is None:\n                raise RuntimeError(\n                    "Universal causal teacher selection evidence is unavailable"\n                )\n            if not bundle.episode_batches:\n                raise RuntimeError(\n                    "Universal causal teacher episode batches are unavailable"\n                )\n            if set(environment_factories) != set(bundle.train_symbols):\n                raise RuntimeError(\n                    "Universal causal teacher holdout environment factories are unavailable"\n                )\n            atomic_write_bytes(\n                output_root / "causal-teacher-selection.json",\n                canonical_json_bytes(selection) + b"\\n",\n            )\n            episode_days = episode_hours / 24.0\n            teacher_metrics: list[CausalAlphaTeacherHoldoutMetric] = []\n            for symbol in bundle.train_symbols:\n                batch = bundle.episode_batches[symbol]\n                if not batch.contracts or len(batch.targets) != len(batch.contracts):\n                    raise RuntimeError(\n                        f"Universal causal teacher holdout batch is invalid for {symbol}"\n                    )\n                evaluation = evaluate_episode_action_path(\n                    environment_factories[symbol],\n                    batch.contracts[-1],\n                    actions=batch.targets[-1],\n                )\n                performance = evaluation.performance\n                teacher_metrics.append(\n                    CausalAlphaTeacherHoldoutMetric(\n                        symbol=symbol,\n                        gross_return=float(performance.gross_return),\n                        net_return=float(performance.net_return),\n                        turnover_per_day=float(performance.turnover_total)\n                        / episode_days,\n                        total_execution_cost=float(performance.cost_total),\n                        trade_count=int(performance.trade_count),\n                        maximum_drawdown=float(performance.maximum_drawdown),\n                    )\n                )\n            teacher_admission = evaluate_causal_alpha_teacher_admission(\n                tuple(teacher_metrics)\n            )\n            atomic_write_bytes(\n                output_root / "causal-teacher-admission.json",\n                canonical_json_bytes(teacher_admission.to_payload()) + b"\\n",\n            )\n            if not teacher_admission.passed:\n                raise RuntimeError(\n                    "Universal causal teacher admission failed before behavior cloning"\n                )\n'''
new_hook = '''        if config.behavior_cloning_teacher == "causal_alpha_ridge":\n            selection = bundle.causal_teacher_selection_evidence\n            admission = bundle.causal_teacher_admission_evidence\n            if selection is None or admission is None:\n                raise RuntimeError(\n                    "Universal causal teacher selection/admission evidence is unavailable"\n                )\n            atomic_write_bytes(\n                output_root / "causal-teacher-selection.json",\n                canonical_json_bytes(selection) + b"\\n",\n            )\n            atomic_write_bytes(\n                output_root / "causal-teacher-admission.json",\n                canonical_json_bytes(admission) + b"\\n",\n            )\n            if admission.get("passed") is not True:\n                raise RuntimeError(\n                    "Universal causal teacher admission failed before behavior cloning"\n                )\n'''
replace_once(pretraining, old_hook, new_hook)

runtime = "trade_rl/workflows/universal_teacher_runtime.py"
replace_once(
    runtime,
    '''        causal_teacher_episode_hours=(\n            None\n            if causal_teacher_package is None\n            else causal_teacher_package.episode_hours\n        ),\n''',
    '''        causal_teacher_admission_evidence=(\n            None\n            if causal_teacher_package is None\n            else causal_teacher_package.teacher_admission.to_payload()\n        ),\n        causal_teacher_episode_hours=(\n            None\n            if causal_teacher_package is None\n            else causal_teacher_package.episode_hours\n        ),\n''',
)

# Keep older bundle test doubles aligned with the new immutable evidence field.
runtime_test = "tests/workflows/test_universal_teacher_bundle_runtime.py"
replace_once(
    runtime_test,
    '''        causal_teacher_selection_evidence: dict[str, object] | None = None\n        causal_teacher_episode_hours: float | None = None\n''',
    '''        causal_teacher_selection_evidence: dict[str, object] | None = None\n        causal_teacher_admission_evidence: dict[str, object] | None = None\n        causal_teacher_episode_hours: float | None = None\n''',
)
