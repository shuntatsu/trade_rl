from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise SystemExit(f"patch target drifted in {path}: {old[:120]!r}")
    target.write_text(text.replace(old, new), encoding="utf-8")


contracts = "trade_rl/workflows/universal_causal_alpha_contracts.py"
replace_once(
    contracts,
    "from dataclasses import dataclass\n",
    "import math\n\nfrom dataclasses import dataclass\n",
)
replace_once(
    contracts,
    '''class CausalAlphaExpandingFit:\n''',
    '''class CausalAlphaExpandingFit:\n''',
)
# Add compact fit payload before prediction evidence definitions.
replace_once(
    contracts,
    '''        object.__setattr__(self, "digest", expected)\n\n\n@dataclass(frozen=True, slots=True)\nclass CausalAlphaEpisodeEvidence:\n    episode_index: int\n    knowledge_cutoff: int\n    initial_weight: float\n    fit_digest: str\n    max_label_end_24h: int\n    max_label_end_72h: int\n    target_path_digest: str\n    prediction_digest: str\n\n\n@dataclass(frozen=True, slots=True)\nclass CausalAlphaBatchEvidence:\n''',
    '''        object.__setattr__(self, "digest", expected)\n\n    def to_payload(self) -> dict[str, object]:\n        def compact_model(model: CausalAlphaRidgeModel) -> dict[str, object]:\n            payload = dict(model.to_payload())\n            payload.pop("eligible_indices", None)\n            payload["artifact_digest"] = model.digest\n            return payload\n\n        return {\n            "artifact_digest": self.digest,\n            "knowledge_cutoff": self.knowledge_cutoff,\n            "max_label_end_24h": self.max_label_end_24h,\n            "max_label_end_72h": self.max_label_end_72h,\n            "model_24h": compact_model(self.model_24h),\n            "model_72h": compact_model(self.model_72h),\n            "sample_count_24h": self.sample_count_24h,\n            "sample_count_72h": self.sample_count_72h,\n            "sample_scope_digest": self.sample_scope_digest,\n            "schema_version": _CAUSAL_ALPHA_EXPANDING_FIT_SCHEMA,\n            "train_symbols": list(self.train_symbols),\n        }\n\n\n@dataclass(frozen=True, slots=True)\nclass CausalAlphaPredictionDiagnostics:\n    sample_count: int\n    pearson_correlation: float | None\n    directional_accuracy: float\n    prediction_mean: float\n    prediction_std: float\n    prediction_min: float\n    prediction_max: float\n    prediction_quantiles: tuple[float, float, float, float, float, float, float]\n    digest: str = ""\n\n    def __post_init__(self) -> None:\n        if isinstance(self.sample_count, bool) or not isinstance(self.sample_count, int) or self.sample_count <= 0:\n            raise ValueError("causal alpha prediction diagnostics need positive samples")\n        if self.pearson_correlation is not None and (\n            not math.isfinite(self.pearson_correlation)\n            or not -1.0 <= self.pearson_correlation <= 1.0\n        ):\n            raise ValueError("causal alpha prediction correlation is invalid")\n        if not math.isfinite(self.directional_accuracy) or not 0.0 <= self.directional_accuracy <= 1.0:\n            raise ValueError("causal alpha directional accuracy is invalid")\n        scalars = (\n            self.prediction_mean,\n            self.prediction_std,\n            self.prediction_min,\n            self.prediction_max,\n            *self.prediction_quantiles,\n        )\n        if not all(math.isfinite(value) for value in scalars):\n            raise ValueError("causal alpha prediction distribution is non-finite")\n        if self.prediction_std < 0.0 or self.prediction_min > self.prediction_max:\n            raise ValueError("causal alpha prediction distribution is invalid")\n        if len(self.prediction_quantiles) != 7 or any(\n            left > right\n            for left, right in zip(\n                self.prediction_quantiles, self.prediction_quantiles[1:], strict=False\n            )\n        ):\n            raise ValueError("causal alpha prediction quantiles are invalid")\n        payload = self._payload_without_digest()\n        expected = content_digest(payload)\n        if self.digest and self.digest != expected:\n            raise ValueError("causal alpha prediction diagnostics digest mismatch")\n        object.__setattr__(self, "digest", expected)\n\n    def _payload_without_digest(self) -> dict[str, object]:\n        quantiles = self.prediction_quantiles\n        return {\n            "directional_accuracy": self.directional_accuracy,\n            "pearson_correlation": self.pearson_correlation,\n            "prediction_max": self.prediction_max,\n            "prediction_mean": self.prediction_mean,\n            "prediction_min": self.prediction_min,\n            "prediction_quantiles": {\n                "p00": quantiles[0],\n                "p10": quantiles[1],\n                "p25": quantiles[2],\n                "p50": quantiles[3],\n                "p75": quantiles[4],\n                "p90": quantiles[5],\n                "p100": quantiles[6],\n            },\n            "prediction_std": self.prediction_std,\n            "sample_count": self.sample_count,\n            "schema_version": "causal_alpha_prediction_diagnostics_v1",\n        }\n\n    def to_payload(self) -> dict[str, object]:\n        return {**self._payload_without_digest(), "artifact_digest": self.digest}\n\n\n@dataclass(frozen=True, slots=True)\nclass CausalAlphaEpisodeEvidence:\n    episode_index: int\n    scope: str\n    knowledge_cutoff: int\n    initial_weight: float\n    fit_digest: str\n    fit_sample_count_24h: int\n    fit_sample_count_72h: int\n    max_label_end_24h: int\n    max_label_end_72h: int\n    prediction_24h: CausalAlphaPredictionDiagnostics\n    prediction_72h: CausalAlphaPredictionDiagnostics\n    decision_count: int\n    actionable_decision_count: int\n    submitted_change_count: int\n    suppressed_change_count: int\n    sign_flip_count: int\n    target_path_digest: str\n    prediction_digest: str\n    digest: str = ""\n\n    def __post_init__(self) -> None:\n        if self.scope not in {"selection", "holdout"}:\n            raise ValueError("causal alpha episode evidence scope is invalid")\n        for field in (\n            "fit_sample_count_24h",\n            "fit_sample_count_72h",\n            "decision_count",\n            "actionable_decision_count",\n            "submitted_change_count",\n            "suppressed_change_count",\n            "sign_flip_count",\n        ):\n            value = getattr(self, field)\n            if isinstance(value, bool) or not isinstance(value, int) or value < 0:\n                raise ValueError(f"causal alpha episode {field} is invalid")\n        if self.fit_sample_count_24h < 2 or self.fit_sample_count_72h < 2:\n            raise ValueError("causal alpha episode fit sample count is insufficient")\n        if self.decision_count <= 0 or not 0 < self.actionable_decision_count <= self.decision_count:\n            raise ValueError("causal alpha episode decision support is invalid")\n        if self.max_label_end_24h >= self.knowledge_cutoff or self.max_label_end_72h >= self.knowledge_cutoff:\n            raise ValueError("causal alpha episode fit crossed knowledge cutoff")\n        for field in ("fit_digest", "target_path_digest", "prediction_digest"):\n            value = getattr(self, field)\n            if not isinstance(value, str) or len(value) != 64:\n                raise ValueError(f"causal alpha episode {field} is invalid")\n        expected = content_digest(self._payload_without_digest())\n        if self.digest and self.digest != expected:\n            raise ValueError("causal alpha episode evidence digest mismatch")\n        object.__setattr__(self, "digest", expected)\n\n    def _payload_without_digest(self) -> dict[str, object]:\n        return {\n            "actionable_decision_count": self.actionable_decision_count,\n            "decision_count": self.decision_count,\n            "episode_index": self.episode_index,\n            "fit_digest": self.fit_digest,\n            "fit_sample_count_24h": self.fit_sample_count_24h,\n            "fit_sample_count_72h": self.fit_sample_count_72h,\n            "initial_weight": self.initial_weight,\n            "knowledge_cutoff": self.knowledge_cutoff,\n            "max_label_end_24h": self.max_label_end_24h,\n            "max_label_end_72h": self.max_label_end_72h,\n            "prediction_24h": self.prediction_24h.to_payload(),\n            "prediction_72h": self.prediction_72h.to_payload(),\n            "prediction_digest": self.prediction_digest,\n            "schema_version": "causal_alpha_episode_evidence_v2",\n            "scope": self.scope,\n            "sign_flip_count": self.sign_flip_count,\n            "submitted_change_count": self.submitted_change_count,\n            "suppressed_change_count": self.suppressed_change_count,\n            "target_path_digest": self.target_path_digest,\n        }\n\n    def to_payload(self) -> dict[str, object]:\n        return {**self._payload_without_digest(), "artifact_digest": self.digest}\n\n\n@dataclass(frozen=True, slots=True)\nclass CausalAlphaBatchEvidence:\n''',
)
replace_once(
    contracts,
    '''    controller_config_digest: str\n    episodes: tuple[CausalAlphaEpisodeEvidence, ...]\n    digest: str = ""\n''',
    '''    controller_config_digest: str\n    fits: Mapping[str, CausalAlphaExpandingFit]\n    episodes: tuple[CausalAlphaEpisodeEvidence, ...]\n    digest: str = ""\n''',
)
replace_once(
    contracts,
    '''    def __post_init__(self) -> None:\n        if not self.episodes:\n            raise ValueError("causal alpha batch evidence must contain episodes")\n        expected = content_digest(\n            {\n                "controller_config_digest": self.controller_config_digest,\n                "episodes": tuple(\n                    {\n                        "episode_index": item.episode_index,\n                        "fit_digest": item.fit_digest,\n                        "initial_weight": item.initial_weight,\n                        "knowledge_cutoff": item.knowledge_cutoff,\n                        "max_label_end_24h": item.max_label_end_24h,\n                        "max_label_end_72h": item.max_label_end_72h,\n                        "prediction_digest": item.prediction_digest,\n                        "target_path_digest": item.target_path_digest,\n                    }\n                    for item in self.episodes\n                ),\n                "partition_digest": self.partition_digest,\n                "ridge_config_digest": self.ridge_config_digest,\n                "sample_scope_digest": self.sample_scope_digest,\n                "schema_version": _CAUSAL_ALPHA_BATCH_EVIDENCE_SCHEMA,\n                "symbol": self.symbol,\n                "train_symbols": self.train_symbols,\n            }\n        )\n        if self.digest and self.digest != expected:\n            raise ValueError("causal alpha batch evidence digest mismatch")\n        object.__setattr__(self, "digest", expected)\n''',
    '''    def __post_init__(self) -> None:\n        episodes = tuple(self.episodes)\n        fits = dict(self.fits)\n        if not episodes:\n            raise ValueError("causal alpha batch evidence must contain episodes")\n        referenced = {item.fit_digest for item in episodes}\n        if set(fits) != referenced or any(\n            digest != fit.digest for digest, fit in fits.items()\n        ):\n            raise ValueError("causal alpha batch fit evidence does not close over episodes")\n        expected = content_digest(\n            {\n                "controller_config_digest": self.controller_config_digest,\n                "episode_evidence_digests": tuple(item.digest for item in episodes),\n                "fit_digests": tuple(sorted(fits)),\n                "partition_digest": self.partition_digest,\n                "ridge_config_digest": self.ridge_config_digest,\n                "sample_scope_digest": self.sample_scope_digest,\n                "schema_version": _CAUSAL_ALPHA_BATCH_EVIDENCE_SCHEMA,\n                "symbol": self.symbol,\n                "train_symbols": self.train_symbols,\n            }\n        )\n        if self.digest and self.digest != expected:\n            raise ValueError("causal alpha batch evidence digest mismatch")\n        object.__setattr__(self, "episodes", episodes)\n        object.__setattr__(self, "fits", fits)\n        object.__setattr__(self, "digest", expected)\n\n    def to_payload(self) -> dict[str, object]:\n        return {\n            "artifact_digest": self.digest,\n            "controller_config_digest": self.controller_config_digest,\n            "episodes": [item.to_payload() for item in self.episodes],\n            "fits": {digest: self.fits[digest].to_payload() for digest in sorted(self.fits)},\n            "partition_digest": self.partition_digest,\n            "ridge_config_digest": self.ridge_config_digest,\n            "sample_scope_digest": self.sample_scope_digest,\n            "schema_version": _CAUSAL_ALPHA_BATCH_EVIDENCE_SCHEMA,\n            "symbol": self.symbol,\n            "train_symbols": list(self.train_symbols),\n        }\n''',
)
replace_once(
    contracts,
    '''    teacher_config_digest: str\n    episode_hours: float\n''',
    '''    teacher_config_digest: str\n    generator_code_digest: str\n    episode_hours: float\n''',
)
replace_once(
    contracts,
    '''            ("selected_candidate_digest", self.selected_candidate_digest),\n            ("teacher_config_digest", self.teacher_config_digest),\n''',
    '''            ("selected_candidate_digest", self.selected_candidate_digest),\n            ("teacher_config_digest", self.teacher_config_digest),\n            ("generator_code_digest", self.generator_code_digest),\n''',
)
replace_once(
    contracts,
    '''                "episode_hours": self.episode_hours,\n                "schema_version": "universal_causal_alpha_teacher_package_v1",\n''',
    '''                "episode_hours": self.episode_hours,\n                "generator_code_digest": self.generator_code_digest,\n                "schema_version": "universal_causal_alpha_teacher_package_v1",\n''',
)
replace_once(
    contracts,
    '''        object.__setattr__(self, "batch_evidence", batch_evidence)\n        object.__setattr__(self, "digest", expected)\n\n\n__all__ = [\n''',
    '''        object.__setattr__(self, "batch_evidence", batch_evidence)\n        object.__setattr__(self, "digest", expected)\n\n    def to_payload(self) -> dict[str, object]:\n        return {\n            "artifact_digest": self.digest,\n            "batch_evidence": {\n                symbol: self.batch_evidence[symbol].to_payload()\n                for symbol in self.train_symbols\n            },\n            "episode_hours": self.episode_hours,\n            "generator_code_digest": self.generator_code_digest,\n            "partitions": {\n                symbol: {\n                    "artifact_digest": self.partitions[symbol].digest,\n                    "holdout_episode_digest": self.partitions[symbol].holdout_contract.digest,\n                    "selection_episode_digests": [\n                        contract.digest\n                        for contract in self.partitions[symbol].selection_contracts\n                    ],\n                }\n                for symbol in self.train_symbols\n            },\n            "samples": {\n                symbol: {\n                    "artifact_digest": self.samples[symbol].digest,\n                    "context_digest": self.samples[symbol].context_digest,\n                    "feature_schema_digest": self.samples[symbol].feature_schema_digest,\n                    "reference_equity": self.samples[symbol].reference_equity,\n                    "reference_equity_mode": self.samples[symbol].reference_equity_mode,\n                }\n                for symbol in self.train_symbols\n            },\n            "schema_version": "universal_causal_alpha_teacher_package_evidence_v1",\n            "selected_candidate_digest": self.selected_candidate_digest,\n            "selection": self.selection.to_payload(),\n            "teacher_admission": self.teacher_admission.to_payload(),\n            "teacher_config_digest": self.teacher_config_digest,\n            "train_symbols": list(self.train_symbols),\n        }\n\n\n__all__ = [\n''',
)
replace_once(
    contracts,
    '''    "CausalAlphaEpisodePartition",\n    "CausalAlphaExpandingFit",\n''',
    '''    "CausalAlphaEpisodePartition",\n    "CausalAlphaExpandingFit",\n    "CausalAlphaPredictionDiagnostics",\n''',
)

fitting = "trade_rl/workflows/universal_causal_alpha_fitting.py"
replace_once(
    fitting,
    '''    CausalAlphaExpandingFit,\n    CausalAlphaSymbolSamples,\n)\n''',
    '''    CausalAlphaExpandingFit,\n    CausalAlphaPredictionDiagnostics,\n    CausalAlphaSymbolSamples,\n)\n''',
)
insert = '''\n\ndef causal_alpha_prediction_diagnostics(\n    predictions: object, labels: object\n) -> CausalAlphaPredictionDiagnostics:\n    predicted = np.asarray(predictions, dtype=np.float64).reshape(-1)\n    realized = np.asarray(labels, dtype=np.float64).reshape(-1)\n    if predicted.shape != realized.shape or predicted.size == 0:\n        raise ValueError("causal alpha prediction diagnostics require aligned samples")\n    if not np.isfinite(predicted).all() or not np.isfinite(realized).all():\n        raise ValueError("causal alpha prediction diagnostics require finite samples")\n    predicted_std = float(predicted.std())\n    realized_std = float(realized.std())\n    correlation = (\n        None\n        if predicted.size < 2 or predicted_std <= 1e-12 or realized_std <= 1e-12\n        else float(np.corrcoef(predicted, realized)[0, 1])\n    )\n    directional = float(np.mean(np.sign(predicted) == np.sign(realized)))\n    quantiles = tuple(\n        float(value)\n        for value in np.quantile(\n            predicted, np.asarray((0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0))\n        )\n    )\n    return CausalAlphaPredictionDiagnostics(\n        sample_count=int(predicted.size),\n        pearson_correlation=correlation,\n        directional_accuracy=directional,\n        prediction_mean=float(predicted.mean()),\n        prediction_std=predicted_std,\n        prediction_min=float(predicted.min()),\n        prediction_max=float(predicted.max()),\n        prediction_quantiles=quantiles,\n    )\n'''
replace_once(
    fitting,
    '''\ndef build_causal_alpha_episode_batch(\n''',
    insert + '''\n\ndef build_causal_alpha_episode_batch(\n''',
)
replace_once(
    fitting,
    '''    targets: list[np.ndarray] = []\n    episode_evidence: list[CausalAlphaEpisodeEvidence] = []\n''',
    '''    targets: list[np.ndarray] = []\n    episode_evidence: list[CausalAlphaEpisodeEvidence] = []\n    fits: dict[str, CausalAlphaExpandingFit] = {}\n''',
)
replace_once(
    fitting,
    '''        scores = combine_causal_alpha_predictions(\n            prediction_24h,\n            prediction_72h,\n            controller_config.horizon_mix,\n        )\n''',
    '''        scores = combine_causal_alpha_predictions(\n            prediction_24h,\n            prediction_72h,\n            controller_config.horizon_mix,\n        )\n        positions = np.searchsorted(block.decision_indices, decisions)\n        diagnostic_positions = positions[actionable]\n        diagnostics_24h = causal_alpha_prediction_diagnostics(\n            prediction_24h[actionable], block.labels_24h[diagnostic_positions]\n        )\n        diagnostics_72h = causal_alpha_prediction_diagnostics(\n            prediction_72h[actionable], block.labels_72h[diagnostic_positions]\n        )\n''',
)
replace_once(
    fitting,
    '''        targets.append(target_matrix)\n        episode_evidence.append(\n            CausalAlphaEpisodeEvidence(\n                episode_index=contract.episode_index,\n                knowledge_cutoff=contract.start,\n                initial_weight=initial_weight,\n                fit_digest=fitted.digest,\n                max_label_end_24h=fitted.max_label_end_24h,\n                max_label_end_72h=fitted.max_label_end_72h,\n                target_path_digest=target_path.digest,\n                prediction_digest=prediction_digest,\n            )\n        )\n''',
    '''        targets.append(target_matrix)\n        fits[fitted.digest] = fitted\n        episode_evidence.append(\n            CausalAlphaEpisodeEvidence(\n                episode_index=contract.episode_index,\n                scope=(\n                    "holdout"\n                    if contract.episode_index == partition.holdout_contract.episode_index\n                    else "selection"\n                ),\n                knowledge_cutoff=contract.start,\n                initial_weight=initial_weight,\n                fit_digest=fitted.digest,\n                fit_sample_count_24h=fitted.sample_count_24h,\n                fit_sample_count_72h=fitted.sample_count_72h,\n                max_label_end_24h=fitted.max_label_end_24h,\n                max_label_end_72h=fitted.max_label_end_72h,\n                prediction_24h=diagnostics_24h,\n                prediction_72h=diagnostics_72h,\n                decision_count=int(decisions.size),\n                actionable_decision_count=int(np.count_nonzero(actionable)),\n                submitted_change_count=target_path.submitted_change_count,\n                suppressed_change_count=target_path.suppressed_change_count,\n                sign_flip_count=target_path.sign_flip_count,\n                target_path_digest=target_path.digest,\n                prediction_digest=prediction_digest,\n            )\n        )\n''',
)
replace_once(
    fitting,
    '''        controller_config_digest=controller_config.digest,\n        episodes=tuple(episode_evidence),\n''',
    '''        controller_config_digest=controller_config.digest,\n        fits=fits,\n        episodes=tuple(episode_evidence),\n''',
)
replace_once(
    fitting,
    '''    "build_chronological_episode_partition",\n    "fit_expanding_causal_alpha_models",\n''',
    '''    "build_chronological_episode_partition",\n    "causal_alpha_prediction_diagnostics",\n    "fit_expanding_causal_alpha_models",\n''',
)

facade = "trade_rl/workflows/universal_causal_alpha_teacher.py"
replace_once(
    facade,
    '''from __future__ import annotations\n\nfrom functools import partial\nfrom typing import Any, Mapping\n''',
    '''from __future__ import annotations\n\nimport hashlib\nfrom functools import partial\nfrom pathlib import Path\nfrom typing import Any, Mapping\n\nimport trade_rl.learning.causal_alpha_teacher as _causal_learning_module\nimport trade_rl.workflows.universal_causal_alpha_contracts as _causal_contracts_module\nimport trade_rl.workflows.universal_causal_alpha_fitting as _causal_fitting_module\nimport trade_rl.workflows.universal_causal_alpha_selection as _causal_selection_module\n''',
)
replace_once(
    facade,
    '''    CausalAlphaExpandingFit,\n    CausalAlphaSelectionEvidence,\n''',
    '''    CausalAlphaExpandingFit,\n    CausalAlphaPredictionDiagnostics,\n    CausalAlphaSelectionEvidence,\n''',
)
code_helper = '''\n\ndef causal_alpha_generator_code_digest() -> str:\n    \"\"\"Bind teacher identity to the exact causal-generator source files.\"\"\"\n\n    modules = (\n        _causal_learning_module,\n        _causal_contracts_module,\n        _causal_fitting_module,\n        _causal_selection_module,\n    )\n    files: dict[str, str] = {}\n    for module in modules:\n        raw_path = getattr(module, "__file__", None)\n        if not isinstance(raw_path, str):\n            raise RuntimeError("causal alpha generator source path is unavailable")\n        path = Path(raw_path)\n        files[module.__name__] = hashlib.sha256(path.read_bytes()).hexdigest()\n    files[__name__] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()\n    return content_digest(\n        {\n            "files": files,\n            "schema_version": "universal_causal_alpha_generator_code_v1",\n        }\n    )\n'''
replace_once(
    facade,
    '''\ndef evaluate_causal_alpha_teacher_holdouts(\n''',
    code_helper + '''\n\ndef evaluate_causal_alpha_teacher_holdouts(\n''',
)
replace_once(
    facade,
    '''    selected = selected_evidence[0].candidate\n    teacher_config_digest = content_digest(\n        {\n            "feature_schema_digest": feature_schema_digest,\n''',
    '''    selected = selected_evidence[0].candidate\n    generator_code_digest = causal_alpha_generator_code_digest()\n    teacher_config_digest = content_digest(\n        {\n            "feature_schema_digest": feature_schema_digest,\n            "generator_code_digest": generator_code_digest,\n''',
)
replace_once(
    facade,
    '''        teacher_config_digest=teacher_config_digest,\n        episode_hours=resolved_episode_hours,\n''',
    '''        teacher_config_digest=teacher_config_digest,\n        generator_code_digest=generator_code_digest,\n        episode_hours=resolved_episode_hours,\n''',
)
replace_once(
    facade,
    '''    "build_universal_causal_alpha_teacher_package",\n    "default_causal_alpha_candidate_grid",\n''',
    '''    "build_universal_causal_alpha_teacher_package",\n    "causal_alpha_generator_code_digest",\n    "default_causal_alpha_candidate_grid",\n''',
)
replace_once(
    facade,
    '''    "CausalAlphaExpandingFit",\n    "CausalAlphaSelectionEvidence",\n''',
    '''    "CausalAlphaExpandingFit",\n    "CausalAlphaPredictionDiagnostics",\n    "CausalAlphaSelectionEvidence",\n''',
)

runtime = "trade_rl/workflows/universal_teacher_runtime.py"
replace_once(
    runtime,
    '''        causal_teacher_admission_evidence=(\n            None\n            if causal_teacher_package is None\n            else causal_teacher_package.teacher_admission.to_payload()\n        ),\n''',
    '''        causal_teacher_admission_evidence=(\n            None\n            if causal_teacher_package is None\n            else causal_teacher_package.teacher_admission.to_payload()\n        ),\n        causal_teacher_package_evidence=(\n            None\n            if causal_teacher_package is None\n            else causal_teacher_package.to_payload()\n        ),\n''',
)

pretraining = "trade_rl/integrations/universal_pretraining.py"
replace_once(
    pretraining,
    '''    causal_teacher_admission_evidence: Mapping[str, object] | None = None\n    causal_teacher_episode_hours: float | None = None\n''',
    '''    causal_teacher_admission_evidence: Mapping[str, object] | None = None\n    causal_teacher_package_evidence: Mapping[str, object] | None = None\n    causal_teacher_episode_hours: float | None = None\n''',
)
replace_once(
    pretraining,
    '''        admission_evidence = self.causal_teacher_admission_evidence\n        episode_hours = self.causal_teacher_episode_hours\n''',
    '''        admission_evidence = self.causal_teacher_admission_evidence\n        package_evidence = self.causal_teacher_package_evidence\n        episode_hours = self.causal_teacher_episode_hours\n''',
)
replace_once(
    pretraining,
    '''            if admission_evidence is not None or episode_hours is not None:\n                raise ValueError(\n                    "causal teacher admission/episode hours require selection evidence"\n                )\n''',
    '''            if (\n                admission_evidence is not None\n                or package_evidence is not None\n                or episode_hours is not None\n            ):\n                raise ValueError(\n                    "causal teacher package/admission/episode hours require selection evidence"\n                )\n''',
)
replace_once(
    pretraining,
    '''            if admission_evidence is None:\n                raise ValueError("causal teacher admission evidence is unavailable")\n            admission_evidence = dict(admission_evidence)\n''',
    '''            if admission_evidence is None or package_evidence is None:\n                raise ValueError(\n                    "causal teacher package/admission evidence is unavailable"\n                )\n            admission_evidence = dict(admission_evidence)\n            package_evidence = dict(package_evidence)\n            if (\n                package_evidence.get("schema_version")\n                != "universal_causal_alpha_teacher_package_evidence_v1"\n            ):\n                raise ValueError("causal teacher package evidence schema mismatch")\n            package_digest = package_evidence.get("artifact_digest")\n            if not isinstance(package_digest, str) or len(package_digest) != 64:\n                raise ValueError("causal teacher package evidence digest is invalid")\n''',
)
replace_once(
    pretraining,
    '''        object.__setattr__(\n            self, "causal_teacher_admission_evidence", admission_evidence\n        )\n        object.__setattr__(self, "causal_teacher_episode_hours", episode_hours)\n''',
    '''        object.__setattr__(\n            self, "causal_teacher_admission_evidence", admission_evidence\n        )\n        object.__setattr__(\n            self, "causal_teacher_package_evidence", package_evidence\n        )\n        object.__setattr__(self, "causal_teacher_episode_hours", episode_hours)\n''',
)
replace_once(
    pretraining,
    '''            selection = bundle.causal_teacher_selection_evidence\n            admission = bundle.causal_teacher_admission_evidence\n            if selection is None or admission is None:\n                raise RuntimeError(\n                    "Universal causal teacher selection/admission evidence is unavailable"\n                )\n''',
    '''            selection = bundle.causal_teacher_selection_evidence\n            admission = bundle.causal_teacher_admission_evidence\n            package = bundle.causal_teacher_package_evidence\n            if selection is None or admission is None or package is None:\n                raise RuntimeError(\n                    "Universal causal teacher package/selection/admission evidence is unavailable"\n                )\n''',
)
replace_once(
    pretraining,
    '''            atomic_write_bytes(\n                output_root / "causal-teacher-admission.json",\n                canonical_json_bytes(admission) + b"\\n",\n            )\n''',
    '''            atomic_write_bytes(\n                output_root / "causal-teacher-admission.json",\n                canonical_json_bytes(admission) + b"\\n",\n            )\n            atomic_write_bytes(\n                output_root / "causal-teacher-package.json",\n                canonical_json_bytes(package) + b"\\n",\n            )\n''',
)

# Align test double used by runtime bundle tests.
runtime_test = "tests/workflows/test_universal_teacher_bundle_runtime.py"
replace_once(
    runtime_test,
    '''        causal_teacher_admission_evidence: dict[str, object] | None = None\n        causal_teacher_episode_hours: float | None = None\n''',
    '''        causal_teacher_admission_evidence: dict[str, object] | None = None\n        causal_teacher_package_evidence: dict[str, object] | None = None\n        causal_teacher_episode_hours: float | None = None\n''',
)

# Document the reviewed feature-level availability semantics.
for doc_path in (
    "docs/implementation-plans/specs/2026-08-13-universal-causal-alpha-teacher-design.md",
    "docs/implementation-plans/plans/2026-08-13-universal-causal-alpha-teacher.md",
):
    path = Path(doc_path)
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "- every selected feature is finite and causally available;",
        "- selected feature values are finite and their causal availability mask is explicit;",
    )
    text = text.replace(
        "Prefix scaler fits only rows whose label realization ends strictly before `knowledge_cutoff`; nonfinite or unavailable rows are excluded, constant columns scale to zero and are recorded.",
        "Prefix scaler fits only rows whose label realization ends strictly before `knowledge_cutoff`; nonfinite rows are excluded, while feature-level unavailable entries use fitted-mean/standardized-zero semantics and constant or unavailable columns are recorded.",
    )
    path.write_text(text, encoding="utf-8")
