from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise SystemExit(f"patch target drifted in {path}: {old[:80]!r}")
    target.write_text(text.replace(old, new), encoding="utf-8")


learning = "trade_rl/learning/causal_alpha_teacher.py"
replace_once(
    learning,
    '''    def transform(self, features: object) -> np.ndarray:\n        values = np.asarray(features, dtype=np.float64)\n        if values.ndim != 2 or values.shape[1] != len(self.feature_names):\n            raise ValueError("prediction features do not match causal alpha schema")\n        if not np.isfinite(values).all():\n            raise ValueError("prediction features must be finite")\n        scaled = (values - self.location) / self.scale\n        scaled[:, self.constant_mask] = 0.0\n        return scaled\n\n    def predict(self, features: object) -> np.ndarray:\n        scaled = self.transform(features)\n        prediction = self.intercept + scaled @ self.coefficients\n''',
    '''    def transform(\n        self,\n        features: object,\n        *,\n        feature_available: object | None = None,\n    ) -> np.ndarray:\n        values = np.asarray(features, dtype=np.float64)\n        if values.ndim != 2 or values.shape[1] != len(self.feature_names):\n            raise ValueError("prediction features do not match causal alpha schema")\n        if not np.isfinite(values).all():\n            raise ValueError("prediction features must be finite")\n        availability: np.ndarray | None = None\n        if feature_available is not None:\n            availability = np.asarray(feature_available, dtype=np.bool_)\n            if availability.shape != values.shape:\n                raise ValueError(\n                    "prediction feature availability must match prediction features"\n                )\n        scaled = (values - self.location) / self.scale\n        scaled[:, self.constant_mask] = 0.0\n        if availability is not None:\n            scaled = np.where(availability, scaled, 0.0)\n        return scaled\n\n    def predict(\n        self,\n        features: object,\n        *,\n        feature_available: object | None = None,\n    ) -> np.ndarray:\n        scaled = self.transform(features, feature_available=feature_available)\n        prediction = self.intercept + scaled @ self.coefficients\n''',
)
replace_once(
    learning,
    '''class CausalAlphaTargetPath:\n    initial_weight: float\n    targets: np.ndarray\n    submitted_change_count: int\n    suppressed_change_count: int\n    sign_flip_count: int\n    digest: str = ""\n''',
    '''class CausalAlphaTargetPath:\n    initial_weight: float\n    targets: np.ndarray\n    submitted_change_count: int\n    suppressed_change_count: int\n    sign_flip_count: int\n    actionable_mask: np.ndarray | None = None\n    digest: str = ""\n''',
)
replace_once(
    learning,
    '''        targets.setflags(write=False)\n        for field in (\n            "submitted_change_count",\n''',
    '''        targets.setflags(write=False)\n        if self.actionable_mask is None:\n            actionable = np.ones(targets.shape, dtype=np.bool_)\n        else:\n            actionable = np.asarray(self.actionable_mask, dtype=np.bool_).reshape(-1).copy()\n            if actionable.shape != targets.shape:\n                raise ValueError("actionable_mask must align with causal alpha targets")\n        actionable.setflags(write=False)\n        for field in (\n            "submitted_change_count",\n''',
)
replace_once(
    learning,
    '''        object.__setattr__(self, "targets", targets)\n        expected = content_digest(\n            {\n                "initial_weight": float(self.initial_weight),\n                "targets": targets.tolist(),\n''',
    '''        object.__setattr__(self, "targets", targets)\n        object.__setattr__(self, "actionable_mask", actionable)\n        expected = content_digest(\n            {\n                "actionable_mask": actionable.astype(bool).tolist(),\n                "initial_weight": float(self.initial_weight),\n                "targets": targets.tolist(),\n''',
)
replace_once(
    learning,
    '''def causal_alpha_target_path(\n    scores: object,\n    *,\n    config: CausalAlphaControllerConfig,\n    initial_weight: float,\n) -> CausalAlphaTargetPath:\n    values = np.asarray(scores, dtype=np.float64).reshape(-1)\n    if not np.isfinite(values).all():\n        raise ValueError("causal alpha scores must be finite")\n    if not math.isfinite(initial_weight):\n        raise ValueError("initial_weight must be finite")\n    previous = float(initial_weight)\n''',
    '''def causal_alpha_target_path(\n    scores: object,\n    *,\n    config: CausalAlphaControllerConfig,\n    initial_weight: float,\n    actionable_mask: object | None = None,\n) -> CausalAlphaTargetPath:\n    values = np.asarray(scores, dtype=np.float64).reshape(-1)\n    if not np.isfinite(values).all():\n        raise ValueError("causal alpha scores must be finite")\n    if not math.isfinite(initial_weight):\n        raise ValueError("initial_weight must be finite")\n    if actionable_mask is None:\n        actionable = np.ones(values.shape, dtype=np.bool_)\n    else:\n        actionable = np.asarray(actionable_mask, dtype=np.bool_).reshape(-1)\n        if actionable.shape != values.shape:\n            raise ValueError("actionable_mask must align with causal alpha scores")\n    previous = float(initial_weight)\n''',
)
replace_once(
    learning,
    '''    for index, score in enumerate(values):\n        desired = _desired_target(float(score), previous, config)\n''',
    '''    for index, score in enumerate(values):\n        if not bool(actionable[index]):\n            targets[index] = previous\n            continue\n        desired = _desired_target(float(score), previous, config)\n''',
)
replace_once(
    learning,
    '''        sign_flip_count=sign_flips,\n    )\n''',
    '''        sign_flip_count=sign_flips,\n        actionable_mask=actionable,\n    )\n''',
)

contracts = "trade_rl/workflows/universal_causal_alpha_contracts.py"
replace_once(
    contracts,
    '''    def features_for_decisions(self, decision_indices: object) -> np.ndarray:\n        requested = np.asarray(decision_indices, dtype=np.int64).reshape(-1)\n        positions = np.searchsorted(self.decision_indices, requested)\n        if np.any(positions >= self.decision_indices.size) or not np.array_equal(\n            self.decision_indices[positions], requested\n        ):\n            raise ValueError(\n                "causal alpha prediction decisions are absent from samples"\n            )\n        if not np.all(self.feature_available[positions]):\n            raise ValueError("causal alpha prediction features are unavailable")\n        return np.asarray(self.features[positions], dtype=np.float64)\n''',
    '''    def prediction_inputs_for_decisions(\n        self, decision_indices: object\n    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:\n        requested = np.asarray(decision_indices, dtype=np.int64).reshape(-1)\n        if requested.size == 0 or np.any(requested < 0) or np.any(np.diff(requested) <= 0):\n            raise ValueError(\n                "causal alpha prediction decisions must be strictly increasing"\n            )\n        positions = np.searchsorted(self.decision_indices, requested)\n        present = positions < self.decision_indices.size\n        matched = np.zeros(requested.shape, dtype=np.bool_)\n        if np.any(present):\n            matched[present] = (\n                self.decision_indices[positions[present]] == requested[present]\n            )\n        width = len(self.feature_names)\n        features = np.zeros((requested.size, width), dtype=np.float64)\n        availability = np.zeros((requested.size, width), dtype=np.bool_)\n        if np.any(matched):\n            source = positions[matched]\n            features[matched] = self.features[source]\n            availability[matched] = self.feature_available[source]\n        return features, availability, matched\n\n    def features_for_decisions(self, decision_indices: object) -> np.ndarray:\n        features, availability, present = self.prediction_inputs_for_decisions(\n            decision_indices\n        )\n        if not np.all(present):\n            raise ValueError(\n                "causal alpha prediction decisions are absent from samples"\n            )\n        if not np.all(availability):\n            raise ValueError("causal alpha prediction features are unavailable")\n        return features\n''',
)

selection = "trade_rl/workflows/universal_causal_alpha_selection.py"
replace_once(
    selection,
    '''    prediction_features = block.features_for_decisions(decisions)\n    prediction_24h = fitted.model_24h.predict(prediction_features)\n    prediction_72h = fitted.model_72h.predict(prediction_features)\n''',
    '''    prediction_features, prediction_available, actionable = (\n        block.prediction_inputs_for_decisions(decisions)\n    )\n    prediction_24h = fitted.model_24h.predict(\n        prediction_features, feature_available=prediction_available\n    )\n    prediction_72h = fitted.model_72h.predict(\n        prediction_features, feature_available=prediction_available\n    )\n''',
)
replace_once(
    selection,
    '''        config=candidate.controller,\n        initial_weight=float(contract.initial_weights[0]),\n    )\n''',
    '''        config=candidate.controller,\n        initial_weight=float(contract.initial_weights[0]),\n        actionable_mask=actionable,\n    )\n''',
)

fitting = "trade_rl/workflows/universal_causal_alpha_fitting.py"
replace_once(
    fitting,
    '''        prediction_features = block.features_for_decisions(decisions)\n        prediction_24h = fitted.model_24h.predict(prediction_features)\n        prediction_72h = fitted.model_72h.predict(prediction_features)\n''',
    '''        prediction_features, prediction_available, actionable = (\n            block.prediction_inputs_for_decisions(decisions)\n        )\n        prediction_24h = fitted.model_24h.predict(\n            prediction_features, feature_available=prediction_available\n        )\n        prediction_72h = fitted.model_72h.predict(\n            prediction_features, feature_available=prediction_available\n        )\n''',
)
replace_once(
    fitting,
    '''            config=controller_config,\n            initial_weight=initial_weight,\n        )\n''',
    '''            config=controller_config,\n            initial_weight=initial_weight,\n            actionable_mask=actionable,\n        )\n''',
)
replace_once(
    fitting,
    '''                ("prediction_24h", prediction_24h),\n                ("prediction_72h", prediction_72h),\n                ("scores", scores),\n                ("targets", target_matrix),\n''',
    '''                ("prediction_24h", prediction_24h),\n                ("prediction_72h", prediction_72h),\n                ("prediction_feature_available", prediction_available),\n                ("actionable_mask", actionable),\n                ("scores", scores),\n                ("targets", target_matrix),\n''',
)
