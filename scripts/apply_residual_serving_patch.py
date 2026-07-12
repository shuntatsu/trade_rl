from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one match, found {count}")
    path.write_text(text.replace(old, new), encoding="utf-8")


root = Path(__file__).resolve().parents[1]

replace_once(
    root / "mars_lite/env/observation.py",
    '        if self.version != 1:\n            raise ValueError(f"unsupported observation schema version: {self.version}")\n',
    '        if self.version not in {1, 2}:\n            raise ValueError(f"unsupported observation schema version: {self.version}")\n',
)

bundle = root / "mars_lite/serving/bundle.py"
replace_once(
    bundle,
    '''def _validate_model_layout(metadata: dict[str, Any], files: Mapping[str, str]) -> None:
    model_kind = metadata.get("model_kind")
    has_single = "model.zip" in files
    ensemble_members = sorted(
        name for name in files if _ENSEMBLE_MEMBER_RE.fullmatch(name)
    )
    has_other_ensemble_files = any(
        name.startswith("ensemble/") and name not in ensemble_members for name in files
    )
    if model_kind == "single":
        if not has_single or ensemble_members or has_other_ensemble_files:
            raise ValueError("single model_kind requires only model.zip")
    elif model_kind == "ensemble":
        if has_single or not ensemble_members or has_other_ensemble_files:
            raise ValueError(
                "ensemble model_kind requires one or more ensemble/seed_<n>.zip files"
            )
    else:
        raise ValueError("metadata.model_kind must be 'single' or 'ensemble'")
''',
    '''def _validate_model_layout(metadata: dict[str, Any], files: Mapping[str, str]) -> None:
    model_kind = metadata.get("model_kind")
    action_schema = metadata.get("action_schema", "direct_weights_v1")
    policy_mode = metadata.get("policy_mode")
    has_single = "model.zip" in files
    has_alpha = "residual_alpha.json" in files
    ensemble_members = sorted(
        name for name in files if _ENSEMBLE_MEMBER_RE.fullmatch(name)
    )
    has_other_ensemble_files = any(
        name.startswith("ensemble/") and name not in ensemble_members for name in files
    )
    if model_kind == "single":
        if not has_single or ensemble_members or has_other_ensemble_files:
            raise ValueError("single model_kind requires only model.zip")
    elif model_kind == "ensemble":
        if has_single or not ensemble_members or has_other_ensemble_files:
            raise ValueError(
                "ensemble model_kind requires one or more ensemble/seed_<n>.zip files"
            )
    elif model_kind == "baseline_only":
        if has_single or ensemble_members or has_other_ensemble_files:
            raise ValueError("baseline_only model_kind must not contain policy models")
    else:
        raise ValueError(
            "metadata.model_kind must be 'single', 'ensemble', or 'baseline_only'"
        )

    if action_schema == "baseline_residual_v1":
        if not has_alpha:
            raise ValueError("baseline_residual_v1 requires residual_alpha.json")
        if policy_mode == "baseline_only" and model_kind != "baseline_only":
            raise ValueError("baseline_only policy_mode requires baseline_only model_kind")
        if policy_mode == "ppo_residual_ensemble" and model_kind != "ensemble":
            raise ValueError("ppo_residual_ensemble requires ensemble model_kind")
        if policy_mode not in {"baseline_only", "ppo_residual_ensemble"}:
            raise ValueError("unsupported baseline residual policy_mode")
    elif model_kind == "baseline_only":
        raise ValueError("baseline_only requires baseline_residual_v1 action schema")
''',
)
replace_once(
    bundle,
    '''    if metadata.get("observation_schema_version") != 1:
        raise ValueError("unsupported observation_schema_version")
''',
    '''    action_schema = metadata.get("action_schema", "direct_weights_v1")
    if action_schema not in {"direct_weights_v1", "baseline_residual_v1"}:
        raise ValueError("unsupported action_schema")
    expected_schema_version = 2 if action_schema == "baseline_residual_v1" else 1
    if metadata.get("observation_schema_version") != expected_schema_version:
        raise ValueError("unsupported observation_schema_version for action schema")
    if action_schema == "baseline_residual_v1":
        if not isinstance(metadata.get("trend_family"), dict):
            raise ValueError("baseline residual metadata requires trend_family")
        if not isinstance(metadata.get("composer"), dict):
            raise ValueError("baseline residual metadata requires composer")
        if metadata.get("residual_alpha_file") != "residual_alpha.json":
            raise ValueError("baseline residual metadata requires residual_alpha_file")
''',
)
replace_once(
    bundle,
    '''    expected_observation_dim = (
        len(symbols) * (len(feature_names) + 1)
        + len(global_feature_names)
        + 3
        + (4 if obs_risk_state else 0)
    )
''',
    '''    per_symbol_extra = 5 if action_schema == "baseline_residual_v1" else 1
    expected_observation_dim = (
        len(symbols) * (len(feature_names) + per_symbol_extra)
        + len(global_feature_names)
        + 3
        + (4 if obs_risk_state else 0)
    )
''',
)

runtime = root / "mars_lite/serving/runtime.py"
replace_once(
    runtime,
    '''RiskFn = Callable[[np.ndarray, InferenceState, Sequence[str]], Mapping[str, Any]]
ComponentFactory = Callable[[ServingBundle], "RuntimeComponents"]
''',
    '''RiskFn = Callable[[np.ndarray, InferenceState, Sequence[str]], Mapping[str, Any]]
FeatureAugmentFn = Callable[
    ["FeatureSnapshot", np.ndarray], tuple[np.ndarray, Mapping[str, Any]]
]
ContextDecisionFn = Callable[
    [
        np.ndarray,
        InferenceState,
        np.ndarray | None,
        np.ndarray | None,
        Mapping[str, Any],
    ],
    tuple[np.ndarray, Mapping[str, Any]],
]
ComponentFactory = Callable[[ServingBundle], "RuntimeComponents"]
''',
)
replace_once(
    runtime,
    '''    close_history: np.ndarray
    data_age_hours: float

    def validate(self) -> None:
''',
    '''    close_history: np.ndarray
    data_age_hours: float
    timestamps: np.ndarray | None = None

    def validate(self) -> None:
''',
)
replace_once(
    runtime,
    '''        if not np.isfinite(close).all() or np.any(close <= 0):
            raise ValueError("close_history must be finite and positive")
        if not math.isfinite(self.data_age_hours) or self.data_age_hours < 0:
''',
    '''        if not np.isfinite(close).all() or np.any(close <= 0):
            raise ValueError("close_history must be finite and positive")
        if self.timestamps is not None:
            timestamps = np.asarray(self.timestamps).astype("datetime64[ns]")
            if timestamps.ndim != 1 or len(timestamps) != features.shape[0]:
                raise ValueError("timestamps must match feature history")
            timestamp_ns = timestamps.astype(np.int64)
            if len(timestamp_ns) > 1 and np.any(np.diff(timestamp_ns) <= 0):
                raise ValueError("timestamps must be strictly increasing")
        if not math.isfinite(self.data_age_hours) or self.data_age_hours < 0:
''',
)
replace_once(
    runtime,
    '''    serving_progress: float = 0.0
    vol_lookback: int = 0
    htf_feature_name: str | None = None
''',
    '''    serving_progress: float = 0.0
    vol_lookback: int = 0
    htf_feature_name: str | None = None
    augment_features: FeatureAugmentFn | None = None
    decide_with_context: ContextDecisionFn | None = None
''',
)
replace_once(
    runtime,
    '''            latest, recent_returns, htf_trend = self._prepare_features(
                bundle, components, snapshot
            )
            symbols = tuple(bundle.metadata["symbols"])
''',
    '''            latest, recent_returns, htf_trend = self._prepare_features(
                bundle, components, snapshot
            )
            decision_context: Mapping[str, Any] = {}
            if components.augment_features is not None:
                latest, decision_context = components.augment_features(snapshot, latest)
            symbols = tuple(bundle.metadata["symbols"])
''',
)
replace_once(
    runtime,
    '''            raw_action, _ = components.model.predict(observation, deterministic=True)
            target, decision_info = components.decide(
                np.asarray(raw_action, dtype=np.float64).reshape(-1),
                request.state,
                recent_returns,
                htf_trend,
            )
''',
    '''            raw_action, _ = components.model.predict(observation, deterministic=True)
            action_array = np.asarray(raw_action, dtype=np.float64).reshape(-1)
            if components.decide_with_context is not None:
                target, decision_info = components.decide_with_context(
                    action_array,
                    request.state,
                    recent_returns,
                    htf_trend,
                    decision_context,
                )
            else:
                target, decision_info = components.decide(
                    action_array,
                    request.state,
                    recent_returns,
                    htf_trend,
                )
''',
)

provider = root / "mars_lite/serving/feature_provider.py"
replace_once(
    provider,
    '''        rank_window = int(bundle.preprocessing.get("rank_window", 250))
        post_config = dict(bundle.metadata.get("post_processor") or {})
        vol_lookback = int(post_config.get("vol_lookback", 60))
        history_bars = max(rank_window, vol_lookback + 1, 2)
''',
    '''        rank_window = int(bundle.preprocessing.get("rank_window", 250))
        post_config = dict(bundle.metadata.get("post_processor") or {})
        vol_lookback = int(post_config.get("vol_lookback", 60))
        trend_config = dict(bundle.metadata.get("trend_family") or {})
        trend_lookback = max(
            int(trend_config.get("fast_lookback", 0)),
            int(trend_config.get("base_lookback", 0)),
            int(trend_config.get("slow_lookback", 0)),
        )
        history_bars = max(rank_window, vol_lookback + 1, trend_lookback + 1, 2)
''',
)
replace_once(
    provider,
    '''            close_history=close_history_array,
            data_age_hours=endpoint.data_age_hours,
        )
''',
    '''            close_history=close_history_array,
            data_age_hours=endpoint.data_age_hours,
            timestamps=np.asarray(timestamps),
        )
''',
)
