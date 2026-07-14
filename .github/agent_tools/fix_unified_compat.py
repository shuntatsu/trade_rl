from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one match in {path}, found {count}: {old[:120]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


# Session-aware market build configuration while retaining the verified FeatureSpec.
replace_once(
    "trade_rl/data/contracts.py",
    '''    base_timeframe: str
    features: tuple[FeatureSpec, ...]
    schema_version: str = "market_build_v1"
''',
    '''    base_timeframe: str
    features: tuple[FeatureSpec, ...]
    calendar_kind: str = "continuous_24_7"
    session_periods_per_year: int | None = None
    schema_version: str = "market_build_v2"
''',
)
replace_once(
    "trade_rl/data/contracts.py",
    '''        require_non_empty(self.base_timeframe, field="base_timeframe")
        if self.base_timeframe not in _TIMEFRAME_HOURS:
''',
    '''        require_non_empty(self.base_timeframe, field="base_timeframe")
        raw_calendar = getattr(self.calendar_kind, "value", self.calendar_kind)
        if not isinstance(raw_calendar, str) or raw_calendar not in {
            "continuous_24_7",
            "session_calendar",
        }:
            raise ValueError("calendar_kind is not supported")
        object.__setattr__(self, "calendar_kind", raw_calendar)
        if raw_calendar == "session_calendar":
            if (
                isinstance(self.session_periods_per_year, bool)
                or not isinstance(self.session_periods_per_year, int)
                or self.session_periods_per_year <= 0
            ):
                raise ValueError(
                    "session_periods_per_year must be a positive integer for session data"
                )
        elif self.session_periods_per_year is not None:
            raise ValueError(
                "session_periods_per_year is valid only for session calendar data"
            )
        if self.base_timeframe not in _TIMEFRAME_HOURS:
''',
)
replace_once(
    "trade_rl/data/contracts.py",
    '''            "bar_hours": self.bar_hours,
            "features": tuple(spec.canonical_payload() for spec in self.features),
''',
    '''            "bar_hours": self.bar_hours,
            "calendar_kind": self.calendar_kind,
            "session_periods_per_year": self.session_periods_per_year,
            "features": tuple(spec.canonical_payload() for spec in self.features),
''',
)

# Preserve irregular exchange-session clocks without synthesizing weekend bars.
replace_once(
    "trade_rl/data/builder.py",
    '''def _align_series(
    raw: RawMarketSeries,
    timestamps: np.ndarray,
    *,
    step_ns: int,
) -> dict[str, np.ndarray]:
    n_bars = len(timestamps)
    first = int(timestamps[0].astype("datetime64[ns]").astype(np.int64))
    raw_ns = raw.timestamps.astype("datetime64[ns]").astype(np.int64)
    offsets = raw_ns - first
    if np.any(offsets < 0) or np.any(offsets % step_ns != 0):
        raise ValueError("raw timestamps do not align to the configured base timeframe")
    indices = offsets // step_ns
    if np.any(indices >= n_bars):
        raise ValueError("raw timestamps fall outside the resolved market clock")
''',
    '''def _session_clock(series: tuple[RawMarketSeries, ...]) -> np.ndarray:
    values = np.concatenate(
        tuple(item.timestamps.astype("datetime64[ns]") for item in series)
    )
    timestamps = np.unique(values)
    if timestamps.size < 3:
        raise ValueError("market source does not contain a usable session range")
    return timestamps


def _align_series(
    raw: RawMarketSeries,
    timestamps: np.ndarray,
    *,
    step_ns: int | None,
) -> dict[str, np.ndarray]:
    n_bars = len(timestamps)
    raw_ns = raw.timestamps.astype("datetime64[ns]").astype(np.int64)
    clock_ns = timestamps.astype("datetime64[ns]").astype(np.int64)
    if step_ns is None:
        indices = np.searchsorted(clock_ns, raw_ns)
        if np.any(indices >= n_bars) or np.any(clock_ns[indices] != raw_ns):
            raise ValueError("raw timestamps fall outside the resolved session clock")
    else:
        first = int(clock_ns[0])
        offsets = raw_ns - first
        if np.any(offsets < 0) or np.any(offsets % step_ns != 0):
            raise ValueError("raw timestamps do not align to the configured base timeframe")
        indices = offsets // step_ns
        if np.any(indices >= n_bars):
            raise ValueError("raw timestamps fall outside the resolved market clock")
''',
)
replace_once(
    "trade_rl/data/builder.py",
    '''        step_ns = int(round(self.config.bar_hours * _NS_PER_HOUR))
        timestamps = _regular_clock(raw_series, step_ns=step_ns)
''',
    '''        step_ns = int(round(self.config.bar_hours * _NS_PER_HOUR))
        if self.config.calendar_kind == "session_calendar":
            timestamps = _session_clock(raw_series)
            alignment_step: int | None = None
        else:
            timestamps = _regular_clock(raw_series, step_ns=step_ns)
            alignment_step = step_ns
''',
)
replace_once(
    "trade_rl/data/builder.py",
    "            aligned = _align_series(raw, timestamps, step_ns=step_ns)\n",
    "            aligned = _align_series(raw, timestamps, step_ns=alignment_step)\n",
)
replace_once(
    "trade_rl/data/builder.py",
    '''        periods_per_year = int(round(365.0 * 24.0 / self.config.bar_hours))
        return MarketDataset(
''',
    '''        periods_per_year = (
            int(round(365.0 * 24.0 / self.config.bar_hours))
            if self.config.calendar_kind == "continuous_24_7"
            else int(self.config.session_periods_per_year or 0)
        )
        return MarketDataset(
''',
)
replace_once(
    "trade_rl/data/builder.py",
    '''            normalization_digest=normalization_digest,
            periods_per_year=periods_per_year,
        ).with_content_identity(metadata)
''',
    '''            normalization_digest=normalization_digest,
            periods_per_year=periods_per_year,
            calendar_kind=self.config.calendar_kind,
            nominal_bar_hours=self.config.bar_hours,
        ).with_content_identity(metadata)
''',
)

# Keep framework integrations outside the RL core.
replace_once(
    "trade_rl/rl/training.py",
    '''\n\ndef __getattr__(name: str) -> Any:
    if name in {"StableBaselines3Backend", "StableBaselines3PPOBackend"}:
        from trade_rl.integrations import sb3_training

        return getattr(sb3_training, name)
    raise AttributeError(name)
''',
    "",
)
replace_once(
    "trade_rl/rl/training.py",
    '''    "ResidualTrainingConfig",
    "StableBaselines3Backend",
    "StableBaselines3PPOBackend",
    "gamma_from_half_life",
''',
    '''    "ResidualTrainingConfig",
    "gamma_from_half_life",
''',
)
replace_once(
    "trade_rl/workflows/market_walk_forward.py",
    "from trade_rl.rl.training import StableBaselines3Backend, train_residual_ensemble\n",
    "from trade_rl.integrations.sb3_training import StableBaselines3Backend\nfrom trade_rl.rl.training import train_residual_ensemble\n",
)
replace_once(
    "trade_rl/workflows/market_walk_forward.py",
    '''            training_config_digest=config_digest,
            artifact_paths=_artifact_paths(stage),
''',
    '''            training_config_digest=config_digest,
            provenance_digest=config_digest,
            artifact_paths=_artifact_paths(stage),
''',
)

# Type-safe artifact constructors.
replace_once(
    "trade_rl/rl/replay.py",
    '''        replay_digest=payload["replay_digest"],
        size_bytes=payload["size_bytes"],
''',
    '''        replay_digest=str(payload["replay_digest"]),
        size_bytes=int(payload["size_bytes"]),
''',
)
replace_once(
    "trade_rl/release/attestation.py",
    "        return cls(attestation_digest=content_digest(payload), **payload)\n",
    '''        return cls(
            attestation_digest=content_digest(payload),
            bundle_digest=bundle_digest,
            dataset_id=dataset_id,
            selection_evaluation_digest=selection_evaluation_digest,
            gate_evaluation_digest=gate_evaluation_digest,
            gate_evidence_digest=gate_evidence_digest,
            selected_policy_digest=selected_policy_digest,
            git_commit=git_commit,
            dependency_digest=dependency_digest,
            approver=approver,
            approved_at=approved_at,
        )
''',
)
replace_once(
    "trade_rl/artifacts/run_manifest.py",
    "        return cls(digest=content_digest(payload), **payload)\n",
    '''        return cls(
            digest=content_digest(payload),
            run_id=run_id,
            dataset_id=dataset_id,
            environment_digest=environment_digest,
            ensemble_digest=ensemble_digest,
            training_config_digest=training_config_digest,
            provenance_digest=provenance_digest,
            files=files,
            created_at=created_at,
        )
''',
)
replace_once(
    "trade_rl/artifacts/run_manifest.py",
    "        return cls(digest=content_digest(payload), **payload)\n",
    '''        return cls(
            digest=content_digest(payload),
            run_id=run_id,
            dataset_id=dataset_id,
            environment_digest=environment_digest,
            evaluation_digest=evaluation_digest,
            workflow_config_digest=workflow_config_digest,
            policy_set_digest=policy_set_digest,
            provenance_digest=provenance_digest,
            fold_count=fold_count,
            files=files,
            created_at=created_at,
        )
''',
)
replace_once(
    "trade_rl/artifacts/provenance.py",
    "    return RuntimeProvenance(digest=content_digest(payload), **payload)\n",
    '''    return RuntimeProvenance(
        digest=content_digest(payload),
        git_commit=resolved_commit,
        git_dirty=resolved_dirty,
        lockfile_digest=lock_digest,
        python_version=python_version or sys.version.split()[0],
        platform_name=platform_name or platform.platform(),
        hardware_name=hardware_name
        or platform.processor()
        or platform.machine()
        or "unknown",
        package_versions=tuple(sorted(versions.items())),
        deterministic_seed_config_digest=content_digest(deterministic_seed_config),
    )
''',
)

# Fold diagnostics use the verified ExecutionDiagnostics contract.
replace_once(
    "trade_rl/evaluation/fold_metrics.py",
    "result.evidence",
    "result.diagnostics",
)
# The replacement occurs six times; handle the remaining occurrences.
path = ROOT / "trade_rl/evaluation/fold_metrics.py"
path.write_text(path.read_text(encoding="utf-8").replace("result.evidence", "result.diagnostics"), encoding="utf-8")
replace_once(
    "tests/evaluation/test_fold_metrics.py",
    "from trade_rl.evaluation.walk_forward.stitching import ExecutionEvidence, FoldOOSResult\n",
    "from trade_rl.evaluation.evidence import ExecutionDiagnostics\nfrom trade_rl.evaluation.walk_forward.stitching import FoldOOSResult\n",
)
replace_once(
    "tests/evaluation/test_fold_metrics.py",
    '''        evidence=ExecutionEvidence(
            total_cost=cost, turnover_total=2 * cost, n_trades=index + 1
        ),
''',
    '''        diagnostics=ExecutionDiagnostics(
            total_cost=cost, turnover_total=2 * cost, n_trades=index + 1
        ),
''',
)

# Immutable optional multipliers are always materialized by BookState validation.
replace_once(
    "trade_rl/simulation/accounting.py",
    "            contract_multipliers=self.contract_multipliers.copy(),\n",
    "            contract_multipliers=np.asarray(self.contract_multipliers).copy(),\n",
)

# Expose canonical identity state to the atomic publication compatibility wrapper.
replace_once(
    "trade_rl/data/market.py",
    '''        object.__setattr__(self, "identity_payload_json", identity_payload_json)

    def identity_contract_payload(self) -> dict[str, object]:
''',
    '''        object.__setattr__(self, "identity_payload_json", identity_payload_json)

    @property
    def identity_verified(self) -> bool:
        return self.identity_payload_json is not None

    def identity_contract_payload(self) -> dict[str, object]:
''',
)

# Serving compatibility wrapper delegates to the bundle's verified normalizer.
(ROOT / "trade_rl/serving/observations.py").write_text(
    '''"""Verified observation transformation for serving bundles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trade_rl.rl.normalization import ObservationNormalizer
from trade_rl.serving.bundle import ServingBundle

NORMALIZER_FILE = "normalizer.json"


@dataclass(frozen=True, slots=True)
class ServingObservationPipeline:
    observation_size: int
    normalizer: ObservationNormalizer | None

    @classmethod
    def load(cls, bundle: ServingBundle) -> "ServingObservationPipeline":
        manifest = bundle.manifest
        normalizer = bundle.normalizer
        if manifest.normalizer_digest is None:
            if normalizer is not None:
                raise ValueError("serving bundle contains an unbound normalizer")
            return cls(observation_size=manifest.observation_size, normalizer=None)
        if normalizer is None or normalizer.digest != manifest.normalizer_digest:
            raise ValueError("serving normalizer digest mismatch")
        if normalizer.size != manifest.observation_size:
            raise ValueError("serving normalizer observation size mismatch")
        if normalizer.observation_schema != manifest.observation_schema:
            raise ValueError("serving normalizer observation schema mismatch")
        if normalizer.dataset_id is not None and normalizer.dataset_id != manifest.dataset_id:
            raise ValueError("serving normalizer dataset identity mismatch")
        return cls(observation_size=manifest.observation_size, normalizer=normalizer)

    def transform(self, observation: np.ndarray) -> np.ndarray:
        vector = np.asarray(observation, dtype=np.float32).reshape(-1)
        if vector.shape != (self.observation_size,) or not np.isfinite(vector).all():
            raise ValueError("observation violates the active observation schema")
        if self.normalizer is None:
            return vector.copy()
        return self.normalizer.transform(vector)


__all__ = ["NORMALIZER_FILE", "ServingObservationPipeline"]
''',
    encoding="utf-8",
)

# Explicit algorithm constructors avoid untyped **dict expansion.
path = ROOT / "trade_rl/rl/algorithm_configs.py"
text = path.read_text(encoding="utf-8")
start = text.index("def build_algorithm_config(")
end = text.index("\n\n__all__ =", start)
function = '''def build_algorithm_config(
    source: ResidualTrainingConfig,
    *,
    algorithm: str | None = None,
) -> AlgorithmConfig:
    resolved = source.algorithm if algorithm is None else algorithm.lower()
    if resolved == "ppo":
        return PPOConfig(
            timesteps=source.timesteps,
            gamma=source.gamma,
            learning_rate=source.learning_rate,
            batch_size=source.batch_size,
            policy=source.policy,
            device=source.device,
            policy_net_arch=source.policy_net_arch,
            n_steps=source.n_steps,
            n_epochs=source.n_epochs,
            gae_lambda=source.gae_lambda,
            clip_range=source.clip_range,
            normalize_advantage=source.normalize_advantage,
            ent_coef=source.ent_coef,
            vf_coef=source.vf_coef,
            max_grad_norm=source.max_grad_norm,
            log_std_init=source.log_std_init,
            target_kl=source.target_kl,
            use_sde=source.use_sde,
            sde_sample_freq=source.sde_sample_freq,
        )
    common = dict(
        timesteps=source.timesteps,
        gamma=source.gamma,
        learning_rate=source.learning_rate,
        batch_size=source.batch_size,
        policy=source.policy,
        device=source.device,
        policy_net_arch=source.policy_net_arch,
        buffer_size=source.buffer_size,
        learning_starts=source.learning_starts,
        train_freq=source.train_freq,
        gradient_steps=source.gradient_steps,
    )
    if resolved == "sac":
        return SACConfig(
            timesteps=source.timesteps,
            gamma=source.gamma,
            learning_rate=source.learning_rate,
            batch_size=source.batch_size,
            policy=source.policy,
            device=source.device,
            policy_net_arch=source.policy_net_arch,
            buffer_size=source.buffer_size,
            learning_starts=source.learning_starts,
            train_freq=source.train_freq,
            gradient_steps=source.gradient_steps,
            use_sde=source.use_sde,
            sde_sample_freq=source.sde_sample_freq,
        )
    if resolved == "td3":
        return TD3Config(**common)  # type: ignore[arg-type]
    if resolved == "tqc":
        return TQCConfig(
            timesteps=source.timesteps,
            gamma=source.gamma,
            learning_rate=source.learning_rate,
            batch_size=source.batch_size,
            policy=source.policy,
            device=source.device,
            policy_net_arch=source.policy_net_arch,
            buffer_size=source.buffer_size,
            learning_starts=source.learning_starts,
            train_freq=source.train_freq,
            gradient_steps=source.gradient_steps,
            use_sde=source.use_sde,
            sde_sample_freq=source.sde_sample_freq,
        )
    raise ValueError(f"unsupported training algorithm: {resolved}")
'''
path.write_text(text[:start] + function + text[end:], encoding="utf-8")

# Exogenous observation matrix used for leakage-safe normalizer fitting tests.
insert = '''\n\ndef observation_market_matrix(
    dataset: MarketDataset,
    *,
    start: int,
    stop: int,
    action_size: int = 2,
    n_factors: int = 0,
    finite_horizon: bool = False,
) -> np.ndarray:
    if not 0 <= start < stop <= dataset.n_bars:
        raise ValueError("observation matrix range is outside the dataset")
    layout = observation_layout(
        dataset,
        action_size=action_size,
        n_factors=n_factors,
        finite_horizon=finite_horizon,
    )
    rows: list[np.ndarray] = []
    n_features = dataset.n_features
    n_global = len(dataset.global_feature_names)
    for index in range(start, stop):
        per_symbol = np.zeros(
            (dataset.n_symbols, layout.per_symbol_width), dtype=np.float64
        )
        per_symbol[:, :n_features] = dataset.features[index]
        per_symbol[:, n_features : 2 * n_features] = dataset.feature_available[
            index
        ]
        per_symbol[:, 2 * n_features : 3 * n_features] = _feature_staleness(
            dataset, index
        )
        per_symbol[:, 3 * n_features : 4 * n_features] = dataset.resolved_array(
            "feature_missing_reason"
        )[index]
        offset = 4 * n_features
        per_symbol[:, offset] = dataset.resolved_array("asset_active")[index]
        per_symbol[:, offset + 1] = dataset.observable_tradable(index)
        per_symbol[:, offset + n_factors + 15] = dataset.resolved_array(
            "borrow_available"
        )[index]
        per_symbol[:, offset + n_factors + 16] = dataset.resolved_array(
            "borrow_rate"
        )[index]
        per_symbol[:, offset + n_factors + 17] = (
            dataset.resolved_array("mark_price")[index]
            / dataset.resolved_array("index_price")[index]
            - 1.0
        )
        global_values = np.zeros(layout.global_width, dtype=np.float64)
        global_values[:n_global] = dataset.global_features[index]
        global_values[n_global : 2 * n_global] = dataset.resolved_array(
            "global_feature_available"
        )[index]
        global_values[2 * n_global : 3 * n_global] = dataset.resolved_array(
            "global_feature_staleness_hours"
        )[index]
        global_values[3 * n_global : 4 * n_global] = dataset.resolved_array(
            "global_feature_missing_reason"
        )[index]
        if finite_horizon:
            global_values[-1] = (stop - index) / (stop - start)
        rows.append(
            np.concatenate((per_symbol.reshape(-1), global_values)).astype(np.float32)
        )
    return np.stack(rows, axis=0)
'''
replace_once(
    "trade_rl/rl/observations.py",
    "\n\ndef _drawdown(book: BookState) -> float:\n",
    insert + "\n\ndef _drawdown(book: BookState) -> float:\n",
)
