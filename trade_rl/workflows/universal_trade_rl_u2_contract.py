"""Frozen Base PPO training contract for Universal Trade RL U2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.domain.universal_trade_rl_universe import UniversalTradeRLSymbolRole
from trade_rl.rl.training import (
    ResidualTrainingConfig,
    gamma_from_half_life,
)
from trade_rl.rl.training_modes import CudaRuntimeMode
from trade_rl.rl.universal_architecture import (
    UniversalArchitectureName,
    apply_architecture_to_training_config,
    architecture_spec,
)
from trade_rl.rl.universal_episode_router import DETERMINISTIC_INSTRUMENT_ROUTER_SCHEMA
from trade_rl.workflows.universal_trade_rl_data_provenance import (
    UniversalTradeRLFitProvenance,
    UniversalTradeRLFitPurpose,
    require_universal_trade_rl_train_only_provenance,
)
from trade_rl.workflows.universal_trade_rl_run_identity import (
    UniversalTradeRLRunIdentity,
    UniversalTradeRLRunStage,
)
from trade_rl.workflows.universal_trade_rl_u1_contract import UniversalTradeRLU1Contract
from trade_rl.workflows.universal_trade_rl_u2_time_partition import (
    UniversalTradeRLU2TimePartition,
)
from trade_rl.workflows.universal_trade_rl_universe_manifest import (
    UniversalTradeRLUniverseManifest,
)

UNIVERSAL_TRADE_RL_U2_CONTRACT_SCHEMA: Final = "universal_trade_rl_u2_contract_v1"
U2_PRODUCTION_STATUS: Final = "NO-GO"
U2_ARCHITECTURE_NAME: Final = UniversalArchitectureName.U_MEDIUM_DIRECT.value
U2_TRAINING_SEEDS: Final = (0, 1, 2)
U2_PRIMARY_CANDIDATE_SEED: Final = 0
U2_FINAL_TIMESTEPS: Final = 524_288

_CONTRACT_KEYS: Final = (
    "schema_version",
    "universe_manifest_digest",
    "u1_contract_digest",
    "u1_normalizer_digest",
    "u1_normalizer_knowledge_cutoff_ns",
    "time_partition_digest",
    "fit_end_ns",
    "rl_training_provenance_digest",
    "rl_training_knowledge_cutoff_ns",
    "training_config_digest",
    "training_config_payload",
    "architecture_name",
    "architecture_spec_digest",
    "training_seeds",
    "primary_candidate_seed",
    "router_contract_digest",
    "episode_sampling_contract_digest",
    "evaluation_checkpoint_contract_digest",
    "baseline_contract_digest",
    "selection_thresholds_digest",
    "instrument_context_enabled",
    "v4_context_enabled",
    "production_status",
    "artifact_digest",
)


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer and not boolean")
    return value


def _exact_mapping(
    value: object,
    *,
    keys: tuple[str, ...],
    field: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object with exact keys")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{field} keys must be strings")
        result[key] = item
    if set(result) != set(keys) or len(result) != len(keys):
        raise ValueError(f"{field} must use exact keys")
    return result


def _sequence(value: object, *, field: str) -> tuple[object, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be an array")
    return tuple(value)


def _normalize_payload_value(value: object) -> object:
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("training config payload keys must be strings")
            result[key] = _normalize_payload_value(item)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_normalize_payload_value(item) for item in value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ValueError("training config payload contains unsupported values")


def _normalized_training_payload(value: object) -> dict[str, object]:
    normalized = _normalize_payload_value(value)
    if not isinstance(normalized, dict):
        raise ValueError("training config payload must be an object")
    return normalized


def build_universal_trade_rl_u2_training_config() -> ResidualTrainingConfig:
    """Return the one preregistered Base PPO V1 recipe, with inactive defaults frozen."""

    gamma = gamma_from_half_life(decision_hours=0.25, half_life_hours=168.0)
    config = ResidualTrainingConfig(
        timesteps=U2_FINAL_TIMESTEPS,
        gamma=gamma,
        seeds=U2_TRAINING_SEEDS,
        learning_rate=0.00012,
        n_steps=128,
        batch_size=256,
        n_epochs=10,
        gae_lambda=0.95,
        clip_range=0.2,
        normalize_advantage=True,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy="MlpPolicy",
        device="cuda",
        cuda_runtime_mode=CudaRuntimeMode.DETERMINISTIC,
        decision_hours=0.25,
        discount_half_life_hours=168.0,
        log_std_init=-4.0,
        target_kl=0.02,
        use_sde=False,
        sde_sample_freq=-1,
        policy_net_arch=(128, 128),
        value_net_arch=(128, 128),
        observation_encoder="asset_set",
        policy_actor_head=None,
        hierarchical_gate_temperature=1.0,
        sequence_tcn_capacity="standard",
        sequence_d_model=320,
        sequence_timeframe_attention_heads=8,
        sequence_timeframe_attention_layers=2,
        sequence_timeframe_ffn_multiplier=3,
        sequence_timeframe_gate_bias=-2.0,
        sequence_asset_attention_heads=8,
        sequence_asset_attention_layers=2,
        sequence_asset_ffn_multiplier=3,
        sequence_asset_gate_bias=-2.0,
        sequence_dropout=0.05,
        sequence_compile=False,
        sequence_compile_mode="reduce-overhead",
        sequence_transfer_mode="synchronous",
        max_policy_parameters=12_000_000,
        max_rollout_buffer_bytes=805_306_368,
        asset_embedding_dim=64,
        global_embedding_dim=64,
        algorithm="ppo",
        buffer_size=100_000,
        learning_starts=10_000,
        train_freq=1,
        gradient_steps=1,
        cost_learning_rate=3e-4,
        cost_n_epochs=1,
        cost_batch_size=None,
        cost_continuous_hidden_dims=(128, 64),
        cost_event_hidden_dims=(128, 64),
        cost_max_grad_norm=0.5,
        cost_continuous_gae_lambda=0.95,
        cost_event_gae_lambda=0.95,
        cost_value_loss_coefficient=1.0,
        cost_auxiliary_event_loss_coefficient=0.0,
        cost_architecture_variant="family_separated_v1",
        checkpoint_interval_steps=32_768,
        max_checkpoints=8,
        n_envs=8,
        vector_environment_mode="in_process",
        behavior_cloning_epochs=0,
        behavior_cloning_learning_rate=1e-3,
        behavior_cloning_batch_size=256,
        behavior_cloning_validation_fraction=0.0,
        behavior_cloning_patience=3,
        behavior_cloning_minimum_improvement=0.0,
        behavior_cloning_teacher="oracle",
        behavior_cloning_seed=None,
        behavior_cloning_required_relative_improvement=0.0,
        behavior_cloning_critic_warm_start_steps=0,
        behavior_cloning_joint_warm_start_steps=0,
        behavior_cloning_critic_warm_start_learning_rate=3e-4,
        behavior_cloning_joint_warm_start_actor_lr_scale=0.1,
        behavior_cloning_gate_loss_weight=1.0,
        behavior_cloning_target_loss_weight=1.0,
        behavior_cloning_composed_loss_weight=1.0,
        behavior_cloning_gate_change_threshold=0.05,
        behavior_cloning_gate_prediction_threshold=0.5,
        behavior_cloning_max_positive_class_weight=20.0,
        behavior_cloning_min_gate_precision=0.0,
        behavior_cloning_min_gate_recall=0.0,
        behavior_cloning_max_active_target_rmse=1.0,
        behavior_cloning_min_activity_ratio=0.0,
        behavior_cloning_max_activity_ratio=1.0,
        behavior_cloning_min_causal_holdout_trades=0,
        behavior_cloning_min_causal_holdout_episodes=1,
        behavior_cloning_min_causal_holdout_net_return_lower_bound=-1.0,
        behavior_cloning_max_causal_holdout_regret=0.0,
        behavior_cloning_causal_holdout_bootstrap_resamples=2_000,
        behavior_cloning_causal_holdout_confidence_level=0.95,
        lagrangian_budgets=(),
        lagrangian_dual_learning_rates=(),
        lagrangian_ema_betas=(),
        lagrangian_initial_multipliers=(),
        lagrangian_max_multipliers=(),
        lagrangian_warmup_rollouts=(),
        lagrangian_update_interval_rollouts=(),
        lagrangian_minimum_completed_episodes=(),
        lagrangian_probe_episodes=0,
        lagrangian_probe_max_steps_per_episode=0,
        learning_rate_schedule="linear",
        learning_rate_final_ratio=0.1,
        tensorboard_enabled=True,
        tensorboard_log_interval=1,
    )
    return apply_architecture_to_training_config(
        config,
        UniversalArchitectureName.U_MEDIUM_DIRECT,
    )


def _canonical_training_payload() -> dict[str, object]:
    return _normalized_training_payload(
        build_universal_trade_rl_u2_training_config().digest_payload()
    )


def _architecture_contract_payload() -> dict[str, object]:
    spec = architecture_spec(UniversalArchitectureName.U_MEDIUM_DIRECT)
    return {
        "schema_version": "universal_trade_rl_u2_architecture_contract_v1",
        "architecture_name": spec.name.value,
        "observation_encoder": "hierarchical_sequence_v2",
        "policy": "MultiInputPolicy",
        "tcn_capacity": spec.tcn_capacity,
        "d_model": spec.d_model,
        "timeframe_attention_heads": spec.attention_heads,
        "timeframe_attention_layers": spec.attention_layers,
        "timeframe_ffn_multiplier": spec.ffn_multiplier,
        "actor_head": spec.actor_head,
        "actor_mlp": spec.actor_mlp,
        "critic_mlp": spec.critic_mlp,
        "sequence_dropout": spec.sequence_dropout,
        "action_shape": spec.action_shape,
        "architecture_sweep_allowed": False,
    }


def _router_contract_payload(
    *,
    universe_manifest_digest: str,
    time_partition_digest: str,
) -> dict[str, object]:
    return {
        "schema_version": "universal_trade_rl_u2_router_contract_v1",
        "router_schema": DETERMINISTIC_INSTRUMENT_ROUTER_SCHEMA,
        "symbol_source": "u0_manifest_train_role",
        "universe_manifest_digest": universe_manifest_digest,
        "partition_digest": time_partition_digest,
        "cycle_contract": "each_train_symbol_exactly_once_before_repeat",
        "routing_scope": "environment_local",
        "deterministic": True,
    }


def _episode_sampling_contract_payload(
    *,
    time_partition_digest: str,
    fit_end_ns: int,
) -> dict[str, object]:
    return {
        "schema_version": "universal_trade_rl_u2_episode_sampling_contract_v1",
        "time_partition_digest": time_partition_digest,
        "fit_end_ns": fit_end_ns,
        "fit_scope_only": True,
        "episode_hours": 720,
        "eligible_start_sampling": "uniform",
        "outcome_must_not_exceed_fit_end": True,
        "deterministic_resumable": True,
        "regime_oversampling": False,
        "stress_oversampling": False,
    }


def _evaluation_checkpoint_contract_payload(
    *,
    training_config_digest: str,
) -> dict[str, object]:
    return {
        "schema_version": "universal_trade_rl_u2_evaluation_checkpoint_contract_v1",
        "training_config_digest": training_config_digest,
        "policy_action_mode": "deterministic_mean",
        "selected_checkpoint_timesteps": U2_FINAL_TIMESTEPS,
        "selected_checkpoint_rule": "exact_final_fixed_budget_only",
        "intermediate_checkpoint_role": "recovery_debug_only",
        "performance_checkpoint_selection_allowed": False,
        "training_seeds": U2_TRAINING_SEEDS,
        "primary_candidate_seed": U2_PRIMARY_CANDIDATE_SEED,
        "best_seed_selection_allowed": False,
    }


def _baseline_contract_payload(*, u1_contract_digest: str) -> dict[str, object]:
    return {
        "schema_version": "universal_trade_rl_u2_baseline_contract_v1",
        "u1_contract_digest": u1_contract_digest,
        "primary_baseline": {"name": "cash", "constant_action": 0.0},
        "diagnostic_baselines": (
            {"name": "constant_long", "constant_action": 1.0},
            {"name": "constant_short", "constant_action": -1.0},
        ),
        "same_risk_execution_accounting_scope": True,
    }


def _selection_thresholds_payload() -> dict[str, object]:
    return {
        "schema_version": "universal_trade_rl_u2_selection_thresholds_v2",
        "complete_scope_coverage_required": True,
        "symbol_balanced_gross_wealth_min_exclusive": 1.0,
        "symbol_balanced_net_wealth_min_exclusive": 1.0,
        "median_symbol_net_wealth_min_inclusive": 1.0,
        "minimum_symbol_net_wealth_min_inclusive": 1.0,
        "positive_net_scope_fraction_min_inclusive": 0.50,
        "scope_net_return_cvar10_min_inclusive": -0.01,
        "turnover_p95_per_day_max_inclusive": 1.0,
        "meaningful_execution_symbol_fraction_required": 1.0,
        "hard_risk_violation_count_required": 0,
        "unexplained_execution_reject_count_required": 0,
        "positive_gross_log_growth_net_retention_min_inclusive": 0.50,
        "development_seed_gate": {
            "required_seeds": U2_TRAINING_SEEDS,
            "scopes": (
                "development_future_1",
                "development_future_2",
                "development_future_1_plus_2",
            ),
            "all_seeds_must_pass_all_scopes": True,
        },
        "cross_seed_robustness": {
            "median_seed_net_wealth_min_exclusive": 1.0,
            "worst_seed_net_wealth_min_inclusive": 1.0,
            "hard_risk_violation_count_required": 0,
            "all_seed_turnover_p95_per_day_max_inclusive": 1.0,
            "paired_excess_vs_cash_bootstrap_lower_ci_min_exclusive": 0.0,
            "bootstrap_method": "moving_block_mean_test",
            "bootstrap_confidence_level": 0.95,
            "bootstrap_resamples": 2_000,
            "bootstrap_seed": 0,
            "bootstrap_block_length_rule": "ceil_sqrt_n_capped",
        },
        "seed0_seen_and_symbol_generalization_required": True,
        "seed1_seed2_seen_and_symbol_generalization_role": "diagnostic_only",
        "admission_access_count_required": 0,
        "timeout_evidence_required": True,
        "production_status": U2_PRODUCTION_STATUS,
    }


@dataclass(frozen=True, slots=True)
class UniversalTradeRLU2Contract:
    """Content-addressed identity for one preregistered U2 Base PPO generation."""

    universe_manifest_digest: str
    u1_contract_digest: str
    u1_normalizer_digest: str
    u1_normalizer_knowledge_cutoff_ns: int
    time_partition_digest: str
    fit_end_ns: int
    rl_training_provenance_digest: str
    rl_training_knowledge_cutoff_ns: int
    training_config_digest: str
    training_config_payload: Mapping[str, object]
    architecture_name: str
    architecture_spec_digest: str
    training_seeds: tuple[int, ...]
    primary_candidate_seed: int
    router_contract_digest: str
    episode_sampling_contract_digest: str
    evaluation_checkpoint_contract_digest: str
    baseline_contract_digest: str
    selection_thresholds_digest: str
    instrument_context_enabled: bool
    v4_context_enabled: bool
    production_status: str = U2_PRODUCTION_STATUS
    schema_version: str = UNIVERSAL_TRADE_RL_U2_CONTRACT_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != UNIVERSAL_TRADE_RL_U2_CONTRACT_SCHEMA:
            raise ValueError("unsupported Universal Trade RL U2 contract schema")
        for field_name, value in (
            ("universe_manifest_digest", self.universe_manifest_digest),
            ("u1_contract_digest", self.u1_contract_digest),
            ("u1_normalizer_digest", self.u1_normalizer_digest),
            ("time_partition_digest", self.time_partition_digest),
            ("rl_training_provenance_digest", self.rl_training_provenance_digest),
            ("training_config_digest", self.training_config_digest),
            ("architecture_spec_digest", self.architecture_spec_digest),
            ("router_contract_digest", self.router_contract_digest),
            (
                "episode_sampling_contract_digest",
                self.episode_sampling_contract_digest,
            ),
            (
                "evaluation_checkpoint_contract_digest",
                self.evaluation_checkpoint_contract_digest,
            ),
            ("baseline_contract_digest", self.baseline_contract_digest),
            ("selection_thresholds_digest", self.selection_thresholds_digest),
        ):
            require_sha256(value, field=f"U2 contract {field_name}")

        u1_cutoff = _integer(
            self.u1_normalizer_knowledge_cutoff_ns,
            field="U2 U1 normalizer knowledge cutoff",
        )
        fit_end = _integer(self.fit_end_ns, field="U2 FIT end")
        rl_cutoff = _integer(
            self.rl_training_knowledge_cutoff_ns,
            field="U2 RL_TRAINING knowledge cutoff",
        )
        if u1_cutoff <= 0 or fit_end <= 0 or rl_cutoff <= 0:
            raise ValueError("U2 cutoff timestamps must be positive")
        if u1_cutoff != fit_end:
            raise ValueError("U2 U1 normalizer cutoff must equal FIT end")
        if rl_cutoff != fit_end:
            raise ValueError("U2 RL_TRAINING provenance cutoff must equal FIT end")

        payload = _normalized_training_payload(self.training_config_payload)
        canonical_payload = _canonical_training_payload()
        if payload != canonical_payload:
            raise ValueError(
                "U2 training config recipe is not the canonical Base PPO V1"
            )
        expected_training_digest = content_digest(payload)
        if self.training_config_digest != expected_training_digest:
            raise ValueError("U2 training config digest mismatch")
        object.__setattr__(self, "training_config_payload", MappingProxyType(payload))

        if self.architecture_name != U2_ARCHITECTURE_NAME:
            raise ValueError("U2 architecture must be u_medium_direct")
        if self.training_seeds != U2_TRAINING_SEEDS:
            raise ValueError("U2 training seeds must be exactly (0, 1, 2)")
        if self.primary_candidate_seed != U2_PRIMARY_CANDIDATE_SEED:
            raise ValueError("U2 primary candidate seed must be seed 0")
        if self.instrument_context_enabled is not False:
            raise ValueError("U2 instrument context must remain disabled")
        if self.v4_context_enabled is not False:
            raise ValueError("U2 V4 context must remain disabled")
        if self.production_status != U2_PRODUCTION_STATUS:
            raise ValueError("Universal Trade RL U2 remains Production NO-GO")

        expected_rules = {
            "architecture_spec_digest": content_digest(
                _architecture_contract_payload()
            ),
            "router_contract_digest": content_digest(
                _router_contract_payload(
                    universe_manifest_digest=self.universe_manifest_digest,
                    time_partition_digest=self.time_partition_digest,
                )
            ),
            "episode_sampling_contract_digest": content_digest(
                _episode_sampling_contract_payload(
                    time_partition_digest=self.time_partition_digest,
                    fit_end_ns=fit_end,
                )
            ),
            "evaluation_checkpoint_contract_digest": content_digest(
                _evaluation_checkpoint_contract_payload(
                    training_config_digest=self.training_config_digest
                )
            ),
            "baseline_contract_digest": content_digest(
                _baseline_contract_payload(u1_contract_digest=self.u1_contract_digest)
            ),
            "selection_thresholds_digest": content_digest(
                _selection_thresholds_payload()
            ),
        }
        for field_name, expected in expected_rules.items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"U2 {field_name} does not match preregistration")

        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest:
            require_sha256(self.digest, field="U2 contract artifact digest")
            if self.digest != expected:
                raise ValueError("Universal Trade RL U2 contract digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "universe_manifest_digest": self.universe_manifest_digest,
            "u1_contract_digest": self.u1_contract_digest,
            "u1_normalizer_digest": self.u1_normalizer_digest,
            "u1_normalizer_knowledge_cutoff_ns": (
                self.u1_normalizer_knowledge_cutoff_ns
            ),
            "time_partition_digest": self.time_partition_digest,
            "fit_end_ns": self.fit_end_ns,
            "rl_training_provenance_digest": self.rl_training_provenance_digest,
            "rl_training_knowledge_cutoff_ns": self.rl_training_knowledge_cutoff_ns,
            "training_config_digest": self.training_config_digest,
            "training_config_payload": dict(self.training_config_payload),
            "architecture_name": self.architecture_name,
            "architecture_spec_digest": self.architecture_spec_digest,
            "training_seeds": self.training_seeds,
            "primary_candidate_seed": self.primary_candidate_seed,
            "router_contract_digest": self.router_contract_digest,
            "episode_sampling_contract_digest": self.episode_sampling_contract_digest,
            "evaluation_checkpoint_contract_digest": (
                self.evaluation_checkpoint_contract_digest
            ),
            "baseline_contract_digest": self.baseline_contract_digest,
            "selection_thresholds_digest": self.selection_thresholds_digest,
            "instrument_context_enabled": self.instrument_context_enabled,
            "v4_context_enabled": self.v4_context_enabled,
            "production_status": self.production_status,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload

    @classmethod
    def from_payload(cls, payload: object) -> UniversalTradeRLU2Contract:
        values = _exact_mapping(payload, keys=_CONTRACT_KEYS, field="U2 contract")
        string_fields = (
            "schema_version",
            "universe_manifest_digest",
            "u1_contract_digest",
            "u1_normalizer_digest",
            "time_partition_digest",
            "rl_training_provenance_digest",
            "training_config_digest",
            "architecture_name",
            "architecture_spec_digest",
            "router_contract_digest",
            "episode_sampling_contract_digest",
            "evaluation_checkpoint_contract_digest",
            "baseline_contract_digest",
            "selection_thresholds_digest",
            "production_status",
            "artifact_digest",
        )
        strings: dict[str, str] = {}
        for field_name in string_fields:
            value = values[field_name]
            if not isinstance(value, str):
                raise ValueError(f"U2 contract {field_name} must be a string")
            strings[field_name] = value

        raw_seeds = _sequence(values["training_seeds"], field="U2 training seeds")
        seeds = tuple(_integer(item, field="U2 training seed") for item in raw_seeds)
        instrument_context = values["instrument_context_enabled"]
        v4_context = values["v4_context_enabled"]
        if not isinstance(instrument_context, bool) or not isinstance(v4_context, bool):
            raise ValueError("U2 context-enabled flags must be boolean")

        return cls(
            universe_manifest_digest=strings["universe_manifest_digest"],
            u1_contract_digest=strings["u1_contract_digest"],
            u1_normalizer_digest=strings["u1_normalizer_digest"],
            u1_normalizer_knowledge_cutoff_ns=_integer(
                values["u1_normalizer_knowledge_cutoff_ns"],
                field="U2 U1 normalizer knowledge cutoff",
            ),
            time_partition_digest=strings["time_partition_digest"],
            fit_end_ns=_integer(values["fit_end_ns"], field="U2 FIT end"),
            rl_training_provenance_digest=strings["rl_training_provenance_digest"],
            rl_training_knowledge_cutoff_ns=_integer(
                values["rl_training_knowledge_cutoff_ns"],
                field="U2 RL_TRAINING knowledge cutoff",
            ),
            training_config_digest=strings["training_config_digest"],
            training_config_payload=_normalized_training_payload(
                values["training_config_payload"]
            ),
            architecture_name=strings["architecture_name"],
            architecture_spec_digest=strings["architecture_spec_digest"],
            training_seeds=seeds,
            primary_candidate_seed=_integer(
                values["primary_candidate_seed"], field="U2 primary candidate seed"
            ),
            router_contract_digest=strings["router_contract_digest"],
            episode_sampling_contract_digest=strings[
                "episode_sampling_contract_digest"
            ],
            evaluation_checkpoint_contract_digest=strings[
                "evaluation_checkpoint_contract_digest"
            ],
            baseline_contract_digest=strings["baseline_contract_digest"],
            selection_thresholds_digest=strings["selection_thresholds_digest"],
            instrument_context_enabled=instrument_context,
            v4_context_enabled=v4_context,
            production_status=strings["production_status"],
            schema_version=strings["schema_version"],
            digest=strings["artifact_digest"],
        )


def _train_symbols(manifest: UniversalTradeRLUniverseManifest) -> tuple[str, ...]:
    return tuple(
        entry.symbol
        for entry in manifest.entries
        if entry.role is UniversalTradeRLSymbolRole.TRAIN
    )


def build_universal_trade_rl_u2_contract(
    *,
    manifest: UniversalTradeRLUniverseManifest,
    u1_contract: UniversalTradeRLU1Contract,
    time_partition: UniversalTradeRLU2TimePartition,
    rl_training_provenance: UniversalTradeRLFitProvenance,
    training_config: ResidualTrainingConfig,
) -> UniversalTradeRLU2Contract:
    """Bind U0, U1, FIT, Train-only provenance, and the exact Base PPO V1 recipe."""

    if not isinstance(manifest, UniversalTradeRLUniverseManifest):
        raise TypeError("U2 manifest is invalid")
    if not isinstance(u1_contract, UniversalTradeRLU1Contract):
        raise TypeError("U2 U1 contract is invalid")
    if not isinstance(time_partition, UniversalTradeRLU2TimePartition):
        raise TypeError("U2 time partition is invalid")
    if not isinstance(rl_training_provenance, UniversalTradeRLFitProvenance):
        raise TypeError("U2 RL_TRAINING provenance is invalid")
    if not isinstance(training_config, ResidualTrainingConfig):
        raise TypeError("U2 training config is invalid")

    if u1_contract.universe_manifest_digest != manifest.digest:
        raise ValueError("U2 U1 contract universe manifest identity mismatch")
    if time_partition.universe_manifest_digest != manifest.digest:
        raise ValueError("U2 time partition universe manifest identity mismatch")
    require_universal_trade_rl_train_only_provenance(
        rl_training_provenance,
        manifest=manifest,
    )
    if rl_training_provenance.purpose is not UniversalTradeRLFitPurpose.RL_TRAINING:
        raise ValueError("U2 provenance purpose must be RL_TRAINING")

    expected_train_symbols = _train_symbols(manifest)
    if rl_training_provenance.source_symbols != expected_train_symbols:
        raise ValueError(
            "U2 RL_TRAINING provenance must cover the complete Train symbol scope"
        )
    fit_end = time_partition.fit_end_ns
    if u1_contract.normalizer_knowledge_cutoff_ns != fit_end:
        raise ValueError("U2 U1 normalizer cutoff must equal FIT end")
    if rl_training_provenance.knowledge_cutoff != fit_end:
        raise ValueError("U2 RL_TRAINING provenance cutoff must equal FIT end")

    training_payload = _normalized_training_payload(training_config.digest_payload())
    canonical_training_payload = _canonical_training_payload()
    if training_payload != canonical_training_payload:
        raise ValueError("U2 training config recipe is not the canonical Base PPO V1")
    training_digest = content_digest(training_payload)

    architecture_digest = content_digest(_architecture_contract_payload())
    router_digest = content_digest(
        _router_contract_payload(
            universe_manifest_digest=manifest.digest,
            time_partition_digest=time_partition.digest,
        )
    )
    episode_sampling_digest = content_digest(
        _episode_sampling_contract_payload(
            time_partition_digest=time_partition.digest,
            fit_end_ns=fit_end,
        )
    )
    evaluation_checkpoint_digest = content_digest(
        _evaluation_checkpoint_contract_payload(training_config_digest=training_digest)
    )
    baseline_digest = content_digest(
        _baseline_contract_payload(u1_contract_digest=u1_contract.digest)
    )
    selection_digest = content_digest(_selection_thresholds_payload())

    return UniversalTradeRLU2Contract(
        universe_manifest_digest=manifest.digest,
        u1_contract_digest=u1_contract.digest,
        u1_normalizer_digest=u1_contract.normalizer_digest,
        u1_normalizer_knowledge_cutoff_ns=u1_contract.normalizer_knowledge_cutoff_ns,
        time_partition_digest=time_partition.digest,
        fit_end_ns=fit_end,
        rl_training_provenance_digest=rl_training_provenance.digest,
        rl_training_knowledge_cutoff_ns=rl_training_provenance.knowledge_cutoff,
        training_config_digest=training_digest,
        training_config_payload=training_payload,
        architecture_name=U2_ARCHITECTURE_NAME,
        architecture_spec_digest=architecture_digest,
        training_seeds=U2_TRAINING_SEEDS,
        primary_candidate_seed=U2_PRIMARY_CANDIDATE_SEED,
        router_contract_digest=router_digest,
        episode_sampling_contract_digest=episode_sampling_digest,
        evaluation_checkpoint_contract_digest=evaluation_checkpoint_digest,
        baseline_contract_digest=baseline_digest,
        selection_thresholds_digest=selection_digest,
        instrument_context_enabled=False,
        v4_context_enabled=False,
        production_status=U2_PRODUCTION_STATUS,
    )


def build_universal_trade_rl_u2_base_training_identity(
    *,
    contract: UniversalTradeRLU2Contract,
    rl_training_provenance: UniversalTradeRLFitProvenance,
) -> UniversalTradeRLRunIdentity:
    """Build the exact U0 BASE_TRAINING identity for this U2 generation."""

    if not isinstance(contract, UniversalTradeRLU2Contract):
        raise TypeError("U2 contract is invalid")
    if not isinstance(rl_training_provenance, UniversalTradeRLFitProvenance):
        raise TypeError("U2 RL_TRAINING provenance is invalid")
    if rl_training_provenance.purpose is not UniversalTradeRLFitPurpose.RL_TRAINING:
        raise ValueError("U2 provenance purpose must be RL_TRAINING")
    if (
        rl_training_provenance.universe_manifest_digest
        != contract.universe_manifest_digest
    ):
        raise ValueError("U2 BASE_TRAINING provenance universe identity mismatch")
    if rl_training_provenance.digest != contract.rl_training_provenance_digest:
        raise ValueError("U2 BASE_TRAINING fit provenance digest mismatch")
    if rl_training_provenance.knowledge_cutoff != contract.fit_end_ns:
        raise ValueError("U2 BASE_TRAINING provenance cutoff mismatch")

    return UniversalTradeRLRunIdentity(
        stage=UniversalTradeRLRunStage.BASE_TRAINING,
        universe_manifest_digest=contract.universe_manifest_digest,
        model_config_digest=contract.digest,
        fit_provenance_digests=(rl_training_provenance.digest,),
        admission_authorization_digest=None,
    )


def require_universal_trade_rl_u2_base_training_identity(
    identity: UniversalTradeRLRunIdentity,
    *,
    contract: UniversalTradeRLU2Contract,
    rl_training_provenance: UniversalTradeRLFitProvenance,
) -> UniversalTradeRLRunIdentity:
    """Fail closed unless a run identity is exactly the preregistered U2 BASE_TRAINING run."""

    if not isinstance(identity, UniversalTradeRLRunIdentity):
        raise TypeError("U2 BASE_TRAINING run identity is invalid")
    expected = build_universal_trade_rl_u2_base_training_identity(
        contract=contract,
        rl_training_provenance=rl_training_provenance,
    )
    if identity.stage is not UniversalTradeRLRunStage.BASE_TRAINING:
        raise ValueError("U2 run identity stage must be BASE_TRAINING")
    if identity.universe_manifest_digest != expected.universe_manifest_digest:
        raise ValueError("U2 BASE_TRAINING universe manifest identity mismatch")
    if identity.model_config_digest != expected.model_config_digest:
        raise ValueError("U2 BASE_TRAINING model config must equal U2 contract digest")
    if identity.fit_provenance_digests != expected.fit_provenance_digests:
        raise ValueError("U2 BASE_TRAINING fit provenance must be exact")
    if identity.admission_authorization_digest is not None:
        raise ValueError("U2 BASE_TRAINING cannot carry Admission authorization")
    return identity


__all__ = [
    "UNIVERSAL_TRADE_RL_U2_CONTRACT_SCHEMA",
    "U2_ARCHITECTURE_NAME",
    "U2_FINAL_TIMESTEPS",
    "U2_PRIMARY_CANDIDATE_SEED",
    "U2_PRODUCTION_STATUS",
    "U2_TRAINING_SEEDS",
    "UniversalTradeRLU2Contract",
    "build_universal_trade_rl_u2_base_training_identity",
    "build_universal_trade_rl_u2_contract",
    "build_universal_trade_rl_u2_training_config",
    "require_universal_trade_rl_u2_base_training_identity",
]
