from __future__ import annotations

import importlib
from dataclasses import dataclass, replace

import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.universal_trade_rl_universe import UniversalTradeRLUniverseConfig
from trade_rl.rl.universal_architecture import (
    UniversalArchitectureName,
    apply_architecture_to_training_config,
)
from trade_rl.workflows.universal_trade_rl_data_provenance import (
    UniversalTradeRLFitProvenance,
    UniversalTradeRLFitPurpose,
    build_universal_trade_rl_fit_provenance,
)
from trade_rl.workflows.universal_trade_rl_run_identity import (
    UniversalTradeRLRunIdentity,
    UniversalTradeRLRunStage,
)
from trade_rl.workflows.universal_trade_rl_u1_contract import UniversalTradeRLU1Contract
from trade_rl.workflows.universal_trade_rl_u2_time_partition import (
    UniversalTradeRLU2TimePartition,
    build_universal_trade_rl_u2_time_partition,
)
from trade_rl.workflows.universal_trade_rl_universe_access import (
    UniversalTradeRLAccessPhase,
    UniversalTradeRLUniverseAccess,
)
from trade_rl.workflows.universal_trade_rl_universe_config import (
    UniversalTradeRLSymbolSource,
)
from trade_rl.workflows.universal_trade_rl_universe_manifest import (
    UniversalTradeRLUniverseManifest,
    build_universal_trade_rl_universe_manifest,
)

_STEP_NS = 15 * 60 * 1_000_000_000
_BARS_PER_DAY = 96
_START_NS = _STEP_NS * 2_000_000
_TOTAL_BARS = 620 * _BARS_PER_DAY


@dataclass(frozen=True, slots=True)
class U2ContractFixture:
    manifest: UniversalTradeRLUniverseManifest
    time_partition: UniversalTradeRLU2TimePartition
    u1_contract: UniversalTradeRLU1Contract
    rl_training_provenance: UniversalTradeRLFitProvenance


def _module():
    try:
        return importlib.import_module(
            "trade_rl.workflows.universal_trade_rl_u2_contract"
        )
    except ModuleNotFoundError:
        pytest.fail("Universal Trade RL U2 training contract is not implemented")


def _manifest() -> UniversalTradeRLUniverseManifest:
    config = UniversalTradeRLUniverseConfig(
        train_symbols=("BTCUSDT", "ETHUSDT"),
        development_symbols=("SOLUSDT",),
        admission_symbols=("XRPUSDT",),
    )
    sources = tuple(
        UniversalTradeRLSymbolSource(
            symbol=symbol,
            dataset_digest=digest_char * 64,
            first_timestamp_ns=_START_NS,
            last_timestamp_ns=_START_NS + (_TOTAL_BARS - 1) * _STEP_NS,
            row_count=_TOTAL_BARS,
        )
        for symbol, digest_char in (
            ("BTCUSDT", "a"),
            ("ETHUSDT", "b"),
            ("SOLUSDT", "c"),
            ("XRPUSDT", "d"),
        )
    )
    return build_universal_trade_rl_universe_manifest(config=config, sources=sources)


def _u1_contract(
    *,
    manifest: UniversalTradeRLUniverseManifest,
    fit_end_ns: int,
) -> UniversalTradeRLU1Contract:
    return UniversalTradeRLU1Contract(
        universe_manifest_digest=manifest.digest,
        u0_identity_digest="8" * 64,
        policy_contract_digest="7" * 64,
        normalizer_digest="e" * 64,
        normalizer_provenance_digest="f" * 64,
        normalizer_knowledge_cutoff_ns=fit_end_ns,
        normalizer_clip_value=10.0,
        observation_schema_digest="1" * 64,
        state_layout_digest="2" * 64,
        policy_state_fields=("current_weight",),
        runtime_config_digest="3" * 64,
        execution_policy_digest="4" * 64,
        pretrade_risk_digest="5" * 64,
        portfolio_risk_digest="6" * 64,
    )


def _fixture() -> U2ContractFixture:
    manifest = _manifest()
    partition = build_universal_trade_rl_u2_time_partition(manifest=manifest)
    access = UniversalTradeRLUniverseAccess.for_phase(
        manifest=manifest,
        phase=UniversalTradeRLAccessPhase.TRAIN,
    )
    provenance = build_universal_trade_rl_fit_provenance(
        manifest=manifest,
        access=access,
        purpose=UniversalTradeRLFitPurpose.RL_TRAINING,
        source_symbols=("BTCUSDT", "ETHUSDT"),
        knowledge_cutoff=partition.fit_end_ns,
    )
    return U2ContractFixture(
        manifest=manifest,
        time_partition=partition,
        u1_contract=_u1_contract(
            manifest=manifest,
            fit_end_ns=partition.fit_end_ns,
        ),
        rl_training_provenance=provenance,
    )


@pytest.fixture(scope="module")
def u2_fixture() -> U2ContractFixture:
    return _fixture()


def test_u2_training_config_is_exact_preregistered_base_ppo_v1() -> None:
    module = _module()
    config = module.build_universal_trade_rl_u2_training_config()

    assert config.algorithm == "ppo"
    assert config.timesteps == 524_288
    assert config.seeds == (0, 1, 2)
    assert config.n_envs == 8
    assert config.n_steps == 128
    assert config.batch_size == 256
    assert config.n_epochs == 10
    assert config.learning_rate == pytest.approx(0.00012)
    assert config.learning_rate_schedule == "linear"
    assert config.learning_rate_final_ratio == pytest.approx(0.1)
    assert config.gamma == pytest.approx(0.998969062762624)
    assert config.decision_hours == pytest.approx(0.25)
    assert config.discount_half_life_hours == pytest.approx(168.0)
    assert config.gae_lambda == pytest.approx(0.95)
    assert config.clip_range == pytest.approx(0.2)
    assert config.normalize_advantage is True
    assert config.ent_coef == pytest.approx(0.0)
    assert config.vf_coef == pytest.approx(0.5)
    assert config.max_grad_norm == pytest.approx(0.5)
    assert config.target_kl == pytest.approx(0.02)
    assert config.log_std_init == pytest.approx(-4.0)
    assert config.use_sde is False

    assert config.policy == "MultiInputPolicy"
    assert str(config.observation_encoder) == "hierarchical_sequence_v2"
    assert config.policy_actor_head == "shared_target_v1"
    assert config.sequence_tcn_capacity == "compact"
    assert config.sequence_d_model == 256
    assert config.sequence_timeframe_attention_heads == 4
    assert config.sequence_timeframe_attention_layers == 1
    assert config.sequence_timeframe_ffn_multiplier == 3
    assert config.sequence_dropout == pytest.approx(0.0)
    assert config.policy_net_arch == (256, 128)
    assert config.value_net_arch == (256, 128)

    assert config.device == "cuda"
    assert str(config.cuda_runtime_mode) == "deterministic"
    assert config.vector_environment_mode == "in_process"
    assert config.sequence_compile is False
    assert config.sequence_compile_mode == "reduce-overhead"
    assert config.sequence_transfer_mode == "synchronous"
    assert config.hierarchical_gate_temperature == pytest.approx(1.0)
    assert config.checkpoint_interval_steps == 32_768
    assert config.max_checkpoints == 8
    assert config.max_policy_parameters == 12_000_000
    assert config.max_rollout_buffer_bytes == 805_306_368
    assert config.tensorboard_enabled is True
    assert config.tensorboard_log_interval == 1

    assert config.behavior_cloning_epochs == 0
    assert config.behavior_cloning_critic_warm_start_steps == 0
    assert config.behavior_cloning_joint_warm_start_steps == 0
    assert config.lagrangian_budgets == ()
    assert config.rounded_timesteps == 524_288


def test_u2_contract_binds_u0_u1_time_provenance_and_full_training_payload(
    u2_fixture: U2ContractFixture,
) -> None:
    module = _module()
    training = module.build_universal_trade_rl_u2_training_config()
    contract = module.build_universal_trade_rl_u2_contract(
        manifest=u2_fixture.manifest,
        u1_contract=u2_fixture.u1_contract,
        time_partition=u2_fixture.time_partition,
        rl_training_provenance=u2_fixture.rl_training_provenance,
        training_config=training,
    )

    assert contract.universe_manifest_digest == u2_fixture.manifest.digest
    assert contract.u1_contract_digest == u2_fixture.u1_contract.digest
    assert contract.u1_normalizer_digest == u2_fixture.u1_contract.normalizer_digest
    assert (
        contract.u1_normalizer_knowledge_cutoff_ns
        == u2_fixture.time_partition.fit_end_ns
    )
    assert contract.time_partition_digest == u2_fixture.time_partition.digest
    assert contract.fit_end_ns == u2_fixture.time_partition.fit_end_ns
    assert (
        contract.rl_training_provenance_digest
        == u2_fixture.rl_training_provenance.digest
    )
    assert (
        contract.rl_training_knowledge_cutoff_ns == u2_fixture.time_partition.fit_end_ns
    )
    assert contract.training_config_payload == training.digest_payload()
    assert contract.training_config_digest == content_digest(training.digest_payload())
    assert contract.architecture_name == "u_medium_direct"
    assert contract.training_seeds == (0, 1, 2)
    assert contract.primary_candidate_seed == 0
    assert contract.instrument_context_enabled is False
    assert contract.v4_context_enabled is False
    assert contract.production_status == "NO-GO"
    for field in (
        "architecture_spec_digest",
        "router_contract_digest",
        "episode_sampling_contract_digest",
        "evaluation_checkpoint_contract_digest",
        "baseline_contract_digest",
        "selection_thresholds_digest",
    ):
        assert len(getattr(contract, field)) == 64


def test_u2_contract_round_trips_canonically(u2_fixture: U2ContractFixture) -> None:
    module = _module()
    contract = module.build_universal_trade_rl_u2_contract(
        manifest=u2_fixture.manifest,
        u1_contract=u2_fixture.u1_contract,
        time_partition=u2_fixture.time_partition,
        rl_training_provenance=u2_fixture.rl_training_provenance,
        training_config=module.build_universal_trade_rl_u2_training_config(),
    )

    restored = module.UniversalTradeRLU2Contract.from_payload(contract.to_payload())

    assert restored == contract
    assert restored.digest == contract.digest
    assert restored.to_payload() == contract.to_payload()


def test_u2_contract_rejects_u1_or_training_cutoff_drift(
    u2_fixture: U2ContractFixture,
) -> None:
    module = _module()
    training = module.build_universal_trade_rl_u2_training_config()
    wrong_u1 = replace(
        u2_fixture.u1_contract,
        normalizer_knowledge_cutoff_ns=u2_fixture.time_partition.fit_end_ns - _STEP_NS,
        digest="",
    )
    wrong_provenance = replace(
        u2_fixture.rl_training_provenance,
        knowledge_cutoff=u2_fixture.time_partition.fit_end_ns - _STEP_NS,
        digest="",
    )

    with pytest.raises(ValueError, match="U1|normalizer|cutoff"):
        module.build_universal_trade_rl_u2_contract(
            manifest=u2_fixture.manifest,
            u1_contract=wrong_u1,
            time_partition=u2_fixture.time_partition,
            rl_training_provenance=u2_fixture.rl_training_provenance,
            training_config=training,
        )
    with pytest.raises(ValueError, match="RL_TRAINING|provenance|cutoff"):
        module.build_universal_trade_rl_u2_contract(
            manifest=u2_fixture.manifest,
            u1_contract=u2_fixture.u1_contract,
            time_partition=u2_fixture.time_partition,
            rl_training_provenance=wrong_provenance,
            training_config=training,
        )


def test_u2_contract_requires_complete_train_only_rl_training_provenance(
    u2_fixture: U2ContractFixture,
) -> None:
    module = _module()
    training = module.build_universal_trade_rl_u2_training_config()
    wrong_purpose = replace(
        u2_fixture.rl_training_provenance,
        purpose=UniversalTradeRLFitPurpose.FEATURE_NORMALIZATION,
        digest="",
    )
    access = UniversalTradeRLUniverseAccess.for_phase(
        manifest=u2_fixture.manifest,
        phase=UniversalTradeRLAccessPhase.TRAIN,
    )
    incomplete = build_universal_trade_rl_fit_provenance(
        manifest=u2_fixture.manifest,
        access=access,
        purpose=UniversalTradeRLFitPurpose.RL_TRAINING,
        source_symbols=("BTCUSDT",),
        knowledge_cutoff=u2_fixture.time_partition.fit_end_ns,
    )

    with pytest.raises(ValueError, match="RL_TRAINING|purpose"):
        module.build_universal_trade_rl_u2_contract(
            manifest=u2_fixture.manifest,
            u1_contract=u2_fixture.u1_contract,
            time_partition=u2_fixture.time_partition,
            rl_training_provenance=wrong_purpose,
            training_config=training,
        )
    with pytest.raises(ValueError, match="Train|symbols|scope|complete"):
        module.build_universal_trade_rl_u2_contract(
            manifest=u2_fixture.manifest,
            u1_contract=u2_fixture.u1_contract,
            time_partition=u2_fixture.time_partition,
            rl_training_provenance=incomplete,
            training_config=training,
        )


@pytest.mark.parametrize(
    "changes",
    (
        {"learning_rate": 0.00013},
        {"log_std_init": -3.0},
        {"device": "cpu"},
        {"cuda_runtime_mode": "performance"},
        {"vector_environment_mode": "auto"},
        {"checkpoint_interval_steps": 65_536},
        {"tensorboard_enabled": False},
        {"sequence_transfer_mode": "pinned_non_blocking"},
    ),
)
def test_u2_contract_rejects_any_training_recipe_drift(
    u2_fixture: U2ContractFixture,
    changes: dict[str, object],
) -> None:
    module = _module()
    canonical = module.build_universal_trade_rl_u2_training_config()
    changed = replace(canonical, **changes)

    assert content_digest(changed.digest_payload()) != content_digest(
        canonical.digest_payload()
    )
    with pytest.raises(ValueError, match="training|config|recipe|canonical"):
        module.build_universal_trade_rl_u2_contract(
            manifest=u2_fixture.manifest,
            u1_contract=u2_fixture.u1_contract,
            time_partition=u2_fixture.time_partition,
            rl_training_provenance=u2_fixture.rl_training_provenance,
            training_config=changed,
        )


def test_u2_contract_rejects_architecture_drift(u2_fixture: U2ContractFixture) -> None:
    module = _module()
    canonical = module.build_universal_trade_rl_u2_training_config()
    changed = apply_architecture_to_training_config(
        canonical,
        UniversalArchitectureName.U_SMALL_DIRECT,
    )

    with pytest.raises(ValueError, match="architecture|training|config|recipe"):
        module.build_universal_trade_rl_u2_contract(
            manifest=u2_fixture.manifest,
            u1_contract=u2_fixture.u1_contract,
            time_partition=u2_fixture.time_partition,
            rl_training_provenance=u2_fixture.rl_training_provenance,
            training_config=changed,
        )


def test_u2_contract_rejects_rehashed_training_payload_tampering(
    u2_fixture: U2ContractFixture,
) -> None:
    module = _module()
    contract = module.build_universal_trade_rl_u2_contract(
        manifest=u2_fixture.manifest,
        u1_contract=u2_fixture.u1_contract,
        time_partition=u2_fixture.time_partition,
        rl_training_provenance=u2_fixture.rl_training_provenance,
        training_config=module.build_universal_trade_rl_u2_training_config(),
    )
    payload = contract.to_payload()
    training_payload = dict(payload["training_config_payload"])
    training_payload["learning_rate"] = 0.00013
    payload["training_config_payload"] = training_payload
    payload["training_config_digest"] = content_digest(training_payload)
    payload["artifact_digest"] = content_digest(
        {key: value for key, value in payload.items() if key != "artifact_digest"}
    )

    with pytest.raises(ValueError, match="training|config|recipe|canonical"):
        module.UniversalTradeRLU2Contract.from_payload(payload)


def test_u2_base_training_identity_is_exactly_bound_and_validated(
    u2_fixture: U2ContractFixture,
) -> None:
    module = _module()
    contract = module.build_universal_trade_rl_u2_contract(
        manifest=u2_fixture.manifest,
        u1_contract=u2_fixture.u1_contract,
        time_partition=u2_fixture.time_partition,
        rl_training_provenance=u2_fixture.rl_training_provenance,
        training_config=module.build_universal_trade_rl_u2_training_config(),
    )
    identity = module.build_universal_trade_rl_u2_base_training_identity(
        contract=contract,
        rl_training_provenance=u2_fixture.rl_training_provenance,
    )

    assert identity.stage is UniversalTradeRLRunStage.BASE_TRAINING
    assert identity.universe_manifest_digest == u2_fixture.manifest.digest
    assert identity.model_config_digest == contract.digest
    assert identity.fit_provenance_digests == (
        u2_fixture.rl_training_provenance.digest,
    )
    assert identity.admission_authorization_digest is None
    assert (
        module.require_universal_trade_rl_u2_base_training_identity(
            identity,
            contract=contract,
            rl_training_provenance=u2_fixture.rl_training_provenance,
        )
        is identity
    )

    wrong_model = UniversalTradeRLRunIdentity(
        stage=UniversalTradeRLRunStage.BASE_TRAINING,
        universe_manifest_digest=u2_fixture.manifest.digest,
        model_config_digest="0" * 64,
        fit_provenance_digests=(u2_fixture.rl_training_provenance.digest,),
    )
    extra_fit = UniversalTradeRLRunIdentity(
        stage=UniversalTradeRLRunStage.BASE_TRAINING,
        universe_manifest_digest=u2_fixture.manifest.digest,
        model_config_digest=contract.digest,
        fit_provenance_digests=tuple(
            sorted((u2_fixture.rl_training_provenance.digest, "f" * 64))
        ),
    )

    with pytest.raises(ValueError, match="model|config|contract"):
        module.require_universal_trade_rl_u2_base_training_identity(
            wrong_model,
            contract=contract,
            rl_training_provenance=u2_fixture.rl_training_provenance,
        )
    with pytest.raises(ValueError, match="fit|provenance"):
        module.require_universal_trade_rl_u2_base_training_identity(
            extra_fit,
            contract=contract,
            rl_training_provenance=u2_fixture.rl_training_provenance,
        )
