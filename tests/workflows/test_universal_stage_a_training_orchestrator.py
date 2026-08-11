from __future__ import annotations

from types import SimpleNamespace

from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.stage_a_zero_shot_contracts import StageACandidate
from trade_rl.rl.training import ResidualTrainingConfig
from trade_rl.rl.universal_architecture import (
    UniversalArchitectureName,
    apply_architecture_to_training_config,
)
from trade_rl.rl.universal_instrument_binding import InstrumentDatasetBinding
from trade_rl.workflows.universal_training import universal_training_contract_digest


def _digest(label: str) -> str:
    return content_digest(label)


def _binding(symbol: str) -> InstrumentDatasetBinding:
    dataset_id = _digest(f"dataset:{symbol}")
    return InstrumentDatasetBinding(
        concrete_symbol=symbol,
        source_dataset_id=dataset_id,
        symbol_dataset_digest=dataset_id,
        execution_metadata_digest=_digest(f"metadata:{symbol}"),
        instrument_descriptor_digest=_digest(f"descriptor:{symbol}"),
        split="train",
    )


def _base_training() -> ResidualTrainingConfig:
    return ResidualTrainingConfig(
        timesteps=64,
        gamma=1.0,
        seeds=(7, 11),
        n_envs=4,
        n_steps=8,
        batch_size=8,
        behavior_cloning_epochs=2,
        behavior_cloning_teacher="oracle",
        behavior_cloning_seed=17,
        behavior_cloning_validation_fraction=0.1,
    )


def _base_routed_factory():
    from trade_rl.workflows.universal_training_runner import (
        UniversalRoutedEnvironmentFactory,
    )

    return UniversalRoutedEnvironmentFactory(
        train_symbols=("AAAUSDT",),
        partition_digest=_digest("partition"),
        bindings=(_binding("AAAUSDT"),),
        concrete_environment_factory=lambda _binding: object(),
        instrument_context_provider=lambda *_args, **_kwargs: object(),
        training_contract_digest=_digest("placeholder-training-contract"),
        run_seed=23,
    )


def test_build_universal_training_runtime_rebinds_candidate_training_contract() -> None:
    from trade_rl.workflows.universal_training_runner import (
        build_universal_training_runtime,
    )

    training = apply_architecture_to_training_config(
        _base_training(),
        UniversalArchitectureName.U_MEDIUM_DIRECT,
    )
    runtime = build_universal_training_runtime(
        train_symbols=("AAAUSDT",),
        catalog_digest=_digest("catalog"),
        partition_digest=_digest("partition"),
        split_manifest_digest=_digest("split"),
        feature_schema_digest=_digest("features"),
        statistics_digest=_digest("statistics"),
        instrument_context_schema_digest=_digest("context"),
        routed_environment_factory=_base_routed_factory(),
        training=training,
    )

    expected_contract = universal_training_contract_digest(
        partition_digest=_digest("partition"),
        feature_schema_digest=_digest("features"),
        statistics_digest=_digest("statistics"),
        instrument_context_schema_digest=_digest("context"),
        training_config_digest=content_digest(training.digest_payload()),
    )
    assert runtime.training_contract_digest == expected_contract
    assert (
        runtime.routed_environment_factory.training_contract_digest == expected_contract
    )
    assert runtime.pretraining_artifact_digest is None

    bound = runtime.with_pretraining_artifact(_digest("pretraining"))
    assert bound.pretraining_artifact_digest == _digest("pretraining")
    assert runtime.pretraining_artifact_digest is None


def test_train_universal_stage_a_ablation_reuses_oracle_batches_and_closes_four_candidates(
    monkeypatch,
    tmp_path,
) -> None:
    import trade_rl.workflows.universal_stage_a_training as module
    from trade_rl.workflows.universal_stage_a import UniversalStageACandidate
    from trade_rl.workflows.universal_stage_a_training import (
        train_universal_stage_a_ablation,
    )
    from trade_rl.workflows.universal_training_runner import (
        build_universal_training_runtime,
    )

    base_training = _base_training()
    base_factory = _base_routed_factory()
    runtime_calls: list[UniversalArchitectureName] = []
    assemble_calls: list[UniversalArchitectureName] = []
    train_calls: list[UniversalArchitectureName] = []
    shared_batches = object()

    def runtime_factory(architecture, training):
        runtime_calls.append(architecture)
        return build_universal_training_runtime(
            train_symbols=("AAAUSDT",),
            catalog_digest=_digest("catalog"),
            partition_digest=_digest("partition"),
            split_manifest_digest=_digest("split"),
            feature_schema_digest=_digest("features"),
            statistics_digest=_digest("statistics"),
            instrument_context_schema_digest=_digest("context"),
            routed_environment_factory=base_factory,
            training=training,
        )

    monkeypatch.setattr(
        module,
        "build_universal_oracle_batches",
        lambda **_kwargs: shared_batches,
    )

    def assemble(*, routed_environment_factory, training, oracle_batches, **_kwargs):
        assert oracle_batches is shared_batches
        architecture = UniversalArchitectureName(
            next(
                name.value
                for name in UniversalArchitectureName
                if apply_architecture_to_training_config(base_training, name)
                == training
            )
        )
        assemble_calls.append(architecture)
        return (
            SimpleNamespace(architecture=architecture),
            SimpleNamespace(
                teacher_artifact=SimpleNamespace(
                    artifact_digest=_digest(f"teacher:{architecture.value}")
                )
            ),
        )

    monkeypatch.setattr(module, "assemble_universal_sb3_training_backend", assemble)

    def train(*, runtime, training, backend, output_root, architecture_name):
        architecture = UniversalArchitectureName(architecture_name)
        assert backend.architecture is architecture
        assert runtime.pretraining_artifact_digest == _digest(
            f"teacher:{architecture.value}"
        )
        assert output_root == tmp_path / architecture.value
        train_calls.append(architecture)
        payload = {
            "schema_version": "universal_training_run_v1",
            "architecture_name": architecture.value,
            "training_config_digest": content_digest(training.digest_payload()),
            "members": [],
        }
        return {**payload, "run_digest": content_digest(payload)}

    monkeypatch.setattr(module, "train_universal_seeds", train)

    def candidate_adapter(
        *, architecture, training_config, training_manifest, output_root
    ):
        assert training_manifest["architecture_name"] == architecture.value
        assert output_root == tmp_path / architecture.value
        checkpoint_digests = tuple(
            (seed, _digest(f"checkpoint:{architecture.value}:{seed}"))
            for seed in training_config.seeds
        )
        stage = StageACandidate.create(
            candidate_id=architecture.value,
            candidate_config_digest=content_digest(training_config.digest_payload()),
            final_training_completion_digest=training_manifest["run_digest"],
            policy_identity=_digest(f"architecture:{architecture.value}"),
            checkpoint_digests=checkpoint_digests,
        )
        return UniversalStageACandidate(
            architecture=architecture,
            stage_a_candidate=stage,
            training_config=training_config,
        )

    monkeypatch.setattr(
        module,
        "build_universal_stage_a_candidate_from_training",
        candidate_adapter,
    )

    candidates = train_universal_stage_a_ablation(
        base_training=base_training,
        runtime_factory=runtime_factory,
        fold_train_range=(10, 50),
        normalizer_digest=_digest("normalizer"),
        feature_schema_digest=_digest("features"),
        output_root=tmp_path,
        verbose=1,
    )

    expected = tuple(UniversalArchitectureName)
    assert tuple(item.architecture for item in candidates) == expected
    assert tuple(runtime_calls) == expected
    assert tuple(assemble_calls) == expected
    assert tuple(train_calls) == expected
    assert len({item.fixed_condition_digest for item in candidates}) == 1
