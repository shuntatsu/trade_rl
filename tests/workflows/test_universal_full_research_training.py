from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.rl.environment_constraints import CONSTRAINT_COST_NAMES
from trade_rl.rl.training import ResidualTrainingConfig
from trade_rl.rl.universal_architecture import UniversalArchitectureName
from trade_rl.rl.universal_instrument_binding import InstrumentDatasetBinding
from trade_rl.workflows.universal_research import (
    FullResearchAlgorithm,
    build_full_research_pair_closure,
)
from trade_rl.workflows.universal_training_runner import (
    UniversalRoutedEnvironmentFactory,
    build_universal_training_runtime,
)


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


def _common(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "timesteps": 64,
        "gamma": 1.0,
        "seeds": (7, 11),
        "n_envs": 4,
        "n_steps": 8,
        "batch_size": 8,
        "behavior_cloning_epochs": 2,
        "behavior_cloning_teacher": "oracle",
        "behavior_cloning_seed": 17,
        "behavior_cloning_validation_fraction": 0.1,
    }
    values.update(overrides)
    return values


def _ppo() -> ResidualTrainingConfig:
    return ResidualTrainingConfig(**_common(algorithm="ppo"))  # type: ignore[arg-type]


def _lagrangian() -> ResidualTrainingConfig:
    count = len(CONSTRAINT_COST_NAMES)
    return ResidualTrainingConfig(
        **_common(algorithm="lagrangian_ppo"),  # type: ignore[arg-type]
        lagrangian_budgets=(0.1,) * count,
        lagrangian_dual_learning_rates=(0.05,) * count,
        lagrangian_ema_betas=(0.9,) * count,
        lagrangian_initial_multipliers=(0.0,) * count,
        lagrangian_max_multipliers=(10.0,) * count,
        lagrangian_warmup_rollouts=(0,) * count,
        lagrangian_update_interval_rollouts=(1,) * count,
        lagrangian_minimum_completed_episodes=(1,) * count,
        lagrangian_probe_episodes=2,
        lagrangian_probe_max_steps_per_episode=16,
    )


def _configs() -> dict[FullResearchAlgorithm, ResidualTrainingConfig]:
    lagrangian = _lagrangian()
    return {
        FullResearchAlgorithm.PPO: _ppo(),
        FullResearchAlgorithm.LAGRANGIAN: lagrangian,
        FullResearchAlgorithm.DISCOUNTED: replace(lagrangian, gamma=0.99),
    }


def _factory() -> UniversalRoutedEnvironmentFactory:
    return UniversalRoutedEnvironmentFactory(
        train_symbols=("AAAUSDT",),
        partition_digest=_digest("partition"),
        bindings=(_binding("AAAUSDT"),),
        concrete_environment_factory=lambda _binding: object(),
        instrument_context_provider=lambda *_args, **_kwargs: object(),
        training_contract_digest=_digest("placeholder"),
        run_seed=23,
    )


def _runtime(training: ResidualTrainingConfig):
    return build_universal_training_runtime(
        train_symbols=("AAAUSDT",),
        catalog_digest=_digest("catalog"),
        partition_digest=_digest("partition"),
        split_manifest_digest=_digest("split"),
        feature_schema_digest=_digest("features"),
        statistics_digest=_digest("statistics"),
        instrument_context_schema_digest=_digest("context"),
        routed_environment_factory=_factory(),
        training=training,
    )


def test_prepare_full_research_configs_projects_only_selected_architecture() -> None:
    from trade_rl.workflows.universal_full_research_training import (
        prepare_universal_full_research_training_configs,
    )

    prepared = prepare_universal_full_research_training_configs(
        selected_architecture=UniversalArchitectureName.U_MEDIUM_DIRECT,
        algorithm_configs=_configs(),
    )

    assert tuple(item.algorithm for item in prepared) == tuple(FullResearchAlgorithm)
    by_algorithm = {item.algorithm: item for item in prepared}
    assert by_algorithm[FullResearchAlgorithm.PPO].training_config.algorithm == "ppo"
    assert by_algorithm[FullResearchAlgorithm.PPO].training_config.gamma == 1.0
    assert (
        by_algorithm[FullResearchAlgorithm.LAGRANGIAN].training_config.algorithm
        == "lagrangian_ppo"
    )
    assert by_algorithm[FullResearchAlgorithm.LAGRANGIAN].training_config.gamma == 1.0
    assert (
        by_algorithm[FullResearchAlgorithm.DISCOUNTED].training_config.algorithm
        == "lagrangian_ppo"
    )
    assert (
        0.0 < by_algorithm[FullResearchAlgorithm.DISCOUNTED].training_config.gamma < 1.0
    )
    assert len({item.fixed_condition_digest for item in prepared}) == 1


def test_prepare_full_research_configs_rejects_discounted_non_lagrangian() -> None:
    from trade_rl.workflows.universal_full_research_training import (
        prepare_universal_full_research_training_configs,
    )

    configs = _configs()
    configs[FullResearchAlgorithm.DISCOUNTED] = replace(_ppo(), gamma=0.99)

    with pytest.raises(ValueError, match="Discounted Lagrangian PPO"):
        prepare_universal_full_research_training_configs(
            selected_architecture=UniversalArchitectureName.U_MEDIUM_DIRECT,
            algorithm_configs=configs,
        )


def test_train_full_research_comparison_reuses_oracle_targets_and_closes_algorithms(
    monkeypatch,
    tmp_path,
) -> None:
    import trade_rl.workflows.universal_full_research_training as module
    from trade_rl.workflows.universal_full_research_training import (
        train_universal_full_research_comparison,
    )

    shared_batches = object()
    oracle_calls: list[dict[str, object]] = []
    assembled: list[FullResearchAlgorithm] = []
    trained: list[FullResearchAlgorithm] = []

    def runtime_factory(algorithm, training):
        del algorithm
        return _runtime(training)

    def oracle_batches(**kwargs: object) -> object:
        oracle_calls.append(dict(kwargs))
        return shared_batches

    monkeypatch.setattr(module, "build_universal_oracle_batches", oracle_batches)

    prepared_configs = {
        item.algorithm: item.training_config
        for item in module.prepare_universal_full_research_training_configs(
            selected_architecture=UniversalArchitectureName.U_MEDIUM_DIRECT,
            algorithm_configs=_configs(),
        )
    }

    def assemble(*, training, oracle_batches, **_kwargs):
        assert oracle_batches is shared_batches
        algorithm = next(
            name for name, config in prepared_configs.items() if config == training
        )
        assembled.append(algorithm)
        return (
            SimpleNamespace(algorithm=algorithm),
            SimpleNamespace(
                teacher_artifact=SimpleNamespace(
                    artifact_digest=_digest(f"teacher:{algorithm.value}")
                )
            ),
        )

    monkeypatch.setattr(module, "assemble_universal_sb3_training_backend", assemble)

    def train(*, runtime, training, backend, output_root, architecture_name):
        algorithm = backend.algorithm
        assert training == prepared_configs[algorithm]
        assert runtime.pretraining_artifact_digest == _digest(
            f"teacher:{algorithm.value}"
        )
        assert output_root == tmp_path / algorithm.value
        assert architecture_name == UniversalArchitectureName.U_MEDIUM_DIRECT.value
        trained.append(algorithm)
        payload = {
            "schema_version": "universal_training_run_v1",
            "architecture_name": architecture_name,
            "training_config_digest": content_digest(training.digest_payload()),
            "members": [],
            "research_success": False,
            "research_success_reason": "test",
        }
        return {**payload, "run_digest": content_digest(payload)}

    monkeypatch.setattr(module, "train_universal_seeds", train)

    comparison = train_universal_full_research_comparison(
        selected_architecture=UniversalArchitectureName.U_MEDIUM_DIRECT,
        algorithm_configs=_configs(),
        runtime_factory=runtime_factory,
        fold_train_range=(10, 50),
        normalizer_digest=_digest("normalizer"),
        feature_schema_digest=_digest("features"),
        baseline_names=("supervised_allocator",),
        folds=(0, 1),
        output_root=tmp_path,
        verbose=1,
    )

    assert len(oracle_calls) == 1
    assert tuple(assembled) == tuple(FullResearchAlgorithm)
    assert tuple(trained) == tuple(FullResearchAlgorithm)
    assert tuple(run.algorithm for run in comparison.runs) == tuple(
        FullResearchAlgorithm
    )
    assert comparison.required_pairs == build_full_research_pair_closure(
        algorithms=tuple(FullResearchAlgorithm),
        baseline_names=("supervised_allocator",),
        folds=(0, 1),
        seeds=(7, 11),
    )
    assert comparison.completed_pairs == ()
