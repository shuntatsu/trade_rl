from __future__ import annotations

from types import SimpleNamespace

from trade_rl.artifacts.hashing import content_digest
from trade_rl.rl.training import ResidualTrainingConfig
from trade_rl.rl.universal_architecture import UniversalArchitectureName
from trade_rl.workflows.universal_research import FullResearchAlgorithm


def _digest(label: str) -> str:
    return content_digest(label)


def _training() -> ResidualTrainingConfig:
    return ResidualTrainingConfig(
        timesteps=32,
        gamma=1.0,
        seeds=(3,),
        n_envs=2,
        n_steps=8,
        batch_size=8,
        behavior_cloning_epochs=2,
        behavior_cloning_teacher="causal_alpha_ridge",
        behavior_cloning_seed=17,
        behavior_cloning_validation_fraction=0.1,
    )


class _FakeRuntime:
    def __init__(self, training: ResidualTrainingConfig) -> None:
        self.training = training
        self.train_symbols = ("AAAUSDT", "BBBUSDT")
        self.catalog_digest = _digest("catalog")
        self.partition_digest = _digest("partition")
        self.split_manifest_digest = _digest("split")
        self.feature_schema_digest = _digest("features")
        self.statistics_digest = _digest("statistics")
        self.instrument_context_schema_digest = _digest("context-schema")
        self.routed_environment_factory = SimpleNamespace(
            train_symbols=self.train_symbols,
            bindings=(
                SimpleNamespace(digest=_digest("binding-a")),
                SimpleNamespace(digest=_digest("binding-b")),
            ),
            partition_digest=self.partition_digest,
            run_seed=23,
            max_cached_environments=2,
            concrete_environment_factory=lambda _binding: object(),
            instrument_context_provider=SimpleNamespace(
                digest=_digest("provider"),
                schema_digest=self.instrument_context_schema_digest,
            ),
        )

    def with_pretraining_artifact(self, _digest_value: str):
        return self


def test_u5_builds_causal_package_once_and_reuses_it_for_all_architectures(
    monkeypatch, tmp_path
) -> None:
    import trade_rl.workflows.universal_stage_a_training as module

    base = _training()
    package = SimpleNamespace(digest=_digest("package"))
    package_calls: list[dict[str, object]] = []
    assembly_packages: list[object] = []

    monkeypatch.setattr(module, "UniversalTrainingRuntime", _FakeRuntime)
    monkeypatch.setattr(
        module,
        "apply_architecture_to_training_config",
        lambda _base, _architecture: base,
    )

    def build_package(**kwargs):
        package_calls.append(kwargs)
        return package

    monkeypatch.setattr(
        module,
        "build_universal_causal_alpha_teacher_package",
        build_package,
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "build_universal_oracle_batches",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("U5 must not build Oracle batches for causal alpha")
        ),
    )

    def assemble(**kwargs):
        assert kwargs.get("oracle_batches") is None
        assembly_packages.append(kwargs.get("causal_teacher_package"))
        return object(), SimpleNamespace(
            teacher_artifact=SimpleNamespace(artifact_digest=_digest("artifact"))
        )

    monkeypatch.setattr(module, "assemble_universal_sb3_training_backend", assemble)
    monkeypatch.setattr(module, "train_universal_seeds", lambda **_kwargs: object())
    monkeypatch.setattr(
        module,
        "build_universal_stage_a_candidate_from_training",
        lambda *, architecture, **_kwargs: SimpleNamespace(
            architecture=architecture,
            fixed_condition_digest=_digest("fixed"),
        ),
    )

    result = module.train_universal_stage_a_ablation(
        base_training=base,
        runtime_factory=lambda _architecture, training: _FakeRuntime(training),
        fold_train_range=(19, 211),
        normalizer_digest=_digest("normalizer"),
        feature_schema_digest=_digest("features"),
        output_root=tmp_path,
    )

    assert len(result) == len(tuple(UniversalArchitectureName))
    assert len(package_calls) == 1
    assert len(assembly_packages) == len(tuple(UniversalArchitectureName))
    assert all(value is package for value in assembly_packages)


def test_u6_builds_causal_package_once_and_reuses_it_for_all_algorithms(
    monkeypatch, tmp_path
) -> None:
    import trade_rl.workflows.universal_full_research_training as module

    training = _training()
    architecture = UniversalArchitectureName.U_MEDIUM_DIRECT
    specs = tuple(
        SimpleNamespace(
            algorithm=algorithm,
            selected_architecture=architecture,
            training_config=training,
        )
        for algorithm in FullResearchAlgorithm
    )
    package = SimpleNamespace(digest=_digest("package"))
    package_calls: list[dict[str, object]] = []
    assembly_packages: list[object] = []

    monkeypatch.setattr(module, "UniversalTrainingRuntime", _FakeRuntime)
    monkeypatch.setattr(
        module,
        "prepare_universal_full_research_training_configs",
        lambda **_kwargs: specs,
    )

    def build_package(**kwargs):
        package_calls.append(kwargs)
        return package

    monkeypatch.setattr(
        module,
        "build_universal_causal_alpha_teacher_package",
        build_package,
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "build_universal_teacher_batches",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("U6 must not build legacy teacher batches for causal alpha")
        ),
    )

    def assemble(**kwargs):
        assert kwargs.get("oracle_batches") is None
        assembly_packages.append(kwargs.get("causal_teacher_package"))
        return object(), SimpleNamespace(
            teacher_artifact=SimpleNamespace(artifact_digest=_digest("artifact"))
        )

    monkeypatch.setattr(module, "assemble_universal_sb3_training_backend", assemble)
    monkeypatch.setattr(
        module,
        "train_universal_seeds",
        lambda **_kwargs: {"schema_version": "ignored-by-test-double"},
    )
    monkeypatch.setattr(
        module,
        "UniversalFullResearchAlgorithmRun",
        lambda *, algorithm, selected_architecture, training_config, training_manifest, output_root: SimpleNamespace(
            algorithm=algorithm,
            selected_architecture=selected_architecture,
            training_config=training_config,
            training_manifest=training_manifest,
            output_root=output_root,
            digest=_digest(f"run:{algorithm.value}"),
        ),
    )
    monkeypatch.setattr(
        module,
        "build_full_research_pair_closure",
        lambda **_kwargs: ("required-pair",),
    )

    result = module.train_universal_full_research_comparison(
        selected_architecture=architecture,
        algorithm_configs={algorithm: training for algorithm in FullResearchAlgorithm},
        runtime_factory=lambda _algorithm, config: _FakeRuntime(config),
        fold_train_range=(19, 211),
        normalizer_digest=_digest("normalizer"),
        feature_schema_digest=_digest("features"),
        baseline_names=("supervised_allocator",),
        folds=(0,),
        output_root=tmp_path,
    )

    assert len(result.runs) == len(tuple(FullResearchAlgorithm))
    assert len(package_calls) == 1
    assert len(assembly_packages) == len(tuple(FullResearchAlgorithm))
    assert all(value is package for value in assembly_packages)
