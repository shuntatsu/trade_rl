from __future__ import annotations

from types import SimpleNamespace

from trade_rl.artifacts.hashing import content_digest
from trade_rl.rl.training import ResidualTrainingConfig
from trade_rl.rl.universal_instrument_binding import InstrumentDatasetBinding
from trade_rl.workflows.universal_training_runner import (
    UniversalRoutedEnvironmentFactory,
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


def _routed_factory() -> UniversalRoutedEnvironmentFactory:
    bindings = (_binding("AAAUSDT"), _binding("BBBUSDT"))

    class Provider:
        digest = _digest("provider")
        schema_digest = _digest("provider-schema")

        def __call__(self, *_args, **_kwargs):
            return object()

    return UniversalRoutedEnvironmentFactory(
        train_symbols=("AAAUSDT", "BBBUSDT"),
        partition_digest=_digest("partition"),
        bindings=bindings,
        concrete_environment_factory=lambda _binding: object(),
        instrument_context_provider=Provider(),
        training_contract_digest=_digest("training-contract"),
        run_seed=23,
    )


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


def test_assemble_routes_causal_teacher_through_shared_package(monkeypatch) -> None:
    import trade_rl.workflows.universal_training_runner as module
    from trade_rl.workflows.universal_training_runner import (
        assemble_universal_sb3_training_backend,
    )

    routed = _routed_factory()
    package = SimpleNamespace(
        batches={"AAAUSDT": object(), "BBBUSDT": object()},
        teacher_config_digest=_digest("teacher-config"),
        digest=_digest("teacher-package"),
    )
    bundle = SimpleNamespace(
        teacher_artifact=SimpleNamespace(artifact_digest=_digest("teacher-artifact"))
    )
    observed: dict[str, object] = {}

    def build_package(**kwargs):
        observed["package_kwargs"] = kwargs
        return package

    monkeypatch.setattr(
        module,
        "build_universal_causal_alpha_teacher_package",
        build_package,
    )
    monkeypatch.setattr(
        module,
        "build_universal_teacher_batches",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("legacy teacher builder must not run for causal alpha")
        ),
    )

    def build_bundle(**kwargs):
        observed["bundle_kwargs"] = kwargs
        return bundle

    monkeypatch.setattr(
        module,
        "build_universal_pretraining_bundle_from_batches",
        build_bundle,
    )
    monkeypatch.setattr(
        module,
        "build_universal_pretraining_hook",
        lambda *_args, **_kwargs: object(),
    )

    class Backend:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

    monkeypatch.setattr(module, "StableBaselines3Backend", Backend)

    _, actual_bundle = assemble_universal_sb3_training_backend(
        routed_environment_factory=routed,
        training=_training(),
        fold_train_range=(19, 211),
        normalizer_digest=_digest("normalizer"),
        feature_schema_digest=_digest("features"),
    )

    assert actual_bundle is bundle
    package_kwargs = observed["package_kwargs"]
    assert isinstance(package_kwargs, dict)
    assert package_kwargs["train_symbols"] == routed.train_symbols
    assert package_kwargs["bindings"] == routed.bindings
    assert package_kwargs["fold_train_range"] == (19, 211)
    assert package_kwargs["feature_schema_digest"] == _digest("features")
    bundle_kwargs = observed["bundle_kwargs"]
    assert isinstance(bundle_kwargs, dict)
    assert bundle_kwargs["batches"] == package.batches
    assert all(
        bundle_kwargs["batches"][symbol] is package.batches[symbol]
        for symbol in routed.train_symbols
    )
    assert bundle_kwargs["teacher_kind"] == "causal_alpha_ridge"
    assert bundle_kwargs["causal_teacher_package"] is package


def test_assemble_reuses_explicit_causal_package(monkeypatch) -> None:
    import trade_rl.workflows.universal_training_runner as module
    from trade_rl.workflows.universal_training_runner import (
        assemble_universal_sb3_training_backend,
    )

    routed = _routed_factory()
    package = SimpleNamespace(
        batches={"AAAUSDT": object(), "BBBUSDT": object()},
        teacher_config_digest=_digest("teacher-config"),
        digest=_digest("teacher-package"),
    )
    bundle = SimpleNamespace(
        teacher_artifact=SimpleNamespace(artifact_digest=_digest("teacher-artifact"))
    )
    monkeypatch.setattr(
        module,
        "build_universal_causal_alpha_teacher_package",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("explicit package must be reused")
        ),
    )
    monkeypatch.setattr(
        module,
        "build_universal_pretraining_bundle_from_batches",
        lambda **_kwargs: bundle,
    )
    monkeypatch.setattr(
        module,
        "build_universal_pretraining_hook",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        module,
        "StableBaselines3Backend",
        lambda *_args, **_kwargs: object(),
    )

    _, actual_bundle = assemble_universal_sb3_training_backend(
        routed_environment_factory=routed,
        training=_training(),
        fold_train_range=(19, 211),
        normalizer_digest=_digest("normalizer"),
        feature_schema_digest=_digest("features"),
        causal_teacher_package=package,
    )
    assert actual_bundle is bundle
