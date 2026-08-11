from __future__ import annotations

from types import SimpleNamespace

import pytest

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


def _training(**overrides: object) -> ResidualTrainingConfig:
    values: dict[str, object] = {
        "timesteps": 32,
        "gamma": 1.0,
        "seeds": (3, 5),
        "n_envs": 4,
        "n_steps": 8,
        "batch_size": 8,
        "behavior_cloning_epochs": 2,
        "behavior_cloning_teacher": "oracle",
        "behavior_cloning_seed": 17,
        "behavior_cloning_validation_fraction": 0.1,
    }
    values.update(overrides)
    return ResidualTrainingConfig(**values)  # type: ignore[arg-type]


def _routed_factory() -> UniversalRoutedEnvironmentFactory:
    bindings = (_binding("AAAUSDT"), _binding("BBBUSDT"))
    return UniversalRoutedEnvironmentFactory(
        train_symbols=("AAAUSDT", "BBBUSDT"),
        partition_digest=_digest("partition"),
        bindings=bindings,
        concrete_environment_factory=lambda _binding: object(),
        instrument_context_provider=lambda *_args, **_kwargs: object(),
        training_contract_digest=_digest("training-contract"),
        run_seed=23,
    )


def test_assemble_universal_sb3_training_backend_connects_oracle_bundle_and_hook(
    monkeypatch,
) -> None:
    import trade_rl.workflows.universal_training_runner as module
    from trade_rl.workflows.universal_training_runner import (
        assemble_universal_sb3_training_backend,
    )

    routed = _routed_factory()
    training = _training()
    batches = {"AAAUSDT": object(), "BBBUSDT": object()}
    bundle = SimpleNamespace(
        teacher_artifact=SimpleNamespace(artifact_digest=_digest("teacher-artifact"))
    )
    hook = object()
    observed: dict[str, object] = {}

    def build_batches(**kwargs: object) -> object:
        observed["batch_kwargs"] = kwargs
        return batches

    def build_bundle(**kwargs: object) -> object:
        observed["bundle_kwargs"] = kwargs
        return bundle

    monkeypatch.setattr(module, "build_universal_oracle_batches", build_batches)
    monkeypatch.setattr(
        module,
        "build_universal_pretraining_bundle_from_batches",
        build_bundle,
    )
    monkeypatch.setattr(
        module,
        "build_universal_pretraining_hook",
        lambda value: hook if value is bundle else None,
    )

    class Backend:
        def __init__(
            self,
            environment_factory: object,
            *,
            verbose: int,
            universal_pretraining_hook: object,
        ) -> None:
            observed["backend_environment_factory"] = environment_factory
            observed["backend_verbose"] = verbose
            observed["backend_hook"] = universal_pretraining_hook

    monkeypatch.setattr(module, "StableBaselines3Backend", Backend)

    backend, actual_bundle = assemble_universal_sb3_training_backend(
        routed_environment_factory=routed,
        training=training,
        fold_train_range=(19, 211),
        normalizer_digest=_digest("normalizer"),
        feature_schema_digest=_digest("features"),
        verbose=2,
    )

    assert isinstance(backend, Backend)
    assert actual_bundle is bundle
    batch_kwargs = observed["batch_kwargs"]
    assert isinstance(batch_kwargs, dict)
    assert batch_kwargs["train_symbols"] == routed.train_symbols
    assert batch_kwargs["bindings"] == routed.bindings
    assert (
        batch_kwargs["concrete_environment_factory"]
        is routed.concrete_environment_factory
    )
    assert batch_kwargs["fold_train_range"] == (19, 211)
    assert batch_kwargs["behavior_cloning_seed"] == 17
    assert batch_kwargs["n_envs"] == 4

    bundle_kwargs = observed["bundle_kwargs"]
    assert isinstance(bundle_kwargs, dict)
    assert bundle_kwargs["batches"] is batches
    assert (
        bundle_kwargs["instrument_context_provider"]
        is routed.instrument_context_provider
    )
    assert bundle_kwargs["partition_digest"] == routed.partition_digest
    assert bundle_kwargs["training_contract_digest"] == routed.training_contract_digest
    assert bundle_kwargs["run_seed"] == routed.run_seed
    assert bundle_kwargs["gamma"] == 1.0
    assert bundle_kwargs["validation_fraction"] == pytest.approx(0.1)
    assert bundle_kwargs["normalizer_digest"] == _digest("normalizer")
    assert bundle_kwargs["feature_schema_digest"] == _digest("features")
    assert observed["backend_environment_factory"] is routed
    assert observed["backend_verbose"] == 2
    assert observed["backend_hook"] is hook


def test_assemble_universal_sb3_training_backend_requires_fixed_bc_seed() -> None:
    from trade_rl.workflows.universal_training_runner import (
        assemble_universal_sb3_training_backend,
    )

    with pytest.raises(ValueError, match="behavior_cloning_seed"):
        assemble_universal_sb3_training_backend(
            routed_environment_factory=_routed_factory(),
            training=_training(behavior_cloning_seed=None),
            fold_train_range=(5, 30),
            normalizer_digest=_digest("normalizer"),
            feature_schema_digest=_digest("features"),
        )
