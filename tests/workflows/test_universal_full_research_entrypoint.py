from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.rl.training_run_config import TrainingRunConfig
from trade_rl.rl.universal_architecture import (
    UniversalArchitectureName,
    apply_architecture_to_training_config,
)
from trade_rl.workflows.universal_research import FullResearchAlgorithm


def _digest(label: str) -> str:
    return content_digest(label)


def _run_configs() -> dict[FullResearchAlgorithm, TrainingRunConfig]:
    root = Path("examples/binance-multitimeframe")
    common = TrainingRunConfig.from_json(
        root / "training-target-weight-growth-ppo.json"
    )
    lagrangian_training = TrainingRunConfig.from_json(
        root / "training-target-weight-constrained-growth.json"
    ).training
    discounted_training = TrainingRunConfig.from_json(
        root / "training-target-weight-constrained-growth-discounted.json"
    ).training
    return {
        FullResearchAlgorithm.PPO: common,
        FullResearchAlgorithm.LAGRANGIAN: replace(
            common,
            training=lagrangian_training,
        ),
        FullResearchAlgorithm.DISCOUNTED: replace(
            common,
            training=discounted_training,
        ),
    }


def test_run_universal_full_research_training_binds_projected_full_configs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import trade_rl.workflows.universal_full_research_entrypoint as module
    from trade_rl.workflows.universal_full_research_entrypoint import (
        run_universal_full_research_training,
    )

    authored = _run_configs()
    factory_calls: list[tuple[FullResearchAlgorithm, TrainingRunConfig]] = []

    def runtime_factory(*, algorithm, run_config):
        factory_calls.append((algorithm, run_config))
        return SimpleNamespace(algorithm=algorithm, training=run_config.training)

    run_digests = {
        algorithm: _digest(f"run:{algorithm.value}")
        for algorithm in FullResearchAlgorithm
    }

    def train(**kwargs):
        raw_configs = kwargs["algorithm_configs"]
        projected = {
            algorithm: apply_architecture_to_training_config(
                raw_configs[algorithm], UniversalArchitectureName.U_MEDIUM_DIRECT
            )
            for algorithm in FullResearchAlgorithm
        }
        wrapped_factory = kwargs["runtime_factory"]
        for algorithm in FullResearchAlgorithm:
            runtime = wrapped_factory(algorithm, projected[algorithm])
            assert runtime.algorithm is algorithm
            assert runtime.training == projected[algorithm]
        return SimpleNamespace(
            digest=_digest("comparison"),
            runs=tuple(
                SimpleNamespace(algorithm=algorithm, run_digest=run_digests[algorithm])
                for algorithm in FullResearchAlgorithm
            ),
            required_pairs=("required-pair",),
            completed_pairs=(),
        )

    monkeypatch.setattr(module, "UniversalTrainingRuntime", SimpleNamespace)
    monkeypatch.setattr(module, "train_universal_full_research_comparison", train)

    result = run_universal_full_research_training(
        selected_architecture=UniversalArchitectureName.U_MEDIUM_DIRECT,
        run_configs=authored,
        runtime_factory=runtime_factory,
        fold_train_range=(100, 500),
        normalizer_digest=_digest("normalizer"),
        feature_schema_digest=_digest("features"),
        baseline_names=("supervised_allocator",),
        folds=(0, 1),
        output_root=tmp_path,
        verbose=2,
    )

    assert result.manifest_path == tmp_path / "universal-full-research-training.json"
    assert result.comparison_digest == _digest("comparison")
    assert result.research_success is False
    assert tuple(call[0] for call in factory_calls) == tuple(FullResearchAlgorithm)
    for algorithm, projected_config in factory_calls:
        assert projected_config.training != authored[algorithm].training
        assert projected_config.environment == authored[algorithm].environment
        assert projected_config.risk == authored[algorithm].risk
        assert projected_config.reward == authored[algorithm].reward
        assert projected_config.action == authored[algorithm].action

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert (
        manifest["schema_version"] == "universal_full_research_training_entrypoint_v1"
    )
    assert manifest["selected_architecture"] == "u_medium_direct"
    assert manifest["comparison_digest"] == _digest("comparison")
    assert manifest["required_pairs"] == ["required-pair"]
    assert manifest["completed_pairs"] == []
    assert manifest["research_success"] is False
    assert manifest["algorithm_run_digests"] == {
        algorithm.value: run_digests[algorithm] for algorithm in FullResearchAlgorithm
    }


def test_run_universal_full_research_training_rejects_non_training_surface_drift(
    tmp_path: Path,
) -> None:
    from trade_rl.workflows.universal_full_research_entrypoint import (
        run_universal_full_research_training,
    )

    configs = _run_configs()
    lagrangian = configs[FullResearchAlgorithm.LAGRANGIAN]
    configs[FullResearchAlgorithm.LAGRANGIAN] = replace(
        lagrangian,
        risk=replace(lagrangian.risk, max_gross=0.9, max_abs_weight=0.9),
    )

    with pytest.raises(ValueError, match="non-training run surfaces"):
        run_universal_full_research_training(
            selected_architecture=UniversalArchitectureName.U_MEDIUM_DIRECT,
            run_configs=configs,
            runtime_factory=lambda **_kwargs: object(),
            fold_train_range=(100, 500),
            normalizer_digest=_digest("normalizer"),
            feature_schema_digest=_digest("features"),
            baseline_names=("supervised_allocator",),
            folds=(0,),
            output_root=tmp_path,
        )


def test_load_universal_runtime_factory_requires_module_function(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.workflows.universal_full_research_entrypoint import (
        load_universal_runtime_factory,
    )

    module_path = tmp_path / "fake_universal_runtime.py"
    module_path.write_text(
        "def build_runtime(*, algorithm, run_config):\n"
        "    return (algorithm, run_config)\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    factory = load_universal_runtime_factory("fake_universal_runtime:build_runtime")
    assert callable(factory)
    with pytest.raises(ValueError, match="module:function"):
        load_universal_runtime_factory("fake_universal_runtime")
