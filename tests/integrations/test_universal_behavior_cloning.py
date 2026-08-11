from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.behavior_cloning import BehaviorCloningConfig
from trade_rl.learning.episode_behavior_cloning import BehaviorCloningSplit
from trade_rl.learning.teacher_artifact import SupervisedPolicyDataset
from trade_rl.learning.universal_bc import SymbolBalancedBatchSampler


def _digest(label: str) -> str:
    return content_digest({"label": label})


def _dataset() -> SupervisedPolicyDataset:
    return SupervisedPolicyDataset(
        observations=np.arange(8, dtype=np.float32).reshape(8, 1),
        actions=np.linspace(-0.8, 0.8, 8, dtype=np.float32).reshape(8, 1),
        dataset_id=_digest("dataset"),
        train_start=0,
        train_stop=9,
        environment_digest=_digest("environment"),
        action_spec_digest=_digest("action"),
        teacher_config_digest=_digest("teacher"),
    )


def _split() -> BehaviorCloningSplit:
    return BehaviorCloningSplit(
        train_indices=np.asarray([0, 1, 2, 4, 5, 6], dtype=np.int64),
        validation_indices=np.asarray([3, 7], dtype=np.int64),
        train_episode_ids=np.asarray([0], dtype=np.int64),
        validation_episode_ids=np.asarray([1], dtype=np.int64),
    )


def test_symbol_balanced_epoch_batches_are_equal_and_cover_train_scope() -> None:
    sampler = SymbolBalancedBatchSampler(
        sample_indices={"A": (0, 1, 2, 3, 4), "B": (10, 11, 12)},
        seed=17,
    )

    first = sampler.epoch_batches(batch_size=4, epoch=2)
    second = sampler.epoch_batches(batch_size=4, epoch=2)

    assert first == second
    assert first
    for batch in first:
        assert len(batch) == 4
        assert sum(index < 10 for index in batch) == 2
        assert sum(index >= 10 for index in batch) == 2
    flattened = {index for batch in first for index in batch}
    assert {0, 1, 2, 3, 4, 10, 11, 12} <= flattened


def test_universal_bc_forwards_balanced_train_batches_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = importlib.import_module("trade_rl.integrations.universal_behavior_cloning")
    captured: dict[str, Any] = {}
    sentinel = object()

    def fake_pretrain_policy(*args: object, **kwargs: object) -> object:
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(module, "pretrain_policy", fake_pretrain_policy)
    result = module.pretrain_universal_policy(
        object(),
        _dataset(),
        symbol_sample_indices={"A": (0, 1, 2), "B": (4, 5, 6)},
        train_symbols=("A", "B"),
        config=BehaviorCloningConfig(
            epochs=2,
            learning_rate=1e-3,
            batch_size=4,
            validation_fraction=0.25,
        ),
        split=_split(),
        seed=7,
        observation_provider=None,
        output_root=tmp_path,
    )

    assert result is sentinel
    batch_provider = captured["training_batch_provider"]
    batches = batch_provider(1, _split().train_indices, 4)
    assert batches
    assert all(set(batch.tolist()) <= {0, 1, 2, 4, 5, 6} for batch in batches)
    for batch in batches:
        assert sum(int(index) in {0, 1, 2} for index in batch) == 2
        assert sum(int(index) in {4, 5, 6} for index in batch) == 2


def test_universal_bc_rejects_validation_index_in_symbol_scope(tmp_path: Path) -> None:
    module = importlib.import_module("trade_rl.integrations.universal_behavior_cloning")

    with pytest.raises(ValueError, match="train scope"):
        module.pretrain_universal_policy(
            object(),
            _dataset(),
            symbol_sample_indices={"A": (0, 1, 3), "B": (4, 5, 6)},
            train_symbols=("A", "B"),
            config=BehaviorCloningConfig(
                epochs=1,
                learning_rate=1e-3,
                batch_size=4,
                validation_fraction=0.25,
            ),
            split=_split(),
            seed=7,
            observation_provider=None,
            output_root=tmp_path,
        )
