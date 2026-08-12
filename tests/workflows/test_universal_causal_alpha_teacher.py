from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.teacher_artifact import SupervisedPolicyDataset
from trade_rl.workflows.universal_causal_alpha_teacher import (
    build_chronological_episode_partition,
    latest_complete_episode_split,
    validate_universal_causal_alpha_partitions,
)


def _environment(*, dataset_id: str, episode_bars: int = 4):
    dataset = SimpleNamespace(
        dataset_id=dataset_id,
        n_bars=40,
        n_symbols=1,
    )

    def initial_weights(mode: str, start: int) -> np.ndarray:
        if mode == "cash":
            return np.zeros(1, dtype=np.float64)
        if mode == "baseline":
            return np.asarray([0.1 + 0.001 * start], dtype=np.float64)
        raise ValueError(mode)

    return SimpleNamespace(
        dataset=dataset,
        minimum_start_index=1,
        episode_bars=episode_bars,
        decision_bars=1,
        config=SimpleNamespace(initial_state_modes=("cash", "baseline")),
        initial_weights_for_reset=initial_weights,
    )


def test_chronological_partition_reserves_latest_complete_episode() -> None:
    dataset_id = content_digest("dataset-A")
    partition = build_chronological_episode_partition(
        _environment(dataset_id=dataset_id),
        train_range=(1, 32),
    )

    assert len(partition.selection_contracts) >= 1
    assert partition.holdout_contract == partition.contracts[-1]
    assert partition.holdout_contract.stop == 32
    assert partition.holdout_contract.start == 27
    assert partition.selection_contracts[-1].stop <= partition.holdout_contract.start
    assert tuple(contract.episode_index for contract in partition.contracts) == tuple(
        range(len(partition.contracts))
    )
    assert all(contract.dataset_id == dataset_id for contract in partition.contracts)


def test_partition_is_deterministic_and_uses_declared_initial_weights() -> None:
    dataset_id = content_digest("dataset-B")
    environment = _environment(dataset_id=dataset_id)
    first = build_chronological_episode_partition(environment, train_range=(1, 32))
    second = build_chronological_episode_partition(environment, train_range=(1, 32))

    assert first.digest == second.digest
    assert [contract.digest for contract in first.contracts] == [
        contract.digest for contract in second.contracts
    ]
    baseline_contracts = [
        contract
        for contract in first.contracts
        if contract.initial_state_mode == "baseline"
    ]
    assert baseline_contracts
    assert all(contract.initial_weights[0] > 0.0 for contract in baseline_contracts)


def test_partition_rejects_missing_selection_history() -> None:
    environment = _environment(dataset_id=content_digest("too-short"), episode_bars=8)
    with pytest.raises(ValueError, match="selection episode"):
        build_chronological_episode_partition(environment, train_range=(1, 11))


def _episode_dataset() -> SupervisedPolicyDataset:
    dataset = SupervisedPolicyDataset(
        observations=np.arange(18, dtype=np.float32).reshape(9, 2),
        actions=np.linspace(-0.3, 0.3, 9, dtype=np.float32)[:, None],
        dataset_id=content_digest("teacher-dataset"),
        train_start=0,
        train_stop=10,
        environment_digest=content_digest("environment"),
        action_spec_digest=content_digest("action"),
        teacher_config_digest=content_digest("teacher"),
    )
    object.__setattr__(
        dataset,
        "episode_ids",
        np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int64),
    )
    object.__setattr__(
        dataset,
        "decision_indices",
        np.asarray([0, 1, 2, 4, 5, 6, 8, 9, 10], dtype=np.int64),
    )
    return dataset


def test_latest_complete_episode_split_holds_out_exact_episode() -> None:
    split = latest_complete_episode_split(_episode_dataset(), holdout_episode_id=2)

    assert split.train_episode_ids.tolist() == [0, 1]
    assert split.validation_episode_ids.tolist() == [2]
    assert split.purged_episode_ids.tolist() == []
    assert split.train_indices.tolist() == [0, 1, 2, 3, 4, 5]
    assert split.validation_indices.tolist() == [6, 7, 8]


def test_universal_scope_requires_one_partition_for_each_train_symbol() -> None:
    first = build_chronological_episode_partition(
        _environment(dataset_id=content_digest("AAA")),
        train_range=(1, 32),
    )
    second = build_chronological_episode_partition(
        _environment(dataset_id=content_digest("BBB")),
        train_range=(1, 32),
    )
    scope = validate_universal_causal_alpha_partitions(
        train_symbols=("AAAUSDT", "BBBUSDT"),
        partitions={"AAAUSDT": first, "BBBUSDT": second},
    )

    assert tuple(scope) == ("AAAUSDT", "BBBUSDT")
    assert len(scope) == 2

    with pytest.raises(ValueError, match="exactly match train_symbols"):
        validate_universal_causal_alpha_partitions(
            train_symbols=("AAAUSDT", "BBBUSDT"),
            partitions={"AAAUSDT": first},
        )
