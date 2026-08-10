#!/usr/bin/env python3
"""Bind behavior-cloning result identity to the resolved sample partition."""

from __future__ import annotations

from pathlib import Path


LEARNING_PATH = Path("trade_rl/learning/behavior_cloning.py")
INTEGRATION_PATH = Path("trade_rl/integrations/behavior_cloning.py")


def replace_once(path: Path, old: str, new: str, *, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise SystemExit(f"{label} marker changed")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def patch_learning_contract() -> None:
    helper_marker = '''class ObservationBatchProvider(Protocol):
    sample_count: int

    def get(self, indices: np.ndarray) -> object: ...


'''
    helper_replacement = helper_marker + '''def _partition_indices(
    value: object,
    *,
    field: str,
    sample_count: int,
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"{field} must be an integer vector")
    resolved = np.asarray(raw, dtype=np.int64)
    if np.any(resolved < 0) or np.any(resolved >= sample_count):
        raise ValueError(f"{field} contains an out-of-range sample index")
    if np.unique(resolved).size != resolved.size:
        raise ValueError(f"{field} must not contain duplicates")
    return resolved


def behavior_cloning_split_digest(
    *,
    sample_count: int,
    training_indices: object,
    validation_indices: object,
    excluded_indices: object,
) -> str:
    """Return a canonical identity for one complete BC sample partition."""

    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count <= 0
    ):
        raise ValueError("sample_count must be a positive integer")
    training = _partition_indices(
        training_indices,
        field="training_indices",
        sample_count=sample_count,
    )
    validation = _partition_indices(
        validation_indices,
        field="validation_indices",
        sample_count=sample_count,
    )
    excluded = _partition_indices(
        excluded_indices,
        field="excluded_indices",
        sample_count=sample_count,
    )
    partition = np.concatenate((training, validation, excluded))
    if partition.size != sample_count or not np.array_equal(
        np.sort(partition), np.arange(sample_count, dtype=np.int64)
    ):
        raise ValueError("behavior-cloning split must partition every sample")
    return content_digest(
        {
            "excluded_indices": excluded.tolist(),
            "sample_count": sample_count,
            "schema_version": "behavior_cloning_split_v1",
            "training_indices": training.tolist(),
            "validation_indices": validation.tolist(),
        }
    )


'''
    replace_once(
        LEARNING_PATH,
        helper_marker,
        helper_replacement,
        label="BC split digest helper",
    )

    result_fields = '''    config: BehaviorCloningConfig
    seed: int
    validation_mse: float | None = None
'''
    result_fields_replacement = '''    config: BehaviorCloningConfig
    seed: int
    training_sample_count: int
    excluded_sample_count: int
    split_digest: str
    validation_mse: float | None = None
'''
    replace_once(
        LEARNING_PATH,
        result_fields,
        result_fields_replacement,
        label="BC result split fields",
    )

    digest_marker = '''                "observation_digest": self.observation_digest,
                "sample_count": self.sample_count,
                "schema_version": "behavior_cloning_result_v3",
                "seed": self.seed,
                "teacher_config_digest": self.teacher_config_digest,
'''
    digest_replacement = '''                "excluded_sample_count": self.excluded_sample_count,
                "observation_digest": self.observation_digest,
                "sample_count": self.sample_count,
                "schema_version": "behavior_cloning_result_v4",
                "seed": self.seed,
                "split_digest": self.split_digest,
                "teacher_config_digest": self.teacher_config_digest,
                "training_sample_count": self.training_sample_count,
'''
    replace_once(
        LEARNING_PATH,
        digest_marker,
        digest_replacement,
        label="BC result digest payload",
    )

    exports_marker = '''    "BehaviorCloningConfig",
    "BehaviorCloningResult",
    "ObservationBatchProvider",
'''
    exports_replacement = '''    "BehaviorCloningConfig",
    "BehaviorCloningResult",
    "ObservationBatchProvider",
    "behavior_cloning_split_digest",
'''
    replace_once(
        LEARNING_PATH,
        exports_marker,
        exports_replacement,
        label="BC split digest export",
    )


def patch_torch_integration() -> None:
    import_marker = '''from trade_rl.learning.behavior_cloning import (
    BehaviorCloningConfig,
    BehaviorCloningResult,
    ObservationBatchProvider,
)
'''
    import_replacement = '''from trade_rl.learning.behavior_cloning import (
    BehaviorCloningConfig,
    BehaviorCloningResult,
    ObservationBatchProvider,
    behavior_cloning_split_digest,
)
'''
    replace_once(
        INTEGRATION_PATH,
        import_marker,
        import_replacement,
        label="BC split digest import",
    )

    signature_marker = ''') -> tuple[np.ndarray, np.ndarray]:
    sample_count = dataset.sample_count
'''
    signature_replacement = ''') -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample_count = dataset.sample_count
'''
    replace_once(
        INTEGRATION_PATH,
        signature_marker,
        signature_replacement,
        label="BC partition return type",
    )

    episode_return = '''        return episode_split.train_indices, episode_split.validation_indices
'''
    episode_return_replacement = '''        return (
            episode_split.train_indices,
            episode_split.validation_indices,
            episode_split.purged_indices,
        )
'''
    replace_once(
        INTEGRATION_PATH,
        episode_return,
        episode_return_replacement,
        label="implicit episode partition",
    )

    plain_return = '''        return indices[:train_count], indices[train_count:]
'''
    plain_return_replacement = '''        return (
            indices[:train_count],
            indices[train_count:],
            np.asarray([], dtype=np.int64),
        )
'''
    replace_once(
        INTEGRATION_PATH,
        plain_return,
        plain_return_replacement,
        label="plain BC partition",
    )

    explicit_episode_return = '''        return train_indices, validation_indices

    expected_validation_count = (
'''
    explicit_episode_return_replacement = '''        return train_indices, validation_indices, purged_indices

    expected_validation_count = (
'''
    replace_once(
        INTEGRATION_PATH,
        explicit_episode_return,
        explicit_episode_return_replacement,
        label="explicit episode partition",
    )

    final_return = '''    return train_indices, validation_indices


def pretrain_policy(
'''
    final_return_replacement = '''    return train_indices, validation_indices, purged_indices


def pretrain_policy(
'''
    replace_once(
        INTEGRATION_PATH,
        final_return,
        final_return_replacement,
        label="explicit plain partition",
    )

    unpack_marker = '''    train_indices, validation_indices = _behavior_cloning_indices(
        dataset=dataset,
        config=config,
        split=split,
    )
    train_count = int(train_indices.size)
    validation_count = int(validation_indices.size)
'''
    unpack_replacement = '''    train_indices, validation_indices, excluded_indices = (
        _behavior_cloning_indices(
            dataset=dataset,
            config=config,
            split=split,
        )
    )
    train_count = int(train_indices.size)
    validation_count = int(validation_indices.size)
    excluded_count = int(excluded_indices.size)
    split_digest = behavior_cloning_split_digest(
        sample_count=sample_count,
        training_indices=train_indices,
        validation_indices=validation_indices,
        excluded_indices=excluded_indices,
    )
'''
    replace_once(
        INTEGRATION_PATH,
        unpack_marker,
        unpack_replacement,
        label="resolved BC partition identity",
    )

    result_marker = '''        config=config,
        seed=seed,
        validation_mse=validation_mse,
        validation_sample_count=validation_count,
'''
    result_replacement = '''        config=config,
        seed=seed,
        training_sample_count=train_count,
        excluded_sample_count=excluded_count,
        split_digest=split_digest,
        validation_mse=validation_mse,
        validation_sample_count=validation_count,
'''
    replace_once(
        INTEGRATION_PATH,
        result_marker,
        result_replacement,
        label="BC result partition fields",
    )


def main() -> None:
    patch_learning_contract()
    patch_torch_integration()


if __name__ == "__main__":
    main()
