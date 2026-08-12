"""Symbol-balanced behavior-cloning adapter for universal single-instrument training."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from trade_rl.artifacts.atomic_write import atomic_write_bytes
from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.integrations.behavior_cloning import pretrain_policy
from trade_rl.learning.behavior_cloning import (
    BehaviorCloningConfig,
    BehaviorCloningResult,
    ObservationBatchProvider,
)
from trade_rl.learning.episode_behavior_cloning import BehaviorCloningSplit
from trade_rl.learning.hierarchical_teacher_labels import HierarchicalTeacherLabels
from trade_rl.learning.teacher_artifact import SupervisedPolicyDataset
from trade_rl.learning.universal_bc import SymbolBalancedBatchSampler


def _validated_symbol_sample_indices(
    symbol_sample_indices: Mapping[str, Sequence[int]],
    *,
    train_symbols: Sequence[str],
    split: BehaviorCloningSplit,
) -> dict[str, tuple[int, ...]]:
    symbols = tuple(train_symbols)
    if (
        not symbols
        or len(set(symbols)) != len(symbols)
        or any(not symbol for symbol in symbols)
    ):
        raise ValueError("Universal BC train_symbols must be non-empty and unique")
    if set(symbol_sample_indices) != set(symbols):
        raise ValueError(
            "Universal BC symbol sample scope must exactly match train_symbols"
        )

    expected_train = {int(value) for value in split.train_indices}
    resolved: dict[str, tuple[int, ...]] = {}
    observed: set[int] = set()
    for symbol in symbols:
        raw = tuple(symbol_sample_indices[symbol])
        if not raw:
            raise ValueError("every Universal BC train symbol requires samples")
        indices: list[int] = []
        for value in raw:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    "Universal BC sample indices must be non-negative integers"
                )
            indices.append(value)
        values = tuple(indices)
        if len(set(values)) != len(values):
            raise ValueError(
                "Universal BC sample indices must be unique within each symbol"
            )
        if not set(values) <= expected_train:
            raise ValueError(
                "Universal BC symbol samples must remain inside the BC train scope"
            )
        overlap = observed.intersection(values)
        if overlap:
            raise ValueError("Universal BC samples cannot belong to multiple symbols")
        observed.update(values)
        resolved[symbol] = values
    if observed != expected_train:
        raise ValueError(
            "Universal BC symbol samples must close exactly over the BC train scope"
        )
    return resolved


def pretrain_universal_policy(
    policy: Any,
    dataset: SupervisedPolicyDataset,
    *,
    symbol_sample_indices: Mapping[str, Sequence[int]],
    train_symbols: Sequence[str],
    config: BehaviorCloningConfig,
    split: BehaviorCloningSplit,
    seed: int,
    observation_provider: ObservationBatchProvider | None,
    output_root: Path,
    hierarchical_labels: HierarchicalTeacherLabels | None = None,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> BehaviorCloningResult:
    """Run shared BC with equal symbol contribution in every training mini-batch."""

    resolved = _validated_symbol_sample_indices(
        symbol_sample_indices,
        train_symbols=train_symbols,
        split=split,
    )
    sampler = SymbolBalancedBatchSampler(sample_indices=resolved, seed=seed)
    expected_train = np.asarray(split.train_indices, dtype=np.int64)

    def training_batch_provider(
        epoch: int,
        train_indices: np.ndarray,
        batch_size: int,
    ) -> tuple[np.ndarray, ...]:
        supplied = np.asarray(train_indices, dtype=np.int64)
        if supplied.ndim != 1 or not np.array_equal(
            np.sort(supplied), np.sort(expected_train)
        ):
            raise ValueError(
                "Universal BC batch provider received a different train scope"
            )
        return tuple(
            np.asarray(batch, dtype=np.int64)
            for batch in sampler.epoch_batches(batch_size=batch_size, epoch=epoch)
        )

    result = pretrain_policy(
        policy,
        dataset,
        config=config,
        split=split,
        seed=seed,
        observation_provider=observation_provider,
        training_batch_provider=training_batch_provider,
        hierarchical_labels=hierarchical_labels,
        progress_callback=progress_callback,
    )
    if isinstance(result, BehaviorCloningResult):
        payload: dict[str, object] = {
            "schema_version": "universal_behavior_cloning_run_v1",
            "behavior_cloning_digest": result.digest,
            "train_symbols": tuple(train_symbols),
            "train_sample_count": len(expected_train),
            "symbol_sample_counts": {
                symbol: len(resolved[symbol]) for symbol in tuple(train_symbols)
            },
            "sampling_contract": "equal_symbol_per_minibatch_cycle_shorter_symbols_v1",
            "batch_size": config.batch_size,
            "seed": seed,
            "hierarchical_label_digest": (
                None
                if hierarchical_labels is None
                else hierarchical_labels.label_config_digest
            ),
        }
        artifact_digest = content_digest(payload)
        output_root.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(
            output_root / "universal-behavior-cloning.json",
            canonical_json_bytes({**payload, "artifact_digest": artifact_digest}),
        )
    return result


__all__ = ["pretrain_universal_policy"]
