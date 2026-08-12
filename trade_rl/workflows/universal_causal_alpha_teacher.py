"""Train-only chronological episode contracts for the Universal causal alpha teacher."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.episode_behavior_cloning import BehaviorCloningSplit
from trade_rl.learning.episode_oracle_bc import resolve_episode_initial_weights
from trade_rl.learning.episode_oracle_teacher import OracleEpisodeContract

_CAUSAL_ALPHA_EPISODE_PARTITION_SCHEMA = "universal_causal_alpha_episode_partition_v1"


@dataclass(frozen=True, slots=True)
class CausalAlphaEpisodePartition:
    """Chronological selection episodes plus one untouched latest holdout."""

    contracts: tuple[OracleEpisodeContract, ...]
    selection_contracts: tuple[OracleEpisodeContract, ...]
    holdout_contract: OracleEpisodeContract
    train_start: int
    train_stop: int
    digest: str = ""

    def __post_init__(self) -> None:
        contracts = tuple(self.contracts)
        selection = tuple(self.selection_contracts)
        if len(contracts) < 2 or selection != contracts[:-1]:
            raise ValueError(
                "causal alpha partition requires selection episodes and one holdout"
            )
        if self.holdout_contract != contracts[-1]:
            raise ValueError("causal alpha holdout must be the latest complete episode")
        if self.train_start < 0 or self.train_stop <= self.train_start:
            raise ValueError("causal alpha partition train range is invalid")
        if tuple(contract.episode_index for contract in contracts) != tuple(
            range(len(contracts))
        ):
            raise ValueError("causal alpha episode indices must be chronological")
        dataset_ids = {contract.dataset_id for contract in contracts}
        if len(dataset_ids) != 1:
            raise ValueError("causal alpha episode dataset identity drifted")
        for previous, current in zip(contracts[:-1], contracts[1:], strict=True):
            if previous.start >= current.start or previous.stop > current.start:
                raise ValueError("causal alpha chronological episodes overlap")
        if selection[-1].stop > self.holdout_contract.start:
            raise ValueError("selection episode support crosses the holdout boundary")
        expected = content_digest(
            {
                "contracts": tuple(contract.digest for contract in contracts),
                "holdout_contract": self.holdout_contract.digest,
                "schema_version": _CAUSAL_ALPHA_EPISODE_PARTITION_SCHEMA,
                "train_start": self.train_start,
                "train_stop": self.train_stop,
            }
        )
        if self.digest and self.digest != expected:
            raise ValueError("causal alpha episode partition digest mismatch")
        object.__setattr__(self, "contracts", contracts)
        object.__setattr__(self, "selection_contracts", selection)
        object.__setattr__(self, "digest", expected)


def _train_range(
    environment: Any,
    train_range: tuple[int, int],
) -> tuple[int, int, int]:
    start, stop = train_range
    if (
        isinstance(start, bool)
        or isinstance(stop, bool)
        or not isinstance(start, int)
        or not isinstance(stop, int)
        or start < 0
        or stop <= start
    ):
        raise ValueError("causal alpha train range is invalid")
    dataset = getattr(environment, "dataset", None)
    n_bars = getattr(dataset, "n_bars", None)
    if isinstance(n_bars, bool) or not isinstance(n_bars, int) or n_bars <= 0:
        raise ValueError("causal alpha environment dataset size is unavailable")
    minimum_start = getattr(environment, "minimum_start_index", None)
    if (
        isinstance(minimum_start, bool)
        or not isinstance(minimum_start, int)
        or minimum_start < 0
    ):
        raise ValueError("causal alpha environment minimum start is unavailable")
    effective_start = max(start, minimum_start)
    effective_stop = min(stop, n_bars)
    if effective_stop <= effective_start:
        raise ValueError("causal alpha effective train range is empty")
    return effective_start, effective_stop, n_bars


def build_chronological_episode_partition(
    environment: Any,
    *,
    train_range: tuple[int, int],
) -> CausalAlphaEpisodePartition:
    """Reserve the latest complete episode and use only earlier complete episodes."""

    if getattr(environment, "decision_bars", None) != 1:
        raise ValueError("causal alpha teacher currently requires one bar per decision")
    episode_bars = getattr(environment, "episode_bars", None)
    if (
        isinstance(episode_bars, bool)
        or not isinstance(episode_bars, int)
        or episode_bars <= 0
    ):
        raise ValueError("causal alpha episode horizon must be positive")
    dataset = getattr(environment, "dataset", None)
    dataset_id = getattr(dataset, "dataset_id", None)
    n_symbols = getattr(dataset, "n_symbols", None)
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError("causal alpha dataset identity is unavailable")
    if isinstance(n_symbols, bool) or not isinstance(n_symbols, int) or n_symbols <= 0:
        raise ValueError("causal alpha dataset symbol count is unavailable")
    effective_start, effective_stop, _ = _train_range(environment, train_range)

    stride = episode_bars + 1
    latest_start = effective_stop - stride
    if latest_start < effective_start:
        raise ValueError("causal alpha train range contains no complete holdout episode")
    starts: list[int] = []
    cursor = latest_start
    while cursor >= effective_start:
        starts.append(cursor)
        cursor -= stride
    starts.reverse()
    if len(starts) < 2:
        raise ValueError(
            "causal alpha train range requires at least one selection episode "
            "before the holdout"
        )

    config = getattr(environment, "config", None)
    modes = tuple(getattr(config, "initial_state_modes", ()))
    if not modes or any(mode not in {"cash", "baseline"} for mode in modes):
        raise ValueError(
            "causal alpha episodes support only declared cash and baseline reset modes"
        )

    contracts: list[OracleEpisodeContract] = []
    for episode_index, contract_start in enumerate(starts):
        mode = modes[episode_index % len(modes)]
        initial_weights = resolve_episode_initial_weights(
            environment,
            mode,
            contract_start,
        )
        if initial_weights.shape != (n_symbols,):
            raise ValueError("causal alpha initial weights do not match dataset symbols")
        contracts.append(
            OracleEpisodeContract(
                dataset_id=dataset_id,
                episode_index=episode_index,
                start=contract_start,
                stop=contract_start + stride,
                initial_state_mode=mode,
                initial_weights=initial_weights,
            )
        )
    resolved = tuple(contracts)
    return CausalAlphaEpisodePartition(
        contracts=resolved,
        selection_contracts=resolved[:-1],
        holdout_contract=resolved[-1],
        train_start=effective_start,
        train_stop=effective_stop,
    )


def _sample_int_vector(dataset: Any, field: str, sample_count: int) -> np.ndarray:
    raw = np.asarray(getattr(dataset, field, None))
    if (
        raw.ndim != 1
        or len(raw) != sample_count
        or not np.issubdtype(raw.dtype, np.integer)
    ):
        raise ValueError(f"{field} must be a sample-aligned integer vector")
    values = np.asarray(raw, dtype=np.int64)
    if np.any(values < 0):
        raise ValueError(f"{field} must be non-negative")
    return values


def latest_complete_episode_split(
    dataset: Any,
    *,
    holdout_episode_id: int,
) -> BehaviorCloningSplit:
    """Return an explicit split whose validation set is exactly one latest episode."""

    sample_count = int(getattr(dataset, "sample_count", 0))
    if sample_count <= 0:
        raise ValueError("causal alpha teacher dataset must contain samples")
    if (
        isinstance(holdout_episode_id, bool)
        or not isinstance(holdout_episode_id, int)
        or holdout_episode_id < 0
    ):
        raise ValueError("holdout_episode_id must be non-negative")
    episode_ids = _sample_int_vector(dataset, "episode_ids", sample_count)
    decision_indices = _sample_int_vector(dataset, "decision_indices", sample_count)
    holdout_mask = episode_ids == holdout_episode_id
    if not np.any(holdout_mask):
        raise ValueError("causal alpha holdout episode is absent from the dataset")
    holdout_start = int(np.min(decision_indices[holdout_mask]))

    train_episode_ids: list[int] = []
    purged_episode_ids: list[int] = []
    for raw_episode_id in np.unique(episode_ids):
        episode_id = int(raw_episode_id)
        if episode_id == holdout_episode_id:
            continue
        mask = episode_ids == episode_id
        episode_start = int(np.min(decision_indices[mask]))
        support_stop = int(np.max(decision_indices[mask])) + 2
        if episode_start >= holdout_start:
            raise ValueError("causal alpha holdout episode must be latest")
        if support_stop <= holdout_start:
            train_episode_ids.append(episode_id)
        else:
            purged_episode_ids.append(episode_id)
    if not train_episode_ids:
        raise ValueError("causal alpha holdout leaves no BC training episodes")

    train_ids = np.asarray(sorted(train_episode_ids), dtype=np.int64)
    purged_ids = np.asarray(sorted(purged_episode_ids), dtype=np.int64)
    validation_ids = np.asarray([holdout_episode_id], dtype=np.int64)
    return BehaviorCloningSplit(
        train_indices=np.flatnonzero(np.isin(episode_ids, train_ids)),
        validation_indices=np.flatnonzero(holdout_mask),
        train_episode_ids=train_ids,
        validation_episode_ids=validation_ids,
        purged_indices=np.flatnonzero(np.isin(episode_ids, purged_ids)),
        purged_episode_ids=purged_ids,
    )


def validate_universal_causal_alpha_partitions(
    *,
    train_symbols: tuple[str, ...],
    partitions: Mapping[str, CausalAlphaEpisodePartition],
) -> dict[str, CausalAlphaEpisodePartition]:
    """Close the causal teacher episode scope over exactly the train symbols."""

    symbols = tuple(train_symbols)
    if not symbols or len(set(symbols)) != len(symbols) or any(not symbol for symbol in symbols):
        raise ValueError("causal alpha train_symbols must be non-empty and unique")
    if set(partitions) != set(symbols):
        raise ValueError("causal alpha partitions must exactly match train_symbols")
    ordered: dict[str, CausalAlphaEpisodePartition] = {}
    for symbol in symbols:
        partition = partitions[symbol]
        if not isinstance(partition, CausalAlphaEpisodePartition):
            raise TypeError("causal alpha partition has an invalid type")
        if not partition.selection_contracts:
            raise ValueError("causal alpha partition has no selection episode")
        ordered[symbol] = partition
    return ordered


__all__ = [
    "CausalAlphaEpisodePartition",
    "build_chronological_episode_partition",
    "latest_complete_episode_split",
    "validate_universal_causal_alpha_partitions",
]
