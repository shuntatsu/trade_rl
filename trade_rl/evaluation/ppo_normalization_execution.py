"""Execution primitives for the sealed PPO normalization replication."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.directional_contract import DIRECTIONAL_BASE_EXECUTION_COST
from trade_rl.strategies.interface import SingleSymbolStrategy
from trade_rl.strategies.rl.ppo import PPOIntentStrategy, fit_ppo_strategy


class _ReplicationConfig(Protocol):
    feature_indices: tuple[int, ...]
    fit_symbol_indices: tuple[int, ...]
    fit_cutoff: str


@dataclass(frozen=True, slots=True)
class ReplicationArmSpec:
    """One immutable fresh-fit slot in the paired replication."""

    slot: str
    protocol_arm: str
    seed: int
    normalize_features: bool


def replication_arm_specs() -> tuple[ReplicationArmSpec, ...]:
    """Return the exact ten-slot roster in deterministic order."""

    controls = tuple(
        ReplicationArmSpec(
            slot=f"control_raw_seed{seed}",
            protocol_arm="control_raw",
            seed=seed,
            normalize_features=False,
        )
        for seed in range(5)
    )
    candidates = tuple(
        ReplicationArmSpec(
            slot=f"candidate_normalized_seed{seed}",
            protocol_arm="candidate_normalized",
            seed=seed,
            normalize_features=True,
        )
        for seed in range(5)
    )
    return controls + candidates


def _fit_stop_index(dataset: MarketDataset, config: _ReplicationConfig) -> int:
    fit_cutoff = np.datetime64(config.fit_cutoff)
    stop_index = int(np.searchsorted(dataset.timestamps, fit_cutoff)) - 1
    if (
        stop_index <= 0
        or stop_index >= dataset.n_bars
        or dataset.timestamps[stop_index] >= fit_cutoff
    ):
        raise ValueError("fit cutoff does not define a valid pre-development window")
    return stop_index


def fit_replication_strategy(
    dataset: MarketDataset,
    config: _ReplicationConfig,
    spec: ReplicationArmSpec,
) -> PPOIntentStrategy:
    """Fit one fresh control/candidate policy under the sealed common contract."""

    if spec not in replication_arm_specs():
        raise ValueError("replication arm is outside the sealed ten-slot roster")
    return fit_ppo_strategy(
        dataset,
        feature_indices=tuple(config.feature_indices),
        fit_symbol_indices=tuple(config.fit_symbol_indices),
        start_index=0,
        stop_index=_fit_stop_index(dataset, config),
        gross_budget=0.1,
        total_timesteps=262_144,
        seed=spec.seed,
        initial_capital=10_000.0,
        execution_cost=DIRECTIONAL_BASE_EXECUTION_COST,
        training_layout="sequential",
        risk_config=None,
        normalize_features=spec.normalize_features,
        settle_terminal_position=True,
    )


def replication_strategy_factory(
    frozen: PPOIntentStrategy,
) -> Callable[[], SingleSymbolStrategy]:
    """Create fresh mutable wrappers while sharing frozen policy/preprocessing."""

    feature_indices = tuple(frozen.feature_indices)
    policy = frozen.policy
    normalizer = frozen.feature_normalizer

    def factory() -> SingleSymbolStrategy:
        return PPOIntentStrategy(
            policy,
            feature_indices=feature_indices,
            feature_normalizer=normalizer,
        )

    return factory


__all__ = [
    "ReplicationArmSpec",
    "fit_replication_strategy",
    "replication_arm_specs",
    "replication_strategy_factory",
]
