"""Generic teacher-side environment assembly for Universal research."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any

from trade_rl.integrations.sb3_runtime import (
    build_episode_oracle_batch_for_environment,
    oracle_teacher_config_for_environment,
)
from trade_rl.integrations.universal_pretraining import (
    UniversalPretrainingBundle,
    combine_symbol_teachers,
)
from trade_rl.learning.episode_behavior_cloning import behavior_cloning_split
from trade_rl.learning.episode_oracle_teacher import EpisodeOracleBatch
from trade_rl.rl.universal_instrument_binding import InstrumentDatasetBinding
from trade_rl.rl.universal_single_instrument_env import (
    EpisodeRoutedSingleInstrumentEnv,
    InstrumentContextProvider,
)
from trade_rl.workflows.universal_training import collect_universal_episode_teacher

DEFAULT_UNIVERSAL_ORACLE_MAX_EPISODES_PER_SYMBOL = 1
DEFAULT_UNIVERSAL_TEACHER_SAMPLE_STRIDE = 16


def _oracle_max_episodes_per_symbol() -> int:
    name = "TRADE_RL_UNIVERSAL_ORACLE_MAX_EPISODES_PER_SYMBOL"
    raw = os.environ.get(
        name, str(DEFAULT_UNIVERSAL_ORACLE_MAX_EPISODES_PER_SYMBOL)
    ).strip()
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _teacher_sample_stride() -> int:
    name = "TRADE_RL_UNIVERSAL_TEACHER_SAMPLE_STRIDE"
    raw = os.environ.get(name, str(DEFAULT_UNIVERSAL_TEACHER_SAMPLE_STRIDE)).strip()
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def build_universal_symbol_teacher_environment(
    *,
    symbol: str,
    binding: InstrumentDatasetBinding,
    concrete_environment_factory: Callable[[InstrumentDatasetBinding], Any],
    instrument_context_provider: InstrumentContextProvider,
    partition_digest: str,
    training_contract_digest: str,
    run_seed: int,
) -> EpisodeRoutedSingleInstrumentEnv:
    """Expose one concrete Oracle market through the ticker-free policy surface."""

    if not isinstance(symbol, str) or not symbol:
        raise ValueError("Universal teacher symbol must be non-empty")
    if not isinstance(binding, InstrumentDatasetBinding):
        raise TypeError("Universal teacher binding must be an InstrumentDatasetBinding")
    if binding.concrete_symbol != symbol or binding.split != "train":
        raise ValueError("Universal teacher binding must match one train symbol")
    if not callable(concrete_environment_factory):
        raise TypeError(
            "Universal teacher concrete environment factory must be callable"
        )
    if not callable(instrument_context_provider):
        raise TypeError(
            "Universal teacher instrument context provider must be callable"
        )
    return EpisodeRoutedSingleInstrumentEnv(
        train_symbols=(symbol,),
        partition_digest=partition_digest,
        bindings=(binding,),
        environment_factory=concrete_environment_factory,
        run_seed=run_seed,
        environment_index=0,
        instrument_context_provider=instrument_context_provider,
        training_contract_digest=training_contract_digest,
        max_cached_environments=1,
    )


def build_universal_oracle_batches(
    *,
    train_symbols: Sequence[str],
    bindings: Sequence[InstrumentDatasetBinding],
    concrete_environment_factory: Callable[[InstrumentDatasetBinding], Any],
    fold_train_range: tuple[int, int],
    behavior_cloning_seed: int,
    n_envs: int,
) -> dict[str, EpisodeOracleBatch]:
    """Solve paired Oracle episodes for every train symbol inside one fold."""

    symbols = tuple(train_symbols)
    binding_values = tuple(bindings)
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("Universal Oracle train_symbols must be non-empty and unique")
    if tuple(binding.concrete_symbol for binding in binding_values) != symbols:
        raise ValueError("Universal Oracle bindings must follow train_symbols exactly")
    if any(binding.split != "train" for binding in binding_values):
        raise ValueError("Universal Oracle accepts train bindings only")
    if not callable(concrete_environment_factory):
        raise TypeError(
            "Universal Oracle concrete environment factory must be callable"
        )
    start, stop = fold_train_range
    if (
        isinstance(start, bool)
        or isinstance(stop, bool)
        or not isinstance(start, int)
        or not isinstance(stop, int)
        or start < 0
        or stop <= start
    ):
        raise ValueError("Universal Oracle fold_train_range is invalid")
    if (
        isinstance(behavior_cloning_seed, bool)
        or not isinstance(behavior_cloning_seed, int)
        or not 0 <= behavior_cloning_seed <= 0xFFFFFFFF
    ):
        raise ValueError("Universal Oracle behavior_cloning_seed is invalid")
    if isinstance(n_envs, bool) or not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError("Universal Oracle n_envs must be a positive integer")

    batches: dict[str, EpisodeOracleBatch] = {}
    max_episodes = _oracle_max_episodes_per_symbol()
    for symbol, binding in zip(symbols, binding_values, strict=True):
        environment = concrete_environment_factory(binding)
        try:
            minimum_start = getattr(environment, "minimum_start_index", None)
            dataset_bars = getattr(getattr(environment, "dataset", None), "n_bars", None)
            if (
                isinstance(minimum_start, bool)
                or not isinstance(minimum_start, int)
                or isinstance(dataset_bars, bool)
                or not isinstance(dataset_bars, int)
            ):
                raise ValueError("Universal Oracle environment trainable range is unavailable")
            effective_range = (max(start, minimum_start), min(stop, dataset_bars))
            if effective_range[1] <= effective_range[0]:
                raise ValueError("Universal Oracle environment trainable range is empty")
            batch = build_episode_oracle_batch_for_environment(
                environment,
                train_range=effective_range,
                seed=behavior_cloning_seed,
                n_envs=n_envs,
                max_episodes=max_episodes,
            )
        finally:
            environment.close()
        if batch.dataset_id != binding.source_dataset_id:
            raise ValueError("Universal Oracle batch dataset identity mismatch")
        batches[symbol] = batch
    return batches


def build_universal_pretraining_bundle_from_batches(
    *,
    train_symbols: Sequence[str],
    bindings: Sequence[InstrumentDatasetBinding],
    batches: Mapping[str, EpisodeOracleBatch],
    concrete_environment_factory: Callable[[InstrumentDatasetBinding], Any],
    instrument_context_provider: InstrumentContextProvider,
    partition_digest: str,
    training_contract_digest: str,
    run_seed: int,
    gamma: float,
    validation_fraction: float,
    normalizer_digest: str,
    feature_schema_digest: str,
) -> UniversalPretrainingBundle:
    """Collect, split, and combine train-only Oracle batches on the generic surface."""

    symbols = tuple(train_symbols)
    binding_values = tuple(bindings)
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("Universal teacher train_symbols must be non-empty and unique")
    if tuple(binding.concrete_symbol for binding in binding_values) != symbols:
        raise ValueError("Universal teacher bindings must follow train_symbols exactly")
    if any(binding.split != "train" for binding in binding_values):
        raise ValueError("Universal teacher accepts train bindings only")
    if set(batches) != set(symbols):
        raise ValueError("Universal teacher batches must exactly match train_symbols")
    if (
        isinstance(run_seed, bool)
        or not isinstance(run_seed, int)
        or run_seed < 0
        or run_seed + len(symbols) - 1 > 0xFFFFFFFF
    ):
        raise ValueError("Universal teacher run_seed range is invalid")

    symbol_teachers: dict[str, tuple[Any, Any, Any]] = {}
    sample_stride = _teacher_sample_stride()
    for index, (symbol, binding) in enumerate(
        zip(symbols, binding_values, strict=True)
    ):
        batch = batches[symbol]
        if not isinstance(batch, EpisodeOracleBatch):
            raise TypeError("Universal teacher batch must be an EpisodeOracleBatch")
        if batch.dataset_id != binding.source_dataset_id:
            raise ValueError("Universal teacher batch dataset identity mismatch")
        concrete_environment = concrete_environment_factory(binding)
        close_concrete = getattr(concrete_environment, "close", None)
        if not callable(close_concrete):
            raise TypeError("Universal teacher concrete environment must be closable")
        try:
            candidate_teacher_config = oracle_teacher_config_for_environment(
                concrete_environment
            )
        finally:
            close_concrete()
        if candidate_teacher_config.digest != batch.teacher_config_digest:
            raise ValueError("Universal Oracle teacher config identity mismatch")
        environment = build_universal_symbol_teacher_environment(
            symbol=symbol,
            binding=binding,
            concrete_environment_factory=concrete_environment_factory,
            instrument_context_provider=instrument_context_provider,
            partition_digest=partition_digest,
            training_contract_digest=training_contract_digest,
            run_seed=run_seed + index,
        )
        try:
            collected = collect_universal_episode_teacher(
                environment,
                batch,
                teacher_config_digest=batch.teacher_config_digest,
                gamma=gamma,
                sample_stride=sample_stride,
            )
        finally:
            environment.close()
        split = behavior_cloning_split(
            collected.dataset,
            validation_fraction=validation_fraction,
        )
        symbol_teachers[symbol] = (
            collected.dataset,
            split,
            collected.critic_targets,
        )

    return replace(
        combine_symbol_teachers(
            symbol_teachers,
            train_symbols=symbols,
            normalizer_digest=normalizer_digest,
            feature_schema_digest=feature_schema_digest,
        ),
        episode_batches=dict(batches),
    )


__all__ = [
    "DEFAULT_UNIVERSAL_ORACLE_MAX_EPISODES_PER_SYMBOL",
    "DEFAULT_UNIVERSAL_TEACHER_SAMPLE_STRIDE",
    "build_universal_oracle_batches",
    "build_universal_pretraining_bundle_from_batches",
    "build_universal_symbol_teacher_environment",
]
