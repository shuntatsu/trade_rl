"""Generic teacher-side environment assembly for Universal research."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

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
        raise TypeError("Universal teacher concrete environment factory must be callable")
    if not callable(instrument_context_provider):
        raise TypeError("Universal teacher instrument context provider must be callable")
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
    for index, (symbol, binding) in enumerate(zip(symbols, binding_values, strict=True)):
        batch = batches[symbol]
        if not isinstance(batch, EpisodeOracleBatch):
            raise TypeError("Universal teacher batch must be an EpisodeOracleBatch")
        if batch.dataset_id != binding.source_dataset_id:
            raise ValueError("Universal teacher batch dataset identity mismatch")
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

    return combine_symbol_teachers(
        symbol_teachers,
        train_symbols=symbols,
        normalizer_digest=normalizer_digest,
        feature_schema_digest=feature_schema_digest,
    )


__all__ = [
    "build_universal_pretraining_bundle_from_batches",
    "build_universal_symbol_teacher_environment",
]
