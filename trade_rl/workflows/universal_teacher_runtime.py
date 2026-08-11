"""Generic teacher-side environment assembly for Universal research."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from trade_rl.rl.universal_instrument_binding import InstrumentDatasetBinding
from trade_rl.rl.universal_single_instrument_env import (
    EpisodeRoutedSingleInstrumentEnv,
    InstrumentContextProvider,
)


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


__all__ = ["build_universal_symbol_teacher_environment"]
