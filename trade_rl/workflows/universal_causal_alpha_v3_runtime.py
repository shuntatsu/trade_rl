"""Verified runtime preparation for the hardened causal alpha V3 workflow."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, TypeVar

import trade_rl
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.risk.portfolio import PortfolioRiskConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.workflows.universal_causal_alpha_contracts import (
    CausalAlphaEpisodePartition,
    CausalAlphaSymbolSamples,
)
from trade_rl.workflows.universal_causal_alpha_fitting import (
    build_causal_alpha_symbol_samples,
    build_chronological_episode_partition,
)
from trade_rl.workflows.universal_causal_alpha_v3_identity import (
    CausalAlphaV3ExecutionIdentity,
)
from trade_rl.workflows.universal_training_runner import UniversalTrainingRuntime

T = TypeVar("T")


def _frozen_mapping(value: Mapping[str, T]) -> Mapping[str, T]:
    return MappingProxyType(dict(value))


def causal_alpha_v3_source_tree_digest() -> str:
    """Bind replay evidence to every Python module in the installed trade_rl tree."""

    package_root = Path(trade_rl.__file__).resolve().parent
    files = tuple(
        (
            path.relative_to(package_root).as_posix(),
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in sorted(package_root.rglob("*.py"))
    )
    if not files:
        raise RuntimeError("V3 source tree is unavailable")
    return content_digest(
        {"files": files, "schema_version": "causal_alpha_v3_source_tree_v1"}
    )


@dataclass(frozen=True, slots=True)
class CausalAlphaV3PreparedResearchData:
    """Immutable train-only inputs plus exact execution semantics identity."""

    train_symbols: tuple[str, ...]
    partitions: Mapping[str, CausalAlphaEpisodePartition]
    samples: Mapping[str, CausalAlphaSymbolSamples]
    environment_factories: Mapping[str, Callable[[], Any]]
    episode_hours: float
    execution_costs: Mapping[str, ExecutionCostConfig]
    signal_delays: Mapping[str, int]
    decision_bars: Mapping[str, int]
    max_position_to_market_notional: float
    catalog_digest: str
    partition_digest: str
    split_manifest_digest: str
    feature_schema_digest: str
    statistics_digest: str
    execution_identity: CausalAlphaV3ExecutionIdentity

    def __post_init__(self) -> None:
        symbols = tuple(self.train_symbols)
        if (
            not symbols
            or len(set(symbols)) != len(symbols)
            or any(not symbol for symbol in symbols)
        ):
            raise ValueError("V3 prepared train_symbols must be unique")
        partitions = dict(self.partitions)
        samples = dict(self.samples)
        factories = dict(self.environment_factories)
        execution_costs = dict(self.execution_costs)
        signal_delays = dict(self.signal_delays)
        decision_bars = dict(self.decision_bars)
        for name, values in (
            ("partitions", partitions),
            ("samples", samples),
            ("environment_factories", factories),
            ("execution_costs", execution_costs),
            ("signal_delays", signal_delays),
            ("decision_bars", decision_bars),
        ):
            if set(values) != set(symbols):
                raise ValueError(f"V3 prepared {name} must match train_symbols")
        for symbol in symbols:
            partition = partitions[symbol]
            sample = samples[symbol]
            if not isinstance(partition, CausalAlphaEpisodePartition):
                raise TypeError("V3 prepared partition type is invalid")
            if not isinstance(sample, CausalAlphaSymbolSamples):
                raise TypeError("V3 prepared sample type is invalid")
            if sample.dataset_id != partition.holdout_contract.dataset_id:
                raise ValueError("V3 prepared sample/partition dataset identity drifted")
            if not callable(factories[symbol]):
                raise TypeError("V3 prepared environment factory must be callable")
            if not isinstance(execution_costs[symbol], ExecutionCostConfig):
                raise TypeError("V3 prepared execution cost config is invalid")
            delay = signal_delays[symbol]
            if isinstance(delay, bool) or not isinstance(delay, int) or delay not in {0, 1}:
                raise ValueError("V3 prepared signal delay must be 0 or 1")
            bars = decision_bars[symbol]
            if isinstance(bars, bool) or not isinstance(bars, int) or bars <= 0:
                raise ValueError("V3 prepared decision_bars must be positive")
        if not math.isfinite(self.episode_hours) or self.episode_hours <= 0.0:
            raise ValueError("V3 prepared episode_hours must be positive")
        if (
            not math.isfinite(self.max_position_to_market_notional)
            or abs(self.max_position_to_market_notional - 0.02) > 1e-12
        ):
            raise ValueError("V3 hard market-notional cap must remain 0.02")
        for name in (
            "catalog_digest",
            "partition_digest",
            "split_manifest_digest",
            "feature_schema_digest",
            "statistics_digest",
        ):
            require_sha256(getattr(self, name), field=f"V3 prepared {name}")
        if not isinstance(self.execution_identity, CausalAlphaV3ExecutionIdentity):
            raise TypeError("V3 prepared execution identity is invalid")
        if self.execution_identity.train_symbols != symbols:
            raise ValueError("V3 prepared execution identity scope drifted")
        object.__setattr__(self, "train_symbols", symbols)
        object.__setattr__(self, "partitions", _frozen_mapping(partitions))
        object.__setattr__(self, "samples", _frozen_mapping(samples))
        object.__setattr__(self, "environment_factories", _frozen_mapping(factories))
        object.__setattr__(self, "execution_costs", _frozen_mapping(execution_costs))
        object.__setattr__(self, "signal_delays", _frozen_mapping(signal_delays))
        object.__setattr__(self, "decision_bars", _frozen_mapping(decision_bars))

    @property
    def generator_code_digest(self) -> str:
        return content_digest(
            {
                "schema_version": "universal_causal_alpha_v3_generator_code_v2",
                "source_tree_digest": self.execution_identity.source_tree_digest,
            }
        )


def prepare_causal_alpha_v3_research_data(
    *, runtime: UniversalTrainingRuntime, fold_train_range: tuple[int, int]
) -> CausalAlphaV3PreparedResearchData:
    """Resolve train artifacts once and close runtime/execution identities."""

    if not isinstance(runtime, UniversalTrainingRuntime):
        raise TypeError("V3 research preparation requires UniversalTrainingRuntime")
    start, stop = fold_train_range
    if (
        isinstance(start, bool)
        or isinstance(stop, bool)
        or not isinstance(start, int)
        or not isinstance(stop, int)
        or not 0 <= start < stop
    ):
        raise ValueError("V3 fold_train_range must be a valid half-open range")
    routed = runtime.routed_environment_factory
    symbols = tuple(runtime.train_symbols)
    bindings = tuple(routed.bindings)
    if tuple(binding.concrete_symbol for binding in bindings) != symbols:
        raise ValueError("V3 runtime bindings drifted from train_symbols")
    concrete = routed.concrete_environment_factory
    provider = routed.instrument_context_provider
    if not callable(concrete) or not callable(provider):
        raise ValueError("V3 concrete factory/context provider is unavailable")

    partitions: dict[str, CausalAlphaEpisodePartition] = {}
    samples: dict[str, CausalAlphaSymbolSamples] = {}
    costs: dict[str, ExecutionCostConfig] = {}
    delays: dict[str, int] = {}
    bars_by_symbol: dict[str, int] = {}
    factories: dict[str, Callable[[], Any]] = {}
    episode_hours: list[float] = []
    market_caps: list[float] = []
    runtime_digests: list[tuple[str, str]] = []

    for symbol, binding in zip(symbols, bindings, strict=True):
        environment = concrete(binding)
        close = getattr(environment, "close", None)
        if not callable(close):
            raise TypeError("V3 concrete environment must be closable")
        try:
            partition = build_chronological_episode_partition(
                environment, train_range=fold_train_range
            )
            sample = build_causal_alpha_symbol_samples(
                environment=environment,
                binding=binding,
                instrument_context_provider=provider,
                train_range=fold_train_range,
                feature_schema_digest=runtime.feature_schema_digest,
            )
            if partition.holdout_contract.dataset_id != sample.dataset_id:
                raise ValueError("V3 partition/sample dataset identity drifted")
            config = getattr(environment, "config", None)
            execution = getattr(config, "execution_cost", None)
            if not isinstance(execution, ExecutionCostConfig):
                raise TypeError("V3 execution cost config is unavailable")
            delay = getattr(config, "signal_delay_decisions", None)
            if isinstance(delay, bool) or not isinstance(delay, int) or delay not in {0, 1}:
                raise ValueError("V3 signal delay is unavailable")
            decision_bars = getattr(environment, "decision_bars", None)
            if (
                isinstance(decision_bars, bool)
                or not isinstance(decision_bars, int)
                or decision_bars <= 0
            ):
                raise ValueError("V3 decision_bars is unavailable")
            hours = getattr(config, "episode_hours", None)
            if isinstance(hours, bool) or not isinstance(hours, int | float):
                raise ValueError("V3 episode_hours is unavailable")
            risk = getattr(getattr(environment, "portfolio_risk", None), "config", None)
            if not isinstance(risk, PortfolioRiskConfig):
                raise TypeError("V3 portfolio risk config is unavailable")
            cap = risk.max_position_to_market_notional
            if cap is None or not math.isfinite(cap) or cap <= 0.0:
                raise ValueError("V3 hard market-notional cap is unavailable")
            dataset_id = getattr(environment.dataset, "dataset_id", None)
            if not isinstance(dataset_id, str):
                raise ValueError(f"V3 dataset identity is unavailable for {symbol}")
            require_sha256(dataset_id, field=f"V3 dataset identity {symbol}")
            runtime_digest = content_digest(
                {
                    "dataset_id": dataset_id,
                    "decision_bars": decision_bars,
                    "episode_hours": float(hours),
                    "execution_cost": asdict(execution),
                    "hard_market_notional_cap": float(cap),
                    "risk_config": asdict(risk),
                    "schema_version": "causal_alpha_v3_symbol_runtime_v1",
                    "signal_delay_decisions": delay,
                    "symbol": symbol,
                }
            )
            partitions[symbol] = partition
            samples[symbol] = sample
            costs[symbol] = execution
            delays[symbol] = delay
            bars_by_symbol[symbol] = decision_bars
            factories[symbol] = partial(concrete, binding)
            episode_hours.append(float(hours))
            market_caps.append(float(cap))
            runtime_digests.append((symbol, runtime_digest))
        finally:
            close()

    if len(set(episode_hours)) != 1:
        raise ValueError("V3 episode_hours differs across train symbols")
    if len(set(market_caps)) != 1 or abs(market_caps[0] - 0.02) > 1e-12:
        raise ValueError("V3 hard market-notional cap must remain exactly 0.02")
    execution_identity = CausalAlphaV3ExecutionIdentity(
        train_symbols=symbols,
        training_contract_digest=runtime.training_contract_digest,
        instrument_context_schema_digest=runtime.instrument_context_schema_digest,
        source_tree_digest=causal_alpha_v3_source_tree_digest(),
        symbol_runtime_digests=tuple(runtime_digests),
    )
    return CausalAlphaV3PreparedResearchData(
        train_symbols=symbols,
        partitions=partitions,
        samples=samples,
        environment_factories=factories,
        episode_hours=episode_hours[0],
        execution_costs=costs,
        signal_delays=delays,
        decision_bars=bars_by_symbol,
        max_position_to_market_notional=market_caps[0],
        catalog_digest=runtime.catalog_digest,
        partition_digest=runtime.partition_digest,
        split_manifest_digest=runtime.split_manifest_digest,
        feature_schema_digest=runtime.feature_schema_digest,
        statistics_digest=runtime.statistics_digest,
        execution_identity=execution_identity,
    )


__all__ = [
    "CausalAlphaV3PreparedResearchData",
    "causal_alpha_v3_source_tree_digest",
    "prepare_causal_alpha_v3_research_data",
]
