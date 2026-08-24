"""Concrete Binance adapter for immutable Causal Alpha V4 context materialization."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC
from pathlib import Path
from typing import Protocol

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data import load_market_dataset_artifact
from trade_rl.data.v4_context import (
    CausalBetaConfig,
    V4CrossMarketInputs,
    V4GlobalMarketInputs,
    V4TargetContext,
    build_causal_beta_series,
    build_cross_market_context,
    build_global_market_context,
)
from trade_rl.integrations.binance import BinanceMarket
from trade_rl.integrations.binance_cache import inspect_binance_vision_urls
from trade_rl.integrations.binance_v4_cached_sources import (
    build_cached_v4_cross_market_inputs,
)
from trade_rl.integrations.binance_v4_context_capability import (
    BinanceV4ProfileCapability,
    inspect_binance_v4_derivative_capability,
)
from trade_rl.integrations.binance_v4_source_plan import (
    BinanceV4SymbolSourcePlan,
    V4ReferenceDecisionClock,
    build_v4_reference_decision_clock,
    plan_binance_v4_symbol_sources,
)
from trade_rl.integrations.frozen_binance_metadata import (
    FrozenBinanceExchangeInfoTransport,
)
from trade_rl.integrations.postgres_universal_source import MAINTAINED_SYMBOLS
from trade_rl.workflows.binance_metadata_modes import resolve_frozen_snapshot
from trade_rl.workflows.universal_causal_alpha_v4_manifest import (
    CausalAlphaV4ContextManifest,
)
from trade_rl.workflows.universal_runtime_manifest import (
    UniversalRuntimeManifest,
    load_universal_runtime_manifest,
)

CapabilityResolver = Callable[[str], object]
ContextBuilder = Callable[[str, str], V4TargetContext]


class MaterializeGeneration(Protocol):
    def __call__(
        self,
        *,
        base_runtime: object,
        output_root: Path,
        requested_profile: str,
        capability_resolver: CapabilityResolver,
        context_builder: ContextBuilder,
    ) -> CausalAlphaV4ContextManifest: ...


def _ordered_symbols(manifest: UniversalRuntimeManifest) -> tuple[str, ...]:
    symbols = (
        *manifest.train_symbols,
        *manifest.validation_symbols,
        *manifest.test_symbols,
    )
    if set(symbols) != set(MAINTAINED_SYMBOLS):
        raise ValueError("V4 materialization runtime symbol identity mismatch")
    return symbols


def _validate_frozen_metadata(
    *,
    manifest: UniversalRuntimeManifest,
    symbols: tuple[str, ...],
    frozen_metadata_root: Path,
) -> None:
    resolution = resolve_frozen_snapshot(
        transport=FrozenBinanceExchangeInfoTransport(frozen_metadata_root),
        market=BinanceMarket.USDS_M,
        symbols=symbols,
        start_time=manifest.research_start,
        end_time=manifest.research_end,
    )
    if resolution.evidence_digest != manifest.metadata_evidence_digest:
        raise ValueError("V4 materialization frozen metadata identity mismatch")


def _reference_clock(
    *,
    manifest: UniversalRuntimeManifest,
    runtime_manifest_path: Path,
) -> V4ReferenceDecisionClock:
    expected_datasets = dict(manifest.dataset_digests)
    expected_btc_digest = expected_datasets.get("BTCUSDT")
    if expected_btc_digest is None:
        raise ValueError(
            "V4 materialization requires BTCUSDT in train dataset identity"
        )
    dataset_root = runtime_manifest_path.parent / manifest.dataset_artifact_relpath
    dataset = load_market_dataset_artifact(dataset_root / "BTCUSDT")
    if getattr(dataset, "dataset_id", None) != expected_btc_digest:
        raise ValueError("V4 reference BTCUSDT dataset identity mismatch")
    return build_v4_reference_decision_clock(dataset)


def _explicit_core_capability(
    *,
    manifest: UniversalRuntimeManifest,
    symbols: tuple[str, ...],
    reference_clock_digest: str,
) -> BinanceV4ProfileCapability:
    source_digest = content_digest(
        {
            "base_runtime_manifest_digest": manifest.manifest_digest,
            "profile_name": "cross_market_core_v1",
            "reference_clock_digest": reference_clock_digest,
            "requested_profile": "core",
            "schema_version": "binance_v4_explicit_core_capability_v1",
            "symbols": symbols,
        }
    )
    return BinanceV4ProfileCapability(
        symbols=symbols,
        start_time=manifest.research_start.astimezone(UTC),
        end_time=manifest.research_end.astimezone(UTC),
        required_archive_count=0,
        cached_archive_count=0,
        missing_archive_count=0,
        invalid_archive_count=0,
        derivative_metrics_complete=False,
        profile_name="cross_market_core_v1",
        source_digest=source_digest,
    )


def _resolve_capability(
    *,
    requested_profile: str,
    manifest: UniversalRuntimeManifest,
    symbols: tuple[str, ...],
    reference_clock_digest: str,
    market_data_cache_root: Path,
) -> BinanceV4ProfileCapability:
    if requested_profile == "core":
        return _explicit_core_capability(
            manifest=manifest,
            symbols=symbols,
            reference_clock_digest=reference_clock_digest,
        )
    if requested_profile != "derivatives-auto":
        raise ValueError("V4 materialization requested profile is unsupported")
    return inspect_binance_v4_derivative_capability(
        symbols=symbols,
        start_time=manifest.research_start,
        end_time=manifest.research_end,
        cache_root=market_data_cache_root,
    )


def _source_plans(
    *,
    symbols: tuple[str, ...],
    decision_timestamps: object,
    include_metrics: bool,
) -> dict[str, BinanceV4SymbolSourcePlan]:
    return {
        symbol: plan_binance_v4_symbol_sources(
            symbol=symbol,
            decision_timestamps=decision_timestamps,
            include_metrics=include_metrics,
        )
        for symbol in symbols
    }


def _required_urls(
    plans: Mapping[str, BinanceV4SymbolSourcePlan],
    *,
    symbols: tuple[str, ...],
) -> tuple[str, ...]:
    urls: list[str] = []
    for symbol in symbols:
        plan = plans[symbol]
        urls.extend(plan.spot_kline_urls)
        urls.extend(plan.perp_kline_urls)
        urls.extend(plan.mark_price_kline_urls)
        urls.extend(plan.funding_urls)
        urls.extend(plan.metrics_urls)
    result = tuple(dict.fromkeys(urls))
    if not result:
        raise ValueError("V4 materialization source plan is empty")
    return result


def _require_cached_sources(
    *,
    plans: Mapping[str, BinanceV4SymbolSourcePlan],
    symbols: tuple[str, ...],
    market_data_cache_root: Path,
) -> None:
    urls = _required_urls(plans, symbols=symbols)
    report = inspect_binance_vision_urls(urls, cache_root=market_data_cache_root)
    if not report.complete or report.cached_count != len(urls):
        raise FileNotFoundError(
            "V4 materialization Binance cache is incomplete: "
            f"missing={len(report.missing_urls)} "
            f"empty={len(report.empty_urls)} invalid={len(report.invalid_urls)}"
        )


def _load_inputs(
    *,
    symbols: tuple[str, ...],
    plans: Mapping[str, BinanceV4SymbolSourcePlan],
    decision_indices: object,
    decision_timestamps: object,
    market_data_cache_root: Path,
) -> dict[str, V4CrossMarketInputs]:
    return {
        symbol: build_cached_v4_cross_market_inputs(
            decision_indices=decision_indices,
            decision_timestamps=decision_timestamps,
            cache_root=market_data_cache_root,
            spot_kline_urls=plans[symbol].spot_kline_urls,
            perp_kline_urls=plans[symbol].perp_kline_urls,
            mark_price_kline_urls=plans[symbol].mark_price_kline_urls,
            funding_urls=plans[symbol].funding_urls,
            metrics_urls=plans[symbol].metrics_urls,
            expected_symbol=symbol,
        )
        for symbol in symbols
    }


def _build_contexts(
    *,
    symbols: tuple[str, ...],
    inputs: Mapping[str, V4CrossMarketInputs],
    capability: BinanceV4ProfileCapability,
) -> dict[str, V4TargetContext]:
    if "BTCUSDT" not in inputs or "ETHUSDT" not in inputs:
        raise ValueError("V4 materialization requires BTCUSDT and ETHUSDT inputs")
    include_derivatives = capability.derivative_metrics_complete
    btc = inputs["BTCUSDT"]
    eth = inputs["ETHUSDT"]
    global_inputs = V4GlobalMarketInputs(
        btc=btc,
        eth=eth,
        source_digest=content_digest(
            {
                "btc_source_digest": btc.source_digest,
                "capability_digest": capability.source_digest,
                "eth_source_digest": eth.source_digest,
                "schema_version": "causal_alpha_v4_global_inputs_materialization_v1",
            }
        ),
    )
    global_market = build_global_market_context(
        global_inputs,
        include_derivatives=include_derivatives,
    )
    beta_config = CausalBetaConfig()
    contexts: dict[str, V4TargetContext] = {}
    for symbol in symbols:
        source = inputs[symbol]
        beta = build_causal_beta_series(
            symbol=symbol,
            decision_indices=source.decision_indices,
            target_close=source.perp_close,
            btc_close=btc.perp_close,
            target_row_available=source.perp_row_available,
            btc_row_available=btc.perp_row_available,
            bars_per_4h=16,
            config=beta_config,
            target_source_digest=source.source_digest,
            btc_source_digest=btc.source_digest,
        )
        contexts[symbol] = V4TargetContext(
            symbol=symbol,
            local=build_cross_market_context(
                source,
                include_derivatives=include_derivatives,
            ),
            global_market=global_market,
            beta=beta.beta,
            beta_available=beta.available,
            beta_source_digest=beta.source_digest,
            profile_name=capability.profile_name,
        )
    return contexts


def materialize_binance_causal_alpha_v4_context(
    *,
    runtime_manifest_path: Path,
    frozen_metadata_root: Path,
    market_data_cache_root: Path,
    output_root: Path,
    requested_profile: str,
    materialize_generation: MaterializeGeneration,
) -> CausalAlphaV4ContextManifest:
    """Materialize V4 context from one artifact-bound Universal runtime and cache."""

    if not callable(materialize_generation):
        raise TypeError("V4 materialize_generation must be callable")
    manifest_path = Path(runtime_manifest_path)
    metadata_root = Path(frozen_metadata_root)
    cache_root = Path(market_data_cache_root)
    base_runtime = load_universal_runtime_manifest(manifest_path)
    symbols = _ordered_symbols(base_runtime)
    _validate_frozen_metadata(
        manifest=base_runtime,
        symbols=symbols,
        frozen_metadata_root=metadata_root,
    )
    reference_clock = _reference_clock(
        manifest=base_runtime,
        runtime_manifest_path=manifest_path,
    )
    capability = _resolve_capability(
        requested_profile=requested_profile,
        manifest=base_runtime,
        symbols=symbols,
        reference_clock_digest=reference_clock.source_digest,
        market_data_cache_root=cache_root,
    )
    plans = _source_plans(
        symbols=symbols,
        decision_timestamps=reference_clock.decision_timestamps,
        include_metrics=capability.derivative_metrics_complete,
    )
    _require_cached_sources(
        plans=plans,
        symbols=symbols,
        market_data_cache_root=cache_root,
    )
    inputs = _load_inputs(
        symbols=symbols,
        plans=plans,
        decision_indices=reference_clock.decision_indices,
        decision_timestamps=reference_clock.decision_timestamps,
        market_data_cache_root=cache_root,
    )
    contexts = _build_contexts(
        symbols=symbols,
        inputs=inputs,
        capability=capability,
    )

    def capability_resolver(profile: str) -> object:
        if profile != requested_profile:
            raise ValueError("V4 materialization capability request drifted")
        return capability

    def context_builder(symbol: str, profile_name: str) -> V4TargetContext:
        if profile_name != capability.profile_name:
            raise ValueError("V4 materialization context profile drifted")
        try:
            return contexts[symbol]
        except KeyError as error:
            raise ValueError("V4 materialization context symbol is unknown") from error

    result = materialize_generation(
        base_runtime=base_runtime,
        output_root=Path(output_root),
        requested_profile=requested_profile,
        capability_resolver=capability_resolver,
        context_builder=context_builder,
    )
    if not isinstance(result, CausalAlphaV4ContextManifest):
        raise TypeError("V4 materialize_generation returned an invalid manifest")
    return result


__all__ = ["materialize_binance_causal_alpha_v4_context"]
