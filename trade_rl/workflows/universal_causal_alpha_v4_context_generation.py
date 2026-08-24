"""Immutable Causal Alpha V4 context-generation assembly."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.v4_context import (
    CausalBetaConfig,
    V4CrossMarketInputs,
    V4GlobalMarketInputs,
    V4TargetContext,
    build_causal_beta_series,
    build_cross_market_context,
    build_global_market_context,
)
from trade_rl.data.v4_context_artifact import (
    load_v4_target_context_artifact,
    write_v4_target_context_artifact,
)
from trade_rl.integrations.binance_v4_context_capability import (
    BinanceV4ProfileCapability,
)
from trade_rl.workflows.universal_causal_alpha_v4_manifest import (
    CausalAlphaV4ContextManifest,
    load_causal_alpha_v4_context_manifest,
    validate_causal_alpha_v4_context_manifest_against_base,
    write_causal_alpha_v4_context_manifest,
)


@dataclass(frozen=True, slots=True)
class CausalAlphaV4ContextGenerationResult:
    """Published paths and identity for one immutable V4 context generation."""

    manifest_path: Path
    manifest_digest: str
    context_paths: tuple[tuple[str, Path], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifest_path", Path(self.manifest_path))
        paths = tuple((symbol, Path(path)) for symbol, path in self.context_paths)
        if not paths or len({symbol for symbol, _ in paths}) != len(paths):
            raise ValueError("V4 context generation paths must contain unique symbols")
        if not isinstance(self.manifest_digest, str) or len(self.manifest_digest) != 64:
            raise ValueError("V4 context generation manifest_digest is invalid")
        object.__setattr__(self, "context_paths", paths)


def _base_symbols(base_runtime: Any) -> tuple[str, ...]:
    train = tuple(getattr(base_runtime, "train_symbols", ()))
    validation = tuple(getattr(base_runtime, "validation_symbols", ()))
    test = tuple(getattr(base_runtime, "test_symbols", ()))
    symbols = (*train, *validation, *test)
    if not symbols or any(
        not isinstance(symbol, str) or not symbol for symbol in symbols
    ):
        raise ValueError("V4 base runtime symbols are unavailable")
    if len(set(symbols)) != len(symbols):
        raise ValueError("V4 base runtime symbols overlap")
    return symbols


def _validate_input_scope(
    *,
    symbols: tuple[str, ...],
    inputs: Mapping[str, V4CrossMarketInputs],
    capability: BinanceV4ProfileCapability,
) -> dict[str, V4CrossMarketInputs]:
    if set(inputs) != set(symbols):
        raise ValueError(
            "V4 context input scope must exactly match base runtime symbols"
        )
    if tuple(capability.symbols) != symbols:
        raise ValueError("V4 source capability symbols must match base runtime order")
    ordered: dict[str, V4CrossMarketInputs] = {}
    for symbol in symbols:
        value = inputs[symbol]
        if not isinstance(value, V4CrossMarketInputs):
            raise TypeError("V4 context inputs must be V4CrossMarketInputs")
        ordered[symbol] = value
    if "BTCUSDT" not in ordered or "ETHUSDT" not in ordered:
        raise ValueError("V4 context generation requires BTCUSDT and ETHUSDT anchors")
    include_derivatives = capability.derivative_metrics_complete
    if include_derivatives and any(
        ordered[symbol].open_interest_value is None for symbol in symbols
    ):
        raise ValueError(
            "V4 derivative profile requires complete derivative inputs for every symbol"
        )
    if not include_derivatives and any(
        ordered[symbol].open_interest_value is not None for symbol in symbols
    ):
        # Core generation is deliberately fixed to its authored feature profile even
        # when inert derivative arrays happen to be available upstream.
        pass
    return ordered


def _schema_digest(*, kind: str, feature_names: tuple[str, ...]) -> str:
    return content_digest(
        {
            "feature_names": feature_names,
            "kind": kind,
            "schema_version": "causal_alpha_v4_context_feature_schema_v1",
        }
    )


def _build_expected_contexts(
    *,
    symbols: tuple[str, ...],
    inputs: Mapping[str, V4CrossMarketInputs],
    capability: BinanceV4ProfileCapability,
    beta_config: CausalBetaConfig,
) -> tuple[V4TargetContext, ...]:
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
                "schema_version": "causal_alpha_v4_global_inputs_generation_v1",
            }
        ),
    )
    global_block = build_global_market_context(
        global_inputs,
        include_derivatives=include_derivatives,
    )
    contexts: list[V4TargetContext] = []
    for symbol in symbols:
        source = inputs[symbol]
        local_block = build_cross_market_context(
            source,
            include_derivatives=include_derivatives,
        )
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
        contexts.append(
            V4TargetContext(
                symbol=symbol,
                local=local_block,
                global_market=global_block,
                beta=beta.beta,
                beta_available=beta.available,
                beta_source_digest=beta.source_digest,
                profile_name=capability.profile_name,
            )
        )
    return tuple(contexts)


def _require_published_contexts_exist(
    manifest: CausalAlphaV4ContextManifest,
    *,
    root: Path,
) -> None:
    context_root = root / manifest.context_artifact_relpath
    for symbol, expected_digest in manifest.context_digests:
        path = context_root / symbol
        if not path.is_dir():
            raise FileNotFoundError(
                f"published V4 context manifest references missing context: {symbol}"
            )
        loaded = load_v4_target_context_artifact(path)
        if loaded.digest != expected_digest:
            raise ValueError(f"published V4 context digest mismatch for {symbol}")


def materialize_causal_alpha_v4_context_generation(
    *,
    base_runtime: Any,
    inputs: Mapping[str, V4CrossMarketInputs],
    capability: BinanceV4ProfileCapability,
    output_root: str | Path,
    beta_config: CausalBetaConfig,
) -> CausalAlphaV4ContextGenerationResult:
    """Publish contexts first and the generation manifest only after closure."""

    if not isinstance(capability, BinanceV4ProfileCapability):
        raise TypeError("capability must be BinanceV4ProfileCapability")
    if not isinstance(beta_config, CausalBetaConfig):
        raise TypeError("beta_config must be CausalBetaConfig")
    symbols = _base_symbols(base_runtime)
    ordered_inputs = _validate_input_scope(
        symbols=symbols,
        inputs=inputs,
        capability=capability,
    )
    root = Path(output_root)
    manifest_path = root / "manifest.json"
    existing_manifest: CausalAlphaV4ContextManifest | None = None
    if manifest_path.exists():
        existing_manifest = load_causal_alpha_v4_context_manifest(manifest_path)
        validate_causal_alpha_v4_context_manifest_against_base(
            existing_manifest,
            base_runtime,
        )
        _require_published_contexts_exist(existing_manifest, root=root)

    expected_contexts = _build_expected_contexts(
        symbols=symbols,
        inputs=ordered_inputs,
        capability=capability,
        beta_config=beta_config,
    )
    contexts_root = root / "contexts"
    context_paths: list[tuple[str, Path]] = []
    context_digests: list[tuple[str, str]] = []
    for context in expected_contexts:
        path = contexts_root / context.symbol
        written = write_v4_target_context_artifact(path, context)
        loaded = load_v4_target_context_artifact(written)
        if loaded.digest != context.digest:
            raise ValueError(
                f"published V4 target context changed identity: {context.symbol}"
            )
        context_paths.append((context.symbol, written))
        context_digests.append((context.symbol, context.digest))

    first = expected_contexts[0]
    local_schema_digest = _schema_digest(
        kind="local_cross_market",
        feature_names=first.local.feature_names,
    )
    global_schema_digest = _schema_digest(
        kind="global_market",
        feature_names=first.global_market.feature_names,
    )
    if any(
        context.local.feature_names != first.local.feature_names
        or context.global_market.feature_names != first.global_market.feature_names
        for context in expected_contexts
    ):
        raise ValueError("V4 context feature schema drifted across symbols")
    base_digest = getattr(base_runtime, "manifest_digest", None)
    if not isinstance(base_digest, str):
        raise ValueError("V4 base runtime manifest digest is unavailable")
    manifest = CausalAlphaV4ContextManifest(
        base_runtime_manifest_digest=base_digest,
        profile_name=capability.profile_name,
        context_artifact_relpath=Path("contexts"),
        context_digests=tuple(context_digests),
        local_schema_digest=local_schema_digest,
        global_schema_digest=global_schema_digest,
        pit_flow_profile=None,
        source_capability_digest=capability.source_digest,
    )
    validate_causal_alpha_v4_context_manifest_against_base(manifest, base_runtime)
    if (
        existing_manifest is not None
        and existing_manifest.manifest_digest != manifest.manifest_digest
    ):
        raise ValueError(
            "existing V4 context manifest identity differs from current generation"
        )
    write_causal_alpha_v4_context_manifest(manifest_path, manifest)
    return CausalAlphaV4ContextGenerationResult(
        manifest_path=manifest_path,
        manifest_digest=manifest.manifest_digest,
        context_paths=tuple(context_paths),
    )


__all__ = [
    "CausalAlphaV4ContextGenerationResult",
    "materialize_causal_alpha_v4_context_generation",
]
