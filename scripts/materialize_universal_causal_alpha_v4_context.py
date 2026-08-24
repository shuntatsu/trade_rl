"""Materialize one immutable Causal Alpha V4 auxiliary-context generation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.v4_context import V4TargetContext
from trade_rl.data.v4_context_artifact import (
    load_v4_target_context_artifact,
    write_v4_target_context_artifact,
)
from trade_rl.domain.common import require_sha256
from trade_rl.workflows.universal_causal_alpha_v4_manifest import (
    CausalAlphaV4ContextManifest,
    load_causal_alpha_v4_context_manifest,
    validate_causal_alpha_v4_context_manifest_against_base,
    write_causal_alpha_v4_context_manifest,
)

_REQUESTED_PROFILES = frozenset({"core", "derivatives-auto"})
_RESOLVED_PROFILES = frozenset({"cross_market_core_v1", "cross_market_derivatives_v1"})
_CONTEXT_ARTIFACT_RELPATH = Path("contexts")
_MANIFEST_NAME = "manifest.json"


class _SourceCapability(Protocol):
    profile_name: str
    source_digest: str


CapabilityResolver = Callable[[str], object]
ContextBuilder = Callable[[str, str], V4TargetContext]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize immutable Causal Alpha V4 auxiliary context"
    )
    parser.add_argument("--runtime-manifest", required=True, type=Path)
    parser.add_argument("--frozen-metadata-root", required=True, type=Path)
    parser.add_argument("--market-data-cache-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--profile",
        required=True,
        choices=tuple(sorted(_REQUESTED_PROFILES)),
    )
    return parser


def _ordered_runtime_symbols(base_runtime: object) -> tuple[str, ...]:
    digest = getattr(base_runtime, "manifest_digest", None)
    if not isinstance(digest, str):
        raise ValueError("V4 materializer base runtime digest is unavailable")
    require_sha256(digest, field="V4 materializer base runtime manifest_digest")
    symbols = (
        *tuple(getattr(base_runtime, "train_symbols", ())),
        *tuple(getattr(base_runtime, "validation_symbols", ())),
        *tuple(getattr(base_runtime, "test_symbols", ())),
    )
    if (
        not symbols
        or len(set(symbols)) != len(symbols)
        or any(not isinstance(symbol, str) or not symbol for symbol in symbols)
    ):
        raise ValueError(
            "V4 materializer runtime symbols must be non-empty, unique strings"
        )
    return symbols


def _resolve_capability(
    requested_profile: str,
    resolver: CapabilityResolver,
) -> tuple[str, str]:
    if requested_profile not in _REQUESTED_PROFILES:
        raise ValueError("V4 materializer requested profile is unsupported")
    if not callable(resolver):
        raise TypeError("V4 materializer capability_resolver must be callable")
    capability = resolver(requested_profile)
    profile_name = getattr(capability, "profile_name", None)
    source_digest = getattr(capability, "source_digest", None)
    if profile_name not in _RESOLVED_PROFILES:
        raise ValueError("V4 materializer resolved profile is unsupported")
    if requested_profile == "core" and profile_name != "cross_market_core_v1":
        raise ValueError("V4 core request cannot resolve to a derivatives profile")
    if not isinstance(source_digest, str):
        raise ValueError("V4 materializer capability digest is unavailable")
    require_sha256(source_digest, field="V4 materializer source capability digest")
    return profile_name, source_digest


def _validate_context(
    context: object,
    *,
    symbol: str,
    profile_name: str,
) -> V4TargetContext:
    if not isinstance(context, V4TargetContext):
        raise TypeError("V4 materializer context builder returned an invalid artifact")
    if context.symbol != symbol:
        raise ValueError("V4 materializer context symbol identity drifted")
    if context.profile_name != profile_name:
        raise ValueError("V4 materializer context profile identity drifted")
    return context


def _schema_digest(
    *,
    profile_name: str,
    feature_names: tuple[str, ...],
    kind: str,
) -> str:
    return content_digest(
        {
            "feature_names": feature_names,
            "kind": kind,
            "profile_name": profile_name,
            "schema_version": "causal_alpha_v4_context_feature_schema_v1",
        }
    )


def _load_or_build_context(
    *,
    root: Path,
    symbol: str,
    profile_name: str,
    context_builder: ContextBuilder,
) -> V4TargetContext:
    if root.exists():
        return _validate_context(
            load_v4_target_context_artifact(root),
            symbol=symbol,
            profile_name=profile_name,
        )
    if not callable(context_builder):
        raise TypeError("V4 materializer context_builder must be callable")
    context = _validate_context(
        context_builder(symbol, profile_name),
        symbol=symbol,
        profile_name=profile_name,
    )
    write_v4_target_context_artifact(root, context)
    loaded = load_v4_target_context_artifact(root)
    if loaded.digest != context.digest:
        raise ValueError("V4 materializer published context identity drifted")
    return loaded


def _require_common_schemas(
    contexts: tuple[V4TargetContext, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not contexts:
        raise ValueError("V4 materializer requires at least one context")
    local_names = contexts[0].local.feature_names
    global_names = contexts[0].global_market.feature_names
    for context in contexts[1:]:
        if context.local.feature_names != local_names:
            raise ValueError("V4 materializer local context schema drifted")
        if context.global_market.feature_names != global_names:
            raise ValueError("V4 materializer global context schema drifted")
    return local_names, global_names


def _resume_existing_manifest(
    *,
    manifest: CausalAlphaV4ContextManifest,
    base_runtime: object,
    output_root: Path,
    symbols: tuple[str, ...],
    profile_name: str,
    source_capability_digest: str,
    context_builder: ContextBuilder,
) -> CausalAlphaV4ContextManifest:
    validate_causal_alpha_v4_context_manifest_against_base(manifest, base_runtime)
    if manifest.profile_name != profile_name:
        raise ValueError("V4 materializer existing manifest profile drifted")
    if manifest.source_capability_digest != source_capability_digest:
        raise ValueError("V4 materializer existing capability identity drifted")
    if manifest.context_artifact_relpath != _CONTEXT_ARTIFACT_RELPATH:
        raise ValueError("V4 materializer existing artifact layout drifted")
    expected = dict(manifest.context_digests)
    contexts: list[V4TargetContext] = []
    for symbol in symbols:
        context = _load_or_build_context(
            root=output_root / manifest.context_artifact_relpath / symbol,
            symbol=symbol,
            profile_name=profile_name,
            context_builder=context_builder,
        )
        if context.digest != expected[symbol]:
            raise ValueError("V4 materializer resumed context identity drifted")
        contexts.append(context)
    local_names, global_names = _require_common_schemas(tuple(contexts))
    if manifest.local_schema_digest != _schema_digest(
        profile_name=profile_name,
        feature_names=local_names,
        kind="local",
    ):
        raise ValueError("V4 materializer local schema identity drifted")
    if manifest.global_schema_digest != _schema_digest(
        profile_name=profile_name,
        feature_names=global_names,
        kind="global",
    ):
        raise ValueError("V4 materializer global schema identity drifted")
    return manifest


def materialize_causal_alpha_v4_context_generation(
    *,
    base_runtime: object,
    output_root: Path,
    requested_profile: str,
    capability_resolver: CapabilityResolver,
    context_builder: ContextBuilder,
) -> CausalAlphaV4ContextManifest:
    """Publish every target context before atomically publishing its manifest."""

    symbols = _ordered_runtime_symbols(base_runtime)
    profile_name, source_capability_digest = _resolve_capability(
        requested_profile,
        capability_resolver,
    )
    root = Path(output_root)
    manifest_path = root / _MANIFEST_NAME
    if manifest_path.exists():
        manifest = load_causal_alpha_v4_context_manifest(manifest_path)
        return _resume_existing_manifest(
            manifest=manifest,
            base_runtime=base_runtime,
            output_root=root,
            symbols=symbols,
            profile_name=profile_name,
            source_capability_digest=source_capability_digest,
            context_builder=context_builder,
        )

    contexts = tuple(
        _load_or_build_context(
            root=root / _CONTEXT_ARTIFACT_RELPATH / symbol,
            symbol=symbol,
            profile_name=profile_name,
            context_builder=context_builder,
        )
        for symbol in symbols
    )
    local_names, global_names = _require_common_schemas(contexts)
    base_digest = getattr(base_runtime, "manifest_digest")
    assert isinstance(base_digest, str)
    manifest = CausalAlphaV4ContextManifest(
        base_runtime_manifest_digest=base_digest,
        profile_name=profile_name,
        context_artifact_relpath=_CONTEXT_ARTIFACT_RELPATH,
        context_digests=tuple(
            (symbol, context.digest)
            for symbol, context in zip(symbols, contexts, strict=True)
        ),
        local_schema_digest=_schema_digest(
            profile_name=profile_name,
            feature_names=local_names,
            kind="local",
        ),
        global_schema_digest=_schema_digest(
            profile_name=profile_name,
            feature_names=global_names,
            kind="global",
        ),
        pit_flow_profile=None,
        source_capability_digest=source_capability_digest,
    )
    validate_causal_alpha_v4_context_manifest_against_base(manifest, base_runtime)
    write_causal_alpha_v4_context_manifest(manifest_path, manifest)
    loaded = load_causal_alpha_v4_context_manifest(manifest_path)
    if loaded.manifest_digest != manifest.manifest_digest:
        raise ValueError("V4 materializer published manifest identity drifted")
    return loaded


def materialize_causal_alpha_v4_context_from_paths(
    *,
    runtime_manifest_path: Path,
    frozen_metadata_root: Path,
    market_data_cache_root: Path,
    output_root: Path,
    requested_profile: str,
) -> CausalAlphaV4ContextManifest:
    """Resolve concrete Binance inputs lazily and run the pure materializer."""

    from trade_rl.workflows.universal_causal_alpha_v4_materialization import (
        materialize_binance_causal_alpha_v4_context,
    )

    return materialize_binance_causal_alpha_v4_context(
        runtime_manifest_path=Path(runtime_manifest_path),
        frozen_metadata_root=Path(frozen_metadata_root),
        market_data_cache_root=Path(market_data_cache_root),
        output_root=Path(output_root),
        requested_profile=requested_profile,
        materialize_generation=materialize_causal_alpha_v4_context_generation,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = materialize_causal_alpha_v4_context_from_paths(
        runtime_manifest_path=args.runtime_manifest,
        frozen_metadata_root=args.frozen_metadata_root,
        market_data_cache_root=args.market_data_cache_root,
        output_root=args.output_root,
        requested_profile=args.profile,
    )
    print(
        json.dumps(
            {
                "context_count": len(manifest.context_digests),
                "manifest_digest": manifest.manifest_digest,
                "output_root": str(args.output_root.resolve()),
                "profile_name": manifest.profile_name,
                "source_capability_digest": manifest.source_capability_digest,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
