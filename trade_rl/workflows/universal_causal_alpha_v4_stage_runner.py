"""Concrete stage assembly helpers for the research-only Causal Alpha V4 runner."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256

CAUSAL_ALPHA_V4_STAGE_RUN_IDENTITY_SCHEMA: Final = (
    "causal_alpha_v4_stage_run_identity_v1"
)


def build_causal_alpha_v4_stage_run_identity(
    *,
    base_runtime_manifest_digest: str,
    v4_context_manifest_digest: str,
    config_digest: str,
    execution_identity_digest: str,
    nested_partition_digest: str,
    generator_code_digest: str,
) -> str:
    """Bind every immutable upstream identity into one V4 run identity."""

    payload = {
        "base_runtime_manifest_digest": base_runtime_manifest_digest,
        "config_digest": config_digest,
        "execution_identity_digest": execution_identity_digest,
        "generator_code_digest": generator_code_digest,
        "nested_partition_digest": nested_partition_digest,
        "schema_version": CAUSAL_ALPHA_V4_STAGE_RUN_IDENTITY_SCHEMA,
        "v4_context_manifest_digest": v4_context_manifest_digest,
    }
    for field_name, value in payload.items():
        if field_name == "schema_version":
            continue
        require_sha256(str(value), field=f"V4 stage {field_name}")
    return content_digest(payload)


def require_causal_alpha_v4_context_scope(
    *,
    train_symbols: tuple[str, ...],
    provider: object | None,
    manifest: object,
) -> dict[str, Any]:
    """Return exact ordered train contexts only after provider/manifest closure."""

    symbols = tuple(train_symbols)
    if (
        not symbols
        or len(set(symbols)) != len(symbols)
        or any(not isinstance(symbol, str) or not symbol for symbol in symbols)
    ):
        raise ValueError("V4 stage train_symbols must be non-empty and unique")
    if provider is None:
        raise ValueError("V4 stage requires V4 context provider")
    raw_contexts = getattr(provider, "contexts", None)
    if not isinstance(raw_contexts, Mapping):
        raise TypeError("V4 stage context provider does not expose contexts")
    contexts = dict(raw_contexts)
    raw_digests = getattr(manifest, "context_digests", None)
    if not isinstance(raw_digests, (tuple, list)):
        raise TypeError("V4 stage context manifest does not expose context_digests")
    try:
        expected = tuple((str(symbol), str(digest)) for symbol, digest in raw_digests)
    except (TypeError, ValueError) as error:
        raise ValueError("V4 stage context manifest digests are invalid") from error
    if tuple(symbol for symbol, _ in expected) != symbols or set(contexts) != set(symbols):
        raise ValueError("V4 stage context scope does not match train_symbols")

    ordered: dict[str, Any] = {}
    for symbol, expected_digest in expected:
        require_sha256(expected_digest, field=f"V4 stage context digest {symbol}")
        context = contexts[symbol]
        if getattr(context, "symbol", None) != symbol:
            raise ValueError("V4 stage context artifact symbol identity drifted")
        if getattr(context, "digest", None) != expected_digest:
            raise ValueError("V4 stage context artifact identity drifted")
        ordered[symbol] = context
    return ordered


__all__ = [
    "CAUSAL_ALPHA_V4_STAGE_RUN_IDENTITY_SCHEMA",
    "build_causal_alpha_v4_stage_run_identity",
    "require_causal_alpha_v4_context_scope",
]
