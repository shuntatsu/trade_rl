from __future__ import annotations

from types import SimpleNamespace

import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.workflows.universal_causal_alpha_v4_stage_runner import (
    build_causal_alpha_v4_stage_run_identity,
    require_causal_alpha_v4_context_scope,
)


def _digest(char: str) -> str:
    return char * 64


def test_v4_stage_run_identity_binds_all_immutable_inputs() -> None:
    baseline = build_causal_alpha_v4_stage_run_identity(
        base_runtime_manifest_digest=_digest("a"),
        v4_context_manifest_digest=_digest("b"),
        config_digest=_digest("c"),
        execution_identity_digest=_digest("d"),
        nested_partition_digest=_digest("e"),
        generator_code_digest=_digest("f"),
    )

    assert len(baseline) == 64
    for field, replacement in (
        ("base_runtime_manifest_digest", _digest("1")),
        ("v4_context_manifest_digest", _digest("2")),
        ("config_digest", _digest("3")),
        ("execution_identity_digest", _digest("4")),
        ("nested_partition_digest", _digest("5")),
        ("generator_code_digest", _digest("6")),
    ):
        values = {
            "base_runtime_manifest_digest": _digest("a"),
            "v4_context_manifest_digest": _digest("b"),
            "config_digest": _digest("c"),
            "execution_identity_digest": _digest("d"),
            "nested_partition_digest": _digest("e"),
            "generator_code_digest": _digest("f"),
        }
        values[field] = replacement
        assert build_causal_alpha_v4_stage_run_identity(**values) != baseline


def _context(symbol: str, digest_char: str) -> SimpleNamespace:
    return SimpleNamespace(symbol=symbol, digest=_digest(digest_char))


def test_v4_stage_context_scope_requires_provider_and_exact_manifest_order() -> None:
    manifest = SimpleNamespace(
        context_digests=(("BTCUSDT", _digest("1")), ("ETHUSDT", _digest("2")))
    )

    with pytest.raises(ValueError, match="requires V4 context provider"):
        require_causal_alpha_v4_context_scope(
            train_symbols=("BTCUSDT", "ETHUSDT"),
            provider=None,
            manifest=manifest,
        )

    provider = SimpleNamespace(
        contexts={
            "BTCUSDT": _context("BTCUSDT", "1"),
            "ETHUSDT": _context("ETHUSDT", "2"),
        }
    )
    resolved = require_causal_alpha_v4_context_scope(
        train_symbols=("BTCUSDT", "ETHUSDT"),
        provider=provider,
        manifest=manifest,
    )
    assert tuple(resolved) == ("BTCUSDT", "ETHUSDT")


def test_v4_stage_context_scope_rejects_missing_or_drifted_context() -> None:
    manifest = SimpleNamespace(
        context_digests=(("BTCUSDT", _digest("1")), ("ETHUSDT", _digest("2")))
    )
    missing = SimpleNamespace(contexts={"BTCUSDT": _context("BTCUSDT", "1")})
    with pytest.raises(ValueError, match="context scope"):
        require_causal_alpha_v4_context_scope(
            train_symbols=("BTCUSDT", "ETHUSDT"),
            provider=missing,
            manifest=manifest,
        )

    drifted = SimpleNamespace(
        contexts={
            "BTCUSDT": _context("BTCUSDT", "1"),
            "ETHUSDT": _context("ETHUSDT", "3"),
        }
    )
    with pytest.raises(ValueError, match="context artifact identity"):
        require_causal_alpha_v4_context_scope(
            train_symbols=("BTCUSDT", "ETHUSDT"),
            provider=drifted,
            manifest=manifest,
        )


def test_v4_stage_run_identity_is_canonical_content_digest() -> None:
    payload = {
        "base_runtime_manifest_digest": _digest("a"),
        "config_digest": _digest("c"),
        "execution_identity_digest": _digest("d"),
        "generator_code_digest": _digest("f"),
        "nested_partition_digest": _digest("e"),
        "schema_version": "causal_alpha_v4_stage_run_identity_v1",
        "v4_context_manifest_digest": _digest("b"),
    }
    assert build_causal_alpha_v4_stage_run_identity(
        base_runtime_manifest_digest=payload["base_runtime_manifest_digest"],
        v4_context_manifest_digest=payload["v4_context_manifest_digest"],
        config_digest=payload["config_digest"],
        execution_identity_digest=payload["execution_identity_digest"],
        nested_partition_digest=payload["nested_partition_digest"],
        generator_code_digest=payload["generator_code_digest"],
    ) == content_digest(payload)
