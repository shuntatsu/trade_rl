from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from trade_rl.workflows.universal_causal_alpha_v4_manifest import (
    CausalAlphaV4ContextManifest,
    load_causal_alpha_v4_context_manifest,
    validate_causal_alpha_v4_context_manifest_against_base,
    write_causal_alpha_v4_context_manifest,
)


def _digest(char: str) -> str:
    return char * 64


def _symbols() -> tuple[str, ...]:
    return (
        "BTCUSDT",
        "ETHUSDT",
        "BNBUSDT",
        "SOLUSDT",
        "XRPUSDT",
        "ADAUSDT",
        "DOGEUSDT",
        "LINKUSDT",
        "AVAXUSDT",
        "LTCUSDT",
        "BCHUSDT",
        "DOTUSDT",
        "APTUSDT",
        "ARBUSDT",
        "SUIUSDT",
    )


def _manifest() -> CausalAlphaV4ContextManifest:
    return CausalAlphaV4ContextManifest(
        base_runtime_manifest_digest=_digest("a"),
        profile_name="cross_market_core_v1",
        context_artifact_relpath=Path("v4-context"),
        context_digests=tuple(
            (symbol, f"{index + 1:064x}") for index, symbol in enumerate(_symbols())
        ),
        local_schema_digest=_digest("b"),
        global_schema_digest=_digest("c"),
        pit_flow_profile=None,
        source_capability_digest=_digest("d"),
    )


def _base():
    symbols = _symbols()
    return SimpleNamespace(
        manifest_digest=_digest("a"),
        train_symbols=symbols[:9],
        validation_symbols=symbols[9:12],
        test_symbols=symbols[12:],
    )


def test_v4_context_manifest_round_trip_and_strict_identity(tmp_path: Path) -> None:
    manifest = _manifest()
    path = write_causal_alpha_v4_context_manifest(tmp_path / "v4.json", manifest)
    loaded = load_causal_alpha_v4_context_manifest(path)
    assert loaded.manifest_digest == manifest.manifest_digest
    assert loaded.context_digests == manifest.context_digests
    assert loaded.context_artifact_relpath == Path("v4-context")
    validate_causal_alpha_v4_context_manifest_against_base(loaded, _base())


def test_v4_context_manifest_rejects_symbol_order_drift() -> None:
    manifest = _manifest()
    reversed_contexts = tuple(reversed(manifest.context_digests))
    drifted = CausalAlphaV4ContextManifest(
        base_runtime_manifest_digest=manifest.base_runtime_manifest_digest,
        profile_name=manifest.profile_name,
        context_artifact_relpath=manifest.context_artifact_relpath,
        context_digests=reversed_contexts,
        local_schema_digest=manifest.local_schema_digest,
        global_schema_digest=manifest.global_schema_digest,
        pit_flow_profile=manifest.pit_flow_profile,
        source_capability_digest=manifest.source_capability_digest,
    )
    with pytest.raises(ValueError, match="symbol order"):
        validate_causal_alpha_v4_context_manifest_against_base(drifted, _base())


def test_v4_context_manifest_rejects_base_digest_drift() -> None:
    manifest = _manifest()
    base = _base()
    base.manifest_digest = _digest("e")
    with pytest.raises(ValueError, match="base runtime"):
        validate_causal_alpha_v4_context_manifest_against_base(manifest, base)


def test_v4_context_manifest_rejects_duplicate_symbol_and_bad_path() -> None:
    manifest = _manifest()
    contexts = list(manifest.context_digests)
    contexts[-1] = (contexts[0][0], contexts[-1][1])
    with pytest.raises(ValueError, match="unique"):
        CausalAlphaV4ContextManifest(
            base_runtime_manifest_digest=manifest.base_runtime_manifest_digest,
            profile_name=manifest.profile_name,
            context_artifact_relpath=manifest.context_artifact_relpath,
            context_digests=tuple(contexts),
            local_schema_digest=manifest.local_schema_digest,
            global_schema_digest=manifest.global_schema_digest,
            pit_flow_profile=None,
            source_capability_digest=manifest.source_capability_digest,
        )

    with pytest.raises(ValueError, match="relative"):
        CausalAlphaV4ContextManifest(
            base_runtime_manifest_digest=manifest.base_runtime_manifest_digest,
            profile_name=manifest.profile_name,
            context_artifact_relpath=Path("../outside"),
            context_digests=manifest.context_digests,
            local_schema_digest=manifest.local_schema_digest,
            global_schema_digest=manifest.global_schema_digest,
            pit_flow_profile=None,
            source_capability_digest=manifest.source_capability_digest,
        )


def test_v4_context_manifest_rejects_unknown_json_field(tmp_path: Path) -> None:
    manifest = _manifest()
    path = write_causal_alpha_v4_context_manifest(tmp_path / "v4.json", manifest)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown|missing|fields"):
        load_causal_alpha_v4_context_manifest(path)


def test_v4_context_manifest_write_is_idempotent_but_not_overwritable(
    tmp_path: Path,
) -> None:
    path = tmp_path / "v4.json"
    manifest = _manifest()
    first = write_causal_alpha_v4_context_manifest(path, manifest)
    second = write_causal_alpha_v4_context_manifest(path, manifest)
    assert first == second == path

    changed = CausalAlphaV4ContextManifest(
        base_runtime_manifest_digest=manifest.base_runtime_manifest_digest,
        profile_name="cross_market_derivatives_v1",
        context_artifact_relpath=manifest.context_artifact_relpath,
        context_digests=manifest.context_digests,
        local_schema_digest=manifest.local_schema_digest,
        global_schema_digest=manifest.global_schema_digest,
        pit_flow_profile=None,
        source_capability_digest=manifest.source_capability_digest,
    )
    with pytest.raises(FileExistsError, match="different"):
        write_causal_alpha_v4_context_manifest(path, changed)
