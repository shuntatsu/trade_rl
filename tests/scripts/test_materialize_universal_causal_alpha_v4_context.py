from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.materialize_universal_causal_alpha_v4_context import (
    _parser,
    materialize_causal_alpha_v4_context_generation,
)
from trade_rl.data.v4_context import (
    CROSS_MARKET_CORE_NAMES,
    GLOBAL_MARKET_CORE_NAMES,
    V4ContextBlock,
    V4TargetContext,
)
from trade_rl.workflows.universal_causal_alpha_v4_manifest import (
    load_causal_alpha_v4_context_manifest,
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


def _base() -> SimpleNamespace:
    symbols = _symbols()
    return SimpleNamespace(
        manifest_digest=_digest("a"),
        train_symbols=symbols[:9],
        validation_symbols=symbols[9:12],
        test_symbols=symbols[12:],
    )


def _context(symbol: str, profile_name: str) -> V4TargetContext:
    rows = 4
    decisions = np.arange(rows, dtype=np.int64)
    local_values = np.zeros((rows, len(CROSS_MARKET_CORE_NAMES)), dtype=np.float64)
    global_values = np.zeros((rows, len(GLOBAL_MARKET_CORE_NAMES)), dtype=np.float64)
    local = V4ContextBlock(
        feature_names=CROSS_MARKET_CORE_NAMES,
        decision_indices=decisions,
        values=local_values,
        available=np.zeros(local_values.shape, dtype=np.bool_),
        staleness_hours=np.zeros(local_values.shape, dtype=np.float64),
        source_digest=_digest("b"),
    )
    global_market = V4ContextBlock(
        feature_names=GLOBAL_MARKET_CORE_NAMES,
        decision_indices=decisions,
        values=global_values,
        available=np.zeros(global_values.shape, dtype=np.bool_),
        staleness_hours=np.zeros(global_values.shape, dtype=np.float64),
        source_digest=_digest("c"),
    )
    return V4TargetContext(
        symbol=symbol,
        local=local,
        global_market=global_market,
        beta=np.zeros(rows, dtype=np.float64),
        beta_available=np.zeros(rows, dtype=np.bool_),
        beta_source_digest=_digest("d"),
        profile_name=profile_name,
    )


def _capability() -> SimpleNamespace:
    return SimpleNamespace(
        profile_name="cross_market_core_v1",
        source_digest=_digest("e"),
    )


def test_v4_context_materializer_parser_requires_explicit_artifact_inputs() -> None:
    args = _parser().parse_args(
        [
            "--runtime-manifest",
            "runtime/manifest.json",
            "--frozen-metadata-root",
            "runtime/frozen-metadata/usds-m",
            "--market-data-cache-root",
            "cache/binance-vision",
            "--output-root",
            "runtime/v4-context/run-1",
            "--profile",
            "derivatives-auto",
        ]
    )

    assert args.runtime_manifest == Path("runtime/manifest.json")
    assert args.profile == "derivatives-auto"


def test_v4_context_materializer_publishes_manifest_last_and_recovers_one_missing(
    tmp_path: Path,
) -> None:
    base = _base()
    output_root = tmp_path / "v4"
    build_calls: list[str] = []

    def build_context(symbol: str, profile_name: str) -> V4TargetContext:
        build_calls.append(symbol)
        assert profile_name == "cross_market_core_v1"
        return _context(symbol, profile_name)

    manifest = materialize_causal_alpha_v4_context_generation(
        base_runtime=base,
        output_root=output_root,
        requested_profile="derivatives-auto",
        capability_resolver=lambda _profile: _capability(),
        context_builder=build_context,
    )

    assert tuple(symbol for symbol, _ in manifest.context_digests) == _symbols()
    assert manifest.base_runtime_manifest_digest == base.manifest_digest
    assert manifest.profile_name == "cross_market_core_v1"
    assert manifest.source_capability_digest == _digest("e")
    assert build_calls == list(_symbols())
    manifest_path = output_root / "manifest.json"
    assert manifest_path.is_file()
    assert load_causal_alpha_v4_context_manifest(manifest_path) == manifest

    missing_symbol = _symbols()[-1]
    shutil.rmtree(output_root / "contexts" / missing_symbol)
    build_calls.clear()
    recovered = materialize_causal_alpha_v4_context_generation(
        base_runtime=base,
        output_root=output_root,
        requested_profile="derivatives-auto",
        capability_resolver=lambda _profile: _capability(),
        context_builder=build_context,
    )

    assert recovered.manifest_digest == manifest.manifest_digest
    assert build_calls == [missing_symbol]
    assert (output_root / "contexts" / missing_symbol / "manifest.json").is_file()


def test_v4_context_materializer_rejects_existing_artifact_identity_drift(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "v4"
    materialize_causal_alpha_v4_context_generation(
        base_runtime=_base(),
        output_root=output_root,
        requested_profile="core",
        capability_resolver=lambda _profile: _capability(),
        context_builder=_context,
    )
    artifact_manifest = output_root / "contexts" / "BTCUSDT" / "manifest.json"
    payload = json.loads(artifact_manifest.read_text(encoding="utf-8"))
    payload["context_digest"] = _digest("f")
    artifact_manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="digest mismatch|identity"):
        materialize_causal_alpha_v4_context_generation(
            base_runtime=_base(),
            output_root=output_root,
            requested_profile="core",
            capability_resolver=lambda _profile: _capability(),
            context_builder=lambda *_: pytest.fail(
                "drifted artifact must fail before build"
            ),
        )


def test_v4_context_materializer_never_publishes_manifest_after_partial_failure(
    tmp_path: Path,
) -> None:
    calls = 0

    def fail_on_third(symbol: str, profile_name: str) -> V4TargetContext:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("source failure")
        return _context(symbol, profile_name)

    with pytest.raises(RuntimeError, match="source failure"):
        materialize_causal_alpha_v4_context_generation(
            base_runtime=_base(),
            output_root=tmp_path / "v4",
            requested_profile="core",
            capability_resolver=lambda _profile: _capability(),
            context_builder=fail_on_third,
        )

    assert not (tmp_path / "v4" / "manifest.json").exists()
