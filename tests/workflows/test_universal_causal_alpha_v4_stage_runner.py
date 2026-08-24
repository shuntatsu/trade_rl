from __future__ import annotations

from types import SimpleNamespace

import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.workflows import universal_causal_alpha_v4_stage_runner as stage_runner
from trade_rl.workflows.universal_causal_alpha_v4_stage_runner import (
    build_causal_alpha_v4_stage_run_identity,
    prepare_causal_alpha_v4_stage_data,
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


def test_v4_stage_preparation_builds_each_symbol_once_and_binds_nested_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    symbols = ("BTCUSDT", "ETHUSDT")
    contexts = {
        "BTCUSDT": _context("BTCUSDT", "1"),
        "ETHUSDT": _context("ETHUSDT", "2"),
    }
    provider = SimpleNamespace(contexts=contexts)
    runtime = SimpleNamespace(
        routed_environment_factory=SimpleNamespace(v4_context_provider=provider)
    )
    context_manifest = SimpleNamespace(
        manifest_digest=_digest("b"),
        context_digests=(("BTCUSDT", _digest("1")), ("ETHUSDT", _digest("2"))),
    )
    runtime_context = SimpleNamespace(
        manifest=SimpleNamespace(
            manifest_digest=_digest("a"),
            fold_train_range=(10, 500),
        ),
        v4_context_manifest=context_manifest,
    )
    closed: list[str] = []

    class _Env:
        def __init__(self, symbol: str) -> None:
            self.dataset = f"dataset:{symbol}"
            self.symbol = symbol

        def close(self) -> None:
            closed.append(self.symbol)

    prepared_v3 = SimpleNamespace(
        train_symbols=symbols,
        partitions={symbol: f"partition:{symbol}" for symbol in symbols},
        samples={symbol: f"base:{symbol}" for symbol in symbols},
        environment_factories={
            symbol: (lambda symbol=symbol: _Env(symbol)) for symbol in symbols
        },
        signal_delays={symbol: 1 for symbol in symbols},
        decision_bars={symbol: 1 for symbol in symbols},
        execution_identity=SimpleNamespace(digest=_digest("d")),
    )
    nested = {
        "BTCUSDT": SimpleNamespace(digest=_digest("4")),
        "ETHUSDT": SimpleNamespace(digest=_digest("5")),
    }
    build_calls: list[tuple[object, object, object, int, int, int]] = []

    def fake_build_samples(**kwargs: object) -> object:
        build_calls.append(
            (
                kwargs["base_samples"],
                kwargs["context"],
                kwargs["dataset"],
                int(kwargs["train_stop"]),
                int(kwargs["signal_delay_decisions"]),
                int(kwargs["decision_bars"]),
            )
        )
        return f"v4:{getattr(kwargs['context'], 'symbol')}"

    def fake_validate_scope(
        *, train_symbols: tuple[str, ...], samples: object
    ) -> object:
        assert train_symbols == symbols
        return samples

    def fake_split(*_args: object, **kwargs: object) -> object:
        assert kwargs["train_symbols"] == symbols
        assert kwargs["signal_contract_count"] == 8
        assert kwargs["minimum_economic_contract_count"] == 4
        return nested

    monkeypatch.setattr(
        stage_runner, "build_causal_alpha_v4_symbol_samples", fake_build_samples
    )
    monkeypatch.setattr(
        stage_runner, "validate_causal_alpha_v4_train_sample_scope", fake_validate_scope
    )
    monkeypatch.setattr(stage_runner, "split_causal_alpha_v3_partitions", fake_split)

    prepared = prepare_causal_alpha_v4_stage_data(
        config_digest=_digest("c"),
        generator_code_digest=_digest("f"),
        runtime_context=runtime_context,
        runtime=runtime,
        prepared_v3=prepared_v3,
    )

    assert tuple(prepared.samples) == symbols
    assert prepared.samples == {"BTCUSDT": "v4:BTCUSDT", "ETHUSDT": "v4:ETHUSDT"}
    assert prepared.nested_partitions is nested
    assert closed == ["BTCUSDT", "ETHUSDT"]
    assert build_calls == [
        ("base:BTCUSDT", contexts["BTCUSDT"], "dataset:BTCUSDT", 500, 1, 1),
        ("base:ETHUSDT", contexts["ETHUSDT"], "dataset:ETHUSDT", 500, 1, 1),
    ]
    nested_digest = content_digest(
        {
            "partitions": (
                ("BTCUSDT", _digest("4")),
                ("ETHUSDT", _digest("5")),
            ),
            "schema_version": "causal_alpha_v4_nested_scope_v1",
        }
    )
    assert prepared.nested_partition_digest == nested_digest
    assert prepared.run_manifest_digest == build_causal_alpha_v4_stage_run_identity(
        base_runtime_manifest_digest=_digest("a"),
        v4_context_manifest_digest=_digest("b"),
        config_digest=_digest("c"),
        execution_identity_digest=_digest("d"),
        nested_partition_digest=nested_digest,
        generator_code_digest=_digest("f"),
    )
