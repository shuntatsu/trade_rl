from __future__ import annotations

from types import SimpleNamespace

from trade_rl.workflows.universal_causal_alpha_v4_stage_runner import (
    require_causal_alpha_v4_context_scope,
)


def _digest(char: str) -> str:
    return char * 64


def _context(symbol: str, digest_char: str) -> SimpleNamespace:
    return SimpleNamespace(symbol=symbol, digest=_digest(digest_char))


def test_v4_stage_context_scope_selects_train_subset_from_complete_manifest() -> None:
    manifest = SimpleNamespace(
        context_digests=(
            ("BTCUSDT", _digest("1")),
            ("ETHUSDT", _digest("2")),
            ("SOLUSDT", _digest("3")),
            ("XRPUSDT", _digest("4")),
        )
    )
    provider = SimpleNamespace(
        contexts={
            "BTCUSDT": _context("BTCUSDT", "1"),
            "ETHUSDT": _context("ETHUSDT", "2"),
            "SOLUSDT": _context("SOLUSDT", "3"),
            "XRPUSDT": _context("XRPUSDT", "4"),
        }
    )

    resolved = require_causal_alpha_v4_context_scope(
        train_symbols=("BTCUSDT", "ETHUSDT"),
        provider=provider,
        manifest=manifest,
    )

    assert tuple(resolved) == ("BTCUSDT", "ETHUSDT")
    assert resolved["BTCUSDT"].digest == _digest("1")
    assert resolved["ETHUSDT"].digest == _digest("2")
