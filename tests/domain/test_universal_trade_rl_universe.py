from __future__ import annotations

import pytest

from trade_rl.domain.universal_trade_rl_universe import (
    UniversalTradeRLSymbolExclusion,
    UniversalTradeRLSymbolRole,
    UniversalTradeRLUniverseConfig,
)


def _config() -> UniversalTradeRLUniverseConfig:
    return UniversalTradeRLUniverseConfig(
        train_symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT"),
        development_symbols=("LINKUSDT",),
        admission_symbols=("AVAXUSDT",),
        exclusions=(
            UniversalTradeRLSymbolExclusion(
                symbol="LUNA2USDT",
                reason="insufficient_contiguous_history",
            ),
        ),
    )


def test_roles_are_disjoint_sorted_and_digest_bound() -> None:
    config = _config()

    assert config.role_for("BTCUSDT") is UniversalTradeRLSymbolRole.TRAIN
    assert config.role_for("LINKUSDT") is UniversalTradeRLSymbolRole.DEVELOPMENT
    assert config.role_for("AVAXUSDT") is UniversalTradeRLSymbolRole.ADMISSION
    assert config.role_for("LUNA2USDT") is None
    assert len(config.digest) == 64
    assert config.to_payload()["artifact_digest"] == config.digest


def test_role_overlap_is_rejected() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        UniversalTradeRLUniverseConfig(
            train_symbols=("BTCUSDT", "ETHUSDT"),
            development_symbols=("ETHUSDT",),
            admission_symbols=("AVAXUSDT",),
        )


def test_role_groups_must_be_non_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        UniversalTradeRLUniverseConfig(
            train_symbols=("BTCUSDT",),
            development_symbols=(),
            admission_symbols=("AVAXUSDT",),
        )


def test_role_groups_must_be_sorted_and_unique() -> None:
    with pytest.raises(ValueError, match="sorted and unique"):
        UniversalTradeRLUniverseConfig(
            train_symbols=("ETHUSDT", "BTCUSDT"),
            development_symbols=("LINKUSDT",),
            admission_symbols=("AVAXUSDT",),
        )


def test_btc_market_proxy_must_be_train() -> None:
    with pytest.raises(ValueError, match="BTCUSDT"):
        UniversalTradeRLUniverseConfig(
            train_symbols=("ETHUSDT",),
            development_symbols=("LINKUSDT",),
            admission_symbols=("AVAXUSDT",),
        )


def test_assigned_symbol_cannot_also_be_excluded() -> None:
    with pytest.raises(ValueError, match="assigned and excluded"):
        UniversalTradeRLUniverseConfig(
            train_symbols=("BTCUSDT", "ETHUSDT"),
            development_symbols=("LINKUSDT",),
            admission_symbols=("AVAXUSDT",),
            exclusions=(
                UniversalTradeRLSymbolExclusion(
                    symbol="LINKUSDT",
                    reason="should_not_overlap",
                ),
            ),
        )


def test_exclusion_reason_must_be_non_empty() -> None:
    with pytest.raises(ValueError, match="reason"):
        UniversalTradeRLSymbolExclusion(symbol="LUNA2USDT", reason="  ")


def test_symbols_use_canonical_uppercase_market_syntax() -> None:
    with pytest.raises(ValueError, match="canonical"):
        UniversalTradeRLUniverseConfig(
            train_symbols=("BTCUSDT", "ethusdt"),
            development_symbols=("LINKUSDT",),
            admission_symbols=("AVAXUSDT",),
        )


def test_digest_tampering_is_rejected() -> None:
    config = _config()

    with pytest.raises(ValueError, match="digest"):
        UniversalTradeRLUniverseConfig(
            train_symbols=config.train_symbols,
            development_symbols=config.development_symbols,
            admission_symbols=config.admission_symbols,
            exclusions=config.exclusions,
            digest="f" * 64,
        )
