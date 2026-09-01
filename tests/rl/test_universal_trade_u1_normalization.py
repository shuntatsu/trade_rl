from __future__ import annotations

from pathlib import Path

import pytest

import trade_rl.rl.universal_normalization as universal_normalization
from tests.rl.universal_trade_test_support import make_u1_feature_specs
from trade_rl.domain.universal_trade_rl_universe import UniversalTradeRLUniverseConfig
from trade_rl.rl.universal_trade_contract import UniversalTradePolicyContract
from trade_rl.workflows.universal_trade_rl_universe_access import (
    UniversalTradeRLAccessPhase,
    UniversalTradeRLUniverseAccess,
)
from trade_rl.workflows.universal_trade_rl_universe_config import (
    UniversalTradeRLSymbolSource,
)
from trade_rl.workflows.universal_trade_rl_universe_manifest import (
    UniversalTradeRLUniverseManifest,
    build_universal_trade_rl_universe_manifest,
)


def _source(symbol: str, digest_char: str) -> UniversalTradeRLSymbolSource:
    return UniversalTradeRLSymbolSource(
        symbol=symbol,
        dataset_digest=digest_char * 64,
        first_timestamp_ns=1,
        last_timestamp_ns=2,
        row_count=2,
    )


def _manifest(
    train: tuple[UniversalTradeRLSymbolSource, ...],
) -> UniversalTradeRLUniverseManifest:
    by_symbol = {item.symbol: item for item in train}
    by_symbol["LINKUSDT"] = _source("LINKUSDT", "c")
    by_symbol["AVAXUSDT"] = _source("AVAXUSDT", "a")
    config = UniversalTradeRLUniverseConfig(
        train_symbols=("BTCUSDT", "ETHUSDT"),
        development_symbols=("LINKUSDT",),
        admission_symbols=("AVAXUSDT",),
    )
    return build_universal_trade_rl_universe_manifest(
        config=config,
        sources=tuple(by_symbol[symbol] for symbol in sorted(by_symbol)),
    )


def test_scope_fails_before_missing_artifact_path_is_touched() -> None:
    fit = getattr(
        universal_normalization,
        "fit_universal_trade_sequence_normalizer",
        None,
    )
    published_source_type = getattr(
        universal_normalization,
        "UniversalTradePublishedSource",
        None,
    )
    assert callable(fit), "U1 sequence normalizer fitter is not implemented"
    assert published_source_type is not None, "U1 published source contract is missing"

    train = (
        _source("BTCUSDT", "b"),
        _source("ETHUSDT", "e"),
    )
    manifest = _manifest(train)
    access = UniversalTradeRLUniverseAccess.for_phase(
        manifest=manifest,
        phase=UniversalTradeRLAccessPhase.DEVELOPMENT,
    )

    with pytest.raises(PermissionError, match="normalization|Train"):
        fit(
            manifest=manifest,
            access=access,
            sources=(
                published_source_type("BTCUSDT", Path("/missing/btc")),
                published_source_type("ETHUSDT", Path("/missing/eth")),
            ),
            contract=UniversalTradePolicyContract(
                feature_specs=make_u1_feature_specs()
            ),
            knowledge_cutoff_ns=2,
        )
