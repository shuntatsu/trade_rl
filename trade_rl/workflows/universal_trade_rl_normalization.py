"""U1 normalization orchestration above the RL statistics layer."""

from __future__ import annotations

from collections.abc import Sequence

from trade_rl.rl.universal_normalization import UniversalTradePublishedSource
from trade_rl.rl.universal_trade_contract import UniversalTradePolicyContract
from trade_rl.workflows.universal_trade_rl_universe_access import (
    UniversalTradeRLAccessPhase,
    UniversalTradeRLUniverseAccess,
)
from trade_rl.workflows.universal_trade_rl_universe_manifest import (
    UniversalTradeRLUniverseManifest,
)


def fit_universal_trade_sequence_normalizer(
    *,
    manifest: UniversalTradeRLUniverseManifest,
    access: UniversalTradeRLUniverseAccess,
    sources: Sequence[UniversalTradePublishedSource],
    contract: UniversalTradePolicyContract,
    knowledge_cutoff_ns: int,
) -> None:
    """Fit U1 normalization only from an explicitly authorized Train phase."""

    del manifest, contract, knowledge_cutoff_ns
    if access.phase is not UniversalTradeRLAccessPhase.TRAIN:
        raise PermissionError("Universal Trade RL normalization fitting is Train-only")
    access.require_normalization_scope(tuple(source.symbol for source in sources))
    raise NotImplementedError("U1 sequence normalizer fitting is not implemented yet")


__all__ = ["fit_universal_trade_sequence_normalizer"]
