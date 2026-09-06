"""Authoritative U2 Development replay boundary after pre-development freeze."""

from __future__ import annotations

from typing import Any, Final

from trade_rl.workflows.universal_trade_rl_u2_predevelopment import (
    universal_trade_rl_u2_evaluation_seed,
)
from trade_rl.workflows.universal_trade_rl_u2_replay import (
    UniversalTradeRLU2ReplayEvidence,
    UniversalTradeRLU2ReplayRequest,
    UniversalTradeRLU2ReplaySession,
)

U2_DEVELOPMENT_REPLAY_AUTHORITY_SCHEMA: Final = (
    "universal_trade_rl_u2_development_replay_authority_v1"
)


def replay_universal_trade_rl_u2_development_scope(
    *,
    session: UniversalTradeRLU2ReplaySession,
    request: UniversalTradeRLU2ReplayRequest,
    model: Any | None = None,
) -> UniversalTradeRLU2ReplayEvidence:
    """Replay one scope only when its preregistered common RNG seed is used.

    ``UniversalTradeRLU2ReplaySession`` remains the lower-level deterministic
    replay engine used by synthetic diagnostics.  U2 Development Selection is
    authoritative only through this boundary, which rejects a seed mismatch
    before environment creation or numeric stepping.
    """

    if not isinstance(session, UniversalTradeRLU2ReplaySession):
        raise TypeError("U2 Development replay authority requires a replay session")
    if not isinstance(request, UniversalTradeRLU2ReplayRequest):
        raise TypeError("U2 Development replay authority requires a replay request")

    scope = session.scope(request.scope_digest)
    expected_seed = universal_trade_rl_u2_evaluation_seed(
        u2_contract_digest=session.u2_contract.digest,
        scope_digest=scope.digest,
    )
    if request.evaluation_seed != expected_seed:
        raise ValueError(
            "U2 Development replay evaluation seed must equal the scope-common RNG seed"
        )
    return session.replay(request, model=model)


__all__ = [
    "U2_DEVELOPMENT_REPLAY_AUTHORITY_SCHEMA",
    "replay_universal_trade_rl_u2_development_scope",
]
