from __future__ import annotations

from trade_rl.workflows import universal_trade_rl_u2_development_authority as authority
from trade_rl.workflows import universal_trade_rl_u2_replay as replay


def test_u2_numeric_development_session_has_only_lock_bound_public_entry() -> None:
    unlocked_name = "build_universal_trade_rl_u2_development_replay_session"
    private_name = "_build_universal_trade_rl_u2_development_replay_session_unlocked"

    assert unlocked_name not in replay.__all__
    assert not hasattr(replay, unlocked_name)
    assert hasattr(replay, private_name)
    assert hasattr(
        authority,
        "build_authoritative_universal_trade_rl_u2_development_replay_session",
    )
