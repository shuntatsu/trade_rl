from __future__ import annotations

import trade_rl.evaluation as evaluation


def test_shared_cash_replay_api_exists() -> None:
    run_replay = getattr(evaluation, "run_shared_cash_replay", None)
    assert callable(run_replay), "shared-cash replay is not implemented"
