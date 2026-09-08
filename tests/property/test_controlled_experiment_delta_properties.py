from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from trade_rl.evaluation.experiments.contracts import ControlledFactor
from trade_rl.evaluation.experiments.delta import FACTOR_RULES, _classify_resolved_delta


@given(
    has_allowed_change=st.booleans(),
    has_forbidden_change=st.booleans(),
)
def test_delta_partition_matches_allowed_and_forbidden_paths(
    has_allowed_change: bool,
    has_forbidden_change: bool,
) -> None:
    baseline: dict[str, object] = {
        "ppo_total_timesteps": 256,
        "initial_capital": 100_000.0,
        "evaluation_start": "2026-02-01T00:00:00.000000000",
    }
    candidate = dict(baseline)
    if has_allowed_change:
        candidate["ppo_total_timesteps"] = 512
    if has_forbidden_change:
        candidate["initial_capital"] = 200_000.0

    changed, forbidden = _classify_resolved_delta(
        baseline,
        candidate,
        FACTOR_RULES[ControlledFactor.PPO_TRAINING_BUDGET],
    )

    controlled = bool(changed) and not forbidden
    assert controlled is (has_allowed_change and not has_forbidden_change)
    assert (("ppo_total_timesteps",) in changed) is has_allowed_change
    assert (("initial_capital",) in forbidden) is has_forbidden_change
