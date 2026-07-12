from types import SimpleNamespace

import pytest

from mars_lite.pipeline.training_engine import build_env_kwargs
from mars_lite.trading.post_processor import make_legacy_processor


def _args(decision_every: int, *, scan_horizons: bool = False):
    return SimpleNamespace(
        action_mode="baseline-residual",
        decision_every=decision_every,
        scan_horizons=scan_horizons,
        fee_profile="taker",
        min_trade_delta=0.0,
        reward_scale=100.0,
        htf_gate=False,
        obs_risk_state=False,
    )


def test_residual_decision_every_one_is_explicitly_forwarded() -> None:
    args = _args(1)

    kwargs = build_env_kwargs(args, make_legacy_processor(0.0), horizon=12)

    assert kwargs["decision_every"] == 1
    assert args.decision_every == 1


def test_auto_decision_interval_is_recorded_back_to_args() -> None:
    args = _args(1, scan_horizons=True)

    kwargs = build_env_kwargs(args, make_legacy_processor(0.0), horizon=12)

    assert kwargs["decision_every"] == 6
    assert args.decision_every == 6


def test_invalid_decision_interval_is_rejected() -> None:
    with pytest.raises(ValueError, match="decision_every"):
        build_env_kwargs(_args(0), make_legacy_processor(0.0), horizon=12)
