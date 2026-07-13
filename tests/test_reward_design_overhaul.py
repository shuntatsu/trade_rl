from types import SimpleNamespace

import numpy as np
import pytest

from mars_lite.env.baseline_residual_env import BaselineResidualTradingEnv
from mars_lite.env.portfolio_env import PortfolioTradingEnv
from mars_lite.eval import relative_evaluation
from mars_lite.features.feature_pipeline import FeatureSet
from mars_lite.pipeline import production_pipeline, training_engine
from mars_lite.pipeline.cli import build_parser
from mars_lite.trading.post_processor import make_legacy_processor
from mars_lite.trading.trend_family import TrendFamily, TrendFamilyConfig


def _feature_set(
    *, n_bars: int = 160, n_symbols: int = 2, alternating: bool = False
) -> FeatureSet:
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        n_bars
    ) * np.timedelta64(1, "h")
    if alternating:
        step_returns = np.where(np.arange(n_bars - 1) % 2 == 0, 0.02, -0.015)
        one = np.concatenate([[1.0], np.cumprod(1.0 + step_returns)])
    else:
        one = np.ones(n_bars, dtype=np.float64)
    close = np.column_stack([one * (1.0 + 0.001 * idx) for idx in range(n_symbols)])
    return FeatureSet(
        symbols=[f"S{idx}" for idx in range(n_symbols)],
        timestamps=timestamps,
        features=np.zeros((n_bars, n_symbols, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        close=close,
        open_next=close.copy(),
        funding_rate=np.zeros_like(close),
        feature_names=["dummy"],
        global_feature_names=["dummy_global"],
    )


def _residual_env(**overrides: object) -> BaselineResidualTradingEnv:
    kwargs: dict[str, object] = {
        "trend_family": TrendFamily(
            TrendFamilyConfig(
                fast_lookback=12,
                base_lookback=24,
                slow_lookback=48,
                rebalance_every=12,
            )
        ),
        "decision_every": 4,
        "episode_bars": 48,
        "post_processor": make_legacy_processor(0.0),
        "min_trade_delta": 0.0,
        "fee_rate": 0.0,
        "spread_rate": 0.0,
        "impact_rate": 0.0,
    }
    kwargs.update(overrides)
    return BaselineResidualTradingEnv(_feature_set(), **kwargs)


def test_pipeline_default_gamma_is_long_horizon() -> None:
    args = build_parser().parse_args([])
    assert args.gamma == pytest.approx(0.99)


def test_residual_low_gamma_requires_explicit_override() -> None:
    with pytest.raises(ValueError, match="residual gamma"):
        training_engine.resolve_training_gamma(
            "baseline-residual", 0.5, allow_low_residual_gamma=False
        )
    assert training_engine.resolve_training_gamma(
        "baseline-residual", 0.5, allow_low_residual_gamma=True
    ) == pytest.approx(0.5)
    assert training_engine.resolve_training_gamma(
        "direct", 0.5, allow_low_residual_gamma=False
    ) == pytest.approx(0.5)


def test_residual_observation_exposes_paired_state() -> None:
    env = _residual_env()
    obs, _ = env.reset(options={"start_idx": 72})

    assert env.n_per_symbol == env.fs.n_features + 6
    assert env.n_global == env.fs.global_features.shape[1] + 7
    assert obs.shape == env.observation_space.shape
    np.testing.assert_allclose(obs[-4:], np.zeros(4), atol=1e-12)


def test_shadow_only_insolvency_does_not_apply_hybrid_penalty() -> None:
    env = _residual_env(reward_scale=1.0, hybrid_insolvency_penalty=5.0)
    env.reset(options={"start_idx": 72})
    env.shadow.portfolio_value = 1e-12

    _, reward, terminated, _, info = env.step(np.zeros(2))

    assert terminated is True
    assert info["shadow_insolvent"] is True
    assert info["hybrid_insolvent"] is False
    assert info["rollout_valid"] is False
    assert reward > -5.0


def test_excess_log_return_series_is_the_paired_statistic() -> None:
    hybrid = np.asarray([0.50, -0.20], dtype=np.float64)
    shadow = np.asarray([0.40, -0.10], dtype=np.float64)
    expected = np.log1p(hybrid) - np.log1p(shadow)

    actual = relative_evaluation.excess_log_return_series(hybrid, shadow)

    np.testing.assert_allclose(actual, expected)
    assert not np.allclose(actual, hybrid - shadow)


def test_direct_turnover_penalty_is_invariant_to_reward_scale() -> None:
    fs = _feature_set(n_bars=12, n_symbols=1)

    def run(scale: float) -> float:
        env = PortfolioTradingEnv(
            fs,
            episode_bars=6,
            fee_rate=0.0,
            spread_rate=0.0,
            impact_rate=0.0,
            min_trade_delta=0.0,
            post_processor=make_legacy_processor(0.0),
            turnover_penalty_rate=0.001,
            reward_scale=scale,
        )
        env.reset(options={"start_idx": 0})
        _, reward, _, _, _ = env.step(np.asarray([0.5]))
        return reward

    reward_100 = run(100.0)
    reward_200 = run(200.0)

    assert reward_100 == pytest.approx(-0.05)
    assert reward_200 == pytest.approx(-0.10)
    assert reward_100 / 100.0 == pytest.approx(reward_200 / 200.0)


def test_direct_step_advances_one_complete_decision_interval() -> None:
    fs = _feature_set(n_bars=12, n_symbols=1, alternating=True)
    env = PortfolioTradingEnv(
        fs,
        episode_bars=6,
        decision_every=3,
        fee_rate=0.0,
        spread_rate=0.0,
        impact_rate=0.0,
        min_trade_delta=0.0,
        post_processor=make_legacy_processor(0.0),
        turnover_penalty_rate=0.0,
    )
    env.reset(options={"start_idx": 0})

    _, _, _, _, info = env.step(np.asarray([1.0]))

    assert env.t == 3
    assert info["bars_advanced"] == 3
    assert info["decision_step_index"] == 1
    assert len(env._returns_history) == 3
    assert env.n_trades == 1


def test_dsr_requires_explicit_research_mode_and_exposes_state() -> None:
    fs = _feature_set(n_bars=12, n_symbols=1, alternating=True)
    with pytest.raises(ValueError, match="experimental"):
        PortfolioTradingEnv(fs, use_dsr=True)

    env = PortfolioTradingEnv(
        fs,
        episode_bars=6,
        fee_rate=0.0,
        spread_rate=0.0,
        impact_rate=0.0,
        min_trade_delta=0.0,
        post_processor=make_legacy_processor(0.0),
        experimental_dsr=True,
        dsr_clip=0.25,
        reward_scale=1.0,
    )
    obs, _ = env.reset(options={"start_idx": 0})
    assert env.n_global == fs.global_features.shape[1] + 5
    np.testing.assert_allclose(obs[-2:], np.zeros(2), atol=1e-12)

    for action in (np.asarray([1.0]), np.asarray([-1.0]), np.asarray([1.0])):
        obs, reward, terminated, truncated, _ = env.step(action)
        assert abs(reward) <= 0.25
        if terminated or truncated:
            break
    assert np.isfinite(obs).all()


def test_experimental_dsr_disqualifies_release_intent() -> None:
    args = SimpleNamespace(
        force=False,
        skip_p0=False,
        skip_wf=False,
        skip_gate=False,
        experimental_dsr=True,
    )
    reasons = production_pipeline.release_disqualifying_reasons(args)
    assert "experimental_dsr" in reasons
