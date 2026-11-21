"""Synthetic tests of the learning-lab overlay environment and the fixed-stride baselines."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from tests.evaluation import learning_lab_support as support
from tests.strategies.test_agy_anchor_policy import _snapshot
from trade_rl.evaluation.agy_anchor_oos import protocol
from trade_rl.evaluation.learning_lab import baselines
from trade_rl.evaluation.learning_lab import overlay_env as oe
from trade_rl.strategies.agy_anchor.policy import risk_scale_of
from trade_rl.strategies.portfolio import PortfolioTargetProposal


def _fixed_stride_policy(stride: int) -> Callable[[int, Any], int]:
    """Closed-loop policy that trades fully on every ``stride // 6``-th decision."""
    every = stride // 6
    return lambda step, _obs: 3 if step % every == 0 else 0


def _baseline(
    st: Any, method: str, *, start: int = support.START, stop: int = support.STOP
) -> Any:
    return baselines.run_baseline(
        method,
        st.dataset,
        st.signals,
        start_index=start,
        stop_index=stop,
        execution_cost=protocol.scenario_cost("BASE"),
        risk_config=protocol.risk_config(),
        initial_capital=support.CAPITAL,
    )


# ------------------------------------------------------------------ causality


def test_future_bars_do_not_change_past_observations_actions_or_rewards() -> None:
    st = support.stage()
    cut = support.START + 6 * 10
    future = support.perturb_future(st, cut=cut, seed=99)

    def policy(step: int, obs: np.ndarray) -> int:
        # closed loop: the action is a pure function of the observation
        return int(abs(obs[0] * 7.0 + obs[2] * 50.0 + step)) % 4

    obs_a, rew_a, act_a, _ = support.run_episode(support.make_env(st), policy)
    obs_b, rew_b, act_b, _ = support.run_episode(support.make_env(future), policy)

    # decision k sits at row START + 6 k; its observation reads rows <= that row
    through = 10  # decisions 0..10 are at rows <= cut
    for k in range(through + 1):
        np.testing.assert_array_equal(obs_a[k], obs_b[k])
    # step k covers rows START+6k .. START+6k+6; steps 0..9 end at or before the cut
    assert act_a[:through] == act_b[:through]
    assert rew_a[:through] == rew_b[:through]
    # the perturbation is real: the future does differ
    assert rew_a != rew_b


def test_the_observation_has_the_documented_shape_and_bounds() -> None:
    env = support.make_env(support.stage())
    observation, _ = env.reset(seed=0)
    assert observation.shape == (len(oe.OBSERVATION_FIELDS),)
    assert observation.dtype == np.float32
    assert env.observation_space.contains(observation)
    assert oe.OBSERVATION_FIELDS[-1] == "days_since_last_trade"


# --------------------------------------------------------------------- reward


def test_reward_is_the_engine_equity_delta_net_of_costs() -> None:
    env = support.make_env(support.stage())
    rng = np.random.default_rng(5)
    env.reset(seed=0)
    total = 0.0
    done = False
    previous = support.CAPITAL
    info: dict[str, Any] = {}
    while not done:
        _, reward, done, _, info = env.step(int(rng.integers(0, 4)))
        total += reward
        assert reward * support.CAPITAL == pytest.approx(info["equity"] - previous)
        assert info["equity_delta"] == pytest.approx(info["equity"] - previous)
        previous = info["equity"]
    assert total * support.CAPITAL == pytest.approx(env.final_equity - support.CAPITAL)
    # the engine's per-bar net returns compound to the same equity
    compounded = support.CAPITAL * float(
        np.prod(1.0 + np.asarray(env.episode_bar_returns))
    )
    assert compounded == pytest.approx(env.final_equity, rel=1e-9)
    assert info["total_cost"] > 0.0  # costs are inside the equity change


def test_every_decision_full_trade_equals_the_stride_6_replay_exactly() -> None:
    st = support.stage()
    env = support.make_env(st)
    _, rewards, _, _ = support.run_episode(env, lambda *_: 3)
    base = _baseline(st, "BASE-STRIDE-6")
    assert env.final_equity == base.final_equity
    assert sum(rewards) * support.CAPITAL == pytest.approx(
        base.final_equity - support.CAPITAL
    )


# ----------------------------------------------------- fixed-stride equivalence


@pytest.mark.parametrize("stride", baselines.FIXED_STRIDES)
def test_full_trade_every_stride_reproduces_the_fixed_stride_baseline(
    stride: int,
) -> None:
    st = support.stage()
    env = support.make_env(st)
    support.run_episode(env, _fixed_stride_policy(stride))
    base = _baseline(st, baselines.stride_method_id(stride))
    assert env.episode_bar_returns == base.bar_returns  # bit-identical
    assert env.final_equity == base.final_equity
    book = env.session.book
    assert float(book.total_cost) == base.total_cost
    assert float(book.turnover_total) == base.turnover_total


def test_stride_equivalence_also_holds_through_the_drawdown_ramp() -> None:
    st = support.stage(n_bars=260, seed=3, volatility=0.03, drift=-0.001)
    start, stop = 20, 250
    env = support.make_env(st, start=start, stop=stop)
    support.run_episode(env, _fixed_stride_policy(18))
    base = _baseline(st, "BASE-STRIDE-18", start=start, stop=stop)
    equity = support.CAPITAL * np.cumprod(1.0 + np.asarray(base.bar_returns))
    drawdown = float((1.0 - equity / np.maximum.accumulate(equity)).max())
    assert drawdown > protocol.RISK_DRAWDOWN_START  # the risk ramp was active
    assert env.episode_bar_returns == base.bar_returns


# ----------------------------------------------------------------------- hold


def test_zero_fraction_holds_the_flat_book_and_pays_nothing() -> None:
    env = support.make_env(support.stage())
    _, rewards, _, info = support.run_episode(env, lambda *_: 0)
    assert env.final_equity == support.CAPITAL
    assert all(r == 0.0 for r in rewards)
    assert info["turnover_total"] == 0.0
    assert info["total_cost"] == 0.0


def test_zero_fraction_after_a_trade_keeps_the_exact_quantities() -> None:
    env = support.make_env(support.stage())
    env.reset(seed=0)
    env.step(3)  # build the book
    quantities = tuple(float(q) for q in env.session.book.quantities)
    done = False
    held_steps = 0
    while not done:
        _, _, done, _, _ = env.step(0)
        if not done:
            held_steps += 1
            assert tuple(float(q) for q in env.session.book.quantities) == quantities
    assert held_steps > 5
    assert quantities != (0.0,) * len(quantities)


def test_a_partial_fraction_trades_less_than_a_full_one() -> None:
    turnover: list[float] = []
    for action in (0, 1, 2, 3):
        env = support.make_env(support.stage())
        env.reset(seed=0)
        _, _, _, _, info = env.step(action)
        turnover.append(float(info["turnover_total"]))
    assert turnover[0] == 0.0
    assert turnover == sorted(turnover) and turnover[1] > 0.0
    assert turnover[1] < turnover[2] < turnover[3]


def test_blend_moves_the_effective_weights_by_the_fraction_and_keeps_holds() -> None:
    current = (0.1, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0)
    snapshot = _snapshot(5, weights=current)
    target = (0.3, 0.2, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0)
    proposal = PortfolioTargetProposal(target_weights=target)
    half = oe.blend_proposal(proposal, snapshot, 0.5)
    np.testing.assert_allclose(
        half.target_weights, np.asarray(current) + 0.5 * (np.asarray(target) - current)
    )
    assert oe.blend_proposal(proposal, snapshot, 1.0) is proposal
    held = PortfolioTargetProposal(
        target_weights=(0.1, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0),
        hold_current_quantity_symbols=(0, 2),
    )
    blended = oe.blend_proposal(held, snapshot, 0.25)
    assert blended.hold_current_quantity_symbols == (0, 2)
    assert blended.target_weights[0] == 0.1 and blended.target_weights[2] == 0.2
    assert blended.target_weights[1] == pytest.approx(0.05)


def test_blend_applies_the_risk_scale_exactly_once() -> None:
    dead = _snapshot(5, value=0.75, peak=1.0)  # drawdown 0.25: scale 0
    full = PortfolioTargetProposal(target_weights=(0.3,) * 8)
    assert oe.blend_proposal(full, dead, 0.5) is full  # scale 0: nothing to blend
    mid = _snapshot(5, value=0.85, peak=1.0)  # drawdown 0.15: scale 0.5
    scale = oe.risk_scale(mid)
    assert scale == pytest.approx(0.5)
    blended = oe.blend_proposal(
        PortfolioTargetProposal(target_weights=(0.4,) * 8), mid, 0.5
    )
    # risk multiplies by `scale`: the executed weight is current + f * (scale * target)
    assert scale * blended.target_weights[0] == pytest.approx(0.5 * scale * 0.4)


def test_the_local_risk_scale_matches_the_anchors() -> None:
    for value in (1.0, 0.95, 0.9, 0.85, 0.8, 0.7):
        snapshot = _snapshot(3, value=value, peak=1.0)
        assert oe.risk_scale(snapshot) == risk_scale_of(snapshot)


# ------------------------------------------------------------- continuous mode


def test_continuous_endpoints_equal_the_discrete_endpoints_and_clip() -> None:
    st = support.stage()
    discrete = support.make_env(st)
    support.run_episode(discrete, _fixed_stride_policy(18))
    continuous = support.make_env(st, mode=oe.ACTION_CONTINUOUS)
    support.run_episode(
        continuous,
        lambda step, _o: np.array([1.0 if step % 3 == 0 else 0.0], dtype=np.float32),
    )
    assert continuous.episode_bar_returns == discrete.episode_bar_returns
    clipped = support.make_env(st, mode=oe.ACTION_CONTINUOUS)
    support.run_episode(
        clipped,
        lambda step, _o: np.array([7.0 if step % 3 == 0 else -2.0], dtype=np.float32),
    )
    assert clipped.episode_bar_returns == discrete.episode_bar_returns
    assert continuous.action_space.shape == (1,)


def test_a_continuous_midpoint_trades_between_hold_and_full() -> None:
    values = {}
    for level in (0.0, 0.5, 1.0):
        env = support.make_env(support.stage(), mode=oe.ACTION_CONTINUOUS)
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.array([level], dtype=np.float32))
        values[level] = info["turnover_total"]
    assert values[0.0] == 0.0 < values[0.5] < values[1.0]


# ----------------------------------------------------------- randomization hooks


def _cost_of(perturbation: oe.PerturbationSource | None) -> float:
    env = support.make_env(support.stage(), perturbation=perturbation)
    support.run_episode(env, lambda *_: 3)
    return float(env.session.book.total_cost)


def test_the_cost_multiplier_hook_scales_the_engine_costs() -> None:
    base = _cost_of(None)
    doubled = _cost_of(lambda _rng: oe.OverlayPerturbation(cost_multiplier=2.0))
    free = _cost_of(lambda _rng: oe.OverlayPerturbation(cost_multiplier=0.0))
    assert doubled > 1.9 * base
    assert free < 0.05 * base


def test_the_delay_and_shock_hooks_change_the_engine_outcome_reproducibly() -> None:
    st = support.stage()
    plain = support.make_env(st)
    support.run_episode(plain, lambda *_: 3)
    delayed = support.make_env(
        st, perturbation=lambda _r: oe.OverlayPerturbation(extra_latency_bars=1)
    )
    support.run_episode(delayed, lambda *_: 3)
    assert delayed.session.execution_cost.order_latency_bars == 1
    assert delayed.final_equity != plain.final_equity

    def shock(_rng: np.random.Generator) -> oe.OverlayPerturbation:
        return oe.OverlayPerturbation(return_shock_sigma=0.01, shock_seed=4)

    shocked = support.make_env(st, perturbation=shock)
    support.run_episode(shocked, lambda *_: 3)
    assert shocked.final_equity != plain.final_equity
    again = support.make_env(st, perturbation=shock)
    support.run_episode(again, lambda *_: 3)
    assert again.episode_bar_returns == shocked.episode_bar_returns


def test_the_return_shock_leaves_rows_up_to_the_start_untouched() -> None:
    st = support.stage()
    shocked = oe.shock_returns(st.dataset, sigma=0.02, seed=1, from_index=50)
    np.testing.assert_array_equal(shocked.close[:51], st.dataset.close[:51])
    assert not np.array_equal(shocked.close[52:], st.dataset.close[52:])
    assert oe.shock_returns(st.dataset, sigma=0.0, seed=1, from_index=50) is st.dataset


def test_uniform_randomization_is_a_seeded_perturbation_source() -> None:
    source = oe.UniformRandomization(
        cost_multiplier=(1.0, 2.0),
        extra_latency_bars=(0, 1),
        return_shock_sigma=(0.0, 0.01),
    )
    first = source(np.random.default_rng(3))
    assert first == source(np.random.default_rng(3))
    assert 1.0 <= first.cost_multiplier <= 2.0
    one = support.make_env(support.stage(), perturbation=source)
    two = support.make_env(support.stage(), perturbation=source)
    assert (
        one.reset(seed=11)[1]["perturbation"] == two.reset(seed=11)[1]["perturbation"]
    )


def test_an_adversary_cannot_leave_its_bounds() -> None:
    with pytest.raises(oe.OverlayError):
        oe.OverlayPerturbation(cost_multiplier=oe.MAX_COST_MULTIPLIER + 1)
    with pytest.raises(oe.OverlayError):
        oe.OverlayPerturbation(extra_latency_bars=oe.MAX_EXTRA_LATENCY_BARS + 1)
    with pytest.raises(oe.OverlayError):
        oe.OverlayPerturbation(return_shock_sigma=-0.1)
    env = support.make_env(
        support.stage(),
        perturbation=lambda _r: "adversary",  # type: ignore[arg-type,return-value]
    )
    with pytest.raises(TypeError):
        env.reset(seed=0)


# --------------------------------------------------------------------- misuse


def test_actions_and_lifecycle_are_validated() -> None:
    env = support.make_env(support.stage())
    with pytest.raises(oe.OverlayError):
        env.step(0)  # before reset
    env.reset(seed=0)
    for bad in (4, -1, 1.5, True):
        with pytest.raises(oe.OverlayError):
            env.step(bad)
    support.run_episode(env, lambda *_: 0)
    with pytest.raises(oe.OverlayError):
        env.step(0)  # episode finished
    with pytest.raises(oe.OverlayError):
        env.reset(options={"x": 1})


@pytest.mark.parametrize("bad_symbol", (-1, True, 1.0))
def test_negative_or_nonnative_hold_symbols_are_already_rejected_by_proposal(
    bad_symbol: object,
) -> None:
    # The contract is in PortfolioTargetProposal, not duplicated in the overlay.
    with pytest.raises((TypeError, ValueError), match="hold symbol"):
        PortfolioTargetProposal(
            target_weights=(0.0,) * 8,
            hold_current_quantity_symbols=(bad_symbol,),  # type: ignore[arg-type]
        )


def test_observation_time_uses_native_bar_duration_not_decision_stride() -> None:
    st = support.stage()
    env = oe.ExecutionOverlayEnv(
        st.dataset,
        start_index=support.START,
        stop_index=support.STOP,
        base_factory=baselines.anchor_stack_factory(
            st.signals, start_index=support.START
        ),
        execution_cost=protocol.scenario_cost("BASE"),
        risk_config=protocol.risk_config(),
        config=oe.OverlayConfig(
            decision_every_bars=18, initial_capital=support.CAPITAL
        ),
    )
    assert env._bars_per_day == 6
    env.reset(seed=0)
    observation, _, done, _, _ = env.step(0)
    assert not done
    index = oe.OBSERVATION_FIELDS.index("days_since_last_trade")
    assert observation[index] == pytest.approx(3.0)


def test_a_hold_that_moves_the_book_is_refused() -> None:
    snapshot = _snapshot(3, weights=(0.1,) + (0.0,) * 7)
    bad = PortfolioTargetProposal(
        target_weights=(0.2,) + (0.0,) * 7, hold_current_quantity_symbols=(0,)
    )
    with pytest.raises(oe.OverlayError):
        oe.ExecutionOverlayEnv._validated(bad, snapshot)


def test_the_cash_baseline_never_trades_and_unknown_baselines_are_refused() -> None:
    st = support.stage()
    result = _baseline(st, "BASE-CASH")
    assert result.final_equity == support.CAPITAL
    assert result.turnover_total == 0.0
    assert set(result.bar_returns) == {0.0}
    with pytest.raises(ValueError):
        _baseline(st, "BASE-STRIDE-7")
