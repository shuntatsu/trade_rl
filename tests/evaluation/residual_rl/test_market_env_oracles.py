"""G2 oracles for the synthetic residual-PPO environment (no PPO runtime needed)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from trade_rl.evaluation.residual_rl.env import (
    EnvConfig,
    ResidualAllocatorEnv,
    apply_residual,
)
from trade_rl.evaluation.residual_rl.evaluate import env_config
from trade_rl.evaluation.residual_rl.market import (
    NS_PER_DAY,
    MarketSpec,
    Regime,
    SyntheticMarket,
    generate_market,
)
from trade_rl.evaluation.residual_rl.protocol import PROTOCOL_V1
from trade_rl.strategies.allocation.cost_aware import CostAwareAllocationPolicy
from trade_rl.strategies.portfolio import (
    PortfolioAccountSnapshot,
    PortfolioDecisionSnapshot,
    PortfolioRiskLimits,
)

# A short horizon keeps the allocator-backed oracles fast; the mechanism is identical.
SPEC = MarketSpec(horizon_days=40)
M_IDLE, M_HALF, M_ONE, M_BOOST = 0, 1, 2, 3  # indexes into (0, 0.5, 1, 1.5)


def _config(regime: Regime = Regime.PLANTED_ALPHA, **kwargs: float) -> EnvConfig:
    base = env_config(PROTOCOL_V1, regime, **kwargs)
    return EnvConfig(
        market=SPEC,
        regime=regime,
        allocator=base.allocator,
        costs=base.costs,
        scales=base.scales,
        multipliers=base.multipliers,
        reward_scale=base.reward_scale,
    )


def _run(
    config: EnvConfig, market: SyntheticMarket, actions: list[int]
) -> tuple[ResidualAllocatorEnv, list[np.ndarray], list[float]]:
    env = ResidualAllocatorEnv(config, (market.seed,))
    obs, _ = env.reset(options={"market": market})
    observations = [obs]
    rewards: list[float] = []
    for action in actions:
        obs, reward, terminated, truncated, _ = env.step(action)
        observations.append(obs)
        rewards.append(reward)
        assert not terminated
        if truncated:
            break
    return env, observations, rewards


def _full_actions(value: int) -> list[int]:
    return [value] * SPEC.horizon_days


def _random_actions(seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    return [int(v) for v in rng.integers(0, 4, size=SPEC.horizon_days)]


# -- market ------------------------------------------------------------------


def test_market_is_deterministic_and_regimes_share_the_random_stream() -> None:
    first = generate_market(SPEC, Regime.PLANTED_ALPHA, 7)
    again = generate_market(SPEC, Regime.PLANTED_ALPHA, 7)
    other = generate_market(SPEC, Regime.PLANTED_ALPHA, 8)
    zero = generate_market(SPEC, Regime.ZERO_ALPHA, 7)
    for name in ("signal", "mu", "returns", "prices"):
        assert np.array_equal(getattr(first, name), getattr(again, name))
    assert not np.array_equal(first.returns, other.returns)
    assert np.array_equal(first.mu, zero.mu)  # the allocator sees the same forecast
    assert np.array_equal(first.shock, zero.shock)
    assert np.all(zero.alpha == 0.0)
    assert np.array_equal(first.alpha, SPEC.signal_scale * first.signal)
    assert np.array_equal(
        first.returns, first.alpha + np.asarray(SPEC.daily_vol) * first.shock
    )
    assert np.all(first.prices > 0.0)


def test_planted_signal_predicts_returns_and_zero_regime_does_not() -> None:
    spec = MarketSpec(horizon_days=1500)
    planted = generate_market(spec, Regime.PLANTED_ALPHA, 1)
    zero = generate_market(spec, Regime.ZERO_ALPHA, 1)

    def correlation(market: SyntheticMarket) -> float:
        return float(np.corrcoef(market.mu.ravel(), market.returns.ravel())[0, 1])

    assert correlation(planted) > 0.07  # theory: ~0.10 at these constants
    assert abs(correlation(zero)) < 0.05


def test_market_spec_rejects_invalid_constants() -> None:
    with pytest.raises(ValueError):
        MarketSpec(n_assets=2, daily_vol=(0.01, 0.01))
    with pytest.raises(ValueError):
        MarketSpec(warmup_days=10)
    with pytest.raises(ValueError):
        MarketSpec(signal_persistence=1.0)


# -- causality (no look-ahead) ------------------------------------------------


@pytest.mark.parametrize("decision", [5, 20, 33])
def test_observation_is_causal_in_market_data(decision: int) -> None:
    market = generate_market(SPEC, Regime.PLANTED_ALPHA, 11)
    donor = generate_market(SPEC, Regime.PLANTED_ALPHA, 12)
    day = SPEC.warmup_days + decision
    # Same history up to and including the information set of decision ``day``;
    # the return realised on ``day`` and every later input are different.
    counterfactual = market.replace_from(day, shock=donor.shock).replace_from(
        day + 1, signal=donor.signal, forecast_noise=donor.forecast_noise
    )
    actions = _random_actions(3)
    config = _config()
    _, base_obs, base_rewards = _run(config, market, actions)
    _, alt_obs, alt_rewards = _run(config, counterfactual, actions)
    for index in range(decision + 1):
        assert np.array_equal(base_obs[index], alt_obs[index]), index
    for index in range(decision):
        assert base_rewards[index] == alt_rewards[index]
    # Negative controls: the oracle can see a difference when one is allowed.
    assert not np.array_equal(base_obs[decision + 1], alt_obs[decision + 1])
    assert base_rewards[decision] != alt_rewards[decision]


def test_observation_depends_on_the_current_forecast_only_through_mu() -> None:
    market = generate_market(SPEC, Regime.PLANTED_ALPHA, 11)
    day = SPEC.warmup_days + 10
    donor = generate_market(SPEC, Regime.PLANTED_ALPHA, 12)
    changed = market.replace_from(day, forecast_noise=donor.forecast_noise)
    config = _config()
    _, base_obs, _ = _run(config, market, _random_actions(4))
    _, alt_obs, _ = _run(config, changed, _random_actions(4))
    n = SPEC.n_assets
    assert not np.array_equal(base_obs[10][:n], alt_obs[10][:n])  # mu block differs
    assert np.array_equal(base_obs[10][n : 2 * n], alt_obs[10][n : 2 * n])  # weights
    assert np.array_equal(base_obs[9], alt_obs[9])


# -- ledger conservation -------------------------------------------------------


def _reference_ledger(
    market: SyntheticMarket, executed: list[np.ndarray], one_way: float
) -> list[float]:
    """Separate value ledger driven only by market returns and executed weights."""

    values = np.zeros(market.spec.n_assets)
    cash = 1.0
    equities: list[float] = []
    for step, weights in enumerate(executed):
        day = market.spec.warmup_days + step
        equity = cash + float(values.sum())
        trade = weights * equity - values
        cost = one_way * float(np.abs(trade).sum())
        values = values + trade
        cash = cash - float(trade.sum()) - cost
        values = values * (1.0 + market.returns[day])
        equities.append(cash + float(values.sum()))
    return equities


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_reward_equals_the_ledger_equity_change(seed: int) -> None:
    market = generate_market(SPEC, Regime.PLANTED_ALPHA, 20 + seed)
    config = _config()
    env, _, rewards = _run(config, market, _random_actions(seed))
    reference = _reference_ledger(market, env.executed, config.costs.one_way)
    previous = 1.0
    for step, (reward, equity) in enumerate(zip(rewards, reference, strict=True)):
        assert reward == pytest.approx(math.log(equity / previous), abs=1e-12)
        assert env.ledger[step]["equity_after"] == pytest.approx(equity, rel=1e-12)
        previous = equity
    assert sum(rewards) == pytest.approx(math.log(reference[-1]), abs=1e-11)
    assert env.equity == pytest.approx(reference[-1], rel=1e-12)


def test_pnl_minus_costs_conserves_equity() -> None:
    market = generate_market(SPEC, Regime.PLANTED_ALPHA, 31)
    config = _config()
    env, _, _ = _run(config, market, _random_actions(9))
    gross = 0.0
    costs = 0.0
    values = np.zeros(SPEC.n_assets)
    for step, row in enumerate(env.ledger):
        trade = env.executed[step] * row["equity_before"] - values
        values = values + trade
        gross += float(values @ market.returns[SPEC.warmup_days + step])
        values = values * (1.0 + market.returns[SPEC.warmup_days + step])
        costs += row["fee"] + row["spread_cost"]
    assert env.equity - 1.0 == pytest.approx(gross - costs, abs=1e-11)


def test_reward_scale_only_scales_the_reward() -> None:
    market = generate_market(SPEC, Regime.PLANTED_ALPHA, 5)
    plain, _, plain_rewards = _run(_config(), market, _random_actions(1))
    scaled, _, scaled_rewards = _run(
        _config(reward_scale=100.0), market, _random_actions(1)
    )
    assert plain.ledger == scaled.ledger
    assert scaled_rewards == [100.0 * r for r in plain_rewards]


def test_costs_are_fee_plus_half_spread_on_traded_notional() -> None:
    market = generate_market(SPEC, Regime.PLANTED_ALPHA, 6)
    env, _, _ = _run(_config(), market, _full_actions(M_ONE))
    traded_rows = [row for row in env.ledger if row["traded_value"] > 0.0]
    assert traded_rows
    for row in traded_rows:
        assert row["fee"] == pytest.approx(0.0005 * row["traded_value"], rel=1e-12)
        assert row["spread_cost"] == pytest.approx(
            0.0001 * row["traded_value"], rel=1e-12
        )


# -- residual semantics --------------------------------------------------------


def test_zero_multiplier_never_trades_and_keeps_equity_exactly() -> None:
    market = generate_market(SPEC, Regime.PLANTED_ALPHA, 3)
    env, _, rewards = _run(_config(), market, _full_actions(M_IDLE))
    assert env.equity == 1.0
    assert all(row["traded_value"] == 0.0 for row in env.ledger)
    assert all(reward == 0.0 for reward in rewards)


def _independent_allocator_alone(
    market: SyntheticMarket, config: EnvConfig
) -> tuple[list[float], list[tuple[float, ...]]]:
    spec = market.spec
    n = spec.n_assets
    warm = spec.warmup_days

    class Forecast:
        def expected_simple_returns(
            self, snapshot: PortfolioDecisionSnapshot
        ) -> tuple[float, ...]:
            return tuple(float(v) for v in market.mu[snapshot.index])

    allocator = config.allocator
    policy = CostAwareAllocationPolicy(
        symbols=spec.symbols,
        forecast_source=Forecast(),
        cost_rates=(config.costs.fee_rate + 0.5 * config.costs.spread_rate,) * n,
        gamma=allocator.gamma,
        ewma_halflife_days=allocator.ewma_halflife_days,
        shrinkage=allocator.shrinkage,
        w_max=allocator.w_max,
        gross_max=allocator.gross_max,
        min_trade_fraction=allocator.min_trade_fraction,
        min_history_periods=allocator.min_history_periods,
        warmup_timestamps_ns=tuple(d * NS_PER_DAY for d in range(warm)),
        warmup_mark_prices=tuple(
            tuple(map(float, market.prices[d])) for d in range(warm)
        ),
    )
    units = np.zeros(n)
    cash = 1.0
    equities: list[float] = []
    targets: list[tuple[float, ...]] = []
    for day in range(warm, spec.total_days):
        prices = market.prices[day]
        equity = cash + float(units @ prices)
        weights = tuple(float(v) for v in units * prices / equity)
        account = PortfolioAccountSnapshot(
            cash=cash,
            portfolio_value=equity,
            peak_value=max(equity, 1.0),
            max_drawdown=0.0,
            exact_quantities=tuple(repr(float(q)) for q in units),
            quantities=tuple(float(q) for q in units),
            weights=weights,
            mark_prices=tuple(float(p) for p in prices),
            contract_multipliers=(1.0,) * n,
            turnover_total=0.0,
            total_cost=0.0,
            funding_pnl=0.0,
            borrow_cost=0.0,
            margin_used=0.0,
            maintenance_margin=0.0,
            maintenance_requirement=0.0,
            margin_deficit=0.0,
            insolvent=False,
            termination_reason=None,
        )
        snapshot = PortfolioDecisionSnapshot(
            index=day,
            timestamp_ns=day * NS_PER_DAY,
            agent_stop_index=spec.total_days,
            remaining_decisions=spec.total_days - day,
            symbols=spec.symbols,
            feature_names=(),
            global_feature_names=(),
            features=((),) * n,
            feature_available=((),) * n,
            feature_staleness=((),) * n,
            global_features=(),
            global_feature_available=(),
            account=account,
            active_orders=(),
            risk=PortfolioRiskLimits(1.0, 0.5, None, 1.0, 1.0, False, 1e-9),
        )
        target = np.asarray(policy.decide_portfolio(snapshot).target_weights)
        targets.append(tuple(float(v) for v in target))
        trade_units = target * equity / prices - units
        notional = trade_units * prices
        cost = config.costs.one_way * float(np.abs(notional).sum())
        units = units + trade_units
        cash = cash - float(notional.sum()) - cost
        equities.append(cash + float(units @ market.prices[day + 1]))
    return equities, targets


@pytest.mark.parametrize("seed", [40, 41])
def test_multiplier_one_reproduces_allocator_alone(seed: int) -> None:
    market = generate_market(SPEC, Regime.PLANTED_ALPHA, seed)
    config = _config()
    env, _, _ = _run(config, market, _full_actions(M_ONE))
    # Executed weights are *exactly* the allocator proposal at every step.
    for target, executed in zip(env.targets, env.executed, strict=True):
        assert np.array_equal(target, executed)
    # And a separately written quantity ledger around a fresh allocator agrees.
    equities, targets = _independent_allocator_alone(market, config)
    for step, target in enumerate(targets):
        assert env.targets[step] == pytest.approx(np.asarray(target), abs=1e-9)
    ledger_equity = [row["equity_after"] for row in env.ledger]
    assert ledger_equity == pytest.approx(equities, rel=1e-9)
    assert max(abs(a - b) for a, b in zip(ledger_equity, equities, strict=True)) < 1e-9


def test_apply_residual_is_exact_at_zero_and_one_and_stays_in_the_box() -> None:
    rng = np.random.default_rng(0)
    for _ in range(200):
        current = rng.uniform(-0.25, 0.25, size=4)
        target = rng.uniform(-0.5, 0.5, size=4)
        target *= min(1.0, 1.0 / float(np.abs(target).sum()))
        assert np.array_equal(
            apply_residual(current, target, 1.0, w_max=0.5, gross_max=1.0), target
        )
        assert np.array_equal(
            apply_residual(current, target, 0.0, w_max=0.5, gross_max=1.0), current
        )
        for multiplier in (0.5, 1.5, 1.2):
            executed = apply_residual(
                current, target, multiplier, w_max=0.5, gross_max=1.0
            )
            assert np.all(np.abs(executed) <= 0.5 + 1e-12)
            assert float(np.abs(executed).sum()) <= 1.0 + 1e-9
        half = apply_residual(current, target, 0.5, w_max=0.5, gross_max=1.0)
        assert np.allclose(half, 0.5 * (current + target))


def test_throttle_halves_the_trade_when_unprojected() -> None:
    market = generate_market(SPEC, Regime.PLANTED_ALPHA, 3)
    env, _, _ = _run(_config(), market, _full_actions(M_HALF))
    first_target = env.targets[0]
    assert np.allclose(env.executed[0], 0.5 * first_target)  # starts from cash


def test_allocator_never_fails_closed_on_the_synthetic_markets() -> None:
    for regime in (Regime.PLANTED_ALPHA, Regime.ZERO_ALPHA):
        market = generate_market(SPEC, regime, 50)
        env, _, _ = _run(_config(regime), market, _full_actions(M_ONE))
        assert env._policy is not None
        outcomes = {record.outcome for record in env._policy.records}
        assert "fail_closed" not in outcomes
        assert "rebalance" in outcomes


def test_environment_is_deterministic_and_obeys_the_gym_contract() -> None:
    market = generate_market(SPEC, Regime.PLANTED_ALPHA, 60)
    actions = _random_actions(7)
    first, first_obs, _ = _run(_config(), market, actions)
    second, second_obs, _ = _run(_config(), market, actions)
    assert first.ledger == second.ledger
    assert all(np.array_equal(a, b) for a, b in zip(first_obs, second_obs, strict=True))
    env = ResidualAllocatorEnv(_config(), (60,))
    obs, info = env.reset(seed=123)
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert info["market_seed"] == 60
    with pytest.raises(ValueError):
        env.step(9)


def test_episode_truncates_at_the_horizon() -> None:
    market = generate_market(SPEC, Regime.PLANTED_ALPHA, 61)
    env = ResidualAllocatorEnv(_config(), (61,))
    env.reset(options={"market": market})
    for step in range(SPEC.horizon_days):
        _, _, terminated, truncated, _ = env.step(M_ONE)
        assert not terminated
        assert truncated is (step == SPEC.horizon_days - 1)
