from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.replay import (
    SharedCashPortfolioReplayResult,
    run_shared_portfolio_replay,
)
from trade_rl.simulation import ExecutionCostConfig
from trade_rl.strategies.allocation import solver as solver_module
from trade_rl.strategies.allocation.cost_aware import (
    CostAwareAllocationPolicy,
    ExpectedReturnSource,
    ewma_covariance,
)
from trade_rl.strategies.allocation.costs import (
    one_way_cost_rate,
    one_way_cost_rates,
)
from trade_rl.strategies.portfolio import (
    PortfolioDecisionSnapshot,
    PortfolioTargetProposal,
)

SYMBOLS = ("AAAUSDT", "BBBUSDT", "CCCUSDT")
CAPITAL = 10_000.0
N_DAYS = 90
BARS = N_DAYS * 24


def _dataset(
    daily_returns: np.ndarray,
    *,
    symbols: tuple[str, ...] = SYMBOLS,
) -> MarketDataset:
    """Hourly bars whose 00:00 UTC marks follow ``daily_returns`` exactly."""

    n_days, n_symbols = daily_returns.shape
    n_bars = n_days * 24 + 1
    daily_close = 100.0 * np.cumprod(1.0 + daily_returns, axis=0)
    daily_close = np.vstack((np.full((1, n_symbols), 100.0), daily_close))
    close = np.empty((n_bars, n_symbols))
    for bar in range(n_bars):
        day, hour = divmod(bar, 24)
        if hour == 0 or day >= n_days:
            close[bar] = daily_close[min(day, n_days)]
        else:
            fraction = hour / 24.0
            close[bar] = (
                daily_close[day] * (daily_close[day + 1] / daily_close[day]) ** fraction
            )
    open_price = np.vstack((close[0], close[:-1]))
    features = np.zeros((n_bars, n_symbols, 1), dtype=np.float32)
    return MarketDataset(
        dataset_id="d" * 64,
        symbols=symbols,
        timestamps=np.datetime64("2026-01-01T00:00:00", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=features,
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=open_price,
        high=np.maximum(open_price, close),
        low=np.minimum(open_price, close),
        close=close,
        volume=np.full((n_bars, n_symbols), 1e12),
        funding_rate=np.zeros((n_bars, n_symbols)),
        tradable=np.ones((n_bars, n_symbols), dtype=np.bool_),
        feature_available=np.ones((n_bars, n_symbols, 1), dtype=np.bool_),
        feature_names=("signal",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def _noise(seed: int = 7, scale: float = 0.02, n_symbols: int = 3) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, scale, size=(N_DAYS, n_symbols))


@dataclass
class _ConstantSource:
    mu: tuple[float, ...]

    def expected_simple_returns(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> tuple[float, ...]:
        return self.mu


class _CallableSource:
    def __init__(
        self, function: Callable[[PortfolioDecisionSnapshot], tuple[float, ...]]
    ) -> None:
        self._function = function

    def expected_simple_returns(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> tuple[float, ...]:
        return self._function(snapshot)


def _policy(
    source: ExpectedReturnSource,
    *,
    cost: float = 0.0006,
    **overrides: object,
) -> CostAwareAllocationPolicy:
    kwargs: dict[str, object] = {
        "symbols": SYMBOLS,
        "forecast_source": source,
        "cost_rates": (cost,) * len(SYMBOLS),
        "min_history_periods": 5,
        "min_trade_fraction": 0.005,
    }
    kwargs.update(overrides)
    return CostAwareAllocationPolicy(**kwargs)  # type: ignore[arg-type]


def _replay(
    dataset: MarketDataset,
    policy: CostAwareAllocationPolicy,
    *,
    execution_cost: ExecutionCostConfig | None = None,
    stop_day: int | None = None,
) -> SharedCashPortfolioReplayResult:
    stop = (stop_day if stop_day is not None else N_DAYS) * 24
    return run_shared_portfolio_replay(
        dataset,
        policy,
        start_index=0,
        stop_index=stop,
        initial_capital=CAPITAL,
        execution_cost=execution_cost or ExecutionCostConfig(),
        settle_terminal_position=False,
    )


def test_policy_satisfies_the_portfolio_policy_and_source_protocols() -> None:
    source = _ConstantSource((0.0, 0.0, 0.0))
    assert isinstance(source, ExpectedReturnSource)
    policy = _policy(source)
    assert callable(policy.decide_portfolio)
    assert policy.symbols == SYMBOLS


def test_zero_signal_market_with_fees_never_trades() -> None:
    dataset = _dataset(_noise())
    policy = _policy(_ConstantSource((0.0, 0.0, 0.0)))
    result = _replay(dataset, policy)
    assert result.book.turnover_total == 0.0
    assert result.book.total_cost == 0.0
    assert result.book.fill_count == 0
    assert result.book.portfolio_value == pytest.approx(CAPITAL)
    scheduled = [r for r in policy.records if r.outcome == "hold_no_trade"]
    assert scheduled, "history warm-up should end and the solver should run"
    assert all(r.outcome in {"hold_no_trade", "fail_closed"} for r in policy.records)


def test_strong_positive_signal_goes_long_and_profits_net_of_fees() -> None:
    drift = np.full((N_DAYS, 3), 0.01)
    drift += _noise(scale=0.002)
    dataset = _dataset(drift)
    policy = _policy(_ConstantSource((0.01, 0.01, 0.01)), gamma=5.0)
    result = _replay(dataset, policy, stop_day=N_DAYS - 1)
    assert result.book.total_cost > 0.0
    assert all(w > 0.0 for w in result.book.weights)
    assert result.book.portfolio_value > CAPITAL * 1.3  # fees do not erase the drift


def test_higher_cost_rates_shrink_turnover() -> None:
    returns = _noise(seed=11)

    def flip_flop(snapshot: PortfolioDecisionSnapshot) -> tuple[float, ...]:
        day = snapshot.timestamp_ns // (24 * 3_600 * 1_000_000_000)
        sign = 1.0 if day % 2 == 0 else -1.0
        return (0.0005 * sign, -0.0005 * sign, 0.00025 * sign)

    turnovers: list[float] = []
    for cost in (0.0, 0.0001, 0.0003, 0.002):
        policy = _policy(_CallableSource(flip_flop), cost=cost)
        result = _replay(_dataset(returns), policy)
        turnovers.append(float(result.book.turnover_total))
    assert turnovers[0] > turnovers[1] > turnovers[2] >= turnovers[3]
    assert turnovers[0] > 0.0


def test_covariance_makes_equal_signals_diversify() -> None:
    rng = np.random.default_rng(3)
    returns = rng.normal(0.0, 0.02, size=(N_DAYS, 3))
    policy = _policy(
        _ConstantSource((0.004, 0.004, 0.004)),
        cost=0.0,
        gamma=5.0,
        w_max=1.0,
        gross_max=1.0,
        shrinkage=0.0,
    )
    result = _replay(_dataset(returns), policy, stop_day=N_DAYS - 1)
    last_rebalance = [r for r in policy.records if r.outcome == "rebalance"][-1]
    assert last_rebalance.traded_symbols
    positive = np.asarray(result.book.weights)
    assert np.all(positive > 0.1)
    assert positive.max() < 0.7


def _correlated_returns(seed: int, correlation: float) -> np.ndarray:
    """Assets A and B share a common factor; asset C is independent."""

    rng = np.random.default_rng(seed)
    factor = rng.normal(0.0, 0.02, size=N_DAYS)
    own = np.sqrt(1.0 - correlation)
    shared = np.sqrt(correlation)
    return np.column_stack(
        (
            shared * factor + own * rng.normal(0.0, 0.02, size=N_DAYS),
            shared * factor + own * rng.normal(0.0, 0.02, size=N_DAYS),
            rng.normal(0.0, 0.02, size=N_DAYS),
        )
    )


def test_correlated_assets_are_diversified_with_the_full_covariance() -> None:
    """Equal signals: the offsetting off-diagonal terms must tilt away from A/B."""

    returns = _correlated_returns(seed=17, correlation=0.8)
    mu = (0.0005, 0.0005, 0.0005)
    decision_day = 80
    policy = _policy(
        _ConstantSource(mu),
        cost=0.0,
        gamma=5.0,
        w_max=5.0,
        gross_max=50.0,
        shrinkage=0.0,
        min_trade_fraction=0.0,
    )
    result = _replay(_dataset(returns), policy, stop_day=decision_day + 2)
    proposal = np.asarray(
        next(
            d.proposal_weights for d in result.decisions if d.index == decision_day * 24
        )
    )
    # Reference: the unconstrained optimum (gamma Sigma)^-1 mu of the same EWMA
    # covariance of the daily returns observed up to the decision day.
    sigma = ewma_covariance(
        returns[:decision_day],
        halflife_periods=60.0,
        shrinkage=0.0,
        variance_floor=1e-8,
    )
    expected = np.linalg.solve(5.0 * sigma, np.asarray(mu))
    assert np.all(np.abs(expected) < 5.0)  # no constraint binds in the reference
    np.testing.assert_allclose(proposal, expected, rtol=1e-6, atol=1e-8)
    # Economically: the independent asset receives the largest weight and a
    # diagonal-only covariance (which would give nearly equal weights) is rejected.
    assert proposal[2] > proposal[0] and proposal[2] > proposal[1]
    diagonal_only = np.asarray(mu) / (5.0 * np.diag(sigma))
    assert np.abs(proposal - diagonal_only).max() > 0.05


def test_solver_failure_holds_current_quantities_and_places_no_orders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(**_: object) -> object:
        raise solver_module.AllocationSolverError("not_converged")

    monkeypatch.setattr(
        "trade_rl.strategies.allocation.cost_aware.solve_cost_aware_allocation", broken
    )
    policy = _policy(_ConstantSource((0.01, 0.01, 0.01)))
    result = _replay(_dataset(_noise()), policy)
    assert result.book.fill_count == 0
    assert result.book.turnover_total == 0.0
    failures = [r for r in policy.records if r.outcome == "fail_closed"]
    assert any(r.reason == "solver_not_converged" for r in failures)
    assert not any(r.outcome == "rebalance" for r in policy.records)


@pytest.mark.parametrize(
    "bad",
    [
        (float("nan"), 0.0, 0.0),
        (float("inf"), 0.0, 0.0),
        (0.01, 0.01),
        (-1.5, 0.0, 0.0),
    ],
)
def test_invalid_forecast_fails_closed(bad: tuple[float, ...]) -> None:
    policy = _policy(_ConstantSource(bad))
    result = _replay(_dataset(_noise()), policy)
    assert result.book.fill_count == 0
    assert any(r.reason == "invalid_forecast" for r in policy.records)


def test_insufficient_history_holds_until_enough_scheduled_marks() -> None:
    policy = _policy(_ConstantSource((0.05, 0.05, 0.05)), min_history_periods=20)
    result = _replay(_dataset(_noise()), policy)
    early = [r for r in policy.records if r.reason == "insufficient_history"]
    assert len(early) == 20
    assert result.book.fill_count > 0
    first_rebalance = min(r.index for r in policy.records if r.outcome == "rebalance")
    assert first_rebalance == 20 * 24


def test_constraints_are_never_violated_under_extreme_signals() -> None:
    returns = _noise(seed=5, scale=0.03)

    def extreme(snapshot: PortfolioDecisionSnapshot) -> tuple[float, ...]:
        day = snapshot.timestamp_ns // (24 * 3_600 * 1_000_000_000)
        return (0.5, -0.5 if day % 3 else 0.5, 0.3)

    policy = _policy(_CallableSource(extreme), w_max=0.3, gross_max=0.5, cost=0.0005)
    result = _replay(_dataset(returns), policy)
    rebalance_indexes = {r.index for r in policy.records if r.outcome == "rebalance"}
    assert rebalance_indexes
    checked = 0
    for decision in result.decisions:
        if decision.index not in rebalance_indexes:
            continue  # off-schedule HOLD proposals echo drifted current weights
        weights = np.asarray(decision.proposal_weights)
        assert np.all(np.abs(weights) <= 0.3 + 1e-9)
        assert np.abs(weights).sum() <= 0.5 + 1e-9
        checked += 1
    assert checked == len(rebalance_indexes)


def test_long_only_never_proposes_a_short() -> None:
    def short_signal(_: PortfolioDecisionSnapshot) -> tuple[float, ...]:
        return (-0.05, 0.02, -0.03)

    policy = _policy(_CallableSource(short_signal), long_only=True)
    result = _replay(_dataset(_noise()), policy)
    for decision in result.decisions:
        assert min(decision.proposal_weights) >= -1e-12
    assert result.book.weights[0] == pytest.approx(0.0, abs=1e-12)


def test_retargets_only_on_the_decision_schedule() -> None:
    dataset = _dataset(_noise())
    policy = _policy(_ConstantSource((0.003, 0.002, 0.004)), cost=0.0002)
    result = _replay(dataset, policy)
    rebalance_indexes = {r.index for r in policy.records if r.outcome == "rebalance"}
    assert rebalance_indexes
    for decision in result.decisions:
        hour = int(
            (dataset.timestamps[decision.index] - np.datetime64("2026-01-01T00:00"))
            / np.timedelta64(1, "h")
        )
        if hour % 24 != 0:
            assert decision.requested_hold_symbols == (0, 1, 2)
    assert all(index % 24 == 0 for index in rebalance_indexes)


def _perturb_after(dataset: MarketDataset, last_unchanged_bar: int) -> MarketDataset:
    """Return a copy whose every market array differs strictly after the bar."""

    rng = np.random.default_rng(1234)
    first = last_unchanged_bar + 1
    close = dataset.close.copy()
    n_rows = close.shape[0] - first
    close[first:] *= np.exp(rng.normal(0.0, 0.03, size=(n_rows, close.shape[1])))
    open_price = dataset.open.copy()
    open_price[first:] = np.vstack((close[first - 1], close[first:-1]))
    funding = dataset.funding_rate.copy()
    funding[first:] = rng.normal(0.001, 0.002, size=funding[first:].shape)
    return replace(
        dataset,
        close=close,
        open=open_price,
        high=np.maximum(open_price, close) * 1.01,
        low=np.minimum(open_price, close) * 0.99,
        volume=np.where(
            np.arange(close.shape[0])[:, None] >= first, 1e11, dataset.volume
        ),
        funding_rate=funding,
    )


def _phase_forecast(snapshot: PortfolioDecisionSnapshot) -> tuple[float, ...]:
    day = snapshot.timestamp_ns // (24 * 3_600 * 1_000_000_000)
    phase = (day % 5) - 2
    return (0.002 * phase, -0.001 * phase, 0.0015)


def _decisions_through_bar_are_identical(
    base: MarketDataset,
    changed: MarketDataset,
    bar: int,
    make_source: Callable[[MarketDataset], ExpectedReturnSource],
) -> bool:
    first = _policy(make_source(base))
    second = _policy(make_source(changed))
    result_a = _replay(base, first)
    result_b = _replay(changed, second)
    # The perturbed future may legitimately end the replay early; only decisions
    # up to the unchanged bar are compared (and they must all exist).
    left_by_index = {d.index: d for d in result_a.decisions}
    right_by_index = {d.index: d for d in result_b.decisions}
    for index in range(bar + 1):
        left, right = left_by_index[index], right_by_index[index]
        if (
            left.proposal_weights != right.proposal_weights
            or left.requested_hold_symbols != right.requested_hold_symbols
        ):
            return False
    records_a = [r for r in first.records if r.index <= bar]
    records_b = [r for r in second.records if r.index <= bar]
    assert any(r.outcome == "rebalance" for r in records_a)
    return records_a == records_b


@pytest.mark.parametrize("last_unchanged_bar", [1176, 1199, 1200, 1201, 1223, 1224])
def test_everything_strictly_after_a_bar_cannot_change_decisions_up_to_it(
    last_unchanged_bar: int,
) -> None:
    """Perturb every array from the very next bar (incl. the next daily mark)."""

    bar = last_unchanged_bar
    base = _dataset(_noise(seed=21))
    changed = _perturb_after(base, bar)
    assert np.array_equal(base.close[: bar + 1], changed.close[: bar + 1])
    assert not np.array_equal(base.close[bar + 1], changed.close[bar + 1])
    assert _decisions_through_bar_are_identical(
        base, changed, bar, lambda _: _CallableSource(_phase_forecast)
    )


def test_the_look_ahead_check_detects_a_one_day_forward_leak() -> None:
    """Sensitivity guard: a source peeking one daily mark ahead must be caught."""

    class _Peek:
        def __init__(self, dataset: MarketDataset) -> None:
            self._dataset = dataset

        def expected_simple_returns(
            self, snapshot: PortfolioDecisionSnapshot
        ) -> tuple[float, ...]:
            ahead = min(snapshot.index + 24, self._dataset.close.shape[0] - 1)
            move = self._dataset.close[ahead] / self._dataset.close[snapshot.index] - 1
            return tuple(float(value) / 10.0 for value in move)

    base = _dataset(_noise(seed=21))
    changed = _perturb_after(base, 1199)  # first changed bar 1200 = next mark of 1176
    assert not _decisions_through_bar_are_identical(base, changed, 1199, _Peek)


def test_warmup_marks_seed_the_covariance_before_the_first_snapshot() -> None:
    rng = np.random.default_rng(1)
    warm_returns = rng.normal(0.0, 0.02, size=(10, 3))
    warm_prices = 100.0 * np.cumprod(1.0 + warm_returns, axis=0)
    day_ns = 24 * 3_600 * 1_000_000_000
    start_ns = int(np.datetime64("2026-01-01T00:00:00", "ns").astype("int64"))
    timestamps = tuple(start_ns - (10 - k) * day_ns for k in range(10))
    prices = tuple(tuple(float(v) for v in row) for row in warm_prices)
    policy = _policy(
        _ConstantSource((0.003, 0.0, 0.0)),
        min_history_periods=9,
        warmup_timestamps_ns=timestamps,
        warmup_mark_prices=prices,
    )
    _replay(_dataset(_noise()), policy)
    assert not any(r.reason == "insufficient_history" for r in policy.records)
    assert policy.records[0].index == 0


def test_ewma_covariance_weights_recent_returns_and_shrinks() -> None:
    old_then_new = np.array([[0.10, 0.10], [0.0, 0.0], [0.0, 0.0], [0.01, -0.01]])
    plain = ewma_covariance(
        old_then_new, halflife_periods=1e9, shrinkage=0.0, variance_floor=1e-12
    )
    recent = ewma_covariance(
        old_then_new, halflife_periods=1.0, shrinkage=0.0, variance_floor=1e-12
    )
    assert recent[0, 0] < plain[0, 0]
    np.testing.assert_allclose(plain, plain.T)
    shrunk = ewma_covariance(
        old_then_new, halflife_periods=1e9, shrinkage=0.5, variance_floor=1e-12
    )
    assert shrunk[0, 1] == pytest.approx(0.5 * plain[0, 1])
    assert shrunk[0, 0] == pytest.approx(plain[0, 0] + 1e-12)
    assert np.all(np.linalg.eigvalsh(shrunk) > 0.0)
    constant = ewma_covariance(
        np.zeros((5, 2)), halflife_periods=3.0, shrinkage=0.2, variance_floor=1e-8
    )
    assert np.all(np.linalg.eigvalsh(constant) > 0.0)


@pytest.mark.parametrize(
    "overrides",
    [
        {"gamma": 0.0},
        {"gamma": float("nan")},
        {"ewma_halflife_days": 0.0},
        {"shrinkage": 1.5},
        {"w_max": 0.0},
        {"gross_max": -1.0},
        {"long_only": 1},
        {"min_trade_fraction": 1.0},
        {"decision_interval_hours": 0},
        {"decision_anchor_hour_utc": 24},
        {"min_history_periods": 1},
        {"variance_floor": 0.0},
        {"max_iter": 0},
        {"tol": 0.0},
        {"cost_rates": (0.001, 0.001)},
        {"cost_rates": (-0.001, 0.0, 0.0)},
        {"symbols": ("A", "A", "B")},
        {"forecast_source": object()},
        {"warmup_timestamps_ns": (1,), "warmup_mark_prices": ()},
        {
            "warmup_timestamps_ns": (24 * 3_600 * 10**9,),
            "warmup_mark_prices": ((1.0, -1.0, 1.0),),
        },
    ],
)
def test_constructor_validation(overrides: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        _policy(_ConstantSource((0.0, 0.0, 0.0)), **overrides)


def test_snapshot_roster_and_ordering_are_validated() -> None:
    captured: list[PortfolioDecisionSnapshot] = []

    def capture(snapshot: PortfolioDecisionSnapshot) -> tuple[float, ...]:
        captured.append(snapshot)
        return (0.0, 0.0, 0.0)

    policy = _policy(_CallableSource(capture), min_history_periods=2)
    _replay(_dataset(_noise()), policy, stop_day=5)
    snapshot = captured[0]
    with pytest.raises(ValueError, match="increase"):
        policy.decide_portfolio(snapshot)
    other = _policy(
        _ConstantSource((0.0, 0.0)), symbols=("X", "Y"), cost_rates=(0.0, 0.0)
    )
    with pytest.raises(ValueError, match="roster"):
        other.decide_portfolio(snapshot)
    with pytest.raises(TypeError):
        policy.decide_portfolio(object())  # type: ignore[arg-type]


def test_proposals_are_portfolio_target_proposals() -> None:
    captured: list[PortfolioTargetProposal] = []
    policy = _policy(_ConstantSource((0.004, 0.0, 0.0)))
    original = policy.decide_portfolio

    class _Spy:
        def decide_portfolio(
            self, snapshot: PortfolioDecisionSnapshot
        ) -> PortfolioTargetProposal:
            proposal = original(snapshot)
            captured.append(proposal)
            return proposal

    run_shared_portfolio_replay(
        _dataset(_noise()),
        _Spy(),
        start_index=0,
        stop_index=40 * 24,
        initial_capital=CAPITAL,
        execution_cost=ExecutionCostConfig(),
    )
    assert captured
    assert all(isinstance(p, PortfolioTargetProposal) for p in captured)


def test_one_way_cost_rate_mirrors_the_executor_unit_cost() -> None:
    default = ExecutionCostConfig()
    assert one_way_cost_rate(default) == pytest.approx(0.0005 + 0.0 + 0.0002)
    doubled = ExecutionCostConfig(multiplier=2.0)
    assert one_way_cost_rate(doubled) == pytest.approx(2 * one_way_cost_rate(default))
    limit = ExecutionCostConfig(
        order_type="limit", maker_fee_rate=0.0001, spread_rate=0.0004
    )
    assert one_way_cost_rate(limit) == pytest.approx(0.0005 + 0.0001 + 0.0002)
    assert one_way_cost_rates(default, n_symbols=3) == (one_way_cost_rate(default),) * 3
    assert one_way_cost_rate(ExecutionCostConfig.zero()) == 0.0
    with pytest.raises(TypeError):
        one_way_cost_rate(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        one_way_cost_rates(default, n_symbols=0)
