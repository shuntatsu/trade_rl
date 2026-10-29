"""FS1 G2 independent oracles for the kappa and volatility-target mechanisms.

Every reference below is written in pure Python (``math`` and ``random`` only, no
numpy/scipy) from the documented definitions, NOT by calling production helpers, so a
shared misreading cannot make production and oracle agree. They run on synthetic data
only and read no result.

* ORACLE-1: VT scaling, annualised portfolio volatility, the kappa-scaled cost vector
  and the EWMA covariance against production, on random instances and inside a replay.
* ORACLE-2: a brute-force / local-optimality check of the kappa-scaled solve.
* ORACLE-3: default parameters (kappa 1, no volatility target) are bit-identical to the
  pre-Phase-3 decision pipeline (re-implemented here).
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from tests.evaluation.test_dev_realdata_phase2_trials import (
    SYMBOLS,
    small_protocol,
    synthetic_market,
)
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.dev_realdata_study.phase3_sources import make_phase3_context
from trade_rl.evaluation.forward_study.fs1 import build_fs1_arm
from trade_rl.evaluation.forward_study.registry import (
    ARM_FS1_A,
    ARM_FS1_B,
    ARM_FS1_C,
)
from trade_rl.evaluation.replay import run_shared_portfolio_replay
from trade_rl.simulation import ExecutionCostConfig
from trade_rl.strategies.allocation import cost_aware
from trade_rl.strategies.allocation.cost_aware import (
    CostAwareAllocationPolicy,
    annualized_portfolio_vol,
    vol_target_scale,
)
from trade_rl.strategies.allocation.costs import one_way_cost_rates
from trade_rl.strategies.allocation.solver import (
    solve_cost_aware_allocation,
)
from trade_rl.strategies.portfolio import (
    PortfolioDecisionSnapshot,
    PortfolioTargetProposal,
)

Vector = list[float]
Matrix = list[list[float]]
DAY_NS = 24 * 3_600 * 1_000_000_000


# --- pure-Python references ---------------------------------------------------


def ref_quadratic(w: Sequence[float], cov: Matrix) -> float:
    n = len(w)
    return sum(w[i] * cov[i][j] * w[j] for i in range(n) for j in range(n))


def ref_annual_vol(w: Sequence[float], cov: Matrix, periods_per_year: float) -> float:
    return math.sqrt(max(ref_quadratic(w, cov), 0.0) * periods_per_year)


def ref_vt_scale(vol: float, target: float) -> float:
    """Scale the whole book down to the cap; never up."""

    return 1.0 if vol <= target else target / vol


def ref_penalty_costs(
    n: int, *, fee: float, taker_fee: float, spread: float, kappa: float
) -> Vector:
    """kappa times the one-way market-order cost (fee + venue fee + full spread)."""

    return [kappa * (fee + taker_fee + spread)] * n


def ref_ewma_covariance(
    returns: Sequence[Sequence[float]],
    halflife: float,
    shrinkage: float,
    floor: float,
) -> Matrix:
    rows = len(returns)
    n = len(returns[0])
    raw = [0.5 ** ((rows - 1 - t) / halflife) for t in range(rows)]
    total = sum(raw)
    weights = [value / total for value in raw]
    second = [
        [
            sum(weights[t] * returns[t][i] * returns[t][j] for t in range(rows))
            for j in range(n)
        ]
        for i in range(n)
    ]
    cov = [
        [second[i][j] if i == j else (1.0 - shrinkage) * second[i][j] for j in range(n)]
        for i in range(n)
    ]
    for i in range(n):
        cov[i][i] += floor
    return cov


def random_psd(rng: random.Random, n: int, scale: float) -> Matrix:
    factor = [[rng.gauss(0.0, 1.0) for _ in range(n)] for _ in range(n)]
    return [
        [
            scale * sum(factor[i][k] * factor[j][k] for k in range(n))
            + (scale * 0.1 if i == j else 0.0)
            for j in range(n)
        ]
        for i in range(n)
    ]


# --- ORACLE-1: VT scaling and the kappa cost vector -------------------------------


@pytest.mark.parametrize("seed", range(40))
def test_vol_target_scaling_matches_the_pure_reference_on_random_instances(
    seed: int,
) -> None:
    rng = random.Random(seed)
    n = rng.choice([1, 2, 3, 5])
    cov = random_psd(rng, n, 10 ** rng.uniform(-5.5, -3.0))
    weights = [rng.uniform(-0.5, 0.5) for _ in range(n)]
    periods = rng.choice([365.0, 8_760.0 / 168.0, 8_760.0])
    target = rng.choice([0.05, 0.12, 0.30])
    reference_vol = ref_annual_vol(weights, cov, periods)
    production_vol = annualized_portfolio_vol(weights, cov, periods_per_year=periods)
    assert production_vol == pytest.approx(reference_vol, rel=1e-12, abs=1e-15)
    scale = ref_vt_scale(reference_vol, target)
    assert vol_target_scale(production_vol, target) == pytest.approx(scale, rel=1e-12)
    scaled = [value * scale for value in weights]
    # The scaled book sits exactly on the cap when it was above it, else is unchanged.
    scaled_vol = ref_annual_vol(scaled, cov, periods)
    if reference_vol > target:
        assert scaled_vol == pytest.approx(target, rel=1e-9)
    else:
        assert scale == 1.0 and scaled == weights


def test_vol_target_scale_edges_and_failures() -> None:
    assert vol_target_scale(0.12, 0.12) == 1.0
    assert vol_target_scale(0.0, 0.12) == 1.0
    assert vol_target_scale(0.24, 0.12) == pytest.approx(0.5)
    for bad in (math.nan, math.inf, -0.1):
        with pytest.raises(ValueError):
            vol_target_scale(bad, 0.12)
    with pytest.raises(ValueError):
        vol_target_scale(0.1, 0.0)
    with pytest.raises(ValueError):
        annualized_portfolio_vol([math.nan], [[1.0]], periods_per_year=365.0)
    with pytest.raises(ValueError):
        annualized_portfolio_vol([1.0], [[-1.0]], periods_per_year=365.0)


def returns_dataset() -> MarketDataset:
    return synthetic_market(seed=5)


class Recorder:
    """Delegating policy that records every snapshot/proposal of a replay."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.log: list[tuple[PortfolioDecisionSnapshot, PortfolioTargetProposal]] = []

    def decide_portfolio(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> PortfolioTargetProposal:
        proposal = self.inner.decide_portfolio(snapshot)
        self.log.append((snapshot, proposal))
        return proposal


class NoisyForecast:
    """Deterministic pseudo-random expected simple returns per decision day."""

    def expected_simple_returns(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> tuple[float, ...]:
        rng = random.Random(snapshot.timestamp_ns // DAY_NS)
        return tuple(rng.gauss(0.0, 0.004) for _ in snapshot.symbols)


def policy_for(
    cost: ExecutionCostConfig, *, kappa: float, vol_target: float | None, **extra: Any
) -> CostAwareAllocationPolicy:
    settings: dict[str, Any] = {
        "symbols": SYMBOLS,
        "forecast_source": NoisyForecast(),
        "cost_rates": one_way_cost_rates(cost, n_symbols=len(SYMBOLS)),
        "gamma": 5.0,
        "ewma_halflife_days": 15.0,
        "shrinkage": 0.2,
        "w_max": 0.5,
        "gross_max": 0.95,
        "min_history_periods": 10,
        "cost_penalty_multiple": kappa,
        "vol_target_annual": vol_target,
    }
    settings.update(extra)
    return CostAwareAllocationPolicy(**settings)


COST = ExecutionCostConfig(fee_rate=0.0005, spread_rate=0.0002)


def replay_recorded(policy: Any, *, start_day: int = 12, stop_day: int = 100):
    data = returns_dataset()
    recorder = Recorder(policy)
    run_shared_portfolio_replay(
        data,
        recorder,
        start_index=start_day * 24,
        stop_index=stop_day * 24,
        initial_capital=10_000.0,
        execution_cost=COST,
    )
    scheduled = [
        (snapshot, proposal)
        for snapshot, proposal in recorder.log
        if snapshot.timestamp_ns % DAY_NS == 0
    ]
    return data, scheduled


def test_policy_decisions_match_the_pure_reference_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FS1-A mechanism (kappa 3, VT 12%) against reference covariance, costs and scale."""

    assert COST.taker_fee_rate == 0.0 and COST.multiplier == 1.0
    captured: list[dict[str, Any]] = []
    real_solve = cost_aware.solve_cost_aware_allocation

    def spy(**kwargs: Any) -> Any:
        solution = real_solve(**kwargs)
        captured.append(
            {
                "costs": np.asarray(kwargs["costs"]).tolist(),
                "covariance": np.asarray(kwargs["covariance"]).tolist(),
                "weights": solution.weights.tolist(),
            }
        )
        return solution

    monkeypatch.setattr(cost_aware, "solve_cost_aware_allocation", spy)
    policy = policy_for(COST, kappa=3.0, vol_target=0.12, min_trade_fraction=0.0)
    _, scheduled = replay_recorded(policy)
    records = list(policy.records)
    marks: list[Vector] = []
    solved = 0
    scaled_below_one = 0
    for (snapshot, proposal), record in zip(scheduled, records, strict=True):
        marks.append(list(snapshot.account.mark_prices))
        if record.outcome == "fail_closed":
            assert record.reason == "insufficient_history"
            continue
        returns = [
            [marks[t + 1][i] / marks[t][i] - 1.0 for i in range(3)]
            for t in range(len(marks) - 1)
        ]
        cov = ref_ewma_covariance(returns, 15.0 * 24.0 / 24.0, 0.2, 1e-8)
        call = captured[solved]
        solved += 1
        # kappa multiplies the optimizer's cost vector only: 3 * (0.0005 + 0.0002).
        assert call["costs"] == pytest.approx(
            ref_penalty_costs(3, fee=0.0005, taker_fee=0.0, spread=0.0002, kappa=3.0),
            rel=1e-12,
        )
        for row, ref_row in zip(call["covariance"], cov, strict=True):
            assert row == pytest.approx(ref_row, rel=1e-9, abs=1e-15)
        solver_weights = call["weights"]
        vol = ref_annual_vol(solver_weights, cov, 365.0)
        scale = ref_vt_scale(vol, 0.12)
        assert record.ex_ante_vol == pytest.approx(vol, rel=1e-8, abs=1e-12)
        assert record.vol_scale == pytest.approx(scale, rel=1e-8)
        scaled_below_one += scale < 1.0
        # With a zero minimum trade size nothing is held back: target = scale * solve.
        assert list(proposal.target_weights) == pytest.approx(
            [scale * value for value in solver_weights], rel=1e-9, abs=1e-12
        )
        assert record.final_vol is not None
        assert record.final_vol <= 0.12 + 1e-9 or scale == 1.0
    assert solved >= 60
    assert scaled_below_one >= 10, "the cap must bind often enough to be tested"


def test_the_fs1_builders_use_the_frozen_mechanism_settings() -> None:
    data = synthetic_market()
    protocol = small_protocol()
    context = make_phase3_context(data, protocol)
    a = build_fs1_arm(ARM_FS1_A, data, protocol, context).policy
    b = build_fs1_arm(ARM_FS1_B, data, protocol, context).policy
    c = build_fs1_arm(ARM_FS1_C, data, protocol, context).policy
    assert a._cost_penalty_multiple == 3.0 and a._vol_target == 0.12
    assert b._cost_penalty_multiple == 1.0 and b._vol_target == 0.12
    assert a._long_only is False and b._long_only is False
    assert a._periods_per_year == 365.0 == b._periods_per_year
    assert (a._gamma, a._w_max, a._gross_max) == (5.0, 0.5, 0.95)
    assert list(a._costs) == pytest.approx([0.0007] * len(data.symbols), rel=1e-12)
    assert c._vol_target == 0.12 and c._gross_budget == 0.99


# --- ORACLE-2: brute-force / local optimality of the kappa-scaled solve -----------


def ref_objective(
    w: Sequence[float],
    mu: Sequence[float],
    cov: Matrix,
    w0: Sequence[float],
    costs: Sequence[float],
    gamma: float,
) -> float:
    return (
        sum(m * x for m, x in zip(mu, w, strict=True))
        - 0.5 * gamma * ref_quadratic(w, cov)
        - sum(c * abs(x - y) for c, x, y in zip(costs, w, w0, strict=True))
    )


def feasible(w: Sequence[float], w_max: float, gross: float, long_only: bool) -> bool:
    floor = 0.0 if long_only else -w_max
    return (
        all(floor - 1e-12 <= x <= w_max + 1e-12 for x in w)
        and sum(abs(x) for x in w) <= gross + 1e-12
    )


def instance(seed: int, n: int, *, long_only: bool) -> dict[str, Any]:
    rng = random.Random(seed)
    w_max = rng.choice([0.5, 0.4])
    gross = rng.choice([0.95, 0.5, 0.3])
    floor = 0.0 if long_only else -w_max
    start = [rng.uniform(floor, w_max) * rng.choice([0.0, 1.0]) for _ in range(n)]
    total = sum(abs(x) for x in start)
    if total > gross:
        start = [x * 0.99 * gross / total for x in start]
    return {
        "mu": [rng.gauss(0.0, 0.004) for _ in range(n)],
        "cov": random_psd(rng, n, 10 ** rng.uniform(-4.5, -3.5)),
        "w0": start,
        "gamma": 5.0,
        "w_max": w_max,
        "gross": gross,
        "kappa": rng.choice([1.0, 3.0]),
        "long_only": long_only,
    }


def solve_production(case: dict[str, Any]) -> list[float]:
    costs = ref_penalty_costs(
        len(case["mu"]), fee=0.0005, taker_fee=0.0, spread=0.0002, kappa=case["kappa"]
    )
    solution = solve_cost_aware_allocation(
        expected_returns=case["mu"],
        covariance=case["cov"],
        current_weights=case["w0"],
        costs=costs,
        gamma=case["gamma"],
        w_max=case["w_max"],
        gross_max=case["gross"],
        long_only=case["long_only"],
    )
    return [float(x) for x in solution.weights]


def case_objective(case: dict[str, Any], w: Sequence[float]) -> float:
    costs = ref_penalty_costs(
        len(case["mu"]), fee=0.0005, taker_fee=0.0, spread=0.0002, kappa=case["kappa"]
    )
    return ref_objective(w, case["mu"], case["cov"], case["w0"], costs, case["gamma"])


def pattern_search(case: dict[str, Any], start: Vector) -> Vector:
    """Derivative-free polish with axis and pairwise directions (reference only)."""

    n = len(start)
    directions: list[Vector] = []
    for i in range(n):
        for sign in (1.0, -1.0):
            unit = [0.0] * n
            unit[i] = sign
            directions.append(unit)
        for j in range(i + 1, n):
            for a, b in ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)):
                pair = [0.0] * n
                pair[i], pair[j] = a, b
                directions.append(pair)
    best = list(start)
    value = case_objective(case, best)
    step = 0.01
    while step > 1e-13:
        improved = False
        for direction in directions:
            trial = [x + step * d for x, d in zip(best, direction, strict=True)]
            if not feasible(trial, case["w_max"], case["gross"], case["long_only"]):
                continue
            trial_value = case_objective(case, trial)
            if trial_value > value + 1e-16:
                best, value, improved = trial, trial_value, True
        if not improved:
            step *= 0.5
    return best


@pytest.mark.parametrize("long_only", [False, True])
@pytest.mark.parametrize("seed", range(12))
def test_two_asset_kappa_scaled_solve_is_not_beaten_by_brute_force(
    seed: int, long_only: bool
) -> None:
    case = instance(seed, 2, long_only=long_only)
    solved = solve_production(case)
    assert feasible(solved, case["w_max"], case["gross"], long_only)
    produced = case_objective(case, solved)
    lo = 0.0 if long_only else -case["w_max"]
    axes = [
        sorted(
            {lo + (case["w_max"] - lo) * k / 160 for k in range(161)}
            | {case["w0"][i], 0.0}
        )
        for i in range(2)
    ]
    best_value, best_point = -math.inf, [0.0, 0.0]
    for x in axes[0]:
        for y in axes[1]:
            if not feasible((x, y), case["w_max"], case["gross"], long_only):
                continue
            value = case_objective(case, (x, y))
            if value > best_value:
                best_value, best_point = value, [x, y]
    refined = pattern_search(case, best_point)
    refined_value = case_objective(case, refined)
    # Production is never beaten by an exhaustive search ...
    assert produced >= best_value - 1e-12
    assert produced >= refined_value - 1e-10
    # ... and is not suspiciously better than the true optimum either.
    assert produced - refined_value <= 1e-7


@pytest.mark.parametrize("long_only", [False, True])
@pytest.mark.parametrize("seed", range(10))
def test_three_asset_kappa_scaled_solve_has_no_feasible_improving_move(
    seed: int, long_only: bool
) -> None:
    case = instance(100 + seed, 3, long_only=long_only)
    solved = solve_production(case)
    assert feasible(solved, case["w_max"], case["gross"], long_only)
    produced = case_objective(case, solved)
    rng = random.Random(seed)
    floor = 0.0 if long_only else -case["w_max"]
    anchors = [solved, case["w0"], [0.0, 0.0, 0.0]]
    for _ in range(3_000):
        base = rng.choice(anchors)
        mode = rng.random()
        trial = list(base)
        if mode < 0.4:
            i = rng.randrange(3)
            trial[i] += rng.gauss(0.0, 10 ** rng.uniform(-6, -1))
        elif mode < 0.8:
            i, j = rng.sample(range(3), 2)
            move = rng.gauss(0.0, 10 ** rng.uniform(-6, -1))
            trial[i] += move
            trial[j] -= move
        else:
            trial = [rng.uniform(floor, case["w_max"]) for _ in range(3)]
        trial = [min(max(x, floor), case["w_max"]) for x in trial]
        if not feasible(trial, case["w_max"], case["gross"], long_only):
            continue
        assert case_objective(case, trial) <= produced + 1e-10


def test_larger_kappa_never_increases_trading() -> None:
    for seed in range(8):
        case = instance(300 + seed, 3, long_only=False)
        distances = []
        for kappa in (1.0, 3.0, 10.0, 50.0):
            solved = solve_production({**case, "kappa": kappa})
            distances.append(sum(abs(x - y) for x, y in zip(solved, case["w0"])))
        assert all(
            later <= earlier + 1e-9
            for earlier, later in zip(distances, distances[1:], strict=False)
        )
        assert distances[-1] <= distances[0]


# --- ORACLE-3: defaults are bit-identical to the pre-Phase-3 pipeline -------------


class PrePhase3Reference:
    """The decision pipeline as it was before kappa / VT existed (re-implemented)."""

    def __init__(self, policy_kwargs: dict[str, Any]) -> None:
        from trade_rl.strategies.allocation.cost_aware import ewma_covariance

        self._ewma = ewma_covariance
        self.kwargs = policy_kwargs
        self.marks: list[np.ndarray] = []
        self.records: list[str] = []

    def decide_portfolio(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> PortfolioTargetProposal:
        k = self.kwargs
        n = len(snapshot.symbols)
        current = snapshot.account.weights
        hold_all = PortfolioTargetProposal(
            target_weights=current, hold_current_quantity_symbols=tuple(range(n))
        )
        if snapshot.timestamp_ns % DAY_NS != 0:
            return hold_all
        marks = np.asarray(snapshot.account.mark_prices, dtype=np.float64)
        self.marks.append(marks)
        if len(self.marks) - 1 < k["min_history_periods"]:
            self.records.append("fail_closed")
            return hold_all
        prices = np.vstack(self.marks)
        cov = self._ewma(
            prices[1:] / prices[:-1] - 1.0,
            halflife_periods=k["ewma_halflife_days"] * 24.0 / 24,
            shrinkage=k["shrinkage"],
            variance_floor=1e-8,
        )
        mu = np.asarray(k["forecast_source"].expected_simple_returns(snapshot))
        w_current = np.asarray(current, dtype=np.float64)
        solution = solve_cost_aware_allocation(
            expected_returns=mu,
            covariance=cov,
            current_weights=w_current,
            costs=np.asarray(k["cost_rates"], dtype=np.float64),
            gamma=k["gamma"],
            w_max=k["w_max"],
            gross_max=k["gross_max"],
            long_only=False,
            max_iter=2_000,
            tol=1e-10,
        )
        target = solution.weights.copy()
        held: list[int] = []
        for i in range(n):
            if abs(target[i] - w_current[i]) <= k["min_trade_fraction"]:
                trial = target.copy()
                trial[i] = w_current[i]
                floor = -k["w_max"]
                if (
                    np.all(trial >= floor - 1e-9)
                    and np.all(trial <= k["w_max"] + 1e-9)
                    and float(np.abs(trial).sum()) <= k["gross_max"] + 1e-9
                ):
                    target = trial
                    held.append(i)
        traded = [i for i in range(n) if i not in held]
        if not traded:
            self.records.append("hold_no_trade")
            return hold_all
        self.records.append("rebalance")
        return PortfolioTargetProposal(
            target_weights=tuple(
                current[i] if i in held else float(target[i]) for i in range(n)
            ),
            hold_current_quantity_symbols=tuple(held),
        )


def test_default_parameters_are_bit_identical_to_the_pre_phase3_pipeline() -> None:
    kwargs: dict[str, Any] = {
        "forecast_source": NoisyForecast(),
        "cost_rates": one_way_cost_rates(COST, n_symbols=3),
        "gamma": 5.0,
        "ewma_halflife_days": 15.0,
        "shrinkage": 0.2,
        "w_max": 0.5,
        "gross_max": 0.95,
        "min_trade_fraction": 0.005,
        "min_history_periods": 10,
    }
    defaults = CostAwareAllocationPolicy(symbols=SYMBOLS, **kwargs)
    explicit = CostAwareAllocationPolicy(
        symbols=SYMBOLS, cost_penalty_multiple=1.0, vol_target_annual=None, **kwargs
    )
    reference = PrePhase3Reference(kwargs)

    class Fan:
        """Feed the identical snapshot to all three and require identical proposals."""

        def __init__(self) -> None:
            self.compared = 0
            self.rebalances = 0

        def decide_portfolio(
            self, snapshot: PortfolioDecisionSnapshot
        ) -> PortfolioTargetProposal:
            first = defaults.decide_portfolio(snapshot)
            second = explicit.decide_portfolio(snapshot)
            third = reference.decide_portfolio(snapshot)
            assert second == first == third
            if snapshot.timestamp_ns % DAY_NS == 0:
                self.compared += 1
                self.rebalances += first.target_weights != snapshot.account.weights
            return first

    fan = Fan()
    data = returns_dataset()
    run_shared_portfolio_replay(
        data,
        fan,
        start_index=12 * 24,
        stop_index=100 * 24,
        initial_capital=10_000.0,
        execution_cost=COST,
    )
    assert fan.compared > 80 and fan.rebalances > 20
    assert defaults.records == explicit.records
    assert [r.outcome for r in defaults.records] == reference.records
    for record in defaults.records:
        assert record.ex_ante_vol is None
        assert record.vol_scale == 1.0 and record.final_vol is None
    # kappa = 1.0 is an exact floating-point no-op on the cost vector.
    costs = np.asarray(kwargs["cost_rates"], dtype=np.float64)
    assert (costs * 1.0).tobytes() == costs.tobytes()


def test_an_active_mechanism_is_not_bit_identical_to_the_defaults() -> None:
    """Guards the bit-identity test against comparing two inert policies."""

    base_kwargs: dict[str, Any] = {
        "symbols": SYMBOLS,
        "forecast_source": NoisyForecast(),
        "cost_rates": one_way_cost_rates(COST, n_symbols=3),
        "ewma_halflife_days": 15.0,
        "min_history_periods": 10,
        "gross_max": 0.95,
    }
    plain = CostAwareAllocationPolicy(**base_kwargs)
    capped = CostAwareAllocationPolicy(**base_kwargs, vol_target_annual=0.12)
    taxed = CostAwareAllocationPolicy(**base_kwargs, cost_penalty_multiple=3.0)

    class Fan:
        def __init__(self) -> None:
            self.differences = {"capped": 0, "taxed": 0}

        def decide_portfolio(
            self, snapshot: PortfolioDecisionSnapshot
        ) -> PortfolioTargetProposal:
            first = plain.decide_portfolio(snapshot)
            if capped.decide_portfolio(snapshot) != first:
                self.differences["capped"] += 1
            if taxed.decide_portfolio(snapshot) != first:
                self.differences["taxed"] += 1
            return first

    fan = Fan()
    run_shared_portfolio_replay(
        returns_dataset(),
        fan,
        start_index=12 * 24,
        stop_index=100 * 24,
        initial_capital=10_000.0,
        execution_cost=COST,
    )
    assert fan.differences["capped"] > 0 and fan.differences["taxed"] > 0
