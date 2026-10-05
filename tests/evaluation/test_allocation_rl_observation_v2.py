"""Live opt-in state projection through the existing canonical episode."""

from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from tests.evaluation.test_allocation_rl_env import parameters
from tests.simulation.test_stateful_execution_characterization import _normalize
from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.data.contracts import MarketCalendarKind
from trade_rl.evaluation.rl_allocation.env import AllocationTradingEnv
from trade_rl.simulation import MarketExecutor
from trade_rl.strategies.rl.allocation_observation_v2 import AllocationObservationSchema
from trade_rl.strategies.rl.allocation_policy import AllocationRuntimeProfile
from trade_rl.strategies.rl.allocation_recipe_v2 import allocation_recipe_payload_v2


def opt_in(args):
    args = dict(args)
    dataset, bound = args["dataset"], args["bound"]
    features = tuple(dataset.feature_names[i] for i in args["feature_indices"])
    schema = AllocationObservationSchema(
        features,
        2,
        bound.objective.capital.initial_equities[0],
        args["stop_index"] - args["start_index"],
    )
    profile = AllocationRuntimeProfile(
        MarketExecutor(
            dataset, args["execution_cost"], insolvency_valuation="retain_debt"
        ).execution_policy_digest,
        content_digest(args["risk_config"]),
        schema.initial_capital,
        bound.objective.capital.currency,
        bound.clock.decision_interval_seconds,
        bound.clock.economic_horizon_seconds,
        calendar_kind=MarketCalendarKind(dataset.calendar_kind).value,
        execution_bar_hours=dataset.bar_hours,
    )
    recipe = allocation_recipe_payload_v2(
        args["action_contract"],
        features,
        allocator=args["allocator"],
        expected_horizon_seconds=3600,
        runtime_profile=profile,
        observation_schema=schema,
    )
    args["bound"] = replace(
        bound,
        objective=replace(
            bound.objective,
            economics_digest=profile.economics_digest,
            risk_digest=profile.risk_digest,
            deployment_recipe_digest=content_digest(recipe),
        ),
    )
    args["observation_schema"] = schema
    return args


def test_opt_in_reset_initial_margin_and_native_live_columns():
    args = parameters()
    args["execution_cost"] = replace(
        args["execution_cost"], maintenance_margin_rate=0.25
    )
    env = AllocationTradingEnv(**opt_in(args))
    values, _ = env.reset()
    fields = env.observation_schema.fields
    assert values.shape == (71,) and env.observation_space.contains(values)
    assert values[fields.index("cash_over_initial_capital")] == 1
    assert values[fields.index("maintenance_margin")] == 0.25
    assert not values[fields.index("order[0].presence") :].any()
    values, reward, done, truncated, _ = env.step(3)
    assert reward == pytest.approx(0.049) and not done and not truncated
    assert values[fields.index("position_notional_over_initial_capital")] <= 0.55
    assert values[fields.index("current_drawdown")] == 0
    assert values[fields.index("remaining_horizon_fraction")] == 0.5


def test_identical_actions_preserve_v1_native_books_orders_and_rewards():
    args = parameters()
    legacy, current = AllocationTradingEnv(**args), AllocationTradingEnv(**opt_in(args))
    before = canonical_json_bytes(legacy.recipe)
    explicit_none = AllocationTradingEnv(**args, observation_schema=None)
    left, _ = legacy.reset()
    default_values, _ = explicit_none.reset()
    np.testing.assert_array_equal(left, default_values)
    assert canonical_json_bytes(explicit_none.recipe) == before
    right, _ = current.reset()
    assert left.shape == (17,) and right.shape == (71,)
    assert legacy.book.maintenance_margin == 0
    assert (
        current.book.maintenance_margin
        == args["execution_cost"].maintenance_margin_rate
    )
    for action in (3, 2):
        a, b = legacy.step(action), current.step(action)
        assert a[1:4] == b[1:4]
        assert _normalize(legacy.book) == _normalize(current.book)
        assert _normalize(legacy.order_book) == _normalize(current.order_book)
    assert not b[0].any() and b[2] and not b[3]
    assert canonical_json_bytes(legacy.recipe) == before


def test_actual_partial_order_columns_use_exact_remaining_and_live_mark():
    args = parameters()
    args["execution_cost"] = replace(
        args["execution_cost"], max_participation_rate=0.00001
    )
    env = AllocationTradingEnv(**opt_in(args))
    env.reset()
    values, _, done, _, _ = env.step(3)
    assert not done and len(env.order_book.active_orders) == 1
    for name, exact in (
        ("requested_notional_over_initial_capital", Fraction(11, 20)),
        ("cumulative_quantity_notional_over_initial_capital", Fraction(11, 100)),
        ("remaining_notional_over_initial_capital", Fraction(11, 25)),
    ):
        value = values[env.observation_schema.fields.index(f"order[0].{name}")]
        assert 0 < Fraction.from_float(float(value)) <= exact
        assert (
            Fraction.from_float(float(np.nextafter(value, np.float32(np.inf)))) > exact
        )


@pytest.mark.parametrize("early", [False, True])
def test_true_termination_never_requests_live_snapshot(early, monkeypatch):
    from trade_rl.evaluation.rl_allocation import env as module

    args = parameters(stop=9 if early else 7)
    if early:
        args["allocator"] = replace(args["allocator"], lower_weight=-1.0)
        dataset = args["dataset"]
        close = dataset.close.copy()
        close[7:] = 400
        args["dataset"] = replace(
            dataset,
            close=close,
            mark_price=close,
            index_price=close,
            high=np.maximum(dataset.open, close),
            low=np.minimum(dataset.open, close),
        )
    env = AllocationTradingEnv(**opt_in(args))
    env.reset()
    monkeypatch.setattr(
        module,
        "snapshot_allocation_account",
        lambda *a, **k: pytest.fail("terminal read"),
    )
    values, reward, terminated, truncated, _ = env.step(1 if early else 3)
    assert terminated and not truncated and not values.any()
    assert env.index == 7
    if early:
        assert env.book.cash == env.book.portfolio_value == -501 and reward == -1.501


@pytest.mark.parametrize(
    "change",
    [{"initial_capital": 999}, {"episode_steps": 3}, {"feature_names": ("other",)}],
)
def test_env_rejects_schema_binding_mismatch(change):
    args = opt_in(parameters())
    args["observation_schema"] = replace(args["observation_schema"], **change)
    with pytest.raises(ValueError, match="match"):
        AllocationTradingEnv(**args)


@pytest.mark.parametrize(
    "field,value",
    [("max_active_orders", 1), ("initial_capital", 999), ("episode_steps", 3)],
)
def test_runtime_schema_drift_rejects_before_action_side_effects(field, value):
    env = AllocationTradingEnv(**opt_in(parameters()))
    env.reset()
    before = _normalize(env.book), _normalize(env.order_book)
    env.observation_schema = replace(env.observation_schema, **{field: value})
    with pytest.raises(ValueError):
        env.step(3)
    assert before == (_normalize(env.book), _normalize(env.order_book))


def test_live_snapshot_source_mismatch_is_not_admitted_to_the_policy(monkeypatch):
    from trade_rl.evaluation.rl_allocation import env as module

    original = module.snapshot_allocation_account

    def wrong_source(*args, **kwargs):
        return replace(original(*args, **kwargs), source_state_digest="f" * 64)

    monkeypatch.setattr(module, "snapshot_allocation_account", wrong_source)
    with pytest.raises(ValueError, match="match"):
        AllocationTradingEnv(**opt_in(parameters())).reset()
