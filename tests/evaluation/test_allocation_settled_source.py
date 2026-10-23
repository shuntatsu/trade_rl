"""Actor declarations and the entire native closing envelope have separate scope."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from tests.evaluation.test_allocation_settled_env import arguments, environment
from trade_rl.evaluation.rl_allocation.settled_training_source import (
    settled_training_source_receipt,
)


@pytest.mark.parametrize("wrapped_source", [True, False])
def test_unrepresentable_utc_dates_cannot_alias_a_native_nanosecond_window(
    wrapped_source,
):
    from trade_rl.evaluation.rl_allocation.settled_binding import (
        validate_settled_runtime_binding,
    )
    from trade_rl.simulation import MarketExecutor

    args, _ = arguments()
    bound = args["bound"]
    objective = replace(
        bound.objective,
        evaluation_start=datetime(2600, 1, 1, 6, tzinfo=UTC),
        evaluation_stop_exclusive=datetime(2600, 1, 1, 10, tzinfo=UTC),
    )
    bound = replace(bound, objective=objective)
    # Build no account or source packet at this date. Exercise the clock reader
    # directly with both original coarse stamps and their wrapped ns aliases.
    original = np.datetime64("2600-01-01T00:00:00", "us") + np.arange(
        12
    ) * np.timedelta64(1, "h")
    timestamps = original.astype("datetime64[ns]") if wrapped_source else original
    dataset = args["dataset"]
    object.__setattr__(dataset, "timestamps", timestamps)
    executor = MarketExecutor(
        dataset, args["execution_cost"], insolvency_valuation="retain_debt"
    )
    with pytest.raises(ValueError, match="nanoseconds"):
        validate_settled_runtime_binding(
            dataset,
            bound,
            executor,
            args["risk_config"],
            objective.deployment_recipe_digest,
            start_index=6,
            actor_stop_index=8,
            settlement_stop_index=10,
        )


@pytest.mark.parametrize("field", ["funding_price_rate", "cash_rate", "volume"])
def test_future_reserve_financial_drift_rejects_before_next_actor_execution(field):
    env = environment()
    env.reset()
    env.step(np.int64(3))
    native_index, actor_step, book = env.index, env.actor_step, env.book
    changed = getattr(env.dataset, field).copy()
    changed[env.settlement_stop_index] += 0.5
    # Same declared Dataset identity, mutated native bytes: a digest label alone
    # must not authorize the later reserve. Never run the altered account.
    object.__setattr__(env.dataset, field, changed)
    with pytest.raises(ValueError, match="envelope") as failure:
        env.step(np.int64(0))
    assert (
        env.book is book and env.index == native_index and env.actor_step == actor_step
    )
    assert env.publication_invalid is True
    assert failure.value.settled_execution_receipt


def test_reserve_features_are_not_pinned_or_read_as_actor_inputs():
    env = environment()
    declared = settled_training_source_receipt(env)
    features = env.dataset.features.copy()
    features[env.actor_stop_index :] = 777.0
    object.__setattr__(env.dataset, "features", features)
    assert settled_training_source_receipt(env) == declared
    env.reset()
    env.step(np.int64(3))
    _, _, terminal, _, _ = env.step(np.int64(0))
    assert terminal is True and env.book.cash == pytest.approx(998.0)


def test_actor_feature_mutation_rejects_before_account_reset():
    env = environment()
    features = env.dataset.features.copy()
    features[env.start_index, env.symbol_index, env.feature_indices[0]] += 0.5
    object.__setattr__(env.dataset, "features", features)
    with pytest.raises(ValueError, match="envelope"):
        env.reset()
    assert not hasattr(env, "book"), "source preflight precedes actual account creation"


def test_envelope_covers_reference_and_processing_rows_and_is_detached():
    env = environment()
    receipt = settled_training_source_receipt(env)
    assert receipt["consumption_proven"] is False
    assert receipt["native_scope"]["physical_source_indices"] == [6, 7, 8, 9, 10]
    assert receipt["native_scope"]["processing_indices"] == [7, 8, 9, 10]
    assert receipt["actor_scope"]["actor_indices"] == [6, 7]
    assert receipt["native_financial_arrays"]["funding_price_rate"]["shape"] == [5, 1]
    assert receipt["native_financial_arrays"]["contract_multipliers"]["shape"] == [1]
    receipt["actor_scope"]["actor_indices"].append(99)
    assert settled_training_source_receipt(env)["actor_scope"]["actor_indices"] == [
        6,
        7,
    ]


def test_v4_recipe_is_window_independent_and_reconstructs_closed_native_fields():
    from trade_rl.artifacts import content_digest
    from trade_rl.strategies.rl.allocation_recipe_validation import (
        validate_allocation_recipe,
    )

    env = environment()
    validate_allocation_recipe(env.recipe)
    original = env.recipe_digest
    # Objective clock endpoints bind a window separately from deployable recipe.
    shifted = replace(
        env.bound.objective,
        evaluation_start=env.bound.objective.evaluation_start + timedelta(days=1),
        evaluation_stop_exclusive=env.bound.objective.evaluation_stop_exclusive
        + timedelta(days=1),
    )
    before_bound = env.bound.digest
    env.bound = replace(env.bound, objective=shifted)
    assert content_digest(env.recipe) == original and env.bound.digest != before_bound
    with pytest.raises(ValueError, match="changed"):
        env.validate_binding()
    payload = env.recipe
    payload["settled_clock"]["physical_horizon_steps"] = 3
    with pytest.raises(ValueError):
        validate_allocation_recipe(payload)
    payload = env.recipe
    payload["terminal_valuation"] = "marked_continuation"
    with pytest.raises(ValueError):
        validate_allocation_recipe(payload)
