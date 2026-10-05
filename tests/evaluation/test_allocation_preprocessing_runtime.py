"""Finite causal-prefix admission and raw-economic isolation oracles."""

from dataclasses import fields, replace

import numpy as np
import pytest

from tests.evaluation.test_allocation_rl_env import parameters
from tests.evaluation.test_allocation_rl_observation_v2 import opt_in
from tests.simulation.test_stateful_execution_characterization import _normalize
from tests.strategies.test_allocation_protocol_receipt import protocol
from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.evaluation.rl_allocation import preprocessing, training
from trade_rl.evaluation.rl_allocation.env import AllocationTradingEnv
from trade_rl.strategies.rl.allocation_recipe_v3 import allocation_recipe_payload_v3


def finite_forecast(dataset, indices=(0,)):
    from trade_rl.strategies.forecasts.simple_prequential import (
        fit_prequential_simple_ridge,
    )
    from trade_rl.strategies.forecasts.stream import ForecastBlock

    times = dataset.timestamps
    return fit_prequential_simple_ridge(
        dataset,
        blocks=(
            ForecastBlock(
                times[5], times[5] + np.timedelta64(15, "m"), times[6], times[10]
            ),
        ),
        feature_indices=indices,
        horizon_hours=1,
        alpha=1e6,
    )


def frozen_args(*, stop=8, early=False, partial=False):
    args = parameters(stop=stop)
    dataset = args["dataset"]
    values, available = (
        dataset.features.copy(),
        dataset.resolved_array("available_at").copy(),
    )
    values[:, 0, 0] = 4
    values[:3, 0, 0] = [0, 1000, 2]
    values[6:, 0, 0] = 4 + 2 * np.arange(dataset.n_bars - 6)
    available[1, 0] = dataset.timestamps[2]
    args["dataset"] = replace(
        dataset, features=values, available_at=available, information_available=None
    )
    if early:
        prices = args["dataset"].close.copy()
        prices[7:] = 400
        args["dataset"] = replace(
            args["dataset"],
            close=prices,
            mark_price=prices,
            index_price=prices,
            high=np.maximum(dataset.open, prices),
            low=np.minimum(dataset.open, prices),
        )
        args["allocator"] = replace(args["allocator"], lower_weight=-1.0)
    if partial:
        args["execution_cost"] = replace(
            args["execution_cost"], max_participation_rate=0.00001
        )
    args["stream"] = finite_forecast(args["dataset"])
    args["bound"] = replace(
        args["bound"], clock=replace(args["bound"].clock, rollout_steps=2)
    )
    args = opt_in(args)
    frozen = preprocessing.fit_allocation_feature_preprocessing(
        args["dataset"],
        feature_indices=(0,),
        fit_symbol_indices=(0,),
        fit_start=0,
        fit_stop=3,
        fit_as_of=args["dataset"].timestamps[3],
        first_decision_index=6,
    )
    return bind_frozen(args, frozen)


def bind_frozen(args, frozen):
    args = dict(args)
    args.pop("feature_preprocessing", None)
    old = AllocationTradingEnv(**opt_in(args))
    raw = old.recipe["runtime_profile"]
    from trade_rl.strategies.rl.allocation_policy import AllocationRuntimeProfile

    recipe = allocation_recipe_payload_v3(
        args["action_contract"],
        ("signal",),
        allocator=args["allocator"],
        expected_horizon_seconds=3600,
        observation_schema=args["observation_schema"],
        feature_preprocessing=frozen,
        runtime_profile=AllocationRuntimeProfile(
            **{k: v for k, v in raw.items() if k != "schema"}
        ),
    )
    args["bound"] = replace(
        args["bound"],
        objective=replace(
            args["bound"].objective, deployment_recipe_digest=content_digest(recipe)
        ),
    )
    return args | {"feature_preprocessing": frozen}


def raw_args(args):
    args = dict(args)
    del args["feature_preprocessing"]
    return opt_in(args)


@pytest.mark.parametrize(
    "early,partial", [(False, False), (False, True), (True, False)]
)
def test_first_feature_only_and_fixed_actions_preserve_complete_native_trace(
    early, partial
):
    args = frozen_args(stop=9 if early else 8, early=early, partial=partial)
    left, right = AllocationTradingEnv(**raw_args(args)), AllocationTradingEnv(**args)
    raw, normalized = left.reset()[0], right.reset()[0]
    assert args["feature_preprocessing"].normalizer.mean == (1.0,)
    assert args["feature_preprocessing"].normalizer.scale == (1.0,)
    assert raw[0] == 4 and normalized[0] == 3
    assert raw[1:].tobytes() == normalized[1:].tobytes()
    for action in [1] if early else [3, 0]:
        a, b = left.step(action), right.step(action)
        assert a[1:4] == b[1:4]
        assert _normalize(left.book) == _normalize(right.book)
        assert _normalize(left.order_book) == _normalize(right.order_book)
        assert left.decision.decision_digest == right.decision.decision_digest
    assert not b[0].any() and b[2] and not b[3]


def test_reset_never_refits_and_true_termination_never_observes_or_transforms(
    monkeypatch,
):
    args = frozen_args(stop=7)
    env = AllocationTradingEnv(**args)
    pin = args["feature_preprocessing"].digest
    monkeypatch.setattr(
        preprocessing,
        "fit_allocation_feature_preprocessing",
        lambda *a, **k: pytest.fail("reset refit"),
    )
    assert env.reset()[0][0] == env.reset()[0][0] == 3
    monkeypatch.setattr(
        env, "_observation", lambda: pytest.fail("terminal live observation")
    )
    assert not env.step(0)[0].any()
    assert env.feature_preprocessing.digest == pin


def test_new_lane_requires_explicit_protocol_before_backend_import(monkeypatch):
    env = AllocationTradingEnv(**frozen_args())
    monkeypatch.setattr(
        training.importlib, "import_module", lambda _: pytest.fail("backend")
    )
    with pytest.raises(ValueError, match="explicit.*protocol"):
        training.build_allocation_ppo(env)


@pytest.mark.parametrize(
    "kind", ["source", "coefficient", "rows", "policy_index", "policy_time"]
)
def test_training_prefix_is_reconstructed_before_backend_import(kind, monkeypatch):
    args = frozen_args()
    frozen = args["feature_preprocessing"]
    if kind == "source":
        frozen = replace(
            frozen, normalizer=replace(frozen.normalizer, source_dataset_id="f" * 64)
        )
    elif kind == "coefficient":
        frozen = replace(frozen, normalizer=replace(frozen.normalizer, mean=(2.0,)))
    elif kind == "rows":
        frozen = replace(frozen, admitted_row_indices=((0, 1),))
    elif kind == "policy_index":
        frozen = replace(frozen, policy_start_index=5)
    else:
        frozen = replace(frozen, policy_start_time_ns=frozen.policy_start_time_ns - 1)
    # Rebind a coherent recipe; only actual training source admission can reject.
    args["feature_preprocessing"] = frozen
    env = AllocationTradingEnv(**bind_frozen(args, frozen))
    monkeypatch.setattr(
        training.importlib, "import_module", lambda _: pytest.fail("backend")
    )
    with pytest.raises(ValueError, match="prefix|preprocessing"):
        training.build_allocation_ppo(
            env, training_protocol=protocol(n_steps=2, batch_size=2)
        )


def test_none_keeps_valid_v2_recipe_and_observation_bytes():
    args = raw_args(frozen_args())
    a, b = (
        AllocationTradingEnv(**args),
        AllocationTradingEnv(**args, feature_preprocessing=None),
    )
    assert canonical_json_bytes(a.recipe) == canonical_json_bytes(b.recipe)
    assert a.reset()[0].tobytes() == b.reset()[0].tobytes()


@pytest.mark.parametrize("kind", ["build", "upstream", "early"])
def test_actual_application_rejects_build_or_policy_clock_before_prediction(kind):
    args = frozen_args()
    if kind in ("build", "upstream"):
        field = "feature_config_digest" if kind == "build" else "normalization_digest"
        args["dataset"] = replace(args["dataset"], **{field: "f" * 64})
    else:
        frozen = replace(
            args["feature_preprocessing"],
            policy_start_time_ns=args["feature_preprocessing"].policy_start_time_ns + 1,
        )
        args = bind_frozen(args, frozen)
    with pytest.raises(ValueError, match="application"):
        AllocationTradingEnv(**args)


def test_future_inference_allows_new_source_id_column_coordinates_and_window():
    from datetime import timedelta

    args = frozen_args()
    dataset = args["dataset"]
    values = np.full((dataset.n_bars, 1, 2), 4.0, dtype=np.float32)
    args["dataset"] = replace(
        dataset,
        dataset_id="f" * 64,
        features=values,
        feature_available=np.ones_like(values, dtype=bool),
        feature_staleness_hours=None,
        feature_missing_reason=None,
        feature_staleness=None,
        feature_names=("unused", "signal"),
    )
    args["stream"] = finite_forecast(args["dataset"], (1,))
    future = args["dataset"]
    args["dataset"] = replace(
        future,
        **{
            field.name: value[7:]
            for field in fields(future)
            if field.init
            and isinstance(value := getattr(future, field.name), np.ndarray)
            and value.ndim > 0
            and value.shape[0] == future.n_bars
        },
    )
    args.update(feature_indices=(1,), start_index=0, stop_index=2)
    args["estimates"] = tuple(
        replace(
            c,
            decision_time=c.decision_time + np.timedelta64(1, "h"),
            available_at=c.available_at + np.timedelta64(1, "h"),
            horizon_end=c.horizon_end + np.timedelta64(1, "h"),
        )
        for c in args["estimates"]
    )
    args["bound"] = replace(
        args["bound"],
        objective=replace(
            args["bound"].objective,
            evaluation_start=args["bound"].objective.evaluation_start
            + timedelta(hours=1),
            evaluation_stop_exclusive=args["bound"].objective.evaluation_stop_exclusive
            + timedelta(hours=1),
        ),
    )
    env = AllocationTradingEnv(**args)
    assert env.reset()[0][0] == 3 and env.index == 0
    with pytest.raises(ValueError, match="prefix/source"):
        training.build_allocation_ppo(
            env, training_protocol=protocol(n_steps=2, batch_size=2)
        )
