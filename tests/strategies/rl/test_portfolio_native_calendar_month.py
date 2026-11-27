"""Generated calendar-month RL must use calendar boundaries, not nominal seconds."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from fractions import Fraction

import numpy as np
import pytest

from tests.strategies.rl.test_portfolio_inference_bundle import (
    _dataset,
    _load,
    _publish,
)
from tests.strategies.rl.test_portfolio_native_fixed import _cost, _risk
from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.artifacts.publication import publish_market_dataset_artifact
from trade_rl.data.native_clock import NativeBarClock, NativeClockKind
from trade_rl.data.native_execution_clock import calendar_month_execution_frequency
from trade_rl.evaluation.native_calendar_returns import (
    native_calendar_month_return_payload,
)
from trade_rl.evaluation.portfolio_inference import (
    run_published_portfolio_inference_replay,
)
from trade_rl.evaluation.replay import run_shared_portfolio_replay
from trade_rl.risk.pretrade import PreTradeRisk
from trade_rl.simulation.objectives.binding import BoundObjectiveClock
from trade_rl.simulation.objectives.contract import CapitalContract, ObjectiveContract
from trade_rl.strategies.rl.portfolio_env import (
    ACTION_HOLD,
    ACTION_LONG,
    SharedPortfolioGymEnv,
)
from trade_rl.strategies.rl.portfolio_training import (
    SharedPortfolioPPOConfig,
    fit_shared_portfolio_ppo,
)
from trade_rl.strategies.rl.portfolio_transform import (
    PortfolioObservationRuntimeBinding,
    PortfolioObservationTransform,
    fit_portfolio_observation_transform,
)


def _month_dataset():
    base = _dataset(n_bars=10)
    closes = np.arange("2024-01", "2024-11", dtype="datetime64[M]").astype(
        "datetime64[ns]"
    )
    native = NativeBarClock(NativeClockKind.UTC_MONTH)
    return replace(
        base,
        timestamps=closes,
        available_at=np.broadcast_to(closes[:, None], (10, 2)).copy(),
        native_clock=native,
        nominal_bar_hours=None,
        periods_per_year=calendar_month_execution_frequency(native, closes),
        funding_event_count=np.zeros((10, 2), dtype=np.int32),
        funding_due=np.zeros((10, 2), dtype=np.bool_),
        funding_price_rate=np.zeros((10, 2)),
        identity_payload_json=None,
    ).with_content_identity({"fixture": "generated-monthly-ppo"})


def _objective(start: int, stop: int):
    return ObjectiveContract(
        capital=CapitalContract("shared_portfolio", "USDT", (1000.0,)),
        evaluation_start=datetime(2024, start + 1, 1, tzinfo=UTC),
        evaluation_stop_exclusive=datetime(2024, stop + 1, 1, tzinfo=UTC),
        terminal_valuation="settled",
        economics_digest="b" * 64,
        risk_digest="c" * 64,
        deployment_recipe_digest="d" * 64,
    )


def _bound(start=0, stop=5):
    from trade_rl.simulation.objectives.clock import CalendarMonthFinancialClockContract

    objective = _objective(start, stop)
    seconds = int(
        (
            objective.evaluation_stop_exclusive - objective.evaluation_start
        ).total_seconds()
    )
    return BoundObjectiveClock(
        objective,
        CalendarMonthFinancialClockContract(
            economic_horizon_months=stop - start,
            economic_horizon_seconds=seconds,
            rollout_steps=4,
            gamma=1.0,
            gae_lambda=0.95,
            reward_schema="equity_delta_v1",
        ),
    )


def _environment(dataset, start=0, stop=5, *, bound=None):
    return SharedPortfolioGymEnv(
        dataset,
        start_index=start,
        stop_index=stop,
        bound=_bound(start, stop) if bound is None else bound,
        target_abs_weights=(0.25, 0.25),
        execution_cost=_cost(),
        risk_config=_risk(),
    )


def test_monthly_clock_is_not_a_nominal_fixed_duration():
    bound = _bound(0, 5)
    assert bound.clock.decision_interval_seconds is None
    assert bound.clock.payload()["schema"] == "calendar_month_financial_clock_v1"
    assert bound.clock.economic_horizon_seconds == 152 * 86400
    assert (
        bound.clock.discount_after(29 * 86400)
        == bound.clock.discount_after(31 * 86400)
        == 1.0
    )
    with pytest.raises(ValueError, match="calendar month"):
        BoundObjectiveClock(
            bound.objective, replace(bound.clock, economic_horizon_months=4)
        )
    with pytest.raises(ValueError):
        replace(bound.clock, gamma=0.99)
    with pytest.raises(ValueError):
        BoundObjectiveClock(
            replace(bound.objective, evaluation_start=datetime(2024, 1, 2, tzinfo=UTC)),
            replace(bound.clock, economic_horizon_seconds=151 * 86400),
        )


def test_monthly_gym_uses_actual_close_and_settles():
    dataset = _month_dataset()
    env = _environment(dataset)
    transform = fit_portfolio_observation_transform(env)
    assert transform.native_interval_seconds is None
    payload = transform.to_payload()
    assert payload["schema"] != "shared_portfolio_observation_transform_v1"
    assert (
        PortfolioObservationTransform.from_payload(
            payload, expected_digest=transform.digest
        ).digest
        == transform.digest
    )
    raw, _ = env.reset(seed=42)
    result = transform.transform_observation(raw)
    assert result["clock"][2] == 0.0
    raw, reward, done, truncated, info = env.step(np.array([ACTION_LONG, ACTION_LONG]))
    rewards = [reward]
    assert not done and not truncated
    assert info["elapsed_intervals"] == 1
    assert transform.transform_observation(raw)["clock"][2] == pytest.approx(31 / 152)
    bad = {key: value.copy() for key, value in raw.items()}
    bad["clock"][1] += 86400 * 10**9
    with pytest.raises(ValueError, match="timestamp"):
        transform.transform_observation(bad)
    while not done:
        raw, reward, done, truncated, info = env.step(
            np.array([ACTION_HOLD, ACTION_HOLD])
        )
        rewards.append(reward)
    assert not truncated and info["settlement_complete"]
    assert env._session is not None
    assert all(q == 0 for q in env._session.book.exact_quantities)
    assert sum(rewards) == pytest.approx(env._session.book.portfolio_value / 1000.0 - 1)


def test_monthly_clock_refuses_fixed_dataset_and_mutated_window():
    with pytest.raises(ValueError, match="calendar month"):
        _environment(_dataset(n_bars=10))
    env = _environment(_month_dataset())
    env.stop_index = 6
    with pytest.raises(ValueError, match="window|horizon"):
        env.reset()
    assert env._session is None


def test_actual_monthly_ppo_roundtrip_and_replay(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    dataset = _month_dataset()
    env = _environment(dataset)
    transform = fit_portfolio_observation_transform(env)
    trained = fit_shared_portfolio_ppo(
        env,
        transform=transform,
        output_root=tmp_path / "attempt",
        config=SharedPortfolioPPOConfig(
            total_timesteps=8,
            n_steps=4,
            batch_size=2,
            n_epochs=1,
            reset_seed=29,
            ppo_seed=31,
        ),
    )
    assert trained.realized_timesteps == 8
    context = json.loads((trained.output_root / "training_context.json").read_bytes())
    assert context["financial_clock"]["schema"] == "calendar_month_financial_clock_v1"
    publication = _publish(trained, tmp_path / "bundle")
    runtime = _environment(dataset, 5, 9)
    policy = _load(publication, runtime)
    binding = policy.runtime_binding
    assert binding.native_interval_seconds is None
    assert binding.runtime_horizon_seconds == 122 * 86400
    assert (
        PortfolioObservationRuntimeBinding.from_payload(
            binding.to_payload(), expected_digest=binding.digest
        )
        == binding
    )
    for change in (
        {"schema": "shared_portfolio_observation_runtime_binding_v1"},
        {"runtime_horizon_seconds": 120 * 86400},
        {"runtime_start_ns": binding.runtime_start_ns + 1},
    ):
        changed = binding.to_payload() | change
        with pytest.raises(ValueError):
            PortfolioObservationRuntimeBinding.from_payload(
                changed, expected_digest=content_digest(changed)
            )
    raw, _ = runtime.reset(seed=43)
    rewards = []
    done = False
    while not done:
        raw, reward, done, truncated, info = runtime.step(policy.predict(raw))
        rewards.append(reward)
    assert not truncated and info["settlement_complete"]
    replay = run_shared_portfolio_replay(
        dataset,
        policy,
        start_index=5,
        stop_index=9,
        initial_capital=1000.0,
        execution_cost=_cost(),
        risk=PreTradeRisk(_risk()),
        settle_terminal_position=True,
        capture_ledger_evidence=True,
        native_calendar_month=True,
    )
    assert replay.book.portfolio_value == pytest.approx(1000 * (1 + sum(rewards)))
    assert all(q == 0 for q in replay.book.exact_quantities)
    payload = native_calendar_month_return_payload(
        dataset, replay.returns, start_index=5
    )
    assert payload["elapsed_years"] == {"numerator": 122, "denominator": 365}

    artifact = publish_market_dataset_artifact(tmp_path / "dataset", dataset)
    readback, paired = run_published_portfolio_inference_replay(
        dataset_artifact_root=artifact.root,
        expected_dataset_artifact_sha256=artifact.artifact_digest,
        bundle_root=publication.root,
        expected_bundle_sha256=publication.bundle_sha256,
        expected_manifest_sha256=publication.training_manifest_sha256,
        expected_completion_sha256=publication.training_completion_sha256,
        expected_training_context_sha256=publication.training_context_sha256,
        expected_checkpoint_sha256=publication.checkpoint_sha256,
        bound=runtime.bound,
        start_index=5,
        stop_index=9,
        initial_capital=1000.0,
        target_abs_weights=runtime.target_abs_weights,
        execution_cost=_cost(),
        risk_config=_risk(),
        market_order_profile=None,
        capture_ledger_evidence=True,
    )
    assert readback.artifact_digest == artifact.artifact_digest
    assert paired.returns == replay.returns
    assert paired.book.portfolio_value == replay.book.portfolio_value


def test_monthly_flat_interest_matches_independent_actual_day_oracle():
    dataset = replace(
        _month_dataset(),
        cash_rate=np.full(10, 0.12),
        identity_payload_json=None,
    ).with_content_identity({"fixture": "calendar-interest"})
    env = _environment(dataset)
    env.reset(seed=42)
    rewards = []
    done = False
    while not done:
        _, reward, done, truncated, info = env.step(
            np.array([ACTION_HOLD, ACTION_HOLD])
        )
        rewards.append(reward)
    expected = Fraction(1000)
    # Literal Jan-May 2024, including leap February and terminal interval.
    for days in (31, 29, 31, 30, 31):
        expected *= 1 + Fraction(12, 100) * Fraction(days, 365)
    assert not truncated and info["settlement_complete"]
    assert env._session is not None
    assert env._session.book.cash == pytest.approx(float(expected), rel=1e-13)
    assert sum(rewards) == pytest.approx(float(expected / 1000 - 1), abs=1e-13)


@pytest.mark.parametrize(
    "field,value",
    [
        ("economic_horizon_months", True),
        ("economic_horizon_months", 0),
        ("economic_horizon_seconds", -1),
        ("rollout_steps", 2.0),
        ("gamma", True),
        ("gamma", 0.99),
        ("gae_lambda", 1.1),
        ("reward_schema", "net_log_return_v1"),
    ],
)
def test_monthly_clock_rejects_invalid_fields(field, value):
    with pytest.raises(ValueError):
        replace(_bound().clock, **{field: value})


@pytest.mark.parametrize(
    "change",
    [
        {"schema": "shared_portfolio_observation_transform_v1"},
        {"economic_horizon_seconds": 150 * 86400},
        {"objective_start_ns": 1704067200000000001},
        {"native_interval_seconds": 30 * 86400},
    ],
)
def test_monthly_transform_rejects_rehashed_malformed_clock(change):
    transform = fit_portfolio_observation_transform(_environment(_month_dataset()))
    payload = transform.to_payload() | change
    with pytest.raises(ValueError):
        PortfolioObservationTransform.from_payload(
            payload, expected_digest=content_digest(payload)
        )


def test_monthly_fit_refuses_unavailable_terminal_mark():
    dataset = _month_dataset()
    availability = dataset.resolved_array("available_at").copy()
    information = dataset.resolved_array("information_available").copy()
    availability[5, 0] += np.timedelta64(1, "ns")
    information[5, 0] = False
    late = replace(
        dataset,
        available_at=availability,
        information_available=information,
        identity_payload_json=None,
    ).with_content_identity({"fixture": "month-late-terminal"})
    with pytest.raises(ValueError, match="unavailable"):
        _environment(late)


@pytest.mark.parametrize("mutation", ["clock", "capital", "latency"])
def test_monthly_reset_rechecks_objective_and_terminal_before_session(mutation):
    env = _environment(_month_dataset())
    if mutation == "clock":
        env.bound = BoundObjectiveClock(
            replace(env.bound.objective, terminal_valuation="marked_continuation"),
            env.bound.clock,
        )
    elif mutation == "capital":
        env.initial_capital = 500
    else:
        env.execution_cost = replace(env.execution_cost, order_latency_bars=4)
    with pytest.raises(ValueError, match="calendar month Gym"):
        env.reset()
    assert env._session is None


def test_monthly_four_family_fixed_manifest_refuses_new_clock():
    from trade_rl.evaluation.experiments.four_family_input_manifest import WindowSpec

    with pytest.raises(ValueError, match="calendar month"):
        WindowSpec("training", 0, 5, _bound())


def test_calendar_timestamp_arithmetic_pre_epoch_and_leap_year():
    from trade_rl.strategies.rl.portfolio_transform import _native_timestamp_ns_after

    for year in (1900, 1969, 1970, 2000, 2024, 2100, 2200):
        start = int(np.datetime64(f"{year}-01-01", "ns").astype(np.int64))
        for step, month in enumerate(("01", "02", "03", "04")):
            expected = int(np.datetime64(f"{year}-{month}-01", "ns").astype(np.int64))
            assert _native_timestamp_ns_after(start, step, None) == expected
    with pytest.raises(ValueError):
        _native_timestamp_ns_after(1, 1, None)
    with pytest.raises(ValueError):
        _native_timestamp_ns_after(0, 10**20, None)
