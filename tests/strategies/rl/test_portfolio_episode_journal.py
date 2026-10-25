from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.risk.pretrade import PreTradeRiskConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.objectives.binding import BoundObjectiveClock
from trade_rl.simulation.objectives.clock import FinancialClockContract
from trade_rl.simulation.objectives.contract import CapitalContract, ObjectiveContract
from trade_rl.strategies.rl.portfolio_env import (
    ACTION_FLAT,
    ACTION_HOLD,
    ACTION_SHORT,
    SharedPortfolioGymEnv,
)
from trade_rl.strategies.rl.portfolio_episode_journal import (
    PortfolioEpisodeJournalWrapper,
)

_START = datetime(2026, 1, 1, tzinfo=UTC)


def _environment(*, insolvency_gap: bool = False) -> SharedPortfolioGymEnv:
    close = np.full((6, 1), 100.0, dtype=np.float64)
    open_price = close.copy()
    fee_rate = np.zeros_like(close)
    if insolvency_gap:
        close[2:, 0] = 500.0
        open_price[2:, 0] = 500.0
        fee_rate[:] = 0.01
    zeros = np.zeros_like(close)
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        6
    ) * np.timedelta64(1, "h")
    dataset = MarketDataset(
        dataset_id="d" * 64,
        symbols=("AAA",),
        timestamps=timestamps,
        features=np.arange(6, dtype=np.float32).reshape(6, 1, 1),
        global_features=np.arange(6, dtype=np.float32).reshape(6, 1),
        open=open_price,
        high=np.maximum(open_price, close),
        low=np.minimum(open_price, close),
        close=close,
        volume=np.full_like(close, 1_000_000.0),
        funding_rate=zeros.copy(),
        funding_due=None,
        borrow_rate=zeros.copy(),
        tradable=np.ones((6, 1), dtype=np.bool_),
        feature_available=np.ones((6, 1, 1), dtype=np.bool_),
        feature_names=("literal",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
        fee_rate=fee_rate,
        maker_fee_rate=zeros.copy(),
        taker_fee_rate=zeros.copy(),
        spread_rate=zeros.copy(),
        contract_multipliers=np.ones(1, dtype=np.float64),
    )
    stop = _START + timedelta(hours=5)
    bound = BoundObjectiveClock(
        ObjectiveContract(
            capital=CapitalContract("shared_portfolio", "USDT", (1_000.0,)),
            evaluation_start=_START,
            evaluation_stop_exclusive=stop,
            terminal_valuation="settled",
            economics_digest="a" * 64,
            risk_digest="b" * 64,
            deployment_recipe_digest="c" * 64,
            maximum_drawdown=0.20,
        ),
        FinancialClockContract(
            decision_interval_seconds=3_600,
            execution_interval_seconds=3_600,
            reward_interval_seconds=3_600,
            economic_horizon_seconds=5 * 3_600,
            rollout_steps=1,
            gamma=1.0,
            gae_lambda=0.95,
            reward_schema="equity_delta_v1",
        ),
    )
    execution_cost = ExecutionCostConfig.zero()
    if insolvency_gap:
        execution_cost = replace(execution_cost, fee_rate=0.01)
    return SharedPortfolioGymEnv(
        dataset,
        start_index=0,
        stop_index=5,
        bound=bound,
        target_abs_weights=(0.5,),
        execution_cost=execution_cost,
        risk_config=PreTradeRiskConfig(
            max_gross=1.0,
            max_abs_weight=0.5,
            max_turnover=None,
            drawdown_start=0.20,
            drawdown_stop=0.20,
        ),
    )


def _events(root: Path) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / "events").glob("event-*.json"))
    ]


def test_journal_root_collision_fails_before_environment_reset(tmp_path: Path) -> None:
    root = tmp_path / "already-owned"
    root.mkdir()
    marker = root / "preserve.txt"
    marker.write_text("keep", encoding="utf-8")
    env = _environment()

    with pytest.raises(FileExistsError):
        PortfolioEpisodeJournalWrapper(
            env,
            root=root,
            declared_context_digest="a" * 64,
        )

    assert env._session is None
    assert marker.read_text(encoding="utf-8") == "keep"


def test_wrapper_preserves_source_transition_and_writes_terminal_facts(
    tmp_path: Path,
) -> None:
    raw = _environment()
    journal = PortfolioEpisodeJournalWrapper(
        _environment(),
        root=tmp_path / "journal",
        declared_context_digest="a" * 64,
    )

    raw_observation, raw_reset_info = raw.reset(seed=19)
    wrapped_observation, wrapped_reset_info = journal.reset(seed=19)
    assert wrapped_reset_info == raw_reset_info
    for key in raw.observation_keys:
        np.testing.assert_array_equal(wrapped_observation[key], raw_observation[key])

    terminal_info: dict[str, object] | None = None
    for _ in range(8):
        action = np.asarray([ACTION_FLAT], dtype=np.int64)
        raw_result = raw.step(action)
        wrapped_result = journal.step(action)
        np.testing.assert_array_equal(
            wrapped_result[0]["clock"], raw_result[0]["clock"]
        )
        assert wrapped_result[1:4] == raw_result[1:4]
        assert wrapped_result[4] == raw_result[4]
        if wrapped_result[2]:
            terminal_info = wrapped_result[4]
            break

    assert terminal_info is not None
    terminal_state = terminal_info["terminal_state"]
    assert isinstance(terminal_state, dict)
    assert terminal_state["objective_endpoint_valid"] is True
    completed = [
        event
        for event in _events(tmp_path / "journal")
        if event["event"] == "step_result" and event["data"]["terminated"]
    ]
    assert len(completed) == 1
    assert completed[0]["data"]["terminal_state"]["objective_endpoint_valid"] is True
    assert completed[0]["data"]["raw_action_codes"] == [ACTION_FLAT]


def test_dummy_vec_env_records_invalid_episode_before_auto_reset(
    tmp_path: Path,
) -> None:
    dummy_vec_env = pytest.importorskip("stable_baselines3.common.vec_env")
    root = tmp_path / "journal"
    env = _environment(insolvency_gap=True)
    original_reset = env.reset
    reset_count = 0

    def inspect_auto_reset(
        *, seed: int | None = None, options: dict[str, object] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        nonlocal reset_count
        if reset_count:
            terminal_events = [
                event
                for event in _events(root)
                if event["event"] == "step_result" and event["data"]["terminated"]
            ]
            assert terminal_events, "terminal journal event must precede SB3 reset"
        reset_count += 1
        return original_reset(seed=seed, options=options)

    env.reset = inspect_auto_reset  # type: ignore[method-assign]
    wrapped = PortfolioEpisodeJournalWrapper(
        env,
        root=root,
        declared_context_digest="b" * 64,
    )
    vector_env = dummy_vec_env.DummyVecEnv([lambda: wrapped])
    try:
        vector_env.seed(7)
        vector_env.reset()
        vector_env.step(np.asarray([[ACTION_SHORT]], dtype=np.int64))
        _, _, dones, infos = vector_env.step(
            np.asarray([[ACTION_HOLD]], dtype=np.int64)
        )
        assert dones.tolist() == [True]
        assert infos[0]["terminal_state"]["termination_reason"] == "insolvency"
        assert reset_count == 2

        for _ in range(8):
            _, _, dones, _ = vector_env.step(
                np.asarray([[ACTION_FLAT]], dtype=np.int64)
            )
            if dones[0]:
                break
        assert dones.tolist() == [True]
    finally:
        vector_env.close()

    terminal = [
        event["data"]["terminal_state"]
        for event in _events(root)
        if event["event"] == "step_result" and event["data"]["terminated"]
    ]
    assert terminal[0]["termination_reason"] == "insolvency"
    assert terminal[0]["settlement_complete"] is False
    assert terminal[0]["objective_endpoint_valid"] is False
    assert terminal[1]["settlement_complete"] is True
    assert terminal[1]["objective_endpoint_valid"] is True


def test_step_write_failure_prevents_dummy_vec_auto_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dummy_vec_env = pytest.importorskip("stable_baselines3.common.vec_env")
    import trade_rl.strategies.rl.portfolio_episode_journal as journal_module

    root = tmp_path / "journal"
    env = _environment(insolvency_gap=True)
    original_reset = env.reset
    reset_count = 0

    def count_reset(
        *, seed: int | None = None, options: dict[str, object] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        nonlocal reset_count
        reset_count += 1
        return original_reset(seed=seed, options=options)

    env.reset = count_reset  # type: ignore[method-assign]
    wrapped = journal_module.PortfolioEpisodeJournalWrapper(
        env,
        root=root,
        declared_context_digest="c" * 64,
    )
    vector_env = dummy_vec_env.DummyVecEnv([lambda: wrapped])
    vector_env.seed(7)
    vector_env.reset()
    vector_env.step(np.asarray([[ACTION_SHORT]], dtype=np.int64))
    original_write = journal_module.atomic_write_bytes

    def fail_terminal_result(path: Path, payload: bytes) -> Path:
        record = json.loads(payload.decode("utf-8"))
        if record.get("event") == "step_result" and record["data"]["terminated"]:
            raise OSError("injected terminal journal failure")
        return original_write(path, payload)

    monkeypatch.setattr(journal_module, "atomic_write_bytes", fail_terminal_result)
    try:
        with pytest.raises(OSError, match="injected terminal journal failure"):
            vector_env.step(np.asarray([[ACTION_HOLD]], dtype=np.int64))
        assert reset_count == 1
        with pytest.raises(RuntimeError, match="poisoned"):
            wrapped.reset(seed=8)
        assert reset_count == 1
        events = _events(root)
        assert any(event["event"] == "step_started" for event in events)
        assert not any(
            event["event"] == "step_result" and event["data"]["terminated"]
            for event in events
        )
    finally:
        vector_env.close()


def test_invalid_action_is_recorded_without_coercion_and_poisoned(
    tmp_path: Path,
) -> None:
    root = tmp_path / "journal"
    env = _environment()
    wrapped = PortfolioEpisodeJournalWrapper(
        env,
        root=root,
        declared_context_digest="d" * 64,
    )
    wrapped.reset(seed=11)

    with pytest.raises(TypeError, match="action codes"):
        wrapped.step(np.asarray([True], dtype=np.bool_))

    events = _events(root)
    failed = [event for event in events if event["event"] == "step_failed"]
    assert len(failed) == 1
    started = [event for event in events if event["event"] == "step_started"]
    assert started[-1]["data"]["submitted_action"]["values"] == [True]
    assert started[-1]["data"]["submitted_action"]["dtype"] == "bool"
    assert failed[0]["data"]["native_result_returned"] is False
    assert not any(event["event"] == "step_result" for event in events)
    with pytest.raises(RuntimeError, match="poisoned"):
        wrapped.step(np.asarray([ACTION_FLAT], dtype=np.int64))


def test_manual_reset_marks_open_episode_incomplete_before_new_start(
    tmp_path: Path,
) -> None:
    root = tmp_path / "journal"
    wrapped = PortfolioEpisodeJournalWrapper(
        _environment(),
        root=root,
        declared_context_digest="e" * 64,
    )
    wrapped.reset(seed=1)
    wrapped.step(np.asarray([ACTION_FLAT], dtype=np.int64))
    wrapped.reset(seed=2)

    events = _events(root)
    interrupted = next(
        i for i, event in enumerate(events) if event["event"] == "episode_interrupted"
    )
    second_start = next(
        i
        for i, event in enumerate(events)
        if event["event"] == "episode_started" and event["episode_index"] == 1
    )
    assert interrupted < second_start
    assert events[interrupted]["data"]["reason"] == "manual_reset"
    assert events[interrupted]["data"]["completed"] is False
    wrapped.close()


def test_reset_failure_is_recorded_and_cannot_be_retried(
    tmp_path: Path,
) -> None:
    root = tmp_path / "journal"
    env = _environment()
    original_reset = env.reset
    reset_calls = 0

    def mutate_then_fail(
        *, seed: int | None = None, options: dict[str, object] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        nonlocal reset_calls
        reset_calls += 1
        original_reset(seed=seed, options=options)
        raise OSError("injected post-reset failure")

    env.reset = mutate_then_fail  # type: ignore[method-assign]
    wrapped = PortfolioEpisodeJournalWrapper(
        env,
        root=root,
        declared_context_digest="f" * 64,
    )

    with pytest.raises(OSError, match="injected post-reset failure"):
        wrapped.reset(seed=31)

    events = _events(root)
    assert [event["event"] for event in events] == [
        "episode_started",
        "episode_reset_failed",
    ]
    assert events[1]["data"]["native_result_returned"] is False
    assert events[1]["data"]["native_state_outcome"] == "unknown"
    with pytest.raises(RuntimeError, match="poisoned"):
        wrapped.reset(seed=32)
    assert reset_calls == 1
