from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from trade_rl.data.artifacts.publication import publish_market_dataset_artifact
from trade_rl.data.market import MarketDataset
from trade_rl.risk.pretrade import PreTradeRiskConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.objectives.binding import BoundObjectiveClock
from trade_rl.simulation.objectives.clock import FinancialClockContract
from trade_rl.simulation.objectives.contract import CapitalContract, ObjectiveContract
from trade_rl.strategies.forecasts.ridge import RidgeForecastModel
from trade_rl.strategies.forecasts.ridge_artifact import (
    RidgeForecastArtifactBinding,
    publish_ridge_forecast_artifact,
)
from trade_rl.strategies.portfolio import (
    PortfolioDecisionSnapshot,
    PortfolioTargetProposal,
)
from trade_rl.strategies.rules.trend import TrendIntentConfig

_START = datetime(2026, 1, 1, tzinfo=UTC)
_FAMILY_ORDER = ("cash", "fixed_trend", "ridge_forecast", "joint_ppo")


def _dataset(*, force_insolvency: bool = False) -> MarketDataset:
    n_bars = 7
    n_symbols = 2
    close = np.full((n_bars, n_symbols), 100.0, dtype=np.float64)
    if force_insolvency:
        close[3:, 0] = 1.0
        close[3:, 1] = 500.0
    zeros = np.zeros_like(close)
    fee = np.full_like(close, 0.01)
    features = np.tile(np.asarray([[1.0], [-1.0]], dtype=np.float32), (n_bars, 1, 1))
    funding = np.zeros_like(close)
    funding[4] = (0.01, 0.005)
    funding_due = np.zeros_like(close, dtype=np.bool_)
    funding_due[4] = True
    borrow = np.zeros_like(close)
    borrow[:, 1] = 0.365
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        n_bars
    ) * np.timedelta64(1, "h")
    return MarketDataset(
        dataset_id="a" * 64,
        symbols=("AAA", "BBB"),
        timestamps=timestamps,
        features=features,
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=close.copy(),
        high=close.copy(),
        low=close.copy(),
        close=close,
        volume=np.full_like(close, 1_000_000.0),
        funding_rate=funding,
        funding_due=funding_due,
        borrow_rate=borrow,
        borrow_available=np.ones_like(close, dtype=np.bool_),
        tradable=np.ones_like(close, dtype=np.bool_),
        feature_available=np.ones((n_bars, n_symbols, 1), dtype=np.bool_),
        feature_names=("signal",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
        fee_rate=fee,
        maker_fee_rate=zeros.copy(),
        taker_fee_rate=zeros.copy(),
        spread_rate=zeros.copy(),
        contract_multipliers=np.ones(n_symbols, dtype=np.float64),
    ).with_content_identity({"fixture": "four-family-portfolio-replay"})


def _prepared_inputs(
    tmp_path: Path, *, force_insolvency: bool = False
) -> tuple[dict[str, Any], MarketDataset]:
    dataset = _dataset(force_insolvency=force_insolvency)
    dataset_artifact = publish_market_dataset_artifact(tmp_path / "dataset", dataset)
    model = RidgeForecastModel(
        feature_indices=(0,),
        feature_mean=np.asarray([0.0], dtype=np.float64),
        feature_scale=np.asarray([1.0], dtype=np.float64),
        coefficients=np.asarray([0.01], dtype=np.float64),
        intercept=0.0,
        horizon_hours=1,
        alpha=1.0,
        n_samples=2,
        fit_cutoff=dataset.timestamps[1],
    )
    binding = RidgeForecastArtifactBinding(
        dataset_id=dataset.dataset_id,
        dataset_schema_version=dataset_artifact.schema_version,
        dataset_artifact_sha256=dataset_artifact.artifact_digest,
        fit_symbol_indices=(0, 1),
        fit_symbol_names=dataset.symbols,
        feature_names=dataset.feature_names,
        feature_indices=model.feature_indices,
        fit_cutoff_ns=int(model.fit_cutoff.astype("datetime64[ns]").astype(np.int64)),
        horizon_hours=model.horizon_hours,
        alpha_hex=model.alpha.hex(),
        n_samples=model.n_samples,
        vintage_id="synthetic-vintage",
        source_commit="1" * 40,
        implementation_sha256="2" * 64,
        runtime_sha256="3" * 64,
    )
    ridge_root = tmp_path / "ridge"
    ridge_digest = publish_ridge_forecast_artifact(ridge_root, model, binding)

    stop_index = 5
    risk_config = PreTradeRiskConfig(
        max_gross=1.0,
        max_abs_weight=0.5,
        max_turnover=None,
        drawdown_start=0.1,
        drawdown_stop=0.2,
    )
    execution_cost = ExecutionCostConfig(
        fee_rate=0.01,
        maker_fee_rate=0.0,
        taker_fee_rate=0.0,
        spread_rate=0.0,
        impact_rate=0.0,
        max_participation_rate=1.0,
        max_leverage=1.0,
        maintenance_margin_rate=0.0,
        borrow_rate_multiplier=1.0,
        order_latency_bars=0,
        allow_short=True,
    )
    start = _START + timedelta(hours=1)
    stop = _START + timedelta(hours=stop_index)
    bound = BoundObjectiveClock(
        ObjectiveContract(
            capital=CapitalContract("shared_portfolio", "USDT", (1_000.0,)),
            evaluation_start=start,
            evaluation_stop_exclusive=stop,
            terminal_valuation="settled",
            economics_digest="4" * 64,
            risk_digest="5" * 64,
            deployment_recipe_digest="6" * 64,
            maximum_drawdown=0.2,
        ),
        FinancialClockContract(
            decision_interval_seconds=3_600,
            execution_interval_seconds=3_600,
            reward_interval_seconds=3_600,
            economic_horizon_seconds=(stop_index - 1) * 3_600,
            rollout_steps=4,
            gamma=1.0,
            gae_lambda=0.95,
            reward_schema="equity_delta_v1",
        ),
    )
    return (
        {
            "dataset_artifact_root": dataset_artifact.root,
            "expected_dataset_artifact_sha256": dataset_artifact.artifact_digest,
            "expected_symbols": dataset.symbols,
            "ridge_artifact_root": ridge_root,
            "expected_ridge_artifact_sha256": ridge_digest,
            "expected_ridge_binding": binding,
            "bundle_root": tmp_path / "ppo-bundle",
            "expected_bundle_sha256": "7" * 64,
            "expected_manifest_sha256": "8" * 64,
            "expected_completion_sha256": "9" * 64,
            "expected_training_context_sha256": "a" * 64,
            "expected_checkpoint_sha256": "b" * 64,
            "bound": bound,
            "start_index": 1,
            "stop_index": stop_index,
            "initial_capital": 1_000.0,
            "target_abs_weights": (0.5, 0.5),
            "trend_config": TrendIntentConfig(
                signal_index=0,
                entry_threshold=0.5,
                exit_threshold=0.1,
            ),
            "ridge_entry_threshold": 0.005,
            "ridge_exit_threshold": 0.001,
            "ridge_one_way_switch_cost": None,
            "execution_cost": execution_cost,
            "risk_config": risk_config,
            "market_order_profile": None,
        },
        dataset,
    )


class _FixedPPOPolicy:
    def __init__(self) -> None:
        self.indexes: list[int] = []

    def decide_portfolio(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> PortfolioTargetProposal:
        self.indexes.append(snapshot.index)
        return PortfolioTargetProposal(target_weights=(0.5, -0.5))


def test_four_family_comparison_preflights_once_then_reuses_native_engine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.data.artifacts import publication as publication_module
    from trade_rl.evaluation import replay as replay_module
    from trade_rl.evaluation.runs import (
        portfolio_family_comparison as comparison_module,
    )
    from trade_rl.evaluation.runs.portfolio_family_comparison import (
        run_four_family_shared_portfolio_replay,
    )
    from trade_rl.simulation.stateful import portfolio_session as session_module
    from trade_rl.strategies.rl import portfolio_env as env_module

    inputs, expected_dataset = _prepared_inputs(tmp_path)
    loaded_pairs: list[tuple[Any, MarketDataset]] = []
    validation_envs: list[object] = []
    session_records: list[tuple[object, MarketDataset, dict[str, object]]] = []
    replay_calls: list[tuple[MarketDataset, object, dict[str, object]]] = []
    ppo_policy = _FixedPPOPolicy()

    load_dataset = publication_module.load_published_market_dataset_artifact

    def record_dataset_load(root: str | Path) -> tuple[Any, MarketDataset]:
        pair = load_dataset(root)
        loaded_pairs.append(pair)
        return pair

    def load_fixed_ppo_policy(
        root: str | Path,
        *,
        expected_bundle_sha256: str,
        expected_manifest_sha256: str,
        expected_completion_sha256: str,
        expected_training_context_sha256: str,
        expected_checkpoint_sha256: str,
        runtime_env: object,
    ) -> _FixedPPOPolicy:
        del (
            root,
            expected_bundle_sha256,
            expected_manifest_sha256,
            expected_completion_sha256,
            expected_training_context_sha256,
            expected_checkpoint_sha256,
        )
        assert runtime_env.dataset is loaded_pairs[0][1]
        assert runtime_env._session is None
        validation_envs.append(runtime_env)
        return ppo_policy

    original_env_init = env_module.SharedPortfolioGymEnv.__init__

    def record_env_init(
        self: object, dataset: MarketDataset, *args: object, **kwargs: object
    ) -> None:
        original_env_init(self, dataset, *args, **kwargs)
        validation_envs.append(self)

    def forbidden_validation_step(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("the PPO validation Gym must remain unreset and unstepped")

    original_session_init = session_module.SharedPortfolioSession.__init__

    def record_session_init(
        self: object, dataset: MarketDataset, *args: object, **kwargs: object
    ) -> None:
        session_records.append((self, dataset, dict(kwargs)))
        original_session_init(self, dataset, *args, **kwargs)

    run_replay = replay_module.run_shared_portfolio_replay

    def record_replay(
        dataset: MarketDataset, policy: object, **kwargs: object
    ) -> object:
        replay_calls.append((dataset, policy, dict(kwargs)))
        return run_replay(dataset, policy, **kwargs)

    monkeypatch.setattr(
        comparison_module,
        "load_published_market_dataset_artifact",
        record_dataset_load,
    )
    monkeypatch.setattr(
        comparison_module,
        "load_portfolio_inference_bundle",
        load_fixed_ppo_policy,
    )
    monkeypatch.setattr(env_module.SharedPortfolioGymEnv, "__init__", record_env_init)
    monkeypatch.setattr(
        env_module.SharedPortfolioGymEnv, "reset", forbidden_validation_step
    )
    monkeypatch.setattr(
        env_module.SharedPortfolioGymEnv, "step", forbidden_validation_step
    )
    monkeypatch.setattr(
        session_module.SharedPortfolioSession,
        "__init__",
        record_session_init,
    )
    monkeypatch.setattr(comparison_module, "run_shared_portfolio_replay", record_replay)

    result = run_four_family_shared_portfolio_replay(**inputs)

    assert (
        result.dataset_artifact.artifact_digest
        == inputs["expected_dataset_artifact_sha256"]
    )
    assert tuple(outcome.family for outcome in result.families) == _FAMILY_ORDER
    assert len(loaded_pairs) == 1
    paired_dataset = loaded_pairs[0][1]
    assert paired_dataset.dataset_id == expected_dataset.dataset_id
    assert len(validation_envs) == 2  # one constructor spy plus one loader observation
    assert validation_envs[0] is validation_envs[1]
    assert validation_envs[0].dataset is paired_dataset
    assert validation_envs[0]._session is None
    assert len(replay_calls) == 4
    assert len({id(policy) for _, policy, _ in replay_calls}) == 4
    assert all(dataset is paired_dataset for dataset, _, _ in replay_calls)
    assert len(session_records) == 4
    assert len({id(session) for session, _, _ in session_records}) == 4
    assert all(dataset is paired_dataset for _, dataset, _ in session_records)

    risk_objects = []
    for _, _, kwargs in session_records:
        assert kwargs["execution_cost"] is inputs["execution_cost"]
        assert kwargs["market_order_profile"] is inputs["market_order_profile"]
        assert kwargs["settle_terminal_position"] is True
        risk = kwargs["risk"]
        assert risk.config is inputs["risk_config"]
        risk_objects.append(risk)
    assert len({id(risk) for risk in risk_objects}) == 4

    expected_intervals = inputs["stop_index"] - inputs["start_index"]
    for outcome in result.families:
        assert outcome.expected_interval_count == expected_intervals
        assert outcome.actual_interval_count == expected_intervals
        assert outcome.termination_reason is None
        assert outcome.terminal_flat is True
        assert outcome.no_active_orders is True
        assert outcome.replay.ledger_evidence is not None
    assert ppo_policy.indexes == [
        decision.index for decision in result.families[-1].replay.decisions
    ]
    assert all(outcome.window_complete for outcome in result.families)

    for outcome in result.families[1:]:
        evidence = outcome.replay.ledger_evidence
        assert evidence is not None
        assert evidence.intervals.final_total_cost > 0.0
        assert evidence.intervals.final_funding_pnl != 0.0
        assert evidence.intervals.final_borrow_cost > 0.0


def test_dataset_identity_mismatch_fails_before_any_native_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.evaluation.runs import (
        portfolio_family_comparison as comparison_module,
    )
    from trade_rl.evaluation.runs.portfolio_family_comparison import (
        run_four_family_shared_portfolio_replay,
    )

    inputs, _ = _prepared_inputs(tmp_path)
    inputs["expected_dataset_artifact_sha256"] = "c" * 64
    replay_calls = 0

    def forbidden_replay(*args: object, **kwargs: object) -> object:
        nonlocal replay_calls
        replay_calls += 1
        raise AssertionError("native replay must not start on an identity mismatch")

    monkeypatch.setattr(
        comparison_module, "run_shared_portfolio_replay", forbidden_replay
    )

    with pytest.raises(ValueError, match="Dataset artifact"):
        run_four_family_shared_portfolio_replay(**inputs)
    assert replay_calls == 0


def test_bundle_validation_failure_precedes_every_native_arm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.evaluation.runs import (
        portfolio_family_comparison as comparison_module,
    )
    from trade_rl.evaluation.runs.portfolio_family_comparison import (
        run_four_family_shared_portfolio_replay,
    )

    inputs, _ = _prepared_inputs(tmp_path)
    replay_calls = 0

    def fail_bundle_load(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise ValueError("synthetic bundle binding mismatch")

    def forbidden_replay(*args: object, **kwargs: object) -> object:
        nonlocal replay_calls
        replay_calls += 1
        raise AssertionError("native replay must not start before bundle validation")

    monkeypatch.setattr(
        comparison_module,
        "load_portfolio_inference_bundle",
        fail_bundle_load,
    )
    monkeypatch.setattr(
        comparison_module, "run_shared_portfolio_replay", forbidden_replay
    )

    with pytest.raises(ValueError, match="synthetic bundle binding mismatch"):
        run_four_family_shared_portfolio_replay(**inputs)
    assert replay_calls == 0


def test_ridge_dataset_binding_mismatch_fails_before_any_native_arm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.evaluation.runs import (
        portfolio_family_comparison as comparison_module,
    )
    from trade_rl.evaluation.runs.portfolio_family_comparison import (
        run_four_family_shared_portfolio_replay,
    )

    inputs, _ = _prepared_inputs(tmp_path)
    inputs["expected_ridge_binding"] = replace(
        inputs["expected_ridge_binding"], dataset_id="d" * 64
    )
    replay_calls = 0

    def forbidden_replay(*args: object, **kwargs: object) -> object:
        nonlocal replay_calls
        replay_calls += 1
        raise AssertionError("native replay must not start on a Ridge binding mismatch")

    monkeypatch.setattr(
        comparison_module, "run_shared_portfolio_replay", forbidden_replay
    )

    with pytest.raises(ValueError, match="context"):
        run_four_family_shared_portfolio_replay(**inputs)
    assert replay_calls == 0


def test_native_insolvency_is_retained_as_an_incomplete_family_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.evaluation.runs import (
        portfolio_family_comparison as comparison_module,
    )
    from trade_rl.evaluation.runs.portfolio_family_comparison import (
        run_four_family_shared_portfolio_replay,
    )

    inputs, _ = _prepared_inputs(tmp_path, force_insolvency=True)
    monkeypatch.setattr(
        comparison_module,
        "load_portfolio_inference_bundle",
        lambda *args, **kwargs: _FixedPPOPolicy(),
    )

    result = run_four_family_shared_portfolio_replay(**inputs)

    assert tuple(outcome.family for outcome in result.families) == _FAMILY_ORDER
    assert result.families[0].window_complete is True
    insolvent = [outcome for outcome in result.families if outcome.insolvent]
    assert insolvent
    for outcome in insolvent:
        assert outcome.termination_reason == "insolvency"
        assert outcome.actual_interval_count < outcome.expected_interval_count
        assert outcome.window_complete is False
        assert outcome.replay.book.insolvent is True
        assert outcome.terminal_flat is True
        assert outcome.no_active_orders is True


def test_completed_family_is_observed_before_a_later_arm_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.evaluation.runs import (
        portfolio_family_comparison as comparison_module,
    )
    from trade_rl.evaluation.runs.portfolio_family_comparison import (
        run_four_family_shared_portfolio_replay,
    )

    inputs, _ = _prepared_inputs(tmp_path)
    completed: list[Any] = []
    replay_calls = 0
    actual_replay = comparison_module.run_shared_portfolio_replay

    def fail_second_arm(*args: object, **kwargs: object) -> object:
        nonlocal replay_calls
        replay_calls += 1
        if replay_calls == 2:
            raise RuntimeError("synthetic second-family failure")
        return actual_replay(*args, **kwargs)

    monkeypatch.setattr(
        comparison_module, "run_shared_portfolio_replay", fail_second_arm
    )
    monkeypatch.setattr(
        comparison_module,
        "load_portfolio_inference_bundle",
        lambda *args, **kwargs: _FixedPPOPolicy(),
    )

    with pytest.raises(RuntimeError, match="synthetic second-family failure"):
        run_four_family_shared_portfolio_replay(
            **inputs,
            on_family_completed=completed.append,
        )

    assert replay_calls == 2
    assert len(completed) == 1
    assert completed[0].family == "cash"
    assert completed[0].terminal_flat is True
    assert completed[0].no_active_orders is True


def test_completion_observer_failure_stops_before_the_next_arm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.evaluation.runs import (
        portfolio_family_comparison as comparison_module,
    )
    from trade_rl.evaluation.runs.portfolio_family_comparison import (
        run_four_family_shared_portfolio_replay,
    )

    inputs, _ = _prepared_inputs(tmp_path)
    completed: list[Any] = []
    replay_calls = 0
    actual_replay = comparison_module.run_shared_portfolio_replay

    def record_replay(*args: object, **kwargs: object) -> object:
        nonlocal replay_calls
        replay_calls += 1
        return actual_replay(*args, **kwargs)

    def fail_observer(outcome: object) -> None:
        completed.append(outcome)
        raise RuntimeError("synthetic observer persistence failure")

    monkeypatch.setattr(comparison_module, "run_shared_portfolio_replay", record_replay)
    monkeypatch.setattr(
        comparison_module,
        "load_portfolio_inference_bundle",
        lambda *args, **kwargs: _FixedPPOPolicy(),
    )

    with pytest.raises(RuntimeError, match="synthetic observer persistence failure"):
        run_four_family_shared_portfolio_replay(
            **inputs,
            on_family_completed=fail_observer,
        )

    assert replay_calls == 1
    assert len(completed) == 1
    assert completed[0].family == "cash"
