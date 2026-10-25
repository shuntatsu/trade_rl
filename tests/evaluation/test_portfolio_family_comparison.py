from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from trade_rl.data.artifacts.publication import publish_market_dataset_artifact
from trade_rl.data.market import MarketDataset
from trade_rl.risk.pretrade import PreTradeRisk, PreTradeRiskConfig
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
    observed_bundle_pins: dict[str, str] = {}
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
        del root
        observed_bundle_pins.update(
            {
                "expected_bundle_sha256": expected_bundle_sha256,
                "expected_manifest_sha256": expected_manifest_sha256,
                "expected_completion_sha256": expected_completion_sha256,
                "expected_training_context_sha256": expected_training_context_sha256,
                "expected_checkpoint_sha256": expected_checkpoint_sha256,
            }
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
    assert observed_bundle_pins == {
        "expected_bundle_sha256": inputs["expected_bundle_sha256"],
        "expected_manifest_sha256": inputs["expected_manifest_sha256"],
        "expected_completion_sha256": inputs["expected_completion_sha256"],
        "expected_training_context_sha256": inputs["expected_training_context_sha256"],
        "expected_checkpoint_sha256": inputs["expected_checkpoint_sha256"],
    }
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


@pytest.mark.parametrize(
    ("capital", "error_type"),
    [
        pytest.param(True, TypeError, id="bool"),
        pytest.param(np.bool_(True), TypeError, id="numpy-bool"),
        pytest.param("1000", TypeError, id="numeric-string"),
        pytest.param(float("nan"), ValueError, id="nan"),
        pytest.param(float("inf"), ValueError, id="infinity"),
        pytest.param(0.0, ValueError, id="zero"),
        pytest.param(-1.0, ValueError, id="negative"),
        pytest.param(2**4096, ValueError, id="float-overflow"),
    ],
)
def test_invalid_initial_capital_is_rejected_before_any_artifact_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capital: object,
    error_type: type[Exception],
) -> None:
    from trade_rl.evaluation.runs import (
        portfolio_family_comparison as comparison_module,
    )
    from trade_rl.evaluation.runs.portfolio_family_comparison import (
        run_four_family_shared_portfolio_replay,
    )

    inputs, _ = _prepared_inputs(tmp_path)
    inputs["initial_capital"] = capital

    def forbidden_dataset_load(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("invalid capital must fail before any artifact load")

    monkeypatch.setattr(
        comparison_module,
        "load_published_market_dataset_artifact",
        forbidden_dataset_load,
    )

    with pytest.raises(error_type):
        run_four_family_shared_portfolio_replay(**inputs)


def test_dataset_digest_mismatch_precedes_ridge_artifact_load(
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

    def forbidden_ridge_load(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("Dataset mismatch must fail before Ridge artifact load")

    monkeypatch.setattr(
        comparison_module, "load_ridge_forecast_artifact", forbidden_ridge_load
    )

    with pytest.raises(ValueError, match="Dataset artifact digest"):
        run_four_family_shared_portfolio_replay(**inputs)


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


def test_native_ledger_matches_independent_fraction_cost_and_cash_oracle() -> None:
    from trade_rl.evaluation.replay import run_shared_portfolio_replay

    dataset = _dataset()
    fee_rate = Fraction(1, 100)
    dataset_fee_rate = Fraction(1, 100)
    annual_borrow_rate = Fraction(365, 1_000)
    execution_cost = ExecutionCostConfig(
        fee_rate=float(fee_rate),
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

    class FixedQuantityPolicy:
        def decide_portfolio(
            self, snapshot: PortfolioDecisionSnapshot
        ) -> PortfolioTargetProposal:
            equity = Fraction(str(snapshot.account.portfolio_value))
            prices = dataset.resolved_array("mark_price")[snapshot.index]
            quantities = (Fraction(2), Fraction(-1))
            return PortfolioTargetProposal(
                target_weights=tuple(
                    float(quantity * Fraction(str(price)) / equity)
                    for quantity, price in zip(quantities, prices, strict=True)
                )
            )

    result = run_shared_portfolio_replay(
        dataset,
        FixedQuantityPolicy(),
        start_index=0,
        stop_index=5,
        initial_capital=1_000.0,
        execution_cost=execution_cost,
        risk=PreTradeRisk(PreTradeRiskConfig(max_gross=1.0, max_abs_weight=0.5)),
        settle_terminal_position=True,
        capture_ledger_evidence=True,
    )

    ledger = result.ledger_evidence
    assert ledger is not None
    observed_fills = tuple(
        (
            event.processing_index,
            dataset.symbols[event.symbol_index],
            Fraction(str(event.filled_quantity)),
            Fraction(str(event.execution_price)),
        )
        for interval in ledger.intervals.intervals
        for event in interval.order_events
        if event.filled_quantity != 0.0
    )
    literal_fills = (
        (1, "AAA", Fraction(2), Fraction(100)),
        (1, "BBB", Fraction(-1), Fraction(100)),
        (5, "AAA", Fraction(-2), Fraction(100)),
        (5, "BBB", Fraction(1), Fraction(100)),
    )
    assert observed_fills == literal_fills

    signed_trade_cash = sum(
        (-quantity * price for _, _, quantity, price in literal_fills),
        start=Fraction(0),
    )
    expected_fees = sum(
        (
            abs(quantity) * price * (fee_rate + dataset_fee_rate)
            for _, _, quantity, price in literal_fills
        ),
        start=Fraction(0),
    )
    expected_funding = -Fraction(200) * Fraction(1, 100) + Fraction(100) * Fraction(
        1, 200
    )
    expected_borrow = Fraction(100) * annual_borrow_rate * 4 / 8_760
    expected_terminal_cash = (
        Fraction(1_000)
        + signed_trade_cash
        - expected_fees
        + expected_funding
        - expected_borrow
    )
    assert signed_trade_cash == 0
    assert expected_fees == 12
    assert expected_funding == Fraction(-3, 2)
    assert expected_borrow == Fraction(1, 60)
    assert expected_terminal_cash == Fraction(59_189, 60)
    assert result.book.cash == pytest.approx(float(expected_terminal_cash), abs=1e-10)
    assert ledger.intervals.final_cash == pytest.approx(
        float(expected_terminal_cash), abs=1e-10
    )
    assert result.book.total_cost == pytest.approx(float(expected_fees), abs=1e-10)
    assert ledger.intervals.final_total_cost == pytest.approx(
        float(expected_fees), abs=1e-10
    )
    assert result.book.funding_pnl == pytest.approx(float(expected_funding), abs=1e-10)
    assert ledger.intervals.final_funding_pnl == pytest.approx(
        float(expected_funding), abs=1e-10
    )
    assert result.book.borrow_cost == pytest.approx(float(expected_borrow), abs=1e-10)
    assert ledger.intervals.final_borrow_cost == pytest.approx(
        float(expected_borrow), abs=1e-10
    )
    assert ledger.intervals.terminal_exact_quantities == ("0", "0")
    assert ledger.intervals.active_order_remainders == ()


def _actual_ppo_bundle_case(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[dict[str, Any], MarketDataset, Any]:
    pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    from trade_rl.strategies.rl.portfolio_env import SharedPortfolioGymEnv
    from trade_rl.strategies.rl.portfolio_inference_bundle import (
        publish_portfolio_inference_bundle,
    )
    from trade_rl.strategies.rl.portfolio_training import (
        SharedPortfolioPPOConfig,
        fit_shared_portfolio_ppo,
    )
    from trade_rl.strategies.rl.portfolio_transform import (
        fit_portfolio_observation_transform,
    )

    root = tmp_path_factory.mktemp("stage4a-actual-ppo")
    inputs, dataset = _prepared_inputs(root)
    base_bound = inputs["bound"]
    training_bound = BoundObjectiveClock(
        replace(
            base_bound.objective,
            evaluation_start=_START,
            evaluation_stop_exclusive=_START + timedelta(hours=4),
        ),
        replace(base_bound.clock, economic_horizon_seconds=4 * 3_600),
    )
    runtime_bound = BoundObjectiveClock(
        replace(
            base_bound.objective,
            evaluation_start=_START + timedelta(hours=4),
            evaluation_stop_exclusive=_START + timedelta(hours=6),
        ),
        replace(base_bound.clock, economic_horizon_seconds=2 * 3_600),
    )
    inputs.update(
        {
            "start_index": 4,
            "stop_index": 6,
            "bound": runtime_bound,
        }
    )
    training_env = SharedPortfolioGymEnv(
        dataset,
        start_index=0,
        stop_index=4,
        bound=training_bound,
        target_abs_weights=inputs["target_abs_weights"],
        execution_cost=inputs["execution_cost"],
        risk_config=inputs["risk_config"],
        market_order_profile=inputs["market_order_profile"],
    )
    transform = fit_portfolio_observation_transform(training_env)
    attempt = fit_shared_portfolio_ppo(
        training_env,
        transform=transform,
        output_root=root / "training-attempt",
        config=SharedPortfolioPPOConfig(
            total_timesteps=8,
            n_steps=4,
            batch_size=2,
            n_epochs=1,
            reset_seed=47,
            ppo_seed=53,
        ),
    )
    publication = publish_portfolio_inference_bundle(
        attempt.output_root,
        root / "published-bundle",
        expected_manifest_sha256=attempt.manifest_sha256,
        expected_completion_sha256=attempt.completion_sha256,
        expected_training_context_sha256=attempt.training_context_sha256,
        expected_checkpoint_sha256=attempt.final_checkpoint_sha256,
    )
    inputs.update(
        {
            "bundle_root": publication.root,
            "expected_bundle_sha256": publication.bundle_sha256,
            "expected_manifest_sha256": publication.training_manifest_sha256,
            "expected_completion_sha256": publication.training_completion_sha256,
            "expected_training_context_sha256": publication.training_context_sha256,
            "expected_checkpoint_sha256": publication.checkpoint_sha256,
        }
    )
    return inputs, dataset, publication


@pytest.fixture(scope="module")
def actual_ppo_bundle_case(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[dict[str, Any], MarketDataset, Any]:
    return _actual_ppo_bundle_case(tmp_path_factory)


def test_actual_published_ppo_bundle_uses_strict_loader_in_four_family_runner(
    actual_ppo_bundle_case: tuple[dict[str, Any], MarketDataset, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.evaluation.runs import (
        portfolio_family_comparison as comparison_module,
    )
    from trade_rl.evaluation.runs.portfolio_family_comparison import (
        run_four_family_shared_portfolio_replay,
    )

    inputs, dataset, publication = actual_ppo_bundle_case
    strict_loader = comparison_module.load_portfolio_inference_bundle
    dataset_loader = comparison_module.load_published_market_dataset_artifact
    replay_runner = comparison_module.run_shared_portfolio_replay
    forwarded: dict[str, object] = {}
    loaded_pairs: list[tuple[Any, MarketDataset]] = []
    replay_datasets: list[MarketDataset] = []

    def record_dataset_load(root: str | Path) -> tuple[Any, MarketDataset]:
        pair = dataset_loader(root)
        loaded_pairs.append(pair)
        return pair

    def record_native_replay(
        replay_dataset: MarketDataset, policy: object, **kwargs: object
    ) -> object:
        replay_datasets.append(replay_dataset)
        return replay_runner(replay_dataset, policy, **kwargs)

    def record_and_load(root: str | Path, **kwargs: object) -> object:
        forwarded.update(kwargs)
        return strict_loader(root, **kwargs)

    monkeypatch.setattr(
        comparison_module, "load_portfolio_inference_bundle", record_and_load
    )
    monkeypatch.setattr(
        comparison_module, "load_published_market_dataset_artifact", record_dataset_load
    )
    monkeypatch.setattr(
        comparison_module, "run_shared_portfolio_replay", record_native_replay
    )
    result = run_four_family_shared_portfolio_replay(**inputs)

    assert result.ppo_bundle_sha256 == publication.bundle_sha256
    assert tuple(outcome.family for outcome in result.families) == _FAMILY_ORDER
    assert all(
        outcome.replay.ledger_evidence is not None for outcome in result.families
    )
    assert forwarded == {
        "expected_bundle_sha256": publication.bundle_sha256,
        "expected_manifest_sha256": publication.training_manifest_sha256,
        "expected_completion_sha256": publication.training_completion_sha256,
        "expected_training_context_sha256": publication.training_context_sha256,
        "expected_checkpoint_sha256": publication.checkpoint_sha256,
        "runtime_env": forwarded["runtime_env"],
    }
    runtime_env = forwarded["runtime_env"]
    assert len(loaded_pairs) == 1
    loaded_dataset = loaded_pairs[0][1]
    assert loaded_dataset.dataset_id == dataset.dataset_id
    assert runtime_env.dataset is loaded_dataset
    assert runtime_env._session is None
    assert len(replay_datasets) == 4
    assert all(replay_dataset is loaded_dataset for replay_dataset in replay_datasets)


@pytest.mark.parametrize(
    "pin_name",
    (
        "expected_bundle_sha256",
        "expected_manifest_sha256",
        "expected_completion_sha256",
        "expected_training_context_sha256",
        "expected_checkpoint_sha256",
    ),
)
def test_actual_ppo_bundle_pin_mismatch_precedes_native_session_creation(
    actual_ppo_bundle_case: tuple[dict[str, Any], MarketDataset, Any],
    monkeypatch: pytest.MonkeyPatch,
    pin_name: str,
) -> None:
    from trade_rl.evaluation.runs.portfolio_family_comparison import (
        run_four_family_shared_portfolio_replay,
    )
    from trade_rl.simulation.stateful import portfolio_session as session_module

    inputs, _, _ = actual_ppo_bundle_case
    bad_inputs = dict(inputs)
    current = str(bad_inputs[pin_name])
    bad_inputs[pin_name] = "0" * 64 if current != "0" * 64 else "1" * 64
    sessions = 0

    def forbidden_session_init(self: object, *args: object, **kwargs: object) -> None:
        del self, args, kwargs
        nonlocal sessions
        sessions += 1
        raise AssertionError("bundle pin mismatch must precede native sessions")

    monkeypatch.setattr(
        session_module.SharedPortfolioSession, "__init__", forbidden_session_init
    )

    with pytest.raises(ValueError):
        run_four_family_shared_portfolio_replay(**bad_inputs)
    assert sessions == 0
