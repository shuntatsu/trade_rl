from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.experiments.ppo_interleaved_evaluation import (
    PPOReturnPathEvidence,
    PPOSeedEvidence,
    PPOSymbolEvidence,
    canonical_ppo_interleaved_evaluator_spec,
    evaluate_ppo_training_seed,
    return_path_sha256,
)
from trade_rl.evaluation.metrics import compound_return
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.position_intent import PositionIntent
from trade_rl.strategies.rl.ppo import PPOIntentStrategy

_SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
_FEATURE_INDICES = (0, 1, 2, 4, 5, 6, 7, 10, 60, 63, 114, 118)
_FEATURE_NAMES = (
    "1h__log_return_1bar",
    "1h__log_return_4bar",
    "1h__log_return_24bar",
    "1h__realized_volatility_24bar",
    "1h__volume_zscore_24bar",
    "1h__funding_bps",
    "1h__rsi_14bar",
    "1h__macd_histogram_12_26_9",
    "4h__log_return_4bar",
    "4h__realized_volatility_24bar",
    "1d__log_return_1bar",
    "1d__realized_volatility_24bar",
)


def _feature_names(*, corrupt_index: int | None = None) -> tuple[str, ...]:
    names = [f"unused_{index}" for index in range(119)]
    for index, name in zip(_FEATURE_INDICES, _FEATURE_NAMES, strict=True):
        names[index] = name
    if corrupt_index is not None:
        names[corrupt_index] = "corrupt_feature"
    return tuple(names)


def _dataset(*, corrupt_feature_index: int | None = None) -> MarketDataset:
    timestamps = np.asarray(
        [
            "2022-12-31T22:00:00",
            "2022-12-31T23:00:00",
            "2023-01-01T00:00:00",
            "2025-01-01T00:00:00",
            "2025-01-01T01:00:00",
        ],
        dtype="datetime64[ns]",
    )
    n_bars = len(timestamps)
    n_symbols = len(_SYMBOLS)
    n_features = 119
    close = np.full((n_bars, n_symbols), 100.0, dtype=np.float64)
    return MarketDataset(
        dataset_id="6c0b040d317a1bb73a9273f4135879b31691634aa837f30f0eec005ac7531518",
        symbols=_SYMBOLS,
        timestamps=timestamps,
        features=np.zeros((n_bars, n_symbols, n_features), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=close.copy(),
        high=close.copy(),
        low=close.copy(),
        close=close,
        volume=np.full((n_bars, n_symbols), 1_000_000.0, dtype=np.float64),
        funding_rate=np.zeros((n_bars, n_symbols), dtype=np.float64),
        tradable=np.ones((n_bars, n_symbols), dtype=np.bool_),
        feature_available=np.ones((n_bars, n_symbols, n_features), dtype=np.bool_),
        feature_names=_feature_names(corrupt_index=corrupt_feature_index),
        global_feature_names=("regime",),
        periods_per_year=8_760,
        calendar_kind="session_calendar",
        fee_rate=np.full((n_bars, n_symbols), 0.0005, dtype=np.float64),
        taker_fee_rate=np.zeros((n_bars, n_symbols), dtype=np.float64),
        spread_rate=np.full((n_bars, n_symbols), 0.0002, dtype=np.float64),
    )


class _Policy:
    def __init__(self, num_timesteps: int) -> None:
        self.num_timesteps = num_timesteps

    def predict(
        self, observation: np.ndarray, *, deterministic: bool
    ) -> tuple[int, None]:
        del observation, deterministic
        return 1, None


def _strategy(num_timesteps: int) -> PPOIntentStrategy:
    return PPOIntentStrategy(
        _Policy(num_timesteps),
        feature_indices=_FEATURE_INDICES,
    )


def _max_drawdown(values: tuple[float, ...]) -> float:
    wealth = 1.0
    peak = 1.0
    maximum = 0.0
    for value in values:
        wealth *= 1.0 + value
        peak = max(peak, wealth)
        maximum = max(maximum, 1.0 - wealth / peak)
    return maximum


def _entry(
    name: str,
    *,
    returns: tuple[float, ...],
    total_cost: float,
    turnover_total: float,
    termination_reasons: tuple[str, ...] = (),
) -> Any:
    return SimpleNamespace(
        name=name,
        metrics=SimpleNamespace(
            total_return=compound_return(returns),
            total_cost=total_cost,
            turnover_total=turnover_total,
            max_drawdown=_max_drawdown(returns),
            termination_count=len(termination_reasons),
            n_periods=len(returns),
        ),
        replay=SimpleNamespace(
            returns=SimpleNamespace(values=returns, periods_per_year=8_760),
            diagnostics=SimpleNamespace(
                total_cost=total_cost,
                turnover_total=turnover_total,
                termination_count=len(termination_reasons),
                termination_reasons=termination_reasons,
            ),
        ),
    )


def _comparison() -> Any:
    by_symbol = []
    for index, symbol in enumerate(_SYMBOLS):
        ppo_returns = (0.01 + index * 0.001, -0.002)
        cash_returns = (0.0, 0.0)
        long_returns = ppo_returns if index == 0 else (0.005, 0.0)
        short_returns = (-0.005, 0.0)
        entries = (
            _entry(
                "ppo",
                returns=ppo_returns,
                total_cost=5.0 + index,
                turnover_total=2.0 + index,
            ),
            _entry(
                "cash",
                returns=cash_returns,
                total_cost=0.0,
                turnover_total=0.0,
            ),
            _entry(
                "constant_long",
                returns=long_returns,
                total_cost=6.0 + index,
                turnover_total=1.0,
            ),
            _entry(
                "constant_short",
                returns=short_returns,
                total_cost=6.0 + index,
                turnover_total=1.0,
            ),
        )
        by_symbol.append(
            SimpleNamespace(
                symbol_index=index,
                symbol=symbol,
                comparison=SimpleNamespace(entries=entries),
            )
        )
    return SimpleNamespace(by_symbol=tuple(by_symbol))


def _path(
    name: str,
    values: tuple[float, ...],
    *,
    total_cost: float = 1.0,
    turnover_total: float = 2.0,
    termination_reasons: tuple[str, ...] = (),
) -> PPOReturnPathEvidence:
    return PPOReturnPathEvidence(
        strategy_name=name,
        returns=values,
        return_sha256=return_path_sha256(values),
        total_return=compound_return(values),
        total_cost=total_cost,
        turnover_total=turnover_total,
        max_drawdown=_max_drawdown(values),
        termination_count=len(termination_reasons),
        termination_reasons=termination_reasons,
        n_periods=len(values),
        periods_per_year=8_760,
    )


def _symbol_evidence(symbol: str, index: int) -> PPOSymbolEvidence:
    ppo = _path("ppo", (0.01, -0.002))
    cash = _path("cash", (0.0, 0.0), total_cost=0.0, turnover_total=0.0)
    long = _path("constant_long", (0.01, -0.002))
    short = _path("constant_short", (-0.01, 0.001))
    return PPOSymbolEvidence(
        symbol_index=index,
        symbol=symbol,
        ppo=ppo,
        cash=cash,
        constant_long=long,
        constant_short=short,
        ppo_returns_equal_cash=False,
        ppo_returns_equal_constant_long=True,
        ppo_returns_equal_constant_short=False,
    )


def test_canonical_spec_binds_sealed_protocol_without_authorizing_training() -> None:
    spec = canonical_ppo_interleaved_evaluator_spec()

    assert spec.issue_number == 632
    assert spec.protocol_issue_number == 629
    assert spec.protocol_head == "f1187dacae78e679a322cc53cbf03f3371f457b1"
    assert spec.protocol_module_blob == "62a71b1ca8b7d22fdfc14282508754c44d96091a"
    assert (
        spec.protocol_digest
        == "a34aee66bf3f51ce02675b955f835c292b023770aab841f25b46815f9b399c2c"
    )
    assert spec.protocol_seal_run_id == 35205354305
    assert spec.protocol_primary_artifact_id == 10489866637
    assert spec.protocol_fresh_artifact_id == 10489661856
    assert spec.implementation_head == "cddef3532dd582d0f1066a61006f64265f0cabbe"
    assert spec.symbols == _SYMBOLS
    assert spec.feature_names == _FEATURE_NAMES
    assert spec.feature_indices == _FEATURE_INDICES
    assert spec.ppo_seeds == (0, 1, 2, 3, 4)
    assert spec.ppo_total_timesteps == 100_000
    assert spec.baseline_training_layout == "sequential"
    assert spec.baseline_rollout_steps_per_env is None
    assert spec.candidate_training_layout == "interleaved"
    assert spec.candidate_rollout_steps_per_env == 384
    assert spec.baseline_training_authorized is False
    assert spec.candidate_training_authorized is False
    assert spec.economic_result_inspected is False
    assert spec.final_test_accessed is False
    assert spec.shared_cash_profitability_established is False
    assert spec.production_eligible is False
    assert spec.live_trading_authorized is False
    assert spec.merge_authorized is False

    with pytest.raises(ValueError, match="candidate_rollout_steps_per_env"):
        replace(spec, candidate_rollout_steps_per_env=448)


def test_return_path_digest_is_deterministic_and_fail_closed() -> None:
    values = (0.01, -0.02, 0.03)
    assert return_path_sha256(values) == return_path_sha256(values)
    assert return_path_sha256(values) != return_path_sha256((0.01, -0.02))
    with pytest.raises(ValueError, match="one-dimensional"):
        return_path_sha256(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="finite"):
        return_path_sha256((0.0, float("nan")))


@pytest.mark.parametrize(
    ("arm", "expected_layout", "expected_rollout", "num_timesteps"),
    [
        ("baseline", "sequential", None, 100_352),
        ("candidate", "interleaved", 384, 101_760),
    ],
)
def test_evaluator_isolates_only_training_layout_and_uses_common_replay(
    monkeypatch: pytest.MonkeyPatch,
    arm: str,
    expected_layout: str,
    expected_rollout: int | None,
    num_timesteps: int,
) -> None:
    import trade_rl.evaluation.experiments.ppo_interleaved_evaluation as module

    dataset = _dataset()
    spec = canonical_ppo_interleaved_evaluator_spec()
    strategy = _strategy(num_timesteps)
    fit_calls: list[dict[str, object]] = []
    compare_calls: list[dict[str, object]] = []

    def fit(dataset_arg: MarketDataset, **kwargs: object) -> PPOIntentStrategy:
        assert dataset_arg is dataset
        fit_calls.append(dict(kwargs))
        return strategy

    def compare(
        dataset_arg: MarketDataset,
        strategies: dict[str, object],
        **kwargs: object,
    ) -> Any:
        assert dataset_arg is dataset
        assert tuple(strategies) == (
            "ppo",
            "cash",
            "constant_long",
            "constant_short",
        )
        assert strategies["ppo"] is strategy
        compare_calls.append({"strategies": strategies, **kwargs})
        return _comparison()

    monkeypatch.setattr(module, "fit_ppo_strategy", fit)
    monkeypatch.setattr(module, "compare_strategies_by_symbol", compare)

    result = evaluate_ppo_training_seed(dataset, spec, arm=arm, seed=2)

    assert len(fit_calls) == 1
    assert len(compare_calls) == 1
    fit_call = fit_calls[0]
    assert fit_call["feature_indices"] == spec.feature_indices
    assert fit_call["fit_symbol_indices"] == (0, 1, 2, 3, 4)
    assert fit_call["start_index"] == 0
    assert fit_call["stop_index"] == 1
    assert fit_call["gross_budget"] == spec.gross_budget
    assert fit_call["total_timesteps"] == 100_000
    assert fit_call["seed"] == 2
    assert fit_call["initial_capital"] == 100_000.0
    assert fit_call["training_layout"] == expected_layout
    assert fit_call["rollout_steps_per_env"] == expected_rollout
    execution_cost = fit_call["execution_cost"]
    assert isinstance(execution_cost, ExecutionCostConfig)
    assert execution_cost.processing_bar_volume_capacity is False
    assert execution_cost.slippage_std == 0.0

    compare_call = compare_calls[0]
    assert compare_call["start_index"] == 2
    assert compare_call["stop_index"] == 3
    assert compare_call["gross_budget"] == spec.gross_budget
    assert compare_call["initial_capital"] == spec.initial_capital
    assert compare_call["execution_cost"] is execution_cost
    assert compare_call["risk"] is None

    assert result.arm == arm
    assert result.seed == 2
    assert result.training_layout == expected_layout
    assert result.rollout_steps_per_env == expected_rollout
    assert result.caller_total_timesteps == 100_000
    assert result.realized_num_timesteps == num_timesteps
    assert tuple(item.symbol for item in result.by_symbol) == _SYMBOLS
    assert result.by_symbol[0].ppo_returns_equal_constant_long is True
    assert result.by_symbol[0].ppo_returns_equal_cash is False
    assert result.final_test_accessed is False
    assert result.shared_cash_profitability_established is False
    assert len(result.digest) == 64


def test_invalid_arm_seed_dataset_or_execution_fails_before_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trade_rl.evaluation.experiments.ppo_interleaved_evaluation as module

    spec = canonical_ppo_interleaved_evaluator_spec()
    calls = 0

    def fit(*args: object, **kwargs: object) -> PPOIntentStrategy:
        nonlocal calls
        calls += 1
        return _strategy(100_352)

    monkeypatch.setattr(module, "fit_ppo_strategy", fit)

    with pytest.raises(ValueError, match="arm"):
        evaluate_ppo_training_seed(_dataset(), spec, arm="other", seed=0)
    with pytest.raises(ValueError, match="seed"):
        evaluate_ppo_training_seed(_dataset(), spec, arm="baseline", seed=9)
    with pytest.raises(ValueError, match="feature identity"):
        evaluate_ppo_training_seed(
            _dataset(corrupt_feature_index=0), spec, arm="baseline", seed=0
        )

    bad_cost = replace(ExecutionCostConfig.zero(), slippage_std=0.01)
    monkeypatch.setattr(module, "execution_cost_for_overlay", lambda overlay: bad_cost)
    with pytest.raises(ValueError, match="execution cost"):
        evaluate_ppo_training_seed(_dataset(), spec, arm="candidate", seed=0)
    assert calls == 0


def test_return_path_evidence_fails_closed_on_raw_arithmetic_and_identity() -> None:
    path = _path("ppo", (0.01, -0.002))
    assert len(path.digest) == 64

    with pytest.raises(ValueError, match="return_sha256"):
        replace(path, return_sha256="f" * 64)
    with pytest.raises(ValueError, match="total_return"):
        replace(path, total_return=0.5)
    with pytest.raises(ValueError, match="n_periods"):
        replace(path, n_periods=99)
    with pytest.raises(ValueError, match="max_drawdown"):
        replace(path, max_drawdown=0.5)
    with pytest.raises(ValueError, match="termination"):
        replace(path, termination_count=1)


def test_symbol_evidence_derives_control_equality_from_exact_paths() -> None:
    symbol = _symbol_evidence("BTCUSDT", 0)
    assert symbol.ppo_returns_equal_constant_long is True

    with pytest.raises(ValueError, match="constant_long equality"):
        replace(symbol, ppo_returns_equal_constant_long=False)
    with pytest.raises(ValueError, match="strategy_name"):
        replace(symbol, ppo=replace(symbol.ppo, strategy_name="cash"))


def test_seed_evidence_fails_closed_on_timestep_roster_and_authorization() -> None:
    spec = canonical_ppo_interleaved_evaluator_spec()
    rows = tuple(
        _symbol_evidence(symbol, index) for index, symbol in enumerate(_SYMBOLS)
    )
    result = PPOSeedEvidence(
        spec_digest=spec.digest,
        protocol_head=spec.protocol_head,
        protocol_digest=spec.protocol_digest,
        implementation_head=spec.implementation_head,
        successor_bundle_run_id=spec.successor_bundle_run_id,
        successor_bundle_artifact_id=spec.successor_bundle_artifact_id,
        successor_bundle_artifact_digest=spec.successor_bundle_artifact_digest,
        dataset_id=spec.dataset_id,
        dataset_artifact_digest=spec.dataset_artifact_digest,
        study_digest=spec.study_digest,
        execution_overlay=spec.execution_overlay,
        slippage_std=spec.slippage_std,
        symbols=spec.symbols,
        feature_names=spec.feature_names,
        feature_indices=spec.feature_indices,
        fit_symbol_names=spec.fit_symbol_names,
        fit_cutoff=spec.fit_cutoff,
        evaluation_start=spec.evaluation_start,
        evaluation_stop_exclusive=spec.evaluation_stop_exclusive,
        gross_budget=spec.gross_budget,
        initial_capital=spec.initial_capital,
        arm="baseline",
        seed=0,
        training_layout="sequential",
        rollout_steps_per_env=None,
        caller_total_timesteps=100_000,
        realized_num_timesteps=100_352,
        by_symbol=rows,
    )
    assert len(result.digest) == 64

    with pytest.raises(ValueError, match="realized_num_timesteps"):
        replace(result, realized_num_timesteps=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="symbol roster"):
        replace(result, by_symbol=tuple(reversed(rows)))
    with pytest.raises(ValueError, match="production"):
        replace(result, production_eligible=True)


def test_evaluator_module_does_not_interpret_frozen_economic_decision() -> None:
    import trade_rl.evaluation.experiments.ppo_interleaved_evaluation as module

    assert not hasattr(module, "classify_ppo_interleaved_evaluation")
    assert not hasattr(module, "PROMOTE_TO_SHARED_CASH_EVALUATION")
    assert PositionIntent.FLAT.value == 0
