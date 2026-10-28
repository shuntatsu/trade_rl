"""Synthetic TD4 one-shot native comparison and failure evidence regression tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.runs.synthetic_value_attempt import (
    SyntheticValueAttemptConfig,
    load_synthetic_value_attempt,
    run_synthetic_value_attempt,
)
from trade_rl.evaluation.runs.synthetic_value_costaware_attempt import (
    SyntheticCostAwareAttemptConfig,
    _ExactQuantityCostAwareRidgePolicy,
    load_synthetic_cost_aware_attempt,
    run_synthetic_cost_aware_attempt,
    synthetic_ridge_fixture_digest,
)
from trade_rl.evaluation.replay import run_shared_portfolio_replay
from trade_rl.risk.pretrade import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.decision_value import (
    LinearDecisionValueModel,
    serialize_linear_decision_value,
)
from trade_rl.strategies.forecasts.ridge import RidgeForecastModel
from trade_rl.strategies.portfolio import PortfolioDecisionSnapshot, PortfolioTargetProposal
from trade_rl.strategies.portfolio_policy_codec import (
    ACTION_HOLD,
    ACTION_LONG,
    ACTION_SHORT,
    portfolio_action_to_proposal,
)

_CASH = (ACTION_HOLD, ACTION_HOLD)
_LONG = (ACTION_LONG, ACTION_HOLD)
_SHORT = (ACTION_SHORT, ACTION_HOLD)
_COST = replace(ExecutionCostConfig.zero(), fee_rate=0.001, spread_rate=0.0002)
_RISK = PreTradeRiskConfig(
    max_gross=0.5,
    max_abs_weight=0.5,
    max_turnover=None,
    drawdown_start=0.20,
    drawdown_stop=0.20,
)


def _dataset(*, signal: float, move: float) -> MarketDataset:
    marks = np.array(
        [
            100.0,
            100.0,
            100.0,
            100.0 * (1 + move / 3),
            100.0 * (1 + 2 * move / 3),
            100.0 * (1 + move),
            100.0 * (1 + move),
            100.0 * (1 + move),
        ],
        dtype=np.float64,
    )
    close = np.column_stack((marks, np.full(8, 200.0)))
    opening = np.vstack((close[0], close[:-1]))
    features = np.zeros((8, 2, 1), dtype=np.float32)
    features[:, 0, 0] = signal
    dataset_id = hashlib.sha256(
        f"issue1025-td4/artificial/{signal}/{move}".encode()
    ).hexdigest()
    return MarketDataset(
        dataset_id=dataset_id,
        symbols=("AAA", "BBB"),
        timestamps=np.datetime64("2026-03-03T00:00:00", "ns")
        + np.arange(8) * np.timedelta64(1, "h"),
        features=features,
        global_features=np.zeros((8, 1), dtype=np.float32),
        open=opening,
        high=np.maximum(close, opening),
        low=np.minimum(close, opening),
        close=close,
        volume=np.full_like(close, 1_000_000.0),
        funding_rate=np.zeros_like(close),
        tradable=np.ones_like(close, dtype=np.bool_),
        feature_available=np.ones((8, 2, 1), dtype=np.bool_),
        feature_names=("observed_signal",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def _model() -> LinearDecisionValueModel:
    return LinearDecisionValueModel(
        schema_version="synthetic_linear_decision_value_v1",
        source_scope="synthetic_only",
        symbols=("AAA", "BBB"),
        feature_names=("observed_signal",),
        global_feature_names=("regime",),
        feature_config_digest="0" * 64,
        normalization_digest="0" * 64,
        execution_policy_digest=_COST.execution_policy_digest,
        reference_policy_digest="a" * 64,
        target_abs_weights=(0.4, 0.4),
        candidate_actions=(_CASH, _LONG, _SHORT),
        fit_cutoff_ns=int(np.datetime64("2026-02-01", "ns").astype(np.int64)),
        evaluation_start_ns=int(np.datetime64("2026-03-01", "ns").astype(np.int64)),
        ridge_penalty=1e-4,
        training_rows=18,
        overlap_groups=1,
        coefficients=(
            (-0.001, 0.010, 0.0, 0.0, 0.0, 0.0),
            (-0.001, -0.010, 0.0, 0.0, 0.0, 0.0),
        ),
    )


def _config(
    dataset: MarketDataset,
    *,
    execution_cost: ExecutionCostConfig = _COST,
) -> SyntheticValueAttemptConfig:
    payload, digest = serialize_linear_decision_value(_model())
    return SyntheticValueAttemptConfig(
        source_scope="synthetic_only",
        dataset=dataset,
        expected_dataset_id=dataset.dataset_id,
        model_json=payload,
        expected_model_sha256=digest,
        execution_cost=execution_cost,
        risk_config=_RISK,
        start_index=0,
        stop_index=7,
        initial_capital=1_000.0,
        fixed_action=_LONG,
    )


@pytest.mark.parametrize(
    ("signal", "move", "expected_action"),
    (
        (0.0, 0.0, _CASH),
        (0.8, 0.018, _LONG),
        (-0.8, -0.018, _SHORT),
    ),
)
def test_native_three_arm_comparison_persists_and_strictly_reloads(
    tmp_path: Path,
    signal: float,
    move: float,
    expected_action: tuple[int, ...],
) -> None:
    dataset = _dataset(signal=signal, move=move)
    config = _config(dataset)
    root = tmp_path / "attempt"
    observed: list[str] = []

    def after_arm(family: str) -> None:
        assert (root / "arms" / family / "record.json").is_file()
        observed.append(family)

    published = run_synthetic_value_attempt(root, config, on_arm_completed=after_arm)
    assert observed == ["cash", "fixed_long", "linear_value"]
    assert published.root == root
    loaded = load_synthetic_value_attempt(
        root,
        expected_result_sha256=published.result_sha256,
        expected_model_sha256=config.expected_model_sha256,
        expected_dataset_id=dataset.dataset_id,
    )
    assert loaded.result_sha256 == published.result_sha256
    assert loaded.dataset_id == dataset.dataset_id
    assert loaded.model_sha256 == config.expected_model_sha256
    assert tuple(row["family"] for row in loaded.arms) == tuple(observed)
    assert all(row["valid_endpoint"] is True for row in loaded.arms)
    for row in loaded.arms:
        assert len(row["ledger"]["intervals"]) == 7
        assert row["ledger"]["dataset_id"] == dataset.dataset_id
        assert row["ledger"]["execution_policy_digest"] == (
            _COST.execution_policy_digest
        )
        assert row["ledger"]["terminal_exact_quantities"] == ["0", "0"]

    cash, fixed, value = (float(row["terminal_equity"]) for row in loaded.arms)
    assert cash == pytest.approx(1_000.0)
    if expected_action == _CASH:
        assert value == pytest.approx(cash)
        # Independent fee-only roundtrip arithmetic: 2 * 400 * (.001 + .0002).
        assert fixed == pytest.approx(1_000.0 - 0.96, abs=1e-8)
        assert float(loaded.arms[1]["total_cost"]) == pytest.approx(0.96)
    else:
        assert value > cash
        if expected_action == _SHORT:
            assert fixed < cash
        else:
            assert fixed > cash

    assert not (root / "failure" / "record.json").exists()
    assert (root / "complete" / "record.json").is_file()
    with pytest.raises(FileExistsError):
        run_synthetic_value_attempt(root, config)


def test_failure_after_durable_arm_keeps_raw_row_and_blocks_completion(
    tmp_path: Path,
) -> None:
    root = tmp_path / "attempt"
    config = _config(_dataset(signal=0.8, move=0.018))

    def fail_after_long(family: str) -> None:
        if family == "fixed_long":
            raise RuntimeError("deliberate observer failure")

    with pytest.raises(RuntimeError, match="deliberate observer failure"):
        run_synthetic_value_attempt(root, config, on_arm_completed=fail_after_long)
    assert (root / "arms" / "cash" / "record.json").is_file()
    assert (root / "arms" / "fixed_long" / "record.json").is_file()
    assert not (root / "arms" / "linear_value" / "record.json").exists()
    assert not (root / "complete" / "record.json").exists()
    failure = json.loads((root / "failure" / "record.json").read_text())
    assert failure["phase"] == "fixed_long"
    assert failure["completed_families"] == ["cash", "fixed_long"]
    with pytest.raises(ValueError, match="complete|failure"):
        load_synthetic_value_attempt(
            root,
            expected_result_sha256="f" * 64,
            expected_model_sha256=config.expected_model_sha256,
            expected_dataset_id=config.dataset.dataset_id,
        )


def test_unliquidated_native_arm_keeps_bad_row_without_usable_result(
    tmp_path: Path,
) -> None:
    dataset = _dataset(signal=0.8, move=0.018)
    volume = dataset.volume.copy()
    volume[-1, 0] = 0.0
    blocked = replace(dataset, volume=volume)
    root = tmp_path / "blocked"
    with pytest.raises(RuntimeError, match="endpoint"):
        run_synthetic_value_attempt(root, _config(blocked))
    assert (root / "failure" / "record.json").is_file()
    row = json.loads((root / "arms" / "fixed_long" / "record.json").read_text())
    assert row["valid_endpoint"] is False
    assert row["ledger"]["terminal_exact_quantities"] != ["0", "0"]
    assert not (root / "complete" / "record.json").exists()


def test_preflight_rejects_cost_or_source_mismatch_before_reserving(
    tmp_path: Path,
) -> None:
    market = _dataset(signal=0.0, move=0.0)
    config = _config(market)
    root = tmp_path / "unused"
    with pytest.raises(ValueError, match="execution_policy_digest"):
        run_synthetic_value_attempt(
            root,
            replace(
                config,
                execution_cost=replace(_COST, fee_rate=0.003),
            ),
        )
    assert not root.exists()
    with pytest.raises(ValueError, match="synthetic"):
        run_synthetic_value_attempt(root, replace(config, source_scope="real_market"))
    assert not root.exists()
    with pytest.raises(ValueError, match="dataset"):
        run_synthetic_value_attempt(root, replace(config, expected_dataset_id="b" * 64))
    assert not root.exists()


def test_tampered_row_or_complete_marker_is_rejected_by_strict_loader(
    tmp_path: Path,
) -> None:
    dataset = _dataset(signal=0.8, move=0.018)
    config = _config(dataset)
    root = tmp_path / "attempt"
    result = run_synthetic_value_attempt(root, config)
    row_path = root / "arms" / "fixed_long" / "record.json"
    original = row_path.read_text()
    row_path.write_text(
        original.replace('"valid_endpoint":true', '"valid_endpoint":false')
    )
    with pytest.raises(ValueError, match="digest|tamper|valid"):
        load_synthetic_value_attempt(
            root,
            expected_result_sha256=result.result_sha256,
            expected_model_sha256=config.expected_model_sha256,
            expected_dataset_id=dataset.dataset_id,
        )
    row_path.write_text(original)
    completion = root / "complete" / "record.json"
    completion.unlink()
    with pytest.raises(ValueError, match="complete"):
        load_synthetic_value_attempt(
            root,
            expected_result_sha256=result.result_sha256,
            expected_model_sha256=config.expected_model_sha256,
            expected_dataset_id=dataset.dataset_id,
        )


# TD4b: a *separate*, versioned cost-aware extension of the closed v1 attempt.
# The Ridge return-forecast is frozen artificial fixture input, NOT a learned
# real-market source, and winning by construction is not a G4 economic result.


def _frozen_cost_aware_forecast() -> RidgeForecastModel:
    return RidgeForecastModel(
        feature_indices=(0,),
        feature_mean=np.array((0.0,), dtype=np.float64),
        feature_scale=np.array((1.0,), dtype=np.float64),
        coefficients=np.array((0.03,), dtype=np.float64),
        intercept=0.0,
        horizon_hours=1,
        alpha=1.0,
        n_samples=18,
        fit_cutoff=np.datetime64("2026-02-01T00:00:00", "ns"),
    )


def _cost_aware_config(
    market: MarketDataset,
) -> SyntheticCostAwareAttemptConfig:
    forecast = _frozen_cost_aware_forecast()
    return SyntheticCostAwareAttemptConfig(
        value_attempt=_config(market),
        forecast_model=forecast,
        expected_forecast_sha256=synthetic_ridge_fixture_digest(forecast),
        entry_threshold=0.002,
        exit_threshold=0.0005,
        one_way_switch_cost=0.0012,
    )


@pytest.mark.parametrize(
    ("signal", "move", "should_profit"),
    (
        (0.0, 0.0, False),
        (0.8, 0.018, True),
        (-0.8, -0.018, True),
        (0.8, -0.018, False),  # false signal: do not erase losing outcomes
    ),
)
def test_four_arm_cost_aware_v2_native_ledger_and_v1_backward_compatibility(
    tmp_path: Path, signal: float, move: float, should_profit: bool
) -> None:
    market = _dataset(signal=signal, move=move)
    config = _cost_aware_config(market)
    root = tmp_path / "four-arm"
    observed: list[str] = []

    def capture(family: str) -> None:
        rel = (
            "three_arm_v1/arms/" + family + "/record.json"
            if family != "cost_aware_ridge"
            else "arms/cost_aware_ridge/record.json"
        )
        assert (root / rel).is_file()
        observed.append(family)

    published = run_synthetic_cost_aware_attempt(root, config, on_arm_completed=capture)
    assert observed == ["cash", "fixed_long", "linear_value", "cost_aware_ridge"]
    assert (root / "three_arm_v1/complete/record.json").is_file()
    assert (root / "complete/record.json").is_file()
    loaded = load_synthetic_cost_aware_attempt(
        root,
        expected_result_sha256=published.result_sha256,
        expected_model_sha256=config.value_attempt.expected_model_sha256,
        expected_forecast_sha256=config.expected_forecast_sha256,
        expected_dataset_id=market.dataset_id,
    )
    assert loaded.result_sha256 == published.result_sha256
    assert loaded.forecast_sha256 == config.expected_forecast_sha256
    assert tuple(row["family"] for row in loaded.arms) == tuple(observed)
    assert len(loaded.arms) == 4
    for row in loaded.arms:
        assert row["valid_endpoint"] is True
        assert len(row["ledger"]["intervals"]) == 7
        assert row["ledger"]["dataset_id"] == market.dataset_id
        assert row["ledger"]["execution_policy_digest"] == (
            _COST.execution_policy_digest
        )
        assert row["ledger"]["terminal_exact_quantities"] == ["0", "0"]
        assert row["ledger"]["active_order_remainders"] == []

    cash = float(loaded.arms[0]["terminal_equity"])
    learned = float(loaded.arms[2]["terminal_equity"])
    cost_aware = float(loaded.arms[3]["terminal_equity"])
    assert cash == pytest.approx(1000.0)
    if signal == 0.0:
        assert learned == pytest.approx(cash)
        assert cost_aware == pytest.approx(cash)
        assert float(loaded.arms[3]["total_cost"]) == pytest.approx(0.0)
    elif should_profit:
        assert learned > cash
        assert cost_aware > cash
    else:
        assert learned < cash
        assert cost_aware < cash
    # Future labels may change, but the first proposal is causal / frozen.
    proposed = loaded.arms[3]["ledger"]["portfolio_decisions"][0]["proposal_weights"]
    if signal > 0:
        assert proposed[0] > 0.0
    elif signal < 0:
        assert proposed[0] < 0.0
    else:
        assert proposed[0] == 0.0

    with pytest.raises(FileExistsError):
        run_synthetic_cost_aware_attempt(root, config)


def test_cost_aware_v2_rejects_wrong_model_and_cost_proxy_before_reserving(
    tmp_path: Path,
) -> None:
    config = _cost_aware_config(_dataset(signal=0.8, move=0.018))
    root = tmp_path / "rejected"
    with pytest.raises(ValueError, match="forecast.*digest"):
        run_synthetic_cost_aware_attempt(
            root, replace(config, expected_forecast_sha256="f" * 64)
        )
    assert not root.exists()
    with pytest.raises(ValueError, match="cost|proxy"):
        run_synthetic_cost_aware_attempt(root, replace(config, one_way_switch_cost=0.0))
    assert not root.exists()
    with pytest.raises(ValueError, match="fit.cutoff|future"):
        run_synthetic_cost_aware_attempt(
            root,
            replace(
                config,
                forecast_model=replace(
                    config.forecast_model,
                    fit_cutoff=np.datetime64("2026-04-01T00:00:00", "ns"),
                ),
                expected_forecast_sha256=synthetic_ridge_fixture_digest(
                    replace(
                        config.forecast_model,
                        fit_cutoff=np.datetime64("2026-04-01T00:00:00", "ns"),
                    )
                ),
            ),
        )
    assert not root.exists()


def test_cost_aware_v2_retains_completed_v1_and_fourth_arm_on_observer_failure(
    tmp_path: Path,
) -> None:
    config = _cost_aware_config(_dataset(signal=0.8, move=0.018))
    root = tmp_path / "partial"

    def fail_after_fourth(family: str) -> None:
        if family == "cost_aware_ridge":
            raise RuntimeError("injected fourth arm observer failure")

    with pytest.raises(RuntimeError, match="fourth arm observer"):
        run_synthetic_cost_aware_attempt(
            root, config, on_arm_completed=fail_after_fourth
        )
    assert (root / "three_arm_v1/complete/record.json").is_file()
    assert (root / "arms/cost_aware_ridge/record.json").is_file()
    assert not (root / "complete/record.json").exists()
    assert json.loads((root / "failure/record.json").read_text())[
        "completed_families"
    ] == ["cash", "fixed_long", "linear_value", "cost_aware_ridge"]
    with pytest.raises(ValueError, match="failure|complete"):
        load_synthetic_cost_aware_attempt(
            root,
            expected_result_sha256="f" * 64,
            expected_model_sha256=config.value_attempt.expected_model_sha256,
            expected_forecast_sha256=config.expected_forecast_sha256,
            expected_dataset_id=config.value_attempt.expected_dataset_id,
        )


def test_cost_aware_v2_detects_nested_v1_tamper_and_frozen_forecast_tamper(
    tmp_path: Path,
) -> None:
    config = _cost_aware_config(_dataset(signal=-0.8, move=-0.018))
    root = tmp_path / "complete"
    published = run_synthetic_cost_aware_attempt(root, config)
    params = {
        "expected_result_sha256": published.result_sha256,
        "expected_model_sha256": config.value_attempt.expected_model_sha256,
        "expected_forecast_sha256": config.expected_forecast_sha256,
        "expected_dataset_id": config.value_attempt.expected_dataset_id,
    }
    nested = root / "three_arm_v1/arms/cash/record.json"
    raw = nested.read_bytes()
    nested.write_bytes(raw.replace(b'"valid_endpoint":true', b'"valid_endpoint":false'))
    with pytest.raises(ValueError, match="digest|tamper|valid"):
        load_synthetic_cost_aware_attempt(root, **params)
    nested.write_bytes(raw)

    model = root / "input/forecast.json"
    original_model = model.read_bytes()
    model.write_bytes(original_model + b"\n")
    with pytest.raises(ValueError, match="forecast|canonical|digest"):
        load_synthetic_cost_aware_attempt(root, **params)
    model.write_bytes(original_model)

    (root / "complete/record.json").unlink()
    with pytest.raises(ValueError, match="complete"):
        load_synthetic_cost_aware_attempt(root, **params)


def test_cost_aware_v2_exact_quantity_bridge_preserves_native_account_state() -> None:
    market = _dataset(signal=0.8, move=0.018)
    config = _cost_aware_config(market)

    class _CaptureNative:
        def __init__(self) -> None:
            self.observed: PortfolioDecisionSnapshot | None = None

        def decide_portfolio(
            self, snapshot: PortfolioDecisionSnapshot
        ) -> PortfolioTargetProposal:
            if snapshot.index == 2:
                self.observed = snapshot
            action = _LONG if snapshot.index == 0 else _CASH
            return portfolio_action_to_proposal(action, snapshot, (0.4, 0.4))

    capturer = _CaptureNative()
    run_shared_portfolio_replay(
        market,
        capturer,
        start_index=0,
        stop_index=7,
        initial_capital=1000.0,
        execution_cost=_COST,
        risk=PreTradeRisk(_RISK),
        settle_terminal_position=True,
    )
    observed = capturer.observed
    assert observed is not None
    assert Fraction(observed.account.exact_quantities[0]) > 0

    # The native BookState exact quantity is authoritative; a native floating
    # view may differ by a few ULPs. Do not weaken the legacy portfolio adapter.
    adapter = _ExactQuantityCostAwareRidgePolicy(market, config, (0.4, 0.4))
    proposal = adapter.decide_portfolio(observed)
    assert proposal.target_weights == (0.4, 0.0)

    broken = replace(
        observed,
        account=replace(
            observed.account,
            quantities=(observed.account.quantities[0] + 1.0, 0.0),
        ),
    )
    poisoned = _ExactQuantityCostAwareRidgePolicy(market, config, (0.4, 0.4))
    with pytest.raises(ValueError, match="materially disagree"):
        poisoned.decide_portfolio(broken)
    with pytest.raises(RuntimeError, match="poison"):
        poisoned.decide_portfolio(observed)
