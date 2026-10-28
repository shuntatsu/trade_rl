"""Synthetic TD4 one-shot native comparison and failure evidence regression tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.runs.synthetic_value_attempt import (
    SyntheticValueAttemptConfig,
    load_synthetic_value_attempt,
    run_synthetic_value_attempt,
)
from trade_rl.risk.pretrade import PreTradeRiskConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.decision_value import (
    LinearDecisionValueModel,
    serialize_linear_decision_value,
)
from trade_rl.strategies.portfolio_policy_codec import (
    ACTION_HOLD,
    ACTION_LONG,
    ACTION_SHORT,
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

    published = run_synthetic_value_attempt(
        root, config, on_arm_completed=after_arm
    )
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

    cash, fixed, value = (
        float(row["terminal_equity"]) for row in loaded.arms
    )
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
        run_synthetic_value_attempt(
            root, config, on_arm_completed=fail_after_long
        )
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
        run_synthetic_value_attempt(
            root, replace(config, source_scope="real_market")
        )
    assert not root.exists()
    with pytest.raises(ValueError, match="dataset"):
        run_synthetic_value_attempt(
            root, replace(config, expected_dataset_id="b" * 64)
        )
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
    row_path.write_text(original.replace('"valid_endpoint":true', '"valid_endpoint":false'))
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
