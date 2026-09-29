from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from trade_rl.artifacts.canonical import to_json_value
from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.experiments.analysis import (
    PPO_HOLDING_DURATION_COMPARISON_SCHEMA,
    compare_evidence_sets,
)
from trade_rl.evaluation.experiments.contracts import ExperimentComparison, StudyPlan
from trade_rl.evaluation.experiments.protocols import ppo_holding_metrics
from trade_rl.evaluation.metrics import evaluate_performance
from trade_rl.evaluation.runs.artifact import (
    inspect_candidate_run_artifact,
    load_candidate_run_artifact,
    publish_candidate_run,
)
from trade_rl.evaluation.series import ReturnKind, ReturnSeries
from trade_rl.risk import PreTradeRiskConfig
from trade_rl.strategies.position_intent import PositionIntent
from trade_rl.strategies.rl.intent import PPO_OBSERVATION_SCHEMA_V3
from trade_rl.strategies.rl.ppo import ppo_observation_contract_payload


def _provenance() -> dict[str, object]:
    implementation: dict[str, object] = {
        "schema_version": "candidate_run_implementation_v1",
        "files": [],
    }
    runtime: dict[str, object] = {"schema_version": "candidate_run_runtime_v1"}
    return {
        "schema_version": "candidate_run_provenance_v1",
        "implementation": implementation,
        "implementation_digest": content_digest(implementation),
        "runtime_environment": runtime,
        "runtime_environment_digest": content_digest(runtime),
        "research_context_digest": None,
    }


def _summary(*, schema: str, observation: object | None) -> dict[str, object]:
    summary: dict[str, object] = {
        "schema_version": schema,
        "dataset_id": "b" * 64,
        "dataset_artifact": {
            "schema_version": "market_dataset_artifact_v3",
            "artifact_digest": "d" * 64,
        },
        "symbols": ["BTCUSDT"],
        "candidate_config": {},
        "evaluation": {},
        "by_symbol": [
            {
                "symbol_index": 0,
                "symbol": "BTCUSDT",
                "strategies": [
                    {
                        "name": "cash",
                        "return_key": "symbol_0_strategy_0",
                        "metrics": {},
                        "diagnostics": {},
                        "final_portfolio_value": 1000.0,
                        "fill_count": 0,
                    }
                ],
            }
        ],
    }
    if observation is not None:
        summary["ppo_observation"] = observation
    return summary


def _write_root(root: Path, summary: dict[str, object]) -> None:
    root.mkdir()
    (root / "summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    (root / "provenance.json").write_text(
        json.dumps(_provenance(), sort_keys=True, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(
        root / "returns.npz",
        symbol_0_strategy_0=np.asarray([0.0, 0.01], dtype=np.float64),
    )


def _result() -> object:
    config = SimpleNamespace(
        signal_name="signal",
        feature_names=("signal",),
        fit_symbol_names=("BTCUSDT",),
        evaluation_start=np.datetime64("2026-01-02T00:00:00", "ns"),
        evaluation_stop_exclusive=np.datetime64("2026-01-03T00:00:00", "ns"),
        gross_budget=0.5,
        initial_capital=1000.0,
        ppo_observation_schema="ppo_observation_v2",
        ppo_settle_terminal_position=False,
        pretrade_risk_config=None,
    )
    lean_config = SimpleNamespace(
        signal_index=0,
        feature_indices=(0,),
        fit_symbol_indices=(0,),
        fit_cutoff=np.datetime64("2026-01-02T00:00:00", "ns"),
        rule_entry_threshold=0.1,
        rule_exit_threshold=0.02,
        forecast_entry_threshold=0.01,
        forecast_exit_threshold=0.002,
        ppo_total_timesteps=256,
        ppo_seed=7,
        ppo_training_layout="sequential",
        ppo_rollout_steps_per_env=None,
        ppo_minimum_hold_bars=0,
    )
    metrics = SimpleNamespace(
        total_return=0.0,
        sharpe=0.0,
        sortino=0.0,
        max_drawdown=0.0,
        turnover_total=0.0,
        total_cost=0.0,
        funding_pnl=0.0,
        borrow_cost=0.0,
        n_trades=0,
        rebalance_events=0,
        termination_count=0,
        n_periods=1,
        return_kind=SimpleNamespace(value="base_bar"),
        periods_per_year=8760,
    )
    diagnostics = SimpleNamespace(
        turnover_total=0.0,
        total_cost=0.0,
        funding_pnl=0.0,
        borrow_cost=0.0,
        n_trades=0,
        rebalance_events=0,
        termination_reasons=(),
    )
    entry = SimpleNamespace(
        name="cash",
        replay=SimpleNamespace(
            returns=SimpleNamespace(values=(0.0,)),
            decisions=(),
            active_order_remainders=(),
            terminal_order_reasons=(),
            diagnostics=diagnostics,
            book=SimpleNamespace(
                portfolio_value=1000.0,
                fill_count=0,
                quantities=np.asarray([0.0]),
            ),
        ),
        metrics=metrics,
    )
    comparison = SimpleNamespace(
        by_symbol=(
            SimpleNamespace(
                symbol_index=0,
                symbol="BTCUSDT",
                comparison=SimpleNamespace(entries=(entry,)),
            ),
        )
    )
    return SimpleNamespace(
        spec=SimpleNamespace(
            config=config,
            lean_config=lean_config,
            dataset_id="b" * 64,
            dataset_artifact_schema="market_dataset_artifact_v3",
            dataset_artifact_digest="d" * 64,
        ),
        symbols=("BTCUSDT",),
        comparison=comparison,
        ppo_training_timesteps=2048,
        ppo_training_minimum_hold_suppressed_count=0,
    )


def _holding_result(
    *,
    seed: int,
    ppo_returns: tuple[float, ...],
    minimum_hold_bars: int = 168,
    final_quantity: float = 0.0,
    active_order_remainders: tuple[tuple[str, float], ...] = (),
) -> object:
    result = _result()
    result.spec.config.ppo_observation_schema = PPO_OBSERVATION_SCHEMA_V3
    result.spec.config.ppo_settle_terminal_position = True
    result.spec.config.pretrade_risk_config = PreTradeRiskConfig(
        max_gross=0.5,
        max_abs_weight=0.1,
        max_turnover=None,
        drawdown_start=0.1,
        drawdown_stop=0.2,
    )
    result.spec.lean_config.ppo_seed = seed
    result.spec.lean_config.ppo_observation_schema = PPO_OBSERVATION_SCHEMA_V3
    result.spec.lean_config.ppo_minimum_hold_bars = minimum_hold_bars

    diagnostics = SimpleNamespace(
        turnover_total=0.0,
        total_cost=0.0,
        funding_pnl=0.0,
        borrow_cost=0.0,
        n_trades=0,
        rebalance_events=0,
        termination_reasons=(),
    )
    entries = []
    for name in StudyPlan.STRATEGY_NAMES:
        values = ppo_returns if name == "ppo" else (0.0,) * len(ppo_returns)
        returns = ReturnSeries(
            values=values,
            kind=ReturnKind.BASE_BAR,
            periods_per_year=8760,
        )
        metrics = evaluate_performance(returns)
        quantity = final_quantity if name == "ppo" else 0.0
        remainders = active_order_remainders if name == "ppo" else ()
        replay = SimpleNamespace(
            returns=returns,
            decisions=(),
            active_order_remainders=remainders,
            terminal_order_reasons=(),
            diagnostics=diagnostics,
            book=SimpleNamespace(
                portfolio_value=1000.0 * (1.0 + metrics.total_return),
                fill_count=int(quantity != 0.0),
                quantities=np.asarray([quantity]),
            ),
        )
        entries.append(SimpleNamespace(name=name, replay=replay, metrics=metrics))
    result.comparison.by_symbol[0].comparison.entries = tuple(entries)
    return result


def test_new_candidate_write_records_observation_v2_contract(tmp_path: Path) -> None:
    artifact = publish_candidate_run(
        tmp_path / "run",
        _result(),  # type: ignore[arg-type]
        _provenance(),
    )

    summary = json.loads(artifact.summary_path.read_text(encoding="utf-8"))

    assert summary["schema_version"] == "lean_candidate_result_v5"
    assert summary["ppo_observation"] == ppo_observation_contract_payload()
    loaded = load_candidate_run_artifact(artifact.root)
    candidate_config = loaded.summary["candidate_config"]
    assert candidate_config["ppo_training_timesteps"] == 2048


def test_candidate_v5_binds_age_observation_and_holding_duration_and_risk(
    tmp_path: Path,
) -> None:
    result = _result()
    result.spec.config.ppo_observation_schema = PPO_OBSERVATION_SCHEMA_V3
    result.spec.config.ppo_settle_terminal_position = True
    result.spec.config.pretrade_risk_config = PreTradeRiskConfig(
        max_gross=0.5,
        max_abs_weight=0.1,
        max_turnover=None,
        drawdown_start=0.1,
        drawdown_stop=0.2,
    )
    result.spec.lean_config.ppo_observation_schema = PPO_OBSERVATION_SCHEMA_V3
    result.spec.lean_config.ppo_minimum_hold_bars = 168
    result.comparison.by_symbol[0].comparison.entries[0].replay.decisions = (
        SimpleNamespace(
            index=1,
            intent=PositionIntent.FLAT,
            effective_intent=PositionIntent.LONG,
            position_age_bars=167,
            position_age_bars_after=168,
            position_quantity_before=0.25,
            position_quantity_after=0.25,
            target_weight=0.025,
            minimum_hold_suppressed=True,
            minimum_hold_unlocked=False,
            risk_reasons=(),
        ),
    )

    artifact = publish_candidate_run(
        tmp_path / "run",
        result,  # type: ignore[arg-type]
        _provenance(),
    )

    summary = json.loads(artifact.summary_path.read_text(encoding="utf-8"))
    assert summary["schema_version"] == "lean_candidate_result_v5"
    assert summary["ppo_observation"] == ppo_observation_contract_payload(
        PPO_OBSERVATION_SCHEMA_V3
    )
    assert summary["candidate_config"]["ppo_minimum_hold_bars"] == 168
    assert summary["candidate_config"]["ppo_observation_schema"] == (
        PPO_OBSERVATION_SCHEMA_V3
    )
    assert summary["evaluation"]["ppo_settle_terminal_position"] is True
    ppo_summary = summary["by_symbol"][0]["strategies"][0]
    assert ppo_summary["minimum_hold_audit"] == [
        {
            "index": 1,
            "requested_intent": int(PositionIntent.FLAT),
            "effective_intent": int(PositionIntent.LONG),
            "position_age_bars": 167,
            "position_age_bars_after": 168,
            "position_quantity_before": 0.25,
            "position_quantity_after": 0.25,
            "target_weight": 0.025,
            "minimum_hold_suppressed": True,
            "minimum_hold_unlocked": False,
            "risk_reasons": [],
        }
    ]
    assert ppo_summary["final_quantities"] == [0.0]
    assert ppo_summary["terminal_settlement_complete"] is True
    assert ppo_summary["active_order_remainders"] == []

    loaded = load_candidate_run_artifact(artifact.root)
    assert loaded.summary["schema_version"] == "lean_candidate_result_v5"


@pytest.mark.parametrize(
    ("final_quantity", "active_order_remainders"),
    [
        pytest.param(0.25, (), id="residual-terminal-inventory"),
        pytest.param(
            0.0,
            (("terminal-order", 0.25),),
            id="active-terminal-order-remainder",
        ),
    ],
)
def test_loaded_terminal_inventory_or_remainder_rejects_holding_arm(
    tmp_path: Path,
    final_quantity: float,
    active_order_remainders: tuple[tuple[str, float], ...],
) -> None:
    seeds = (2, 7)
    baseline_runs = {}
    candidate_runs = {}
    candidate_returns = (0.01, 0.01, 0.01, 0.01)
    for seed in seeds:
        baseline_artifact = publish_candidate_run(
            tmp_path / "baseline" / f"seed-{seed}",
            _holding_result(
                seed=seed,
                ppo_returns=(0.0, 0.0, 0.0, 0.0),
                minimum_hold_bars=0,
            ),  # type: ignore[arg-type]
            _provenance(),
        )
        candidate_artifact = publish_candidate_run(
            tmp_path / "candidate" / f"seed-{seed}",
            _holding_result(
                seed=seed,
                ppo_returns=candidate_returns,
                final_quantity=final_quantity,
                active_order_remainders=active_order_remainders,
            ),  # type: ignore[arg-type]
            _provenance(),
        )
        baseline_runs[seed] = load_candidate_run_artifact(baseline_artifact.root)
        candidate_runs[seed] = load_candidate_run_artifact(candidate_artifact.root)

        ppo_summary = candidate_runs[seed].summary["by_symbol"][0]["strategies"][-1]
        assert ppo_summary["final_quantities"] == (final_quantity,)
        assert ppo_summary["active_order_remainders"] == tuple(
            {
                "order_id": order_id,
                "remaining_quantity": quantity,
            }
            for order_id, quantity in active_order_remainders
        )
        assert ppo_summary["terminal_settlement_complete"] is False

    factor_effect = compare_evidence_sets(
        baseline_runs,
        candidate_runs,
        n_bootstrap=32,
        bootstrap_seed=13,
        schema_version=PPO_HOLDING_DURATION_COMPARISON_SCHEMA,
    )
    ppo_effect = factor_effect["cross_seed"]["ppo"]
    assert ppo_effect["candidate_terminal_settlement_complete_account_count"] == 0
    factor_digest = factor_effect["analysis_digest"]
    assert isinstance(factor_digest, str)
    comparison = ExperimentComparison(
        study_digest="a" * 64,
        experiment_digest="b" * 64,
        baseline_evidence_digest="c" * 64,
        candidate_evidence_digest="d" * 64,
        verification_digest="e" * 64,
        baseline_analysis_digest="f" * 64,
        candidate_analysis_digest="1" * 64,
        factor_effect_digest=factor_digest,
        factor_effect=factor_effect,
    )
    metrics = ppo_holding_metrics(
        comparison,
        expected_symbols=("BTCUSDT",),
        expected_seeds=seeds,
    )

    assert metrics.score > 0.0
    assert metrics.median_excess_return > 0.0
    assert metrics.worst_max_drawdown <= 0.20
    assert metrics.terminal_settlement_complete is False
    assert metrics.eligible is False


def test_candidate_v5_rejects_inconsistent_typed_risk_config(tmp_path: Path) -> None:
    result = _result()
    result.spec.config.ppo_observation_schema = PPO_OBSERVATION_SCHEMA_V3
    result.spec.config.ppo_settle_terminal_position = True
    result.spec.config.pretrade_risk_config = PreTradeRiskConfig(
        max_gross=0.5,
        max_abs_weight=0.1,
        max_turnover=None,
        drawdown_start=0.1,
        drawdown_stop=0.2,
    )
    result.spec.lean_config.ppo_observation_schema = PPO_OBSERVATION_SCHEMA_V3
    result.spec.lean_config.ppo_minimum_hold_bars = 168
    artifact = publish_candidate_run(
        tmp_path / "run",
        result,  # type: ignore[arg-type]
        _provenance(),
    )

    summary = json.loads(artifact.summary_path.read_text(encoding="utf-8"))
    for field in ("candidate_config", "evaluation"):
        risk_config = dict(summary[field]["pretrade_risk_config"])
        risk_config["drawdown_start"] = 0.3
        summary[field]["pretrade_risk_config"] = risk_config
    artifact.summary_path.write_text(
        json.dumps(summary, sort_keys=True, indent=2),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="pre-trade risk config is malformed"):
        load_candidate_run_artifact(artifact.root)


def test_candidate_v3_rejects_inconsistent_realized_ppo_timesteps(
    tmp_path: Path,
) -> None:
    artifact = publish_candidate_run(
        tmp_path / "run",
        _result(),  # type: ignore[arg-type]
        _provenance(),
    )
    summary = json.loads(artifact.summary_path.read_text(encoding="utf-8"))
    summary["candidate_config"]["ppo_training_timesteps"] = 4096
    artifact.summary_path.write_text(
        json.dumps(summary, sort_keys=True, indent=2),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="realized timesteps"):
        load_candidate_run_artifact(artifact.root)


def test_historical_candidate_v1_without_observation_contract_still_loads(
    tmp_path: Path,
) -> None:
    root = tmp_path / "v1"
    _write_root(root, _summary(schema="lean_candidate_result_v1", observation=None))

    loaded = load_candidate_run_artifact(root)
    identity = inspect_candidate_run_artifact(root)

    assert loaded.summary["schema_version"] == "lean_candidate_result_v1"
    assert "ppo_observation" not in loaded.summary
    assert identity.result_schema_version == "lean_candidate_result_v1"


def test_candidate_v2_requires_and_loads_exact_observation_contract(
    tmp_path: Path,
) -> None:
    root = tmp_path / "v2"
    _write_root(
        root,
        _summary(
            schema="lean_candidate_result_v2",
            observation=ppo_observation_contract_payload(),
        ),
    )

    loaded = load_candidate_run_artifact(root)
    identity = inspect_candidate_run_artifact(root)

    assert (
        to_json_value(loaded.summary["ppo_observation"])
        == ppo_observation_contract_payload()
    )
    assert identity.result_schema_version == "lean_candidate_result_v2"


def test_candidate_v2_rejects_tampered_observation_contract(tmp_path: Path) -> None:
    root = tmp_path / "v2"
    observation = ppo_observation_contract_payload()
    observation["global_feature_names"] = ["market_return_mean"]
    _write_root(
        root,
        _summary(schema="lean_candidate_result_v2", observation=observation),
    )

    with pytest.raises(ValueError, match="PPO observation contract"):
        load_candidate_run_artifact(root)


def test_loaded_candidate_run_is_deeply_immutable(tmp_path: Path) -> None:
    root = tmp_path / "immutable"
    _write_root(
        root,
        _summary(
            schema="lean_candidate_result_v2",
            observation=ppo_observation_contract_payload(),
        ),
    )

    loaded = load_candidate_run_artifact(root)
    key = "symbol_0_strategy_0"

    with pytest.raises(TypeError):
        loaded.summary["dataset_id"] = "c" * 64

    symbols = loaded.summary["symbols"]
    assert symbols[0] == "BTCUSDT"  # type: ignore[index]
    with pytest.raises((AttributeError, TypeError)):
        symbols.append("ETHUSDT")  # type: ignore[attr-defined]

    implementation = loaded.provenance["implementation"]
    assert (
        implementation["schema_version"] == "candidate_run_implementation_v1"  # type: ignore[index]
    )
    with pytest.raises(TypeError):
        implementation["files"] = ["tampered.py"]  # type: ignore[index]

    with pytest.raises(TypeError):
        loaded.returns[key] = np.asarray([1.0], dtype=np.float64)

    values = loaded.returns[key]
    assert values.flags.writeable is False
    with pytest.raises(ValueError):
        values.setflags(write=True)
    with pytest.raises(ValueError):
        values[0] = 1.0
