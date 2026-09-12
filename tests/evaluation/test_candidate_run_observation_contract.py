from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.runs.artifact import (
    inspect_candidate_run_artifact,
    load_candidate_run_artifact,
    publish_candidate_run,
)
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
            diagnostics=diagnostics,
            book=SimpleNamespace(portfolio_value=1000.0, fill_count=0),
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
    )


def test_new_candidate_write_records_observation_v2_contract(tmp_path: Path) -> None:
    artifact = publish_candidate_run(
        tmp_path / "run",
        _result(),  # type: ignore[arg-type]
        _provenance(),
    )

    summary = json.loads(artifact.summary_path.read_text(encoding="utf-8"))

    assert summary["schema_version"] == "lean_candidate_result_v2"
    assert summary["ppo_observation"] == ppo_observation_contract_payload()


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

    assert loaded.summary["ppo_observation"] == ppo_observation_contract_payload()
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
