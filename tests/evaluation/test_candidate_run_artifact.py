from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.comparison.strategies import (
    SharedCashStrategyComparisonEntry,
    UniversalStrategyComparison,
    compare_strategies_by_symbol,
)
from trade_rl.evaluation.metrics import evaluate_performance
from trade_rl.evaluation.replay import run_shared_cash_replay
from trade_rl.evaluation.runs import artifact as candidate_artifact
from trade_rl.evaluation.runs.config import (
    parse_candidate_run_config,
    resolve_candidate_run_spec,
)
from trade_rl.evaluation.runs.execute import CandidateRunResult
from trade_rl.evaluation.runs.provenance import build_candidate_run_provenance
from trade_rl.risk import PreTradeRisk, PreTradeRiskConfig
from trade_rl.strategies.controls import ConstantIntentStrategy
from trade_rl.strategies.position_intent import PositionIntent
from trade_rl.strategies.rl.intent import PPO_OBSERVATION_SCHEMA_V3
from trade_rl.strategies.rl.ppo import ppo_observation_contract_payload


def market() -> MarketDataset:
    n = 8
    close = np.asarray(
        [
            [100.0, 100.0],
            [100.0, 100.0],
            [105.0, 95.0],
            [110.0, 90.0],
            [115.0, 85.0],
            [120.0, 80.0],
            [125.0, 75.0],
            [130.0, 70.0],
        ]
    )
    signal = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    features = np.stack((signal, -signal), axis=1).reshape(n, 2, 1)
    return MarketDataset(
        dataset_id="b" * 64,
        symbols=("BTCUSDT", "ETHUSDT"),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(n) * np.timedelta64(1, "h"),
        features=features,
        global_features=np.zeros((n, 1), dtype=np.float32),
        open=close.copy(),
        high=close.copy(),
        low=close.copy(),
        close=close,
        volume=np.full((n, 2), 1_000_000.0),
        funding_rate=np.zeros((n, 2)),
        tradable=np.ones((n, 2), dtype=np.bool_),
        feature_available=np.ones((n, 2, 1), dtype=np.bool_),
        feature_names=("signal",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def run_config() -> dict[str, object]:
    return {
        "signal_name": "signal",
        "feature_names": ["signal"],
        "fit_symbol_names": ["BTCUSDT"],
        "fit_cutoff": "2026-01-01T04:00:00",
        "evaluation_start": "2026-01-01T04:00:00",
        "evaluation_stop_exclusive": "2026-01-01T07:00:00",
        "rule_entry_threshold": 0.10,
        "rule_exit_threshold": 0.02,
        "forecast_entry_threshold": 0.01,
        "forecast_exit_threshold": 0.002,
        "ppo_total_timesteps": 256,
        "ppo_seed": 7,
        "gross_budget": 0.5,
        "initial_capital": 1000.0,
    }


def test_run_candidate_artifact_writes_summary_and_raw_returns(
    tmp_path,
    monkeypatch,
) -> None:
    from trade_rl.evaluation.runs import candidate as candidate_run

    dataset = market()
    comparison = compare_strategies_by_symbol(
        dataset,
        {
            "cash": ConstantIntentStrategy(PositionIntent.FLAT),
            "long": ConstantIntentStrategy(PositionIntent.LONG),
        },
        start_index=4,
        stop_index=7,
        gross_budget=0.5,
        initial_capital=1_000.0,
    )
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        candidate_run,
        "inspect_published_market_dataset_artifact",
        lambda path: SimpleNamespace(
            schema_version="market_dataset_artifact_v3",
            artifact_digest="d" * 64,
        ),
    )
    monkeypatch.setattr(
        candidate_run,
        "load_market_dataset_artifact",
        lambda path: dataset,
    )

    def fake_execute(loaded, spec):
        calls["dataset"] = loaded
        calls["config"] = spec.lean_config
        calls["kwargs"] = {
            "start_index": spec.evaluation_start_index,
            "stop_index": spec.evaluation_stop_index,
            "gross_budget": spec.config.gross_budget,
            "initial_capital": spec.config.initial_capital,
            "execution_cost": None,
            "risk": None,
        }
        return CandidateRunResult(
            spec=spec,
            symbols=tuple(loaded.symbols),
            comparison=comparison,
            ppo_training_timesteps=2048,
        )

    monkeypatch.setattr(candidate_run, "execute_candidate_run", fake_execute)

    config_path = tmp_path / "run.json"
    config_path.write_text(json.dumps(run_config()), encoding="utf-8")
    output = tmp_path / "result"

    artifact = candidate_run.run_candidate_artifact(
        dataset_root=tmp_path / "dataset",
        config_path=config_path,
        output_root=output,
    )

    assert artifact.root == output
    assert artifact.summary_path == output / "summary.json"
    assert artifact.returns_path == output / "returns.npz"
    assert artifact.provenance_path == output / "provenance.json"
    summary = json.loads(artifact.summary_path.read_text(encoding="utf-8"))
    assert summary["schema_version"] == "lean_candidate_result_v6"
    assert summary["ppo_observation"] == ppo_observation_contract_payload()
    assert summary["dataset_id"] == dataset.dataset_id
    assert summary["dataset_artifact"] == {
        "schema_version": "market_dataset_artifact_v3",
        "artifact_digest": "d" * 64,
    }
    assert summary["symbols"] == ["BTCUSDT", "ETHUSDT"]
    assert summary["candidate_config"] == {
        "signal_name": "signal",
        "signal_index": 0,
        "feature_names": ["signal"],
        "feature_indices": [0],
        "fit_symbol_names": ["BTCUSDT"],
        "fit_symbol_indices": [0],
        "fit_cutoff": "2026-01-01T04:00:00.000000000",
        "rule_entry_threshold": 0.10,
        "rule_exit_threshold": 0.02,
        "forecast_entry_threshold": 0.01,
        "forecast_exit_threshold": 0.002,
        "ppo_total_timesteps": 256,
        "ppo_seed": 7,
        "ppo_training_layout": "sequential",
        "ppo_rollout_steps_per_env": None,
        "ppo_minimum_hold_bars": 0,
        "ppo_observation_schema": "ppo_observation_v2",
        "ppo_settle_terminal_position": False,
        "ppo_training_timesteps": 2048,
        "ppo_training_minimum_hold_suppressed_count": 0,
        "pretrade_risk_config": None,
    }

    age_aware_without_risk = dict(summary)
    age_aware_without_risk["candidate_config"] = {
        **summary["candidate_config"],
        "ppo_observation_schema": PPO_OBSERVATION_SCHEMA_V3,
        "ppo_settle_terminal_position": True,
    }
    age_aware_without_risk["evaluation"] = {
        **summary["evaluation"],
        "ppo_settle_terminal_position": True,
    }
    age_aware_without_risk["ppo_observation"] = ppo_observation_contract_payload(
        PPO_OBSERVATION_SCHEMA_V3
    )
    with pytest.raises(ValueError, match="explicit.*risk"):
        candidate_artifact._validate_ppo_training_evidence(
            age_aware_without_risk,
            result_schema="lean_candidate_result_v6",
        )
    assert summary["evaluation"] == {
        "start": "2026-01-01T04:00:00.000000000",
        "stop_exclusive": "2026-01-01T07:00:00.000000000",
        "gross_budget": 0.5,
        "initial_capital": 1_000.0,
        "ppo_settle_terminal_position": False,
        "expected_periods": 3,
        "pretrade_risk_config": None,
        "execution_overlay": "zero_overlay_dataset_fields_authoritative",
    }
    assert [item["symbol"] for item in summary["by_symbol"]] == [
        "BTCUSDT",
        "ETHUSDT",
    ]
    assert summary["by_symbol"][0]["strategies"][0]["name"] == "cash"
    assert summary["by_symbol"][0]["strategies"][1]["name"] == "long"

    with np.load(artifact.returns_path, allow_pickle=False) as arrays:
        assert set(arrays.files) == {
            "symbol_0_strategy_0",
            "symbol_0_strategy_1",
            "symbol_1_strategy_0",
            "symbol_1_strategy_1",
        }
        assert arrays["symbol_0_strategy_1"].shape == (3,)

    provenance = json.loads(artifact.provenance_path.read_text(encoding="utf-8"))
    assert provenance["schema_version"] == "candidate_run_provenance_v1"
    assert provenance["research_context_digest"] is None

    lean_config = calls["config"]
    assert getattr(lean_config, "signal_index") == 0
    assert getattr(lean_config, "feature_indices") == (0,)
    assert getattr(lean_config, "fit_symbol_indices") == (0,)
    assert calls["kwargs"] == {
        "start_index": 4,
        "stop_index": 7,
        "gross_budget": 0.5,
        "initial_capital": 1_000.0,
        "execution_cost": None,
        "risk": None,
    }


def test_run_candidate_artifact_rejects_unknown_config_keys(
    tmp_path,
    monkeypatch,
) -> None:
    from trade_rl.evaluation.runs import candidate as candidate_run

    monkeypatch.setattr(
        candidate_run,
        "inspect_published_market_dataset_artifact",
        lambda path: SimpleNamespace(
            schema_version="market_dataset_artifact_v3",
            artifact_digest="d" * 64,
        ),
    )
    monkeypatch.setattr(
        candidate_run,
        "load_market_dataset_artifact",
        lambda path: market(),
    )
    raw = run_config()
    raw["hidden_override"] = 1
    config_path = tmp_path / "run.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown config keys"):
        candidate_run.run_candidate_artifact(
            dataset_root=tmp_path / "dataset",
            config_path=config_path,
            output_root=tmp_path / "result",
        )


def test_run_candidate_artifact_rejects_unknown_fit_symbol(
    tmp_path,
    monkeypatch,
) -> None:
    from trade_rl.evaluation.runs import candidate as candidate_run

    monkeypatch.setattr(
        candidate_run,
        "inspect_published_market_dataset_artifact",
        lambda path: SimpleNamespace(
            schema_version="market_dataset_artifact_v3",
            artifact_digest="d" * 64,
        ),
    )
    monkeypatch.setattr(
        candidate_run,
        "load_market_dataset_artifact",
        lambda path: market(),
    )
    raw = run_config()
    raw["fit_symbol_names"] = ["UNKNOWN"]
    config_path = tmp_path / "run.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown fit symbol name: UNKNOWN"):
        candidate_run.run_candidate_artifact(
            dataset_root=tmp_path / "dataset",
            config_path=config_path,
            output_root=tmp_path / "result",
        )


def test_run_candidate_artifact_refuses_overwrite(tmp_path, monkeypatch) -> None:
    from trade_rl.evaluation.runs import candidate as candidate_run

    output = tmp_path / "result"
    output.mkdir()
    monkeypatch.setattr(
        candidate_run,
        "load_market_dataset_artifact",
        lambda path: market(),
    )
    config_path = tmp_path / "run.json"
    config_path.write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError):
        candidate_run.run_candidate_artifact(
            dataset_root=tmp_path / "dataset",
            config_path=config_path,
            output_root=output,
        )


def test_shared_cash_ppo_artifact_binds_portfolio_returns_and_ledger_digest(
    tmp_path,
) -> None:
    dataset = market()
    risk_config = PreTradeRiskConfig(
        max_gross=0.5,
        max_abs_weight=0.1,
        max_turnover=None,
        drawdown_start=0.1,
        drawdown_stop=0.2,
    )
    config = replace(
        parse_candidate_run_config(run_config()),
        ppo_observation_schema=PPO_OBSERVATION_SCHEMA_V3,
        ppo_settle_terminal_position=True,
        pretrade_risk_config=risk_config,
    )
    spec = resolve_candidate_run_spec(
        dataset,
        dataset_artifact_schema="market_dataset_artifact_v3",
        dataset_artifact_digest="d" * 64,
        config=config,
    )
    risk = PreTradeRisk(risk_config)
    per_symbol = compare_strategies_by_symbol(
        dataset,
        {"ppo": ConstantIntentStrategy(PositionIntent.FLAT)},
        start_index=4,
        stop_index=7,
        gross_budget=0.5,
        initial_capital=1_000.0,
        risk=risk,
        settle_terminal_position=True,
    )
    shared_replay = run_shared_cash_replay(
        dataset,
        (
            ConstantIntentStrategy(PositionIntent.FLAT),
            ConstantIntentStrategy(PositionIntent.FLAT),
        ),
        start_index=4,
        stop_index=7,
        gross_budget=0.5,
        initial_capital=1_000.0,
        risk=risk,
        minimum_hold_bars=0,
        settle_terminal_position=True,
        capture_ledger_evidence=True,
    )
    diagnostics = shared_replay.diagnostics
    shared_entry = SharedCashStrategyComparisonEntry(
        name="ppo",
        replay=shared_replay,
        metrics=evaluate_performance(
            shared_replay.returns,
            turnover_total=diagnostics.turnover_total,
            total_cost=diagnostics.total_cost,
            funding_pnl=diagnostics.funding_pnl,
            borrow_cost=diagnostics.borrow_cost,
            n_trades=diagnostics.n_trades,
            rebalance_events=diagnostics.rebalance_events,
            termination_count=diagnostics.termination_count,
        ),
    )
    result = CandidateRunResult(
        spec=spec,
        symbols=tuple(dataset.symbols),
        comparison=UniversalStrategyComparison(
            by_symbol=per_symbol.by_symbol,
            shared_cash_ppo=shared_entry,
            ppo_training_timesteps=2048,
        ),
        ppo_training_timesteps=2048,
    )

    published = candidate_artifact.publish_candidate_run(
        tmp_path / "shared-cash-result",
        result,
        build_candidate_run_provenance(),
    )
    loaded = candidate_artifact.load_candidate_run_artifact(published.root)

    assert loaded.has_verified_full_evaluation_coverage
    assert loaded.summary["schema_version"] == "lean_candidate_result_v8"
    portfolio = loaded.summary["shared_cash_ppo"]
    assert portfolio["name"] == "ppo"
    assert portfolio["final_portfolio_value"] == 1_000.0
    assert portfolio["terminal_settlement_complete"] is True
    assert portfolio["ledger_evidence"]["interval_count"] == 3
    assert len(portfolio["ledger_evidence"]["digest"]) == 64
    ledger_payload = portfolio["ledger_evidence"]["payload"]
    assert content_digest(ledger_payload) == portfolio["ledger_evidence"]["digest"]
    assert len(ledger_payload["intervals"]) == 3
    assert (
        len(ledger_payload["decisions"])
        == portfolio["ledger_evidence"]["decision_count"]
    )
    assert np.array_equal(loaded.returns["shared_cash_ppo"], np.zeros(3))

    v8_summary_text = published.summary_path.read_text(encoding="utf-8")
    legacy_summary = json.loads(v8_summary_text)
    legacy_summary["schema_version"] = "lean_candidate_result_v7"
    del legacy_summary["shared_cash_ppo"]["ledger_evidence"]["payload"]
    published.summary_path.write_text(json.dumps(legacy_summary), encoding="utf-8")
    legacy_loaded = candidate_artifact.load_candidate_run_artifact(published.root)
    assert legacy_loaded.summary["schema_version"] == "lean_candidate_result_v7"
    assert legacy_loaded.has_verified_full_evaluation_coverage

    published.summary_path.write_text(v8_summary_text, encoding="utf-8")
    tampered_summary = json.loads(published.summary_path.read_text(encoding="utf-8"))
    tampered_summary["shared_cash_ppo"]["ledger_evidence"]["payload"]["intervals"][0][
        "cash_after"
    ] += 1.0
    published.summary_path.write_text(json.dumps(tampered_summary), encoding="utf-8")
    with pytest.raises(ValueError, match="ledger.*digest|digest.*ledger"):
        candidate_artifact.load_candidate_run_artifact(published.root)
