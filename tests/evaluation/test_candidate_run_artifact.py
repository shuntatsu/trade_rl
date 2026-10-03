from __future__ import annotations

import json
from copy import deepcopy
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
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.controls import ConstantIntentStrategy
from trade_rl.strategies.interface import StrategyObservation
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


class _SequenceIntentStrategy:
    observation_schema = PPO_OBSERVATION_SCHEMA_V3

    def __init__(self, intents: tuple[PositionIntent, ...]) -> None:
        self._intents = intents
        self._index = 0
        self.observed_indices: list[int] = []
        self.observed_ages: list[int] = []

    def decide(self, observation: StrategyObservation) -> PositionIntent:
        self.observed_indices.append(observation.index)
        self.observed_ages.append(observation.position_age_bars)
        intent = self._intents[min(self._index, len(self._intents) - 1)]
        self._index += 1
        return intent


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


def test_shared_cash_ppo_artifact_binds_portfolio_returns_ledger_and_intrabar_drawdown(
    tmp_path,
) -> None:
    dataset = market()
    close = np.full(dataset.close.shape, 100.0, dtype=np.float64)
    open_prices = close.copy()
    open_prices[5, 0] = 60.0
    low_prices = np.minimum(open_prices, close)
    low_prices[5, 0] = 40.0
    dataset = replace(
        dataset,
        open=open_prices,
        high=np.maximum(open_prices, close),
        low=low_prices,
        close=close,
        mark_price=close.copy(),
    )
    risk_config = PreTradeRiskConfig(
        max_gross=0.5,
        max_abs_weight=0.5,
        max_turnover=None,
        drawdown_start=0.1,
        drawdown_stop=0.2,
    )
    raw_config = run_config()
    raw_config.update(
        {
            "fit_cutoff": "2026-01-01T03:00:00",
            "evaluation_start": "2026-01-01T03:00:00",
            "evaluation_stop_exclusive": "2026-01-01T07:00:00",
        }
    )
    config = replace(
        parse_candidate_run_config(raw_config),
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
        start_index=3,
        stop_index=7,
        gross_budget=0.5,
        initial_capital=1_000.0,
        risk=risk,
        settle_terminal_position=True,
    )
    shared_replay = run_shared_cash_replay(
        dataset,
        (
            _SequenceIntentStrategy(
                (
                    PositionIntent.LONG,
                    PositionIntent.LONG,
                    PositionIntent.FLAT,
                )
            ),
            ConstantIntentStrategy(PositionIntent.FLAT),
        ),
        start_index=3,
        stop_index=7,
        gross_budget=0.5,
        initial_capital=1_000.0,
        risk=risk,
        minimum_hold_bars=0,
        settle_terminal_position=True,
        capture_ledger_evidence=True,
        capture_accounting_evidence=True,
    )
    diagnostics = shared_replay.diagnostics
    metrics = evaluate_performance(
        shared_replay.returns,
        observed_max_drawdown=shared_replay.book.max_drawdown,
        turnover_total=diagnostics.turnover_total,
        total_cost=diagnostics.total_cost,
        funding_pnl=diagnostics.funding_pnl,
        borrow_cost=diagnostics.borrow_cost,
        n_trades=diagnostics.n_trades,
        rebalance_events=diagnostics.rebalance_events,
        termination_count=diagnostics.termination_count,
    )
    shared_entry = SharedCashStrategyComparisonEntry(
        name="ppo",
        replay=shared_replay,
        metrics=metrics,
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
    assert loaded.summary["schema_version"] == "lean_candidate_result_v11"
    portfolio = loaded.summary["shared_cash_ppo"]
    assert portfolio["name"] == "ppo"
    assert portfolio["final_portfolio_value"] == 1_000.0
    assert portfolio["metrics"]["max_drawdown"] == pytest.approx(0.3)
    assert portfolio["terminal_settlement_complete"] is True
    assert portfolio["ledger_evidence"]["interval_count"] == 4
    assert len(portfolio["ledger_evidence"]["digest"]) == 64
    ledger_payload = portfolio["ledger_evidence"]["payload"]
    assert content_digest(ledger_payload) == portfolio["ledger_evidence"]["digest"]
    assert ledger_payload["schema_version"] == "shared_cash_replay_ledger_v4"
    transitions = [
        transition
        for interval in ledger_payload["intervals"]
        for transition in interval["accounting_transitions"]
    ]
    assert transitions
    assert any(
        transition["transition_type"] == "mark_revaluation"
        for transition in transitions
    )
    assert len(ledger_payload["intervals"]) == 4
    assert ledger_payload["final_max_drawdown"] == pytest.approx(0.3)
    assert (
        len(ledger_payload["decisions"])
        == portfolio["ledger_evidence"]["decision_count"]
    )
    assert np.array_equal(loaded.returns["shared_cash_ppo"], np.zeros(4))
    assert loaded.summary["evaluation"]["ppo_policy_decision_stop_index"] == 6

    v10_summary_text = published.summary_path.read_text(encoding="utf-8")

    forged_drawdown_summary = json.loads(v10_summary_text)
    forged_drawdown_portfolio = forged_drawdown_summary["shared_cash_ppo"]
    forged_drawdown_ledger = forged_drawdown_portfolio["ledger_evidence"]
    forged_drawdown_payload = forged_drawdown_ledger["payload"]
    forged_drawdown_payload["final_max_drawdown"] = 0.0
    forged_drawdown_portfolio["metrics"]["max_drawdown"] = 0.0
    for interval in forged_drawdown_payload["intervals"]:
        interval["max_drawdown_before"] = 0.0
        interval["max_drawdown_after"] = 0.0
    forged_drawdown_ledger["digest"] = content_digest(forged_drawdown_payload)
    published.summary_path.write_text(
        json.dumps(forged_drawdown_summary), encoding="utf-8"
    )
    with pytest.raises(
        ValueError, match="candidate v10 accounting drawdown link is inconsistent"
    ):
        candidate_artifact.load_candidate_run_artifact(published.root)

    # v8 keeps the prior v1/v2 ledger read contract; strict semantic validation
    # is introduced by v9.
    compatibility_error = None
    for ledger_schema in (
        "shared_cash_replay_ledger_v1",
        "shared_cash_replay_ledger_v2",
    ):
        legacy_v8_summary = json.loads(v10_summary_text)
        legacy_v8_summary["schema_version"] = "lean_candidate_result_v8"
        legacy_v8_summary["shared_cash_ppo"]["metrics"]["max_drawdown"] = 0.0
        legacy_v8_ledger = legacy_v8_summary["shared_cash_ppo"]["ledger_evidence"]
        legacy_v8_payload = legacy_v8_ledger["payload"]
        legacy_v8_ledger["schema_version"] = ledger_schema
        legacy_v8_payload["schema_version"] = ledger_schema
        legacy_v8_payload["final_max_drawdown"] = 0.0
        legacy_v8_payload.pop("contract_multipliers", None)
        legacy_v8_payload.pop("initial_mark_prices", None)
        for interval in legacy_v8_payload["intervals"]:
            interval.pop("accounting_transitions", None)
            interval["max_drawdown_before"] = 0.0
            interval["max_drawdown_after"] = 0.0
        if ledger_schema == "shared_cash_replay_ledger_v1":
            del legacy_v8_payload["decisions"]
        legacy_v8_ledger["digest"] = content_digest(legacy_v8_payload)
        published.summary_path.write_text(
            json.dumps(legacy_v8_summary), encoding="utf-8"
        )
        try:
            legacy_v8_loaded = candidate_artifact.load_candidate_run_artifact(
                published.root
            )
        except ValueError as error:
            compatibility_error = error
        else:
            assert (
                legacy_v8_loaded.summary["schema_version"] == "lean_candidate_result_v8"
            )
            assert legacy_v8_loaded.has_verified_full_evaluation_coverage

    legacy_v9_summary = json.loads(v10_summary_text)
    legacy_v9_summary["schema_version"] = "lean_candidate_result_v9"
    legacy_v9_summary["shared_cash_ppo"]["metrics"]["max_drawdown"] = 0.0
    legacy_v9_ledger = legacy_v9_summary["shared_cash_ppo"]["ledger_evidence"]
    legacy_v9_payload = legacy_v9_ledger["payload"]
    legacy_v9_ledger["schema_version"] = "shared_cash_replay_ledger_v2"
    legacy_v9_payload["schema_version"] = "shared_cash_replay_ledger_v2"
    legacy_v9_payload["final_max_drawdown"] = 0.0
    legacy_v9_payload.pop("contract_multipliers", None)
    legacy_v9_payload.pop("initial_mark_prices", None)
    for interval in legacy_v9_payload["intervals"]:
        interval.pop("accounting_transitions", None)
        interval["max_drawdown_before"] = 0.0
        interval["max_drawdown_after"] = 0.0
    legacy_v9_ledger["digest"] = content_digest(legacy_v9_payload)
    published.summary_path.write_text(json.dumps(legacy_v9_summary), encoding="utf-8")
    legacy_v9_loaded = candidate_artifact.load_candidate_run_artifact(published.root)
    assert legacy_v9_loaded.summary["schema_version"] == "lean_candidate_result_v9"

    legacy_summary = json.loads(v10_summary_text)
    legacy_summary["schema_version"] = "lean_candidate_result_v7"
    legacy_summary["shared_cash_ppo"]["metrics"]["max_drawdown"] = 0.0
    legacy_v7_ledger = legacy_summary["shared_cash_ppo"]["ledger_evidence"]
    legacy_v7_ledger["schema_version"] = "shared_cash_replay_ledger_v2"
    del legacy_summary["shared_cash_ppo"]["ledger_evidence"]["payload"]
    published.summary_path.write_text(json.dumps(legacy_summary), encoding="utf-8")
    legacy_loaded = candidate_artifact.load_candidate_run_artifact(published.root)
    assert legacy_loaded.summary["schema_version"] == "lean_candidate_result_v7"
    assert legacy_loaded.has_verified_full_evaluation_coverage

    def downgrade_to_ledger_v1(summary, payload) -> None:
        summary["shared_cash_ppo"]["ledger_evidence"]["schema_version"] = (
            "shared_cash_replay_ledger_v1"
        )
        payload["schema_version"] = "shared_cash_replay_ledger_v1"
        payload.pop("contract_multipliers", None)
        payload.pop("initial_mark_prices", None)
        del payload["decisions"]
        for interval in payload["intervals"]:
            interval.pop("accounting_transitions", None)

    def remove_first_policy_decision(summary, payload) -> None:
        del payload["decisions"][0]
        summary["shared_cash_ppo"]["ledger_evidence"]["decision_count"] -= 1

    def remove_last_policy_decision(summary, payload) -> None:
        del payload["decisions"][-1]
        summary["shared_cash_ppo"]["ledger_evidence"]["decision_count"] -= 1

    def create_policy_decision_gap(summary, payload) -> None:
        payload["decisions"][1]["index"] += 1

    def alter_policy_decision_stop(summary, payload) -> None:
        summary["evaluation"]["ppo_policy_decision_stop_index"] -= 1

    def forge_first_interval_start_capital(summary, payload) -> None:
        payload["intervals"][0]["cash_before"] += 1.0
        payload["intervals"][0]["portfolio_value_before"] += 1.0

    def shift_ledger_window(summary, payload) -> None:
        payload["start_index"] += 1
        payload["stop_index"] += 1
        summary["evaluation"]["ppo_policy_decision_stop_index"] += 1
        for interval in payload["intervals"]:
            interval["start_index"] += 1
            interval["next_index"] += 1
        for decision in payload["decisions"]:
            decision["index"] += 1

    def alter_decision_quantity_before(summary, payload) -> None:
        payload["decisions"][0]["position_quantity_before"][0] += 1.0

    def alter_decision_quantity_after(summary, payload) -> None:
        payload["decisions"][0]["position_quantity_after"][0] += 1.0

    def alter_decision_target_weight(summary, payload) -> None:
        payload["decisions"][0]["target_weights"][0] += 0.2

    def replace_order_events_with_scalar(summary, payload) -> None:
        payload["intervals"][0]["order_events"] = [7]

    def replace_capacity_events_with_scalar(summary, payload) -> None:
        payload["intervals"][0]["capacity_events"] = [7]

    def replace_funding_events_with_scalar(summary, payload) -> None:
        payload["intervals"][0]["funding_events"] = [7]

    def link_non_fill_transition_to_order_event(summary, payload) -> None:
        transition = next(
            item
            for item in payload["intervals"][0]["accounting_transitions"]
            if item["transition_type"] == "mark_revaluation"
        )
        transition["order_event_sequence"] = 0

    def make_interval_financial_value_boolean(summary, payload) -> None:
        payload["intervals"][0]["cash_after"] = True

    def break_interval_continuity(summary, payload) -> None:
        payload["intervals"][1]["cash_before"] += 1.0

    def break_terminal_row_link(summary, payload) -> None:
        payload["intervals"][-1]["cash_after"] += 1.0

    def break_portfolio_summary_link(summary, payload) -> None:
        payload["intervals"][-1]["cash_after"] += 1.0
        payload["final_cash"] += 1.0

    cases = (
        ("ledger schema", downgrade_to_ledger_v1, "ledger schema"),
        (
            "missing first policy decision",
            remove_first_policy_decision,
            "decision coverage",
        ),
        (
            "missing last policy decision",
            remove_last_policy_decision,
            "decision coverage",
        ),
        ("policy decision gap", create_policy_decision_gap, "decision index"),
        (
            "policy decision stop mismatch",
            alter_policy_decision_stop,
            "decision coverage boundary",
        ),
        (
            "forged first interval start capital",
            forge_first_interval_start_capital,
            "initial ledger link",
        ),
        (
            "shifted ledger evaluation window",
            shift_ledger_window,
            "evaluation index link",
        ),
        (
            "decision quantity before link",
            alter_decision_quantity_before,
            "decision interval link",
        ),
        (
            "decision quantity after link",
            alter_decision_quantity_after,
            "decision interval link",
        ),
        (
            "decision target weight risk link",
            alter_decision_target_weight,
            "decision risk link",
        ),
        ("scalar order event", replace_order_events_with_scalar, "order event"),
        (
            "scalar capacity event",
            replace_capacity_events_with_scalar,
            "capacity event",
        ),
        (
            "scalar funding event",
            replace_funding_events_with_scalar,
            "funding event",
        ),
        (
            "non-fill transition linked to order event",
            link_non_fill_transition_to_order_event,
            "event sequence",
        ),
        (
            "boolean interval financial value",
            make_interval_financial_value_boolean,
            "financial value",
        ),
        ("interval continuity", break_interval_continuity, "continuity"),
        ("terminal row link", break_terminal_row_link, "terminal ledger link"),
        (
            "portfolio summary link",
            break_portfolio_summary_link,
            "portfolio ledger link",
        ),
    )
    failures = []
    for name, mutate, expected_message in cases:
        tampered_summary = json.loads(v10_summary_text)
        ledger_evidence = tampered_summary["shared_cash_ppo"]["ledger_evidence"]
        ledger_payload = ledger_evidence["payload"]
        mutate(tampered_summary, ledger_payload)
        ledger_evidence["digest"] = content_digest(ledger_payload)
        published.summary_path.write_text(
            json.dumps(tampered_summary), encoding="utf-8"
        )
        try:
            candidate_artifact.load_candidate_run_artifact(published.root)
        except ValueError as error:
            if expected_message not in str(error):
                failures.append(f"{name}: rejected for {error!s}")
        else:
            failures.append(f"{name}: accepted")

    return_link_summary = json.loads(v10_summary_text)
    original_returns_bytes = published.returns_path.read_bytes()
    try:
        with np.load(published.returns_path, allow_pickle=False) as archive:
            changed_returns = {
                key: np.asarray(archive[key]).copy() for key in archive.files
            }
        changed_returns["shared_cash_ppo"][0] += 0.1
        wealth = 1.0
        for value in changed_returns["shared_cash_ppo"]:
            wealth *= 1.0 + float(value)

        return_metrics = return_link_summary["shared_cash_ppo"]["metrics"]
        return_metrics["total_return"] = wealth - 1.0
        np.savez(published.returns_path, **changed_returns)
        published.summary_path.write_text(
            json.dumps(return_link_summary), encoding="utf-8"
        )
        try:
            candidate_artifact.load_candidate_run_artifact(published.root)
        except ValueError as error:
            if "return ledger link" not in str(error):
                failures.append(f"persisted return link: rejected for {error!s}")
        else:
            failures.append("persisted return link: accepted")
    finally:
        published.returns_path.write_bytes(original_returns_bytes)
        published.summary_path.write_text(v10_summary_text, encoding="utf-8")

    arithmetic_summary = json.loads(v10_summary_text)
    arithmetic_ledger = arithmetic_summary["shared_cash_ppo"]["ledger_evidence"]
    arithmetic_payload = arithmetic_ledger["payload"]
    forged_return = arithmetic_payload["intervals"][0]["interval_net_return"] + 0.1
    arithmetic_payload["intervals"][0]["interval_net_return"] = forged_return
    arithmetic_ledger["digest"] = content_digest(arithmetic_payload)
    original_returns_bytes = published.returns_path.read_bytes()
    try:
        with np.load(published.returns_path, allow_pickle=False) as archive:
            forged_returns = {
                key: np.asarray(archive[key]).copy() for key in archive.files
            }
        forged_returns["shared_cash_ppo"][0] = forged_return
        wealth = 1.0
        for value in forged_returns["shared_cash_ppo"]:
            wealth *= 1.0 + float(value)
        arithmetic_metrics = arithmetic_summary["shared_cash_ppo"]["metrics"]
        arithmetic_metrics["total_return"] = wealth - 1.0
        np.savez(published.returns_path, **forged_returns)
        published.summary_path.write_text(
            json.dumps(arithmetic_summary), encoding="utf-8"
        )
        try:
            candidate_artifact.load_candidate_run_artifact(published.root)
        except ValueError as error:
            if "interval return accounting" not in str(error):
                failures.append(f"return/value reconciliation: rejected for {error!s}")
        else:
            failures.append("return/value reconciliation: accepted")
    finally:
        published.returns_path.write_bytes(original_returns_bytes)
        published.summary_path.write_text(v10_summary_text, encoding="utf-8")

    if compatibility_error is not None:
        failures.append(f"v8/v1 compatibility: rejected for {compatibility_error!s}")
    assert failures == [], "; ".join(failures)


def test_v10_shared_cash_minimum_hold_suppression_and_unlock_survive_artifact_round_trip(
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
    raw_config = run_config()
    raw_config.update(
        {
            "fit_cutoff": "2026-01-01T02:00:00",
            "evaluation_start": "2026-01-01T02:00:00",
            "evaluation_stop_exclusive": "2026-01-01T07:00:00",
            "ppo_minimum_hold_bars": 3,
            "ppo_observation_schema": PPO_OBSERVATION_SCHEMA_V3,
            "ppo_settle_terminal_position": True,
            "pretrade_risk_config": {
                "max_gross": 0.5,
                "max_abs_weight": 0.1,
                "max_turnover": None,
                "drawdown_start": 0.1,
                "drawdown_stop": 0.2,
                "emergency_turnover_override": True,
                "fail_closed_tolerance": 1e-10,
            },
        }
    )
    config = parse_candidate_run_config(raw_config)
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
        start_index=2,
        stop_index=7,
        gross_budget=0.5,
        initial_capital=1_000.0,
        risk=risk,
        settle_terminal_position=True,
    )
    strategies = (
        _SequenceIntentStrategy(
            (
                PositionIntent.LONG,
                PositionIntent.FLAT,
                PositionIntent.FLAT,
                PositionIntent.FLAT,
            )
        ),
        _SequenceIntentStrategy(
            (
                PositionIntent.LONG,
                PositionIntent.FLAT,
                PositionIntent.FLAT,
                PositionIntent.FLAT,
            )
        ),
    )
    shared_replay = run_shared_cash_replay(
        dataset,
        strategies,
        start_index=2,
        stop_index=7,
        gross_budget=0.5,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
        risk=risk,
        minimum_hold_bars=3,
        settle_terminal_position=True,
        capture_ledger_evidence=True,
        capture_accounting_evidence=True,
    )
    diagnostics = shared_replay.diagnostics
    shared_entry = SharedCashStrategyComparisonEntry(
        name="ppo",
        replay=shared_replay,
        metrics=evaluate_performance(
            shared_replay.returns,
            observed_max_drawdown=shared_replay.book.max_drawdown,
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
            ppo_training_timesteps=2_048,
        ),
        ppo_training_timesteps=2_048,
    )

    published = candidate_artifact.publish_candidate_run(
        tmp_path / "shared-cash-minimum-hold-result",
        result,
        build_candidate_run_provenance(),
    )
    loaded = candidate_artifact.load_candidate_run_artifact(published.root)
    portfolio = loaded.summary["shared_cash_ppo"]
    ledger_payload = portfolio["ledger_evidence"]["payload"]
    decisions = ledger_payload["decisions"]

    assert strategies[0].observed_indices == [2, 3, 4, 5]
    assert strategies[1].observed_indices == [2, 3, 4, 5]
    assert strategies[0].observed_ages == [0, 1, 2, 3]
    assert strategies[1].observed_ages == [0, 1, 2, 3]
    assert [decision["minimum_hold_suppressed"] for decision in decisions] == [
        (False, False),
        (True, True),
        (True, True),
        (False, False),
    ]
    assert [decision["minimum_hold_unlocked"] for decision in decisions] == [
        (False, False),
        (False, False),
        (False, False),
        (True, True),
    ]
    assert [decision["position_age_bars_before"] for decision in decisions] == [
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 3),
    ]
    assert loaded.summary["evaluation"]["ppo_minimum_hold_bars"] == 3
    assert portfolio["terminal_settlement_complete"] is True
    assert decisions[-1]["effective_intents"] == (
        PositionIntent.FLAT.value,
        PositionIntent.FLAT.value,
    )
    assert decisions[-1]["position_quantity_after"] == (0.0, 0.0)
    np.testing.assert_allclose(
        loaded.returns["shared_cash_ppo"], shared_replay.returns.values
    )

    tampered_summary = json.loads(published.summary_path.read_text(encoding="utf-8"))
    tampered_ledger = tampered_summary["shared_cash_ppo"]["ledger_evidence"]
    tampered_payload = tampered_ledger["payload"]
    tampered_payload["decisions"][-1]["minimum_hold_unlocked"] = [False, False]
    tampered_ledger["digest"] = content_digest(tampered_payload)
    published.summary_path.write_text(json.dumps(tampered_summary), encoding="utf-8")
    with pytest.raises(
        ValueError, match="candidate v10 decision intent link is inconsistent"
    ):
        candidate_artifact.load_candidate_run_artifact(published.root)


def test_v10_shared_cash_artifact_rejects_forged_execution_state(tmp_path) -> None:
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
    by_symbol = compare_strategies_by_symbol(
        dataset,
        {"ppo": ConstantIntentStrategy(PositionIntent.LONG)},
        start_index=4,
        stop_index=7,
        gross_budget=0.5,
        initial_capital=1_000.0,
        risk=risk,
        settle_terminal_position=True,
    )
    replay = run_shared_cash_replay(
        dataset,
        (
            ConstantIntentStrategy(PositionIntent.LONG),
            ConstantIntentStrategy(PositionIntent.LONG),
        ),
        start_index=4,
        stop_index=7,
        gross_budget=0.5,
        initial_capital=1_000.0,
        risk=risk,
        minimum_hold_bars=0,
        settle_terminal_position=True,
        capture_ledger_evidence=True,
        capture_accounting_evidence=True,
    )
    assert replay.ledger_evidence is not None
    from fractions import Fraction

    exact_close_fill_observed = False
    for interval in replay.ledger_evidence.intervals:
        for transition in interval.accounting_transitions:
            if transition.transition_type != "fill":
                continue
            symbol_index = int(transition.evidence["symbol_index"])
            before_quantity = Fraction(
                transition.state_before.exact_quantities[symbol_index]
            )
            after_quantity = Fraction(
                transition.state_after.exact_quantities[symbol_index]
            )
            accepted_quantity = Fraction(
                str(transition.evidence["filled_quantity_exact"])
            )
            if after_quantity == 0 and before_quantity + accepted_quantity != 0:
                exact_close_fill_observed = True
    assert exact_close_fill_observed
    diagnostics = replay.diagnostics
    shared_cash_ppo = SharedCashStrategyComparisonEntry(
        name="ppo",
        replay=replay,
        metrics=evaluate_performance(
            replay.returns,
            observed_max_drawdown=replay.book.max_drawdown,
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
            by_symbol=by_symbol.by_symbol,
            shared_cash_ppo=shared_cash_ppo,
            ppo_training_timesteps=2048,
        ),
        ppo_training_timesteps=2048,
    )

    published = candidate_artifact.publish_candidate_run(
        tmp_path / "shared-cash-execution-result",
        result,
        build_candidate_run_provenance(),
    )
    loaded = candidate_artifact.load_candidate_run_artifact(published.root)
    ledger_payload = loaded.summary["shared_cash_ppo"]["ledger_evidence"]["payload"]
    intervals = ledger_payload["intervals"]
    exact_close_transition = next(
        transition
        for interval in intervals
        for transition in interval["accounting_transitions"]
        if transition["transition_type"] == "fill"
        and Fraction(transition["state_after"]["exact_quantities"][0]) == 0
        and Fraction(transition["state_before"]["exact_quantities"][0])
        + Fraction(transition["evidence"]["filled_quantity_exact"])
        != 0
    )
    exact_close_evidence = exact_close_transition["evidence"]
    assert exact_close_evidence["filled_lot_count"] is None
    assert Fraction(exact_close_evidence["book_applied_quantity_exact"]) == -Fraction(
        exact_close_transition["state_before"]["exact_quantities"][0]
    )
    order_events = [
        event for interval in intervals for event in interval["order_events"]
    ]
    capacity_events = [
        event for interval in intervals for event in interval["capacity_events"]
    ]

    assert any(
        event["event_type"] in {"filled", "partial_fill"} for event in order_events
    )
    assert capacity_events

    original_summary = published.summary_path.read_text(encoding="utf-8")

    legacy_summary = json.loads(original_summary)
    legacy_ledger = legacy_summary["shared_cash_ppo"]["ledger_evidence"]
    legacy_payload = legacy_ledger["payload"]
    legacy_fill_count = 0
    for interval in legacy_payload["intervals"]:
        for transition in interval["accounting_transitions"]:
            if transition["transition_type"] != "fill":
                continue
            evidence = transition["evidence"]
            symbol_index = evidence["symbol_index"]
            actual_delta = Fraction(
                transition["state_after"]["exact_quantities"][symbol_index]
            ) - Fraction(transition["state_before"]["exact_quantities"][symbol_index])
            if actual_delta == Fraction(evidence["filled_quantity_exact"]):
                evidence.pop("book_applied_quantity_exact")
                evidence.pop("filled_lot_count")
                evidence.pop("filled_lot_size")
                legacy_fill_count += 1
    assert legacy_fill_count > 0
    legacy_ledger["digest"] = content_digest(legacy_payload)
    published.summary_path.write_text(json.dumps(legacy_summary), encoding="utf-8")
    candidate_artifact.load_candidate_run_artifact(published.root)

    stripped_close_summary = json.loads(original_summary)
    stripped_close_ledger = stripped_close_summary["shared_cash_ppo"]["ledger_evidence"]
    stripped_close_payload = stripped_close_ledger["payload"]
    stripped_close_transition = next(
        transition
        for interval in stripped_close_payload["intervals"]
        for transition in interval["accounting_transitions"]
        if transition["transition_type"] == "fill"
        and transition["state_after"]["exact_quantities"][0] == "0"
        and Fraction(transition["state_before"]["exact_quantities"][0])
        + Fraction(transition["evidence"]["filled_quantity_exact"])
        != 0
    )
    for field in (
        "book_applied_quantity_exact",
        "filled_lot_count",
        "filled_lot_size",
    ):
        stripped_close_transition["evidence"].pop(field)
    stripped_close_ledger["digest"] = content_digest(stripped_close_payload)
    published.summary_path.write_text(
        json.dumps(stripped_close_summary), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="applied fill quantity is inconsistent"):
        candidate_artifact.load_candidate_run_artifact(published.root)

    forged_summary = json.loads(original_summary)
    forged_ledger = forged_summary["shared_cash_ppo"]["ledger_evidence"]
    forged_payload = forged_ledger["payload"]
    forged_transition = next(
        transition
        for interval in forged_payload["intervals"]
        for transition in interval["accounting_transitions"]
        if transition["transition_type"] == "fill"
        and transition["state_after"]["exact_quantities"][0] == "0"
        and Fraction(transition["state_before"]["exact_quantities"][0])
        + Fraction(transition["evidence"]["filled_quantity_exact"])
        != 0
    )
    forged_transition["evidence"]["book_applied_quantity_exact"] = forged_transition[
        "evidence"
    ]["filled_quantity_exact"]
    forged_ledger["digest"] = content_digest(forged_payload)
    published.summary_path.write_text(json.dumps(forged_summary), encoding="utf-8")
    with pytest.raises(ValueError, match="applied fill quantity is inconsistent"):
        candidate_artifact.load_candidate_run_artifact(published.root)

    forged_summary = json.loads(original_summary)
    forged_ledger = forged_summary["shared_cash_ppo"]["ledger_evidence"]
    forged_payload = forged_ledger["payload"]
    forged_intervals = forged_payload["intervals"]
    forged_decisions = forged_payload["decisions"]
    forged_quantity = str(
        Fraction(forged_intervals[0]["exact_quantities_after"][0]) + 1
    )
    forged_intervals[0]["exact_quantities_after"][0] = forged_quantity
    forged_intervals[1]["exact_quantities_before"][0] = forged_quantity
    forged_decisions[0]["position_quantity_after"][0] += 1.0
    forged_decisions[1]["position_quantity_before"][0] += 1.0
    forged_intervals[0]["cash_after"] += 1.0
    forged_intervals[1]["cash_before"] += 1.0
    forged_ledger["digest"] = content_digest(forged_payload)
    published.summary_path.write_text(json.dumps(forged_summary), encoding="utf-8")
    with pytest.raises(ValueError, match="accounting transition"):
        candidate_artifact.load_candidate_run_artifact(published.root)


@pytest.mark.parametrize(
    ("scenario", "expected_transition"),
    (
        ("split", "split"),
        ("delisting", "delisting_settlement"),
        ("funding_and_borrow", "funding_mark"),
        ("termination", "termination_flatten"),
    ),
)
def test_v10_shared_cash_ledger_records_ordered_accounting_transitions(
    scenario: str,
    expected_transition: str,
) -> None:
    dataset = market()
    execution_cost = ExecutionCostConfig.zero()
    risk_config = PreTradeRiskConfig(
        max_gross=0.5,
        max_abs_weight=0.5,
        max_turnover=None,
        drawdown_start=0.5,
        drawdown_stop=0.9,
    )
    intent = PositionIntent.LONG
    gross_budget = 0.5
    if scenario == "split":
        split_factor = np.ones((dataset.n_bars, dataset.n_symbols))
        split_factor[6, 0] = 2.0
        dataset = replace(dataset, split_factor=split_factor)
    elif scenario == "delisting":
        asset_active = np.ones((dataset.n_bars, dataset.n_symbols), dtype=np.bool_)
        asset_active[6, 0] = False
        tradable = np.ones((dataset.n_bars, dataset.n_symbols), dtype=np.bool_)
        tradable[6, 0] = False
        feature_available = np.ones(
            (dataset.n_bars, dataset.n_symbols, dataset.features.shape[2]),
            dtype=np.bool_,
        )
        feature_available[6, 0] = False
        feature_staleness = np.asarray(dataset.feature_staleness).copy()
        feature_staleness[6, 0] = 1.0
        information_available = np.asarray(
            dataset.information_available, dtype=np.bool_
        ).copy()
        information_available[6, 0] = False
        dataset = replace(
            dataset,
            asset_active=asset_active,
            symbol_active=asset_active,
            tradable=tradable,
            feature_available=feature_available,
            feature_staleness=feature_staleness,
            information_available=information_available,
        )
    elif scenario == "funding_and_borrow":
        funding_rate = np.zeros((dataset.n_bars, dataset.n_symbols))
        funding_rate[5, :] = 0.01
        funding_due = np.zeros((dataset.n_bars, dataset.n_symbols), dtype=np.bool_)
        funding_due[5, :] = True
        borrow_rate = np.zeros((dataset.n_bars, dataset.n_symbols))
        borrow_rate[5, :] = 0.1
        dataset = replace(
            dataset,
            funding_rate=funding_rate,
            funding_due=funding_due,
            borrow_rate=borrow_rate,
        )
        intent = PositionIntent.SHORT
    elif scenario == "termination":
        execution_cost = replace(
            execution_cost,
            maintenance_margin_rate=1.0,
            collateral_haircut=0.01,
        )
        risk_config = replace(
            risk_config,
            max_gross=1.0,
            max_abs_weight=1.0,
        )
        gross_budget = 1.0
    else:
        raise AssertionError(f"unexpected synthetic accounting scenario: {scenario}")

    replay = run_shared_cash_replay(
        dataset,
        tuple(ConstantIntentStrategy(intent) for _ in dataset.symbols),
        start_index=4,
        stop_index=7,
        gross_budget=gross_budget,
        initial_capital=1_000.0,
        execution_cost=execution_cost,
        risk=PreTradeRisk(risk_config),
        minimum_hold_bars=0,
        settle_terminal_position=False,
        capture_ledger_evidence=True,
        capture_accounting_evidence=True,
    )

    assert replay.ledger_evidence is not None
    ledger = replay.ledger_evidence.to_mapping()
    assert ledger["schema_version"] == "shared_cash_replay_ledger_v4"
    transitions = [
        transition
        for interval in ledger["intervals"]
        for transition in interval["accounting_transitions"]
    ]
    assert any(
        transition["transition_type"] == expected_transition
        for transition in transitions
    )
    if scenario == "funding_and_borrow":
        assert any(
            transition["transition_type"] == "borrow_charge"
            for transition in transitions
        )
        assert any(
            event["funding_amount"] != 0.0
            for interval in ledger["intervals"]
            for event in interval["funding_events"]
        )
    candidate_artifact._validate_v10_accounting_transitions(
        payload=ledger,
        intervals=ledger["intervals"],
        symbols=dataset.symbols,
        summary={
            "evaluation": {"initial_capital": 1_000.0},
            "shared_cash_ppo": {"metrics": {"max_drawdown": replay.book.max_drawdown}},
        },
    )
    if scenario == "split":
        forged_ledger = deepcopy(ledger)
        fill_stress = next(
            transition
            for interval in forged_ledger["intervals"]
            for transition in interval["accounting_transitions"]
            if transition["transition_type"] == "ohlc_drawdown_stress"
            and transition["evidence"]["phase"] == "after_fill"
        )
        fill_stress["evidence"]["fill_event_sequence"] += 1000
        with pytest.raises(ValueError, match="OHLC stress fill event link"):
            candidate_artifact._validate_v10_accounting_transitions(
                payload=forged_ledger,
                intervals=forged_ledger["intervals"],
                symbols=dataset.symbols,
                summary={
                    "evaluation": {"initial_capital": 1_000.0},
                    "shared_cash_ppo": {
                        "metrics": {"max_drawdown": replay.book.max_drawdown}
                    },
                },
            )
    if scenario == "split":
        forged_ledger = deepcopy(ledger)
        fill_interval = next(
            interval
            for interval in forged_ledger["intervals"]
            if any(
                event["event_type"] in {"filled", "partial_fill"}
                for event in interval["order_events"]
            )
        )
        fill_event = next(
            event
            for event in fill_interval["order_events"]
            if event["event_type"] in {"filled", "partial_fill"}
        )
        fill_transition = next(
            transition
            for transition in fill_interval["accounting_transitions"]
            if transition["order_event_sequence"] == fill_event["sequence"]
        )
        original_notional = fill_event["filled_notional"]
        forged_notional = original_notional * 0.9
        notional_delta = forged_notional - original_notional
        fill_event["filled_notional"] = forged_notional
        fill_transition["evidence"]["filled_notional"] = forged_notional
        fill_transition["evidence"]["turnover"] = (
            forged_notional / fill_interval["portfolio_value_before"]
        )
        fill_interval["turnover_total_after"] += (
            notional_delta / fill_interval["portfolio_value_before"]
        )
        capacity = fill_interval["capacity_events"][0]
        capacity["consumed_capacity_notional"] += notional_delta
        capacity["remaining_capacity_notional"] -= notional_delta
        candidate_artifact._validate_v9_interval_accounting(
            [fill_interval],
            symbols=dataset.symbols,
            dataset_id=forged_ledger["dataset_id"],
            execution_policy_digest=forged_ledger["execution_policy_digest"],
        )
        with pytest.raises(ValueError, match="accounting fill event link"):
            candidate_artifact._validate_v10_accounting_transitions(
                payload=forged_ledger,
                intervals=forged_ledger["intervals"],
                symbols=dataset.symbols,
                summary={"evaluation": {"initial_capital": 1_000.0}},
            )
    for interval in ledger["intervals"]:
        interval_transitions = interval["accounting_transitions"]
        assert [event["sequence"] for event in interval_transitions] == list(
            range(len(interval_transitions))
        )


def test_v10_shared_cash_ledger_rejects_unexplained_mark_jump() -> None:
    dataset = market()
    risk = PreTradeRisk(
        PreTradeRiskConfig(
            max_gross=0.5,
            max_abs_weight=0.5,
            max_turnover=None,
            drawdown_start=0.5,
            drawdown_stop=0.9,
        )
    )
    replay = run_shared_cash_replay(
        dataset,
        tuple(ConstantIntentStrategy(PositionIntent.FLAT) for _ in dataset.symbols),
        start_index=4,
        stop_index=7,
        gross_budget=0.5,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
        risk=risk,
        minimum_hold_bars=0,
        settle_terminal_position=False,
        capture_ledger_evidence=True,
        capture_accounting_evidence=True,
    )

    assert replay.ledger_evidence is not None
    ledger = replay.ledger_evidence.to_mapping()
    intervals = ledger["intervals"]
    assert len(intervals) >= 2
    assert not any(
        event["event_type"] in {"filled", "partial_fill"}
        for interval in intervals
        for event in interval["order_events"]
    )

    forged = deepcopy(ledger)
    later_interval = forged["intervals"][1]
    for transition in later_interval["accounting_transitions"]:
        for state_key in ("state_before", "state_after"):
            state = transition[state_key]
            state["mark_prices"] = tuple(value * 0.5 for value in state["mark_prices"])
        evidence = transition["evidence"]
        if "mark_prices" in evidence:
            evidence["mark_prices"] = tuple(
                value * 0.5 for value in evidence["mark_prices"]
            )
        if transition["transition_type"] == "ohlc_drawdown_stress":
            evidence["adverse_prices"] = tuple(
                value * 0.5 for value in evidence["adverse_prices"]
            )
            evidence["favorable_prices"] = tuple(
                value * 0.5 for value in evidence["favorable_prices"]
            )
    for event in later_interval["funding_events"]:
        event["mark_prices"] = tuple(value * 0.5 for value in event["mark_prices"])

    with pytest.raises(ValueError, match="accounting interval state continuity"):
        candidate_artifact._validate_v10_accounting_transitions(
            payload=forged,
            intervals=forged["intervals"],
            symbols=dataset.symbols,
            summary={"evaluation": {"initial_capital": 1_000.0}},
        )


def test_v10_shared_cash_ledger_v3_accepts_fill_without_ohlc_stress() -> None:
    dataset = market()
    risk = PreTradeRisk(
        PreTradeRiskConfig(
            max_gross=0.5,
            max_abs_weight=0.5,
            max_turnover=None,
            drawdown_start=0.5,
            drawdown_stop=0.9,
        )
    )
    replay = run_shared_cash_replay(
        dataset,
        tuple(ConstantIntentStrategy(PositionIntent.LONG) for _ in dataset.symbols),
        start_index=4,
        stop_index=7,
        gross_budget=0.5,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
        risk=risk,
        minimum_hold_bars=0,
        settle_terminal_position=False,
        capture_ledger_evidence=True,
        capture_accounting_evidence=True,
    )

    assert replay.ledger_evidence is not None
    ledger = deepcopy(replay.ledger_evidence.to_mapping())
    ledger["schema_version"] = "shared_cash_replay_ledger_v3"
    intervals = ledger["intervals"]
    fill_count = 0
    for interval in intervals:
        interval["accounting_transitions"] = [
            transition
            for transition in interval["accounting_transitions"]
            if transition["transition_type"] != "ohlc_drawdown_stress"
        ]
        fill_count += sum(
            transition["transition_type"] == "fill"
            for transition in interval["accounting_transitions"]
        )
        for sequence, transition in enumerate(interval["accounting_transitions"]):
            transition["sequence"] = sequence

    assert fill_count > 0
    candidate_artifact._validate_v10_accounting_transitions(
        payload=ledger,
        intervals=intervals,
        symbols=dataset.symbols,
        summary={
            "evaluation": {"initial_capital": 1_000.0},
            "shared_cash_ppo": {"metrics": {"max_drawdown": replay.book.max_drawdown}},
        },
    )


def test_v10_shared_cash_ledger_requires_termination_flatten_evidence() -> None:
    import math
    from fractions import Fraction

    dataset = market()
    funding_due = np.zeros((dataset.n_bars, dataset.n_symbols), dtype=np.bool_)
    funding_due[5, :] = True
    funding_rate = np.zeros((dataset.n_bars, dataset.n_symbols))
    funding_rate[5, :] = 2.0
    dataset = replace(dataset, funding_rate=funding_rate, funding_due=funding_due)

    execution_cost = replace(
        ExecutionCostConfig.zero(),
        maintenance_margin_rate=0.1,
        collateral_haircut=1.0,
    )
    risk = PreTradeRisk(
        PreTradeRiskConfig(
            max_gross=1.0,
            max_abs_weight=1.0,
            max_turnover=None,
            drawdown_start=0.5,
            drawdown_stop=0.9,
        )
    )
    replay = run_shared_cash_replay(
        dataset,
        tuple(ConstantIntentStrategy(PositionIntent.LONG) for _ in dataset.symbols),
        start_index=4,
        stop_index=7,
        gross_budget=1.0,
        initial_capital=1_000.0,
        execution_cost=execution_cost,
        risk=risk,
        minimum_hold_bars=0,
        settle_terminal_position=False,
        capture_ledger_evidence=True,
        capture_accounting_evidence=True,
    )

    assert replay.ledger_evidence is not None
    ledger = deepcopy(replay.ledger_evidence.to_mapping())
    terminal_interval = ledger["intervals"][-1]
    assert terminal_interval["termination_reason"] == "margin_call"
    transitions = terminal_interval["accounting_transitions"]
    flatten_indices = [
        index
        for index, transition in enumerate(transitions)
        if transition["transition_type"] == "termination_flatten"
    ]
    assert flatten_indices
    assert flatten_indices == list(range(flatten_indices[0], len(transitions)))

    terminal_interval["accounting_transitions"] = [
        transition
        for transition in transitions
        if transition["transition_type"] != "termination_flatten"
    ]
    for sequence, transition in enumerate(terminal_interval["accounting_transitions"]):
        transition["sequence"] = sequence

    final_state = terminal_interval["accounting_transitions"][-1]["state_after"]
    final_cash = float(final_state["cash"])
    final_quantities = tuple(
        Fraction(value) for value in final_state["exact_quantities"]
    )
    assert any(quantity != 0 for quantity in final_quantities)
    final_portfolio_value = final_cash + math.fsum(
        float(quantity) * float(mark) * float(multiplier)
        for quantity, mark, multiplier in zip(
            final_quantities,
            final_state["mark_prices"],
            final_state["contract_multipliers"],
            strict=True,
        )
    )
    terminal_interval["cash_after"] = final_cash
    terminal_interval["exact_quantities_after"] = [
        str(quantity) for quantity in final_quantities
    ]
    terminal_interval["portfolio_value_after"] = final_portfolio_value
    portfolio_value_before = terminal_interval["portfolio_value_before"]
    terminal_interval["interval_net_return"] = max(
        final_portfolio_value / portfolio_value_before - 1.0, -1.0 + 1e-12
    )
    ledger["final_cash"] = final_cash
    ledger["final_portfolio_value"] = final_portfolio_value
    ledger["terminal_exact_quantities"] = [
        str(quantity) for quantity in final_quantities
    ]

    with pytest.raises(ValueError, match="termination flatten evidence is missing"):
        candidate_artifact._validate_v10_accounting_transitions(
            payload=ledger,
            intervals=ledger["intervals"],
            symbols=dataset.symbols,
            summary={"evaluation": {"initial_capital": 1_000.0}},
        )


def test_v10_accounting_accepts_exact_lot_fill_with_lossy_float_projection() -> None:
    from fractions import Fraction

    dataset = market()
    dataset = replace(
        dataset,
        lot_size=np.full((dataset.n_bars, dataset.n_symbols), 3e-17),
    )
    risk = PreTradeRisk(
        PreTradeRiskConfig(
            max_gross=0.5,
            max_abs_weight=0.5,
            max_turnover=None,
            drawdown_start=0.5,
            drawdown_stop=0.9,
        )
    )
    replay = run_shared_cash_replay(
        dataset,
        tuple(ConstantIntentStrategy(PositionIntent.LONG) for _ in dataset.symbols),
        start_index=4,
        stop_index=7,
        gross_budget=0.5,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
        risk=risk,
        minimum_hold_bars=0,
        settle_terminal_position=False,
        capture_ledger_evidence=True,
        capture_accounting_evidence=True,
    )

    assert replay.ledger_evidence is not None
    ledger = replay.ledger_evidence.to_mapping()
    fill_transitions = [
        transition
        for interval in ledger["intervals"]
        for transition in interval["accounting_transitions"]
        if transition["transition_type"] == "fill"
    ]
    assert fill_transitions
    assert any(
        Fraction(
            transition["state_after"]["exact_quantities"][
                transition["evidence"]["symbol_index"]
            ]
        )
        - Fraction(
            transition["state_before"]["exact_quantities"][
                transition["evidence"]["symbol_index"]
            ]
        )
        != Fraction(str(float(transition["evidence"]["filled_quantity"])))
        for transition in fill_transitions
    )

    candidate_artifact._validate_v10_accounting_transitions(
        payload=ledger,
        intervals=ledger["intervals"],
        symbols=dataset.symbols,
        summary={
            "evaluation": {"initial_capital": 1_000.0},
            "shared_cash_ppo": {"metrics": {"max_drawdown": replay.book.max_drawdown}},
        },
    )
