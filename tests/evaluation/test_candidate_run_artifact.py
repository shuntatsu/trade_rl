from __future__ import annotations

import json
import shutil
from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

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
        "load_published_market_dataset_artifact",
        lambda path: (
            SimpleNamespace(
                schema_version="market_dataset_artifact_v3",
                artifact_digest="d" * 64,
            ),
            dataset,
        ),
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
    assert summary["schema_version"] == "lean_candidate_result_v10"
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
        "supervised_forecast_horizon_hours": 24,
        "forecast_switch_cost": None,
    }

    for invalid_horizon in (None, True, 0, -1, 24.0, float("nan"), 2_562_048):
        malformed = dict(summary)
        malformed["candidate_config"] = {
            **summary["candidate_config"],
            "supervised_forecast_horizon_hours": invalid_horizon,
        }
        with pytest.raises(ValueError, match="supervised forecast horizon"):
            candidate_artifact._validate_ppo_training_evidence(
                malformed,
                result_schema="lean_candidate_result_v8",
            )

    for invalid_switch_cost in (True, -0.1, float("inf"), float("nan")):
        malformed_switch = dict(summary)
        malformed_switch["candidate_config"] = {
            **summary["candidate_config"],
            "forecast_switch_cost": invalid_switch_cost,
        }
        with pytest.raises(ValueError, match="forecast switch cost"):
            candidate_artifact._validate_ppo_training_evidence(
                malformed_switch,
                result_schema="lean_candidate_result_v10",
            )

    missing_switch = dict(summary)
    missing_switch["candidate_config"] = {
        key: value
        for key, value in summary["candidate_config"].items()
        if key != "forecast_switch_cost"
    }
    with pytest.raises(ValueError, match="switch cost identity"):
        candidate_artifact._validate_ppo_training_evidence(
            missing_switch,
            result_schema="lean_candidate_result_v10",
        )

    missing_horizon = dict(summary)
    missing_horizon["candidate_config"] = {
        key: value
        for key, value in summary["candidate_config"].items()
        if key != "supervised_forecast_horizon_hours"
    }
    with pytest.raises(ValueError, match="supervised forecast horizon"):
        candidate_artifact._validate_ppo_training_evidence(
            missing_horizon,
            result_schema="lean_candidate_result_v8",
        )

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
            result_schema="lean_candidate_result_v8",
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

    # Historical v8 never allowed shared-cash evidence. Downgrade the current
    # v10 summary explicitly before injecting the forbidden field.
    summary["schema_version"] = "lean_candidate_result_v8"
    summary["candidate_config"].pop("forecast_switch_cost", None)
    summary["shared_cash_ppo"] = {
        "return_key": "shared_cash_ppo",
        "metrics": {"total_return": 999.0},
    }
    artifact.summary_path.write_text(
        json.dumps(summary, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    with np.load(artifact.returns_path, allow_pickle=False) as archive:
        returns = {key: archive[key].copy() for key in archive.files}
    returns["shared_cash_ppo"] = np.zeros(3, dtype=np.float64)
    with artifact.returns_path.open("wb") as stream:
        np.savez_compressed(stream, **returns)

    with pytest.raises(ValueError, match="shared-cash.*v8|v8.*shared-cash"):
        candidate_artifact.load_candidate_run_artifact(artifact.root)


def test_run_candidate_artifact_rejects_unknown_config_keys(
    tmp_path,
    monkeypatch,
) -> None:
    from trade_rl.evaluation.runs import candidate as candidate_run

    monkeypatch.setattr(
        candidate_run,
        "load_published_market_dataset_artifact",
        lambda path: (
            SimpleNamespace(
                schema_version="market_dataset_artifact_v3",
                artifact_digest="d" * 64,
            ),
            market(),
        ),
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
        "load_published_market_dataset_artifact",
        lambda path: (
            SimpleNamespace(
                schema_version="market_dataset_artifact_v3",
                artifact_digest="d" * 64,
            ),
            market(),
        ),
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

    def unexpected_dataset_io(path):
        pytest.fail("existing destination must reject before Dataset I/O")

    monkeypatch.setattr(
        candidate_run,
        "load_published_market_dataset_artifact",
        unexpected_dataset_io,
    )
    config_path = tmp_path / "run.json"
    config_path.write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError):
        candidate_run.run_candidate_artifact(
            dataset_root=tmp_path / "dataset",
            config_path=config_path,
            output_root=output,
        )


@pytest.fixture
def shared_cash_candidate_result():
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

    return result


@pytest.fixture
def shared_cash_candidate_artifact(tmp_path, shared_cash_candidate_result):
    return candidate_artifact.publish_candidate_run(
        tmp_path / "shared-cash-result",
        shared_cash_candidate_result,
        build_candidate_run_provenance(),
    )


def test_shared_cash_ppo_artifact_binds_portfolio_returns_and_ledger_digest(
    tmp_path,
    shared_cash_candidate_artifact,
) -> None:
    published = shared_cash_candidate_artifact
    loaded = candidate_artifact.load_candidate_run_artifact(published.root)

    assert loaded.has_verified_full_evaluation_coverage
    assert loaded.summary["schema_version"] == "lean_candidate_result_v10"
    assert loaded.summary["candidate_config"]["supervised_forecast_horizon_hours"] == 24
    portfolio = loaded.summary["shared_cash_ppo"]
    assert portfolio["name"] == "ppo"
    assert portfolio["final_portfolio_value"] == 1_000.0
    assert portfolio["terminal_settlement_complete"] is True
    assert portfolio["ledger_evidence"]["interval_count"] == 3
    assert len(portfolio["ledger_evidence"]["digest"]) == 64
    assert np.array_equal(loaded.returns["shared_cash_ppo"], np.zeros(3))

    legacy_root = tmp_path / "legacy-shared-cash-result"
    shutil.copytree(published.root, legacy_root)
    legacy_summary_path = legacy_root / "summary.json"
    legacy_summary = json.loads(legacy_summary_path.read_text(encoding="utf-8"))
    legacy_summary["schema_version"] = "lean_candidate_result_v7"
    legacy_summary["candidate_config"].pop("supervised_forecast_horizon_hours")
    legacy_summary["candidate_config"].pop("forecast_switch_cost")
    legacy_summary_path.write_text(
        json.dumps(legacy_summary, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    legacy_loaded = candidate_artifact.load_candidate_run_artifact(legacy_root)
    assert legacy_loaded.summary["schema_version"] == "lean_candidate_result_v7"
    assert np.array_equal(legacy_loaded.returns["shared_cash_ppo"], np.zeros(3))


@pytest.mark.parametrize(
    "reader",
    [
        candidate_artifact.load_candidate_run_artifact,
        candidate_artifact.inspect_candidate_run_artifact,
    ],
)
@pytest.mark.parametrize(
    "schema",
    [
        "lean_candidate_result_v7",
        "lean_candidate_result_v9",
        "lean_candidate_result_v10",
    ],
)
@pytest.mark.parametrize("length", [2, 4])
def test_shared_cash_candidate_rejects_mismatched_period_count_before_decode(
    shared_cash_candidate_artifact,
    monkeypatch,
    reader,
    schema,
    length,
) -> None:
    artifact = shared_cash_candidate_artifact
    summary = json.loads(artifact.summary_path.read_text(encoding="utf-8"))
    summary["schema_version"] = schema
    if schema != "lean_candidate_result_v10":
        summary["candidate_config"].pop("forecast_switch_cost")
    if schema == "lean_candidate_result_v7":
        summary["candidate_config"].pop("supervised_forecast_horizon_hours")
    artifact.summary_path.write_text(json.dumps(summary), encoding="utf-8")
    assert reader(artifact.root) is not None
    with np.load(artifact.returns_path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    arrays["shared_cash_ppo"] = np.zeros(length, dtype=np.float64)
    np.savez_compressed(artifact.returns_path, **arrays)
    original = artifact.returns_path.read_bytes()

    def forbidden_decode(*args, **kwargs):
        raise AssertionError(
            "invalid shared cash period count reached array materialization"
        )

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", forbidden_decode)
    with pytest.raises(ValueError, match="return period count"):
        reader(artifact.root)
    assert artifact.returns_path.read_bytes() == original


@pytest.mark.parametrize(
    "schema",
    [
        "lean_candidate_result_v7",
        "lean_candidate_result_v9",
        "lean_candidate_result_v10",
    ],
)
@pytest.mark.parametrize("quantity", ["1e-400", "-1e-400"])
@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize(
    "reader",
    [
        candidate_artifact.load_candidate_run_artifact,
        candidate_artifact.inspect_candidate_run_artifact,
    ],
)
def test_completed_candidate_refuses_underflowed_exact_terminal_quantity(
    shared_cash_candidate_artifact, schema, quantity, index, reader
):
    artifact = shared_cash_candidate_artifact
    summary = json.loads(artifact.summary_path.read_text(encoding="utf-8"))
    summary["schema_version"] = schema
    if schema != "lean_candidate_result_v10":
        summary["candidate_config"].pop("forecast_switch_cost")
    if schema == "lean_candidate_result_v7":
        summary["candidate_config"].pop("supervised_forecast_horizon_hours")
    artifact.summary_path.write_text(json.dumps(summary), encoding="utf-8")
    assert reader(artifact.root) is not None
    assert summary["shared_cash_ppo"]["terminal_settlement_complete"] is True
    assert Fraction(quantity) != 0 and float(quantity) == 0.0
    summary["shared_cash_ppo"]["ledger_evidence"]["terminal_exact_quantities"][
        index
    ] = quantity
    artifact.summary_path.write_text(json.dumps(summary), encoding="utf-8")
    original = artifact.summary_path.read_bytes()
    with pytest.raises(ValueError, match="exact terminal quantities"):
        reader(artifact.root)
    assert artifact.summary_path.read_bytes() == original


@pytest.mark.parametrize("scope", ["per_symbol", "shared_book", "shared_ledger"])
@pytest.mark.parametrize("index", [0, 1])
def test_candidate_writer_does_not_publish_false_terminal_completion(
    tmp_path, shared_cash_candidate_result, scope, index
):
    result = shared_cash_candidate_result
    comparison = result.comparison
    exact = Fraction(1, 10**400)
    assert exact != 0 and float(exact) == 0.0

    def hidden_residue(book):
        # Construct an actual BookState with exact dust that conservatively
        # projects to zero, using the same native setter as book transitions.
        book = book.clone()
        book._set_quantity_values(
            tuple(
                exact if i == index else Fraction(0)
                for i in range(len(book.quantities))
            )
        )
        assert not np.any(book.quantities)
        assert book.exact_quantities[index] == exact
        return book

    if scope == "per_symbol":
        symbol = comparison.by_symbol[index]
        entry = symbol.comparison.entries[0]
        updated = replace(
            entry, replay=replace(entry.replay, book=hidden_residue(entry.replay.book))
        )
        symbol = replace(
            symbol, comparison=replace(symbol.comparison, entries=(updated,))
        )
        comparison = replace(
            comparison,
            by_symbol=tuple(
                symbol if i == index else value
                for i, value in enumerate(comparison.by_symbol)
            ),
        )
    else:
        shared = comparison.shared_cash_ppo
        assert shared is not None
        ledger = shared.replay.ledger_evidence
        assert ledger is not None
        if scope == "shared_book":
            replay = replace(shared.replay, book=hidden_residue(shared.replay.book))
        else:
            replay = replace(
                shared.replay,
                ledger_evidence=replace(
                    ledger,
                    terminal_exact_quantities=tuple(
                        str(exact) if i == index else "0"
                        for i in range(len(ledger.terminal_exact_quantities))
                    ),
                ),
            )
        comparison = replace(comparison, shared_cash_ppo=replace(shared, replay=replay))
    result = replace(result, comparison=comparison)
    destination = tmp_path / "not-settled"
    with pytest.raises(ValueError, match="exact terminal quantities"):
        candidate_artifact.publish_candidate_run(
            destination, result, build_candidate_run_provenance()
        )
    assert not destination.exists()
    assert not tuple(tmp_path.glob(".not-settled.staging-*"))


@pytest.mark.parametrize(
    "quantity", ["0.0", "-0", "0/1", str(Fraction(1, 10**400)), "not a quantity"]
)
def test_completed_shared_cash_ledger_requires_canonical_zero(
    shared_cash_candidate_artifact, quantity
):
    artifact = shared_cash_candidate_artifact
    summary = json.loads(artifact.summary_path.read_text(encoding="utf-8"))
    assert summary["shared_cash_ppo"]["ledger_evidence"][
        "terminal_exact_quantities"
    ] == ["0", "0"]
    summary["shared_cash_ppo"]["ledger_evidence"]["terminal_exact_quantities"][1] = (
        quantity
    )
    artifact.summary_path.write_text(json.dumps(summary), encoding="utf-8")
    for reader in (
        candidate_artifact.load_candidate_run_artifact,
        candidate_artifact.inspect_candidate_run_artifact,
    ):
        with pytest.raises(ValueError, match="exact terminal quantities"):
            reader(artifact.root)


def test_noncompleted_candidate_preserves_exact_residue_evidence(
    tmp_path, shared_cash_candidate_result
):
    result = shared_cash_candidate_result
    shared = result.comparison.shared_cash_ppo
    assert shared is not None
    ledger = shared.replay.ledger_evidence
    assert ledger is not None
    exact = Fraction(1, 10**400)
    book = shared.replay.book.clone()
    book._set_quantity_values((Fraction(0), exact))
    assert book.exact_quantities == (Fraction(0), exact)
    assert not np.any(book.quantities)
    replay = replace(
        shared.replay,
        book=book,
        ledger_evidence=replace(
            ledger,
            terminal_exact_quantities=("0", str(exact)),
            active_order_remainders=(("pending-synthetic", 0.1),),
        ),
    )
    comparison = replace(
        result.comparison, shared_cash_ppo=replace(shared, replay=replay)
    )
    result = replace(result, comparison=comparison)
    artifact = candidate_artifact.publish_candidate_run(
        tmp_path / "not-completed", result, build_candidate_run_provenance()
    )
    loaded = candidate_artifact.load_candidate_run_artifact(artifact.root)
    assert loaded.summary["shared_cash_ppo"]["terminal_settlement_complete"] is False
    assert loaded.summary["shared_cash_ppo"]["ledger_evidence"][
        "terminal_exact_quantities"
    ] == ("0", str(exact))


@pytest.mark.parametrize(
    "values",
    [
        pytest.param(np.zeros((3, 1)), id="two-dimensional"),
        pytest.param(np.array([0.0, np.nan, 0.0]), id="nan"),
        pytest.param(np.array([0.0, np.inf, 0.0]), id="inf"),
        pytest.param(np.array([0j, 1j, 0j]), id="complex"),
        pytest.param(np.zeros(3, dtype=np.bool_), id="bool"),
        pytest.param(np.array(["0"] * 3), id="text"),
    ],
)
def test_shared_cash_coverage_refuses_invalid_direct_return_arrays(
    shared_cash_candidate_artifact, values: np.ndarray
) -> None:
    from trade_rl.evaluation.runs.artifact import (
        LoadedCandidateRun,
        load_candidate_run_artifact,
    )

    loaded = load_candidate_run_artifact(shared_cash_candidate_artifact.root)
    assert loaded.has_verified_full_evaluation_coverage
    direct = LoadedCandidateRun(
        root=loaded.root,
        summary=loaded.summary,
        returns={**loaded.returns, "shared_cash_ppo": values},
        provenance=loaded.provenance,
    )
    assert not direct.has_verified_full_evaluation_coverage


@pytest.mark.parametrize("version", [7, 9, 10])
@pytest.mark.parametrize("mutation", ["missing", "null", "colliding_key"])
def test_shared_cash_candidate_roster_rejects_invalid_evidence_before_decode(
    shared_cash_candidate_artifact, monkeypatch, version, mutation
) -> None:
    artifact = shared_cash_candidate_artifact
    summary = json.loads(artifact.summary_path.read_text(encoding="utf-8"))
    summary["schema_version"] = f"lean_candidate_result_v{version}"
    if version < 10:
        summary["candidate_config"].pop("forecast_switch_cost")
    if version < 9:
        summary["candidate_config"].pop("supervised_forecast_horizon_hours")
    if version == 10 and mutation == "missing":
        # v10 permits per-symbol-only evidence, but the extra NPZ member must
        # still be refused by the reader.
        summary.pop("shared_cash_ppo")
        match = "return keys"
    elif mutation == "missing":
        summary.pop("shared_cash_ppo")
        match = "candidate coverage"
    elif mutation == "null":
        summary["shared_cash_ppo"] = None
        match = "candidate coverage"
    else:
        summary["shared_cash_ppo"]["return_key"] = summary["by_symbol"][0][
            "strategies"
        ][0]["return_key"]
        match = "candidate coverage"
    artifact.summary_path.write_text(json.dumps(summary), encoding="utf-8")

    def forbidden_decode(*args, **kwargs):
        raise AssertionError("malformed roster reached return array decoding")

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", forbidden_decode)
    for reader in (
        candidate_artifact.load_candidate_run_artifact,
        candidate_artifact.inspect_candidate_run_artifact,
    ):
        with pytest.raises(ValueError, match=match):
            reader(artifact.root)
