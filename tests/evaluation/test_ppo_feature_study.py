from __future__ import annotations

import gzip
import hashlib
import json
import weakref
from copy import deepcopy
from math import nan, prod
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import trade_rl.evaluation.ppo_feature_study as feature_study
from tests.evaluation.test_directional_market_profile import _data, _factory
from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.evaluation.directional import evaluate_directional_arm
from trade_rl.evaluation.directional_study import (
    SOURCE_ARTIFACT_DIGEST,
    SOURCE_DATASET_ID,
    SOURCE_STUDY_DIGEST,
)
from trade_rl.evaluation.experiments import ResolvedRunConfig
from trade_rl.evaluation.ppo_feature_study import (
    BASELINE_FACTOR,
    CANDIDATE_FACTOR,
    _interval_year_slices,
    _read_arm,
    _validate_ledger_payload,
    _validate_replay,
    compare_feature_ablation,
    expected_protocol,
    interval_start_year_returns,
    run_arm,
)
from trade_rl.simulation.quantities import exact_quantity

_SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
_RISK = {
    "max_gross": 0.5,
    "max_abs_weight": 0.1,
    "max_turnover": None,
    "drawdown_start": 0.1,
    "drawdown_stop": 0.2,
    "emergency_turnover_override": True,
    "fail_closed_tolerance": 1e-10,
}
_SCENARIOS = {
    "base": {
        "cost_multiplier": 1.0,
        "latency_bars": 0,
        "execution_policy_digest": "base-policy",
    },
    "cost_2x": {
        "cost_multiplier": 2.0,
        "latency_bars": 0,
        "execution_policy_digest": "cost-policy",
    },
    "latency_1": {
        "cost_multiplier": 1.0,
        "latency_bars": 1,
        "execution_policy_digest": "latency-policy",
    },
}
_PROTOCOL = {
    "evaluation_dataset_id": "transformed-dataset",
    "start_index": 10,
    "stop_index": 14,
    "interval_count": 4,
    "seeds": [0, 1, 2, 3, 4],
    "symbols": list(_SYMBOLS),
    "factors": {
        BASELINE_FACTOR: {"feature_names": ["x"], "feature_indices": [0]},
        CANDIDATE_FACTOR: {
            "feature_names": ["x", "relative"],
            "feature_indices": [0, 1],
        },
    },
    "training": {
        "requested_timesteps": 262_144,
        "initial_capital": 10_000.0,
        "gross_budget": 0.1,
        "risk": _RISK,
    },
    "provenance": {},
    "admission": {
        "required_passing_seeds_per_symbol": 4,
        "required_symbols": 4,
        "required_positive_paired_seed_deltas": 4,
    },
    "evaluation": {
        "start_index": 10,
        "stop_index": 14,
        "interval_count": 4,
        "interval_year_slices": {"2023": [0, 2], "2024": [2, 4]},
        "ledger_drawdown_limit": 0.20,
        "risk": _RISK,
        "initial_capital": 10_000.0,
        "gross_budget": 0.1,
        "scenarios": _SCENARIOS,
        "stress_names": ["cost_2x", "latency_1"],
    },
}


def _endpoint_drawdown(returns: tuple[float, ...]) -> float:
    wealth = 1.0
    peak = 1.0
    maximum = 0.0
    for value in returns:
        wealth *= 1.0 + value
        peak = max(peak, wealth)
        maximum = max(maximum, 1.0 - wealth / peak)
    return maximum


def _trace_summary(
    symbol_index: int,
    scenario_name: str,
    drawdown: float,
    *,
    terminal_values: list[float] | None = None,
    remainders: list[tuple[str, float]] | None = None,
) -> dict:
    scenario = _SCENARIOS[scenario_name]
    values = terminal_values or [0.0] * len(_SYMBOLS)
    return {
        "schema_version": "shared_cash_replay_ledger_v1",
        "dataset_id": _PROTOCOL["evaluation_dataset_id"],
        "start_index": _PROTOCOL["start_index"],
        "stop_index": _PROTOCOL["stop_index"],
        "execution_policy_digest": scenario["execution_policy_digest"],
        "interval_count": _PROTOCOL["interval_count"],
        "final_max_drawdown": drawdown,
        "terminal_exact_quantities": [str(exact_quantity(value)) for value in values],
        "termination_reason": None,
        "active_order_remainders": remainders or [],
    }


def _replay(
    symbol_index: int,
    returns: tuple[float, ...],
    *,
    scenario_name: str = "base",
    ledger_drawdown: float = 0.05,
    terminated: bool = False,
    terminal_flat: bool = True,
) -> dict:
    scenario = _SCENARIOS[scenario_name]
    quantities = [0.0] * len(_SYMBOLS)
    if not terminal_flat:
        quantities[symbol_index] = 0.01
    return {
        "schema": "directional_arm_v1",
        "scenario": scenario_name,
        "dataset_id": _PROTOCOL["evaluation_dataset_id"],
        "start_index": _PROTOCOL["start_index"],
        "stop_index": _PROTOCOL["stop_index"],
        "symbol_index": symbol_index,
        "initial_capital": _PROTOCOL["training"]["initial_capital"],
        "gross_budget": _PROTOCOL["training"]["gross_budget"],
        "latency_bars": scenario["latency_bars"],
        "cost_multiplier": scenario["cost_multiplier"],
        "execution_policy_digest": scenario["execution_policy_digest"],
        "risk": _RISK,
        "returns": list(returns),
        "metrics": {
            "total_return": prod(1.0 + value for value in returns) - 1.0,
            "max_drawdown": _endpoint_drawdown(returns),
        },
        "ledger_max_drawdown": ledger_drawdown,
        "termination_reasons": ["test termination"] if terminated else [],
        "terminal_flat": terminal_flat,
        "terminal_quantities": quantities,
        "ledger_trace_summary": _trace_summary(
            symbol_index,
            scenario_name,
            ledger_drawdown,
            terminal_values=quantities,
        ),
    }


def _arm(factor: str, seed: int, return_value: float) -> dict:
    full = (return_value, return_value, return_value, return_value)
    stresses = {
        name: {
            symbol: _replay(index, full, scenario_name=name)
            for index, symbol in enumerate(_SYMBOLS)
        }
        for name in _PROTOCOL["evaluation"]["stress_names"]
    }
    return {
        "schema": "ppo_feature_arm_v1",
        "factor": factor,
        "seed": seed,
        "dataset_id": _PROTOCOL["evaluation_dataset_id"],
        "start_index": _PROTOCOL["start_index"],
        "stop_index": _PROTOCOL["stop_index"],
        "requested_timesteps": _PROTOCOL["training"]["requested_timesteps"],
        "feature_names": _PROTOCOL["factors"][factor]["feature_names"],
        "feature_indices": _PROTOCOL["factors"][factor]["feature_indices"],
        "training_risk": _RISK,
        "by_symbol": {
            symbol: _replay(index, full) for index, symbol in enumerate(_SYMBOLS)
        },
        "stress_by_name": stresses if factor == CANDIDATE_FACTOR else {},
    }


def _roster(factor: str, value: float) -> dict[int, dict]:
    return {seed: _arm(factor, seed, value) for seed in range(5)}


def _ledger_payload(row: dict) -> dict:
    portfolio_value = 10_000.0
    max_drawdown = 0.0
    intervals = []
    for offset, value in enumerate(row["returns"]):
        before = portfolio_value
        portfolio_value *= 1.0 + value
        after_drawdown = max(max_drawdown, row["ledger_max_drawdown"])
        intervals.append(
            {
                "start_index": _PROTOCOL["start_index"] + offset,
                "next_index": _PROTOCOL["start_index"] + offset + 1,
                "exact_quantities_before": ["0"] * len(_SYMBOLS),
                "exact_quantities_after": ["0"] * len(_SYMBOLS),
                "cash_before": before,
                "cash_after": portfolio_value,
                "portfolio_value_before": before,
                "portfolio_value_after": portfolio_value,
                "total_cost_before": 0.0,
                "total_cost_after": 0.0,
                "funding_pnl_before": 0.0,
                "funding_pnl_after": 0.0,
                "borrow_cost_before": 0.0,
                "borrow_cost_after": 0.0,
                "turnover_total_before": 0.0,
                "turnover_total_after": 0.0,
                "max_drawdown_before": max_drawdown,
                "max_drawdown_after": after_drawdown,
                "interval_cost": 0.0,
                "interval_funding": 0.0,
                "interval_borrow_cost": 0.0,
                "interval_dividend": 0.0,
                "interval_cash_interest": 0.0,
                "interval_net_return": value,
                "termination_reason": None,
                "order_events": [],
                "capacity_events": [],
                "funding_events": [],
            }
        )
        max_drawdown = after_drawdown
    return {
        "schema_version": "shared_cash_replay_ledger_v1",
        "dataset_id": _PROTOCOL["evaluation_dataset_id"],
        "execution_policy_digest": row["execution_policy_digest"],
        "start_index": _PROTOCOL["start_index"],
        "stop_index": _PROTOCOL["stop_index"],
        "intervals": intervals,
        "terminal_exact_quantities": ["0"] * len(_SYMBOLS),
        "final_cash": portfolio_value,
        "final_portfolio_value": portfolio_value,
        "final_total_cost": 0.0,
        "final_funding_pnl": 0.0,
        "final_borrow_cost": 0.0,
        "final_turnover_total": 0.0,
        "final_max_drawdown": row["ledger_max_drawdown"],
        "termination_reason": None,
        "active_order_remainders": [],
        "terminal_order_reasons": [],
    }


def _publish_arm(root: Path, arm: dict, factor: str, seed: int) -> Path:
    directory = root / factor / f"ppo{seed}"
    directory.mkdir(parents=True)
    model = b"synthetic test model"
    (directory / "model.zip").write_bytes(model)
    artifact = deepcopy(arm)
    artifact["protocol_digest"] = content_digest(_PROTOCOL)
    artifact["provenance"] = _PROTOCOL["provenance"]
    artifact["model_sha256"] = {"model.zip": hashlib.sha256(model).hexdigest()}
    scenario_rosters = [("base", artifact["by_symbol"])]
    scenario_rosters.extend(artifact["stress_by_name"].items())
    for scenario_name, roster in scenario_rosters:
        for symbol_index, row in enumerate(roster.values()):
            raw_evidence = canonical_json_bytes(_ledger_payload(row))
            compressed = gzip.compress(raw_evidence, mtime=0)
            filename = f"ledger-{scenario_name}-symbol-{symbol_index}.json.gz"
            (directory / filename).write_bytes(compressed)
            row.pop("ledger_trace_summary", None)
            row["ledger_evidence_file"] = filename
            row["ledger_evidence_raw_sha256"] = hashlib.sha256(raw_evidence).hexdigest()
            row["ledger_evidence_gzip_sha256"] = hashlib.sha256(compressed).hexdigest()
    raw_result = canonical_json_bytes(artifact)
    (directory / "result.json").write_bytes(raw_result)
    (directory / "result.sha256.json").write_bytes(
        canonical_json_bytes({"sha256": hashlib.sha256(raw_result).hexdigest()})
    )
    return directory


def test_interval_start_year_returns_keep_terminal_boundary_in_prior_interval() -> None:
    timestamps = np.array(
        ["2022-12-31T23", "2023-01-01T00", "2024-12-31T23", "2025-01-01T00"],
        dtype="datetime64[h]",
    )
    slices = _interval_year_slices(timestamps, 1, 3)
    returns = (0.05, 0.06)

    assert slices == {"2023": [0, 1], "2024": [1, 2]}
    assert interval_start_year_returns(returns, slices) == {
        "2023": pytest.approx(0.05),
        "2024": pytest.approx(0.06),
    }
    with pytest.raises(ValueError, match="cover every interval exactly once"):
        interval_start_year_returns(returns, {"2023": (0, 1)})


def test_generated_protocol_includes_capital_consumed_by_replay_validators(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    (source / "study").mkdir(parents=True)
    (source / "study" / "plan.json").write_bytes(b"frozen plan")
    feature_names = (
        "base_feature",
        "1h__relative_return_to_btc_1bar",
        "4h__relative_return_to_btc_1bar",
        "1d__relative_return_to_btc_1bar",
    )
    dataset = SimpleNamespace(
        dataset_id=SOURCE_DATASET_ID,
        feature_names=feature_names,
        symbols=_SYMBOLS,
        n_symbols=len(_SYMBOLS),
        timestamps=np.arange(
            np.datetime64("2023-01-01T00"),
            np.datetime64("2025-01-01T01"),
            np.timedelta64(1, "h"),
        ),
    )
    baseline = SimpleNamespace(
        feature_names=("base_feature",),
        feature_indices=(0,),
        fit_symbol_indices=tuple(range(len(_SYMBOLS))),
        fit_symbol_names=_SYMBOLS,
        fit_cutoff="2022-12-31T00",
        to_payload=lambda: {"feature_names": ["base_feature"]},
    )
    plan = SimpleNamespace(
        digest=SOURCE_STUDY_DIGEST,
        dataset_id=SOURCE_DATASET_ID,
        baseline_config=baseline,
    )

    monkeypatch.setattr(
        feature_study,
        "inspect_published_market_dataset_artifact",
        lambda _path: SimpleNamespace(artifact_digest=SOURCE_ARTIFACT_DIGEST),
    )
    monkeypatch.setattr(
        feature_study, "load_market_dataset_artifact", lambda _p: dataset
    )
    monkeypatch.setattr(
        feature_study, "inspect_study", lambda _path: SimpleNamespace(plan=plan)
    )
    monkeypatch.setattr(feature_study, "with_price_channels", lambda value: value)
    monkeypatch.setattr(
        feature_study, "development_indices", lambda _dataset: (0, 17_544)
    )
    monkeypatch.setattr(
        feature_study,
        "build_candidate_run_provenance",
        lambda: {"implementation": {"files": []}},
    )
    monkeypatch.setattr(feature_study, "_source_snapshot_bytes", lambda _p: b"snapshot")

    class _Executor:
        def __init__(self, _dataset: object, execution: object) -> None:
            self.execution_policy_digest = (
                f"{execution.multiplier}-{execution.order_latency_bars}"
            )

    monkeypatch.setattr(feature_study, "MarketExecutor", _Executor)

    protocol = expected_protocol(source)

    assert protocol["training"]["initial_capital"] == 10_000.0
    assert protocol["evaluation"]["initial_capital"] == 10_000.0

    protocol = deepcopy(protocol)
    evaluation = protocol["evaluation"]
    evaluation.update(
        start_index=10,
        stop_index=14,
        interval_count=4,
        interval_year_slices={"2023": [0, 4]},
    )
    scenario = evaluation["scenarios"]["base"]
    row = _replay(0, (0.0, 0.0, 0.0, 0.0))
    row.update(
        dataset_id=protocol["evaluation_dataset_id"],
        start_index=evaluation["start_index"],
        stop_index=evaluation["stop_index"],
        initial_capital=evaluation["initial_capital"],
        gross_budget=evaluation["gross_budget"],
        latency_bars=scenario["latency_bars"],
        cost_multiplier=scenario["cost_multiplier"],
        execution_policy_digest=scenario["execution_policy_digest"],
        risk=evaluation["risk"],
    )
    row["ledger_trace_summary"].update(
        dataset_id=protocol["evaluation_dataset_id"],
        start_index=evaluation["start_index"],
        stop_index=evaluation["stop_index"],
        execution_policy_digest=scenario["execution_policy_digest"],
    )
    ledger = _ledger_payload(row)
    ledger.update(
        dataset_id=protocol["evaluation_dataset_id"],
        execution_policy_digest=scenario["execution_policy_digest"],
    )
    summary = _validate_ledger_payload(
        ledger,
        row=row,
        protocol=protocol,
        scenario_name="base",
        symbol="BTCUSDT",
    )
    row["ledger_trace_summary"] = summary
    validated = _validate_replay(
        row,
        protocol=protocol,
        scenario_name="base",
        symbol="BTCUSDT",
        symbol_index=0,
    )

    assert validated["returns"] == (0.0, 0.0, 0.0, 0.0)


def test_run_arm_forwards_protocol_capital_and_budget_to_fit_and_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = deepcopy(_PROTOCOL)
    protocol["training"]["initial_capital"] = 12_000.0
    protocol["training"]["gross_budget"] = 0.12
    protocol["evaluation"]["initial_capital"] = 9_000.0
    protocol["evaluation"]["gross_budget"] = 0.09
    augmented_dataset = _data()
    source_dataset_ref: weakref.ReferenceType | None = None
    config = ResolvedRunConfig(
        signal_name="signal",
        signal_index=0,
        feature_names=("x",),
        feature_indices=(0,),
        fit_symbol_names=("BTCUSDT",),
        fit_symbol_indices=(0,),
        fit_cutoff="2022-12-31T00:00:00.000000000",
        rule_entry_threshold=0.1,
        rule_exit_threshold=0.02,
        forecast_entry_threshold=0.01,
        forecast_exit_threshold=0.002,
        ppo_total_timesteps=262_144,
        ppo_seed=0,
        evaluation_start="2023-01-01T00:00:00.000000000",
        evaluation_stop_exclusive="2025-01-01T00:00:00.000000000",
        gross_budget=0.5,
        initial_capital=100_000.0,
        execution_overlay="zero_overlay_dataset_fields_authoritative",
    )
    captured: dict[str, object] = {"replays": []}

    class WeakReferenceableSourceDataset:
        pass

    def fake_load(_path: Path) -> object:
        nonlocal source_dataset_ref
        # MarketDataset is slotted; this token observes the orchestration lifetime.
        source_dataset = WeakReferenceableSourceDataset()
        source_dataset_ref = weakref.ref(source_dataset)
        return source_dataset

    def fake_fit(
        _arm: str,
        _dataset: object,
        _config: object,
        output: Path,
        *,
        ppo_risk_config: object,
        initial_capital: float,
        gross_budget: float,
    ) -> object:
        assert source_dataset_ref is not None
        assert source_dataset_ref() is None
        captured["fit"] = (initial_capital, gross_budget, ppo_risk_config)
        (output / "model.zip").write_bytes(b"test model")
        return _factory

    def fake_evaluate(
        _dataset: object,
        _factory_fn: object,
        **kwargs: object,
    ) -> dict[str, object]:
        captured["replays"].append((kwargs["initial_capital"], kwargs["gross_budget"]))
        return {"ledger_evidence": {"schema_version": "synthetic-test"}}

    monkeypatch.setattr(feature_study, "validate_protocol", lambda _s, _r: protocol)
    monkeypatch.setattr(feature_study, "load_market_dataset_artifact", fake_load)
    monkeypatch.setattr(
        feature_study, "with_price_channels", lambda _value: augmented_dataset
    )
    monkeypatch.setattr(
        feature_study,
        "inspect_study",
        lambda _p: SimpleNamespace(plan=SimpleNamespace(baseline_config=config)),
    )
    monkeypatch.setattr(feature_study, "build_candidate_run_provenance", lambda: {})
    monkeypatch.setattr(feature_study, "fit_directional_candidate", fake_fit)
    monkeypatch.setattr(feature_study, "evaluate_directional_arm", fake_evaluate)

    run_arm(tmp_path / "source", tmp_path / "study", BASELINE_FACTOR, 0)

    assert captured["fit"] == (12_000.0, 0.12, feature_study.TRAINING_RISK)
    assert captured["replays"] == [(9_000.0, 0.09)]


def test_positive_base_and_stress_votes_must_come_from_same_seeds() -> None:
    baseline = _roster(BASELINE_FACTOR, 0.01)
    candidate = _roster(CANDIDATE_FACTOR, 0.02)
    base_passing_seeds = {0, 1, 2, 3}
    stress_passing_seeds = {1, 2, 3, 4}

    def set_return(row: dict, value: float) -> None:
        returns = [value] * _PROTOCOL["evaluation"]["interval_count"]
        row["returns"] = returns
        row["metrics"]["total_return"] = prod(1.0 + item for item in returns) - 1.0
        row["metrics"]["max_drawdown"] = _endpoint_drawdown(tuple(returns))
        row["ledger_max_drawdown"] = max(0.05, row["metrics"]["max_drawdown"])
        row["ledger_trace_summary"]["final_max_drawdown"] = row["ledger_max_drawdown"]

    for seed in range(len(_PROTOCOL["seeds"])):
        base_value = 0.02 if seed in base_passing_seeds else -0.001
        for symbol in _SYMBOLS:
            set_return(candidate[seed]["by_symbol"][symbol], base_value)
            for stress_name in _PROTOCOL["evaluation"]["stress_names"]:
                stress_value = 0.02 if seed in stress_passing_seeds else -0.001
                set_return(
                    candidate[seed]["stress_by_name"][stress_name][symbol],
                    stress_value,
                )

    report = compare_feature_ablation(baseline, candidate, _PROTOCOL)

    assert report["decision"] == "KEEP_BASELINE"
    assert report["stress_symbols"] == []
    assert report["accepted_symbols"] == []
    assert report["symbol_results"]["BTCUSDT"]["stress_seed_pass_both"] == [1, 2, 3, 4]
    assert report["symbol_results"]["BTCUSDT"][
        "base_and_both_stress_passing_seeds"
    ] == [1, 2, 3]


def test_exactly_four_same_seeds_can_pass_base_and_both_stresses() -> None:
    baseline = _roster(BASELINE_FACTOR, 0.01)
    candidate = _roster(CANDIDATE_FACTOR, 0.02)
    for symbol in _SYMBOLS:
        set_of_rows = [candidate[4]["by_symbol"][symbol]]
        set_of_rows.extend(
            candidate[4]["stress_by_name"][name][symbol]
            for name in _PROTOCOL["evaluation"]["stress_names"]
        )
        for row in set_of_rows:
            row["returns"] = [-0.001] * _PROTOCOL["evaluation"]["interval_count"]
            row["metrics"]["total_return"] = (
                prod(1.0 + value for value in row["returns"]) - 1.0
            )
            row["metrics"]["max_drawdown"] = _endpoint_drawdown(tuple(row["returns"]))
            row["ledger_max_drawdown"] = max(0.05, row["metrics"]["max_drawdown"])
            row["ledger_trace_summary"]["final_max_drawdown"] = row[
                "ledger_max_drawdown"
            ]

    report = compare_feature_ablation(baseline, candidate, _PROTOCOL)

    assert report["decision"] == "PROSPECTIVE_PAPER_REQUIRED"
    for result in report["symbol_results"].values():
        assert result["base_and_both_stress_passing_seeds"] == [0, 1, 2, 3]
        assert result["stress_pass"] is True


def test_feature_candidate_requires_profit_uplift_and_both_stress_gates() -> None:
    report = compare_feature_ablation(
        _roster(BASELINE_FACTOR, 0.01), _roster(CANDIDATE_FACTOR, 0.02), _PROTOCOL
    )

    assert report["decision"] == "PROSPECTIVE_PAPER_REQUIRED"
    assert report["absolute_symbols"] == list(_SYMBOLS)
    assert report["relative_symbols"] == list(_SYMBOLS)
    assert report["stress_symbols"] == list(_SYMBOLS)
    assert report["production_eligible"] is False


@pytest.mark.parametrize(
    ("scenario_name", "field", "wrong_value"),
    [
        ("cost_2x", "cost_multiplier", 1.0),
        ("latency_1", "latency_bars", True),
        ("base", "cost_multiplier", True),
        ("base", "execution_policy_digest", "wrong-policy"),
        ("base", "initial_capital", 9_999.0),
        ("base", "gross_budget", 0.2),
        ("base", "risk", {**_RISK, "max_gross": 0.6}),
        ("base", "risk", {**_RISK, "emergency_turnover_override": 1}),
        ("base", "schema", "other_replay_v1"),
    ],
)
def test_replay_scenario_metadata_is_bound_to_each_cell(
    scenario_name: str, field: str, wrong_value: object
) -> None:
    baseline = _roster(BASELINE_FACTOR, 0.01)
    candidate = _roster(CANDIDATE_FACTOR, 0.02)
    if scenario_name == "base":
        row = candidate[0]["by_symbol"]["BTCUSDT"]
    else:
        row = candidate[0]["stress_by_name"][scenario_name]["BTCUSDT"]
    row[field] = wrong_value

    with pytest.raises(ValueError, match="scenario contract"):
        compare_feature_ablation(baseline, candidate, _PROTOCOL)


def test_symbol_index_requires_integer_identity_not_boolean_alias() -> None:
    baseline = _roster(BASELINE_FACTOR, 0.01)
    candidate = _roster(CANDIDATE_FACTOR, 0.02)
    candidate[0]["by_symbol"]["BTCUSDT"]["symbol_index"] = False

    with pytest.raises(ValueError, match="identity/window"):
        compare_feature_ablation(baseline, candidate, _PROTOCOL)


@pytest.mark.parametrize("scenario", ["base", "cost_2x", "latency_1"])
@pytest.mark.parametrize(
    "violation", ["drawdown", "termination", "nonflat", "remainder"]
)
def test_one_candidate_hard_guard_violation_globally_blocks_admission(
    scenario: str, violation: str
) -> None:
    baseline = _roster(BASELINE_FACTOR, 0.01)
    candidate = _roster(CANDIDATE_FACTOR, 0.02)
    row = (
        candidate[0]["by_symbol"]["BTCUSDT"]
        if scenario == "base"
        else candidate[0]["stress_by_name"][scenario]["BTCUSDT"]
    )
    if violation == "drawdown":
        row["ledger_max_drawdown"] = 0.21
        row["ledger_trace_summary"]["final_max_drawdown"] = 0.21
    elif violation == "termination":
        row["termination_reasons"] = ["risk_stop"]
        row["ledger_trace_summary"]["termination_reason"] = "risk_stop"
    elif violation == "nonflat":
        row["terminal_flat"] = False
        row["terminal_quantities"][0] = 0.01
        row["ledger_trace_summary"]["terminal_exact_quantities"][0] = "1/100"
    else:
        row["ledger_trace_summary"]["active_order_remainders"] = [["order-1", 0.01]]

    report = compare_feature_ablation(baseline, candidate, _PROTOCOL)

    assert report["decision"] == "HARD_GUARD_FAILURE"
    assert report["hard_guards_pass"] is False
    assert report["accepted_symbols"] == []


@pytest.mark.parametrize("bad_return", [-1.0, -3.0])
def test_simple_return_must_be_greater_than_total_loss(bad_return: float) -> None:
    baseline = _roster(BASELINE_FACTOR, 0.01)
    candidate = _roster(CANDIDATE_FACTOR, 0.02)
    row = candidate[0]["by_symbol"]["BTCUSDT"]
    row["returns"][0] = bad_return
    row["metrics"]["total_return"] = prod(1.0 + value for value in row["returns"]) - 1.0
    row["metrics"]["max_drawdown"] = 1.0

    with pytest.raises(ValueError, match="greater than -1"):
        compare_feature_ablation(baseline, candidate, _PROTOCOL)


def test_metric_drawdown_must_match_returns_and_ledger_drawdown_cannot_understate_it() -> (
    None
):
    baseline = _roster(BASELINE_FACTOR, 0.01)
    candidate = _roster(CANDIDATE_FACTOR, 0.02)
    row = candidate[0]["by_symbol"]["BTCUSDT"]
    row["returns"] = [0.2, -0.3, 0.01, 0.01]
    row["metrics"]["total_return"] = prod(1.0 + value for value in row["returns"]) - 1.0
    row["metrics"]["max_drawdown"] = 0.01

    with pytest.raises(ValueError, match="metric drawdown disagrees"):
        compare_feature_ablation(baseline, candidate, _PROTOCOL)

    row["metrics"]["max_drawdown"] = _endpoint_drawdown(tuple(row["returns"]))
    row["ledger_max_drawdown"] = 0.05
    row["ledger_trace_summary"]["final_max_drawdown"] = 0.05
    with pytest.raises(
        ValueError, match="ledger drawdown understates endpoint drawdown"
    ):
        compare_feature_ablation(baseline, candidate, _PROTOCOL)


def test_terminal_quantity_vector_must_cover_the_exact_symbol_roster() -> None:
    baseline = _roster(BASELINE_FACTOR, 0.01)
    candidate = _roster(CANDIDATE_FACTOR, 0.02)
    candidate[0]["by_symbol"]["BTCUSDT"]["terminal_quantities"] = [0.0]

    with pytest.raises(ValueError, match="exact symbol roster"):
        compare_feature_ablation(baseline, candidate, _PROTOCOL)


def test_four_positive_seed_deltas_are_required_and_zero_is_not_a_win() -> None:
    baseline = _roster(BASELINE_FACTOR, 0.01)
    candidate = _roster(CANDIDATE_FACTOR, 0.02)
    for seed in (0, 1):
        candidate[seed] = _arm(CANDIDATE_FACTOR, seed, 0.01)

    report = compare_feature_ablation(baseline, candidate, _PROTOCOL)

    assert report["relative_symbols"] == []
    assert report["decision"] == "RELATIVE_IMPROVEMENT_NOT_ESTABLISHED"


def test_protocol_year_slices_must_match_complete_interval_roster() -> None:
    baseline = _roster(BASELINE_FACTOR, 0.01)
    candidate = _roster(CANDIDATE_FACTOR, 0.02)
    protocol = deepcopy(_PROTOCOL)
    protocol["evaluation"]["interval_year_slices"] = {
        "2023": [0, 1],
        "2024": [2, 4],
    }

    with pytest.raises(ValueError, match="cover every interval exactly once"):
        compare_feature_ablation(baseline, candidate, protocol)


def test_read_arm_loads_hash_bound_ledger_traces(tmp_path: Path) -> None:
    directory = _publish_arm(
        tmp_path,
        _arm(CANDIDATE_FACTOR, 0, 0.02),
        CANDIDATE_FACTOR,
        0,
    )

    loaded = _read_arm(tmp_path, CANDIDATE_FACTOR, 0, _PROTOCOL)

    assert loaded["by_symbol"]["BTCUSDT"]["ledger_trace_summary"]["interval_count"] == 4
    assert (
        loaded["stress_by_name"]["cost_2x"]["BTCUSDT"]["ledger_trace_summary"][
            "execution_policy_digest"
        ]
        == "cost-policy"
    )
    assert directory.is_dir()


def test_read_arm_rejects_rehashed_scenario_metadata_tampering(tmp_path: Path) -> None:
    directory = _publish_arm(
        tmp_path,
        _arm(CANDIDATE_FACTOR, 0, 0.02),
        CANDIDATE_FACTOR,
        0,
    )
    result_path = directory / "result.json"
    result = json.loads(result_path.read_bytes())
    result["stress_by_name"]["cost_2x"]["BTCUSDT"]["cost_multiplier"] = 1.0
    raw = canonical_json_bytes(result)
    result_path.write_bytes(raw)
    (directory / "result.sha256.json").write_bytes(
        canonical_json_bytes({"sha256": hashlib.sha256(raw).hexdigest()})
    )

    with pytest.raises(ValueError, match="scenario contract"):
        _read_arm(tmp_path, CANDIDATE_FACTOR, 0, _PROTOCOL)


def test_read_arm_rejects_ledger_compressed_payload_tampering(tmp_path: Path) -> None:
    directory = _publish_arm(
        tmp_path,
        _arm(CANDIDATE_FACTOR, 0, 0.02),
        CANDIDATE_FACTOR,
        0,
    )
    ledger_path = directory / "ledger-cost_2x-symbol-0.json.gz"
    ledger_path.write_bytes(ledger_path.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="ledger compressed digest mismatch"):
        _read_arm(tmp_path, CANDIDATE_FACTOR, 0, _PROTOCOL)


@pytest.mark.parametrize(
    ("field", "wrong_value", "error_match"),
    [
        ("seed", True, "protocol/provenance mismatch"),
        ("feature_indices", [False, 1], "feature/training configuration"),
    ],
)
def test_read_arm_rejects_boolean_numeric_identity_aliases(
    tmp_path: Path,
    field: str,
    wrong_value: object,
    error_match: str,
) -> None:
    seed = 1
    directory = _publish_arm(
        tmp_path,
        _arm(CANDIDATE_FACTOR, seed, 0.02),
        CANDIDATE_FACTOR,
        seed,
    )
    result_path = directory / "result.json"
    result = json.loads(result_path.read_bytes())
    result[field] = wrong_value
    raw = canonical_json_bytes(result)
    result_path.write_bytes(raw)
    (directory / "result.sha256.json").write_bytes(
        canonical_json_bytes({"sha256": hashlib.sha256(raw).hexdigest()})
    )

    with pytest.raises(ValueError, match=error_match):
        _read_arm(tmp_path, CANDIDATE_FACTOR, seed, _PROTOCOL)


def test_real_capture_schema_passes_ledger_chain_validator() -> None:
    dataset = _data()
    replay = evaluate_directional_arm(
        dataset,
        _factory,
        start_index=0,
        stop_index=5,
        symbol_index=0,
        initial_capital=12_345.0,
        capture_ledger_evidence=True,
    )
    ledger = json.loads(canonical_json_bytes(replay.pop("ledger_evidence")))
    protocol = deepcopy(_PROTOCOL)
    protocol["evaluation_dataset_id"] = dataset.dataset_id
    protocol["symbols"] = list(dataset.symbols)
    protocol["evaluation"].update(
        start_index=0,
        stop_index=5,
        interval_count=len(replay["returns"]),
        initial_capital=replay["initial_capital"],
        gross_budget=0.1,
        risk=replay["risk"],
        interval_year_slices={"2023": [0, len(replay["returns"])]},
    )
    protocol["evaluation"]["scenarios"]["base"]["execution_policy_digest"] = replay[
        "execution_policy_digest"
    ]

    summary = _validate_ledger_payload(
        ledger,
        row=replay,
        protocol=protocol,
        scenario_name="base",
        symbol="BTCUSDT",
    )
    replay["scenario"] = "base"
    replay["ledger_trace_summary"] = summary
    validated = _validate_replay(
        replay,
        protocol=protocol,
        scenario_name="base",
        symbol="BTCUSDT",
        symbol_index=0,
    )

    assert summary["interval_count"] == len(replay["returns"])
    assert validated["ledger_max_drawdown"] == replay["ledger_max_drawdown"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("interval_cost", 1.0),
        ("interval_funding", 1.0),
        ("interval_borrow_cost", 1.0),
        ("interval_dividend", nan),
        ("interval_cash_interest", nan),
    ],
)
def test_ledger_validates_interval_accounting_components(
    field: str, value: float
) -> None:
    row = _replay(0, (0.01, 0.01, 0.01, 0.01))
    ledger = _ledger_payload(row)
    ledger["intervals"][0][field] = value

    with pytest.raises(ValueError, match="interval|finite numeric"):
        _validate_ledger_payload(
            ledger,
            row=row,
            protocol=_PROTOCOL,
            scenario_name="base",
            symbol="BTCUSDT",
        )


@pytest.mark.parametrize(
    "field",
    [
        "final_total_cost",
        "final_funding_pnl",
        "final_borrow_cost",
        "final_turnover_total",
    ],
)
def test_ledger_final_accounting_totals_match_last_interval(field: str) -> None:
    row = _replay(0, (0.01, 0.01, 0.01, 0.01))
    ledger = _ledger_payload(row)
    ledger[field] = 1.0

    with pytest.raises(ValueError, match="final ledger state disagrees"):
        _validate_ledger_payload(
            ledger,
            row=row,
            protocol=_PROTOCOL,
            scenario_name="base",
            symbol="BTCUSDT",
        )


@pytest.mark.parametrize(
    ("before_field", "after_field", "final_field"),
    [
        ("total_cost_before", "total_cost_after", "final_total_cost"),
        ("funding_pnl_before", "funding_pnl_after", "final_funding_pnl"),
        ("borrow_cost_before", "borrow_cost_after", "final_borrow_cost"),
        ("turnover_total_before", "turnover_total_after", "final_turnover_total"),
    ],
)
def test_ledger_cumulative_accounting_starts_at_zero(
    before_field: str, after_field: str, final_field: str
) -> None:
    row = _replay(0, (0.01, 0.01, 0.01, 0.01))
    ledger = _ledger_payload(row)
    for interval in ledger["intervals"]:
        interval[before_field] = 1.0
        interval[after_field] = 1.0
    ledger[final_field] = 1.0

    with pytest.raises(ValueError, match="initial account state"):
        _validate_ledger_payload(
            ledger,
            row=row,
            protocol=_PROTOCOL,
            scenario_name="base",
            symbol="BTCUSDT",
        )


def test_ledger_rejects_opening_capital_mismatch() -> None:
    row = _replay(0, (0.01, 0.01, 0.01, 0.01))
    ledger = _ledger_payload(row)
    protocol = deepcopy(_PROTOCOL)
    protocol["evaluation"]["initial_capital"] = 9_000.0

    with pytest.raises(ValueError, match="initial account state"):
        _validate_ledger_payload(
            ledger,
            row=row,
            protocol=protocol,
            scenario_name="base",
            symbol="BTCUSDT",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cash_before", 11_000.0),
        ("portfolio_value_before", 11_000.0),
        ("interval_net_return", 0.02),
    ],
)
def test_ledger_rejects_interval_chain_discontinuity(field: str, value: float) -> None:
    row = _replay(0, (0.01, 0.01, 0.01, 0.01))
    ledger = _ledger_payload(row)
    ledger["intervals"][1][field] = value

    with pytest.raises(ValueError, match="differs|broken"):
        _validate_ledger_payload(
            ledger,
            row=row,
            protocol=_PROTOCOL,
            scenario_name="base",
            symbol="BTCUSDT",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("final_cash", 9_000.0),
        ("final_portfolio_value", 9_000.0),
        ("final_max_drawdown", 0.1),
        ("terminal_exact_quantities", ["1"] * len(_SYMBOLS)),
    ],
)
def test_ledger_rejects_final_state_mismatch(field: str, value: object) -> None:
    row = _replay(0, (0.01, 0.01, 0.01, 0.01))
    ledger = _ledger_payload(row)
    ledger[field] = value

    with pytest.raises(ValueError, match="final ledger state disagrees"):
        _validate_ledger_payload(
            ledger,
            row=row,
            protocol=_PROTOCOL,
            scenario_name="base",
            symbol="BTCUSDT",
        )
