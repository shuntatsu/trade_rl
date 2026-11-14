"""Result-blind scenario oracles of the SD2 runner on seeded synthetic markets.

Every panel covers the frozen protocol window with the real roster and feature names;
the truth (a planted signal, pure noise, a signal too small to pay the cost, a leaky
feature) is known by construction. The LightGBM regressor is replaced by a weighted
least-squares stand-in so the tests run where the optional extra is absent; one test
exercises the real regressor when it is installed. No market value is read.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import trade_rl.evaluation.signal_diagnosis_run.runner as runner_module
import trade_rl.strategies.forecasts.lightgbm as lightgbm_module
from tests.evaluation.signal_diagnosis_run_support import (
    CLEAN,
    TEST_PLAN,
    WeightedLinearRegressor,
    protocol,
    synthetic_panel,
)
from trade_rl.evaluation.signal_diagnosis.protocol import (
    SignalDiagnosisProtocolError,
)
from trade_rl.evaluation.signal_diagnosis.verdict import (
    DecisionCellEvidence,
    PValueProvenance,
    classify_signal_diagnosis,
)
from trade_rl.evaluation.signal_diagnosis_run.panel import SignalPanel
from trade_rl.evaluation.signal_diagnosis_run.runner import (
    SignalDiagnosisRun,
    run_signal_diagnosis,
    write_outputs,
)
from trade_rl.evaluation.signal_measurement import future_perturbation_probe


@pytest.fixture(scope="module", autouse=True)
def fake_lightgbm() -> Iterator[None]:
    patch = pytest.MonkeyPatch()
    patch.setattr(
        lightgbm_module, "_lightgbm_regressor_factory", lambda: WeightedLinearRegressor
    )
    yield
    patch.undo()


def _clean_probe(_: Any) -> dict[str, Any]:
    return {"leaks": False, "screen": "test_stub"}


def _run(panel: Any, **kwargs: Any) -> SignalDiagnosisRun:
    kwargs.setdefault("builder_probe", _clean_probe)
    kwargs.setdefault("plan", TEST_PLAN)
    return run_signal_diagnosis(
        panel,
        protocol(),
        scope="synthetic_only",
        code_provenance=CLEAN,
        **kwargs,
    )


@pytest.fixture(scope="module")
def planted() -> SignalDiagnosisRun:
    panel = synthetic_panel(seed=11, planted_families=("trend",), signal=0.5)
    return _run(panel)


@pytest.fixture(scope="module")
def noise() -> SignalDiagnosisRun:
    return _run(synthetic_panel(seed=21))


@pytest.fixture(scope="module")
def too_small_to_pay() -> SignalDiagnosisRun:
    # 24-bar moves of about 1.5 bp: a real rank signal, far below the 14 bp round trip.
    panel = synthetic_panel(
        seed=31, planted_families=("trend",), signal=0.5, hourly_vol=0.00003
    )
    return _run(panel)


def _cell(run: SignalDiagnosisRun, cell_id: str) -> dict[str, Any]:
    return run.records[f"cells/{cell_id.replace(':', '__')}.json"]  # type: ignore[no-any-return]


def test_a_planted_signal_is_a_signal_candidate_only_where_it_was_planted(
    planted: SignalDiagnosisRun,
) -> None:
    verdict = planted.records["verdict.json"]
    assert planted.label == "SIGNAL_CANDIDATE" == verdict["verdict"]
    assert set(verdict["signal_cell_ids"]) == {
        "trend:ridge_inner_cv",
        "trend:lightgbm_fixed",
    }
    assert verdict["candidate_cell_ids"] and set(verdict["candidate_cell_ids"]) <= set(
        verdict["signal_cell_ids"]
    )
    assert verdict["claim_ceiling"] == "known_used_exploratory_screening_only"
    for item in verdict["cells"]:
        if not item["cell_id"].startswith("trend:"):
            assert item["rank_ic_p_bh_adjusted"] > 0.1  # the null families stay null
    decisive = _cell(planted, "trend:ridge_inner_cv")["d4"]["decisive"]
    assert decisive["ci_lower_cost_x1"] > 0.0 and decisive["point_cost_x2"] > 0.0


def test_stopped_cells_carry_no_d4_numbers_and_every_cell_is_saved(
    planted: SignalDiagnosisRun,
) -> None:
    cells = [name for name in planted.records if name.startswith("cells/")]
    assert len(cells) == 16
    stopped = _cell(planted, "volume_flow:ridge_inner_cv")
    assert stopped["d4"] == {"status": "not_computed_upper_layer_stopped"}
    computed = _cell(planted, "trend:lightgbm_fixed")["d4"]
    assert set(computed["grid"]) == {"k1_x1", "k1_x2", "k2_x1", "k2_x2"}
    assert sorted(planted.records["verdict.json"]["d4_evaluated_cells"]) == [
        "trend:lightgbm_fixed",
        "trend:ridge_inner_cv",
    ]


def test_pure_noise_is_no_signal(noise: SignalDiagnosisRun) -> None:
    verdict = noise.records["verdict.json"]
    assert noise.label == "NO_SIGNAL" == verdict["verdict"]
    assert verdict["signal_cell_ids"] == [] and verdict["candidate_cell_ids"] == []
    assert verdict["d4_evaluated_cells"] == []


def test_a_signal_that_does_not_survive_costs_is_not_tradable(
    too_small_to_pay: SignalDiagnosisRun,
) -> None:
    verdict = too_small_to_pay.records["verdict.json"]
    assert too_small_to_pay.label == "SIGNAL_NOT_TRADABLE"
    assert set(verdict["signal_cell_ids"]) == {
        "trend:ridge_inner_cv",
        "trend:lightgbm_fixed",
    }
    assert verdict["candidate_cell_ids"] == []
    d4 = _cell(too_small_to_pay, "trend:ridge_inner_cv")["d4"]
    assert d4["grid"]["k1_x1"]["share_of_rows_traded"] < 0.01  # the rule stays flat


def test_a_feature_that_copies_the_label_is_caught_and_the_run_is_invalid() -> None:
    run = _run(synthetic_panel(seed=41, leak_copy_of_label=True))
    verdict = run.records["verdict.json"]
    assert run.label == "INVALID"
    assert "leakage_detected" in verdict["validity_flags_raised"]
    flagged = run.records["run_record.json"]["d0_leakage"]["label_copy"]["flagged"]
    assert flagged and flagged[0]["feature"].endswith("body_return")
    assert set(run.records) == {"run_record.json", "verdict.json"}  # nothing fitted
    assert run.records["run_record.json"]["status"] == "invalid_before_fits"


def test_a_builder_that_reads_the_future_is_caught_by_the_probe() -> None:
    def centred_average(values: np.ndarray) -> np.ndarray:
        padded = np.vstack([values[:1], values, values[-1:]])
        return (padded[:-2] + padded[1:-1] + padded[2:]) / 3.0

    def probe(_: Any) -> dict[str, Any]:
        rows = np.random.default_rng(0).standard_normal((60, 2)) + 3.0
        result = future_perturbation_probe(
            centred_average, rows, cutoffs=(20, 30), seed=0, relative_scale=0.05
        )
        return {"leaks": result.leaks, "screen": "test_leaky_builder"}

    run = _run(synthetic_panel(seed=42), builder_probe=probe)
    assert run.label == "INVALID"
    assert "leakage_detected" in run.records["verdict.json"]["validity_flags_raised"]


def test_the_causal_maintained_builders_are_not_flagged(
    planted: SignalDiagnosisRun,
) -> None:
    from trade_rl.evaluation.signal_diagnosis_run.leakage import (
        maintained_builder_probe,
    )

    report = maintained_builder_probe(protocol())
    assert report["features_probed"] == 167 and report["leaks"] is False
    assert (
        planted.records["run_record.json"]["d0_leakage"]["label_copy"]["leaks"] is False
    )


def test_every_file_is_written_once_without_carriage_returns(
    planted: SignalDiagnosisRun, tmp_path: Path
) -> None:
    paths = write_outputs(tmp_path / "out", planted)
    names = {p.relative_to(tmp_path / "out").as_posix() for p in paths}
    assert {"run_record.json", "verdict.json", "forecasts.npz"} <= names
    assert len([n for n in names if n.startswith("cells/")]) == 16
    assert all(bytes([13]) not in p.read_bytes() for p in paths if p.suffix == ".json")
    with pytest.raises(FileExistsError):
        write_outputs(tmp_path / "out", planted)


def test_provenance_is_enforced_end_to_end(
    planted: SignalDiagnosisRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}
    real = runner_module.classify_signal_diagnosis

    def spy(spec: Any, *, validity: Any, cells: Any) -> Any:
        captured["cells"] = list(cells)
        return real(spec, validity=validity, cells=cells)

    monkeypatch.setattr(runner_module, "classify_signal_diagnosis", spy)
    run = _run(synthetic_panel(seed=11, planted_families=("trend",), signal=0.5))
    spec = protocol()
    cells: list[DecisionCellEvidence] = captured["cells"]
    assert len(cells) == 16
    for cell in cells:
        record = _cell(run, cell.cell_id)
        assert cell.p_value_provenance.procedure_id == spec.significance.procedure_id
        assert cell.p_value_provenance.raw_n_days == record["raw"]["n_days"]
        assert cell.p_value_provenance.demeaned_n_days == record["demeaned"]["n_days"]
        assert cell.rank_ic_p_value == record["raw"]["ewc"]["p_value_one_sided"]
    # The merged verdict refuses what the runner would never produce.
    first = cells[0]
    foreign = dataclasses.replace(
        first,
        p_value_provenance=PValueProvenance("block_bootstrap_v3", 1000, 1000),
    )
    with pytest.raises(SignalDiagnosisProtocolError, match="provenance"):
        classify_signal_diagnosis(
            spec, validity=_no_flags(), cells=[foreign, *cells[1:]]
        )
    short = dataclasses.replace(
        first,
        p_value_provenance=PValueProvenance(spec.significance.procedure_id, 3, 1000),
        rank_ic_p_value=0.01,
    )
    with pytest.raises(SignalDiagnosisProtocolError, match="exactly 1.0"):
        classify_signal_diagnosis(spec, validity=_no_flags(), cells=[short, *cells[1:]])
    # determinism: a second run of the same panel, protocol and code is identical
    assert json.dumps(run.records, sort_keys=True) == json.dumps(
        planted.records, sort_keys=True
    )
    for key, array in planted.forecasts.items():
        np.testing.assert_array_equal(array, run.forecasts[key])


def _no_flags() -> Any:
    from trade_rl.evaluation.signal_diagnosis.verdict import ValidityFlags

    return ValidityFlags(False, False, False, False, False)


def test_a_failing_cell_is_saved_and_no_verdict_is_produced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("fit failed")

    monkeypatch.setattr(runner_module, "forecast_family", boom)
    run = _run(synthetic_panel(seed=51))
    assert run.label is None
    assert run.records["verdict.json"]["verdict"] is None
    failed = _cell(run, "trend:ridge_inner_cv")
    assert failed["status"] == "failed" and "fit failed" in failed["error"]
    assert run.records["run_record.json"]["status"] == "halted_cell_failure"


def test_reported_layers_evaluate_each_feature_horizon_and_transform() -> None:
    from trade_rl.evaluation.signal_diagnosis_run.fitting import (
        build_labels,
        build_vintages,
    )
    from trade_rl.evaluation.signal_diagnosis_run.measure import (
        d0_profile,
        d1_beta,
        d2_features,
    )

    spec = protocol()
    panel = synthetic_panel(seed=61, planted_families=("trend",), signal=0.5)
    labels = {h: build_labels(panel, h) for h in (4, 24, 72)}
    vintages = build_vintages(panel, spec, TEST_PLAN)
    names = ("1h__macd_line_12_26", "1h__rsi_14bar", "1h__funding_bps")
    rows = d2_features(panel, spec, labels, vintages, names=names)["rows"]
    assert len(rows) == len(names) * 3 * 2
    planted_row = next(
        r
        for r in rows
        if r["feature"] == "1h__macd_line_12_26"
        and r["horizon_bars"] == 24
        and r["transform"] == "train_window_standardise_clip3"
    )
    assert planted_row["raw"]["mean_rank_ic"] > 0.05
    assert planted_row["raw"]["p_positive_direction"] < 0.001
    assert planted_row["raw"]["bh_positive"] < 0.1
    assert planted_row["sign_payoff_mean_daily"].keys() == {"x1", "x2"}
    beta = d1_beta(labels[24], vintages, spec.data.symbols, panel.timestamps_ns, 720)
    assert set(beta["beta_by_symbol"]) == set(spec.data.symbols)
    head = SignalPanel(
        identity=panel.identity,
        symbols=panel.symbols,
        feature_names=panel.feature_names,
        timestamps_ns=panel.timestamps_ns[:3000],
        close=panel.close[:3000],
        features=panel.features[:3000],
        feature_available=panel.feature_available[:3000],
        available_at_ns=panel.available_at_ns[:3000],
    )
    profile = d0_profile(head, clip_abs=3.0, min_train_rows=720)
    assert "profiles" in profile["symbols"]["BTCUSDT"]


@pytest.mark.skipif(
    importlib.util.find_spec("lightgbm") is None, reason="LightGBM extra not installed"
)
def test_the_real_lightgbm_regressor_recovers_the_planted_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lightgbm  # noqa: F401

    real_class = _real_factory()
    monkeypatch.setattr(
        lightgbm_module, "_lightgbm_regressor_factory", lambda: real_class
    )
    run = _run(synthetic_panel(seed=11, planted_families=("trend",), signal=0.5))
    cell = _cell(run, "trend:lightgbm_fixed")
    assert cell.get("status") == "measured", cell
    assert cell["demeaned"]["mean_rank_ic_increment"] > 0.02
    assert cell["demeaned"]["ewc"]["p_value_one_sided"] < 0.001


def _real_factory() -> Any:
    import importlib

    module = importlib.import_module("lightgbm")
    return module.LGBMRegressor
