"""Unit contracts of the SD2 runner pieces on synthetic arrays (no market value)."""

from __future__ import annotations

import datetime as dt
import hashlib
from typing import Any

import numpy as np
import pytest

from tests.evaluation.signal_diagnosis_run_support import (
    CLEAN,
    DIRTY,
    SYNTHETIC_IDENTITY,
    TEST_PLAN,
    protocol,
)
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.signal_diagnosis.protocol import (
    ARRAYS_DIGEST,
    ARTIFACT_DIGEST,
    DATASET_ID,
    FEATURE_CONFIG_DIGEST,
    ForbiddenEvaluationWindowError,
    SignalDiagnosisProtocolError,
    protocol_digest,
)
from trade_rl.evaluation.signal_diagnosis_run.fitting import (
    FamilyData,
    RunPlan,
    _stats,  # noqa: PLC2701
    build_labels,
    build_training,
    build_vintages,
    drift_forecast,
    ridge_from_stats,
    select_ridge_penalty,
)
from trade_rl.evaluation.signal_diagnosis_run.measure import (
    grouped_ic,
    grouped_rank_ic_increment,
)
from trade_rl.evaluation.signal_diagnosis_run.panel import (
    DatasetIdentity,
    SignalPanel,
    admit_panel,
    parse_instant_ns,
    protocol_feature_names,
)
from trade_rl.evaluation.signal_diagnosis_run.runner import (
    FROZEN_PROTOCOL_DIGEST,
    G4_AUTHORIZATION_URL_PATTERN,
    authorization_digest,
    run_signal_diagnosis,
    validate_authorization_url,
    write_outputs,
)
from trade_rl.evaluation.signal_measurement import (
    daily_ic,
    daily_rank_ic_increment,
)
from trade_rl.strategies.forecasts.prequential import PrequentialSchedule
from trade_rl.strategies.forecasts.ridge import fit_ridge_forecast_from_training_set
from trade_rl.strategies.forecasts.supervised import build_causal_forecast_training_set

HOUR = 3_600_000_000_000
URL = "https://github.com/shuntatsu/trade_rl/issues/1146#issuecomment-1234567890"


def _tiny_panel(
    stamps: np.ndarray,
    *,
    identity: DatasetIdentity = SYNTHETIC_IDENTITY,
    symbols: tuple[str, ...] | None = None,
) -> SignalPanel:
    spec = protocol()
    names = protocol_feature_names(spec)
    roster = spec.data.symbols if symbols is None else symbols
    rows = stamps.size
    rng = np.random.default_rng(0)
    return SignalPanel(
        identity=identity,
        symbols=roster,
        feature_names=names,
        timestamps_ns=stamps,
        close=100.0 * np.exp(0.01 * rng.standard_normal((rows, len(roster)))),
        features=np.zeros((rows, len(roster), len(names)), dtype=np.float32),
        feature_available=np.ones((rows, len(roster), len(names)), dtype=np.bool_),
        available_at_ns=np.repeat(stamps[:, None], len(roster), axis=1),
    )


def _stamps(start: str, hours: int) -> np.ndarray:
    first = parse_instant_ns(start)
    return first + HOUR * np.arange(hours, dtype=np.int64)


# ------------------------------------------------------------------ grouped IC


@pytest.mark.parametrize("method", ["rank", "linear"])
def test_grouped_ic_equals_the_sd1_daily_ic_including_ties(method: str) -> None:
    rng = np.random.default_rng(3)
    day = np.repeat(np.arange(40), 25).astype(np.int64)
    score = np.round(rng.standard_normal(day.size), 1)  # many ties
    target = score * 0.3 + rng.standard_normal(day.size)
    score[day == 7] = 1.0  # a constant day has no IC
    reference = daily_ic(day, score, target, method=method)  # type: ignore[arg-type]
    days, values = grouped_ic(day, score, target, method=method)
    assert tuple(days) == reference.day_ids
    np.testing.assert_allclose(values, reference.values, atol=1e-12)


def test_grouped_increment_equals_the_sd1_increment_with_ties_nans_and_skips() -> None:
    rng = np.random.default_rng(4)
    day = np.repeat(np.arange(60), 20).astype(np.int64)
    forecast = np.round(rng.standard_normal(day.size), 1)
    drift = np.round(rng.standard_normal(day.size), 1)
    drift[day == 5] = 0.5  # the baseline IC is undefined: skipped
    target = forecast * 0.2 + rng.standard_normal(day.size)
    target[rng.random(day.size) < 0.1] = np.nan
    reference = daily_rank_ic_increment(day, forecast, drift, target)
    days, values, skipped = grouped_rank_ic_increment(day, forecast, drift, target)
    assert tuple(days) == reference.day_ids and skipped == reference.skipped_days
    np.testing.assert_allclose(values, reference.values, atol=1e-12)
    # an unsorted group key (the regime slice) gives the same answer per group
    shuffled = rng.permutation(day.size)
    again = grouped_rank_ic_increment(
        day[shuffled], forecast[shuffled], drift[shuffled], target[shuffled]
    )
    assert again[2] == skipped and len(again[0]) == len(days)


# ------------------------------------------------- training set and Ridge pins


def _small_dataset() -> tuple[MarketDataset, SignalPanel]:
    rows, symbols, features = 400, 3, 4
    rng = np.random.default_rng(5)
    stamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        rows
    ) * np.timedelta64(1, "h")
    close = 100.0 * np.exp(
        np.cumsum(0.01 * rng.standard_normal((rows, symbols)), axis=0)
    )
    values = rng.standard_normal((rows, symbols, features)).astype(np.float32)
    available = np.ones(values.shape, dtype=np.bool_)
    available[:30] = False  # warm-up
    available[200:210, 1, 2] = False  # a gap for one symbol
    dataset = MarketDataset(
        dataset_id="5" * 64,
        symbols=("A", "B", "C"),
        timestamps=stamps,
        features=values,
        global_features=np.zeros((rows, 1), dtype=np.float32),
        open=close.copy(),
        high=close.copy(),
        low=close.copy(),
        close=close,
        volume=np.full((rows, symbols), 1e6),
        funding_rate=np.zeros((rows, symbols)),
        tradable=np.ones((rows, symbols), dtype=np.bool_),
        feature_available=available,
        feature_names=tuple(f"f{i}" for i in range(features)),
        global_feature_names=("g",),
        periods_per_year=8_760,
    )
    ns = stamps.astype(np.int64)
    panel = SignalPanel(
        identity=SYNTHETIC_IDENTITY,
        symbols=dataset.symbols,
        feature_names=dataset.feature_names,
        timestamps_ns=ns,
        close=close,
        features=values,
        feature_available=available,
        available_at_ns=np.repeat(ns[:, None], symbols, axis=1),
    )
    return dataset, panel


def test_array_training_set_equals_the_shared_causal_builder() -> None:
    dataset, panel = _small_dataset()
    labels = build_labels(panel, 24)
    ok = panel.feature_available.all(axis=2) & np.isfinite(panel.features).all(axis=2)
    data = FamilyData("f", panel.features, ok)
    cut_idx = 300
    cutoff = np.datetime64(int(panel.timestamps_ns[cut_idx]), "ns")
    mine = build_training(data, labels, 3, cut_idx)
    shared = build_causal_forecast_training_set(
        dataset, feature_indices=(0, 1, 2, 3), fit_cutoff=cutoff, horizon_hours=24
    )
    assert mine is not None
    np.testing.assert_allclose(mine.X, shared.features)
    np.testing.assert_allclose(mine.y, shared.labels)
    np.testing.assert_allclose(mine.w, shared.sample_weights)
    # the shared builder drops a row whose label closes exactly at the cutoff
    assert int(mine.t_idx.max()) + 24 < cut_idx


@pytest.mark.parametrize("alpha", [0.1, 10.0, 1000.0])
def test_ridge_from_moments_equals_the_shared_ridge_fit(alpha: float) -> None:
    dataset, panel = _small_dataset()
    labels = build_labels(panel, 24)
    ok = panel.feature_available.all(axis=2)
    train = build_training(FamilyData("f", panel.features, ok), labels, 3, 300)
    assert train is not None
    shared = build_causal_forecast_training_set(
        dataset,
        feature_indices=(0, 1, 2, 3),
        fit_cutoff=np.datetime64(int(panel.timestamps_ns[300]), "ns"),
        horizon_hours=24,
    )
    expected = fit_ridge_forecast_from_training_set(shared, alpha=alpha)
    mean, scale, coefficients, intercept = ridge_from_stats(
        _stats(train.X, train.y, train.w), alpha
    )
    np.testing.assert_allclose(
        coefficients, expected.coefficients, rtol=1e-9, atol=1e-12
    )
    np.testing.assert_allclose(mean, expected.feature_mean, rtol=1e-9)
    np.testing.assert_allclose(scale, expected.feature_scale, rtol=1e-9)
    assert intercept == pytest.approx(expected.intercept)


def test_inner_cv_picks_the_signal_penalty_and_ties_go_to_the_larger_penalty() -> None:
    rng = np.random.default_rng(1)
    n = 3000
    t_idx = np.arange(n)
    X = rng.standard_normal((n, 4))
    y = 0.5 * X[:, 0] + 0.1 * rng.standard_normal(n)
    w = np.ones(n)
    grid = (0.1, 1.0, 10.0, 100.0, 1000.0)
    best, scores = select_ridge_penalty(X, y, w, t_idx, horizon=24, grid=grid, folds=5)
    assert best == 0.1 and scores[0] < scores[-1]
    flat, _ = select_ridge_penalty(
        X, np.zeros(n), w, t_idx, horizon=24, grid=grid, folds=5
    )
    assert flat == 1000.0  # identical (zero) scores: the larger penalty wins


def test_inner_cv_validation_never_sees_a_label_that_closes_inside_its_block() -> None:
    n, horizon = 600, 24
    t_idx = np.arange(n)
    # A label that equals the future of the feature: leakage across the fold edge would
    # make every penalty look perfect; the purge keeps the score honest.
    X = np.random.default_rng(2).standard_normal((n, 1))
    y = np.roll(X[:, 0], -horizon)
    _, scores = select_ridge_penalty(
        X, y, np.ones(n), t_idx, horizon=horizon, grid=(0.1, 1.0), folds=5
    )
    assert min(scores) > 0.5  # the feature cannot predict a shifted white noise


def test_drift_baseline_uses_labels_matured_at_or_before_the_refit_instant() -> None:
    _, panel = _small_dataset()
    labels = build_labels(panel, 24)
    first = int(panel.timestamps_ns[100])
    schedule_plan = RunPlan(refit_interval_days=5)
    vintages = build_vintages(panel, _FirstFit(first), schedule_plan)  # type: ignore[arg-type]
    assert len(vintages) >= 2
    drift = drift_forecast(panel, labels, vintages)
    for vintage in vintages:
        last = int(np.searchsorted(panel.timestamps_ns, vintage.cutoff_ns, "right")) - 1
        for s in range(3):
            mature = [
                labels.raw[t, s]
                for t in range(panel.n_rows)
                if t + 24 <= last and labels.valid[t, s]
            ]
            expected = np.mean(mature) if mature else np.nan
            got = drift[vintage.first_row, s]
            assert (np.isnan(expected) and np.isnan(got)) or got == pytest.approx(
                expected
            )


class _FirstFit:
    """Minimal stand-in exposing only what ``build_vintages`` reads."""

    def __init__(self, first_ns: int) -> None:
        text = np.datetime64(first_ns, "ns").astype("datetime64[s]").astype(str) + "Z"
        self.prequential = type(
            "P",
            (),
            {"first_fit_time": text, "activation_delay_hours": 0},
        )()


def test_vintages_follow_the_shared_prequential_schedule() -> None:
    spec = protocol()
    start = parse_instant_ns(spec.prequential.first_fit_time)
    panel = _tiny_panel(start + HOUR * np.arange(24 * 95, dtype=np.int64))
    vintages = build_vintages(panel, spec, TEST_PLAN)
    schedule = PrequentialSchedule(
        first_fit_time=np.datetime64(start, "ns"), refit_interval_days=365
    )
    for t in (0, 1, 24 * 60, 24 * 94):
        decision = np.datetime64(int(panel.timestamps_ns[t]), "ns")
        cutoff = int(schedule.cutoff_for(decision).astype(np.int64))
        owner = next(v for v in vintages if v.first_row <= t < v.stop_row)
        assert owner.cutoff_ns == cutoff


# ------------------------------------------------------------- panel admission


def test_a_bar_after_the_window_end_is_refused_as_forbidden_data() -> None:
    spec = protocol()
    end = parse_instant_ns(spec.data.evaluation_end)
    inside = _tiny_panel(end - HOUR * np.arange(5, 0, -1, dtype=np.int64))
    admit_panel(inside, spec)  # the bar closing exactly at the end is inside
    for extra in (1, 24 * 40, 24 * 100):  # warm-up span, deep warm-up, sealed window
        stamps = (
            end - HOUR * 4 + HOUR * np.arange(5 + extra, dtype=np.int64)[: 5 + extra]
        )
        with pytest.raises(ForbiddenEvaluationWindowError):
            admit_panel(_tiny_panel(stamps), spec)


def test_a_bar_before_the_window_start_a_wrong_roster_or_an_irregular_grid_is_refused() -> (
    None
):
    spec = protocol()
    start = parse_instant_ns(spec.data.evaluation_start)
    with pytest.raises(
        SignalDiagnosisProtocolError, match="before the protocol window"
    ):
        admit_panel(
            _tiny_panel(start - HOUR + HOUR * np.arange(5, dtype=np.int64)), spec
        )
    good = start + HOUR * np.arange(1, 6, dtype=np.int64)
    with pytest.raises(SignalDiagnosisProtocolError, match="roster"):
        admit_panel(_tiny_panel(good, symbols=("A", "B", "C", "D", "E")), spec)
    with pytest.raises(SignalDiagnosisProtocolError, match="regular hourly"):
        admit_panel(_tiny_panel(good * 1 + np.array([0, 1, 2, 4, 5]) * HOUR), spec)


# ------------------------------------------------------- authorization and run


def test_the_authorization_url_must_be_an_owner_comment_url() -> None:
    assert validate_authorization_url(URL) == URL
    for bad in (
        None,
        "",
        "https://example.com/x",
        URL + "x",
        URL.replace("issues", "x"),
    ):
        with pytest.raises(SignalDiagnosisProtocolError, match="G4 authorization"):
            validate_authorization_url(bad)
    assert G4_AUTHORIZATION_URL_PATTERN.fullmatch(URL)
    digest = authorization_digest(URL, FROZEN_PROTOCOL_DIGEST)
    assert (
        digest
        == hashlib.sha256(
            f"signal_diagnosis_g4_authorization_v1\n{URL}\n{FROZEN_PROTOCOL_DIGEST}".encode()
        ).hexdigest()
    )
    assert authorization_digest(URL, "0" * 64) != digest  # bound to the protocol


def test_the_frozen_digest_constant_is_the_merged_v4_digest() -> None:
    assert protocol_digest(protocol()) == FROZEN_PROTOCOL_DIGEST
    assert FROZEN_PROTOCOL_DIGEST.startswith("1f48b065b03bc38299a751bef89456b57d5afc6a")


def _real_identity() -> DatasetIdentity:
    return DatasetIdentity(
        DATASET_ID, ARTIFACT_DIGEST, ARRAYS_DIGEST, FEATURE_CONFIG_DIGEST
    )


def _small_real_panel() -> SignalPanel:
    start = parse_instant_ns(protocol().data.evaluation_start)
    return _tiny_panel(
        start + HOUR * np.arange(1, 50, dtype=np.int64), identity=_real_identity()
    )


def test_a_real_run_is_refused_without_a_g4_url_or_from_a_dirty_tree_or_a_new_plan() -> (
    None
):
    panel, spec = _small_real_panel(), protocol()
    base: dict[str, Any] = {"scope": "real_development", "code_provenance": CLEAN}
    with pytest.raises(SignalDiagnosisProtocolError, match="G4 authorization"):
        run_signal_diagnosis(panel, spec, **base)
    with pytest.raises(SignalDiagnosisProtocolError, match="dirty working tree"):
        run_signal_diagnosis(
            panel,
            spec,
            scope="real_development",
            code_provenance=DIRTY,
            g4_authorization_url=URL,
        )
    with pytest.raises(SignalDiagnosisProtocolError, match="frozen plan"):
        run_signal_diagnosis(
            panel, spec, plan=TEST_PLAN, g4_authorization_url=URL, **base
        )


def test_real_identity_mismatch_and_protocol_digest_mismatch_are_invalid_not_run() -> (
    None
):
    spec = protocol()
    wrong = DatasetIdentity(
        "1" * 64, ARTIFACT_DIGEST, ARRAYS_DIGEST, FEATURE_CONFIG_DIGEST
    )
    start = parse_instant_ns(spec.data.evaluation_start)
    panel = _tiny_panel(start + HOUR * np.arange(1, 50, dtype=np.int64), identity=wrong)
    run = run_signal_diagnosis(
        panel,
        spec,
        scope="real_development",
        code_provenance=CLEAN,
        g4_authorization_url=URL,
    )
    assert run.label == "INVALID"
    assert (
        "dataset_identity_mismatch"
        in run.records["verdict.json"]["validity_flags_raised"]
    )
    assert set(run.records) == {"run_record.json", "verdict.json"}  # nothing computed
    tampered = run_signal_diagnosis(
        _small_real_panel(),
        spec,
        scope="real_development",
        code_provenance=CLEAN,
        g4_authorization_url=URL,
        expected_protocol_digest="0" * 64,
    )
    assert tampered.label == "INVALID"
    assert (
        "protocol_digest_mismatch"
        in tampered.records["verdict.json"]["validity_flags_raised"]
    )


def test_a_synthetic_run_refuses_a_g4_url_and_the_development_identity() -> None:
    spec = protocol()
    start = parse_instant_ns(spec.data.evaluation_start)
    stamps = start + HOUR * np.arange(1, 50, dtype=np.int64)
    with pytest.raises(SignalDiagnosisProtocolError, match="no G4 URL"):
        run_signal_diagnosis(
            _tiny_panel(stamps),
            spec,
            scope="synthetic_only",
            code_provenance=CLEAN,
            g4_authorization_url=URL,
        )
    with pytest.raises(SignalDiagnosisProtocolError, match="is a real run"):
        run_signal_diagnosis(
            _tiny_panel(stamps, identity=_real_identity()),
            spec,
            scope="synthetic_only",
            code_provenance=CLEAN,
        )
    with pytest.raises(SignalDiagnosisProtocolError, match="scope must be"):
        run_signal_diagnosis(
            _tiny_panel(stamps), spec, scope="other", code_provenance=CLEAN
        )


def test_outputs_are_written_once_and_never_overwritten(tmp_path: Any) -> None:
    from trade_rl.evaluation.signal_diagnosis_run.runner import SignalDiagnosisRun

    run = SignalDiagnosisRun("INVALID", {"verdict.json": {"verdict": "INVALID"}}, {})
    paths = write_outputs(tmp_path / "out", run)
    assert [p.name for p in paths] == ["verdict.json"]
    assert b"\r" not in paths[0].read_bytes()
    with pytest.raises(FileExistsError):
        write_outputs(tmp_path / "out", run)


def test_cli_refuses_without_a_url_a_clean_tree_and_a_fresh_output_dir(
    tmp_path: Any,
) -> None:
    from trade_rl.evaluation.signal_diagnosis_run import cli

    base = ["--dataset", str(tmp_path / "none"), "--output-dir", str(tmp_path / "o")]
    assert cli.main([*base, "--g4-authorization-url", "https://example.com"]) == 2
    (tmp_path / "o").mkdir()
    (tmp_path / "o" / "x").write_text("x")
    assert cli.main([*base, "--g4-authorization-url", URL]) == 2
    with pytest.raises(SystemExit):
        cli.main(base)  # the URL is a required argument
    assert dt.UTC is not None
