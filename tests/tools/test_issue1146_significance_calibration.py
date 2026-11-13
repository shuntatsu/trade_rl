"""The #1146 calibration harness measures the real procedures, on synthetic data only."""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest

from tools import issue1146_significance_calibration as study
from trade_rl.evaluation.comparison.bootstrap import moving_block_mean_test
from trade_rl.evaluation.signal_measurement import (
    bh_adjusted,
    daily_rank_ic_increment,
    ewc_mean_test,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("n", "block", "resamples"),
    [(1460, 20, 2000), (1461, 20, 300), (60, 20, 2000), (60, 5, 400), (47, 20, 200)],
)
def test_the_vectorised_v3_procedure_equals_the_repository_bootstrap(
    n: int, block: int, resamples: int
) -> None:
    rng = np.random.default_rng(n + block)
    series = 0.004 + 0.1 * rng.standard_normal((5, n))
    series[0] -= 0.05  # a non-positive mean: the one-sided p-value is exactly 1

    plan = study.bootstrap_plan(n, block, resamples, 0)
    vectorised = study.v3_centred_mbb_p(series, (np.zeros(5),), plan)[0]
    real = [
        moving_block_mean_test(
            tuple(float(v) for v in row),
            n_bootstrap=resamples,
            seed=0,
            block_size=block,
        ).p_value
        for row in series
    ]

    assert vectorised.tolist() == real
    assert vectorised[0] == 1.0 and 0.0 < vectorised[1:].min() < 1.0


def test_a_planted_shift_is_the_same_as_shifting_the_series() -> None:
    rng = np.random.default_rng(3)
    series = 0.1 * rng.standard_normal((8, 240))
    shift = np.full(8, 0.012)

    shifted = study.procedure_p_values(
        series + shift[:, None], np.zeros(8), block=20, resamples=300, seed=0
    )
    planted = study.procedure_p_values(series, shift, block=20, resamples=300, seed=0)

    for name in study.PROCEDURES:
        assert planted[name][1] == pytest.approx(shifted[name][0], abs=1e-12), name


def test_the_structural_increment_is_the_sd1_daily_rank_ic_increment() -> None:
    panel = study.structural_panel(np.random.default_rng(3), 6, heavy=True)
    days = np.repeat(np.arange(6), panel.labels.shape[1])

    fast = study.structural_increments(panel)

    assert fast.shape == (study.CELLS, 6)
    for cell in (0, 7, 15):
        real = daily_rank_ic_increment(
            days,
            panel.forecasts[cell].reshape(-1),
            panel.baseline.reshape(-1),
            panel.labels.reshape(-1),
        )
        assert real.skipped_days == 0
        assert fast[cell] == pytest.approx(np.asarray(real.values), abs=1e-12)


def test_the_structural_market_has_the_derived_overlap_and_shared_baseline() -> None:
    """Adjacent days share returns (lag 1 only) and all cells share the baseline IC."""

    panel = study.structural_panel(np.random.default_rng(11), 1460, heavy=False)
    x = study.structural_increments(panel)
    centred = x - x.mean(axis=1, keepdims=True)
    variance = (centred * centred).sum(axis=1)
    lag1 = (centred[:, 1:] * centred[:, :-1]).sum(axis=1) / variance
    lag2 = (centred[:, 2:] * centred[:, :-2]).sum(axis=1) / variance
    cross = np.corrcoef(x)

    # The closed-form bound for a constant feature is 2300 / 9224 = 0.249.
    assert 0.12 < float(lag1.mean()) < 0.30
    assert abs(float(lag2.mean())) < 0.05
    assert float(cross[np.triu_indices(study.CELLS, k=1)].mean()) > 0.4
    theta = study.OVERLAP_THETA
    assert theta / (1.0 + theta**2) == pytest.approx(0.25)
    overlap = sum(k * (24 - k) for k in range(1, 24))
    total = 24 * 24 + 2 * sum(k * k for k in range(1, 24))
    assert (overlap, total) == (2300, 9224)


def test_reference_laws_and_bh_match_independent_implementations() -> None:
    z = np.array([-1.0, 0.0, 2.5])
    expected = [0.5 * math.erfc(v / math.sqrt(2.0)) for v in z]
    assert study._normal_sf(z) == pytest.approx(expected, rel=1e-12)
    p = np.random.default_rng(5).uniform(size=(40, 16))
    assert study.bh_adjusted_rows(p) == pytest.approx(
        np.stack([bh_adjusted(row) for row in p]), abs=1e-15
    )


def test_series_nulls_have_the_declared_long_run_variance() -> None:
    config = study.Config("check", "series", 400, phi=0.3, overlap=True)
    x = study.series_increments(np.random.default_rng(8), 600, config)

    means = x.mean(axis=2).reshape(-1)
    observed = float(means.std(ddof=1)) * math.sqrt(config.n)

    assert x.shape == (600, study.CELLS, 400)
    # 9600 means: the standard deviation is known to about 0.7%.
    assert observed == pytest.approx(config.long_run_sd, rel=0.04)
    correlated = study.Config("rho", "series", 200, cross_rho=0.5, heavy=True)
    y = study.series_increments(np.random.default_rng(9), 200, correlated)
    assert float(np.corrcoef(y[:, 0, :].ravel(), y[:, 1, :].ravel())[0, 1]) > 0.35


def test_the_study_is_deterministic_and_independent_of_the_worker_count() -> None:
    configs = (
        study.Config("a", "series", 120, phi=0.3, overlap=True),
        study.Config("b", "series", 60, cells=9, block=5, resamples=100,
                     seed_per_replicate=True),
    )  # fmt: skip

    first = study.run_configs(configs, replicates=130, structural_replicates=0)
    again = study.run_configs(configs, replicates=130, structural_replicates=0)
    pooled = study.run_configs(
        configs, replicates=130, structural_replicates=0, workers=2
    )
    other = study.run_configs(
        configs, replicates=130, structural_replicates=0, seed=study.STUDY_SEED + 1
    )

    assert first == again == pooled
    assert first != other
    table = study.render_markdown(first)
    assert table.count("\n") == 1 + len(configs) * len(study.PROCEDURES)


def test_the_size_criterion_is_three_standard_errors_above_nominal() -> None:
    rng = np.random.default_rng(1)
    uniform = rng.uniform(size=(4000, study.CELLS))
    liberal = uniform**1.6  # too many small p-values

    good = study.summarize(uniform, uniform)
    bad = study.summarize(liberal, liberal)

    assert study.holds_size(good)
    assert not study.holds_size(bad)
    assert {name for name, _ in study.size_excesses(bad)} >= {"bh_any", "size@0.05000"}
    assert study.SIZE_TOLERANCE_SE == 3.0
    assert study.BH_Q == 0.10 and study.CELLS == 16


def test_the_harness_reads_no_dataset_network_or_market_store() -> None:
    source = (ROOT / "tools" / "issue1146_significance_calibration.py").read_text(
        encoding="utf-8"
    )
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    roots = {name.split(".", 1)[0] for name in imported}

    assert not roots & {"pandas", "requests", "urllib", "http", "socket", "psycopg"}
    assert {name for name in imported if name.startswith("trade_rl")} == {
        "trade_rl.evaluation.signal_measurement"
    }
    for forbidden in ("market_store", "dev_realdata", "forward_study", "read_bytes"):
        assert forbidden not in source, forbidden


def test_the_vectorised_ewc_test_equals_the_sd1_function() -> None:
    rng = np.random.default_rng(12)
    for n in (3, 4, 60, 125, 1460):
        series = 0.01 + 0.1 * rng.standard_normal((6, n))
        series[0] -= 0.2  # non-positive mean
        series[1] = 0.25  # constant: no variance estimate
        shift = np.full(6, 0.004)

        null, planted = study.ewc_t_p(series, (np.zeros(6), shift))

        for row in range(6):
            real = ewc_mean_test(series[row])
            assert null[row] == pytest.approx(real.p_value_one_sided, rel=1e-9), n
            moved = ewc_mean_test(series[row] + shift[row])
            assert planted[row] == pytest.approx(moved.p_value_one_sided, rel=1e-9)
        assert null[0] == 1.0 and null[1] == 1.0
