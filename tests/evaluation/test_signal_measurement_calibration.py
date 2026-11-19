"""SD1 (#1146): the frozen significance test keeps its size at protocol scale.

A reduced, seeded version of ``tools/issue1146_significance_calibration.py``. Every
series is synthetic. The test would fail if the frozen procedure lost calibration
(a liberal per-test size or a family error above q under the global null) or lost
its power against a planted signal, and it keeps the superseded v3 bootstrap at 60
days as the control that must fail.
"""

from __future__ import annotations

import math

import numpy as np

from tools import issue1146_significance_calibration as study
from trade_rl.evaluation.signal_diagnosis.protocol import (
    default_signal_diagnosis_protocol,
)
from trade_rl.evaluation.signal_measurement import bh_rejections, ewc_mean_test

N_DAYS = 1460
REPLICATES = 300
CELLS = 16
Q = 0.10


def _null_series(config: study.Config, replicates: int) -> np.ndarray:
    rng = np.random.default_rng([20261112, config.n, len(config.name)])
    return study.series_increments(rng, replicates, config)


def _family_rates(
    p: np.ndarray, planted: np.ndarray, cells: int
) -> tuple[float, float, float]:
    """P(any BH rejection) under the null, and FDR and power with planted cells."""

    any_null = float(np.mean([bh_rejections(row, Q).any() for row in p]))
    mixed = p.copy()
    mixed[:, :cells] = planted[:, :cells]
    fdr, power = [], []
    for row in mixed:
        rejected = bh_rejections(row, Q)
        total = int(rejected.sum())
        fdr.append(int(rejected[cells:].sum()) / total if total else 0.0)
        power.append(float(rejected[:cells].mean()))
    return any_null, float(np.mean(fdr)), float(np.mean(power))


def test_the_sd1_test_holds_size_and_power_at_protocol_scale() -> None:
    config = study.Config("regression", "series", N_DAYS, phi=0.3, overlap=True)
    x = _null_series(config, REPLICATES)
    shift = study.PLANTED_DELTA * config.long_run_sd / math.sqrt(N_DAYS)

    p = np.array([[ewc_mean_test(cell).p_value_one_sided for cell in rep] for rep in x])
    planted = np.array(
        [[ewc_mean_test(cell + shift).p_value_one_sided for cell in rep] for rep in x]
    )

    tests = REPLICATES * CELLS
    size_005 = float((p <= 0.05).mean())
    size_q = float((p <= Q / CELLS).mean())
    any_null, fdr, power = _family_rates(p, planted, study.DENSE_CELLS)
    # Nominal plus three Monte Carlo standard errors (the study's criterion).
    assert size_005 <= 0.05 + 3 * math.sqrt(0.05 * 0.95 / tests)
    assert size_q <= Q / CELLS + 3 * math.sqrt((Q / CELLS) / tests)
    assert any_null <= Q + 3 * math.sqrt(Q * 0.9 / REPLICATES)
    assert fdr <= Q + 3 * math.sqrt(Q * 0.9 / REPLICATES)
    # Not conservative to the point of uselessness: a 3-SE signal is mostly found.
    assert size_005 >= 0.04 and power >= 0.75
    assert default_signal_diagnosis_protocol().significance.cosine_terms(N_DAYS) == 51


def test_the_superseded_v3_bootstrap_fails_the_same_criterion_at_sixty_days() -> None:
    """The control: the centred moving-block bootstrap of v3 is liberal at 60 days."""

    config = study.Config("control", "series", 60)
    x = _null_series(config, 200).reshape(-1, 60)
    plan = study.bootstrap_plan(60, 20, 2000, 0)

    p = study.v3_centred_mbb_p(x, (np.zeros(x.shape[0]),), plan)[0].reshape(200, 16)

    any_null = float(np.mean([bh_rejections(row, Q).any() for row in p]))
    assert any_null > Q + 3 * math.sqrt(Q * 0.9 / 200)
    assert float((p <= 0.05).mean()) > 0.05 + 3 * math.sqrt(0.05 * 0.95 / p.size)
