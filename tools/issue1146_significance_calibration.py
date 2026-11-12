"""Synthetic calibration of the signal-diagnosis significance test (Issue #1146).

Result-blind and synthetic only. Every number is drawn from a seeded numpy generator;
no dataset, file, network, price, return or market IC is read. The study asks one
question before any real-data diagnosis runs: does the one-sided p-value that feeds
the Benjamini-Hochberg gate of the signal diagnosis hold its size at protocol scale,
and what does it cost in power?

Procedures compared (all one-sided, positive alternative, on one series of daily
rank-IC increments):

* ``v3_centred_mbb``: the procedure ``signal_diagnosis_protocol_v3`` froze, the
  repository ``moving_block_mean_test`` (circular 20-day blocks, 2000 resamples,
  seed 0, bootstrap means centred at their own average). The vectorised version here
  consumes the same index stream as the real function; a test pins equal p-values.
* ``studentized_mbb``: a percentile-t circular block bootstrap with the same blocks,
  resamples and seed, studentised by the block-mean standard error.
* ``newey_west_rule`` and ``newey_west_block``: a Newey-West (Bartlett) HAC t-test
  against the normal law with lag ``floor(4 (n / 100) ** (2 / 9))`` (never below the
  one-day label overlap) and with lag equal to the 20-day block.
* ``ewc_t``: the equal-weighted cosine (EWC) long-run variance test of Lazarus,
  Lewis, Stock and Watson (2018), ``nu = floor(0.4 n ** (2 / 3))`` cosine terms and
  a Student-t reference with ``nu`` degrees of freedom. This is the SD1 function
  ``ewc_mean_test``; a test pins the vectorised version here to it.

Null models. A series-level model ``x_t = u_t + theta u_{t-1}`` with ``u`` an AR(1) of
coefficient ``phi``: ``phi`` in {0, 0.1, 0.3} is serial dependence of the increments
and ``theta = 2 - sqrt(3)`` is the label overlap. The overlap value is derived, not
tuned: a day's pooled IC uses 24 hourly rows whose 24-bar labels reach into the next
day, so adjacent days share returns and later days do not; for a feature that never
changes the lag-1 autocorrelation of the daily mean of ``feature x label`` is
``2300 / 9224 = 0.249`` (the largest value any non-negatively autocorrelated feature
gives), and an MA(1) has lag-1 autocorrelation ``theta / (1 + theta ** 2) = 0.25`` at
``theta = 2 - sqrt(3)``. Innovations are Gaussian or Student-t with 4 degrees of
freedom; cells are independent or share a common factor (correlation ``rho``; the
heavy-tailed correlated case is a multivariate t whose scale is common to all cells).

A structural model builds the statistic the protocol defines instead of assuming its
dependence: an hourly five-symbol panel, 24-bar forward labels, eight feature
families of different persistence fitted by two correlated "models", the expanding
monthly drift baseline, and the pooled daily rank IC of each forecast minus that of
the baseline on the same rows. Features are independent of returns, so every cell is
null; the 16 cells share the labels and the baseline and are therefore dependent.

Planted signal. All procedures are location-equivariant, so a signal is a constant
added to a series: ``delta`` long-run standard errors of the mean (``delta = 3``), in
one cell (sparse) or four cells (dense) of the 16.

Decision rule, fixed before the full-scale run (see ``holds_size``): a procedure holds
size when no error rate at protocol scale (the per-test size at 0.05, 0.10 and q / 16,
the probability of any Benjamini-Hochberg or Holm rejection under the global null, and
the false-discovery rate with planted cells) exceeds its nominal level by more than
three Monte Carlo standard errors. About a hundred rates are compared per procedure,
so the full-scale study is run twice with independent seeds and an excess counts only
when the same rate of the same configuration exceeds three standard errors in both
runs. Among the procedures that hold size the one with the highest dense-scenario
power is preferred, and a deterministic test is preferred over a resampled one when
their powers differ by less than two standard errors. The 60-day configurations
reproduce a reviewer's finding and are a check of the harness, not part of the rule.

Run ``python -m tools.issue1146_significance_calibration --help``. ``--scale reduced``
(500 series replicates, 500 structural, 200 reviewer-construction) takes about two
minutes on 26 worker processes and is only a smoke of the table; ``--scale full``
(10000 series replicates, 4000 structural, 2000 reviewer-construction) takes 15 to 25
minutes on 26 workers and is the run whose table is recorded in the SD0 section of
``docs/architecture/research-assurance.md``: ``--seed`` 20261111 (the default) and
20261112, written with ``--out``. The result does not depend on ``--workers``.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from trade_rl.evaluation.signal_measurement import (
    bh_rejections,
    block_mean_interval,
    daily_ic,
    ewc_terms,
    student_t_sf,
)

CELLS = 16
BH_Q = 0.10
HOLM_ALPHA = 0.05
BLOCK_DAYS = 20
RESAMPLES = 2000
BOOTSTRAP_SEED = 0
LABEL_OVERLAP_DAYS = 1
# MA(1) coefficient whose lag-1 autocorrelation is 0.25 (see the module docstring).
OVERLAP_THETA = 2.0 - math.sqrt(3.0)
PLANTED_DELTA = 3.0
SPARSE_CELLS = 1
DENSE_CELLS = 4
SIZE_LEVELS = (0.05, 0.10, BH_Q / CELLS)
STUDY_SEED = 20261111
SIZE_TOLERANCE_SE = 3.0

PROCEDURES = (
    "v3_centred_mbb",
    "studentized_mbb",
    "newey_west_rule",
    "newey_west_block",
    "ewc_t",
)

_HOURS = 24
_SYMBOLS = 5
_ROWS_PER_DAY = _HOURS * _SYMBOLS
_WARMUP_DAYS = 90
_DRIFT_UPDATE_DAYS = 30
_FAMILY_PERSISTENCE = (0.0, 0.5, 0.9, 0.97, 0.99, 0.995, 0.998, 0.999)


# ----------------------------------------------------------------- reference laws


def _normal_sf(z: np.ndarray) -> np.ndarray:
    erfc = np.frompyfunc(math.erfc, 1, 1)
    return 0.5 * erfc(z / math.sqrt(2.0)).astype(np.float64)


def _student_t_sf(t: np.ndarray, dof: int) -> np.ndarray:
    sf = np.frompyfunc(lambda value: student_t_sf(float(value), dof), 1, 1)
    return sf(t).astype(np.float64)


# ------------------------------------------------------------------ the procedures


@dataclass(frozen=True)
class BootstrapPlan:
    """Index stream of ``moving_block_mean_test`` for one length, as count matrices."""

    n: int
    block: int
    blocks: int
    resamples: int
    usage: np.ndarray  # (resamples, n): how often each day enters a resample
    starts: np.ndarray  # (resamples, n): how often each day starts a block


@lru_cache(maxsize=8)
def bootstrap_plan(n: int, block: int, resamples: int, seed: int) -> BootstrapPlan:
    """Reproduce the scalar draws of the repository bootstrap as one integer array.

    The real function draws ``ceil(n / block)`` block starts per resample, one scalar
    ``rng.integers(0, n)`` at a time, and keeps the first ``n`` indices. An array
    draw of the same shape from the same seed is the same stream.
    """

    size = min(block, n)
    blocks = math.ceil(n / size)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(resamples, blocks))
    index = ((draws[:, :, None] + np.arange(size)) % n).reshape(resamples, -1)[:, :n]
    usage = np.zeros((resamples, n), dtype=np.float64)
    starts = np.zeros((resamples, n), dtype=np.float64)
    rows = np.arange(resamples)[:, None]
    np.add.at(usage, (rows, index), 1.0)
    np.add.at(starts, (rows, draws), 1.0)
    return BootstrapPlan(n, size, blocks, resamples, usage, starts)


def _one_sided(count: np.ndarray, observed: np.ndarray, resamples: int) -> np.ndarray:
    p = (count + 1.0) / (resamples + 1.0)
    return np.where(observed > 0.0, p, 1.0)


def v3_centred_mbb_p(
    x: np.ndarray, shifts: Sequence[np.ndarray], plan: BootstrapPlan
) -> list[np.ndarray]:
    """p-values of the frozen v3 procedure for ``x + shift`` (rows are series)."""

    mean = x.mean(axis=1)
    boot = x @ plan.usage.T / plan.n
    centred = boot - boot.mean(axis=1, keepdims=True)
    out = []
    for shift in shifts:
        observed = mean + shift
        count = (centred >= observed[:, None]).sum(axis=1)
        out.append(_one_sided(count, observed, plan.resamples))
    return out


def _circular_block_means(x: np.ndarray, block: int) -> np.ndarray:
    n = x.shape[1]
    extended = np.concatenate([x, x[:, : block - 1]], axis=1)
    prefix = np.concatenate(
        [np.zeros((x.shape[0], 1)), np.cumsum(extended, axis=1)], axis=1
    )
    return (prefix[:, block : block + n] - prefix[:, :n]) / block


def studentized_mbb_p(
    x: np.ndarray, shifts: Sequence[np.ndarray], plan: BootstrapPlan
) -> list[np.ndarray]:
    """Percentile-t circular block bootstrap on the same block starts.

    A resample is ``blocks`` whole circular blocks; its statistic is the mean of the
    block means minus the series mean, divided by the standard error of the block
    means. The observed statistic uses the variance of all ``n`` circular block
    means, which is the bootstrap law's own variance.
    """

    b = plan.blocks
    means = _circular_block_means(x, plan.block)
    centre = means.mean(axis=1)
    star_mean = means @ plan.starts.T / b
    star_square = (means * means) @ plan.starts.T
    star_var = np.maximum(star_square - b * star_mean**2, 0.0) / (b - 1.0)
    star_se = np.sqrt(star_var / b)
    with np.errstate(divide="ignore", invalid="ignore"):
        star_t = (star_mean - centre[:, None]) / star_se
    star_t = np.where(np.isfinite(star_t), star_t, np.inf)
    observed_se = np.sqrt(((means - centre[:, None]) ** 2).mean(axis=1) / b)
    out = []
    for shift in shifts:
        observed = centre + shift
        count = (star_t >= (observed / observed_se)[:, None]).sum(axis=1)
        out.append(_one_sided(count, observed, plan.resamples))
    return out


def newey_west_rule_lag(n: int) -> int:
    """The textbook Newey-West lag, never below the one-day label overlap."""

    return max(LABEL_OVERLAP_DAYS, math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def newey_west_p(
    x: np.ndarray, shifts: Sequence[np.ndarray], lag: int
) -> list[np.ndarray]:
    """One-sided Newey-West (Bartlett kernel) HAC t-test against the normal law."""

    n = x.shape[1]
    lag = min(lag, n - 1)
    mean = x.mean(axis=1)
    centred = x - mean[:, None]
    long_run = (centred * centred).sum(axis=1) / n
    for k in range(1, lag + 1):
        gamma = (centred[:, k:] * centred[:, :-k]).sum(axis=1) / n
        long_run = long_run + 2.0 * (1.0 - k / (lag + 1.0)) * gamma
    se = np.sqrt(np.maximum(long_run, 1e-300) / n)
    return [_normal_sf((mean + shift) / se) for shift in shifts]


@lru_cache(maxsize=8)
def _cosine_basis(n: int, terms: int) -> np.ndarray:
    t = np.arange(1, n + 1, dtype=np.float64)[:, None]
    j = np.arange(1, terms + 1, dtype=np.float64)[None, :]
    return math.sqrt(2.0 / n) * np.cos(math.pi * j * (t - 0.5) / n)


def ewc_t_p(x: np.ndarray, shifts: Sequence[np.ndarray]) -> list[np.ndarray]:
    """One-sided EWC t-test: Student-t with ``ewc_terms(n)`` degrees of freedom."""

    n = x.shape[1]
    terms = ewc_terms(n)
    mean = x.mean(axis=1)
    if terms < 1:
        return [np.ones(x.shape[0]) for _ in shifts]
    projections = (x - mean[:, None]) @ _cosine_basis(n, terms)
    long_run = (projections * projections).mean(axis=1)
    se = np.sqrt(long_run / n)
    out = []
    for shift in shifts:
        observed = mean + shift
        usable = (observed > 0.0) & (long_run > 0.0) & (np.ptp(x, axis=1) > 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            p = _student_t_sf(np.where(usable, observed / se, 0.0), terms)
        out.append(np.where(usable, p, 1.0))
    return out


def procedure_p_values(
    x: np.ndarray,
    shift: np.ndarray,
    *,
    block: int,
    resamples: int,
    seed: int,
    procedures: Sequence[str] = PROCEDURES,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Null and shifted p-values of every procedure for the rows of ``x``."""

    n = x.shape[1]
    zero = np.zeros(x.shape[0])
    shifts = (zero, shift)
    plan = bootstrap_plan(n, block, resamples, seed)
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in procedures:
        if name == "v3_centred_mbb":
            null, planted = v3_centred_mbb_p(x, shifts, plan)
        elif name == "studentized_mbb":
            null, planted = studentized_mbb_p(x, shifts, plan)
        elif name == "newey_west_rule":
            null, planted = newey_west_p(x, shifts, newey_west_rule_lag(n))
        elif name == "newey_west_block":
            null, planted = newey_west_p(x, shifts, block)
        elif name == "ewc_t":
            null, planted = ewc_t_p(x, shifts)
        else:
            raise ValueError(f"unknown procedure {name!r}")
        out[name] = (null, planted)
    return out


# ------------------------------------------------------------------ null generators


@dataclass(frozen=True)
class Config:
    """One cell of the calibration grid."""

    name: str
    kind: str  # "series" or "structural"
    n: int
    phi: float = 0.0
    overlap: bool = False
    heavy: bool = False
    cross_rho: float = 0.0
    cells: int = CELLS
    block: int = BLOCK_DAYS
    resamples: int = RESAMPLES
    seed_per_replicate: bool = False

    @property
    def theta(self) -> float:
        return OVERLAP_THETA if self.overlap else 0.0

    @property
    def long_run_sd(self) -> float:
        """Long-run standard deviation of a series-level null with unit innovations."""

        return (1.0 + self.theta) / (1.0 - self.phi)


def series_increments(
    rng: np.random.Generator, replicates: int, config: Config
) -> np.ndarray:
    """``(replicates, cells, n)`` mean-zero increments of the series-level model."""

    burn = 50
    total = config.n + burn + 1
    shape = (replicates, config.cells, total)
    rho = config.cross_rho
    noise = rng.standard_normal(shape)
    if rho > 0.0:
        common = rng.standard_normal((replicates, 1, total))
        noise = math.sqrt(rho) * common + math.sqrt(1.0 - rho) * noise
    if config.heavy:
        scale_shape = (replicates, 1, total) if rho > 0.0 else shape
        mixing = rng.chisquare(4.0, size=scale_shape)
        # Student-t with 4 degrees of freedom has variance 2; rescale to unit variance.
        noise = noise * np.sqrt(4.0 / mixing) / math.sqrt(2.0)
    u = np.empty(shape)
    u[..., 0] = noise[..., 0] / math.sqrt(1.0 - config.phi**2)
    for t in range(1, total):
        u[..., t] = config.phi * u[..., t - 1] + noise[..., t]
    x = u[..., 1:] + config.theta * u[..., :-1]
    return x[..., burn:]


def _ranks(values: np.ndarray) -> np.ndarray:
    """0-based ranks along the last axis (continuous values, so no ties)."""

    order = np.argsort(values, axis=-1, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(values.shape[-1]), axis=-1)
    return ranks.astype(np.float64)


@dataclass(frozen=True)
class StructuralPanel:
    """Row-major ``(day, hour, symbol)`` arrays of one synthetic market."""

    labels: np.ndarray  # (days, rows_per_day)
    forecasts: np.ndarray  # (cells, days, rows_per_day)
    baseline: np.ndarray  # (days, rows_per_day)


def structural_panel(
    rng: np.random.Generator, n_days: int, *, heavy: bool
) -> StructuralPanel:
    """A null market: forecasts independent of returns, labels overlapping 24 bars."""

    total_hours = (_WARMUP_DAYS + n_days + 1) * _HOURS

    def draw(shape: tuple[int, ...]) -> np.ndarray:
        values = rng.standard_normal(shape)
        if heavy:
            values = (
                values * np.sqrt(4.0 / rng.chisquare(4.0, size=shape)) / math.sqrt(2.0)
            )
        return values

    returns = math.sqrt(0.6) * draw((total_hours, 1)) + math.sqrt(0.4) * draw(
        (total_hours, _SYMBOLS)
    )
    if heavy:
        # Volatility clustering: a persistent daily log-volatility common to all symbols.
        days = total_hours // _HOURS
        shocks = 0.25 * rng.standard_normal(days)
        log_vol = np.empty(days)
        log_vol[0] = shocks[0] / math.sqrt(1.0 - 0.95**2)
        for day in range(1, days):
            log_vol[day] = 0.95 * log_vol[day - 1] + shocks[day]
        returns = returns * np.exp(np.repeat(log_vol, _HOURS))[:, None]
    cumulative = np.concatenate([np.zeros((1, _SYMBOLS)), np.cumsum(returns, axis=0)])
    decisions = _WARMUP_DAYS * _HOURS + np.arange(n_days * _HOURS)
    # Label of a decision at bar h: the sum of the next 24 hourly returns.
    labels = cumulative[decisions + _HOURS + 1] - cumulative[decisions + 1]

    # Drift baseline: expanding mean of each symbol's past returns, refreshed every
    # 30 days from the bars closed at the refresh.
    day = np.arange(n_days)
    refresh = (_WARMUP_DAYS + (day // _DRIFT_UPDATE_DAYS) * _DRIFT_UPDATE_DAYS) * _HOURS
    drift = cumulative[refresh] / refresh[:, None]
    baseline = np.repeat(drift[:, None, :], _HOURS, axis=1)

    families = len(_FAMILY_PERSISTENCE)
    psi = np.asarray(_FAMILY_PERSISTENCE)[:, None, None]
    innovations = rng.standard_normal((n_days * _HOURS, families, 2, _SYMBOLS))
    scale = np.sqrt(1.0 - psi**2)
    state = innovations[0].copy()
    features = np.empty_like(innovations)
    features[0] = state
    for hour in range(1, innovations.shape[0]):
        state = psi * state + scale * innovations[hour]
        features[hour] = state
    first = features[:, :, 0, :]
    second = 0.8 * first + 0.6 * features[:, :, 1, :]
    # Cell order: families outer, models inner (the protocol's decision-cell order).
    forecasts = np.stack([first, second], axis=2).reshape(
        n_days * _HOURS, families * 2, _SYMBOLS
    )
    return StructuralPanel(
        labels=labels.reshape(n_days, _ROWS_PER_DAY),
        forecasts=np.moveaxis(forecasts, 1, 0).reshape(
            families * 2, n_days, _ROWS_PER_DAY
        ),
        baseline=baseline.reshape(n_days, _ROWS_PER_DAY),
    )


def structural_increments(panel: StructuralPanel) -> np.ndarray:
    """``(cells, days)`` pooled daily rank IC of each forecast minus the baseline's."""

    count = _ROWS_PER_DAY
    centre = 0.5 * (count - 1)
    label_ranks = _ranks(panel.labels) - centre
    forecast_ranks = _ranks(panel.forecasts) - centre
    spread = count * (count * count - 1) / 12.0
    model_ic = (forecast_ranks * label_ranks[None]).sum(axis=-1) / spread
    # The baseline is constant per symbol within a day: 24-way ties share the mean rank.
    symbol_ranks = _ranks(panel.baseline[:, :_SYMBOLS])
    tied = np.tile(_HOURS * symbol_ranks + 0.5 * (_HOURS - 1) - centre, (1, _HOURS))
    baseline_ic = (tied * label_ranks).sum(axis=-1) / np.sqrt(
        (tied * tied).sum(axis=-1) * spread
    )
    return model_ic - baseline_ic[None, :]


_PILOT_REPLICATES = 256
_PILOT_CHUNK = 16


def _pilot_chunk(task: tuple[int, bool, int, int]) -> np.ndarray:
    n_days, heavy, start, count = task
    means = np.empty((count, CELLS))
    for offset in range(count):
        rng = np.random.default_rng([STUDY_SEED, 7, int(heavy), n_days, start + offset])
        means[offset] = structural_increments(
            structural_panel(rng, n_days, heavy=heavy)
        ).mean(axis=1)
    return means


def structural_long_run_sd(
    n_days: int, heavy: bool, *, workers: int = 1
) -> tuple[float, ...]:
    """Per-cell standard deviation of ``sqrt(n) * mean`` from a fixed, seeded pilot."""

    tasks = [
        (n_days, heavy, start, min(_PILOT_CHUNK, _PILOT_REPLICATES - start))
        for start in range(0, _PILOT_REPLICATES, _PILOT_CHUNK)
    ]
    if workers <= 1:
        chunks = [_pilot_chunk(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            chunks = list(pool.map(_pilot_chunk, tasks))
    means = np.concatenate(chunks)
    return tuple(float(v) for v in means.std(axis=0, ddof=1) * math.sqrt(n_days))


# ------------------------------------------------------------------------- metrics


def bh_adjusted_rows(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values of every row."""

    count = p.shape[1]
    order = np.argsort(p, axis=1, kind="stable")
    ordered = np.take_along_axis(p, order, axis=1)
    scaled = ordered * count / np.arange(1, count + 1)
    adjusted = np.minimum(1.0, np.minimum.accumulate(scaled[:, ::-1], axis=1)[:, ::-1])
    out = np.empty_like(p)
    np.put_along_axis(out, order, adjusted, axis=1)
    return out


def _rate(values: np.ndarray) -> tuple[float, float]:
    """Mean over replicates and its Monte Carlo standard error."""

    count = values.shape[0]
    mean = float(values.mean())
    se = float(values.std(ddof=1) / math.sqrt(count)) if count > 1 else float("nan")
    return mean, se


def _harmonic(count: int) -> float:
    return sum(1.0 / k for k in range(1, count + 1))


def _planted_scenario(
    null: np.ndarray, planted: np.ndarray, cells: int, q: float
) -> dict[str, tuple[float, float]]:
    mixed = null.copy()
    mixed[:, :cells] = planted[:, :cells]
    adjusted = bh_adjusted_rows(mixed)
    out = {}
    # Benjamini-Yekutieli is Benjamini-Hochberg at q / (1 + 1/2 + ... + 1/m).
    for label, level in (("", q), ("by_", q / _harmonic(mixed.shape[1]))):
        rejected = adjusted <= level
        total = rejected.sum(axis=1)
        false = rejected[:, cells:].sum(axis=1)
        out[f"{label}power"] = _rate(rejected[:, :cells].mean(axis=1))
        out[f"{label}fdr"] = _rate(false / np.maximum(total, 1))
    return out


def summarize(null: np.ndarray, planted: np.ndarray) -> dict[str, object]:
    """Size, family error, false-discovery rate and power from ``(replicates, cells)``."""

    cells = null.shape[1]
    out: dict[str, object] = {"replicates": int(null.shape[0]), "cells": int(cells)}
    out["size"] = {
        f"{level:.5f}": _rate((null <= level).mean(axis=1)) for level in SIZE_LEVELS
    }
    adjusted = bh_adjusted_rows(null)
    out["bh_any"] = _rate((adjusted <= BH_Q).any(axis=1).astype(np.float64))
    out["by_any"] = _rate(
        (adjusted <= BH_Q / _harmonic(cells)).any(axis=1).astype(np.float64)
    )
    out["holm_any"] = _rate((null.min(axis=1) <= HOLM_ALPHA / cells).astype(np.float64))
    out["single_test_power_0.05"] = _rate((planted <= 0.05).mean(axis=1))
    out["sparse"] = _planted_scenario(null, planted, SPARSE_CELLS, BH_Q)
    out["dense"] = _planted_scenario(null, planted, DENSE_CELLS, BH_Q)
    return out


def size_excesses(summary: dict[str, object]) -> list[tuple[str, float]]:
    """Error rates above nominal, as ``(metric, excess in standard errors)``."""

    sizes: dict[str, tuple[float, float]] = summary["size"]  # type: ignore[assignment]
    sparse: dict[str, tuple[float, float]] = summary["sparse"]  # type: ignore[assignment]
    dense: dict[str, tuple[float, float]] = summary["dense"]  # type: ignore[assignment]
    checks: list[tuple[str, float, tuple[float, float]]] = [
        ("bh_any", BH_Q, summary["bh_any"]),  # type: ignore[list-item]
        ("holm_any", HOLM_ALPHA, summary["holm_any"]),  # type: ignore[list-item]
        ("fdr_sparse", BH_Q, sparse["fdr"]),
        ("fdr_dense", BH_Q, dense["fdr"]),
    ]
    for level in SIZE_LEVELS:
        checks.append((f"size@{level:.5f}", level, sizes[f"{level:.5f}"]))
    out = []
    for name, nominal, (mean, se) in checks:
        if se > 0.0 and mean > nominal:
            out.append((name, (mean - nominal) / se))
    return out


def holds_size(summary: dict[str, object]) -> bool:
    """The pre-registered criterion: no error rate is above nominal by more than 3 SE.

    The rates are the per-test size at 0.05, 0.10 and q / 16, the probability of any
    Benjamini-Hochberg or Holm rejection under the global null, and the false
    discovery rate with one and with four planted cells.
    """

    return all(excess <= SIZE_TOLERANCE_SE for _, excess in size_excesses(summary))


# --------------------------------------------------------------------------- study


def protocol_scale_configs(n: int = 1460) -> tuple[Config, ...]:
    grid = [
        Config(
            f"gauss phi={phi} overlap={'on' if overlap else 'off'}",
            "series",
            n,
            phi=phi,
            overlap=overlap,
        )
        for phi in (0.0, 0.1, 0.3)
        for overlap in (False, True)
    ]
    grid += [
        Config("t4 phi=0.1 overlap=on", "series", n, phi=0.1, overlap=True, heavy=True),
        Config("t4 phi=0.3 overlap=on", "series", n, phi=0.3, overlap=True, heavy=True),
        Config(
            "gauss phi=0.1 overlap=on rho=0.5",
            "series",
            n,
            phi=0.1,
            overlap=True,
            cross_rho=0.5,
        ),
        Config(
            "gauss phi=0.1 overlap=on rho=0.9",
            "series",
            n,
            phi=0.1,
            overlap=True,
            cross_rho=0.9,
        ),
        Config(
            "t4 phi=0.3 overlap=on rho=0.5",
            "series",
            n,
            phi=0.3,
            overlap=True,
            heavy=True,
            cross_rho=0.5,
        ),
        Config("structural gauss", "structural", n),
        Config("structural t4 vol-clustered", "structural", n, heavy=True),
    ]
    return tuple(grid)


def full_window_configs() -> tuple[Config, ...]:
    """The whole out-of-sample window (2021-04 to 2026-07) instead of 2022-2025."""

    n = 1948
    return (
        Config("n=1948 gauss phi=0.3 overlap=on", "series", n, phi=0.3, overlap=True),
        Config(
            "n=1948 t4 phi=0.3 overlap=on rho=0.5",
            "series",
            n,
            phi=0.3,
            overlap=True,
            heavy=True,
            cross_rho=0.5,
        ),
    )


def small_sample_configs() -> tuple[Config, ...]:
    return (
        # The construction of review 5844379344: 60 days, 5-day blocks, 400 resamples,
        # nine tests, a bootstrap seed that changes with the replicate.
        Config(
            "n=60 reviewer settings (iid, block 5, B=400, 9 tests)",
            "series",
            60,
            cells=9,
            block=5,
            resamples=400,
            seed_per_replicate=True,
        ),
        Config("n=60 protocol settings iid", "series", 60),
        Config(
            "n=60 protocol settings phi=0.3 overlap=on",
            "series",
            60,
            phi=0.3,
            overlap=True,
        ),
    )


def _chunk_size(config: Config) -> int:
    return 25 if config.kind == "structural" else 125


_Task = tuple[Config, int, int, int, int, tuple[float, ...]]


def _run_chunk(task: _Task) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    config, config_index, chunk_index, replicates, seed, long_run_sd = task
    rng = np.random.default_rng([seed, config_index, chunk_index])
    if config.kind == "structural":
        x = np.stack(
            [
                structural_increments(
                    structural_panel(rng, config.n, heavy=config.heavy)
                )
                for _ in range(replicates)
            ]
        )
    else:
        x = series_increments(rng, replicates, config)
    sd = np.asarray(long_run_sd)
    shift = np.tile(PLANTED_DELTA * sd / math.sqrt(config.n), replicates)
    flat = x.reshape(replicates * config.cells, config.n)
    if config.seed_per_replicate:
        parts: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
        for replicate in range(replicates):
            rows = slice(replicate * config.cells, (replicate + 1) * config.cells)
            # The reviewer's bootstrap seed is the replicate index.
            result = procedure_p_values(
                flat[rows],
                shift[rows],
                block=config.block,
                resamples=config.resamples,
                seed=chunk_index * _chunk_size(config) + replicate,
            )
            for name, pair in result.items():
                parts.setdefault(name, []).append(pair)
        merged = {
            name: (
                np.concatenate([a for a, _ in pairs]),
                np.concatenate([b for _, b in pairs]),
            )
            for name, pairs in parts.items()
        }
    else:
        merged = procedure_p_values(
            flat,
            shift,
            block=config.block,
            resamples=config.resamples,
            seed=BOOTSTRAP_SEED,
        )
    return {
        name: (
            null.reshape(replicates, config.cells),
            planted.reshape(replicates, config.cells),
        )
        for name, (null, planted) in merged.items()
    }


def run_configs(
    configs: Sequence[Config],
    *,
    replicates: int,
    structural_replicates: int,
    seed: int = STUDY_SEED,
    workers: int = 1,
) -> list[dict[str, object]]:
    """Run every configuration; the result does not depend on ``workers``."""

    tasks: list[_Task] = []
    for config_index, config in enumerate(configs):
        if config.kind == "structural":
            wanted = structural_replicates
            sd = structural_long_run_sd(config.n, config.heavy, workers=workers)
        else:
            wanted = replicates
            sd = (config.long_run_sd,) * config.cells
        size = _chunk_size(config)
        for chunk_index, start in enumerate(range(0, wanted, size)):
            count = min(size, wanted - start)
            tasks.append((config, config_index, chunk_index, count, seed, sd))
    if workers <= 1:
        chunks = [_run_chunk(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            chunks = list(pool.map(_run_chunk, tasks))
    results: list[dict[str, object]] = []
    for config_index, config in enumerate(configs):
        mine = [
            chunk
            for task, chunk in zip(tasks, chunks, strict=True)
            if task[1] == config_index
        ]
        procedures: dict[str, object] = {}
        for name in PROCEDURES:
            null = np.concatenate([chunk[name][0] for chunk in mine])
            planted = np.concatenate([chunk[name][1] for chunk in mine])
            summary = summarize(null, planted)
            summary["holds_size"] = holds_size(summary)
            summary["size_excesses_se"] = size_excesses(summary)
            procedures[name] = summary
        results.append({"config": asdict(config), "procedures": procedures})
    return results


# ----------------------------------------------------- reviewer construction, verbatim


def _reviewer_chunk(task: tuple[int, np.ndarray]) -> list[bool]:
    first_replicate, permutations = task
    days, symbols, null_features, strength = 60, 5, 8, 0.25
    rng = np.random.default_rng(5)
    rows = days * _HOURS * symbols
    signal = rng.standard_normal(rows)
    target = strength * signal + rng.standard_normal(rows)
    features = (signal, *(rng.standard_normal(rows) for _ in range(null_features)))
    day_ids = np.repeat(np.arange(days * _HOURS) // _HOURS, symbols).astype(np.int64)
    blocks = target.reshape(-1, _HOURS * symbols)
    out = []
    for offset, permutation in enumerate(permutations):
        shuffled = blocks[permutation].reshape(-1)
        p = np.array(
            [
                block_mean_interval(
                    daily_ic(day_ids, feature, shuffled, method="rank").values,
                    block_days=5,
                    resamples=400,
                    seed=first_replicate + offset,
                ).p_value_one_sided
                for feature in features
            ]
        )
        out.append(bool(bh_rejections(p, BH_Q).any()))
    return out


def reviewer_construction(replicates: int, *, workers: int = 1) -> tuple[float, float]:
    """P(any BH rejection) of the SD1 shuffled-future oracle, with the real functions.

    The panel (seed 5), the day shuffle (seed 99), the nine features, the 5-day
    blocks, the 400 resamples and the replicate-indexed bootstrap seed are those of
    ``tests/evaluation/test_signal_measurement_oracles.py``; only the replicate count
    differs.
    """

    rng = np.random.default_rng(99)
    permutations = np.stack([rng.permutation(60) for _ in range(replicates)])
    size = 25
    tasks = [
        (start, permutations[start : start + size])
        for start in range(0, replicates, size)
    ]
    if workers <= 1:
        chunks = [_reviewer_chunk(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            chunks = list(pool.map(_reviewer_chunk, tasks))
    hits = np.array([hit for chunk in chunks for hit in chunk], dtype=np.float64)
    return _rate(hits)


# -------------------------------------------------------------------------- report


def _cell(pair: object, digits: int = 3) -> str:
    mean, se = pair  # type: ignore[misc]
    return f"{mean:.{digits}f} ± {se:.{digits}f}"


def render_markdown(results: Iterable[dict[str, object]]) -> str:
    header = (
        "| configuration | procedure | size α=0.05 | size α=0.10 | size α=q/m "
        "| P(any BH), q=0.10 | P(any Holm), 0.05 | power sparse | FDR sparse "
        "| power dense | FDR dense | holds size |"
    )
    lines = [header, "|" + "---|" * 12]
    for result in results:
        config: dict[str, object] = result["config"]  # type: ignore[assignment]
        procedures: dict[str, dict[str, object]] = result["procedures"]  # type: ignore[assignment]
        for name, summary in procedures.items():
            sizes: dict[str, object] = summary["size"]  # type: ignore[assignment]
            sparse: dict[str, object] = summary["sparse"]  # type: ignore[assignment]
            dense: dict[str, object] = summary["dense"]  # type: ignore[assignment]
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"{config['name']} (R={summary['replicates']})",
                        name,
                        _cell(sizes[f"{SIZE_LEVELS[0]:.5f}"], 4),
                        _cell(sizes[f"{SIZE_LEVELS[1]:.5f}"], 4),
                        _cell(sizes[f"{SIZE_LEVELS[2]:.5f}"], 4),
                        _cell(summary["bh_any"]),
                        _cell(summary["holm_any"]),
                        _cell(sparse["power"]),
                        _cell(sparse["fdr"]),
                        _cell(dense["power"]),
                        _cell(dense["fdr"]),
                        "yes" if summary["holds_size"] else "NO",
                    ]
                )
                + " |"
            )
    return "\n".join(lines)


_SCALES = {
    # name: (series replicates, structural replicates, reviewer replicates)
    "smoke": (100, 50, 50),
    "reduced": (500, 500, 200),
    "full": (10000, 4000, 2000),
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--scale", choices=sorted(_SCALES), default="reduced")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=STUDY_SEED)
    parser.add_argument("--out", type=Path, default=None, help="write the JSON here")
    args = parser.parse_args(argv)
    replicates, structural, reviewer = _SCALES[args.scale]
    configs = (
        *protocol_scale_configs(),
        *full_window_configs(),
        *small_sample_configs(),
    )
    results = run_configs(
        configs,
        replicates=replicates,
        structural_replicates=structural,
        seed=args.seed,
        workers=args.workers,
    )
    reviewer_rate = reviewer_construction(reviewer, workers=args.workers)
    document = {
        "schema": "issue1146_significance_calibration_v1",
        "scale": args.scale,
        "seed": args.seed,
        "synthetic_only": True,
        "planted_delta": PLANTED_DELTA,
        "reviewer_construction": {
            "replicates": reviewer,
            "p_any_bh_rejection": reviewer_rate,
        },
        "results": results,
    }
    if args.out is not None:
        text = json.dumps(document, indent=2, sort_keys=True) + "\n"
        args.out.write_bytes(text.encode("utf-8"))
    print(render_markdown(results))
    print()
    print(
        "Reviewer construction (real SD1 functions), P(any BH rejection): "
        + _cell(reviewer_rate)
        + f" over {reviewer} replicates"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
