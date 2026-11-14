"""Cell statistics of the signal diagnosis (SD2; layers D0, D1, D2, D4, D5).

Decision cells use the SD1 ``ewc_mean_test``, ``block_mean_interval`` and
``drift_excess_payoff``. The daily rank IC and its increment are computed by a
vectorised grouped implementation, because D2 evaluates 167 features at three horizons
and two transforms; tests pin it equal to the SD1 ``daily_ic`` and
``daily_rank_ic_increment`` (ties and skipped days included).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from trade_rl.evaluation.signal_diagnosis.protocol import SignalDiagnosisProtocol
from trade_rl.evaluation.signal_diagnosis.verdict import (
    DecisionCellEvidence,
    PValueProvenance,
    YearStabilityEvidence,
)
from trade_rl.evaluation.signal_diagnosis_run.fitting import (
    Labels,
    Vintage,
)
from trade_rl.evaluation.signal_diagnosis_run.panel import (
    NS_PER_DAY,
    NS_PER_HOUR,
    SignalPanel,
)
from trade_rl.evaluation.signal_measurement import (
    bh_adjusted,
    block_mean_interval,
    daily_mean,
    drift_excess_payoff,
    ewc_mean_test,
    holm_adjusted,
    market_mean,
    nonstationarity_profile,
)

# ------------------------------------------------------------------ grouped IC


def _grouped_average_ranks(day: np.ndarray, values: np.ndarray) -> np.ndarray:
    """1-based ranks within each (contiguous, ascending) day, ties share the mean."""

    n = values.size
    order = np.lexsort((values, day))
    sorted_values, sorted_day = values[order], day[order]
    new_day = np.concatenate(([True], sorted_day[1:] != sorted_day[:-1]))
    new_group = new_day | np.concatenate(
        ([True], sorted_values[1:] != sorted_values[:-1])
    )
    position = np.arange(n)
    day_first = np.maximum.accumulate(np.where(new_day, position, 0))
    group_first = np.maximum.accumulate(np.where(new_group, position, 0))
    group_id = np.cumsum(new_group) - 1
    group_last = np.concatenate((position[new_group][1:], [n])) - 1
    average = (group_first + group_last[group_id]) / 2.0 - day_first + 1.0
    ranks = np.empty(n)
    ranks[order] = average
    return ranks


def grouped_pearson(
    day: np.ndarray, left: np.ndarray, right: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-day Pearson correlation; ``(day ids, values)`` for days where it exists."""

    if day.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0)
    starts = np.flatnonzero(np.concatenate(([True], day[1:] != day[:-1])))
    counts = np.diff(np.concatenate((starts, [day.size])))
    centred_left = left - np.repeat(np.add.reduceat(left, starts) / counts, counts)
    centred_right = right - np.repeat(np.add.reduceat(right, starts) / counts, counts)
    cross = np.add.reduceat(centred_left * centred_right, starts)
    var_left = np.add.reduceat(centred_left**2, starts)
    var_right = np.add.reduceat(centred_right**2, starts)
    denominator = np.sqrt(var_left * var_right)
    keep = (counts >= 2) & (denominator > 0.0) & np.isfinite(denominator)
    value = np.clip(cross[keep] / denominator[keep], -1.0, 1.0)
    return day[starts][keep], value


def grouped_ic(
    day: np.ndarray, scores: np.ndarray, target: np.ndarray, *, method: str
) -> tuple[np.ndarray, np.ndarray]:
    """Daily IC over rows already restricted to finite pairs, sorted by day."""

    if method == "rank":
        scores = _grouped_average_ranks(day, scores)
        target = _grouped_average_ranks(day, target)
    return grouped_pearson(day, scores, target)


# ------------------------------------------------------------------- row sets


def utc_day(timestamps_ns: np.ndarray) -> np.ndarray:
    return (timestamps_ns // NS_PER_DAY).astype(np.int64)


def _year_month(day_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    days = day_ids.astype("datetime64[D]")
    years = days.astype("datetime64[Y]").astype(np.int64) + 1970
    months = days.astype("datetime64[M]").astype(np.int64)
    return years, months


@dataclass(frozen=True, slots=True)
class RowSet:
    """Decision rows of one cell, flattened time-major."""

    day: np.ndarray
    time_idx: np.ndarray
    symbol: np.ndarray
    forecast: np.ndarray
    drift: np.ndarray
    raw: np.ndarray
    demeaned: np.ndarray


def decision_rows(
    forecast: np.ndarray, drift: np.ndarray, labels: Labels, timestamps_ns: np.ndarray
) -> RowSet:
    mask = np.isfinite(forecast) & np.isfinite(drift) & np.isfinite(labels.raw)
    t_idx, s_idx = np.nonzero(mask)
    return RowSet(
        utc_day(timestamps_ns[t_idx]),
        t_idx,
        s_idx,
        forecast[t_idx, s_idx],
        drift[t_idx, s_idx],
        labels.raw[t_idx, s_idx],
        labels.demeaned[t_idx, s_idx],
    )


def grouped_rank_ic_increment(
    group: np.ndarray, forecast: np.ndarray, baseline: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    """Daily rank IC of the forecast minus that of the baseline on identical rows.

    The vectorised twin of the SD1 ``daily_rank_ic_increment`` (a test pins them equal,
    ties and skipped days included): rows are the ones where all three are finite, and a
    group on which either IC is undefined is skipped and counted.
    """

    common = np.isfinite(forecast) & np.isfinite(baseline) & np.isfinite(target)
    keep = np.flatnonzero(common)
    order = keep[np.argsort(group[keep], kind="stable")]
    g = group[order]
    groups_total = int(np.unique(g).size)
    model_days, model_ic = grouped_ic(g, forecast[order], target[order], method="rank")
    base_days, base_ic = grouped_ic(g, baseline[order], target[order], method="rank")
    both, at_model, at_base = np.intersect1d(
        model_days, base_days, assume_unique=True, return_indices=True
    )
    return both, model_ic[at_model] - base_ic[at_base], groups_total - int(both.size)


def _series(rows: RowSet, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    return grouped_rank_ic_increment(rows.day, rows.forecast, rows.drift, target)


def _mean(values: np.ndarray) -> float | None:
    return float(values.mean()) if values.size else None


def _yearly(
    days: np.ndarray, values: np.ndarray, min_days: int
) -> dict[int, dict[str, Any]]:
    years, months = _year_month(days)
    out: dict[int, dict[str, Any]] = {}
    for year in sorted({int(y) for y in years}):
        sel = years == year
        month_ids, counts = np.unique(months[sel], return_counts=True)
        out[year] = {
            "n_days": int(sel.sum()),
            "mean_increment": float(values[sel].mean()),
            "covered_months": int((counts >= min_days).sum()),
            "n_months_with_days": int(month_ids.size),
        }
    return out


# ------------------------------------------------------------- decision cells


@dataclass(frozen=True, slots=True)
class CellMeasurement:
    """Everything D3/D5 says about one cell, plus the verdict-ready evidence."""

    family_id: str
    model_id: str
    record: dict[str, Any]
    evidence: DecisionCellEvidence


def measure_cell(
    family_id: str,
    model_id: str,
    rows: RowSet,
    protocol: SignalDiagnosisProtocol,
    *,
    close_symbols: tuple[str, ...],
    regime: np.ndarray,
) -> CellMeasurement:
    """Primary-horizon decision statistics and D5 slices of one decision cell."""

    stability = protocol.stability
    raw_days, raw_values, raw_skipped = _series(rows, rows.raw)
    dem_days, dem_values, dem_skipped = _series(rows, rows.demeaned)
    raw_test = ewc_mean_test(raw_values)
    dem_test = ewc_mean_test(dem_values)
    yearly_dem = _yearly(dem_days, dem_values, stability.covered_month_min_valid_days)
    yearly_raw = _yearly(raw_days, raw_values, stability.covered_month_min_valid_days)
    roster = tuple(
        YearStabilityEvidence(
            year=year,
            covered_months=int(yearly_dem.get(year, {}).get("covered_months", 0)),
            increment_positive=bool(
                yearly_dem.get(year, {}).get("mean_increment", 0.0) > 0.0
            ),
        )
        for year in stability.full_years
    )
    evidence = DecisionCellEvidence(
        family_id=family_id,
        model_id=model_id,
        rank_ic_vs_drift=_mean(raw_values) or 0.0,
        rank_ic_p_value=raw_test.p_value_one_sided,
        demeaned_rank_ic=_mean(dem_values) or 0.0,
        demeaned_p_value=dem_test.p_value_one_sided,
        p_value_provenance=PValueProvenance(
            procedure_id=protocol.significance.procedure_id,
            raw_n_days=int(raw_values.size),
            demeaned_n_days=int(dem_values.size),
        ),
        year_evidence=roster,
        drift_excess_ci_lower_cost_x1=None,
        drift_excess_ci_upper_cost_x1=None,
        drift_excess_point_cost_x2=None,
    )
    record: dict[str, Any] = {
        "cell_id": evidence.cell_id,
        "status": "measured",
        "n_decision_rows": int(rows.day.size),
        "raw": {
            "n_days": int(raw_values.size),
            "skipped_days": raw_skipped,
            "mean_rank_ic_increment": evidence.rank_ic_vs_drift,
            "ewc": _ewc_record(raw_test),
        },
        "demeaned": {
            "n_days": int(dem_values.size),
            "skipped_days": dem_skipped,
            "mean_rank_ic_increment": evidence.demeaned_rank_ic,
            "ewc": _ewc_record(dem_test),
        },
        "context": _context(rows),
        "d5": {
            "decisive_year_roster": [
                {
                    "year": item.year,
                    "covered_months": item.covered_months,
                    "increment_positive": item.increment_positive,
                }
                for item in roster
            ],
            "year": {
                "demeaned_target": {str(k): v for k, v in yearly_dem.items()},
                "raw_target": {str(k): v for k, v in yearly_raw.items()},
                "partial_years_reported_only": list(
                    stability.partial_years_reported_only
                ),
            },
            "symbol": _symbol_slice(rows, close_symbols),
            "regime": _regime_slice(rows, regime),
            "extreme_day_exclusion": _extreme_slice(
                rows,
                raw_days,
                raw_values,
                dem_days,
                dem_values,
                stability.extreme_day_trim_fraction_per_tail,
            ),
        },
        "d4": {"status": "not_computed_upper_layer_stopped"},
    }
    return CellMeasurement(family_id, model_id, record, evidence)


def _ewc_record(test: Any) -> dict[str, Any]:
    return {
        "n_days": test.n_days,
        "cosine_terms": test.cosine_terms,
        "mean": test.mean,
        "standard_error": test.standard_error,
        "statistic": test.statistic,
        "p_value_one_sided": test.p_value_one_sided,
    }


def _context(rows: RowSet) -> dict[str, Any]:
    """Reported context: MSE skill against the zero and drift forecasts (raw target)."""

    err_model = float(np.mean((rows.raw - rows.forecast) ** 2))
    err_zero = float(np.mean(rows.raw**2))
    err_drift = float(np.mean((rows.raw - rows.drift) ** 2))
    return {
        "mse_skill_vs_zero": None if err_zero == 0.0 else 1.0 - err_model / err_zero,
        "mse_skill_vs_drift": None if err_drift == 0.0 else 1.0 - err_model / err_drift,
        "role": "reported_context_never_decisive",
    }


def _symbol_slice(rows: RowSet, symbols: tuple[str, ...]) -> dict[str, Any]:
    """Within one symbol the drift forecast is constant on a day, so the increment is
    undefined by construction; the forecast's own mean daily rank IC is reported."""

    out: dict[str, Any] = {}
    for index, name in enumerate(symbols):
        sel = rows.symbol == index
        _, values = grouped_ic(
            rows.day[sel], rows.forecast[sel], rows.raw[sel], method="rank"
        )
        out[name] = {
            "mean_forecast_rank_ic": _mean(values),
            "n_days": int(values.size),
        }
    return {"statistic": "forecast_rank_ic_within_symbol_raw_target", "symbols": out}


def _regime_slice(rows: RowSet, regime: np.ndarray) -> dict[str, Any]:
    sign = regime[rows.time_idx, rows.symbol]
    known = np.isfinite(sign)
    group = rows.day[known] * 2 + (sign[known] >= 0.0).astype(np.int64)
    out: dict[str, Any] = {"definition": "sign of the causal trailing market return"}
    for name, target in (("raw", rows.raw[known]), ("demeaned", rows.demeaned[known])):
        ids, values, _ = grouped_rank_ic_increment(
            group, rows.forecast[known], rows.drift[known], target
        )
        out[name] = {
            "non_negative_regime": _mean(values[ids % 2 == 1]),
            "negative_regime": _mean(values[ids % 2 == 0]),
            "n_groups_non_negative": int((ids % 2 == 1).sum()),
            "n_groups_negative": int((ids % 2 == 0).sum()),
        }
    return out


def _extreme_slice(
    rows: RowSet,
    raw_days: np.ndarray,
    raw_values: np.ndarray,
    dem_days: np.ndarray,
    dem_values: np.ndarray,
    fraction: float,
) -> dict[str, Any]:
    days, inverse = np.unique(rows.day, return_inverse=True)
    mean_return = np.bincount(inverse, rows.raw) / np.bincount(inverse)
    order = np.lexsort((days, mean_return))
    trim = int(np.floor(fraction * days.size))
    excluded = (
        days[np.concatenate((order[:trim], order[days.size - trim :]))]
        if trim
        else days[:0]
    )
    out: dict[str, Any] = {"days_excluded": int(excluded.size)}
    for name, ids, values in (
        ("raw", raw_days, raw_values),
        ("demeaned", dem_days, dem_values),
    ):
        keep = ~np.isin(ids, excluded)
        out[name] = {
            "mean_rank_ic_increment": _mean(values[keep]),
            "n_days": int(keep.sum()),
        }
    return out


# ------------------------------------------------------------------------ D4


def d4_record(
    rows: RowSet,
    protocol: SignalDiagnosisProtocol,
) -> dict[str, Any]:
    """After-cost payoff relative to the zero-cost drift-sign policy (D4).

    Every (k, cost multiplier) of the grid is computed for a cell that reached D4; the
    decisive numbers are k = primary, interval at the tradability multiplier and point
    at the stress multiplier.
    """

    costs = protocol.costs
    unit = costs.round_trip_cost_bps / 10_000.0
    block = protocol.bootstrap
    grid: dict[str, Any] = {}
    for k in costs.k_grid:
        for multiplier in costs.cost_multipliers:
            payoff = drift_excess_payoff(
                rows.forecast,
                rows.drift,
                rows.raw,
                k=k,
                round_trip_cost=unit,
                cost_multiplier=multiplier,
            )
            daily = daily_mean(rows.day, payoff)
            result = block_mean_interval(
                np.asarray(daily.values),
                block_days=block.block_days,
                resamples=block.resamples,
                seed=block.seed,
            )
            traded = np.abs(rows.forecast) > k * multiplier * unit
            grid[f"k{k}_x{multiplier}"] = {
                "k": k,
                "cost_multiplier": multiplier,
                "n_days": result.n_days,
                "mean": result.mean,
                "ci_lower": result.lower,
                "ci_upper": result.upper,
                "share_of_rows_traded": float(traded.mean()),
            }
    rules = protocol.decision_rules
    decisive_x1 = grid[f"k{rules.primary_k}_x{rules.tradability_cost_multiplier}"]
    decisive_x2 = grid[f"k{rules.primary_k}_x{rules.stress_cost_multiplier}"]
    return {
        "status": "computed",
        "quantity": "drift_excess_payoff (not a beta-adjusted alpha)",
        "grid": grid,
        "decisive": {
            "ci_lower_cost_x1": decisive_x1["ci_lower"],
            "ci_upper_cost_x1": decisive_x1["ci_upper"],
            "point_cost_x2": decisive_x2["mean"],
        },
    }


# ------------------------------------------------------------------------ D0/D1


def regime_signal(close: np.ndarray, lookback_days: int) -> np.ndarray:
    """Trailing cross-sectional mean log return over closed bars, per row (NaN early)."""

    bars = lookback_days * 24
    out = np.full(close.shape[0], np.nan)
    if close.shape[0] > bars:
        out[bars:] = np.log(close[bars:] / close[:-bars]).mean(axis=1)
    return np.repeat(out[:, None], close.shape[1], axis=1)


def d0_profile(
    panel: SignalPanel, *, clip_abs: float, min_train_rows: int
) -> dict[str, Any]:
    """Per (symbol, feature, calendar year) z-score distribution against the past."""

    years, _ = _year_month(utc_day(panel.timestamps_ns))
    out: dict[str, Any] = {"period": "calendar_year_utc", "symbols": {}}
    for index, symbol in enumerate(panel.symbols):
        values = np.where(
            panel.feature_available[:, index, :], panel.features[:, index, :], np.nan
        ).astype(np.float64)
        values[~np.isfinite(values)] = np.nan
        profile = nonstationarity_profile(
            values,
            years,
            panel.feature_names,
            min_train_rows=min_train_rows,
            clip_abs=clip_abs,
        )
        out["symbols"][symbol] = {
            "profiles": [
                {
                    "feature": p.feature,
                    "period": p.period,
                    "n_train": p.n_train,
                    "n_eval": p.n_eval,
                    "z_mean": p.z_mean,
                    "z_std": p.z_std,
                    "z_p05": p.z_p05,
                    "z_p50": p.z_p50,
                    "z_p95": p.z_p95,
                    "share_abs_z_above_clip": p.share_abs_z_above_clip,
                }
                for p in profile.profiles
            ],
            "skipped": [
                {"feature": s.feature, "period": s.period, "reason": s.reason}
                for s in profile.skipped
            ],
        }
    return out


def d1_beta(
    labels: Labels,
    vintages: tuple[Vintage, ...],
    symbols: tuple[str, ...],
    timestamps_ns: np.ndarray,
    min_rows: int,
) -> dict[str, Any]:
    """Reported context: per-vintage slope of matured returns on the market mean."""

    h = labels.horizon
    market = market_mean(np.where(labels.valid, labels.raw, np.nan))
    out: dict[str, list[float | None]] = {s: [] for s in symbols}
    for vintage in vintages:
        last = int(np.searchsorted(timestamps_ns, vintage.cutoff_ns, side="right")) - 1
        limit = last - h
        for index, symbol in enumerate(symbols):
            if limit < 0:
                out[symbol].append(None)
                continue
            x = market[: limit + 1]
            y = np.where(
                labels.valid[: limit + 1, index], labels.raw[: limit + 1, index], np.nan
            )
            ok = np.isfinite(x) & np.isfinite(y)
            if int(ok.sum()) < min_rows or float(np.var(x[ok])) == 0.0:
                out[symbol].append(None)
                continue
            xc = x[ok] - x[ok].mean()
            out[symbol].append(float(xc @ (y[ok] - y[ok].mean()) / (xc @ xc)))
    return {
        "role": "reported_context_never_decisive",
        "cutoffs_ns": [v.cutoff_ns for v in vintages],
        "beta_by_symbol": out,
    }


# ------------------------------------------------------------------------ D2


def _standardised_feature(
    x: np.ndarray,
    ok: np.ndarray,
    labels: Labels,
    vintages: tuple[Vintage, ...],
    clip: float,
) -> np.ndarray:
    """Train-window standardise and clip, vintage by vintage, from cumulative sums."""

    if not ok.any():
        return np.full(x.shape, np.nan)
    shifted = x - float(x[ok].mean())
    used = ok & labels.valid
    v = np.where(used, shifted, 0.0)
    total, squares = np.cumsum(v, axis=0), np.cumsum(v * v, axis=0)
    count = np.cumsum(used, axis=0)
    z = np.full(x.shape, np.nan)
    for vintage in vintages:
        limit = vintage.cut_idx - labels.horizon - 1
        if limit < 0:
            continue
        n = count[limit]
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = total[limit] / n
            std = np.sqrt(np.maximum(squares[limit] / n - mean**2, 0.0))
        scale = np.where(std > 1e-12, std, 1.0)
        mean = np.where(n > 0, mean, 0.0)
        seg = slice(vintage.first_row, vintage.stop_row)
        z[seg] = np.where(
            ok[seg], np.clip((shifted[seg] - mean) / scale, -clip, clip), np.nan
        )
    return z


def d2_features(
    panel: SignalPanel,
    protocol: SignalDiagnosisProtocol,
    labels_by_horizon: dict[int, Labels],
    vintages: tuple[Vintage, ...],
    *,
    names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Fit-free feature-by-horizon-by-transform rank and linear IC with tests.

    ``names`` restricts the features (a test affordance); the runner passes none and
    so evaluates every protocol feature.
    """

    clip = next(t.clip_abs for t in protocol.transforms if t.standardise)
    costs = protocol.costs
    unit = costs.round_trip_cost_bps / 10_000.0
    ts = panel.timestamps_ns
    first_row = vintages[0].first_row
    names = names or tuple(
        n for family in protocol.families for n in family.feature_names
    )
    available_now = panel.available_at_ns <= ts[:, None]
    records: list[dict[str, Any]] = []
    for name in names:
        f = panel.feature_names.index(name)
        x = panel.features[:, :, f].astype(np.float64)
        ok = panel.feature_available[:, :, f] & np.isfinite(x) & available_now
        ok[:first_row] = False
        for horizon, labels in labels_by_horizon.items():
            standardised = _standardised_feature(x, ok, labels, vintages, clip)
            for transform in protocol.transforms:
                z = standardised if transform.standardise else np.where(ok, x, np.nan)
                t_idx, s_idx = np.nonzero(np.isfinite(z) & labels.valid)
                if t_idx.size == 0:
                    records.append(_d2_empty(name, horizon, transform.transform_id))
                    continue
                day = utc_day(ts[t_idx])
                score, raw = z[t_idx, s_idx], labels.raw[t_idx, s_idx]
                dem = labels.demeaned[t_idx, s_idx]
                record = _d2_record(
                    name,
                    horizon,
                    transform.transform_id,
                    day,
                    score,
                    raw,
                    dem,
                    unit,
                    costs.cost_multipliers,
                )
                records.append(record)
    _adjust_d2(records)
    return {
        "role": "reported_not_decisive",
        "families": ["d2_feature_cells"],
        "rows": records,
    }


def _d2_empty(name: str, horizon: int, transform_id: str) -> dict[str, Any]:
    return {
        "feature": name,
        "horizon_bars": horizon,
        "transform": transform_id,
        "n_rows": 0,
    }


def _d2_record(
    name: str,
    horizon: int,
    transform_id: str,
    day: np.ndarray,
    score: np.ndarray,
    raw: np.ndarray,
    dem: np.ndarray,
    unit: float,
    multipliers: tuple[int, ...],
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "feature": name,
        "horizon_bars": horizon,
        "transform": transform_id,
        "n_rows": int(day.size),
    }
    for target_name, target in (("raw", raw), ("demeaned", dem)):
        keep = np.isfinite(target)
        d, s, t = day[keep], score[keep], target[keep]
        rank_days, rank_values = grouped_ic(d, s, t, method="rank")
        _, linear_values = grouped_ic(d, s, t, method="linear")
        positive = ewc_mean_test(rank_values)
        negative = ewc_mean_test(-rank_values)
        out[target_name] = {
            "n_days": int(rank_values.size),
            "mean_rank_ic": _mean(rank_values),
            "mean_linear_ic": _mean(linear_values),
            "p_positive_direction": positive.p_value_one_sided,
            "p_negative_direction": negative.p_value_one_sided,
        }
    position = np.sign(score)
    payoff: dict[str, float | None] = {}
    for multiplier in multipliers:
        row_payoff = position * raw - np.abs(position) * unit * multiplier
        daily = daily_mean(day, row_payoff)
        payoff[f"x{multiplier}"] = _mean(np.asarray(daily.values))
    out["sign_payoff_mean_daily"] = payoff
    return out


def _adjust_d2(records: list[dict[str, Any]]) -> None:
    """Holm and BH within (horizon, transform, target, direction) over the features."""

    groups: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for record in records:
        if record["n_rows"]:
            groups.setdefault((record["horizon_bars"], record["transform"]), []).append(
                record
            )
    for members in groups.values():
        for target in ("raw", "demeaned"):
            for direction in ("positive", "negative"):
                key = f"p_{direction}_direction"
                p = np.array([m[target][key] for m in members])
                holm, bh = holm_adjusted(p), bh_adjusted(p)
                for member, a, b in zip(members, holm, bh, strict=True):
                    member[target][f"holm_{direction}"] = float(a)
                    member[target][f"bh_{direction}"] = float(b)


__all__ = [
    "NS_PER_HOUR",
    "CellMeasurement",
    "RowSet",
    "d0_profile",
    "d1_beta",
    "d2_features",
    "d4_record",
    "decision_rows",
    "grouped_ic",
    "measure_cell",
    "regime_signal",
    "utc_day",
]
