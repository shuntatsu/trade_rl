"""Expected-return sources, the drawdown overlay and the source registry for Phase 2.

Every source implements the allocator's ``ExpectedReturnSource`` protocol (per
decision-interval simple returns) and exposes the same ``calls`` /
``null_packets`` / ``history`` diagnostics as ``PrequentialSimpleReturnSource`` so
the shared runner reports them uniformly. All signals read only the daily
00:00 UTC marks and the funding of due rows **at or before the decision time**;
the series helpers are causal recursions and the sources index them at the
decision position only. The frozen definitions are in
:mod:`~trade_rl.evaluation.dev_realdata_study.trials`.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.dev_realdata_study import runner
from trade_rl.evaluation.dev_realdata_study.protocol import DevStudyProtocol
from trade_rl.evaluation.dev_realdata_study.trials import (
    CONSTANTS,
    KIND_BLEND,
    KIND_FUNDING_CARRY,
    KIND_LIGHTGBM,
    KIND_RIDGE_ALL,
    KIND_TSMOM,
    KIND_XSMOM,
    PR_C_ARMS,
    TRIALS,
    Phase2Constants,
    TrialSpec,
)
from trade_rl.strategies.allocation.costs import one_way_cost_rate
from trade_rl.strategies.forecasts.lightgbm import LightGBMForecastModel
from trade_rl.strategies.forecasts.prequential import (
    PrequentialForecastModel,
    PrequentialForecastSource,
    PrequentialSchedule,
    RidgeForecastFitter,
)
from trade_rl.strategies.forecasts.supervised import CausalForecastTrainingSet
from trade_rl.strategies.portfolio import (
    PortfolioDecisionSnapshot,
    PortfolioTargetProposal,
)

_NS_PER_HOUR = 3_600 * 1_000_000_000
# Minimum finite pooled observations for the T1/T3 trailing std in the blend. This
# constant is not in the frozen registry; it is declared in the provenance JSON.
BLEND_MIN_POOLED_OBSERVATIONS = 30
_NS_PER_DAY = 24 * _NS_PER_HOUR


# --- causal series helpers ------------------------------------------------


def lagged_return(marks: np.ndarray, lag: int) -> np.ndarray:
    """``marks[d] / marks[d - lag] - 1`` (NaN where ``d < lag``); uses days <= d."""

    if lag < 1:
        raise ValueError("lag must be positive")
    out = np.full(marks.shape, np.nan, dtype=np.float64)
    if marks.shape[0] > lag:
        out[lag:] = marks[lag:] / marks[:-lag] - 1.0
    return out


def ewma_annual_vol(
    marks: np.ndarray,
    *,
    halflife_days: float,
    min_periods: int,
    days_per_year: int,
) -> np.ndarray:
    """Annualized EWMA daily-return volatility, using daily returns up to day ``d``.

    Normalized weights ``0.5 ** (age / halflife)``, non-centered second moment
    (the allocator's own risk estimator convention); NaN until ``min_periods``
    daily returns exist.
    """

    decay = 0.5 ** (1.0 / halflife_days)
    out = np.full(marks.shape, np.nan, dtype=np.float64)
    weighted = np.zeros(marks.shape[1], dtype=np.float64)
    total = 0.0
    count = 0
    for day in range(1, marks.shape[0]):
        daily = marks[day] / marks[day - 1] - 1.0
        weighted = decay * weighted + daily * daily
        total = decay * total + 1.0
        count += 1
        if count >= min_periods:
            out[day] = np.sqrt(weighted / total * days_per_year)
    return out


def tsmom_series(
    marks: np.ndarray, *, one_way_cost: float, constants: Phase2Constants
) -> np.ndarray:
    """T1 expected returns per day: ``2 * s * (target / ann_vol) * one_way_cost``."""

    windows = constants.tsmom_windows_days
    blend = np.zeros(marks.shape, dtype=np.float64)
    for window in windows:
        blend = blend + np.sign(lagged_return(marks, window)) / len(windows)
    vol = ewma_annual_vol(
        marks,
        halflife_days=constants.tsmom_vol_halflife_days,
        min_periods=constants.tsmom_vol_min_periods,
        days_per_year=constants.tsmom_days_per_year,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        scaled = constants.tsmom_vol_target / vol
    scaled = np.where(np.isfinite(scaled) & (vol > 0.0), scaled, np.nan)
    return np.asarray(constants.signal_cost_multiple * blend * scaled * one_way_cost)


def xsmom_row(
    marks: np.ndarray, day: int, *, one_way_cost: float, constants: Phase2Constants
) -> np.ndarray:
    """T2 expected returns at day ``day``: demeaned 90d score, mean |mu| = 2 * cost."""

    skip = constants.xsmom_skip_days
    lookback = constants.xsmom_lookback_days
    n_symbols = marks.shape[1]
    if day - skip - lookback < 0:
        return np.full(n_symbols, np.nan)
    score = marks[day - skip] / marks[day - skip - lookback] - 1.0
    centered = score - score.mean()
    scale = float(np.abs(centered).mean())
    if not math.isfinite(scale) or scale <= 0.0:
        return np.zeros(n_symbols)
    return np.asarray(
        centered * (constants.signal_cost_multiple * one_way_cost / scale)
    )


class DailyPanel:
    """Daily 00:00 UTC marks and a prefix-summed funding series of one Dataset.

    The whole dataset is read at construction, but every consumer indexes it at a
    decision position ``p`` and uses days ``<= p`` only (the series helpers are
    causal recursions); the perturbation tests pin this.
    """

    def __init__(self, dataset: MarketDataset) -> None:
        self.symbols = dataset.symbols
        stamps = dataset.timestamps.astype("datetime64[ns]").astype(np.int64)
        rows = np.flatnonzero(stamps % _NS_PER_DAY == 0)
        if rows.size < 2 or np.any(np.diff(stamps[rows]) != _NS_PER_DAY):
            raise ValueError("dataset needs consecutive daily 00:00 UTC bars")
        marks = np.asarray(dataset.resolved_array("mark_price"), dtype=np.float64)
        if not np.all(np.isfinite(marks)) or np.any(marks <= 0.0):
            raise ValueError("mark prices must be positive and finite")
        self.rows = rows
        self.timestamps_ns = stamps[rows]
        self.marks = marks[rows]
        self._stamps = stamps
        funding = np.asarray(dataset.funding_rate, dtype=np.float64)
        due = np.asarray(dataset.funding_due, dtype=np.bool_)
        paid = np.where(due, funding, 0.0)
        if not np.all(np.isfinite(paid)):
            raise ValueError("funding rates on due rows must be finite")
        self._funding_prefix = np.vstack(
            [np.zeros((1, paid.shape[1])), np.cumsum(paid, axis=0)]
        )

    @property
    def n_days(self) -> int:
        return int(self.rows.size)

    def position(self, timestamp_ns: int) -> int:
        day = int(np.searchsorted(self.timestamps_ns, timestamp_ns))
        if day >= self.n_days or int(self.timestamps_ns[day]) != timestamp_ns:
            raise ValueError("decision time must be a daily 00:00 UTC dataset bar")
        return day

    def funding_carry_series(self, window_days: int) -> np.ndarray:
        """T3 expected returns per day: minus the mean funding per 24h over ``(t-w, t]``."""

        out = np.full(self.marks.shape, np.nan, dtype=np.float64)
        for day in range(self.n_days):
            stamp = int(self.timestamps_ns[day])
            start = stamp - window_days * _NS_PER_DAY
            if start < int(self._stamps[0]):
                continue
            low = int(np.searchsorted(self._stamps, start, side="right"))
            high = int(self.rows[day]) + 1
            out[day] = -(self._funding_prefix[high] - self._funding_prefix[low]) / (
                window_days
            )
        return out


# --- sources --------------------------------------------------------------


class _SeriesSource:
    """Common diagnostics and the snapshot -> daily position -> vector plumbing."""

    def __init__(self, panel: DailyPanel) -> None:
        self._panel = panel
        self.calls = 0
        self.null_packets = 0
        self.history: list[tuple[float, ...]] = []

    def values(self, day: int) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def expected_simple_returns(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> tuple[float, ...]:
        if snapshot.symbols != self._panel.symbols:
            raise ValueError("snapshot symbols do not match the signal roster")
        day = self._panel.position(snapshot.timestamp_ns)
        raw = np.asarray(self.values(day), dtype=np.float64)
        unusable = ~np.isfinite(raw)
        self.null_packets += int(unusable.sum())
        result = tuple(float(value) for value in np.where(unusable, 0.0, raw))
        self.calls += 1
        self.history.append(result)
        return result


class SeriesLookupSource(_SeriesSource):
    """Serve a precomputed causal ``(days, symbols)`` expected-return series."""

    def __init__(self, panel: DailyPanel, series: np.ndarray, *, scale: float = 1.0):
        super().__init__(panel)
        if series.shape != panel.marks.shape:
            raise ValueError("series must match the daily panel")
        self._series = series
        self._scale = scale

    def values(self, day: int) -> np.ndarray:
        return np.asarray(self._series[day] * self._scale)


class XsmomSource(_SeriesSource):
    """T2: cross-sectional 90d momentum (skip the last day), demeaned."""

    def __init__(
        self, panel: DailyPanel, *, one_way_cost: float, constants: Phase2Constants
    ) -> None:
        super().__init__(panel)
        self._cost = one_way_cost
        self._constants = constants

    def values(self, day: int) -> np.ndarray:
        return xsmom_row(
            self._panel.marks[: day + 1],
            day,
            one_way_cost=self._cost,
            constants=self._constants,
        )


class RidgeDailyComponent:
    """Ridge24 expected simple returns (per 24h) at any daily date with a vintage.

    ``value_at`` reads the packet of the vintage usable at that date, so a past
    date always uses a vintage fitted before it. Dates before the first vintage
    are unavailable (``None``). Results are cached by day.
    """

    def __init__(
        self, source: PrequentialForecastSource, panel: DailyPanel, *, divisor: float
    ) -> None:
        schedule = source.schedule
        self._source = source
        self._panel = panel
        self._divisor = divisor
        self._first_usable_ns = int(schedule.first_fit_time.astype(np.int64)) + (
            schedule.activation_delay_hours * _NS_PER_HOUR
        )
        self._cache: dict[int, np.ndarray] = {}

    def value_at(self, day: int) -> np.ndarray | None:
        stamp = int(self._panel.timestamps_ns[day])
        if stamp < self._first_usable_ns:
            return None
        cached = self._cache.get(day)
        if cached is None:
            cached = np.full(len(self._panel.symbols), np.nan)
            when = np.datetime64(stamp, "ns")
            for index in range(cached.size):
                packet = self._source.packet(index, when)
                if packet is not None:
                    cached[index] = (
                        math.expm1(
                            packet.point_forecast + 0.5 * packet.residual_variance
                        )
                        / self._divisor
                    )
            self._cache[day] = cached
        return cached


class BlendSource(_SeriesSource):
    """T6/T7/T8: scale-normalised equal-weight blend of Ridge24, T1 and T3.

    Each component (per-24h expected simple return) is divided by its causal
    trailing population std pooled over symbols and dates ``(t-364d .. t)``; the
    mean of the three is multiplied by the same trailing std of Ridge24 mu and by
    ``interval_scale`` (decision interval in days) so it is an expected return over
    the decision interval. Until ``min_dates`` Ridge24 dates exist it serves NaN
    (zero mu).
    """

    def __init__(
        self,
        panel: DailyPanel,
        ridge: RidgeDailyComponent,
        tsmom: np.ndarray,
        carry: np.ndarray,
        *,
        constants: Phase2Constants,
        interval_scale: float,
    ) -> None:
        super().__init__(panel)
        self._ridge = ridge
        self._tsmom = tsmom
        self._carry = carry
        self._constants = constants
        self._interval_scale = interval_scale

    @staticmethod
    def _pooled_std(values: np.ndarray, *, minimum: int) -> float | None:
        finite = values[np.isfinite(values)]
        if finite.size < minimum:
            return None
        std = float(np.std(finite))
        return std if math.isfinite(std) and std > 0.0 else None

    def values(self, day: int) -> np.ndarray:
        constants = self._constants
        n_symbols = len(self._panel.symbols)
        nothing = np.full(n_symbols, np.nan)
        low = max(0, day - (constants.blend_std_window_days - 1))
        rows = [self._ridge.value_at(q) for q in range(low, day + 1)]
        ridge_rows = [row for row in rows if row is not None]
        if len(ridge_rows) < constants.blend_min_dates:
            return nothing
        ridge_history = np.vstack(ridge_rows)
        ridge_sigma = self._pooled_std(ridge_history, minimum=constants.blend_min_dates)
        tsmom_sigma = self._pooled_std(
            self._tsmom[low : day + 1], minimum=BLEND_MIN_POOLED_OBSERVATIONS
        )
        carry_sigma = self._pooled_std(
            self._carry[low : day + 1], minimum=BLEND_MIN_POOLED_OBSERVATIONS
        )
        if ridge_sigma is None or tsmom_sigma is None or carry_sigma is None:
            return nothing
        today = (
            self._ridge.value_at(day),
            self._tsmom[day],
            self._carry[day],
        )
        sigmas = (ridge_sigma, tsmom_sigma, carry_sigma)
        total = np.zeros(n_symbols)
        for component, sigma in zip(today, sigmas, strict=True):
            if component is None:
                continue
            normalised = component / sigma
            total = total + np.where(np.isfinite(normalised), normalised, 0.0)
        return np.asarray(total / 3.0 * ridge_sigma * self._interval_scale)


# --- LightGBM fitter ------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConfiguredLightGBMFitter:
    """LightGBM fitter with the Phase 2 fixed shallow parameters (T5)."""

    n_estimators: int = CONSTANTS.lgbm_n_estimators
    max_depth: int = CONSTANTS.lgbm_max_depth
    num_leaves: int = CONSTANTS.lgbm_num_leaves
    learning_rate: float = CONSTANTS.lgbm_learning_rate
    min_child_samples: int = CONSTANTS.lgbm_min_child_samples
    reg_lambda: float = CONSTANTS.lgbm_reg_lambda
    random_state: int = CONSTANTS.lgbm_random_state

    @property
    def identity(self) -> str:
        return (
            "lightgbm_phase2:"
            f"n={self.n_estimators},depth={self.max_depth},leaves={self.num_leaves},"
            f"lr={self.learning_rate!r},min_leaf={self.min_child_samples},"
            f"lambda={self.reg_lambda!r},seed={self.random_state}"
        )

    def fit(self, training: CausalForecastTrainingSet) -> PrequentialForecastModel:
        try:
            module = importlib.import_module("lightgbm")
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "T5 requires the optional 'forecast-gbm' dependency; no substitute "
                "model is used"
            ) from exc
        predictor = module.LGBMRegressor(
            objective="regression",
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            min_child_samples=self.min_child_samples,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            n_jobs=1,
            verbosity=-1,
        )
        predictor.fit(
            training.features,
            training.labels,
            sample_weight=training.sample_weights,
        )
        return LightGBMForecastModel(
            feature_indices=training.feature_indices,
            predictor=predictor,
            horizon_hours=training.horizon_hours,
            n_samples=training.n_samples,
            fit_cutoff=training.fit_cutoff,
        )

    def predict_rows(
        self, model: PrequentialForecastModel, features: np.ndarray
    ) -> np.ndarray:
        predictor = getattr(model, "predictor", None)
        predict = getattr(predictor, "predict", None)
        if not callable(predict):
            raise TypeError("ConfiguredLightGBMFitter requires a LightGBM model")
        return np.asarray(predict(features), dtype=np.float64).reshape(-1)


# --- drawdown overlay -----------------------------------------------------


def overlay_scale(drawdown: float, *, start: float, limit: float) -> float:
    """``clip((limit - drawdown) / (limit - start), 0, 1)``."""

    if not 0.0 <= start < limit:
        raise ValueError("overlay start must be non-negative and below the limit")
    return float(min(1.0, max(0.0, (limit - drawdown) / (limit - start))))


class DrawdownLatch:
    """Scale from the running drawdown with a latch at 0 until recovery.

    ``update`` is called once per scheduled decision with the account equity and
    the account's running peak. Once ``DD >= limit`` the scale stays 0 until
    ``equity >= release_fraction * peak``.
    """

    def __init__(self, constants: Phase2Constants = CONSTANTS) -> None:
        self._start = constants.overlay_dd_start
        self._limit = constants.overlay_dd_limit
        self._release = constants.overlay_release_fraction
        self.latched = False

    def update(self, equity: float, peak: float) -> float:
        if not (math.isfinite(equity) and math.isfinite(peak)) or peak <= 0.0:
            raise ValueError("equity and peak must be finite and peak positive")
        drawdown = max(0.0, 1.0 - equity / peak)
        if self.latched and equity >= self._release * peak:
            self.latched = False
        if not self.latched and drawdown >= self._limit - 1e-12:
            self.latched = True
        if self.latched:
            return 0.0
        return overlay_scale(drawdown, start=self._start, limit=self._limit)


class DrawdownOverlayPolicy:
    """Wrap an allocator and cap its gross exposure at ``scale * gross_max``.

    The inner policy is always called (it keeps its own covariance state). At a
    scheduled decision the overlay computes the scale from the account snapshot;
    if the inner target's gross exceeds ``scale * gross_max`` the target is shrunk
    proportionally to that budget (hold flags dropped); otherwise the inner
    proposal is returned untouched, so scale 1 never changes the allocator.
    """

    def __init__(
        self,
        inner: Any,
        *,
        gross_max: float,
        decision_interval_hours: int,
        decision_anchor_hour_utc: int,
        constants: Phase2Constants = CONSTANTS,
    ) -> None:
        self._inner = inner
        self._gross_max = gross_max
        self._interval_ns = decision_interval_hours * _NS_PER_HOUR
        self._anchor_ns = decision_anchor_hour_utc * _NS_PER_HOUR
        self._latch = DrawdownLatch(constants)
        self.scales: list[tuple[int, float]] = []

    @property
    def records(self) -> Any:
        return self._inner.records

    def overlay_summary(self) -> dict[str, object]:
        scales = np.asarray([scale for _, scale in self.scales], dtype=np.float64)
        first_zero = next((stamp for stamp, scale in self.scales if scale == 0.0), None)
        return {
            "n_decisions": int(scales.size),
            "n_decisions_scaled_below_one": int(np.sum(scales < 1.0)),
            "n_decisions_at_zero": int(np.sum(scales == 0.0)),
            "mean_scale": float(scales.mean()) if scales.size else None,
            "first_zero_scale_timestamp": (
                None if first_zero is None else str(np.datetime64(first_zero, "ns"))
            ),
            "latched_at_end": bool(self._latch.latched),
        }

    def decide_portfolio(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> PortfolioTargetProposal:
        proposal = cast(PortfolioTargetProposal, self._inner.decide_portfolio(snapshot))
        if snapshot.timestamp_ns % self._interval_ns != self._anchor_ns:
            return proposal
        account = snapshot.account
        scale = self._latch.update(account.portfolio_value, account.peak_value)
        self.scales.append((snapshot.timestamp_ns, scale))
        if scale >= 1.0:
            return proposal
        budget = scale * self._gross_max
        gross = float(sum(abs(value) for value in proposal.target_weights))
        if gross <= budget:
            return proposal
        factor = 0.0 if budget <= 0.0 else budget / gross
        return PortfolioTargetProposal(
            target_weights=tuple(value * factor for value in proposal.target_weights)
        )


# --- context and registry -------------------------------------------------


class TrialContext:
    """Per-process, per-dataset cache of panels and forecast sources."""

    def __init__(self, dataset: MarketDataset, protocol: DevStudyProtocol) -> None:
        self.dataset = dataset
        self.protocol = protocol
        self.constants = CONSTANTS
        self.one_way_cost = one_way_cost_rate(runner.scenario_cost(protocol, "base"))
        self._panel: DailyPanel | None = None
        self._pr_c_sources: dict[int, PrequentialForecastSource] | None = None
        self._forecast: dict[str, PrequentialForecastSource] = {}
        self._tsmom: np.ndarray | None = None
        self._carry: np.ndarray | None = None

    @property
    def panel(self) -> DailyPanel:
        if self._panel is None:
            self._panel = DailyPanel(self.dataset)
        return self._panel

    @property
    def pr_c_sources(self) -> dict[int, PrequentialForecastSource]:
        if self._pr_c_sources is None:
            self._pr_c_sources = runner.build_forecast_sources(
                self.dataset, self.protocol
            )
        return self._pr_c_sources

    def tsmom_series(self) -> np.ndarray:
        if self._tsmom is None:
            self._tsmom = tsmom_series(
                self.panel.marks,
                one_way_cost=self.one_way_cost,
                constants=self.constants,
            )
        return self._tsmom

    def carry_series(self) -> np.ndarray:
        if self._carry is None:
            self._carry = self.panel.funding_carry_series(
                self.constants.carry_window_days
            )
        return self._carry

    def forecast_source(self, key: str) -> PrequentialForecastSource:
        """``ridge24_1h`` (PR-C), ``ridge24_all`` (T4) or ``lgbm24_1h`` (T5)."""

        if key == "ridge24_1h":
            return self.pr_c_sources[24]
        existing = self._forecast.get(key)
        if existing is not None:
            return existing
        dataset, protocol = self.dataset, self.protocol
        schedule = PrequentialSchedule(
            first_fit_time=np.datetime64(protocol.evaluation_start, "ns"),
            refit_interval_days=protocol.refit_interval_days,
            activation_delay_hours=protocol.activation_delay_hours,
        )
        if key == "ridge24_all":
            prefixes = self.constants.ridge_all_prefixes
            indices = tuple(
                index
                for index, name in enumerate(dataset.feature_names)
                if name.startswith(prefixes)
            )
            fitter: Any = RidgeForecastFitter(alpha=protocol.ridge_alpha)
        elif key == "lgbm24_1h":
            indices = runner.feature_indices(dataset, protocol)
            fitter = ConfiguredLightGBMFitter()
        else:
            raise ValueError(f"unknown forecast source key: {key}")
        if not indices:
            raise ValueError("no features match the trial roster")
        source = PrequentialForecastSource(
            dataset,
            schedule=schedule,
            fitter=fitter,
            feature_indices=indices,
            horizon_hours=24,
        )
        self._forecast[key] = source
        return source


SourceFactory = Callable[[TrialContext, TrialSpec], Any]


def _tsmom_factory(context: TrialContext, spec: TrialSpec) -> Any:
    return SeriesLookupSource(context.panel, context.tsmom_series())


def _xsmom_factory(context: TrialContext, spec: TrialSpec) -> Any:
    return XsmomSource(
        context.panel,
        one_way_cost=context.one_way_cost,
        constants=context.constants,
    )


def _carry_factory(context: TrialContext, spec: TrialSpec) -> Any:
    return SeriesLookupSource(context.panel, context.carry_series())


def _ridge_all_factory(context: TrialContext, spec: TrialSpec) -> Any:
    return runner.PrequentialSimpleReturnSource(
        context.forecast_source("ridge24_all"),
        decision_interval_hours=spec.decision_interval_hours,
    )


def _lightgbm_factory(context: TrialContext, spec: TrialSpec) -> Any:
    return runner.PrequentialSimpleReturnSource(
        context.forecast_source("lgbm24_1h"),
        decision_interval_hours=spec.decision_interval_hours,
    )


def _blend_factory(context: TrialContext, spec: TrialSpec) -> Any:
    ridge = RidgeDailyComponent(
        context.forecast_source("ridge24_1h"), context.panel, divisor=1.0
    )
    return BlendSource(
        context.panel,
        ridge,
        context.tsmom_series(),
        context.carry_series(),
        constants=context.constants,
        interval_scale=spec.decision_interval_hours / 24.0,
    )


SOURCE_REGISTRY: dict[str, SourceFactory] = {
    KIND_TSMOM: _tsmom_factory,
    KIND_XSMOM: _xsmom_factory,
    KIND_FUNDING_CARRY: _carry_factory,
    KIND_RIDGE_ALL: _ridge_all_factory,
    KIND_LIGHTGBM: _lightgbm_factory,
    KIND_BLEND: _blend_factory,
}


def trial_spec(arm: str) -> TrialSpec:
    for spec in TRIALS:
        if spec.arm == arm:
            return spec
    raise ValueError(f"unknown trial arm: {arm}")


def build_trial_source(context: TrialContext, spec: TrialSpec) -> Any:
    try:
        factory = SOURCE_REGISTRY[spec.source_kind]
    except KeyError as exc:
        raise ValueError(f"unregistered source kind: {spec.source_kind}") from exc
    return factory(context, spec)


def build_phase2_arm(
    arm: str,
    dataset: MarketDataset,
    protocol: DevStudyProtocol,
    context: TrialContext,
) -> runner.BuiltArm:
    """Fresh policy for one replay of a PR-C arm or a registered trial."""

    if arm in PR_C_ARMS:
        return runner.build_arm(arm, dataset, protocol, context.pr_c_sources)
    spec = trial_spec(arm)
    source = build_trial_source(context, spec)
    policy: Any = runner.build_allocator(
        dataset,
        protocol,
        source,
        long_only=spec.long_only,
        decision_interval_hours=spec.decision_interval_hours,
        decision_anchor_hour_utc=spec.decision_anchor_hour_utc,
    )
    if spec.overlay == "drawdown_latch":
        policy = DrawdownOverlayPolicy(
            policy,
            gross_max=protocol.gross_max,
            decision_interval_hours=spec.decision_interval_hours,
            decision_anchor_hour_utc=spec.decision_anchor_hour_utc,
            constants=context.constants,
        )
    elif spec.overlay is not None:
        raise ValueError(f"unknown overlay: {spec.overlay}")
    return runner.BuiltArm(policy, source)


__all__ = [
    "BLEND_MIN_POOLED_OBSERVATIONS",
    "SOURCE_REGISTRY",
    "BlendSource",
    "ConfiguredLightGBMFitter",
    "DailyPanel",
    "DrawdownLatch",
    "DrawdownOverlayPolicy",
    "RidgeDailyComponent",
    "SeriesLookupSource",
    "TrialContext",
    "XsmomSource",
    "build_phase2_arm",
    "build_trial_source",
    "ewma_annual_vol",
    "lagged_return",
    "overlay_scale",
    "trial_spec",
    "tsmom_series",
    "xsmom_row",
]
