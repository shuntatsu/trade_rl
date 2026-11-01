"""Causal prequential forecast packets with scheduled expanding-window refits.

A *packet* is the single numeric fact a downstream allocator needs for one
``(symbol, decision_time)`` pair: a log-return point forecast over a fixed
horizon plus the training residual variance of the vintage that produced it.
Vintages are fitted on an expanding window at a fixed schedule and frozen
between refits, so every packet is a pure function of data that was available
at its ``fit_time`` and of the decision-row features.  The module reuses
``build_causal_forecast_training_set`` for all label construction and does not
own an account, a ledger, or any trading decision.

Peeking is structurally impossible through the fitter interface: the source
builds the causal training set itself and hands a fitter only that frozen
object (features, labels, weights, feature metadata) -- never the
``MarketDataset`` -- so a fitter has no handle on rows after the cutoff.  The
residual variance and every point forecast are computed through the same
``fitter.predict_rows`` path, and ``model.predict`` must agree with it.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.strategies.dataset_scope import (
    validated_feature_indices,
    validated_symbol_indices,
)
from trade_rl.strategies.forecasts.lightgbm import (
    fit_lightgbm_forecast_from_training_set,
)
from trade_rl.strategies.forecasts.ridge import (
    RidgeForecastModel,
    fit_ridge_forecast_from_training_set,
)
from trade_rl.strategies.forecasts.supervised import (
    CausalForecastTrainingSet,
    _validated_fit_cutoff_ns,
    build_causal_forecast_training_set,
    validate_supervised_forecast_horizon_hours,
)

LOG_RETURN_UNIT = "log"
_NS_PER_HOUR = 3_600_000_000_000
_NS_PER_DAY = 24 * _NS_PER_HOUR
_AGREEMENT_PROBE_ROWS = 32
_AGREEMENT_RTOL = 1e-9
_AGREEMENT_ATOL = 1e-12


class PrequentialForecastModel(Protocol):
    """Frozen fitted model that scores one full feature vector."""

    @property
    def fit_cutoff(self) -> np.datetime64: ...

    def predict(self, features: np.ndarray) -> float: ...


class PrequentialForecastFitter(Protocol):
    """Small fitter protocol: fit on a causal training set, score rows in batch.

    ``fit`` receives only the frozen :class:`CausalForecastTrainingSet` built by
    the source (rows whose labels were available strictly before the cutoff);
    it never receives the dataset.  ``predict_rows`` is the single scoring path
    used for both the training residuals and every served point forecast.
    """

    @property
    def identity(self) -> str: ...

    def fit(self, training: CausalForecastTrainingSet) -> PrequentialForecastModel: ...

    def predict_rows(
        self,
        model: PrequentialForecastModel,
        features: np.ndarray,
    ) -> np.ndarray:
        """Score a ``(rows, len(feature_indices))`` matrix of selected features."""
        ...


@dataclass(frozen=True, slots=True)
class RidgeForecastFitter:
    """Ridge fitter delegating to the canonical ``fit_ridge_forecast``."""

    alpha: float = 1.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.alpha, bool)
            or not math.isfinite(self.alpha)
            or self.alpha <= 0.0
        ):
            raise ValueError("alpha must be finite and positive")

    @property
    def identity(self) -> str:
        return f"ridge:alpha={self.alpha!r}"

    def fit(self, training: CausalForecastTrainingSet) -> RidgeForecastModel:
        return fit_ridge_forecast_from_training_set(training, alpha=self.alpha)

    def predict_rows(
        self,
        model: PrequentialForecastModel,
        features: np.ndarray,
    ) -> np.ndarray:
        if not isinstance(model, RidgeForecastModel):
            raise TypeError("RidgeForecastFitter requires a RidgeForecastModel")
        standardized = (features - model.feature_mean) / model.feature_scale
        return np.asarray(
            model.intercept + standardized @ model.coefficients,
            dtype=np.float64,
        )


@dataclass(frozen=True, slots=True)
class LightGBMForecastFitter:
    """Optional LightGBM fitter (requires the ``forecast-gbm`` extra at fit time)."""

    random_state: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.random_state, bool) or not isinstance(
            self.random_state, int
        ):
            raise ValueError("random_state must be an integer")

    @property
    def identity(self) -> str:
        return f"lightgbm:random_state={self.random_state}"

    def fit(self, training: CausalForecastTrainingSet) -> PrequentialForecastModel:
        return fit_lightgbm_forecast_from_training_set(
            training, random_state=self.random_state
        )

    def predict_rows(
        self,
        model: PrequentialForecastModel,
        features: np.ndarray,
    ) -> np.ndarray:
        predictor = getattr(model, "predictor", None)
        predict = getattr(predictor, "predict", None)
        if not callable(predict):
            raise TypeError("LightGBMForecastFitter requires a LightGBM model")
        return np.asarray(predict(features), dtype=np.float64).reshape(-1)


def _ns(value: np.datetime64, *, field_name: str) -> int:
    try:
        normalized = _validated_fit_cutoff_ns(value)
    except ValueError as error:
        raise ValueError(
            f"{field_name} must be an exact non-NaT datetime64[ns]"
        ) from error
    return int(normalized.astype(np.int64))


def _iso(value_ns: int) -> str:
    return str(np.datetime64(value_ns, "ns"))


def _sha256_json(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _array_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(repr(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class PrequentialSchedule:
    """Expanding-window refit calendar with frozen vintages between refits.

    Vintage ``k`` uses the causal fit cutoff
    ``first_fit_time + k * refit_interval_days`` and becomes usable at
    ``fit_time = cutoff + activation_delay_hours`` (zero by default: the fit is
    assumed to finish instantly, which is the right model for a backtest whose
    cutoff is the fit instant).  A positive delay is required when the dataset's
    ``available_at`` lags the bar timestamp enough that a label becomes
    available exactly at the cutoff: it only moves ``fit_time`` after the last
    admitted label's availability.  It does *not* relax decision-row
    availability: a decision row whose ``available_at`` is later than its own
    timestamp is never servable and ``packet`` fails closed.  The training
    window always starts at the dataset's first row, so it only grows.
    """

    first_fit_time: np.datetime64
    refit_interval_days: int = 30
    activation_delay_hours: int = 0

    def __post_init__(self) -> None:
        first = _ns(self.first_fit_time, field_name="first_fit_time")
        if (
            isinstance(self.refit_interval_days, bool)
            or not isinstance(self.refit_interval_days, int)
            or self.refit_interval_days <= 0
        ):
            raise ValueError("refit_interval_days must be a positive integer")
        if (
            isinstance(self.activation_delay_hours, bool)
            or not isinstance(self.activation_delay_hours, int)
            or self.activation_delay_hours < 0
        ):
            raise ValueError("activation_delay_hours must be a non-negative integer")
        object.__setattr__(self, "first_fit_time", np.datetime64(first, "ns"))

    def fit_time_for(self, cutoff: np.datetime64) -> np.datetime64:
        """Return the instant the vintage with this fit cutoff becomes usable."""

        cutoff_ns = _ns(cutoff, field_name="cutoff")
        return np.datetime64(
            cutoff_ns + self.activation_delay_hours * _NS_PER_HOUR, "ns"
        )

    def cutoff_for(self, decision_time: np.datetime64) -> np.datetime64:
        """Return the fit cutoff of the latest vintage usable at ``decision_time``."""

        decision_ns = _ns(decision_time, field_name="decision_time")
        first_ns = int(self.first_fit_time.astype(np.int64))
        usable_ns = decision_ns - self.activation_delay_hours * _NS_PER_HOUR
        if usable_ns < first_ns:
            raise ValueError(
                "no vintage is usable at decision_time: "
                f"first fit {_iso(first_ns)} plus activation delay is later"
            )
        step_ns = self.refit_interval_days * _NS_PER_DAY
        index = (usable_ns - first_ns) // step_ns
        return np.datetime64(first_ns + index * step_ns, "ns")

    def cutoffs(self, through: np.datetime64) -> tuple[np.datetime64, ...]:
        """All scheduled fit cutoffs ``<= through``."""

        through_ns = _ns(through, field_name="through")
        first_ns = int(self.first_fit_time.astype(np.int64))
        step_ns = self.refit_interval_days * _NS_PER_DAY
        count = 0 if through_ns < first_ns else (through_ns - first_ns) // step_ns + 1
        return tuple(np.datetime64(first_ns + i * step_ns, "ns") for i in range(count))


@dataclass(frozen=True, slots=True)
class ForecastPacket:
    """One log-return forecast for ``(symbol, decision_time)`` from one vintage.

    Invariant checked at construction:
    ``max_training_label_available_at < fit_time <= forecast_available_at
    <= decision_time`` and ``fit_cutoff <= fit_time``.
    """

    symbol: str
    symbol_index: int
    horizon_hours: int
    decision_time: np.datetime64
    forecast_available_at: np.datetime64
    return_unit: str
    point_forecast: float
    residual_variance: float
    vintage_id: str
    fit_time: np.datetime64
    fit_cutoff: np.datetime64
    max_training_label_available_at: np.datetime64
    n_training_samples: int
    fitter_identity: str
    training_digest: str
    model_digest: str
    feature_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not self.symbol:
            raise ValueError("symbol must be a non-empty string")
        if (
            isinstance(self.symbol_index, bool)
            or not isinstance(self.symbol_index, int)
            or self.symbol_index < 0
        ):
            raise ValueError("symbol_index must be a non-negative integer")
        horizon = validate_supervised_forecast_horizon_hours(self.horizon_hours)
        if self.return_unit != LOG_RETURN_UNIT:
            raise ValueError(f"return_unit must be {LOG_RETURN_UNIT!r}")
        for name in ("point_forecast", "residual_variance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite number")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if self.residual_variance < 0.0:
            raise ValueError("residual_variance must be non-negative")
        if (
            isinstance(self.n_training_samples, bool)
            or not isinstance(self.n_training_samples, int)
            or self.n_training_samples <= 0
        ):
            raise ValueError("n_training_samples must be a positive integer")
        for name in (
            "vintage_id",
            "training_digest",
            "model_digest",
            "feature_digest",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
        if not isinstance(self.fitter_identity, str) or not self.fitter_identity:
            raise ValueError("fitter_identity must be a non-empty string")
        decision = _ns(self.decision_time, field_name="decision_time")
        available = _ns(self.forecast_available_at, field_name="forecast_available_at")
        fit_time = _ns(self.fit_time, field_name="fit_time")
        cutoff = _ns(self.fit_cutoff, field_name="fit_cutoff")
        label_max = _ns(
            self.max_training_label_available_at,
            field_name="max_training_label_available_at",
        )
        if cutoff > fit_time:
            raise ValueError("fit_cutoff must not be later than fit_time")
        if not label_max < fit_time:
            raise ValueError(
                "max_training_label_available_at must be strictly before fit_time"
            )
        if not fit_time <= available:
            raise ValueError("forecast_available_at must not precede fit_time")
        if not available <= decision:
            raise ValueError("forecast_available_at must not be after decision_time")
        object.__setattr__(self, "horizon_hours", horizon)
        object.__setattr__(self, "point_forecast", float(self.point_forecast))
        object.__setattr__(self, "residual_variance", float(self.residual_variance))
        object.__setattr__(self, "decision_time", np.datetime64(decision, "ns"))
        object.__setattr__(
            self, "forecast_available_at", np.datetime64(available, "ns")
        )
        object.__setattr__(self, "fit_time", np.datetime64(fit_time, "ns"))
        object.__setattr__(self, "fit_cutoff", np.datetime64(cutoff, "ns"))
        object.__setattr__(
            self,
            "max_training_label_available_at",
            np.datetime64(label_max, "ns"),
        )

    def to_mapping(self) -> dict[str, object]:
        """Canonical JSON-safe mapping (used for byte-level comparisons)."""

        return {
            "symbol": self.symbol,
            "symbol_index": self.symbol_index,
            "horizon_hours": self.horizon_hours,
            "decision_time": str(self.decision_time),
            "forecast_available_at": str(self.forecast_available_at),
            "return_unit": self.return_unit,
            "point_forecast": self.point_forecast,
            "residual_variance": self.residual_variance,
            "vintage_id": self.vintage_id,
            "fit_time": str(self.fit_time),
            "fit_cutoff": str(self.fit_cutoff),
            "max_training_label_available_at": str(
                self.max_training_label_available_at
            ),
            "n_training_samples": self.n_training_samples,
            "fitter_identity": self.fitter_identity,
            "training_digest": self.training_digest,
            "model_digest": self.model_digest,
            "feature_digest": self.feature_digest,
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_mapping(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


@dataclass(frozen=True, slots=True)
class ForecastVintage:
    """One frozen fit: model, training provenance and residual variance.

    ``residual_variance`` is the *in-sample* weighted training residual variance
    of the vintage, so it understates out-of-sample error (most for flexible
    models such as LightGBM, which biases a later ``expm1(mu + s2 / 2)``
    log-to-simple conversion low).  ``max_training_label_available_at`` is
    taken over all rows of the label end bars that the builder could have
    admitted, which is exact for the default ``available_at == timestamp`` and
    otherwise a conservative (possibly later) bound.
    """

    vintage_id: str
    fit_time: np.datetime64
    fit_cutoff: np.datetime64
    max_training_label_available_at: np.datetime64
    n_training_samples: int
    residual_variance: float
    fitter_identity: str
    horizon_hours: int
    training_digest: str
    model_digest: str
    model: PrequentialForecastModel = field(compare=False, repr=False)


def _max_label_available_at_ns(
    dataset: MarketDataset,
    training: CausalForecastTrainingSet,
    symbols: tuple[int, ...],
) -> int:
    timestamps_ns = dataset.timestamps.astype("datetime64[ns]").astype(np.int64)
    end_ns = np.unique(training.label_end_times.astype(np.int64))
    positions = np.searchsorted(timestamps_ns, end_ns)
    if np.any(positions >= timestamps_ns.size) or np.any(
        timestamps_ns[positions] != end_ns
    ):
        raise ValueError("training label end times must be dataset timestamps")
    available = dataset.resolved_array("available_at").astype("datetime64[ns]")
    cutoff = training.fit_cutoff
    # The builder admits only rows whose end-bar availability is <= cutoff, so
    # larger values belong to excluded rows and are ignored here.
    block = available[positions][:, list(symbols)]
    admitted = block[block <= cutoff]
    if admitted.size == 0:
        raise ValueError("training labels have no admitted availability timestamps")
    return int(admitted.max().astype(np.int64))


class PrequentialForecastSource:
    """Serve causal forecast packets from lazily fitted, frozen vintages.

    ``packet(symbol_index, decision_time)`` answers: "what log-return forecast
    and vintage residual variance may the allocator use for this symbol now?"
    The only data read at a decision is the decision-row feature vector and the
    vintage fitted at ``schedule.fit_time_for(decision_time)`` with a cutoff at
    that fit time.  ``decision_time`` must be a dataset timestamp.
    """

    def __init__(
        self,
        dataset: MarketDataset,
        *,
        schedule: PrequentialSchedule,
        fitter: PrequentialForecastFitter,
        feature_indices: tuple[int, ...],
        horizon_hours: int,
        fit_symbol_indices: tuple[int, ...] | None = None,
    ) -> None:
        self._dataset = dataset
        self._schedule = schedule
        self._fitter = fitter
        self._horizon_hours = validate_supervised_forecast_horizon_hours(horizon_hours)
        # Validate the selections once; every fit repeats the same validation.
        self._feature_indices = validated_feature_indices(
            dataset, tuple(feature_indices)
        )
        self._fit_symbol_indices = (
            None if fit_symbol_indices is None else tuple(fit_symbol_indices)
        )
        self._fit_symbols = validated_symbol_indices(dataset, self._fit_symbol_indices)
        self._timestamps_ns = dataset.timestamps.astype("datetime64[ns]").astype(
            np.int64
        )
        self._vintages: dict[int, ForecastVintage] = {}

    @property
    def horizon_hours(self) -> int:
        return self._horizon_hours

    @property
    def schedule(self) -> PrequentialSchedule:
        return self._schedule

    def vintage_for(self, decision_time: np.datetime64) -> ForecastVintage:
        """Return (fitting on first use) the vintage usable at ``decision_time``."""

        cutoff = self._schedule.cutoff_for(decision_time)
        key = int(cutoff.astype(np.int64))
        cached = self._vintages.get(key)
        if cached is None:
            cached = self._fit_vintage(cutoff)
            self._vintages[key] = cached
        return cached

    def _fit_vintage(self, cutoff: np.datetime64) -> ForecastVintage:
        dataset = self._dataset
        training = build_causal_forecast_training_set(
            dataset,
            feature_indices=self._feature_indices,
            fit_symbol_indices=self._fit_symbol_indices,
            fit_cutoff=cutoff,
            horizon_hours=self._horizon_hours,
        )
        # The fitter sees only the frozen causal training set, never the dataset.
        model = self._fitter.fit(training)
        model_cutoff = _ns(model.fit_cutoff, field_name="model fit_cutoff")
        if model_cutoff != int(cutoff.astype(np.int64)):
            raise ValueError("fitted model cutoff does not match the scheduled cutoff")
        fitted = self._score_rows(model, training.features)
        self._assert_predict_agreement(model, training.features, fitted)
        residuals = training.labels - fitted
        weights = training.sample_weights
        residual_variance = float(np.dot(weights, residuals**2) / float(weights.sum()))
        if not math.isfinite(residual_variance) or residual_variance < 0.0:
            raise ValueError("training residual variance must be finite")
        label_available_ns = _max_label_available_at_ns(
            dataset, training, self._fit_symbols
        )
        cutoff_ns = int(cutoff.astype(np.int64))
        fit_ns = int(self._schedule.fit_time_for(cutoff).astype(np.int64))
        if not label_available_ns < fit_ns:
            raise ValueError(
                "training labels were not available strictly before the fit time"
            )
        training_digest = _array_digest(
            training.features,
            training.labels,
            training.label_end_times.astype(np.int64),
            training.sample_weights,
        )
        model_digest = _array_digest(fitted)
        vintage_id = _sha256_json(
            {
                "fitter": self._fitter.identity,
                "feature_indices": list(self._feature_indices),
                "fit_symbol_indices": list(self._fit_symbols),
                "horizon_hours": self._horizon_hours,
                "fit_cutoff_ns": cutoff_ns,
                "training_digest": training_digest,
            }
        )
        return ForecastVintage(
            vintage_id=vintage_id,
            fit_time=np.datetime64(fit_ns, "ns"),
            fit_cutoff=np.datetime64(cutoff_ns, "ns"),
            max_training_label_available_at=np.datetime64(label_available_ns, "ns"),
            n_training_samples=training.n_samples,
            residual_variance=residual_variance,
            fitter_identity=self._fitter.identity,
            horizon_hours=self._horizon_hours,
            training_digest=training_digest,
            model_digest=model_digest,
            model=model,
        )

    def _score_rows(
        self,
        model: PrequentialForecastModel,
        features: np.ndarray,
    ) -> np.ndarray:
        """Single scoring path for training residuals and served forecasts."""

        scores = np.asarray(
            self._fitter.predict_rows(model, features),
            dtype=np.float64,
        ).reshape(-1)
        if scores.shape != (features.shape[0],) or not np.isfinite(scores).all():
            raise ValueError("fitter must return one finite prediction per row")
        return scores

    def _full_vector(self, selected: np.ndarray) -> np.ndarray:
        """Embed selected features at their dataset positions (others zero)."""

        vector = np.zeros(self._dataset.features.shape[2], dtype=np.float64)
        vector[list(self._feature_indices)] = selected
        return vector

    def _assert_predict_agreement(
        self,
        model: PrequentialForecastModel,
        features: np.ndarray,
        scores: np.ndarray,
    ) -> None:
        """``model.predict`` must reproduce ``fitter.predict_rows`` on a probe set."""

        n_rows = features.shape[0]
        probe = np.unique(
            np.linspace(0, n_rows - 1, min(_AGREEMENT_PROBE_ROWS, n_rows)).astype(
                np.int64
            )
        )
        for row in probe:
            direct = float(model.predict(self._full_vector(features[row])))
            if not math.isclose(
                direct,
                float(scores[row]),
                rel_tol=_AGREEMENT_RTOL,
                abs_tol=_AGREEMENT_ATOL,
            ):
                raise ValueError(
                    "model.predict and fitter.predict_rows disagree on a training row"
                )

    def _row_index(self, decision_time: np.datetime64) -> tuple[int, int]:
        decision_ns = _ns(decision_time, field_name="decision_time")
        row = int(np.searchsorted(self._timestamps_ns, decision_ns))
        if row >= self._timestamps_ns.size or self._timestamps_ns[row] != decision_ns:
            raise ValueError("decision_time must be a dataset timestamp")
        return row, decision_ns

    def packet(
        self,
        symbol_index: int,
        decision_time: np.datetime64,
    ) -> ForecastPacket | None:
        """Packet for one symbol, or ``None`` if its decision features are unusable.

        Raises when the decision time is not a dataset timestamp, no vintage is
        usable yet, or the decision row was not available at the decision time.
        """

        dataset = self._dataset
        if (
            isinstance(symbol_index, bool)
            or not isinstance(symbol_index, int)
            or not 0 <= symbol_index < dataset.n_symbols
        ):
            raise ValueError("symbol_index is outside dataset symbols")
        row, decision_ns = self._row_index(decision_time)
        available_ns = int(
            dataset.resolved_array("available_at")[row, symbol_index]
            .astype("datetime64[ns]")
            .astype(np.int64)
        )
        if available_ns > decision_ns:
            raise ValueError("decision row is not available at decision_time")
        vintage = self.vintage_for(decision_time)
        indices = list(self._feature_indices)
        if not bool(np.all(dataset.feature_available[row, symbol_index, indices])):
            return None
        selected = np.asarray(
            dataset.features[row, symbol_index, :],
            dtype=np.float64,
        )
        if not np.isfinite(selected[indices]).all():
            return None
        row_features = selected[indices].reshape(1, -1)
        forecast = float(self._score_rows(vintage.model, row_features)[0])
        direct = float(vintage.model.predict(self._full_vector(selected[indices])))
        if not math.isclose(
            direct, forecast, rel_tol=_AGREEMENT_RTOL, abs_tol=_AGREEMENT_ATOL
        ):
            raise ValueError(
                "model.predict and fitter.predict_rows disagree on the decision row"
            )
        fit_ns = int(vintage.fit_time.astype(np.int64))
        return ForecastPacket(
            symbol=dataset.symbols[symbol_index],
            symbol_index=symbol_index,
            horizon_hours=self._horizon_hours,
            decision_time=np.datetime64(decision_ns, "ns"),
            forecast_available_at=np.datetime64(max(fit_ns, available_ns), "ns"),
            return_unit=LOG_RETURN_UNIT,
            point_forecast=forecast,
            residual_variance=vintage.residual_variance,
            vintage_id=vintage.vintage_id,
            fit_time=vintage.fit_time,
            fit_cutoff=vintage.fit_cutoff,
            max_training_label_available_at=vintage.max_training_label_available_at,
            n_training_samples=vintage.n_training_samples,
            fitter_identity=vintage.fitter_identity,
            training_digest=vintage.training_digest,
            model_digest=vintage.model_digest,
            feature_digest=_array_digest(selected[indices]),
        )

    def packets_at(
        self,
        decision_time: np.datetime64,
    ) -> tuple[ForecastPacket | None, ...]:
        """One entry per dataset symbol (``None`` where features are unusable)."""

        return tuple(
            self.packet(symbol_index, decision_time)
            for symbol_index in range(self._dataset.n_symbols)
        )

    def iter_packets(
        self,
        decision_times: Iterable[np.datetime64],
    ) -> Iterator[tuple[np.datetime64, tuple[ForecastPacket | None, ...]]]:
        """Yield ``(decision_time, packets_at(decision_time))`` in the given order."""

        for decision_time in decision_times:
            yield decision_time, self.packets_at(decision_time)


__all__ = [
    "LOG_RETURN_UNIT",
    "ForecastPacket",
    "ForecastVintage",
    "LightGBMForecastFitter",
    "PrequentialForecastFitter",
    "PrequentialForecastModel",
    "PrequentialForecastSource",
    "PrequentialSchedule",
    "RidgeForecastFitter",
]
