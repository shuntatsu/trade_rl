from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class FrozenResidualAlpha:
    model: str
    horizon: int
    target: str
    feature_names: tuple[str, ...]
    symbols: tuple[str, ...]
    fit_cutoff_index: int
    fit_cutoff_timestamp: str
    prediction_mean: float
    prediction_std: float
    gate_result: dict[str, Any]
    dataset_identity: str
    ridge_weights: tuple[float, ...] | None = None
    gbm_model_string: str | None = None
    feature_mean: tuple[float, ...] | None = None
    feature_std: tuple[float, ...] | None = None

    @property
    def enabled(self) -> bool:
        return self.gate_result.get("passed") is True

    @classmethod
    def fit(
        cls,
        fs,
        *,
        horizon: int,
        target: str = "cs_demean",
        model: str = "gbm",
        gate_result: Mapping[str, Any] | None = None,
    ) -> "FrozenResidualAlpha":
        if horizon <= 0 or horizon >= fs.n_bars:
            raise ValueError("horizon must be positive and smaller than the dataset")
        if model not in {"ridge", "gbm"}:
            raise ValueError("model must be ridge or gbm")

        if gate_result is None:
            from mars_lite.features.signal_check import run_signal_check

            gate = run_signal_check(fs, horizon=horizon, target=target).to_dict()
        else:
            gate = dict(gate_result)
        cutoff = fs.n_bars - horizon
        cutoff_timestamp = str(np.asarray(fs.timestamps)[cutoff - 1])
        identity = _dataset_identity(fs, cutoff)

        if gate.get("passed") is not True:
            return cls(
                model=model,
                horizon=horizon,
                target=target,
                feature_names=tuple(fs.feature_names),
                symbols=tuple(fs.symbols),
                fit_cutoff_index=cutoff,
                fit_cutoff_timestamp=cutoff_timestamp,
                prediction_mean=0.0,
                prediction_std=1.0,
                gate_result=gate,
                dataset_identity=identity,
            )

        from mars_lite.features.signal_check import _pool

        X, y, _ = _pool(fs, horizon, target=target)
        if len(X) == 0:
            raise ValueError("no finite training samples for residual alpha")

        ridge_weights = None
        gbm_model_string = None
        feature_mean = None
        feature_std = None
        if model == "ridge":
            from mars_lite.features.signal_check import _ridge_fit, _ridge_predict

            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)
            std = np.where(std > 1e-12, std, 1.0)
            standardized = (X - mean) / std
            weights = _ridge_fit(standardized, y)
            train_pred = _ridge_predict(standardized, weights)
            ridge_weights = tuple(float(value) for value in weights)
            feature_mean = tuple(float(value) for value in mean)
            feature_std = tuple(float(value) for value in std)
        else:
            from mars_lite.features.gbm_forecaster import fit_gbm, predict_gbm

            booster = fit_gbm(X, y)
            train_pred = predict_gbm(booster, X)
            gbm_model_string = booster.model_to_string()

        prediction_mean = float(np.mean(train_pred))
        prediction_std = float(np.std(train_pred))
        if prediction_std <= 1e-12:
            prediction_std = 1.0
        return cls(
            model=model,
            horizon=horizon,
            target=target,
            feature_names=tuple(fs.feature_names),
            symbols=tuple(fs.symbols),
            fit_cutoff_index=cutoff,
            fit_cutoff_timestamp=cutoff_timestamp,
            prediction_mean=prediction_mean,
            prediction_std=prediction_std,
            gate_result=gate,
            dataset_identity=identity,
            ridge_weights=ridge_weights,
            gbm_model_string=gbm_model_string,
            feature_mean=feature_mean,
            feature_std=feature_std,
        )

    def predict_at(self, fs, t: int) -> np.ndarray:
        if tuple(fs.feature_names) != self.feature_names:
            raise ValueError(
                "feature order does not match frozen residual alpha artifact"
            )
        if tuple(fs.symbols) != self.symbols:
            raise ValueError(
                "symbol order does not match frozen residual alpha artifact"
            )
        if not 0 <= t < fs.n_bars:
            raise IndexError("t out of range")
        if not self.enabled:
            return np.zeros(fs.n_symbols, dtype=np.float64)

        features = np.asarray(fs.features[t], dtype=np.float64)
        if not np.all(np.isfinite(features)):
            raise ValueError("features contain non-finite values")
        if self.model == "ridge":
            if self.ridge_weights is None:
                raise ValueError("ridge artifact is missing weights")
            if self.feature_mean is None or self.feature_std is None:
                raise ValueError("ridge artifact is missing preprocessing statistics")
            mean = np.asarray(self.feature_mean, dtype=np.float64)
            std = np.asarray(self.feature_std, dtype=np.float64)
            if mean.shape != (features.shape[1],) or std.shape != (features.shape[1],):
                raise ValueError("ridge preprocessing shape does not match features")
            standardized = (features - mean) / std
            weights = np.asarray(self.ridge_weights, dtype=np.float64)
            Xb = np.hstack([standardized, np.ones((len(standardized), 1))])
            raw = Xb @ weights
        else:
            if self.gbm_model_string is None:
                raise ValueError("GBM artifact is missing model data")
            import lightgbm as lgb

            booster = lgb.Booster(model_str=self.gbm_model_string)
            raw = np.asarray(booster.predict(features), dtype=np.float64)

        normalized = (raw - self.prediction_mean) / self.prediction_std
        centered = normalized - normalized.mean()
        gross = float(np.abs(centered).sum())
        return centered / gross if gross > 1.0 else centered

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "FrozenResidualAlpha":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        payload["feature_names"] = tuple(payload["feature_names"])
        payload["symbols"] = tuple(payload["symbols"])
        for key in ("ridge_weights", "feature_mean", "feature_std"):
            if payload.get(key) is not None:
                payload[key] = tuple(payload[key])
        return cls(**payload)


def _update_identity_array(
    digest: "hashlib._Hash", name: str, value: np.ndarray, *, dtype: str
) -> None:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    descriptor = json.dumps(
        {"name": name, "dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(len(descriptor).to_bytes(8, "big"))
    digest.update(descriptor)
    payload = array.tobytes(order="C")
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _dataset_identity(fs, cutoff: int) -> str:
    if cutoff <= 0 or cutoff >= fs.n_bars:
        raise ValueError("dataset identity cutoff is outside the dataset")
    metadata = {
        "schema": "frozen-residual-alpha-dataset-v2",
        "symbols": list(fs.symbols),
        "feature_names": list(fs.feature_names),
        "cutoff": int(cutoff),
    }
    digest = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    timestamps = np.asarray(fs.timestamps).astype("datetime64[ns]").astype("<i8")
    if np.any(timestamps == np.iinfo(np.int64).min):
        raise ValueError("timestamps contain NaT")
    _update_identity_array(digest, "timestamps", timestamps, dtype="<i8")
    _update_identity_array(
        digest,
        "training_features",
        np.asarray(fs.features)[:cutoff],
        dtype="<f8",
    )
    _update_identity_array(
        digest,
        "target_close_history",
        np.asarray(fs.close),
        dtype="<f8",
    )
    return digest.hexdigest()
