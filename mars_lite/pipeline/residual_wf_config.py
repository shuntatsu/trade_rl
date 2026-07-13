from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from mars_lite.data.data_utils import TF_TO_MINUTES


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _hash_array(digest: Any, value: object) -> None:
    array = np.ascontiguousarray(np.asarray(value))
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(repr(tuple(array.shape)).encode("utf-8"))
    if array.dtype.hasobject:
        digest.update(
            "\x1f".join(str(item) for item in array.reshape(-1)).encode("utf-8")
        )
    else:
        digest.update(array.tobytes(order="C"))


def feature_set_identity(fs: object) -> str:
    """Return a deterministic content identity for the exact FeatureSet inputs."""

    digest = hashlib.sha256()
    for sequence_name in ("symbols", "feature_names", "global_feature_names"):
        values = tuple(str(value) for value in getattr(fs, sequence_name))
        digest.update(sequence_name.encode("utf-8"))
        digest.update("\x1f".join(values).encode("utf-8"))
    for array_name in (
        "timestamps",
        "features",
        "global_features",
        "close",
        "open_next",
        "funding_rate",
    ):
        digest.update(array_name.encode("utf-8"))
        _hash_array(digest, getattr(fs, array_name))
    return digest.hexdigest()


@dataclass(frozen=True)
class ResidualWalkForwardConfig:
    requested_decision_every: int
    effective_decision_every: int
    requested_ensemble_size: int
    effective_ensemble_size: int
    requested_n_seeds: int
    requested_folds: int
    requested_purge_bars: int
    effective_purge_bars: int
    base_timeframe: str
    bars_per_year: int
    horizon: int
    run_tier: str
    signal_model: str
    fee_profile: str
    dataset_identity: str
    git_sha: str | None

    @classmethod
    def from_args(
        cls,
        args: object,
        *,
        dataset_identity: str,
    ) -> "ResidualWalkForwardConfig":
        horizon = _positive_int(getattr(args, "horizon"), name="horizon")
        requested_decision_every = _positive_int(
            getattr(args, "decision_every", 1),
            name="decision_every",
        )
        requested_ensemble = _positive_int(
            getattr(args, "ensemble", 1),
            name="ensemble",
        )
        requested_n_seeds = _positive_int(
            getattr(args, "n_seeds", 1),
            name="n_seeds",
        )
        requested_folds = _positive_int(
            getattr(args, "folds", 3),
            name="folds",
        )
        requested_purge = _positive_int(
            getattr(args, "purge_bars", 24),
            name="purge_bars",
        )
        effective_decision_every = requested_decision_every
        if (
            requested_decision_every == 1
            and bool(getattr(args, "scan_horizons", False))
            and horizon > 1
        ):
            effective_decision_every = max(1, horizon // 2)

        base_timeframe = str(getattr(args, "base_timeframe", "1h"))
        if base_timeframe not in TF_TO_MINUTES:
            raise ValueError(f"unsupported base_timeframe: {base_timeframe}")
        bars_per_year = int(24 * 60 / TF_TO_MINUTES[base_timeframe] * 365)
        if not dataset_identity:
            raise ValueError("dataset_identity must not be empty")

        git_sha_value = getattr(args, "git_sha", None)
        git_sha = str(git_sha_value) if git_sha_value else None
        return cls(
            requested_decision_every=requested_decision_every,
            effective_decision_every=effective_decision_every,
            requested_ensemble_size=requested_ensemble,
            effective_ensemble_size=max(requested_ensemble, requested_n_seeds),
            requested_n_seeds=requested_n_seeds,
            requested_folds=requested_folds,
            requested_purge_bars=requested_purge,
            effective_purge_bars=max(requested_purge, horizon, 24),
            base_timeframe=base_timeframe,
            bars_per_year=bars_per_year,
            horizon=horizon,
            run_tier=str(getattr(args, "run_tier", "research")),
            signal_model=str(getattr(args, "signal_model", "gbm")),
            fee_profile=str(getattr(args, "fee_profile", "taker")),
            dataset_identity=dataset_identity,
            git_sha=git_sha,
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
