"""Episode contract selection for the residual market environment."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.rl.environment_config import ResidualMarketEnvConfig


@dataclass(frozen=True, slots=True)
class EpisodeContract:
    """Resolved causal episode boundaries."""

    start_index: int
    end_index: int
    hours: float


class EpisodeContractSampler:
    """Resolve and sample valid episode contracts without environment state access."""

    def __init__(
        self,
        dataset: MarketDataset,
        config: ResidualMarketEnvConfig,
        *,
        minimum_start_index: int,
    ) -> None:
        if not 0 <= minimum_start_index < dataset.n_bars:
            raise ValueError("minimum_start_index is outside the dataset")
        self.dataset = dataset
        self.config = config
        self.minimum_start_index = minimum_start_index
        self._valid_start_cache: dict[tuple[float, int | None], np.ndarray] = {}

    def episode_end(self, start: int, *, hours: float, bars: int | None) -> int:
        if bars is not None:
            end = start + bars
            if end >= self.dataset.n_bars:
                raise ValueError("episode does not fit inside the dataset")
            return end
        end = self.dataset.forward_index(start, hours)
        if self.dataset.elapsed_hours(start, end) + 1e-9 < hours:
            raise ValueError("episode duration does not fit inside the dataset")
        return end

    def valid_starts(self, *, hours: float, bars: int | None) -> np.ndarray:
        key = (float(hours), bars)
        cached = self._valid_start_cache.get(key)
        if cached is not None:
            return cached.copy()
        valid: list[int] = []
        for start in range(self.minimum_start_index, self.dataset.n_bars - 1):
            try:
                self.episode_end(start, hours=hours, bars=bars)
            except ValueError:
                continue
            valid.append(start)
        if not valid:
            raise ValueError("dataset is too short for the configured episode")
        resolved = np.asarray(valid, dtype=np.int64)
        self._valid_start_cache[key] = resolved
        return resolved.copy()

    def sample(
        self,
        options: Mapping[str, object],
        rng: np.random.Generator,
    ) -> EpisodeContract:
        raw_hours = options.get("episode_hours")
        raw_bars = options.get("episode_bars", self.config.episode_bars)
        if raw_hours is not None and raw_bars is not None:
            raise ValueError(
                "episode_hours and episode_bars reset options are mutually exclusive"
            )
        if raw_hours is not None:
            if isinstance(raw_hours, bool) or not isinstance(raw_hours, int | float):
                raise ValueError("episode_hours option must be numeric")
            hours = float(raw_hours)
        elif self.config.episode_hour_choices:
            viable_hours: list[float] = []
            for choice in self.config.episode_hour_choices:
                try:
                    self.valid_starts(hours=float(choice), bars=None)
                except ValueError:
                    continue
                viable_hours.append(float(choice))
            if not viable_hours:
                raise ValueError(
                    "none of the configured episode durations fit the dataset"
                )
            hours = float(rng.choice(viable_hours))
        else:
            hours = self.config.episode_hours
        if not math.isfinite(hours) or hours <= 0.0:
            raise ValueError("episode_hours option must be finite and positive")

        bars: int | None
        if raw_bars is None:
            bars = None
        else:
            if (
                isinstance(raw_bars, bool)
                or not isinstance(raw_bars, int)
                or raw_bars <= 0
            ):
                raise ValueError("episode_bars option must be a positive integer")
            bars = raw_bars

        if "start_idx" in options:
            raw_start = options["start_idx"]
            if isinstance(raw_start, bool) or not isinstance(raw_start, int):
                raise ValueError("start_idx must be an integer")
            start = raw_start
            if start < self.minimum_start_index:
                raise ValueError(
                    "start_idx does not have sufficient causal or signal history"
                )
            end = self.episode_end(start, hours=hours, bars=bars)
        else:
            valid_starts = self.valid_starts(hours=hours, bars=bars)
            if self.config.episode_sampling_mode in {
                "regime_balanced",
                "stress_tail",
            }:
                feature_index = self.config.regime_feature_index
                if feature_index >= len(self.dataset.global_feature_names):
                    raise ValueError("regime_feature_index is outside global features")
                available = self.dataset.resolved_array("global_feature_available")[
                    valid_starts,
                    feature_index,
                ]
                candidate_starts = valid_starts[available]
                if candidate_starts.size == 0:
                    raise ValueError(
                        "episode sampling feature is unavailable for every valid start"
                    )
                regime_values = self.dataset.global_features[
                    candidate_starts,
                    feature_index,
                ]
                if self.config.episode_sampling_mode == "stress_tail":
                    threshold = float(
                        np.quantile(
                            np.abs(regime_values),
                            self.config.stress_quantile,
                        )
                    )
                    stressed = candidate_starts[np.abs(regime_values) >= threshold]
                    if stressed.size:
                        candidate_starts = stressed
                else:
                    quantiles = np.unique(
                        np.quantile(
                            regime_values,
                            np.linspace(0.0, 1.0, self.config.regime_bins + 1),
                        )
                    )
                    if quantiles.size > 2:
                        bins = np.digitize(
                            regime_values,
                            quantiles[1:-1],
                            right=True,
                        )
                        chosen_bin = int(rng.choice(np.unique(bins)))
                        candidate_starts = candidate_starts[bins == chosen_bin]
                start = int(rng.choice(candidate_starts))
            else:
                start = int(rng.choice(valid_starts))
            end = self.episode_end(start, hours=hours, bars=bars)

        return EpisodeContract(
            start_index=start,
            end_index=end,
            hours=self.dataset.elapsed_hours(start, end),
        )
