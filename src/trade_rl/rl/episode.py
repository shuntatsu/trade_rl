"""Pure episode-range and reward pre-roll helpers."""

from __future__ import annotations

from dataclasses import dataclass

from trade_rl.data.market import MarketDataset


@dataclass(frozen=True, slots=True)
class EpisodeRange:
    """Half-open episode range with a causal reward-history prefix."""

    start: int
    reward_start: int
    stop: int

    def __post_init__(self) -> None:
        if not 0 <= self.start <= self.reward_start < self.stop:
            raise ValueError("episode range indices are inconsistent")


def resolve_episode_range(
    *,
    requested_start: int,
    episode_bars: int,
    reward_preroll_bars: int,
    dataset_bars: int,
) -> EpisodeRange:
    """Resolve a fixed-cadence range and fail when the full pre-roll is absent."""

    for field_name, value in (
        ("requested_start", requested_start),
        ("episode_bars", episode_bars),
        ("reward_preroll_bars", reward_preroll_bars),
        ("dataset_bars", dataset_bars),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{field_name} must be an integer")
    if requested_start < 0:
        raise ValueError("requested_start must be non-negative")
    if episode_bars <= 0 or dataset_bars <= 0:
        raise ValueError("episode and dataset sizes must be positive")
    if reward_preroll_bars < 0:
        raise ValueError("reward_preroll_bars must be non-negative")
    start = requested_start - reward_preroll_bars
    if start < 0:
        raise ValueError("insufficient reward pre-roll before requested start")
    stop = requested_start + episode_bars
    if stop > dataset_bars:
        raise ValueError("episode does not fit inside the dataset")
    return EpisodeRange(start=start, reward_start=requested_start, stop=stop)


def complete_reward_history_steps(
    dataset: MarketDataset,
    *,
    reward_start: int,
    window_hours: float,
    window_steps: int,
) -> int:
    """Return the full window size only when complete causal history exists."""

    if (
        isinstance(window_steps, bool)
        or not isinstance(window_steps, int)
        or window_steps <= 0
    ):
        raise ValueError("window_steps must be a positive integer")
    try:
        dataset.lookback_index(reward_start, window_hours)
    except ValueError:
        return 0
    return window_steps


def minimum_reward_start_index(
    dataset: MarketDataset,
    *,
    signal_minimum: int,
    window_hours: float,
) -> int:
    """Find the earliest decision index with signal history plus a full window."""

    if (
        isinstance(signal_minimum, bool)
        or not isinstance(signal_minimum, int)
        or signal_minimum < 0
        or signal_minimum >= dataset.n_bars
    ):
        raise ValueError("signal_minimum is outside the dataset")
    if not isinstance(window_hours, (int, float)) or window_hours <= 0.0:
        raise ValueError("window_hours must be positive")
    for candidate in range(signal_minimum + 1, dataset.n_bars):
        try:
            history_start = dataset.lookback_index(candidate, float(window_hours))
        except ValueError:
            continue
        if history_start >= signal_minimum:
            return candidate
    raise ValueError("dataset cannot provide a complete reward pre-roll window")
