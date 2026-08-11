from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class UniversalZeroShotPair:
    symbol: str
    fold: int
    seed: int
    candidate_return: float
    baseline_return: float
    hard_safety_violations: int = 0

    @property
    def excess_return(self) -> float:
        return self.candidate_return - self.baseline_return


@dataclass(frozen=True)
class UniversalZeroShotSummary:
    pair_count: int
    symbol_count: int
    mean_excess_return: float
    worst_symbol_excess_return: float
    worst_seed_excess_return: float
    pass_fraction: float
    hard_safety_violations: int


def summarize_zero_shot_pairs(
    pairs: Iterable[UniversalZeroShotPair],
    *,
    pair_pass_threshold: float = 0.0,
) -> UniversalZeroShotSummary:
    rows = tuple(pairs)
    if not rows:
        raise ValueError("zero-shot evaluation requires at least one paired result")
    identities = {(row.symbol, row.fold, row.seed) for row in rows}
    if len(identities) != len(rows):
        raise ValueError("zero-shot pair identities must be unique")

    by_symbol: dict[str, list[float]] = {}
    by_seed: dict[int, list[float]] = {}
    for row in rows:
        by_symbol.setdefault(row.symbol, []).append(row.excess_return)
        by_seed.setdefault(row.seed, []).append(row.excess_return)

    symbol_means = {
        symbol: sum(values) / len(values) for symbol, values in by_symbol.items()
    }
    seed_means = {seed: sum(values) / len(values) for seed, values in by_seed.items()}
    excess = [row.excess_return for row in rows]
    return UniversalZeroShotSummary(
        pair_count=len(rows),
        symbol_count=len(by_symbol),
        mean_excess_return=sum(excess) / len(excess),
        worst_symbol_excess_return=min(symbol_means.values()),
        worst_seed_excess_return=min(seed_means.values()),
        pass_fraction=sum(value > pair_pass_threshold for value in excess) / len(excess),
        hard_safety_violations=sum(row.hard_safety_violations for row in rows),
    )


def passes_zero_shot_gate(
    summary: UniversalZeroShotSummary,
    *,
    minimum_mean_excess_return: float = 0.0,
    minimum_worst_symbol_excess_return: float = 0.0,
    minimum_worst_seed_excess_return: float = 0.0,
    minimum_pass_fraction: float = 0.5,
) -> bool:
    return (
        summary.hard_safety_violations == 0
        and summary.mean_excess_return > minimum_mean_excess_return
        and summary.worst_symbol_excess_return > minimum_worst_symbol_excess_return
        and summary.worst_seed_excess_return > minimum_worst_seed_excess_return
        and summary.pass_fraction >= minimum_pass_fraction
    )
