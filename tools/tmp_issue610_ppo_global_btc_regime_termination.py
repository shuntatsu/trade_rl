"""Result-blind hard/economic termination checks for Issue #610."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, TypeAlias, cast


class _RunLike(Protocol):
    summary: object


RunMap: TypeAlias = Mapping[int, _RunLike]


def _mapping(value: object) -> Mapping[str, object] | None:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        return None
    return cast(Mapping[str, object], value)


def _symbol_roster_is_exact(run: _RunLike, symbols: tuple[str, ...]) -> bool:
    summary = _mapping(run.summary)
    if summary is None or summary.get("symbols") != list(symbols):
        return False
    by_symbol = summary.get("by_symbol")
    if not isinstance(by_symbol, list) or len(by_symbol) != len(symbols):
        return False
    for expected_symbol, entry in zip(symbols, by_symbol, strict=True):
        mapping = _mapping(entry)
        if mapping is None or mapping.get("symbol") != expected_symbol:
            return False
    return True


def _termination_reasons(
    run: _RunLike,
    *,
    symbol: str,
    symbols: tuple[str, ...],
) -> tuple[str, ...] | None:
    summary = _mapping(run.summary)
    if summary is None:
        return None
    by_symbol = summary.get("by_symbol")
    if not isinstance(by_symbol, list):
        return None
    try:
        index = symbols.index(symbol)
        symbol_entry = _mapping(by_symbol[index])
    except (IndexError, ValueError):
        return None
    if symbol_entry is None or symbol_entry.get("symbol") != symbol:
        return None
    strategies = symbol_entry.get("strategies")
    if not isinstance(strategies, list):
        return None
    ppo_entries = [
        mapping
        for value in strategies
        if (mapping := _mapping(value)) is not None and mapping.get("name") == "ppo"
    ]
    if len(ppo_entries) != 1:
        return None
    ppo = ppo_entries[0]
    diagnostics = _mapping(ppo.get("diagnostics"))
    metrics = _mapping(ppo.get("metrics"))
    if diagnostics is None or metrics is None:
        return None

    reasons = diagnostics.get("termination_reasons")
    if not isinstance(reasons, (list, tuple)):
        return None
    if any(not isinstance(reason, str) or not reason for reason in reasons):
        return None
    normalized = tuple(cast(str, reason) for reason in reasons)
    if len(set(normalized)) != len(normalized):
        return None

    count = metrics.get("termination_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        return None
    if count != len(normalized):
        return None
    return normalized


def validate_no_new_ppo_terminations(
    baseline: RunMap,
    candidate: RunMap,
    *,
    seeds: tuple[int, ...],
    symbols: tuple[str, ...],
) -> tuple[str, ...]:
    """Fail closed unless candidate PPO introduces no new termination reason."""

    violations: list[str] = []
    if tuple(baseline) != seeds:
        violations.append("baseline termination seed roster/order mismatch")
    if tuple(candidate) != seeds:
        violations.append("candidate termination seed roster/order mismatch")
    if violations:
        return tuple(violations)

    for seed in seeds:
        baseline_run = baseline[seed]
        candidate_run = candidate[seed]
        baseline_roster_ok = _symbol_roster_is_exact(baseline_run, symbols)
        candidate_roster_ok = _symbol_roster_is_exact(candidate_run, symbols)
        if not baseline_roster_ok:
            violations.append(
                f"baseline termination symbol roster/order mismatch: seed={seed}"
            )
        if not candidate_roster_ok:
            violations.append(
                f"candidate termination symbol roster/order mismatch: seed={seed}"
            )
        if not baseline_roster_ok or not candidate_roster_ok:
            continue

        for symbol in symbols:
            baseline_reasons = _termination_reasons(
                baseline_run,
                symbol=symbol,
                symbols=symbols,
            )
            candidate_reasons = _termination_reasons(
                candidate_run,
                symbol=symbol,
                symbols=symbols,
            )
            if baseline_reasons is None:
                violations.append(
                    f"baseline PPO termination evidence malformed: seed={seed} {symbol}"
                )
                continue
            if candidate_reasons is None:
                violations.append(
                    "candidate PPO termination evidence malformed: "
                    f"seed={seed} {symbol}"
                )
                continue

            baseline_set = set(baseline_reasons)
            for reason in candidate_reasons:
                if reason not in baseline_set:
                    violations.append(
                        f"new PPO termination: seed={seed} {symbol} reason={reason}"
                    )
    return tuple(violations)


__all__ = ["validate_no_new_ppo_terminations"]
