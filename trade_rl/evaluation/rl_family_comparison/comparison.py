"""Paired, development-only decision oracle for PPO and A2C replay cells."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from statistics import median
from typing import Any

from trade_rl.evaluation.rl_family_comparison.cells import validate_family_rows
from trade_rl.evaluation.rl_family_comparison.contract import (
    fixed_comparison_contract,
)


def _hard_guard_violations(
    rows: Mapping[tuple[str, int, str, str], Mapping[str, Any]],
    *,
    family: str,
) -> list[dict[str, Any]]:
    maximum_drawdown = fixed_comparison_contract()["evaluation"]["maximum_drawdown"]
    violations: list[dict[str, Any]] = []
    for (row_family, seed, symbol, scenario), cell in rows.items():
        if row_family != family:
            continue
        reasons: list[str] = []
        if cell["ledger_max_drawdown"] > maximum_drawdown:
            reasons.append("drawdown_exceeded")
        if cell["termination_reasons"]:
            reasons.append("terminated")
        if not cell["terminal_flat"]:
            reasons.append("terminal_position_not_flat")
        if cell["active_order_remainders"]:
            reasons.append("active_order_remainder")
        if reasons:
            violations.append(
                {
                    "seed": seed,
                    "symbol": symbol,
                    "scenario": scenario,
                    "reasons": reasons,
                }
            )
    return violations


def _absolute_symbols(
    rows: Mapping[tuple[str, int, str, str], Mapping[str, Any]],
    *,
    family: str,
    symbols: tuple[str, ...],
) -> list[str]:
    contract = fixed_comparison_contract()
    seeds = tuple(contract["seeds"])
    scenarios = tuple(contract["evaluation"]["scenarios"])
    years = tuple(contract["evaluation"]["years"])
    required_votes = contract["admission"]["absolute_seed_votes_per_symbol"]
    passing: list[str] = []
    for symbol in symbols:
        passing_seeds = 0
        for seed in seeds:
            seed_pass = all(
                rows[(family, seed, symbol, scenario)]["total_return"] > 0.0
                and all(
                    value > 0.0
                    for value in rows[(family, seed, symbol, scenario)][
                        "year_returns"
                    ].values()
                )
                for scenario in scenarios
            )
            passing_seeds += int(seed_pass)
        medians_pass = all(
            median(
                rows[(family, seed, symbol, scenario)]["total_return"] for seed in seeds
            )
            > 0.0
            and all(
                median(
                    rows[(family, seed, symbol, scenario)]["year_returns"][year]
                    for seed in seeds
                )
                > 0.0
                for year in years
            )
            for scenario in scenarios
        )
        if passing_seeds >= required_votes and medians_pass:
            passing.append(symbol)
    return passing


def _relative_symbols(
    rows: Mapping[tuple[str, int, str, str], Mapping[str, Any]],
    *,
    symbols: tuple[str, ...],
) -> list[str]:
    contract = fixed_comparison_contract()
    seeds = tuple(contract["seeds"])
    scenarios = tuple(contract["evaluation"]["scenarios"])
    years = tuple(contract["evaluation"]["years"])
    required_votes = contract["admission"]["paired_seed_votes_per_symbol"]
    passing: list[str] = []
    for symbol in symbols:
        passing_seeds = 0
        for seed in seeds:
            seed_pass = all(
                rows[("a2c", seed, symbol, scenario)]["total_return"]
                > rows[("ppo", seed, symbol, scenario)]["total_return"]
                and all(
                    rows[("a2c", seed, symbol, scenario)]["year_returns"][year]
                    > rows[("ppo", seed, symbol, scenario)]["year_returns"][year]
                    for year in years
                )
                for scenario in scenarios
            )
            passing_seeds += int(seed_pass)
        medians_pass = all(
            median(
                rows[("a2c", seed, symbol, scenario)]["total_return"]
                - rows[("ppo", seed, symbol, scenario)]["total_return"]
                for seed in seeds
            )
            > 0.0
            and all(
                median(
                    rows[("a2c", seed, symbol, scenario)]["year_returns"][year]
                    - rows[("ppo", seed, symbol, scenario)]["year_returns"][year]
                    for seed in seeds
                )
                > 0.0
                for year in years
            )
            for scenario in scenarios
        )
        if passing_seeds >= required_votes and medians_pass:
            passing.append(symbol)
    return passing


def _compare_verified_family_cells(
    cells: Sequence[Mapping[str, Any]],
    *,
    symbols: tuple[str, ...],
) -> dict[str, Any]:
    """Apply the frozen oracle to rows returned by the G3 evidence reader."""

    if (
        not isinstance(symbols, tuple)
        or len(symbols) != 5
        or any(not isinstance(symbol, str) or not symbol for symbol in symbols)
        or len(set(symbols)) != len(symbols)
    ):
        raise ValueError("symbols must be the five unique frozen Dataset symbols")
    rows = validate_family_rows(cells, symbols=symbols)
    violations = {
        family: _hard_guard_violations(rows, family=family) for family in ("ppo", "a2c")
    }
    absolute = {
        family: _absolute_symbols(rows, family=family, symbols=symbols)
        for family in ("ppo", "a2c")
    }
    relative = _relative_symbols(rows, symbols=symbols)
    jointly_qualified = set(absolute["a2c"]) & set(absolute["ppo"]) & set(relative)
    a2c_common = [symbol for symbol in symbols if symbol in jointly_qualified]
    required_symbols = fixed_comparison_contract()["admission"][
        "required_common_symbols"
    ]
    family_qualified = {
        family: not violations[family] and len(absolute[family]) >= required_symbols
        for family in ("ppo", "a2c")
    }
    a2c_uplift_pass = len(a2c_common) >= required_symbols
    if family_qualified["ppo"] and family_qualified["a2c"]:
        nominated = ["a2c"] if a2c_uplift_pass else ["ppo", "a2c"]
    elif family_qualified["a2c"]:
        nominated = ["a2c"]
    elif family_qualified["ppo"]:
        nominated = ["ppo"]
    else:
        nominated = []

    decisions = {
        ("a2c",): "NOMINATE_A2C_FOR_NEXT_RESEARCH",
        ("ppo",): "NOMINATE_PPO_FOR_NEXT_RESEARCH",
        ("ppo", "a2c"): "NOMINATE_BOTH_FOR_NEXT_RESEARCH",
        (): "NO_QUALIFIED_RL_FAMILY",
    }
    return {
        "decision": decisions[tuple(nominated)],
        "nominated_families": nominated,
        "a2c_relative_uplift_pass": a2c_uplift_pass,
        "family_qualified": family_qualified,
        "family_absolute_symbols": absolute,
        "family_hard_guards_pass": {
            family: not violations[family] for family in ("ppo", "a2c")
        },
        "hard_guard_violations": violations,
        "a2c_relative_symbols": relative,
        "a2c_common_symbols": a2c_common,
        "paper_authorized": False,
        "production_eligible": False,
        "unused_data_accessed": False,
    }
