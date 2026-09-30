from __future__ import annotations

import math
from typing import Any

import pytest

from trade_rl.evaluation.rl_family_comparison.comparison import (
    _compare_verified_family_cells,
)
from trade_rl.evaluation.rl_family_comparison.contract import (
    fixed_comparison_contract,
)

_SYMBOLS = tuple(f"S{index}" for index in range(5))


def _cells(
    *, a2c_return: float = 0.04, ppo_return: float = 0.02
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family, total_return in (("ppo", ppo_return), ("a2c", a2c_return)):
        annual_return = math.sqrt(1.0 + total_return) - 1.0
        for seed in range(5):
            for symbol in _SYMBOLS:
                for scenario in ("base", "cost_2x", "latency_1"):
                    rows.append(
                        {
                            "family": family,
                            "seed": seed,
                            "symbol": symbol,
                            "scenario": scenario,
                            "interval_count": 17_544,
                            "total_return": total_return,
                            "year_returns": {
                                "2023": annual_return,
                                "2024": annual_return,
                            },
                            "ledger_max_drawdown": 0.05,
                            "termination_reasons": [],
                            "terminal_flat": True,
                            "active_order_remainders": [],
                        }
                    )
    return rows


def _set_cell_return(
    rows: list[dict[str, Any]],
    *,
    family: str,
    seed: int,
    symbol: str,
    scenario: str,
    total_return: float,
) -> None:
    cell = next(
        row
        for row in rows
        if (row["family"], row["seed"], row["symbol"], row["scenario"])
        == (family, seed, symbol, scenario)
    )
    annual_return = math.sqrt(1.0 + total_return) - 1.0
    cell["total_return"] = total_return
    cell["year_returns"] = {"2023": annual_return, "2024": annual_return}


def test_protocol_pins_equal_budgets_cutoff_stresses_and_nonproduction_scope() -> None:
    contract = fixed_comparison_contract()
    assert (
        contract["schema"],
        tuple(contract["families"]),
        tuple(contract["seeds"]),
        contract["training"]["requested_timesteps"],
    ) == (
        "rl_family_comparison_protocol_v1",
        ("ppo", "a2c"),
        (0, 1, 2, 3, 4),
        256_000,
    )
    assert contract["training"]["effective_timesteps"] == dict.fromkeys(
        ("ppo", "a2c"), 256_000
    )
    assert (
        contract["fit_scope"]["latest_timestamp_exclusive"]
        == contract["evaluation"]["start_timestamp"]
    )
    assert set(contract["evaluation"]["scenarios"]) == {
        "base",
        "cost_2x",
        "latency_1",
    }
    assert all(
        contract["training"][field] is True
        for field in (
            "same_market_data_fit_scope",
            "same_observation_action_reward_environment",
            "same_training_risk",
            "same_execution_costs",
            "settle_terminal_position",
        )
    )
    assert contract["evaluation"]["independent_10000_account_per_symbol"] is True
    assert (
        contract["training"]["same_initial_capital"]
        == contract["evaluation"]["initial_capital"]
        == 10_000.0
    )
    assert (
        contract["training"]["same_per_symbol_gross_budget"]
        == contract["evaluation"]["gross_budget"]
        == 0.1
    )
    assert contract["evaluation"]["scenarios"] == {
        "base": {"cost_multiplier": 1.0, "latency_bars": 0},
        "cost_2x": {"cost_multiplier": 2.0, "latency_bars": 0},
        "latency_1": {"cost_multiplier": 1.0, "latency_bars": 1},
    }
    assert not any(
        (
            contract["paper_authorized"],
            contract["production_eligible"],
            contract["unused_data_accessed"],
        )
    )
    assert contract["evaluation"]["maximum_drawdown"] == 0.20


@pytest.mark.parametrize(
    ("a2c_return", "ppo_return", "decision", "nominated"),
    [
        (0.04, 0.02, "NOMINATE_A2C_FOR_NEXT_RESEARCH", ["a2c"]),
        (0.04, -0.02, "NOMINATE_A2C_FOR_NEXT_RESEARCH", ["a2c"]),
        (0.02, 0.04, "NOMINATE_BOTH_FOR_NEXT_RESEARCH", ["ppo", "a2c"]),
        (-0.02, 0.04, "NOMINATE_PPO_FOR_NEXT_RESEARCH", ["ppo"]),
        (-0.04, -0.02, "NO_QUALIFIED_RL_FAMILY", []),
    ],
)
def test_families_qualify_independently_and_relative_uplift_only_selects_between_both(
    a2c_return: float,
    ppo_return: float,
    decision: str,
    nominated: list[str],
) -> None:
    result = _compare_verified_family_cells(
        _cells(a2c_return=a2c_return, ppo_return=ppo_return), symbols=_SYMBOLS
    )

    assert (result["decision"], result["nominated_families"]) == (
        decision,
        nominated,
    )
    assert not any(
        result[key]
        for key in ("paper_authorized", "production_eligible", "unused_data_accessed")
    )


@pytest.mark.parametrize("scenario", ("cost_2x", "latency_1"))
def test_a2c_absolute_qualification_cannot_ignore_stress_scenarios(
    scenario: str,
) -> None:
    rows = _cells()
    for seed in (0, 1):
        for symbol in _SYMBOLS:
            _set_cell_return(
                rows,
                family="a2c",
                seed=seed,
                symbol=symbol,
                scenario=scenario,
                total_return=-0.02,
            )

    result = _compare_verified_family_cells(rows, symbols=_SYMBOLS)

    assert result["family_qualified"]["ppo"] is True
    assert result["family_qualified"]["a2c"] is False
    assert result["family_absolute_symbols"]["a2c"] == []
    assert result["decision"] == "NOMINATE_PPO_FOR_NEXT_RESEARCH"


@pytest.mark.parametrize(
    ("a2c_returns", "expected_uplift_pass"),
    [
        ((0.011, 0.021, 0.031, 0.041, 0.045), True),
        ((0.055, 0.045, 0.035, 0.025, 0.015), False),
    ],
)
def test_a2c_uplift_uses_paired_seed_votes_not_unpaired_medians(
    a2c_returns: tuple[float, ...], expected_uplift_pass: bool
) -> None:
    rows = _cells()
    ppo_returns = (0.01, 0.02, 0.03, 0.04, 0.05)
    scenarios = tuple(fixed_comparison_contract()["evaluation"]["scenarios"])
    for seed, (a2c_return, ppo_return) in enumerate(zip(a2c_returns, ppo_returns)):
        for symbol in _SYMBOLS:
            for scenario in scenarios:
                _set_cell_return(
                    rows,
                    family="a2c",
                    seed=seed,
                    symbol=symbol,
                    scenario=scenario,
                    total_return=a2c_return,
                )
                _set_cell_return(
                    rows,
                    family="ppo",
                    seed=seed,
                    symbol=symbol,
                    scenario=scenario,
                    total_return=ppo_return,
                )

    result = _compare_verified_family_cells(rows, symbols=_SYMBOLS)

    assert result["family_qualified"] == {"ppo": True, "a2c": True}
    assert result["a2c_relative_uplift_pass"] is expected_uplift_pass
    assert result["a2c_relative_symbols"] == (
        list(_SYMBOLS) if expected_uplift_pass else []
    )
    assert result["decision"] == (
        "NOMINATE_A2C_FOR_NEXT_RESEARCH"
        if expected_uplift_pass
        else "NOMINATE_BOTH_FOR_NEXT_RESEARCH"
    )


def test_comparison_rejects_incomplete_and_duplicate_matrices() -> None:
    rows = _cells()
    with pytest.raises(ValueError, match="incomplete"):
        _compare_verified_family_cells(rows[:-1], symbols=_SYMBOLS)
    with pytest.raises(ValueError, match="duplicate"):
        _compare_verified_family_cells([*rows, rows[0]], symbols=_SYMBOLS)


def test_exact_twenty_percent_drawdown_boundary_passes() -> None:
    rows = _cells()
    rows[0]["ledger_max_drawdown"] = 0.20

    result = _compare_verified_family_cells(rows, symbols=_SYMBOLS)

    assert result["family_hard_guards_pass"]["ppo"] is True
    assert result["family_qualified"]["ppo"] is True


@pytest.mark.parametrize(
    ("scenario", "field", "value", "reason"),
    [
        ("base", "ledger_max_drawdown", 0.200_001, "drawdown_exceeded"),
        ("cost_2x", "ledger_max_drawdown", 0.200_001, "drawdown_exceeded"),
        ("latency_1", "ledger_max_drawdown", 0.200_001, "drawdown_exceeded"),
        ("base", "termination_reasons", ["margin_call"], "terminated"),
        ("base", "terminal_flat", False, "terminal_position_not_flat"),
        ("base", "active_order_remainders", ["0.001"], "active_order_remainder"),
    ],
)
def test_any_a2c_execution_guard_vetoes_an_otherwise_profitable_family(
    scenario: str, field: str, value: object, reason: str
) -> None:
    rows = _cells()
    next(row for row in rows if row["family"] == "a2c" and row["scenario"] == scenario)[
        field
    ] = value

    result = _compare_verified_family_cells(rows, symbols=_SYMBOLS)

    assert (
        result["decision"],
        result["family_hard_guards_pass"]["a2c"],
    ) == ("NOMINATE_PPO_FOR_NEXT_RESEARCH", False)
    assert reason in result["hard_guard_violations"]["a2c"][0]["reasons"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("interval_count", 17_543, "incomplete clock coverage"),
        ("interval_count", 17_544.0, "incomplete clock coverage"),
        ("ledger_max_drawdown", float("nan"), "finite number"),
        ("year_returns", {"2023": 0.01}, "incomplete year coverage"),
        ("seed", True, "preregistered integer"),
    ],
)
def test_comparison_rejects_malformed_cells(
    field: str, value: object, message: str
) -> None:
    rows = _cells()
    rows[0][field] = value

    with pytest.raises(ValueError, match=message):
        _compare_verified_family_cells(rows, symbols=_SYMBOLS)


def test_comparison_rejects_nonsequence_and_nonmapping_cells() -> None:
    with pytest.raises(ValueError, match="sequence of evidence rows"):
        _compare_verified_family_cells(None, symbols=_SYMBOLS)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="evidence mapping"):
        _compare_verified_family_cells([*_cells(), None], symbols=_SYMBOLS)  # type: ignore[list-item]
