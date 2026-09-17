"""Predeclared family selection; controls and best-seed selection are excluded."""

from __future__ import annotations

from typing import Any

import numpy as np

from trade_rl.evaluation.directional_candidates import ARMS

COMPLEXITY_ORDER = (
    "trend",
    "mean_reversion",
    "channel_breakout",
    "ridge24",
    "lightgbm24",
    "ppo",
)
SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")


def passes_screen(result: dict[str, Any], *, require_positive_years: bool) -> bool:
    """Recompute eligibility instead of trusting a stored qualified flag."""
    years = result["year_returns"]
    return bool(
        result["metrics"]["total_return"] > 0
        and 0 <= result["ledger_max_drawdown"] <= 0.2
        and result["terminal_flat"]
        and not result["termination_reasons"]
        and result["stop_index"] - result["start_index"] == 17544
        and len(result["returns"]) == 17544
        and set(years) == {"2023", "2024"}
        and all(np.isfinite(value) for value in years.values())
        and np.isfinite(result["metrics"]["total_return"])
        and (not require_positive_years or all(value > 0 for value in years.values()))
    )


def passes_stress(result: dict[str, Any]) -> bool:
    stresses = result.get("stress", [])
    return bool(
        len(stresses) == 2
        and [(row["cost_multiplier"], row["latency_bars"]) for row in stresses]
        == [(2.0, 0), (1.0, 1)]
        and all(passes_screen(row, require_positive_years=False) for row in stresses)
        and set(result.get("by_symbol", {})) == set(SYMBOLS)
    )


def select_development_candidates(
    results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Require every arm, then rank only complete eligible candidate families."""
    if set(results) != set(ARMS):
        raise ValueError(
            "selection requires the complete roster, including five PPO seeds"
        )
    families: dict[str, Any] = {}
    for family in COMPLEXITY_ORDER:
        arms = [f"ppo{seed}" for seed in range(5)] if family == "ppo" else [family]
        rows = [results[arm] for arm in arms]
        screened = [passes_screen(row, require_positive_years=True) for row in rows]
        stressed = [
            base and passes_stress(row)
            for base, row in zip(screened, rows, strict=True)
        ]
        total = float(np.median([row["metrics"]["total_return"] for row in rows]))
        years = {
            year: (
                float(np.median([row["year_returns"][year] for row in rows]))
                if all(year in row["year_returns"] for row in rows)
                else None
            )
            for year in ("2023", "2024")
        }
        needed = 4 if family == "ppo" else 1
        families[family] = {
            "arms": arms,
            "base_pass_count": sum(screened),
            "stress_pass_count": sum(stressed),
            "qualified": sum(stressed) >= needed
            and total > 0
            and all(value is not None and value > 0 for value in years.values()),
            "total_return": total,
            "year_returns": years,
            "turnover_total": float(
                np.median([row["metrics"]["turnover_total"] for row in rows])
            ),
        }
    ranked = sorted(
        (name for name in COMPLEXITY_ORDER if families[name]["qualified"]),
        key=lambda name: (
            -families[name]["total_return"],
            families[name]["turnover_total"],
            COMPLEXITY_ORDER.index(name),
        ),
    )
    return {
        "schema": "directional_selection_v1",
        "families": families,
        "ranked_candidates": ranked,
        "winner": ranked[0] if ranked else None,
        "decision": "PROSPECTIVE_PAPER_REQUIRED"
        if ranked
        else "NO_QUALIFIED_CANDIDATE",
        "production_eligible": False,
        "unused_data_accessed": False,
    }
