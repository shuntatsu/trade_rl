#!/usr/bin/env python3
"""
Phase 2A: 要素別機能シナリオ評価 (Element-Specific Synthetic Scenarios)

各要素(C, E, V)が意図したシナリオで機能を発揮できるか、B0との比較検証。
  - C (Cap=0.4): concentrated_alpha, concentrated_alpha_crash
  - E (EMA=0.5): persistent_cross, fast_reversal
  - V (Vol=0.20): vol_shock_up, vol_shock_down
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from scripts.p0_phase2a_ablation import run_single_trial

SCENARIO_PAIRS = [
    ("C", "concentrated_alpha"),
    ("C", "concentrated_alpha_crash"),
    ("E", "persistent_cross"),
    ("E", "fast_reversal"),
    ("V", "vol_shock_up"),
    ("V", "vol_shock_down"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--output", default="output/phase2a_scenarios_results.json")
    args = parser.parse_args()

    data_seeds = [42, 43, 44, 45, 46]
    model_seeds = [100]  # 各シナリオを5 data_seeds × 1 model_seed で機能比較

    results = []
    for cfg, alpha in SCENARIO_PAIRS:
        print(f"\nEvaluating Scenario: Config={cfg} vs B0 on Alpha={alpha}...")
        for ds in data_seeds:
            for ms in model_seeds:
                res_b0 = run_single_trial(
                    config_id="B0",
                    alpha=alpha,
                    data_seed=ds,
                    model_seed=ms,
                    timesteps=args.timesteps,
                )
                res_cfg = run_single_trial(
                    config_id=cfg,
                    alpha=alpha,
                    data_seed=ds,
                    model_seed=ms,
                    timesteps=args.timesteps,
                )
                results.append(
                    {"type": "B0", "scenario_config": cfg, "alpha": alpha, **res_b0}
                )
                results.append(
                    {"type": cfg, "scenario_config": cfg, "alpha": alpha, **res_cfg}
                )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n=== Scenario Evaluation Summary Saved to {args.output} ===")
    for cfg, alpha in SCENARIO_PAIRS:
        print(f"\n--- Scenario: {alpha} ({cfg} vs B0) ---")
        sub_b0 = [r for r in results if r["type"] == "B0" and r["alpha"] == alpha]
        sub_cfg = [r for r in results if r["type"] == cfg and r["alpha"] == alpha]
        b0_ret = np.median([r["test_return"] for r in sub_b0]) * 100
        cfg_ret = np.median([r["test_return"] for r in sub_cfg]) * 100
        b0_dd = np.median([r["test_max_dd"] for r in sub_b0]) * 100
        cfg_dd = np.median([r["test_max_dd"] for r in sub_cfg]) * 100
        b0_mw = np.max([r["max_abs_weight"] for r in sub_b0])
        cfg_mw = np.max([r["max_abs_weight"] for r in sub_cfg])
        print(
            f"B0   : Ret={b0_ret:>6.2f}%, MaxDD={b0_dd:>6.2f}%, MaxWeight={b0_mw:>5.2f}"
        )
        print(
            f"{cfg:<4} : Ret={cfg_ret:>6.2f}%, MaxDD={cfg_dd:>6.2f}%, MaxWeight={cfg_mw:>5.2f}"
        )


if __name__ == "__main__":
    main()
