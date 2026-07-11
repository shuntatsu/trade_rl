"""
構成Aの保存済み結果を使って、val_confirm 上の MDD 閾値スイープを実施するスクリプト。
再学習は行わず、保存済みの val_confirm_metrics に基づき各閾値での合否とテスト性能を算出する。
"""

import json
from pathlib import Path

import numpy as np


def main():
    report_path = Path("output/p0_phase1b_grid5x5/phase1b_report.json")
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    neg_runs = data["results"]["negative"]
    pos_runs = data["results"]["positive"]

    thresholds = [0.05, 0.06, 0.08, 0.10, 0.12, 0.15]

    print(
        "========================================================================================="
    )
    print(
        " 構成A 保存済み val_confirm 上での MDD上限閾値スイープ (Sortino>=0.5, Return>=+0.5%)"
    )
    print(
        "========================================================================================="
    )
    print(
        f"{'MDD上限':<8} | {'none RL通過(偽陽性)':<16} | {'none Null(正当棄権)':<18} | {'cross RL通過(真陽性)':<18} | {'cross Null(誤棄権)':<16} | {'cross Test中央値':<15} | {'cross Test最悪値':<15}"
    )
    print("-" * 115)

    for mdd_th in thresholds:
        # None評価
        none_rl_pass = 0
        none_null_sel = 0
        none_test_rets = []

        for r in neg_runs:
            cm = r["val_confirm_metrics"]
            passes = (
                cm["sortino"] >= 0.5
                and cm["total_return"] >= 0.005
                and cm["max_drawdown"] <= mdd_th
            )
            if passes:
                none_rl_pass += 1
                none_test_rets.append(r["test_best_rl"]["total_return"])
            else:
                none_null_sel += 1
                none_test_rets.append(0.0)

        # Cross評価
        cross_rl_pass = 0
        cross_null_sel = 0
        cross_test_rets = []

        for r in pos_runs:
            cm = r["val_confirm_metrics"]
            passes = (
                cm["sortino"] >= 0.5
                and cm["total_return"] >= 0.005
                and cm["max_drawdown"] <= mdd_th
            )
            if passes:
                cross_rl_pass += 1
                cross_test_rets.append(r["test_best_rl"]["total_return"])
            else:
                cross_null_sel += 1
                cross_test_rets.append(0.0)

        cross_med = float(np.median(cross_test_rets))
        cross_min = float(np.min(cross_test_rets))

        print(
            f"{mdd_th * 100:5.0f}%    | {none_rl_pass:>6}/25         | {none_null_sel:>7}/25          | {cross_rl_pass:>7}/25          | {cross_null_sel:>6}/25        | {cross_med:>+10.2%}     | {cross_min:>+10.2%}"
        )

    print(
        "========================================================================================="
    )


if __name__ == "__main__":
    main()
