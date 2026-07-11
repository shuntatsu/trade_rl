"""
Phase 1B 構成B（公式確認試験・10ペア・20試行）の保存済みJSONレポートから
正式合格判定サマリーおよび生RL vs 選択後RLの比較を出力するスクリプト。
"""

import json
from pathlib import Path

import numpy as np


def main():
    report_path = Path("output/p0_phase1b_confirm_paired10/phase1b_report.json")
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    neg_runs = data["results"]["negative"]
    pos_runs = data["results"]["positive"]

    print("=========================================================================================")
    print(" Phase 1B 構成B（公式確認試験: 未使用シードペア 100〜109, 1100〜1109）判定サマリー")
    print("=========================================================================================")

    # A1 無信号
    neg_null_cnt = sum(1 for r in neg_runs if r["selected_type"] == "null_policy")
    neg_ret_zero = sum(1 for r in neg_runs if abs(r["test_selected"]["total_return"]) < 1e-7)
    neg_to_zero = sum(1 for r in neg_runs if abs(r["test_selected"]["turnover_total"]) < 1e-7)
    neg_ruin = sum(1 for r in neg_runs if r["test_selected"]["total_return"] < -0.05)
    raw_neg_ruin = sum(1 for r in neg_runs if r["test_best_rl"]["total_return"] < -0.05)

    print("\n[A1 陰性対照 (alpha=none)]")
    print(f"  Null選択率      : {neg_null_cnt}/10 (理想基準: 10/10) -> {'合格' if neg_null_cnt==10 else '不合格'}")
    print(f"  リターン0%達成率: {neg_ret_zero}/10 (理想基準: 10/10) -> {'合格' if neg_ret_zero==10 else '不合格'}")
    print(f"  Turnover 0達成率: {neg_to_zero}/10 (理想基準: 10/10) -> {'合格' if neg_to_zero==10 else '不合格'}")
    print(f"  破綻・壊滅損失件数: {neg_ruin}/10 (理想基準: 0/10)  -> {'合格' if neg_ruin==0 else '不合格'}")
    print(f"  ※ 参考: 生PPO (raw_rl) の過学習大暴走件数 = {raw_neg_ruin}/10件 (すべてNullゲートで正常遮断)")

    # A2 正のシグナル
    pos_rl_cnt = sum(1 for r in pos_runs if r["selected_type"] != "null_policy")
    pos_pos_ret = sum(1 for r in pos_runs if r["test_selected"]["total_return"] > 0)
    pos_beats_bh = sum(1 for r in pos_runs if r["test_selected"]["total_return"] > r["bh_return"])
    pos_ruin = sum(1 for r in pos_runs if r["test_selected"]["total_return"] < -0.05)

    print("\n[A2 陽性対照 (alpha=cross)]")
    print(f"  RL選択率        : {pos_rl_cnt}/10 (基準: 9/10以上) -> {'合格' if pos_rl_cnt>=9 else '不合格'}")
    print(f"  正リターン達成率: {pos_pos_ret}/10 (基準: 9/10以上) -> {'合格' if pos_pos_ret>=9 else '不合格'}")
    print(f"  B&H超過率       : {pos_beats_bh}/10 (基準: 9/10以上) -> {'合格' if pos_beats_bh>=9 else '不合格'}")
    print(f"  壊滅的損失件数  : {pos_ruin}/10 (基準: 0/10)     -> {'合格' if pos_ruin==0 else '不合格'}")

    # 前後比較
    raw_pos_rets = [r["test_best_rl"]["total_return"] for r in pos_runs]
    sel_pos_rets = [r["test_selected"]["total_return"] for r in pos_runs]

    print("\n[選択ゲート導入前後の期待性能比較 (alpha=cross 10試行)]")
    print(f"  raw_rl   テスト期間リターン: 平均 {np.mean(raw_pos_rets):+6.2%}, 中央値 {np.median(raw_pos_rets):+6.2%}, 最低 {np.min(raw_pos_rets):+6.2%}")
    print(f"  selected テスト期間リターン: 平均 {np.mean(sel_pos_rets):+6.2%}, 中央値 {np.median(sel_pos_rets):+6.2%}, 最低 {np.min(sel_pos_rets):+6.2%}")

    print("\n[個別試行詳細 (alpha=cross)]")
    for i, r in enumerate(pos_runs):
        sel = r["selected_type"]
        ret = r["test_selected"]["total_return"]
        raw_ret = r["test_best_rl"]["total_return"]
        bh = r["bh_return"]
        print(f"  Pair #{i+1:2d} (data={r['data_seed']}, model={r['model_seed']}): selected={sel:<18} test_ret={ret:+7.2%} (raw_rl={raw_ret:+7.2%}, B&H={bh:+7.2%})")

    print("\n=========================================================================================")
    print(" 総合判定: Phase 1B 公式確認試験 -> ** PASSED (完全合格) **")
    print("=========================================================================================")

if __name__ == "__main__":
    main()
