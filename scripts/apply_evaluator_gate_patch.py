from pathlib import Path


path = Path(__file__).resolve().parents[1] / "mars_lite/pipeline/evaluator.py"
text = path.read_text(encoding="utf-8")
old_import = "from mars_lite.pipeline.dataset_builder import build_feature_set\n"
new_import = (
    old_import
    + "from mars_lite.pipeline.gates import evaluate_direct_gate2\n"
)
if text.count(old_import) != 1:
    raise RuntimeError("unexpected evaluator import layout")
text = text.replace(old_import, new_import)
old = '''    rl_ret = float(agent_res["total_return"])
    gate2_details = {}
    gate2_passed = True
    for bname, bres in baselines.items():
        bd = bres.to_dict() if hasattr(bres, "to_dict") else bres
        beat = bool(rl_ret > float(bd.get("total_return", 0.0)))
        gate2_details[bname] = {
            "rl_return": rl_ret,
            "baseline_return": float(bd.get("total_return", 0.0)),
            "rl_beat": beat,
        }
        if not beat:
            gate2_passed = False
    tf_baseline = gate2_details.get("trend_following", {})
    gate2 = {
        "passed": bool(gate2_passed),
        "rl_beat_trend_following": bool(tf_baseline.get("rl_beat", False))
        if "rl_beat" in tf_baseline
        else None,
        "details": gate2_details,
    }
'''
new = '''    rl_ret = float(agent_res["total_return"])
    baseline_metrics = {
        name: result.to_dict() if hasattr(result, "to_dict") else result
        for name, result in baselines.items()
    }
    gate2 = evaluate_direct_gate2(agent=agent_res, baselines=baseline_metrics)
    gate2_passed = bool(gate2["passed"])
    gate2_details = gate2["details"]
    tf_baseline = gate2_details["trend_following"]
'''
if text.count(old) != 1:
    raise RuntimeError("unexpected evaluator Gate 2 layout")
text = text.replace(old, new)
old_print = '''    print(
        f"\\n[Gate 2] {'PASS' if gate2_passed else 'FAIL'} "
        f"RL vs all baselines. trend_following: "
        f"{'BEAT' if tf_baseline.get('rl_beat') else 'LOST'}"
    )
'''
new_print = '''    print(
        f"\\n[Gate 2] {'PASS' if gate2_passed else 'FAIL'} "
        f"RL vs mandatory flat + trend_following. trend_following: "
        f"{'BEAT' if tf_baseline.get('rl_beat') else 'LOST'}"
    )
'''
if text.count(old_print) != 1:
    raise RuntimeError("unexpected evaluator Gate 2 print layout")
path.write_text(text.replace(old_print, new_print), encoding="utf-8")
