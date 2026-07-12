from pathlib import Path


path = Path(__file__).resolve().parents[1] / "mars_lite/pipeline/residual_pipeline.py"
text = path.read_text(encoding="utf-8")

old_imports = '''from mars_lite.eval.context_window import with_history_context
from mars_lite.eval.relative_evaluation import (
    _moving_block_mean_test,
    evaluate_relative_agent,
)
from mars_lite.features.signal_check import run_leak_self_test, run_signal_check
'''
new_imports = '''from mars_lite.eval.context_window import with_history_context
from mars_lite.eval.gate1_diagnostics import walk_forward_ic
from mars_lite.eval.relative_evaluation import (
    _moving_block_mean_test,
    evaluate_relative_agent,
)
from mars_lite.features.signal_check import run_leak_self_test
'''
if text.count(old_imports) != 1:
    raise RuntimeError("unexpected residual pipeline feature import layout")
text = text.replace(old_imports, new_imports)

old_gate_import = '''from mars_lite.pipeline.gates import (
    evaluate_baseline_only_gate,
    evaluate_residual_gate2,
)
'''
new_gate_import = '''from mars_lite.pipeline.gates import (
    evaluate_baseline_only_gate,
    evaluate_residual_alpha_gate,
    evaluate_residual_gate2,
)
'''
if text.count(old_gate_import) != 1:
    raise RuntimeError("unexpected residual pipeline gate import layout")
text = text.replace(old_gate_import, new_gate_import)

old_gate = '''    signal_gate = run_signal_check(
        train_fs,
        horizon=args.horizon,
        target="cs_demean",
    )
    alpha = FrozenResidualAlpha.fit(
        train_fs,
        horizon=args.horizon,
        target="cs_demean",
        model=getattr(args, "signal_model", "gbm"),
        gate_result=signal_gate.to_dict(),
    )
'''
new_gate = '''    signal_model = str(getattr(args, "signal_model", "gbm"))
    model_gate_report = walk_forward_ic(
        train_fs,
        horizon=args.horizon,
        target="cs_demean",
        model=signal_model,
    )
    signal_gate = evaluate_residual_alpha_gate(model_gate_report)
    alpha = FrozenResidualAlpha.fit(
        train_fs,
        horizon=args.horizon,
        target="cs_demean",
        model=signal_model,
        gate_result=signal_gate,
    )
'''
if text.count(old_gate) != 1:
    raise RuntimeError("unexpected residual alpha gate layout")
text = text.replace(old_gate, new_gate)

old_report = '        "signal_gate": signal_gate.to_dict(),\n'
new_report = '        "signal_gate": signal_gate,\n'
if text.count(old_report) != 1:
    raise RuntimeError("unexpected signal gate report layout")
text = text.replace(old_report, new_report)

path.write_text(text, encoding="utf-8")
