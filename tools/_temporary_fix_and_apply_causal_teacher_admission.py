from pathlib import Path

script = Path("tools/_temporary_apply_causal_teacher_admission.py")
text = script.read_text(encoding="utf-8")
old = '"horizon_mix": item.candidate.controller.horizon_mix.value,'
new = '"horizon_mix": CausalAlphaHorizonMix(item.candidate.controller.horizon_mix).value,'
if text.count(old) != 1:
    raise SystemExit("causal admission horizon-mix patch target drifted")
text = text.replace(old, new)
exec(compile(text, str(script), "exec"), {"__name__": "__main__"})
