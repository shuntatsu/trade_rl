from pathlib import Path

script = Path("tools/_temporary_apply_selection_persistence.py")
text = script.read_text(encoding="utf-8")
old = '''        if causal_teacher_package is None:\n            causal_teacher_package = build_universal_causal_alpha_teacher_package(\n'''
new = '''        if resolved_causal_package is None:\n            if causal_teacher_evidence_root is None:\n                raise ValueError(\n                    "Universal causal teacher auto-build requires causal_teacher_evidence_root"\n                )\n            resolved_causal_package = build_universal_causal_alpha_teacher_package(\n'''
if text.count(old) != 1:
    raise SystemExit("runner causal package patch-script target drifted")
text = text.replace(old, new)
# The original script had a separate insertion step for the guard. Remove it,
# because the replacement above now installs the guard and assignment together.
old_guard_patch = '''replace_once(\n    runner,\n    ''' + "'''" + '''        if causal_teacher_package is None:\\n            causal_teacher_package = build_universal_causal_alpha_teacher_package(\\n''' + "'''" + ''',\n    ''' + "'''" + '''        if causal_teacher_package is None:\\n            if causal_teacher_evidence_root is None:\\n                raise ValueError(\\n                    "Universal causal teacher auto-build requires causal_teacher_evidence_root"\\n                )\\n            causal_teacher_package = build_universal_causal_alpha_teacher_package(\\n''' + "'''" + ''',\n)\n'''
if old_guard_patch in text:
    text = text.replace(old_guard_patch, "")
exec(compile(text, str(script), "exec"), {"__name__": "__main__"})
