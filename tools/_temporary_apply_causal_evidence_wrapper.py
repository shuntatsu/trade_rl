from __future__ import annotations

from pathlib import Path

main = Path("tools/_temporary_apply_causal_evidence.py")
exec(compile(main.read_text(encoding="utf-8"), str(main), "exec"), {"__name__": "__main__"})

path = Path("tests/integrations/test_universal_causal_teacher_admission.py")
text = path.read_text(encoding="utf-8")
old = '''        causal_teacher_admission_evidence=_failed_admission(symbols),\n        causal_teacher_episode_hours=720.0,\n'''
new = '''        causal_teacher_admission_evidence=_failed_admission(symbols),\n        causal_teacher_package_evidence={\n            "schema_version": "universal_causal_alpha_teacher_package_evidence_v1",\n            "artifact_digest": content_digest("package-evidence"),\n        },\n        causal_teacher_episode_hours=720.0,\n'''
if text.count(old) != 1:
    raise SystemExit("failed-admission package evidence test target drifted")
text = text.replace(old, new)
old = '''    admission_path = tmp_path / "causal-teacher-admission.json"\n    assert selection_path.is_file()\n    assert admission_path.is_file()\n'''
new = '''    admission_path = tmp_path / "causal-teacher-admission.json"\n    package_path = tmp_path / "causal-teacher-package.json"\n    assert selection_path.is_file()\n    assert admission_path.is_file()\n    assert package_path.is_file()\n'''
if text.count(old) != 1:
    raise SystemExit("failed-admission package artifact assertion target drifted")
path.write_text(text.replace(old, new), encoding="utf-8")
