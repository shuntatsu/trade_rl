from pathlib import Path

script = Path("tools/_temporary_apply_causal_teacher_admission.py")
text = script.read_text(encoding="utf-8")
old = '"horizon_mix": item.candidate.controller.horizon_mix.value,'
new = '"horizon_mix": CausalAlphaHorizonMix(item.candidate.controller.horizon_mix).value,'
if text.count(old) != 1:
    raise SystemExit("causal admission horizon-mix patch target drifted")
text = text.replace(old, new)
exec(compile(text, str(script), "exec"), {"__name__": "__main__"})

runtime_test = Path("tests/workflows/test_universal_teacher_bundle_runtime.py")
text = runtime_test.read_text(encoding="utf-8")
old = '''    @dataclass(frozen=True)\n    class _BundleStub:\n        episode_batches: dict[str, EpisodeOracleBatch] | None = None\n'''
new = '''    @dataclass(frozen=True)\n    class _BundleStub:\n        episode_batches: dict[str, EpisodeOracleBatch] | None = None\n        causal_teacher_selection_evidence: dict[str, object] | None = None\n        causal_teacher_episode_hours: float | None = None\n'''
if text.count(old) != 1:
    raise SystemExit("universal runtime bundle test-double target drifted")
runtime_test.write_text(text.replace(old, new), encoding="utf-8")
