from pathlib import Path

runtime_path = Path("trade_rl/workflows/universal_teacher_runtime.py")
runtime = runtime_path.read_text()
old_import = '''from trade_rl.integrations.sb3_runtime import (\n    build_episode_oracle_batch_for_environment,\n)\n'''
new_import = '''from trade_rl.integrations.sb3_runtime import (\n    build_episode_oracle_batch_for_environment,\n    oracle_teacher_config_for_environment,\n)\n'''
if old_import in runtime:
    runtime = runtime.replace(old_import, new_import, 1)
elif "oracle_teacher_config_for_environment" not in runtime:
    raise SystemExit("sb3 runtime import anchor not found")

old_loop = '''        if batch.dataset_id != binding.source_dataset_id:\n            raise ValueError("Universal teacher batch dataset identity mismatch")\n        environment = build_universal_symbol_teacher_environment(\n'''
new_loop = '''        if batch.dataset_id != binding.source_dataset_id:\n            raise ValueError("Universal teacher batch dataset identity mismatch")\n        concrete_environment = concrete_environment_factory(binding)\n        close_concrete = getattr(concrete_environment, "close", None)\n        if not callable(close_concrete):\n            raise TypeError("Universal teacher concrete environment must be closable")\n        try:\n            candidate_teacher_config = oracle_teacher_config_for_environment(\n                concrete_environment\n            )\n        finally:\n            close_concrete()\n        if candidate_teacher_config.digest != batch.teacher_config_digest:\n            raise ValueError("Universal Oracle teacher config identity mismatch")\n        environment = build_universal_symbol_teacher_environment(\n'''
if old_loop in runtime:
    runtime = runtime.replace(old_loop, new_loop, 1)
elif "candidate_teacher_config.digest" not in runtime:
    raise SystemExit("Universal teacher loop anchor not found")
runtime_path.write_text(runtime)
compile(runtime, str(runtime_path), "exec")

test_path = Path("tests/workflows/test_universal_teacher_bundle_runtime.py")
test = test_path.read_text()
anchor = '''    monkeypatch.setattr(\n        module, "build_universal_symbol_teacher_environment", build_environment\n    )\n'''
addition = '''    monkeypatch.setattr(\n        module,\n        "oracle_teacher_config_for_environment",\n        lambda _environment: SimpleNamespace(digest=_digest("teacher-config")),\n    )\n'''
if addition not in test:
    if anchor not in test:
        raise SystemExit("teacher bundle test monkeypatch anchor not found")
    test = test.replace(anchor, anchor + addition, 1)
old_factory = '''        concrete_environment_factory=lambda _binding: object(),\n'''
new_factory = '''        concrete_environment_factory=lambda _binding: SimpleNamespace(\n            close=lambda: None\n        ),\n'''
if old_factory in test:
    test = test.replace(old_factory, new_factory, 1)
elif new_factory not in test:
    raise SystemExit("teacher bundle concrete factory anchor not found")
test_path.write_text(test)
compile(test, str(test_path), "exec")
