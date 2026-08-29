from __future__ import annotations

import runpy
from pathlib import Path

runpy.run_path(".github/patch_v10_lifecycle_phase2b.py", run_name="__main__")

replacements = {
    "tests/learning/test_rollout_execution_lifecycle.py": (
        "risk_weight = self._risk_weights if offset == 0 else float(action[0])",
        "risk_weight = self._risk_weights",
    ),
    "tests/workflows/test_universal_causal_alpha_v10_stage_entry.py": (
        'assert stage_entry._REPLAY_LEAF_SCHEMA == "causal_alpha_v10_replay_leaf_v2"',
        'assert stage_entry._REPLAY_LEAF_SCHEMA == "causal_alpha_v10_replay_leaf_v3"',
    ),
}
for filename, (old, new) in replacements.items():
    path = Path(filename)
    text = path.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise SystemExit(f"{filename}: expected one test-contract match, got {text.count(old)}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")

print("phase2b wrapper applied production and test-contract updates")
