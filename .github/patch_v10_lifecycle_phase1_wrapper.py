from __future__ import annotations

import runpy
from pathlib import Path

EXPECTED_FAILURE = (
    "trade_rl/learning/causal_alpha_v10_hierarchy.py: expected one match, got 0: "
    "'                \"schema_version\": \"causal_alpha_v10_target_compiler_contract_v3\"'"
)

try:
    runpy.run_path(".github/patch_v10_lifecycle_phase1.py", run_name="__main__")
except SystemExit as error:
    if str(error) != EXPECTED_FAILURE:
        raise
else:
    raise SystemExit("phase1 wrapper expected the known compiler-schema patch mismatch")

path = Path("trade_rl/learning/causal_alpha_v10_hierarchy.py")
text = path.read_text(encoding="utf-8")
old = '            "schema_version": "causal_alpha_v10_target_compiler_contract_v3",\n'
new = (
    '            "reduce_only_execution_contract": "explicit_v1",\n'
    '            "schema_version": "causal_alpha_v10_target_compiler_contract_v4",\n'
)
if text.count(old) != 1:
    raise SystemExit(f"expected one actual compiler schema line, got {text.count(old)}")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
print("phase1 wrapper completed known final patch")
