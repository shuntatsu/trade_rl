from __future__ import annotations

from pathlib import Path

from trade_rl.operations._training_capability_audit_impl import (
    run_training_capability_audit as _run_training_capability_audit,
)


def run_training_capability_audit(output_root: Path) -> dict[str, object]:
    return _run_training_capability_audit(output_root)
