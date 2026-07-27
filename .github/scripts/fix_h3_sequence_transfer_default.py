from __future__ import annotations

from pathlib import Path


path = Path("trade_rl/integrations/compact_rollout_buffer.py")
source = path.read_text(encoding="utf-8")
old = """        mode = _validate_sequence_transfer_mode(self.sequence_transfer_mode)
"""
new = """        mode = _validate_sequence_transfer_mode(
            getattr(self, "sequence_transfer_mode", "synchronous")
        )
"""
count = source.count(old)
if count != 1:
    raise SystemExit(
        f"sequence transfer compatibility seam changed: expected one match, got {count}"
    )
path.write_text(source.replace(old, new), encoding="utf-8")
