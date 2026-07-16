from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "trade_rl/rl/environment.py"


def main() -> None:
    text = TARGET.read_text(encoding="utf-8")
    old = '''        pending_target_discarded = bool(
            time_limit_reached
            and self.config.signal_delay_decisions == 1
            and self._pending_hybrid_target is not None
        )
        discarded_pending_target = (
            None
            if not pending_target_discarded
            else self._pending_hybrid_target.copy()
        )
'''
    new = '''        pending_hybrid_target = self._pending_hybrid_target
        pending_target_discarded = bool(
            time_limit_reached
            and self.config.signal_delay_decisions == 1
            and pending_hybrid_target is not None
        )
        discarded_pending_target = (
            pending_hybrid_target.copy()
            if pending_target_discarded and pending_hybrid_target is not None
            else None
        )
'''
    if old not in text:
        raise RuntimeError("Task 10 pending-target narrowing anchor is missing")
    TARGET.write_text(text.replace(old, new, 1), encoding="utf-8")


if __name__ == "__main__":
    main()
