from __future__ import annotations

from pathlib import Path

path = Path("trade_rl/risk/pretrade.py")
text = path.read_text(encoding="utf-8")
old = '''        controlled = requested.copy()\n        entry_changed = False\n        hold_changed = False\n        exit_changed = False\n        reversal_changed = False\n        for index, (target, current) in enumerate(\n            zip(requested, existing, strict=True)\n        ):\n            if emergency_mask[index]:\n                controlled[index] = 0.0\n                continue\n            if reduce_only_mask[index]:\n                continue\n'''
new = '''        controlled = requested.copy()\n        entry_changed = False\n        hold_changed = False\n        exit_changed = False\n        reversal_changed = False\n        reduce_only_satisfied = False\n        for index, (target, current) in enumerate(\n            zip(requested, existing, strict=True)\n        ):\n            if emergency_mask[index]:\n                controlled[index] = 0.0\n                continue\n            if reduce_only_mask[index]:\n                if abs(current) <= _TOLERANCE:\n                    controlled[index] = 0.0\n                    reduce_only_satisfied = True\n                    continue\n                if target * current < -_TOLERANCE:\n                    raise ValueError("reduce-only target cannot change sign")\n                if abs(target) >= abs(current) - _TOLERANCE:\n                    controlled[index] = current\n                    reduce_only_satisfied = True\n                continue\n'''
if text.count(old) != 1:
    raise SystemExit(f"expected one rebalance-control block, got {text.count(old)}")
text = text.replace(old, new, 1)
old = '''        if reversal_changed:\n            reasons.append("reversal_hysteresis")\n\n        small_changes = (\n'''
new = '''        if reversal_changed:\n            reasons.append("reversal_hysteresis")\n        if reduce_only_satisfied:\n            reasons.append("reduce_only_satisfied")\n\n        small_changes = (\n'''
if text.count(old) != 1:
    raise SystemExit(f"expected one rebalance reason block, got {text.count(old)}")
text = text.replace(old, new, 1)
old = '''        for index in np.flatnonzero(reduce_mask):\n            target = float(requested[index])\n            current = float(existing[index])\n            if abs(current) <= _TOLERANCE:\n                raise ValueError("reduce-only target cannot start from flat exposure")\n            if target * current < -_TOLERANCE:\n                raise ValueError("reduce-only target cannot change sign")\n            if abs(target) > abs(current) + _TOLERANCE:\n                raise ValueError("reduce-only target cannot increase exposure")\n        emergency_mask = (\n'''
new = '''        emergency_mask = (\n'''
if text.count(old) != 1:
    raise SystemExit(f"expected one eager reduce-only validation block, got {text.count(old)}")
text = text.replace(old, new, 1)
path.write_text(text, encoding="utf-8")
print("stale reduce-only projection patch applied")
