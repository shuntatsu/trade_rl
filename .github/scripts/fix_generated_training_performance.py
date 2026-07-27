from __future__ import annotations

from pathlib import Path


path = Path("trade_rl/rl/training_performance.py")
source = path.read_text(encoding="utf-8")
old = '''            namespace = getattr(owner, "__dict__", None)
            had_local = isinstance(namespace, dict) and name in namespace
            local_value = namespace.get(name) if had_local else None
'''
new = '''            namespace = getattr(owner, "__dict__", None)
            if isinstance(namespace, dict) and name in namespace:
                had_local = True
                local_value = namespace[name]
            else:
                had_local = False
                local_value = None
'''
count = source.count(old)
if count != 1:
    raise SystemExit(f"performance namespace seam changed: {count}")
path.write_text(source.replace(old, new), encoding="utf-8")
