from __future__ import annotations

from pathlib import Path


path = Path("trade_rl/integrations/cost_critic_ppo.py")
source = path.read_text(encoding="utf-8")
old = "policy.extract_features = local_value  # type: ignore[method-assign,assignment]"
new = "policy.extract_features = local_value  # type: ignore[method-assign]"
count = source.count(old)
if count != 1:
    raise SystemExit(f"H2 Mypy suppression seam changed: {count}")
path.write_text(source.replace(old, new), encoding="utf-8")
