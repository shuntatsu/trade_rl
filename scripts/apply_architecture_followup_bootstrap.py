from __future__ import annotations

import base64
import hashlib
import sys
import zlib
from pathlib import Path

PAYLOAD_SHA256 = "fb26b2639bc42a00c543471a1fd9745438a2283ada05a9435c5bee1059d53f38"
PAYLOAD_ROOT = Path(__file__).with_name("architecture_followup_payload")
payload = "".join(
    item.read_text(encoding="utf-8") for item in sorted(PAYLOAD_ROOT.glob("*.txt"))
)
if hashlib.sha256(payload.encode()).hexdigest() != PAYLOAD_SHA256:
    raise RuntimeError("architecture follow-up payload checksum mismatch")
code_text = zlib.decompress(base64.b64decode(payload)).decode("utf-8")
patch_old = r'''    replace_once(
        "trade_rl/workflows/walk_forward_evaluation.py",
        "    minimum = TrendStrategy(run.trend).minimum_history_for(dataset)\n",
        "    minimum = max(\n"
        "        TrendStrategy(run.trend).minimum_history_for(dataset),\n"
        "        PortfolioRiskModel(run.portfolio_risk).minimum_history_for(dataset),\n"
        "    )\n",
    )
'''
patch_new = r'''    target = path("trade_rl/workflows/walk_forward_evaluation.py")
    text = target.read_text(encoding="utf-8")
    old = "    minimum = TrendStrategy(run.trend).minimum_history_for(dataset)\n"
    replacement = (
        "    minimum = max(\n"
        "        TrendStrategy(run.trend).minimum_history_for(dataset),\n"
        "        PortfolioRiskModel(run.portfolio_risk).minimum_history_for(dataset),\n"
        "    )\n"
    )
    count = text.count(old)
    if count != 2:
        raise RuntimeError(
            "expected two walk-forward history matches, "
            f"found {count}"
        )
    target.write_text(text.replace(old, replacement), encoding="utf-8")
'''
if code_text.count(patch_old) != 1:
    raise RuntimeError("architecture follow-up runtime patch did not match")
code = code_text.replace(patch_old, patch_new).encode("utf-8")
namespace = {"__name__": "__main__", "__file__": __file__}
sys.argv = [__file__, *sys.argv[1:]]
exec(compile(code, __file__, "exec"), namespace)
