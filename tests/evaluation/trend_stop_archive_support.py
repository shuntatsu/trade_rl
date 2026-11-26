"""Immutable source/spec observations for the closed trend-stop study."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_COMMIT = "8d7b97f828a908340ddbd9323bf7d1e564d0b1e7"
PROTOCOL_SOURCE = "trade_rl/evaluation/trend_stop_study/protocol.py"
SPEC_SOURCE = "docs/specs/trend-following-stops.md"
SOURCE_HASHES = {
    PROTOCOL_SOURCE: "0c512fa05b34704ead300f97c1ae3508f7246a7535e06f5f7f9dcd1ad7935d0e",
    SPEC_SOURCE: "4ac04f5cc803ed5af539eab8ef8d4ea50d5f54384c4cc759eff3beb4aa334f7c",
}


def archived_text(path: str) -> str:
    raw = subprocess.check_output(["git", "show", f"{ARCHIVE_COMMIT}:{path}"], cwd=ROOT)
    assert hashlib.sha256(raw).hexdigest() == SOURCE_HASHES[path], path
    return raw.decode("utf-8")
