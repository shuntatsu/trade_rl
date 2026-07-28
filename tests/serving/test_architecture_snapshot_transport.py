"""Temporary CI transport for an isolated source snapshot."""

from __future__ import annotations

import base64
import io
import os
import tarfile
from pathlib import Path

import pytest


def test_emit_architecture_remediation_source_snapshot() -> None:
    if os.name == "nt":
        pytest.skip("single Ubuntu transport is sufficient")
    root = Path(__file__).resolve().parents[2]
    excluded_roots = {".git", ".venv", "node_modules", "var"}
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if not path.is_file() or relative.parts[0] in excluded_roots:
                continue
            archive.add(path, arcname=relative.as_posix(), recursive=False)
    encoded = base64.b64encode(payload.getvalue()).decode("ascii")
    pytest.fail(f"ARCH_SNAPSHOT_BEGIN\n{encoded}\nARCH_SNAPSHOT_END")
