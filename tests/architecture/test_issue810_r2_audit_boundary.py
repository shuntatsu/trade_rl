"""The W01 R2 archival tool is read-only and has no trading runtime authority."""

from __future__ import annotations

import ast
from pathlib import Path

from tools import issue810_r2_release_audit as audit

SOURCE = Path(__file__).resolve().parents[2] / "tools" / "issue810_r2_release_audit.py"
ALLOWED_IMPORTS = {
    "argparse",
    "hashlib",
    "json",
    "re",
    "stat",
    "zipfile",
    "pathlib",
    "typing",
    "__future__",
}


def test_issue810_release_auditor_imports_no_market_or_transport_owner() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add((node.module or "").split(".")[0])
    assert imports == ALLOWED_IMPORTS


def test_issue810_release_auditor_pins_original_market_result_and_no_approval() -> None:
    assert audit.R2_ZIP_BYTES == 34_538_439
    assert audit.R2_ZIP_SHA256 == (
        "03bcbd613a516e9e0504d5c074297a896d3e680d87c29adcdbc2fad2d4dc1556"
    )
    assert audit.R2_MANIFEST_SHA256 == (
        "9e1c4ff1d0f30e9fcb98fc915edcf10348e8dfdce1a04a14b1b7acabe312935e"
    )
    assert audit.R2_HEAD == "30c5ab54fdb9db7f03b947c1687a09c8b929c461"
    assert len(audit.R2_SOURCE_FILES) == 8
    assert len(audit.R2_ROOT_FILES) == 13
